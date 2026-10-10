"""CPU observer contracts; no SDK/physics pass is inferred from mock actors."""

from pathlib import Path
import importlib.util
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = source(
    "pre_action_CPU_contract",
    ROOT / "CrowdSim/humanoid/runtime/pre_action_observation.py",
)


class Normalizer:
    clip = 5.0

    def __init__(self):
        self.mean = torch.ones(1044, dtype=torch.float64)
        self.variance = torch.full((1044,), 4.0, dtype=torch.float64)
        self.count = torch.tensor(27.0, dtype=torch.float64)
        self.forward_calls = 0

    def normalize(self, x, *, clamp=True):
        self.forward_calls += 1
        z = (x - self.mean.float()) / torch.sqrt(self.variance.float() + 1e-5)
        return z.clamp(-5, 5) if clamp else z


class Actor:
    config = SimpleNamespace(self_dim=357, task_dim=1044, normalize_observations=True)

    def __init__(self):
        self._inference = SimpleNamespace(task_normalizer=Normalizer())
        self.model_calls = 0
        self.inputs = None

    def act(self, a, b, *, deterministic):
        assert deterministic is True
        self.model_calls += 1
        self.inputs = (a, b)
        raw = (
            a.mean(-1, keepdim=True)
            + self._inference.task_normalizer.normalize(b).mean(-1, keepdim=True)
        ).expand(-1, 69)
        self.last = {
            "actions": raw,
            "mean_actions": raw,
            "applied_actions": raw.clamp(-1, 1),
        }
        return self.last


def test_selected_call_preserves_inputs_RNG_moments_and_exact_output_objects(tmp_path):
    actor = Actor()
    writer = module.PreActionObservationWriter(tmp_path / "samples")
    observed = module.PreActionObservedActor(actor, writer)
    a = torch.full((4, 357), 0.3)
    b = torch.zeros(4, 1044)
    b[:, 20:] = 100.0
    before_a, before_b = a.clone(), b.clone()
    moments = {
        k: getattr(actor._inference.task_normalizer, k).clone()
        for k in ("mean", "variance", "count")
    }
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    result = observed.act(a, b)
    assert result is actor.last
    assert all(result[k] is actor.last[k] for k in result)
    assert actor.model_calls == 1 and actor.inputs[0] is a and actor.inputs[1] is b
    assert torch.equal(a, before_a) and torch.equal(b, before_b)
    assert (
        torch.equal(torch_rng, torch.get_rng_state())
        and python_rng == random.getstate()
    )
    after = np.random.get_state()
    assert (
        numpy_rng[0] == after[0]
        and np.array_equal(numpy_rng[1], after[1])
        and numpy_rng[2:] == after[2:]
    )
    assert all(
        torch.equal(v, getattr(actor._inference.task_normalizer, k))
        for k, v in moments.items()
    )
    assert (
        actor._inference.task_normalizer.forward_calls == 3
    )  # two diagnostics, one actualactor; one model
    with np.load(tmp_path / "samples/call_00000000.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["self_obs"], a.numpy()) and np.array_equal(
            saved["task_obs"], b.numpy()
        )
        assert (
            saved["task_clip_mask"][:, 20:].all()
            and not saved["task_clip_mask"][:, :20].any()
        )
        assert np.allclose(
            saved["task_z_unclamped"][:, 20:], 99 / np.sqrt(4 + 1e-5), rtol=1e-6
        )
        assert (saved["task_z_clamped"][:, 20:] == 5.0).all()
        assert (
            saved["task_mean"].dtype == np.float64
            and saved["task_obs"].dtype == np.float32
        )
    row = json.loads((tmp_path / "samples/samples.jsonl").read_text())
    assert (
        row["call_index"] == 0
        and row["native_counter"] is None
        and row["runtime_completion_claim"] is False
    )


def test_stride_has_no_extra_actor_calls_and_unobserved_calls_do_not_normalize_extra(
    tmp_path,
):
    actor = Actor()
    writer = module.PreActionObservationWriter(tmp_path / "stride", stride=30)
    observed = module.PreActionObservedActor(actor, writer)
    a, b = torch.zeros(1, 357), torch.zeros(1, 1044)
    for _ in range(31):
        result = observed.act(a, b)
        assert result is actor.last
    rows = [
        json.loads(x) for x in (writer.path / "samples.jsonl").read_text().splitlines()
    ]
    assert [r["call_index"] for r in rows] == [0, 30]
    assert (
        actor.model_calls == 31 and actor._inference.task_normalizer.forward_calls == 35
    )


@pytest.mark.parametrize("bad", ["blend", "dimension", "disabled_normalization"])
def test_unsupported_actor_rejected_without_forward(tmp_path, bad):
    actor = Actor()
    if bad == "blend":
        actor._inference.baseline = object()
    elif bad == "dimension":
        actor.config = SimpleNamespace(
            self_dim=356, task_dim=1044, normalize_observations=True
        )
    else:
        actor.config = SimpleNamespace(
            self_dim=357, task_dim=1044, normalize_observations=False
        )
    with pytest.raises(ValueError):
        module.PreActionObservedActor(
            actor, module.PreActionObservationWriter(tmp_path / bad)
        )
    assert actor.model_calls == 0


@pytest.mark.parametrize(
    "bad", ["nan", "double", "dimension", "batch", "nondeterministic"]
)
def test_invalid_input_fails_without_actor_or_sample(tmp_path, bad):
    actor = Actor()
    writer = module.PreActionObservationWriter(tmp_path / "bad")
    observed = module.PreActionObservedActor(actor, writer)
    a, b = torch.zeros(1, 357), torch.zeros(1, 1044)
    if bad == "nan":
        b[0, 0] = torch.nan
    elif bad == "double":
        b = b.double()
    elif bad == "dimension":
        a = torch.zeros(1, 356)
    elif bad == "batch":
        b = torch.zeros(2, 1044)
    with pytest.raises(ValueError):
        observed.act(a, b, deterministic=bad != "nondeterministic")
    assert actor.model_calls == 0 and not list(writer.path.glob("*.npz"))


def test_selected_sample_survives_subprocess_os_exit(tmp_path):
    script = """import importlib.util,sys,torch
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('observer',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class N:
 clip=5.;mean=torch.zeros(1044,dtype=torch.float64);variance=torch.ones(1044,dtype=torch.float64);count=torch.tensor(1.,dtype=torch.float64)
 def normalize(self,x,clamp=True):return x.clamp(-5,5) if clamp else x
class A:
 config=SimpleNamespace(self_dim=357,task_dim=1044,normalize_observations=True);_inference=SimpleNamespace(task_normalizer=N())
 def act(self,a,b,deterministic=True):return {'actions':torch.zeros(1,69)}
w=m.PreActionObservationWriter(sys.argv[2]);m.PreActionObservedActor(A(),w).act(torch.zeros(1,357),torch.zeros(1,1044))
import os;os._exit(17)
"""
    output = tmp_path / "fast_exit"
    result = subprocess.run(
        [sys.executable, "-c", script, module.__file__, str(output)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )
    assert result.returncode == 17
    row = json.loads((output / "samples.jsonl").read_text())
    assert row["runtime_completion_claim"] is False
    with np.load(output / "call_00000000.npz", allow_pickle=False) as arrays:
        assert arrays["task_obs"].shape == (1, 1044)


def test_actual_driver_AST_act_is_once_before_world_step():
    import ast
    import CrowdSim

    workspace = Path(CrowdSim.__file__).resolve().parents[1]
    tree = ast.parse((workspace / "scripts/train_paper_amp.py").read_text())
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "world"
            and call.func.attr == "step"
            for call in ast.walk(node)
        )
    ]
    loop = min(loops, key=lambda node: len(list(ast.walk(node))))
    calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    ]
    act = [
        node
        for node in calls
        if node.func.value.id == "trainer" and node.func.attr == "act"
    ]
    step = [
        node
        for node in calls
        if node.func.value.id == "world" and node.func.attr == "step"
    ]
    assert len(act) == len(step) == 1 and act[0].lineno < step[0].lineno
    assert (
        isinstance(step[0].args[0], ast.Subscript)
        and step[0].args[0].value.id == "action"
    )


def test_launcher_delegates_one_factory_and_one_actor_call_no_extra_context(
    tmp_path, monkeypatch
):
    launcher = source(
        "preaction_launcher_contract", ROOT / "scripts/probe_paper_pre_action.py"
    )
    args = [
        "--amp-checkpoint",
        str(tmp_path / "step_1000.pt"),
        "--config",
        str(tmp_path / "config.json"),
        "--diffusion-checkpoint",
        str(tmp_path / "step_5000.ckpt"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--duration-seconds",
        ".1",
        "--inference-only",
        "--output",
        str(tmp_path / "output"),
        "--observation-stride",
        "30",
    ]
    actor = Actor()
    loads = []
    native = object()
    calls = []

    def factory(checkpoint, *, device):
        loads.append((checkpoint, device))
        return actor

    def driver(**options):
        assert options["native_scene_factory"] is native
        wrapped = options["evaluation_actor_factory"]("checkpoint", device="cpu")
        result = wrapped.act(torch.zeros(1, 357), torch.zeros(1, 1044))
        assert result is actor.last
        calls.append("existing_driver")
        return 0

    def delegation(forwarded, *, driver_main):
        assert "--observation-stride" not in forwarded
        (tmp_path / "output").mkdir()
        return driver_main(
            evaluation_actor_factory=factory, native_scene_factory=native
        )

    monkeypatch.setattr(launcher.deployment, "main", delegation)
    argv = sys.argv
    assert launcher.main(args, driver_main=driver, observer_module=module) == 0
    assert len(loads) == actor.model_calls == len(calls) == 1 and sys.argv is argv
    metadata = json.loads((tmp_path / "output/pre_action_observer.json").read_text())
    assert metadata["additional_model_context_or_world_created"] is False
    assert len(metadata["observer_loaded_source_files"]) == 2
    assert metadata["runtime_completed"] is False


def test_cold_observer_and_launcher_import_no_Torch_SDK_or_context():
    script = """import importlib.util,sys
for name,path in [('observer',sys.argv[1]),('launcher',sys.argv[2])]:
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert 'torch' not in sys.modules
assert not any(name.startswith(('isaacsim','isaaclab','omni.')) for name in sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            module.__file__,
            str(ROOT / "scripts/probe_paper_pre_action.py"),
        ],
        check=False,
    )
    assert result.returncode == 0


def test_launcher_rejects_existing_SDK_before_any_output(tmp_path, monkeypatch):
    launcher = source(
        "preaction_existing_SDK_contract", ROOT / "scripts/probe_paper_pre_action.py"
    )
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace())
    args = [
        "--amp-checkpoint",
        str(tmp_path / "step_1000.pt"),
        "--config",
        str(tmp_path / "config.json"),
        "--diffusion-checkpoint",
        str(tmp_path / "step_5000.ckpt"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--duration-seconds",
        ".1",
        "--inference-only",
        "--output",
        str(tmp_path / "output"),
    ]
    with pytest.raises(RuntimeError, match="fresh Python process"):
        launcher.main(args)
    assert not (tmp_path / "output").exists()


def test_launcher_rejects_arm_experiment_before_driver(tmp_path):
    launcher = source(
        "preaction_arm_rejection_contract", ROOT / "scripts/probe_paper_pre_action.py"
    )
    args = [
        "--amp-checkpoint",
        str(tmp_path / "step_1000.pt"),
        "--config",
        str(tmp_path / "config.json"),
        "--diffusion-checkpoint",
        str(tmp_path / "step_5000.ckpt"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--duration-seconds",
        ".1",
        "--inference-only",
        "--experimental-arm-style-checkpoint",
        str(tmp_path / "step_2000.pt"),
        "--arm-style-alpha",
        ".25",
        "--output",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit) as error:
        launcher.main(
            args, driver_main=lambda **_: pytest.fail("driver must not be called")
        )
    assert error.value.code == 2 and not (tmp_path / "output").exists()
