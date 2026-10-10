"""Pre-Kit CLI and actual runner AST contract for optional foot capture."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.application.configuration import parse_amp_args
from CrowdSim.humanoid.ground_support_archive import GroundSupportArchive, write_interrupted_ground_support

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "docs/style_improvement_20261002/ground_support_integration_before"


def argv(tmp_path, *flags):
    checkpoint = tmp_path / "diffusion.ckpt"
    checkpoint.write_bytes(b"CLI file-existence fixture, not model weights")
    return ["--diffusion-checkpoint", str(checkpoint), "--output", str(tmp_path / "fresh"), *flags]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flags", [(), ("--evaluate-steps", "10", "--resume", "actor.pt"),
    ("--evaluate-steps", "10", "--resume", "actor.pt", "--pairwise-contacts")])
def test_disabled_new_defaults_preserve_all_old_cli_fields(tmp_path, flags):
    old = load_module(BEFORE / "CrowdSim/humanoid/application/configuration.py", "CrowdSim.humanoid.application.ground_before")
    arguments = argv(tmp_path, *flags)
    previous, current = vars(old.parse_amp_args(arguments)), vars(parse_amp_args(arguments))
    assert current.pop("measure_ground_support") is False
    assert current.pop("ground_support_threshold_n") == 1.
    assert current == previous


@pytest.mark.parametrize("flags", [
    ("--measure-ground-support",),
    ("--measure-ground-support", "--evaluate-steps", "10", "--resume", "actor.pt", "--power-source", "legacy_pd_proxy_misaligned"),
    ("--measure-ground-support", "--evaluate-steps", "10"),
    ("--ground-support-threshold-n", "2"),
    ("--measure-ground-support", "--evaluate-steps", "10", "--resume", "actor.pt", "--ground-support-threshold-n", "0"),
    ("--ground-support-threshold-n", "nan"), ("--ground-support-threshold-n", "inf"),
    ("--ground-support-threshold-n", "-1"),
])
def test_ground_measurement_invalid_requests_fail_before_sdk(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        parse_amp_args(argv(tmp_path, *flags))
    assert error.value.code == 2


def test_measured_ground_support_without_or_with_peer_is_explicit_eval(tmp_path):
    base = argv(tmp_path, "--evaluate-steps", "10", "--resume", "actor.pt", "--measure-ground-support",
                "--ground-support-threshold-n", ".25", "--physics-only")
    args = parse_amp_args(base)
    assert args.measure_ground_support and not args.pairwise_contacts
    assert args.ground_support_threshold_n == .25
    assert parse_amp_args([*base, "--pairwise-contacts"]).pairwise_contacts


@pytest.mark.parametrize("filename", ["ground_support.npz", "ground_support_partial.npz"])
def test_new_artifacts_never_overwritten_and_old_disabled_request_unchanged(tmp_path, filename):
    output = tmp_path / "fresh"
    output.mkdir()
    (output / filename).write_bytes(b"previous evidence")
    base = argv(tmp_path, "--evaluate-steps", "10", "--resume", "actor.pt")
    assert not parse_amp_args(base).measure_ground_support
    with pytest.raises(SystemExit):
        parse_amp_args([*base, "--measure-ground-support"])
    assert (output / filename).read_bytes() == b"previous evidence"


def test_cli_and_runner_import_do_not_start_or_import_torch_kit(tmp_path):
    arguments = argv(tmp_path, "--evaluate-steps", "10", "--resume", "actor.pt", "--measure-ground-support")
    code = f"""
import importlib.util,sys
from CrowdSim.humanoid.application.configuration import parse_amp_args
parse_amp_args({arguments!r})
spec=importlib.util.spec_from_file_location('ground_runner',{str(ROOT / 'scripts/train_paper_amp.py')!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert not any(n=='torch' or n.startswith(('isaacsim','isaaclab','omni','pxr')) for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


def capture_fixture():
    tests = load_module(ROOT / "tests/test_foot_capture.py", "ground_capture_cpu_fixture")
    return tests.fixture, tests.control


def captured_row():
    fixture, control = capture_fixture()
    world, clock, _, _, adapter = fixture(2)
    sample = control(clock, adapter, 0)
    return world, clock, adapter, sample


def masks(actors=2):
    return torch.zeros(actors, dtype=torch.bool)


def test_actual_adapter_data_archive_keeps_four_substeps_com_and_control_phase(tmp_path):
    world, clock, adapter, first = captured_row()
    archive = GroundSupportArchive(2)
    archive.append(first, reset_mask=torch.ones(2, dtype=torch.bool), terminated=masks(), truncated=masks(), control_step_after=1)
    world.episode_ids[1] += 1
    adapter.reset_episode_boundary()
    fixture, control = capture_fixture()
    second = control(clock, adapter, 1)
    archive.append(second, reset_mask=torch.tensor([False, True]), terminated=masks(), truncated=masks(), control_step_after=2)
    path = tmp_path / "ground.npz"
    report = archive.write(path)
    assert report["completed_controls"] == 2 and report["completed_physics_substeps"] == 8
    with np.load(path, allow_pickle=False) as data:
        assert data["foot_floor_normal_force_w"].shape == (2, 4, 2, 4, 2, 3)
        assert data["body_com_pos_w"].shape == (2, 4, 2, 24, 3)
        assert np.array_equal(data["control_step"], [0, 1])
        assert np.array_equal(data["control_step_after"], [1, 2])
        assert np.array_equal(data["physics_step"], [[41, 42, 43, 44], [45, 46, 47, 48]])
        assert data["valid"].dtype == np.bool_ and data["valid"].all()
        assert data["episode_ids"].tolist() == [[0, 1], [0, 2]]
        assert not any(name.startswith("peer_") or name.startswith("static_") for name in data.files)
    with pytest.raises(FileExistsError):
        archive.write(path)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("case", ["partial_force", "gap_physics", "wrong_post_counter", "int_mask", "nonfinite_com", "missing_true_com"])
def test_invalid_completed_archive_record_is_not_appended(case):
    _, _, _, sample = captured_row()
    archive = GroundSupportArchive(2)
    post = 1
    reset = masks()
    if case == "partial_force":
        sample["foot_floor_normal_force_w"] = sample["foot_floor_normal_force_w"][:3]
    elif case == "gap_physics":
        sample["physics_step"] = (41, 43, 44, 45)
    elif case == "wrong_post_counter":
        post = 2
    elif case == "int_mask":
        reset = torch.zeros(2, dtype=torch.long)
    elif case == "nonfinite_com":
        sample["body_com_pos_w"][0, 0, 0, 0] = float("nan")
    elif case == "missing_true_com":
        sample["captured_true_com"] = False
    with pytest.raises(ValueError):
        archive.append(sample, reset_mask=reset, terminated=masks(), truncated=masks(), control_step_after=post)
    assert not archive.rows


def test_interrupted_control_stays_invalid_partial_no_padding(tmp_path):
    fixture, _ = capture_fixture()
    _, clock, _, _, adapter = fixture(2)
    adapter.prepare_control_step(77)
    adapter.begin_control_step()
    for _ in range(2):
        clock.value += 1
        adapter.sample(1 / 120)
    report = write_interrupted_ground_support(tmp_path / "partial.npz", adapter, reason="CPU simulated read failure")
    assert report["valid"] is False and report["completed_state_samples"] == report["completed_floor_force_samples"] == 2
    assert report["missing_substeps_padded"] is False
    with np.load(tmp_path / "partial.npz", allow_pickle=False) as data:
        assert data["valid"].shape == () and not bool(data["valid"])
        assert data["partial_foot_floor_normal_force_w"].shape == (2, 2, 4, 2, 3)
        assert data["partial_body_com_pos_w"].shape == (2, 2, 24, 3)
        assert int(data["control_step"]) == 77
    with pytest.raises(ValueError):
        GroundSupportArchive(2).write(tmp_path / "empty.npz")


def runner_tree():
    return ast.parse((ROOT / "scripts/train_paper_amp.py").read_text())


def test_disabled_ground_branches_preserve_original_control_step_loop_ast():
    """Compare the complete control loop to the saved before-ground source.

    Setup and post-run reporting also contain independent expert opt-in changes;
    this test isolates the physical-control loop's disabled-foot behavior. The
    baseline is the real historical source, not a copy of the current runner.
    Only exact optional-ground guards are removed; no unrelated AST is ignored.
    """
    class DisabledGroundCapture(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) in {
                "ground_capture is not None", "ground_archive is not None", "args.measure_ground_support",
            }:
                return self.visit(ast.Module(body=node.orelse, type_ignores=[])).body
            return self.generic_visit(node)

    def control_loop(tree):
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "step"]
        assert len(loops) == 1
        return loops[0]

    current = DisabledGroundCapture().visit(control_loop(runner_tree()))
    previous = control_loop(ast.parse((BEFORE / "scripts/train_paper_amp.py").read_text()))
    assert ast.dump(current, include_attributes=False) == ast.dump(previous, include_attributes=False)


def test_actual_runner_ast_prepares_measures_once_then_increments_control_before_reset():
    main = next(node for node in runner_tree().body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step")
    step_index = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "world.step")
    selected = loop.body[step_index - 1:step_index + 2]
    counter_increment = next(node for node in loop.body if isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "control_step")
    reset_index = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Expr)
                       and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "world.reset")
    assert reset_index > step_index + 1
    assert "ground_capture.reset_episode_boundary()" in ast.unparse(loop.body[reset_index + 1])
    function = ast.FunctionDef(name="run_actual_step", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in
        ("world", "ground_capture", "ground_archive", "control_step", "action")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*selected, counter_increment, ast.Return(ast.Name(id="control_step", ctx=ast.Load()))], decorator_list=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "actual_ground_runner_ast", "exec"), namespace)
    fixture, _ = capture_fixture()
    world, clock, _, peer, adapter = fixture(2)
    world.age = torch.zeros(2, dtype=torch.long)
    calls = []

    def physical_step(action):
        calls.append(action)
        adapter.begin_control_step()
        for _ in range(4):
            clock.value += 1
            adapter.sample(1 / 120)
        world.age += 1
        return SimpleNamespace(), masks(), masks()

    world.step = physical_step
    archive = GroundSupportArchive(2)
    applied = torch.zeros(2, 69)
    assert namespace["run_actual_step"](world, adapter, archive, 0, {"applied_actions": applied}) == 1
    assert len(calls) == 1 and calls[0] is applied
    assert len(archive.rows) == 1 and int(archive.rows[0]["control_step"]) == 0
    assert len([event for event in peer.events if event[0] == "sample"]) == 4


def test_actual_runner_peer_projection_keeps_original_physical_state_fields():
    node = next(node for node in ast.walk(runner_tree()) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "args.pairwise_contacts" and "values.update" in ast.unparse(node))
    namespace = {"args": SimpleNamespace(pairwise_contacts=True), "ground_capture": object(),
                 "ground_sample": {key: torch.tensor(False) for key in ("static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n")},
                 "values": {"body_pos": "old body field"}}
    namespace["ground_sample"]["body_com_pos_w"] = torch.zeros(4, 2, 24, 3)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_peer_projection", "exec"), namespace)
    assert set(namespace["values"]) == {"body_pos", "static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n"}
    assert "body_com_pos_w" not in namespace["values"]
