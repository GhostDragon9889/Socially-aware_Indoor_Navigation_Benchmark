"""CPU-only facade/real Trainer.load oracle for the sealed 2019D milestone.

Reads one old validation capture, never sealed34. Saved states/commands are
re-encoded on CPU: the original runner did not save 357/1044 observations.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np
import torch

from CrowdSim.humanoid.amp.controller import PacerAMPController
from CrowdSim.humanoid.amp.observations import PacerSMPLMapping, build_self_observation, build_task_observation
from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
from CrowdSim.humanoid.training import replay, trainer


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_1000.pt"
EXPECTED_SHA = "66e03fadd0aabf02210d7427cb3f4d86d6ecadc639395c294a70072f0b1e00e2"
CAPTURE_RESULT = ROOT / "output/diffusion_amp/motion_tracking_amass_root_curriculum_full_validation_v1/result.json"
SOURCES = (
    "CrowdSim/humanoid/runtime/evaluation_actor.py", "CrowdSim/humanoid/runtime/amp_inference.py",
    "CrowdSim/humanoid/runtime/checkpoint_contract.py", "CrowdSim/humanoid/training/trainer.py",
    "CrowdSim/humanoid/training/replay.py", "CrowdSim/humanoid/application/settings.py",
    "CrowdSim/humanoid/amp/observations.py", "protomotions/utils/rotations.py",
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def tensor_record(value):
    value = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def rng():
    value = np.random.get_state()
    return random.getstate(), (value[0], value[1].copy(), *value[2:]), torch.get_rng_state().clone()


def compare_rng(left, right):
    return {"Python": left[0] == right[0],
            "NumPy": left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
            and np.array_equal(left[1][1], right[1][1]),
            "Torch_CPU": torch.equal(left[2], right[2])}


@contextmanager
def forbidden_training_constructors():
    def blocked(*args, **kwargs):
        raise AssertionError("Inference/helper attempted a trainer, optimizer, replay or public PACER load")

    with ExitStack() as stack:
        for owner, name in ((trainer.PaperAMPTrainer, "__init__"), (trainer.PaperAMPTrainer, "load"),
                            (torch.optim, "Adam"), (trainer, "AMPPolicyReplay"),
                            (replay, "AMPPolicyReplay"), (PacerAMPController, "from_checkpoint")):
            stack.enter_context(patch.object(owner, name, blocked))
        yield


def input_batches():
    result = json.loads(CAPTURE_RESULT.read_text())
    assert result["status"] == "passed" and len(result["sequences"]) == 22
    assert "flat" in result["layout"]["method"].casefold()
    sequence = result["sequences"][0]
    capture = Path(sequence["capture"])
    assert sha256(capture) == sequence["capture_sha256"]
    mapping = PacerSMPLMapping(tuple(result["physics"]["body_names"]), tuple(result["physics"]["dof_names"]))
    arrays = []
    rows = []
    with np.load(capture, allow_pickle=False) as data:
        for indices in ([1], list(range(16, 24)), list(range(31, 63))):
            assert np.all(data["metric_frame"][indices])
            assert not np.any(data["native_reset_frame"][indices])
            # One physical actor in this captured clip; batch dimension is the
            # selected source times, not simultaneously simulated actors.
            state = SimpleNamespace(**{name: torch.from_numpy(data[name][indices, 0].copy()) for name in (
                "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel")})
            trajectory = torch.from_numpy(data["trajectory_xy"][indices, 0].copy())
            self_obs = build_self_observation(state, mapping=mapping)
            task_obs = build_task_observation(state, trajectory, lambda xy: torch.zeros(xy.shape[:-1], dtype=xy.dtype), mapping=mapping)
            arrays.append((self_obs, task_obs))
            rows.append({"kind": "CPU_reencoded_real_saved_physical_states_and_commands",
                         "capture_indices": indices, "reference_frames": data["reference_frame"][indices].tolist(),
                         "self_observation": tensor_record(self_obs), "task_observation": tensor_record(task_obs),
                         "original_saved_observation_tensors": False, "terrain": "explicit flat Z=0 sampler; no SDK query"})
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    for count, scale in ((1, 0.), (8, 1.), (32, 50.)):
        own = torch.randn(count, 357, generator=generator) * scale
        task = torch.randn(count, 1044, generator=generator) * scale
        arrays.append((own, task))
        rows.append({"kind": "synthetic_mathematical_observations", "private_input_seed": 20261002,
                     "scale": scale, "self_observation": tensor_record(own), "task_observation": tensor_record(task)})
    return arrays, rows, {"result": file_record(CAPTURE_RESULT), "capture": file_record(capture),
                          "motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
                          "input_phase": "post-physics saved states plus recorded command; not claimed to equal original pre-step actor observations"}


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    source_before = {name: file_record(ROOT / name) for name in SOURCES}
    assert sha256(CHECKPOINT) == EXPECTED_SHA
    initial_rng = rng()
    observations, rows, physical_source = input_batches()
    assert all(compare_rng(initial_rng, rng()).values())
    with forbidden_training_constructors():
        actor = EvaluationActor.from_checkpoint(CHECKPOINT, device="cpu", expected_sha256=EXPECTED_SHA)
    assert all(compare_rng(initial_rng, rng()).values())
    assert actor.config.discriminator_input == "state_action"
    assert actor._inference.model.discriminator[0].in_features == 2019
    assert actor.update_count == 1000 and actor.physical_transition_count == 8192000
    assert not any(hasattr(actor, name) for name in ("optimizer", "policy_replay", "discriminator_optimizer", "amp_reward", "restore_rng"))
    moments_before = {name: tensor_record(value) for name, value in actor._inference.named_buffers()}
    with forbidden_training_constructors():
        actual = [actor.act(own, task, deterministic=True) for own, task in observations]
    assert all(compare_rng(initial_rng, rng()).values())
    calls = []
    seed_all = torch.cuda.manual_seed_all

    def observe_seed(seed):
        calls.append(seed)
        return seed_all(seed)

    with patch.object(torch.cuda, "manual_seed_all", observe_seed):
        oracle, runner = trainer.PaperAMPTrainer.load(CHECKPOINT, device="cpu", restore_rng=False)
    legacy_rng = rng()
    legacy_seed_events = calls.copy()
    assert oracle.policy_replay is not None and runner is not None
    assert asdict(actor.config) == asdict(oracle.config) and actor.provenance == oracle.provenance
    assert oracle.model.discriminator[0].in_features == 2019
    oracle_moments = {name: {key: tensor_record(value) for key, value in getattr(oracle, name).state_dict().items()}
                      for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")}
    for inputs, output, row in zip(observations, actual, rows):
        expected = oracle.act(*inputs, deterministic=True)
        assert set(output) == {"actions", "mean_actions", "applied_actions"}
        row["outputs"] = {}
        for key, value in output.items():
            reference = expected[key]
            assert torch.equal(value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
            row["outputs"][key] = {**tensor_record(value), "bytewise_equal": True,
                                   "maximum_absolute_error": float((value - reference).abs().max())}
    assert all(compare_rng(legacy_rng, rng()).values())
    assert oracle_moments == {name: {key: tensor_record(value) for key, value in getattr(oracle, name).state_dict().items()}
                              for name in oracle_moments}
    assert moments_before == {name: tensor_record(value) for name, value in actor._inference.named_buffers()}
    calls.clear()
    with forbidden_training_constructors(), patch.object(torch.cuda, "manual_seed_all", observe_seed):
        helper = prepare_legacy_evaluation_rng(actor.config)
    helper_comparison = compare_rng(legacy_rng, rng())
    assert all(helper_comparison.values())
    assert legacy_seed_events == calls == [actor.config.seed]
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    saved_rng_restored = torch.equal(legacy_rng[2], saved["rng"]["torch_cpu"])
    assert not saved_rng_restored
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"omni", "isaacsim", "isaaclab"} for name in sys.modules)
    assert source_before == {name: file_record(ROOT / name) for name in SOURCES}
    result = {"schema": "evaluation_actor_state_action_actual_cpu.v1", "status": "passed",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
              "CUDA_initialized": False, "Kit_started": False, "sealed34_read": False,
              "checkpoint": file_record(CHECKPOINT), "source_files": source_before,
              "auditor": file_record(Path(__file__)), "physical_capture_input": physical_source,
              "batches": rows, "input_batches": len(rows), "output_tensors": sum(len(row["outputs"]) for row in rows),
              "D_input_dimensions": 2019, "configuration_and_full_provenance_equal": True,
              "facade_load_and_deterministic_act_global_RNG_preserved": True,
              "facade_and_oracle_RMS_buffers_unchanged": True,
              "training_constructors_and_public_PACER_loader_forbidden_during_facade_and_helper": True,
              "complete_reconstruction_D_and_critic_retained": True,
              "complete_training_resume_validation_performed_here": False,
              "legacy_initialization_RNG": {"helper": helper, "actual_Trainer_load_False_comparison": helper_comparison,
                  "legacy_CUDA_manual_seed_all_events": legacy_seed_events, "helper_CUDA_manual_seed_all_events": calls,
                  "saved_checkpoint_Torch_CPU_rng_restored": saved_rng_restored,
                  "CUDA_state_equivalence_verified": False, "CUDA_random_draw_equivalence_verified": False},
              "limits": ["Real saved state/command inputs are reconstructed on CPU, not original stored actor observations.",
                  "Capture was produced by a different fixed actor; no new physics/controller performance is measured.",
                  "Synthetic observations only test the mathematical adapter, never GPU or gait quality.",
                  "The facade retains the complete reconstructed 2019D/critic for bundle validation; no public1950D/Adam/replay is loaded.",
                  "Matching lazy CUDA seed calls and CPU RNG do not verify initialized CUDA generator or planner/physics equivalence."]}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checkpoint_sha256": EXPECTED_SHA, "D_input": 2019,
                      "input_batches": len(rows), "output_tensors": result["output_tensors"], "bytewise_equal": True,
                      "actual_legacy_CPU_RNG_equal": True, "CUDA_equivalence_verified": False}))
    del saved, oracle, runner, actor
    gc.collect()


if __name__ == "__main__":
    main()
