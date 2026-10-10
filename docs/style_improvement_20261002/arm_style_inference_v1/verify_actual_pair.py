"""CPU-only coordinate proof for two immutable, physically trained actors.

The inputs are finite mathematical observations, not fresh simulated frames.
This checks the composite action contract, not balance or gait quality.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from CrowdSim.humanoid.runtime.arm_style import (
    ARM_ACTION_INDICES,
    BASE_ACTION_INDICES,
    ExperimentalArmStyleInference,
)
from CrowdSim.humanoid.runtime.checkpoint_contract import CANONICAL_ACTION_MASK


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt"
STYLE = ROOT / "output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/step_2000.pt"
BASE_SHA = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
STYLE_SHA = "28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72"


def tensor_record(value):
    value = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def source_record(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rng():
    state = np.random.get_state()
    return random.getstate(), (state[0], state[1].copy(), *state[2:]), torch.get_rng_state().clone()


def assert_rng(expected):
    current = rng()
    assert current[0] == expected[0]
    assert current[1][0] == expected[1][0] and current[1][2:] == expected[1][2:]
    assert np.array_equal(current[1][1], expected[1][1])
    assert byte_equal(current[2], expected[2])


def byte_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                            right.contiguous().reshape(-1).view(torch.uint8)))


def main():
    assert not torch.cuda.is_initialized()
    before = rng()
    composite = ExperimentalArmStyleInference.from_checkpoints(
        BASE, STYLE, device="cpu", base_sha256=BASE_SHA, style_sha256=STYLE_SHA,
    )
    assert_rng(before)
    descriptions = {"baseline": composite.baseline.describe(), "style": composite.style.describe()}
    assert descriptions["baseline"]["update_count"] == 1000
    assert descriptions["baseline"]["physical_transition_count"] == 8192000
    assert descriptions["style"]["update_count"] == 2000
    assert descriptions["style"]["physical_transition_count"] == 16384000
    physics = composite.describe()["physical_compatibility"]
    assert physics["training_domains"] == {
        "baseline": {"terrain_extent_m": 192., "isolated_spawn_margin_m": 48.},
        "style": {"terrain_extent_m": 192., "isolated_spawn_margin_m": 48.},
    }
    controllers = {
        0.: composite,
        .25: ExperimentalArmStyleInference(composite.baseline, composite.style, alpha=.25),
        1.: ExperimentalArmStyleInference(composite.baseline, composite.style, alpha=1.),
    }
    assert_rng(before)
    buffers = {name: value.clone() for name, value in composite.named_buffers()}
    parameters = {name: value.clone() for name, value in composite.named_parameters()}
    arms, other = list(ARM_ACTION_INDICES), list(BASE_ACTION_INDICES)
    frozen = [index for index, active in enumerate(CANONICAL_ACTION_MASK) if not active]
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    rows = []
    for batch, scale in ((1, 0.), (8, 1.), (32, 50.)):
        own = torch.randn(batch, 357, generator=generator) * scale
        task = torch.randn(batch, 1044, generator=generator) * scale
        base, style = composite.baseline.act(own, task), composite.style.act(own, task)
        for alpha, controller in controllers.items():
            actual = controller.act(own, task)
            repeated = controller.act(own, task)
            assert all(byte_equal(actual[key], repeated[key]) for key in actual)
            assert all(torch.isfinite(value).all() for value in actual.values())
            expected = base["mean_actions"].clone()
            if alpha == 1.:
                expected[:, arms] = style["mean_actions"][:, arms]
            elif alpha != 0.:
                expected[:, arms] = ((1. - alpha) * base["mean_actions"][:, arms]
                                     + alpha * style["mean_actions"][:, arms])
            assert byte_equal(actual["actions"], expected)
            assert byte_equal(actual["mean_actions"], expected)
            assert byte_equal(actual["applied_actions"], expected.clamp(-1, 1) * composite.baseline.action_mask)
            assert byte_equal(actual["mean_actions"][:, other], base["mean_actions"][:, other])
            assert byte_equal(actual["applied_actions"][:, other], base["applied_actions"][:, other])
            assert byte_equal(actual["applied_actions"][:, frozen], base["applied_actions"][:, frozen])
            if alpha == 0.:
                assert all(byte_equal(actual[key], base[key]) for key in actual)
            if alpha == 1.:
                assert byte_equal(actual["mean_actions"][:, arms], style["mean_actions"][:, arms])
            assert_rng(before)
            rows.append({"batch": batch, "input_scale": scale, "alpha": alpha,
                "self_observation": tensor_record(own), "task_observation": tensor_record(task),
                "outputs": {key: tensor_record(value) for key, value in actual.items()},
                "raw24_expected_bitwise_equal": True, "other45_raw_and_applied_baseline_bitwise_equal": True,
                "frozen12_applied_baseline_bitwise_equal": True,
                "final_projection_formula_bitwise_equal": True,
                "repeat_bitwise_equal": True,
                "alpha0_all_outputs_baseline_bitwise_equal": True if alpha == 0. else None,
                "alpha1_raw24_style_bitwise_equal": True if alpha == 1. else None})
    for name, value in composite.named_buffers():
        assert byte_equal(value, buffers[name]), name
    for name, value in composite.named_parameters():
        assert byte_equal(value, parameters[name]), name
        assert not value.requires_grad
    assert_rng(before)
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"omni", "isaaclab", "isaacsim"} for name in sys.modules)
    report = {"schema": "navisaaclab.experimental_arm_style_cpu_contract.v1", "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False,
        "SDK_started": False, "physical_validation_performed": False, "runtime_integrated": False,
        "input_protocol": "Private CPU generator seed20261002; batch1/8/32 at scales0/1/50; finite FP32 mathematical observations, not newly simulated states",
        "observation_batches": 3, "alpha_conditions_per_batch": 3, "composite_output_tensor_records": 27,
        "actual_checkpoint_physics_compatibility": physics, "selected_actors": descriptions,
        "conditions": rows, "Python_NumPy_Torch_RNG_unchanged": True,
        "all_model_parameters_and_RMS_mask_buffers_unchanged": True,
        "normalizer_and_mask_buffer_hashes": {name: tensor_record(value) for name, value in buffers.items()},
        "sources": [source_record(ROOT / name) for name in (
            "CrowdSim/humanoid/runtime/arm_style.py", "tests/test_amp_arm_style.py",
            "CrowdSim/humanoid/runtime/amp_inference.py", "CrowdSim/humanoid/runtime/checkpoint_contract.py",
            "CrowdSim/humanoid/amp/observations.py")],
        "training_source_metadata": [source_record(ROOT / name) for name in (
            "docs/style_improvement_20261002/backward_penalty_training_source.json",
            "docs/style_improvement_20261002/mixed_curriculum_training_source_v1.json")],
        "limitations": ["CPU coordinate proof does not test balance, gait style, navigation, or safety.",
            "Upper-limb forces can disturb baseline lower-body dynamics.",
            "Not the paper's single trained actor; no composite Gaussian/log_prob/critic/AMP reward.",
            "Compatibility comes from recorded checkpoint physics; a future runtime must independently match its live rig.",
            "No new physical reset, pose injection, PD decode, simulator, or deployed loop in this module."],
        "sealed34_outputs_read": False}
    (OUT / "actual_pair_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "selected_checkpoints": 2,
        "observation_batches": 3, "alpha_conditions_per_batch": 3,
        "composite_output_tensor_records": 27, "all_required_coordinate_and_projection_checks_bitwise_equal": True,
        "RNG_and_RMS_unchanged": True, "physical_validation_performed": False}))


if __name__ == "__main__":
    main()
