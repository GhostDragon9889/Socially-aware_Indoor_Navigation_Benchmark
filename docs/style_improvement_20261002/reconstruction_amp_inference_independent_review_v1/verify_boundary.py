"""Read-only CPU audit of the frozen prepared inference adapter, not physics QA."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = (
    "CrowdSim/humanoid/runtime/__init__.py",
    "CrowdSim/humanoid/runtime/checkpoint_contract.py",
    "CrowdSim/humanoid/runtime/amp_inference.py",
    "tests/test_reconstruction_amp_inference.py",
)
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt"
CHECKPOINT_SHA = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    source_before = [record(ROOT / name) for name in SOURCES]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    def rng_equal():
        current = np.random.get_state()
        return (random.getstate() == python_rng and current[0] == numpy_rng[0]
                and np.array_equal(current[1], numpy_rng[1]) and current[2:] == numpy_rng[2:]
                and torch.equal(torch.get_rng_state(), torch_rng))

    actor = ReconstructionAMPInference.from_checkpoint(
        CHECKPOINT, device="cpu", expected_sha256=CHECKPOINT_SHA,
    )
    assert rng_equal()
    load_rng_unchanged = rng_equal()
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    own = torch.randn(2, 357, generator=generator)
    task = torch.randn(2, 1044, generator=generator)
    buffers = {name: value.clone() for name, value in actor.named_buffers()}
    first = actor.act(own, task)
    second = actor.act(own, task)
    assert all(torch.equal(first[key].contiguous().view(torch.uint8),
                           second[key].contiguous().view(torch.uint8)) for key in first)
    assert all(torch.equal(value, buffers[name]) for name, value in actor.named_buffers())
    assert rng_equal()
    action_mask = actor.action_mask.clone()
    assert action_mask.dtype == torch.float32 and int(action_mask.sum()) == 57
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in first.values())
    assert torch.equal(first["applied_actions"], first["mean_actions"].clamp(-1, 1) * action_mask)
    absent = ("optimizer", "discriminator_optimizer", "policy_replay", "runner_state", "expert_sampler")
    assert all(not hasattr(actor, name) for name in absent)
    description = actor.describe()
    actor.double()
    double_output = actor.act(own.double(), task.double())
    assert all(value.dtype == torch.float64 and torch.isfinite(value).all() for value in double_output.values())
    try:
        actor.act(own, task)
    except ValueError as error:
        original_float32_after_double_error = str(error)
    else:
        raise AssertionError("Expected unchanged float32 inputs to fail after mask dtype becomes float64")
    assert rng_equal()
    owner_path = ROOT / "docs/style_improvement_20261002/reconstruction_amp_inference_v1/actual_checkpoint_verification.json"
    owner = json.loads(owner_path.read_text())
    counts = [{"name": run["name"], "input_batches": len(run["observations"]),
               "output_tensor_comparisons": sum(len(row["outputs"]) for row in run["observations"]),
               "all_byte_identical": all(output["bitwise_equal"] and output["max_abs_error"] == 0
                                         for row in run["observations"] for output in row["outputs"].values())}
              for run in owner["runs"]]
    source_after = [record(ROOT / name) for name in SOURCES]
    assert source_after == source_before
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"omni", "isaaclab", "isaacsim"} for name in sys.modules)
    report = {
        "schema": "navisaaclab.reconstruction_amp_independent_boundary_review.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "default_fp32_checks_passed_with_nonblocking_dtype_contract_gap",
        "checkpoint": record(CHECKPOINT), "source_records": source_before,
        "source_bytes_unchanged": True,
        "actual_cpu_checks": {"load_python_numpy_torch_rng_unchanged": load_rng_unchanged,
            "repeated_fp32_output_byte_identical": True, "normalizer_and_mask_buffers_unchanged": True,
            "outputs_finite_float32": True, "canonical_mask_active_actions": 57,
            "projection_matches_single_clamp_and_mask": True, "training_objects_not_retained": list(absent),
            "load_act_dtype_conversion_python_numpy_torch_rng_unchanged": rng_equal()},
        "manual_dtype_conversion": {"operation": "actor.double() after validated FP32 load",
            "float64_inputs_accepted": True, "outputs_dtype": "torch.float64",
            "original_float32_inputs_rejected": original_float32_after_double_error,
            "declared_float32_contract_guard_uses_mutable_mask_dtype": True,
            "explicit_dtype_conversion_limitation_present_in_describe": False,
            "blocks_default_fp32_inference_or_existing_gpu_evaluation": False},
        "owner_parity_evidence": {"file": record(owner_path), "runs": counts,
            "total_input_batches": sum(run["input_batches"] for run in counts),
            "total_output_tensor_comparisons": sum(run["output_tensor_comparisons"] for run in counts),
            "independently_reran_owner_two_model_oracle": False},
        "inference_description": description,
        "gpu_used": False, "kit_started": False, "physical_validation_performed": False,
        "read_34_test_outputs": False, "owner_source_modified": False,
        "limitations": ["One immutable checkpoint independently loaded for boundary checks",
            "Owner two-model parity JSON and harness independently inspected, not rerun",
            "Single PD decode is established by source contract and owner unit evidence; no physics was executed",
            "No optimizer/replay/physical-resume integrity or real-scene compatibility is claimed"],
    }
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "default_fp32_checks": "passed",
                      "manual_double_contract_gap_reproduced": True,
                      "owner_output_tensor_comparisons": report["owner_parity_evidence"]["total_output_tensor_comparisons"]}))


if __name__ == "__main__":
    main()
