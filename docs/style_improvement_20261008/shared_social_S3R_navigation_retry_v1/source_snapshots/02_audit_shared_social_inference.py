#!/usr/bin/env python3
"""CPU-only strict actor load and forward on a completed run's actual raw rows.

This checks deployment admission, storage and forward, not locomotion quality.
The process exit must be recorded by the caller separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from CrowdSim.humanoid.runtime.social_popart_inference import FrozenSocialPopArtCheckpointInference  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint, sha, admission = admit_training(args.execution_receipt.resolve())
    result = json.loads(Path(admission["result"]["path"]).read_text())
    if result.get("canonical_entry", {}).get("value_normalization") != "popart":
        raise ValueError("This audit explicitly requires the Pop-Art checkpoint family")
    audit = json.loads(Path(admission["audit"]["path"]).read_text())
    rollout_path = Path(audit["rollout"]["path"])
    before = record(rollout_path)
    if before != audit["rollout"]:
        raise ValueError("Actual saved raw rollout changed since CPU audit")
    import torch

    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU inference audit must not initialize CUDA")
    rng = torch.random.get_rng_state().clone()
    actor = FrozenSocialPopArtCheckpointInference.from_checkpoint(checkpoint, expected_sha256=sha, device="cpu")
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise RuntimeError("Strict loader changed caller CPU RNG")
    saved = torch.load(rollout_path, weights_only=True, map_location="cpu", mmap=True)
    rows = saved["collection"]["rollout"]
    raw_self, raw_task = rows["self_obs"][0].clone(), rows["task_obs"][0].clone()
    inputs_before = raw_self.clone(), raw_task.clone()
    prediction = actor.act(raw_self, raw_task, deterministic=True)
    repeated = actor.act(raw_self, raw_task, deterministic=True)
    for key in ("actions", "mean_actions", "applied_actions"):
        value = prediction[key]
        if value.shape != (len(raw_self), 69) or not torch.isfinite(value).all() or value.device.type != "cpu":
            raise ValueError("Invalid deterministic actor output")
        if not torch.equal(value, repeated[key]):
            raise RuntimeError("Repeated frozen deterministic inference changed")
    if not all(torch.equal(a, b) for a, b in zip((raw_self, raw_task), inputs_before, strict=True)):
        raise RuntimeError("Actor modified raw inputs")
    mask = actor._action_mask
    if int(mask.count_nonzero()) != 57 or not torch.equal(
            prediction["applied_actions"], prediction["mean_actions"].clamp(-1, 1) * mask):
        raise RuntimeError("Physical action mask/clipping changed")
    if record(rollout_path) != before or record(checkpoint)["sha256"] != sha:
        raise RuntimeError("Actual checkpoint/rollout changed during inference audit")
    if torch.cuda.is_initialized() or not torch.equal(rng, torch.random.get_rng_state()):
        raise RuntimeError("CPU audit changed RNG or initialized CUDA")
    report = {"schema": "shared_social.completed_actor_CPU_inference.v1", "status": "passed",
        "reader": record(__file__), "training_execution_admission": admission, "actual_raw_rollout": before,
        "input_rows": len(raw_self), "input_phase": "first actual saved pre-action control",
        "mean_shape": list(prediction["mean_actions"].shape), "active_action_dimensions": 57,
        "maximum_absolute_mean": float(prediction["mean_actions"].abs().max()),
        "repeated_forward_equal": True, "raw_inputs_unchanged": True, "caller_CPU_RNG_preserved": True,
        "CUDA_initialized": False, "frozen_actor": actor.describe(),
        "quality_or_author_equivalence_established": False, "SDK_or_physics_executed": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "input_rows", "mean_shape", "active_action_dimensions",
                                                 "maximum_absolute_mean", "CUDA_initialized")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
