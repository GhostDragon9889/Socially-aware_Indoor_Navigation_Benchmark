#!/usr/bin/env python3
"""Audit a completed shared SocialAMP run without constructing a simulator or NN."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # CPU weights_only data loading only; no model construction or SDK import.
    import torch

    run = args.run_directory.resolve()
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "passed_local_training"
    assert result["cleanup_succeeded"] and result["cleanup_errors"] == []
    assert result["source_profile_before_after_equal"]
    assert result["owned_contact_views_live_after_release"] == 0
    assert result["scene_backend_views_live_after_post_context_gc"] == 0
    runtime = result["profile"]["runtime"]
    actors, controls, updates = (runtime[key] for key in ("actors", "rollout_controls", "updates"))
    transitions = actors * controls * updates
    assert result["completed_updates"] == updates
    assert result["training_update_returned_count"] == updates
    assert result["actual_training_transitions"] == transitions
    checkpoint = run / f"step_{updates:06d}.pt"
    before = identity(checkpoint)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    assert state["format"] == "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"
    assert state["update_count"] == updates and state["physical_transition_count"] == transitions
    assert state["runner_state"] is None and state["physical_runner_state_included"] is False
    assert list(state["model"]["actor.0.weight"].shape) == [2048, 633]
    assert list(state["model"]["task_encoder.cnn.0.weight"].shape) == [16, 3, 4, 4]
    assert list(state["model"]["discriminator.0.weight"].shape) == [1024, 2019]
    finite_tensors = 0

    def finite(value):
        nonlocal finite_tensors
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
            assert bool(torch.isfinite(value).all())
            finite_tensors += 1
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)

    for name in ("model", "optimizer", "discriminator_optimizer", "normalizers", "policy_replay"):
        finite(state[name])
    assert state["optimizer"]["state"] and state["discriminator_optimizer"]["state"]
    experts = min(actors * controls, 4096)
    amp_factor = 10 if state["config"]["amp_normalization"] == "per_frame" else 1
    expected_counts = {"self_normalizer": transitions, "task_normalizer": transitions,
        "value_normalizer": transitions, "amp_normalizer": updates * (actors * controls + experts) * amp_factor}
    actual_counts = {}
    for name, dimension in (("self_normalizer", 357), ("task_normalizer", 12308),
            ("amp_normalizer", 195 if amp_factor == 10 else 1950), ("value_normalizer", 1)):
        normalizer = state["normalizers"][name]
        assert tuple(normalizer["mean"].shape) == (dimension,)
        assert tuple(normalizer["variance"].shape) == (dimension,)
        assert bool((normalizer["variance"] >= 0).all())
        actual_counts[name] = float(normalizer["count"])
        assert actual_counts[name] == expected_counts[name]
    metric_rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(metric_rows) == updates
    assert [item["update_count"] for item in metric_rows] == list(range(1, updates + 1))
    assert metric_rows[-1]["physical_transitions"] == transitions
    assert metric_rows[-1]["actual_normalizer_counts"] == actual_counts
    assert metric_rows[-1]["actual_physics_step"] - result["actual_initialized_physics_step"] == updates * controls * 4
    payload = run / f"rollout_update_{updates:06d}.pt"
    payload_before = identity(payload)
    saved = torch.load(payload, weights_only=True, map_location="cpu", mmap=True)
    rollout, diagnostics = saved["collection"]["rollout"], saved["collection"]["diagnostics"]
    finite(saved)
    for name, suffix in (("self_obs", (357,)), ("task_obs", (12308,)),
            ("amp_history", (10, 195)), ("actions", (69,)), ("next_values", ())):
        assert tuple(rollout[name].shape) == (controls, actors, *suffix)
    assert bool((diagnostics["physics_step_after"] - diagnostics["physics_step_before"] == 4).all())
    assert torch.equal(rollout["history_episode_ids"], rollout["episode_ids"].unsqueeze(-1).expand(-1, -1, 10))
    assert torch.equal(diagnostics["post_history_episode_ids"], diagnostics["post_episode_ids"].unsqueeze(-1).expand(-1, -1, 10))
    assert tuple(saved["expert"]["history"].shape) == (experts, 10, 195)
    assert tuple(saved["expert"]["actions"].shape) == (experts, 69)
    assert before == identity(checkpoint) and payload_before == identity(payload)
    report = {"schema": "shared_social.completed_training_CPU_audit.v1", "status": "passed",
        "reader": identity(Path(__file__)), "checkpoint": before, "rollout": payload_before,
        "updates": updates, "physical_transitions": transitions,
        "normalizer_counts": actual_counts, "finite_tensors_checked": finite_tensors,
        "expert_windows_drawn": updates * experts, "checkpoint_physical_resume_supported": False,
        "CUDA_initialized": torch.cuda.is_initialized(),
        "scope": "Completed-run counts, saved model/Adam/RMS/replay and final rollout consistency; no locomotion quality or paper equivalence claim. Actual process exit must be recorded separately."}
    assert report["CUDA_initialized"] is False
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
