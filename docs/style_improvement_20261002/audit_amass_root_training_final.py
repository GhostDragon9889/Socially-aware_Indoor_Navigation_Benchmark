"""CPU terminal training audit; uses only a sealed checkpoint/result/metrics.

The existing frozen checkpoint auditor supplies full-source, physical/logical
resume and finite-bundle gates. This adds independent logged-update accounting
and exact optimizer/normalizer/replay/RNG serialization diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from CrowdSim.humanoid.amass_corpus import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def validate_metadata(state, result, rows):
    config, n, updates = state["config"], result["arguments"]["num_envs"], state["update_count"]
    batch = n * result["arguments"]["rollout_steps"]
    assert [row["update_count"] for row in rows] == list(range(1, updates + 1))
    assert [row["physical_transitions"] for row in rows] == [i * batch for i in range(1, updates + 1)]
    assert state["physical_transition_count"] == updates * batch
    assert rows[-1]["fall_resets"] == result["fall_resets"]
    assert rows[-1]["time_limit_resets"] == result["time_limit_resets"]
    assert result["arguments"]["resume"] is None  # Separate fresh-training contract.
    for row in rows:
        assert all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
        assert row["diffusion_plans"] == 0 and row["planner_forward_calls"] == 0
    init = result["initialization"]
    cap = config["warm_start_normalizer_count_cap"]
    initial_counts = {key: min(value, cap) if cap is not None else value
                      for key, value in init["normalizer_counts_before_cap"].items()}
    expert_batch = min(batch, 4096)
    history_factor = config["history_frames"] if config["amp_normalization"] == "per_frame" else 1
    added = {"self_normalizer": batch * updates, "task_normalizer": batch * updates,
             "amp_normalizer": (batch + expert_batch) * history_factor * updates,
             "value_normalizer": batch * updates}
    expected = {}
    for key in initial_counts:
        enabled = config["normalize_values"] if key == "value_normalizer" else config["normalize_observations"]
        expected[key] = initial_counts[key] + (added[key] if enabled else 0)
    moments = {}
    for key, value in state["normalizers"].items():
        assert float(value["count"]) == expected[key]
        assert bool(torch.isfinite(value["mean"]).all()) and bool(torch.isfinite(value["variance"]).all())
        assert bool((value["variance"] >= 0).all())
        deviation = value["variance"].sqrt()
        moments[key] = {"dimensions": len(value["mean"]), "count": float(value["count"]),
            "expected_count": expected[key], "initial_count_after_cap": initial_counts[key],
            "mean_abs_max": float(value["mean"].abs().max()), "std_min": float(deviation.min()),
            "std_median": float(deviation.median()), "std_max": float(deviation.max())}
        if key == "task_normalizer":
            height = deviation[-1024:]
            moments[key]["last1024_height_std"] = {"min": float(height.min()), "median": float(height.median()), "max": float(height.max())}
    optimizer_steps = {}
    for key, metric in (("optimizer", "optimizer_minibatches"), ("discriminator_optimizer", "discriminator_minibatches")):
        expected_steps = sum(row[metric] for row in rows)
        optimizer = state[key]
        ids = [i for group in optimizer["param_groups"] for i in group["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == set(optimizer["state"])
        steps = [float(value["step"]) for value in optimizer["state"].values()]
        assert all(step == expected_steps for step in steps)
        optimizer_steps[key] = {"parameter_states": len(steps), "sum_logged_optimizer_minibatches": expected_steps,
                                "all_Adam_steps_equal_logged_count": True,
                                "learning_rates": [group["lr"] for group in optimizer["param_groups"]]}
    replay = state["policy_replay"]
    assert replay["size"] == rows[-1]["replay_size"] == min(replay["accepted"], replay["capacity"])
    assert replay["capacity"] == config["discriminator_replay_capacity"] >= batch
    assert replay["cursor"] == replay["accepted"] % replay["capacity"]
    assert replay["history"].shape == (replay["size"], config["history_frames"], config["amp_frame_dim"])
    assert replay["actions"].shape == (replay["size"], 69)
    inactive = torch.tensor(config["action_mask"]) == 0
    assert not bool((replay["actions"][:, inactive] != 0).any())
    rng = state["rng"]
    random.Random().setstate(rng["python"])
    native_numpy = rng["numpy"]
    numpy_validator = np.random.RandomState()
    numpy_validator.set_state((native_numpy[0], np.asarray(native_numpy[1], dtype=np.uint32),
                               native_numpy[2], native_numpy[3], native_numpy[4]))
    torch.Generator(device="cpu").set_state(rng["torch_cpu"])
    assert all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
               and value.ndim == 1 and value.numel() > 0 for value in rng["torch_cuda"])
    return {"normalizers": moments, "optimizers": optimizer_steps,
        "policy_replay": {key: replay[key] for key in ("capacity", "size", "cursor", "accepted", "history_shape")},
        "frozen_applied_action_coordinates_in_replay_exact_zero": int(inactive.sum()),
        "RNG": {"python_local_validator_accepted": True, "numpy_local_validator_accepted": True,
                "torch_CPU_local_generator_accepted": True, "CUDA_state_count": len(rng["torch_cuda"]),
                "CUDA_serialized_uint8_bytes": [value.numel() for value in rng["torch_cuda"]],
                "CUDA_RNG_not_restored_in_CPU_auditor": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        parser.error("Explicit empty CUDA visibility and fresh output are required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert args.checkpoint.name == "step_1000.pt", "Only immutable terminal step1000 is accepted"
    result = json.loads(args.result.read_text())
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    assert result["status"] == "passed" and result["update_count"] == len(rows) == 1000
    assert result["physical_transition_count"] == 8192000
    before = {key: record(path) for key, path in (("checkpoint", args.checkpoint), ("result", args.result), ("metrics", args.metrics))}
    core_path = args.output.with_name(args.output.stem + "_checkpoint_core.json")
    core_log = args.output.with_name(args.output.stem + "_checkpoint_core.log")
    with core_log.open("x") as stream:
        subprocess.run([sys.executable, str(ROOT / "docs/style_improvement_20261002/audit_amass_root_actual_checkpoint.py"),
            "--checkpoint", str(args.checkpoint), "--result", str(args.result), "--output", str(core_path)],
            check=True, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    metadata = validate_metadata(state, result, rows)
    phases = []
    prior_fall = prior_timeout = 0
    for start in (0, 250, 500, 750):
        subset = rows[start:start + 250]
        last = subset[-1]
        fields = ("reward_mean", "trajectory_rewards_mean", "amp_rewards_mean", "direction_rewards_mean", "energy_rewards_mean",
                  "forward_velocity_mean", "backwards_fraction", "disc_expert_probability", "disc_live_policy_probability", "value_loss")
        phases.append({"updates": [start + 1, start + 250], "transitions": 2048000,
            "fall_resets": last["fall_resets"] - prior_fall, "clip_end_and_cap_resets": last["time_limit_resets"] - prior_timeout,
            "arithmetic_update_means": {key: float(np.mean([row[key] for row in subset])) for key in fields}})
        prior_fall, prior_timeout = last["fall_resets"], last["time_limit_resets"]
    after = {key: record(path) for key, path in (("checkpoint", args.checkpoint), ("result", args.result), ("metrics", args.metrics))}
    assert before == after
    document = {"schema": "amass_root_terminal_training_metadata_audit.v1", "status": "passed", "CPU_only": True,
        "SDK_started": False, "GPU_used": False, "heldout_outputs_read": False, "inputs": after,
        "core_resume_source_and_finite_audit": record(core_path), "updates": 1000, "transitions": 8192000,
        "actual_control_steps": result["actual_control_steps"], "fall_resets": result["fall_resets"],
        "time_limit_resets": result["time_limit_resets"], "curriculum_statistics": result["curriculum"]["statistics"],
        "serialized_training_metadata": metadata, "phases": phases, "script": record(__file__),
        "limits": ["Optimizer/normalizer/RNG admission is not a convergence or locomotion-quality threshold.",
                   "No CUDA RNG restoration or PhysX replay was performed by this CPU auditor.",
                   "The selected34 test captures remain unopened."]}
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "updates": 1000, "transitions": 8192000}))


if __name__ == "__main__":
    main()
