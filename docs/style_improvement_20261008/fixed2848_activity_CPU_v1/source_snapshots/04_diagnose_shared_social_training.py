#!/usr/bin/env python3
"""CPU analysis of actual completed Social AMP reward/critic stability.

Reads saved terminal rollout and metrics only; no NN or SDK construction.
The GAE decomposition is a numerical diagnostic, not evidence of causation or
an independent locomotion score. Requires the actual training exit receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402

CONTRACT = ROOT / "CrowdSim/humanoid/training/contracts.py"
CONTRACT_SHA = "26c7379ba60842c67f062fa5c1a3459ddfb6e0843e1b4952e1d078aa6f78cdad"


def identity(path):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or before.st_nlink != 1:
        raise ValueError("Require an independent ordinary input")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Input changed while hashing")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest}


def windows(rows, controls_per_update, size=100):
    result = []
    for start in range(0, len(rows), size):
        selected = rows[start:start + size]
        first_counts = rows[start - 1]["driver_reason_counts"] if start else {}
        last_counts = selected[-1]["driver_reason_counts"]
        result.append({"first_update": start + 1, "last_update": start + len(selected),
            "actor_controls": len(selected) * controls_per_update,
            "reason_flags": {name: count - first_counts.get(name, 0) for name, count in last_counts.items()},
            "mean_training_reward": sum(row["reward_mean"] for row in selected) / len(selected),
            "mean_bootstrapped_GAE_return": sum(row["return_mean"] for row in selected) / len(selected),
            "mean_applied_PPO_minibatches": sum(row["optimizer_minibatches"] for row in selected) / len(selected),
            "KL_stop_updates": sum(row["policy_target_kl_stopped"] for row in selected)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run with CUDA_VISIBLE_DEVICES='' for CPU-only diagnostics")
    _, _, admission = admit_training(args.execution_receipt.resolve())
    audit = json.loads(Path(admission["audit"]["path"]).read_text())
    result = json.loads(Path(admission["result"]["path"]).read_text())
    rollout_path = Path(audit["rollout"]["path"])
    if identity(rollout_path) != audit["rollout"] or identity(CONTRACT)["sha256"] != CONTRACT_SHA:
        raise ValueError("Audited rollout or original GAE implementation changed")
    run = Path(admission["result"]["path"]).parent
    metric_path = run / "metrics.jsonl"
    metric_identity = identity(metric_path)
    rows = [json.loads(row) for row in metric_path.read_text().splitlines()]
    if len(rows) != result["completed_updates"] or [row["update_count"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Actual metrics are incomplete")

    import torch
    from CrowdSim.humanoid.training.contracts import generalized_advantages

    torch.set_num_threads(4)
    saved = torch.load(rollout_path, weights_only=True, map_location="cpu", mmap=True)
    rollout, metadata = saved["collection"]["rollout"], saved["collection"]["metadata"]
    config = result["actual_configuration"]["configuration"]
    reward, value, next_value = (rollout[name] for name in ("rewards", "values", "next_values"))
    masks = rollout["terminated"], rollout["truncated"]
    numerics = {"gamma": config["gamma"], "gae_lambda": config["gae_lambda"]}
    advantage, returns = generalized_advantages(reward, value, next_value, *masks, **numerics)
    zero = torch.zeros_like(reward)
    _, reward_component = generalized_advantages(reward, zero, zero, *masks, **numerics)
    _, critic_component = generalized_advantages(zero, value, next_value, *masks, **numerics)
    comparison = {"CPU_return_mean": float(returns.mean()), "saved_GPU_return_mean": rows[-1]["return_mean"],
                  "CPU_advantage_std": float(advantage.std(unbiased=False)), "saved_GPU_advantage_std": rows[-1]["advantage_std"],
                  "absolute_tolerance": 1e-5, "relative_tolerance": 1e-6}
    # CPU/GPU float32 reductions need not have the same accumulation order.
    # Preserve exact differences and require a declared small relative error.
    comparison["return_mean_delta"] = comparison["CPU_return_mean"] - comparison["saved_GPU_return_mean"]
    comparison["advantage_std_delta"] = comparison["CPU_advantage_std"] - comparison["saved_GPU_advantage_std"]
    if any(abs(comparison[name + "_delta"]) > 1e-5 + 1e-6 * abs(comparison["saved_GPU_" + name])
           for name in ("return_mean", "advantage_std")):
        raise ValueError("Saved terminal rollout does not reproduce the actual GAE metrics")

    def distribution(values):
        values = values.flatten().double()
        if values.device.type != "cpu" or not torch.isfinite(values).all():
            raise ValueError("Nonfinite or non-CPU actual diagnostic input")
        return {"count": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            "quantiles_01_50_99": torch.quantile(values, values.new_tensor([.01, .5, .99])).tolist()}

    frozen = metadata["frozen_rms"]["value_normalizer"]
    runtime = result["profile"]["runtime"]
    report = {"schema": "shared_social.actual_terminal_stability_diagnostic.v1", "status": "passed",
        "reader": identity(Path(__file__)), "training_admission": admission, "rollout": audit["rollout"],
        "metrics": metric_identity, "GAE_implementation": identity(CONTRACT), "numerics": numerics,
        "window_size_updates": 100, "training_windows": windows(rows, runtime["actors"] * runtime["rollout_controls"]),
        "terminal_rollout": {name: distribution(values) for name, values in (
            ("actual_reward", reward), ("pre_update_value", value), ("pre_reset_next_value", next_value),
            ("GAE_return", returns), ("reward_component", reward_component), ("critic_bootstrap_component", critic_component))},
        "GAE_decomposition_max_float_error": float((returns - reward_component - critic_component).abs().max()),
        "CPU_GPU_GAE_scalar_comparison": comparison,
        "pre_update_value_RMS": {key: tensor.tolist() for key, tensor in frozen.items()},
        "terminal_reason_flags": {name: int(rollout[name].sum()) for name in ("terminated", "truncated")},
        "CUDA_initialized": torch.cuda.is_initialized(), "NN_constructed": False, "SDK_started": False,
        "causal_training_bug_proven": False, "independent_quality_evaluated": False,
        "scope": "Training reason flags and actual saved GAE reward/critic decomposition. Does not substitute for full22 or navigation/style evaluation."}
    if report["CUDA_initialized"] or identity(rollout_path) != audit["rollout"] or identity(metric_path) != metric_identity:
        raise ValueError("CPU-only or frozen input contract failed")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "terminal_rollout": report["terminal_rollout"],
                      "CUDA_initialized": False, "independent_quality_evaluated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
