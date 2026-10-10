"""CPU terminal admission for the real isolated2000 checkpoint and result.

Calls the unchanged milestone audit first, then binds terminal actual JSON.
No future result/checkpoint is synthesized and no simulator is started.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRAIN = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_v1"
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt"
AUDITOR = HERE / "audit_state_action_training_checkpoints_v1.py"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_terminal(report, result):
    """Reject partial/resumed/failed runs; preserve all actual fall counters."""
    rows = report.get("checkpoints", [])
    if (report.get("schema") != "state_action.actual_training_checkpoint_audit.v1" or
            report.get("status") != "passed" or len(rows) != 1 or rows[0].get("updates") != 2000):
        raise ValueError("Actual sole2000 full-payload admission required")
    row = rows[0]
    args = result.get("arguments", {})
    expected = {"num_envs": 256, "updates": 2000, "rollout_steps": 32, "replan_steps": 15,
        "episode_steps": 300, "terrain_extent_m": 192., "spawn_margin_m": 48.,
        "trajectory_source": "mixed_amass_procedural", "curriculum_mode": "smooth",
        "curriculum_speed_max_mps": 2., "curriculum_accel_max_mps2": 1.5,
        "checkpoint_every": 500, "synchronized_replanning": True, "physics_only": True,
        "seed": 7, "resume": None, "from_scratch": False, "expert_sampling": None}
    if (result.get("status") != "passed" or result.get("error") is not None or
            type(result.get("update_count")) is not int or result["update_count"] != 2000 or
            type(result.get("physical_transition_count")) is not int or result["physical_transition_count"] != 16384000 or
            result.get("group_transition_exposure") != row["group_exposure"] or
            result.get("fall_resets") != row["cumulative_training_falls_at_milestone"] or
            result.get("time_limit_resets") != row["cumulative_training_time_limits_at_milestone"]):
        raise ValueError("Actual terminal status/counts/exposure/falls differ from sealed2000")
    for key, value in expected.items():
        if key == "expert_sampling":
            if result.get("training_config", {}).get(key) is not None:
                raise ValueError("Uniform expert protocol changed")
            continue
        if key not in args or type(args[key]) is not type(value) or args[key] != value:
            raise ValueError("Actual fresh2000 recipe differs:" + key)
    if (result.get("training_config", {}).get("discriminator_input") != "state_action" or
            result["training_config"].get("warm_start_discriminator") is not False or
            result["training_config"].get("amp_normalization") != "per_history_coordinate"):
        raise ValueError("Actual terminal state-action/RMS recipe differs")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only fresh terminal report directory required")
    frozen = json.loads((HERE / "state_action_isolated_checkpoint500_review_v1/source_freeze.json").read_text())
    frozen_tool = next(item for item in frozen["files"] if item["path"].endswith(AUDITOR.name))
    actual_tool = record(AUDITOR)
    if actual_tool["sha256"] != frozen_tool["sha256"] or actual_tool["bytes"] != frozen_tool["bytes"]:
        raise ValueError("The1077-check milestone auditor changed")
    if not CHECKPOINT.is_file() or not (TRAIN / "result.json").is_file():
        raise ValueError("Actual terminal checkpoint and result do not yet exist")
    result_path = TRAIN / "result.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    # No allocation/load of the future weight is attempted until terminal JSON
    # explicitly says the complete fresh2000 run succeeded.
    if result.get("status") != "passed" or result.get("update_count") != 2000:
        raise ValueError("Real full2000 training has not completed successfully")
    args.output.mkdir()
    (args.output / "actual_terminal_result.json").write_bytes(result_bytes)
    with (args.output / "full_payload.log").open("x") as stream:
        subprocess.run([sys.executable, str(AUDITOR), "--checkpoint", str(CHECKPOINT),
            "--output", str(args.output / "full_payload")], cwd=ROOT, env=dict(os.environ,
            CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT) + ":" + str(ROOT / "src"),
            OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = deepcopy(json.loads((args.output / "full_payload/audit.json").read_text()))
    row = validate_terminal(report, result)
    actual_result = record(result_path)
    if actual_result["bytes"] != len(result_bytes) or actual_result["sha256"] != hashlib.sha256(result_bytes).hexdigest():
        raise ValueError("Actual terminal result changed during the CPU payload audit")
    report["checks"].append({"name": "actual_terminal_fresh2000_status_recipe_counters_and_saved_exposure", "passed": True})
    report["check_count"] = len(report["checks"])
    report["terminal_training_result"] = actual_result
    report["terminal_wrapper"] = record(__file__)
    report["actual_terminal_fresh2000_confirmed"] = True
    report["author_model_effect_or_convergence_claim"] = False
    # Independently run the exact gate used by the future executor. This
    # catches schema drift without requiring an SDK or constructing a model.
    spec = importlib.util.spec_from_file_location("actual_terminal_registered_executor",
        HERE / "run_isolated_state_action_evaluation_v1.py")
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    executor.validate_admission(report, row["checkpoint"])
    report["checks"].append({"name": "actual_registered_evaluation_adapter_accepts_same_terminal_schema", "passed": True})
    report["check_count"] = len(report["checks"])
    with (args.output / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": report["check_count"], "checkpoint": row["checkpoint"],
        "updates": 2000, "transitions": 16384000}))


if __name__ == "__main__":
    main()
