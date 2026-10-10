#!/usr/bin/env python3
"""Read shared AMP training progress without loading tensors, weights or Isaac Sim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_progress(run_directory, pid=None):
    run = run_directory.resolve()
    configuration = json.loads((run / "resolved_configuration.json").read_text())
    runtime = configuration["entry_profile"]["runtime"]
    rows = []
    incomplete_tail = False
    # A live writer can be observed between write and flush. Only complete lines
    # are counted; never turn a partially written metric into a finished update.
    payload = (run / "metrics.jsonl").read_bytes()
    lines = payload.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            incomplete_tail = True
            break
        rows.append(json.loads(line))
    actors, controls = runtime["actors"], runtime["rollout_controls"]
    for update, item in enumerate(rows, 1):
        if (item["update_count"] != update
                or item["physical_transitions"] != update * actors * controls
                or item["completed_physical_controls"] != update * controls):
            raise ValueError("Metric update/physical transition sequence is inconsistent")
    last = rows[-1] if rows else None
    report = {"run_directory": str(run), "status": "incomplete_execution_state_unknown",
        "logged_updates": len(rows), "planned_updates": runtime["updates"],
        "logged_physical_transitions": last["physical_transitions"] if last else 0,
        "planned_physical_transitions": runtime["updates"] * actors * controls,
        "incomplete_metric_tail_ignored": incomplete_tail, "weights_loaded": False,
        "quality_or_process_exit_inferred_from_metrics": False}
    result_path = run / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        report.update(status="entry_report_available", entry_report_status=result["status"],
            completed_updates_in_entry_report=result["completed_updates"],
            cleanup_succeeded=result["cleanup_succeeded"], error=result.get("error"))
    if pid is not None:
        command_path = Path(f"/proc/{pid}/cmdline")
        try:
            command = command_path.read_bytes().split(b"\0")
        except FileNotFoundError:
            report["selected_process_observed"] = False
        else:
            values = [value.decode() for value in command if value]
            matched = False
            for index, value in enumerate(values[:-1]):
                if value == "--output" and Path(values[index + 1]).resolve() == run:
                    matched = any(Path(argument).name == "train_shared_social.py" for argument in values)
            report["selected_process_observed"] = matched
            if matched and not result_path.exists():
                report["status"] = "running_process_observed"
        report["selected_pid"] = pid
    if last:
        report["last_metric"] = {key: last.get(key) for key in (
            "update_count", "reward_mean", "update_wall_s", "actual_physics_step",
            "actual_normalizer_counts", "driver_reason_counts", "optimizer_minibatches",
            "policy_target_kl_stopped", "disc_replay_exposures_this_update")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.pid is not None and args.pid <= 0:
        parser.error("PID must be positive")
    print(json.dumps(read_progress(args.run_directory, args.pid), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
