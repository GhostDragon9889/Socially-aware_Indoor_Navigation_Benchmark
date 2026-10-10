#!/usr/bin/env python3
"""Read recovery progress and observed child exits without Torch or an SDK."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def read_status(plan_path):
    plan = json.loads(plan_path.read_text())
    registration = json.loads(Path(plan["registration"]).read_text())
    run = Path(registration["stages"]["training"]["output"])
    owner_path = plan_path.parent / "owner_state.json"
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
    rows, incomplete_tail = [], False
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        lines = metrics.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                if index != len(lines) - 1:
                    raise ValueError("Incomplete nonterminal metric line")
                incomplete_tail = True
                break
            row = json.loads(line)
            if row["update_count"] != 12001 + index or row["recovery_segment_update"] != index + 1:
                raise ValueError("Recovery metrics are not a contiguous checkpoint-relative segment")
            rows.append(row)
    last = rows[-1] if rows else None
    training = next((step for step in owner.get("steps", []) if step["name"] == "training"), {})
    completion = plan_path.parent / "completion.json"
    return {"schema": "shared_social.recovery_status.v1",
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "owner_status": owner.get("status", "not_launched"),
        "phase": owner.get("steps", [{}])[-1].get("name") if owner.get("steps") else None,
        "retained_checkpoint_updates": 12000, "logged_effective_updates": last["update_count"] if last else 12000,
        "logged_recovery_segment_updates": len(rows), "planned_recovery_segment_updates": 500,
        "remaining_recovery_updates": 500 - len(rows),
        "logged_effective_physical_transitions": last["physical_transitions"] if last else 98304000,
        "logged_recovery_physical_transitions": len(rows) * 8192,
        "last_update_wall_s": last["update_wall_s"] if last else None,
        "last_reward_mean": last["reward_mean"] if last else None,
        "actual_training_exit_observed": "actual_exit_code" in training,
        "actual_training_exit_code": training.get("actual_exit_code"),
        "completion_exists": completion.exists(), "incomplete_metric_tail_ignored": incomplete_tail,
        "last_saved_checkpoint": next((p.name for p in sorted(run.glob("step_*.pt"), reverse=True)), None),
        "owner_pid_recorded": owner.get("owner_pid"), "training_pid_recorded": training.get("pid"),
        "process_liveness_or_exit_inferred_from_progress": False,
        "physical_world_restarted": True, "paper_effect_achieved": False,
        "error": owner.get("error"), "run_directory": str(run), "owner_state": str(owner_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(read_status(args.plan.resolve()), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
