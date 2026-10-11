#!/usr/bin/env python3
"""Read the registered episode training without loading weights or an SDK."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from social_navigation.artifact_identity import artifact_path, same_artifact

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/style_improvement_20261008/shared_social_episode_after_recovery_v4/owner_plan.json"
PLAN_SHA = "8e8314cb35d40eee4b095ac6a05fd7601e92d57027ba56a0ba20e155e3f79c4d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            finite_numbers(item)
    elif type(value) in (float, int):
        require(math.isfinite(value), "Nonfinite training metric")


def read_metrics(path, runtime):
    """Count only complete contiguous rows, including original control budgets."""
    rows, incomplete = [], False
    payload = path.read_bytes() if path.exists() else b""
    for raw in payload.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            incomplete = True
            break
        row = json.loads(raw)
        finite_numbers(row)
        update = len(rows) + 1
        expected = {
            "update_count": update,
            "physical_transitions": update * runtime["actors"] * runtime["rollout_controls"],
            "completed_physical_controls": update * runtime["rollout_controls"],
        }
        require(all(type(row.get(key)) is int and row[key] == value for key, value in expected.items()),
                "Noncontiguous update/physical/control sequence")
        require(update <= runtime["updates"], "Metrics exceed the registered training budget")
        require(type(row.get("update_wall_s")) in (int, float) and row["update_wall_s"] > 0,
                "Require a positive finite completed-update duration")
        rows.append(row)
    return rows, incomplete


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    try:
        command = directory.joinpath("cmdline").read_bytes().split(b"\0")
        stat = directory.joinpath("stat").read_text()
    except FileNotFoundError:
        return None
    fields = stat[stat.rfind(")") + 2:].split()
    return {"pid": pid, "parent_pid": int(fields[1]), "start_ticks": fields[19],
            "command": [item.decode() for item in command if item]}


def read_status(plan_path=PLAN):
    plan_path = plan_path.resolve()
    payload = plan_path.read_bytes()
    require(plan_path == PLAN and not plan_path.is_symlink() and plan_path.stat().st_mode & 0o222 == 0
            and hashlib.sha256(payload).hexdigest() == PLAN_SHA, "Require the fixed readonly episode v4 plan")
    plan = json.loads(payload)
    selected = plan["stages"][-1]
    runtime = selected["runtime"]
    run = artifact_path(selected["output_directory"], root=ROOT)
    configuration = json.loads((run / "resolved_configuration.json").read_text())
    require(configuration["entry_profile"]["runtime"] == runtime, "Saved runtime differs from the registered plan")
    state_path = plan_path.parent / "owner_state.json"
    state = json.loads(state_path.read_text())
    require(same_artifact(state["plan"], {"path": str(plan_path), "bytes": len(payload), "sha256": PLAN_SHA}, root=ROOT),
            "Owner state selects another plan")
    rows, incomplete = read_metrics(run / "metrics.jsonl", runtime)
    step = next((item for item in state.get("steps", []) if item["name"] == "pilot2000_training"), {})
    owner = process_identity(state["owner_pid"])
    child = process_identity(step["pid"]) if "pid" in step else None
    expected_owner = [step["command"][0], str(ROOT / "scripts/own_shared_social_episode_pipeline.py"),
                      "--plan", str(plan_path)] if step else None
    owner_observed = owner is not None and owner["command"] == expected_owner
    child_observed = (owner_observed and child is not None and child["command"] == step["command"]
                      and child["parent_pid"] == owner["pid"])
    # The independently running comparison observer retained the owner's start
    # ticks at launch; use them to reject a PID reused with the same command.
    observer_path = ROOT / "docs/style_improvement_20261008/shared_social_episode_comparison_preparation_v1/comparison_owner_state.json"
    observer = json.loads(observer_path.read_text()) if observer_path.exists() else {}
    retained = observer.get("observed_episode_owner")
    if retained and owner_observed:
        owner_observed = all(owner[key] == retained[key] for key in ("pid", "start_ticks", "command"))
        child_observed = child_observed and owner_observed
    observed_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    recent = rows[-20:]
    rate = sum(row["update_wall_s"] for row in recent) / len(recent) if recent else None
    remaining = runtime["updates"] - len(rows)
    finish = observed_at + timedelta(seconds=remaining * rate) if child_observed and rate is not None else None
    completion_path = plan_path.parent / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    actual_exit = step.get("actual_exit_code")
    require(actual_exit is None or type(actual_exit) is int, "Owner exit observation must be an integer")
    return {
        "schema": "shared_social.episode_status.v1", "observed_at": observed_at.isoformat(),
        "owner_reported_status": state["status"], "phase": state["steps"][-1]["name"] if state["steps"] else None,
        "owner_pid_recorded": state["owner_pid"], "training_pid_recorded": step.get("pid"),
        "owner_command_and_start_observed": owner_observed, "training_command_and_parent_observed": child_observed,
        "host_process_observation_required": True,
        "logged_updates": len(rows), "planned_updates": runtime["updates"], "remaining_updates": remaining,
        "logged_physical_transitions": rows[-1]["physical_transitions"] if rows else 0,
        "planned_physical_transitions": runtime["updates"] * runtime["actors"] * runtime["rollout_controls"],
        "incomplete_metric_tail_ignored": incomplete, "complete_metric_rows_finite_and_contiguous": True,
        "recent_mean_update_wall_s": rate, "estimated_remaining_training_hours": remaining * rate / 3600 if finish else None,
        "estimated_training_finish_Asia_Shanghai": finish.isoformat() if finish else None,
        "estimate_is_completion_guarantee": False,
        "checkpoint_files_observed_unvalidated": [{"name": p.name, "bytes": p.stat().st_size}
                                                  for p in sorted(run.glob("step_*.pt"))],
        "weights_loaded": False, "checkpoint_completeness_inferred_from_presence": False,
        "actual_training_exit_code_reported_by_owner": actual_exit,
        "exit_observation": step.get("exit_observation"), "completion_exists": completion is not None,
        "completion_owner_reported_status": completion.get("status") if completion else None,
        "comparison_owner_reported_status": observer.get("status"),
        "execution_success_or_quality_inferred_from_progress": False, "training_started_or_restarted": False,
        "run_directory": str(run), "owner_state": str(state_path), "error": state.get("error"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args(argv)
    print(json.dumps(read_status(args.plan), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
