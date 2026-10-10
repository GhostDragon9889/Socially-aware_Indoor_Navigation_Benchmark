"""Verify actual root terminal closure/seal without changing the frozen reader."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "docs/style_improvement_20261003/state_prior_actual_runs_v1"
TRAIN = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_v1"
SNAPSHOTS = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    freeze_path = HERE / "source_freeze_v1.json"
    audit_path = HERE / "terminal2000/audit.json"
    owner_path = RUNS / "terminal2000_seal.json"
    plan_path = ROOT / "docs/style_improvement_20261003/state_prior_training_preparation_v1/root_GPU_launch_commands.json"
    freeze, audit, owner, plan = [json.loads(path.read_text()) for path in (freeze_path, audit_path, owner_path, plan_path)]
    check("frozen_reader_and_dependencies_unchanged_readonly", all(record(row["path"]) == row
          and not Path(row["path"]).stat().st_mode & 0o222 for row in freeze["source_files"]))
    check("actual_complete_terminal_reader_passed", audit["schema"] == "public_state_prior.actual_training_checkpoint_audit.v1"
          and audit["status"] == "passed" and audit["phase"] == "terminal2000"
          and audit["actual_terminal_fresh2000_confirmed"] is True and audit["milestone_only_admission"] is False
          and audit["check_count"] == len(audit["checks"]) and all(row["passed"] is True for row in audit["checks"])
          and audit["auditor"] == record(HERE / "audit_state_prior_training.py")
          and audit["source_manifest"] == freeze["source_manifest"])
    check("root_owner_seal_actual_closed_status", owner["schema"] == "state_prior.actual_terminal_seal.v1"
          and owner["status"] == "sealed_after_actual_exit")
    for key, name, command, env in (("actual_launch", "fullfresh2000", plan["commands"]["fullfresh2000"], plan["env"]),
                                  ("actual_watcher_launch", "watch", plan["watch_command"], plan["CPU_watch_env"])):
        launch = owner[key]
        check(name + "_real_closed_root_record_exact", launch == json.loads((RUNS / (name + "_actual_launch.json")).read_text())
              and launch["schema"] == "state_prior.actual_launch.v1" and launch["name"] == name
              and launch["status"] == "closed" and type(launch["exit_code"]) is int and launch["exit_code"] == 0
              and type(launch["child_pid"]) is int and launch["child_pid"] > 0 and bool(launch["start_utc"]) and bool(launch["end_utc"])
              and launch["command"] == command and launch["env"] == env and launch["cwd"] == plan["cwd"]
              and launch["launch_plan_sha256"] == record(plan_path)["sha256"])
    check("closed_training_and_watcher_are_distinct_actual_processes", owner["actual_launch"]["child_pid"] != owner["actual_watcher_launch"]["child_pid"])
    check("root_owner_terminal_budget_actual", type(owner["actual_update_count"]) is int and owner["actual_update_count"] == 2000
          and type(owner["actual_physical_transitions"]) is int and owner["actual_physical_transitions"] == 16384000
          and owner["actual_complete_metrics_rows"] == 2000)
    selected = audit["checkpoints"]
    check("sole_real_final_checkpoint_expected_identity", len(selected) == 1 and selected[0]["updates"] == 2000
          and selected[0]["transitions"] == 16384000 and selected[0]["full_payload_resume_validated"] is True
          and selected[0]["checkpoint"]["path"] == str(SNAPSHOTS / "step_2000.pt")
          and selected[0]["checkpoint"]["sha256"] == "19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f")
    ledger = json.loads((TRAIN / "sealed_terminal/ledger.json").read_text())
    check("closed_watcher_ledger_source_copy_exact", (TRAIN / "sealed_terminal/ledger.json").read_bytes() == (SNAPSHOTS / "ledger.json").read_bytes()
          and ledger["missing"] == [] and ledger["targets"] == [500, 1000, 2000]
          and [row["update_count"] for row in ledger["snapshots"]] == [500, 1000, 2000])
    check("root_owner_snapshot_set_exact", [row["update_count"] for row in owner["snapshots"]] == [500, 1000, 2000])
    for sealed, transport in zip(owner["snapshots"], ledger["snapshots"], strict=True):
        path = Path(sealed["path"])
        actual = record(path)
        check("owner_snapshot_real_hash_readonly:" + path.name, path.parent == SNAPSHOTS and not path.is_symlink()
              and path.stat().st_nlink == 1 and not path.stat().st_mode & 0o222
              and all(actual[name] == sealed[name] == transport[name] for name in ("path", "bytes", "sha256"))
              and sealed["identical_source_bytes_verified"] is True and transport["identical_source_bytes_verified"] is True
              and sealed["source_bytes_modified"] is False)
    expected_paths = {str(TRAIN / "sealed_terminal" / name) for name in ("result.json", "metrics.jsonl", "ledger.json")}
    check("owner_terminal_metadata_exact_set", {row["path"] for row in owner["files"]} == expected_paths and len(owner["files"]) == 3)
    for sealed in owner["files"]:
        path = Path(sealed["path"])
        actual = record(path)
        check("owner_terminal_copy_real_hash_readonly:" + path.name, not path.is_symlink() and path.stat().st_nlink == 1
              and not path.stat().st_mode & 0o222 and all(actual[name] == sealed[name] for name in ("path", "bytes", "sha256"))
              and record(sealed["source"])["sha256"] == actual["sha256"] and Path(sealed["source"]).stat().st_size == actual["bytes"])
    check("actual_reader_result_metrics_match_owner_seal", audit["terminal_training_result"] == record(TRAIN / "sealed_terminal/result.json")
          and audit["actual_closed_metrics"] == record(TRAIN / "sealed_terminal/metrics.jsonl"))
    check("terminal_state_prior_counts_flags_exact", audit["state_prior_initialization_validated"] is True
          and audit["public_D_or_AMP_RMS_transferred"] is True and selected[0]["state_prior"]["explicit_flag"] is True
          and selected[0]["state_prior"]["current_AMP_count"] == selected[0]["RMS_counts"]["amp_normalizer"] == 24776000.)
    result = {"schema": "public_state_prior.actual_terminal_owner_seal_check.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "frozen_reader_modified": False,
              "verifier": record(__file__), "audit": record(audit_path), "auditor_source_freeze": record(freeze_path),
              "owner_terminal_seal": record(owner_path), "source_manifest": audit["source_manifest"],
              "checkpoint": selected[0]["checkpoint"], "checks": checks, "check_count": len(checks),
              "complete_terminal_state_admitted": True, "style_or_native_follow_on_gain_claim": False,
              "observed_runtime_finalization_not_inferred": True}
    output = HERE / "terminal_owner_seal_check_v1.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    output.chmod(0o444)
    print(json.dumps({"status": "passed", "checks": len(checks), "output": record(output)}))


if __name__ == "__main__":
    run()
