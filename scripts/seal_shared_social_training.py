#!/usr/bin/env python3
"""Seal completed Pop-Art outputs using separately observed process exits.

Exit metadata must come from the external process owner. This script checks
the result/audit/artifact bindings; it cannot itself observe a prior exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from CrowdSim.humanoid.runtime import social_popart_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--exit-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    external = json.loads(args.exit_metadata.read_text())
    if (external.get("schema") != "shared_social.observed_training_and_audit_exits.v1"
            or external.get("run_directory") != str(run)
            or any(external.get(key, {}).get("exit_code") != 0 for key in ("training", "CPU_audit"))):
        raise ValueError("Require the external owner's actual successful training and CPU audit exits")
    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    audit_path = args.audit_report.resolve()
    audit = json.loads(audit_path.read_text())
    updates = result["completed_updates"]
    checkpoint = run / f"step_{updates:06d}.pt"
    rollout = run / f"rollout_update_{updates:06d}.pt"
    numeric = social_popart_training.describe()
    if (result.get("status") != "passed_local_training" or result.get("cleanup_succeeded") is not True
            or result.get("cleanup_errors") or result.get("source_profile_before_after_equal") is not True
            or result["profile"]["runtime"]["updates"] != updates
            or result["canonical_entry"]["numeric_migration"] != numeric
            or audit.get("status") != "passed" or audit.get("CUDA_initialized") is not False
            or audit.get("updates") != updates or audit.get("explicit_value_normalization") != numeric
            or audit.get("physical_transitions") != result["actual_training_transitions"]
            or audit.get("checkpoint") != record(checkpoint) or audit.get("rollout") != record(rollout)):
        raise ValueError("Terminal result, actual CPU audit and Pop-Art identities disagree")
    files = [result_path, checkpoint, rollout, run / "metrics.jsonl", run / "resolved_configuration.json", audit_path]
    for path in files:
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1 or not path.is_relative_to(ROOT):
            raise ValueError("Seal only independent regular project-owned completion artifacts")
    for path in files:
        path.chmod(0o444)
    report = {"schema": "shared_social.actual_training_execution.v1", "status": "completed_training_and_CPU_audit",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "terminal_exit_code": 0,
        "terminal_tool_chunk": external["training"]["chunk_id"], "terminal_session_id": external["training_session_id"],
        "CPU_audit_exit_code": 0, "CPU_audit_terminal_tool_chunk": external["CPU_audit"]["chunk_id"],
        "run_directory": str(run), "external_exit_metadata": record(args.exit_metadata),
        "result_identity": record(result_path), "checkpoint_identity": record(checkpoint), "audit_identity": record(audit_path),
        "completed_updates": updates, "actual_training_transitions": result["actual_training_transitions"],
        "actual_expert_windows_drawn": result["final_expert_description"]["sampled_windows"],
        "sealing_entry": record(__file__), "quality_or_author_equivalence_established": False, "default_model_promoted": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "completed_updates": updates, "checkpoint_sha256": report["checkpoint_identity"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
