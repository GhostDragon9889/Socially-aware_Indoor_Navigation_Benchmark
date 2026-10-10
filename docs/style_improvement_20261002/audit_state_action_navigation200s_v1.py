"""CPU audit of an actual isolated state-action four-person200s capture.

Reuses frozen f851 raw measurement and ee0 idle/phase formulas unchanged.
Only their epoch-specific source admission is supplied by the new v3 reader.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = {
    "audit_ground_support_capture.py": "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f",
    "summarize_navigation_style.py": "ee0d13c3f345cf11aa2981002c21b070bf53eb4b6129d86d4a6f851443ac56d7",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-admission", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only fresh output report directory required")
    for name, expected in FROZEN.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen raw/idle formula changed:" + name)
    evidence = load_module("actual_state_action_navigation_evidence", HERE / "state_action_evaluation_evidence_v1.py")
    raw = load_module("unchanged_raw_ground_support_measurement", HERE / "audit_ground_support_capture.py")
    idle = load_module("unchanged_full200_navigation_idle_formula", HERE / "summarize_navigation_style.py")
    result = json.loads((args.input / "result.json").read_text())
    registration = json.loads((HERE / "state_action_evaluation_registration_v1/registration.json").read_text())
    evidence.validate_protocol(result, "navigation", registration)
    audit = raw.Audit()
    source = evidence.admit_actual_evaluation(result, "navigation", args.checkpoint_admission,
                                            args.import_report, require_check=audit.require)
    # Isolated source admission replaces only the old report's epoch hook.
    # Raw normal forces/COM/clock/episode calculations remain original f851.
    ledger_bytes, event_bytes = source.pop("ledger_bytes"), source.pop("event_prefix_bytes")
    original = raw.verify_sources
    try:
        raw.verify_sources = lambda *_args, **_kwargs: source
        report = raw.audit_capture(args.input, audit)
    finally:
        raw.verify_sources = original
    report.update(schema="state_action.actual_navigation_ground_audit.v1", status="passed", checks=audit.checks,
        check_count=len(audit.checks), wrapper=evidence.record(__file__),
        frozen_measurement_tool=evidence.record(HERE / "audit_ground_support_capture.py"),
        SDK_started=False, GPU_used=False, sealed34_outputs_read=False,
        quality_pass=None, quality_evaluated=False, execution_pass_is_quality_pass=False)
    stationary = idle.stagnation(args.input, report)
    args.output.mkdir(parents=True)
    (args.output / "GPU_import_report_as_observed.json").write_bytes(ledger_bytes)
    (args.output / "GPU_import_events_as_observed.jsonl").write_bytes(event_bytes)
    write(args.output / "audit.json", report)
    write(args.output / "stagnation.json", stationary)
    # Avoid the old markdown's current-source execution command/provenance
    # wording: the new audit source is explicitly isolatedv3/old146.
    (args.output / "README.md").write_text(
        "# Actual isolated state-action200s CPU recording audit\n\n"
        "Read-only actual4×6000 recording; unchanged f851 normal-force/COM/clock and ee0 idle formulas. "
        "Source admitted from exact926 immutable isolated files and actual evaluation import ledger, "
        "not the current checkout or historical127/138/154 epoch.\n\n"
        "All falls/terminations/reset frames retained; only cross-episode derivative intervals excluded. "
        "A passed audit proves recording consistency, not stable gait or paper quality. "
        "Foot-origin supported speed is a rolling/sliding proxy, not contact-patch slip. "
        "AMASS reference contact/true COM unavailable.\n\n"
        "See [audit.json](audit.json) and [stagnation.json](stagnation.json).\n")
    print(json.dumps({"status": "passed", "checks": len(audit.checks), "counts": report["counts"],
                      "output": str(args.output), "quality_evaluated": False}))


if __name__ == "__main__":
    main()
