"""Metadata-only default62c4/newDP adapter around the unchanged recorded raw mathematics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from physical_evidence import HERE, admit_capture, load, record, registration, require, source_binding


def audit(plan, phase):
    case = plan["cases"][phase]
    admission = admit_capture(plan, phase)  # Before any capture array is opened.
    run = Path(case["output"])
    raw = load("default_newDP_original_fixture_raw", plan["math"]["fixture_raw"]["path"])
    original = raw.verify_sources
    try:
        raw.verify_sources = lambda *_args, **_kwargs: admission
        checks = raw.Audit()
        report = raw.audit_capture(run, checks)
    finally:
        raw.verify_sources = original
    report["checks"] = checks.checks
    counts, contacts = report["counts"], report["original_obstacle_peer_channels"]
    gate = {"controls": counts["controls"], "actors": counts["actors"], "duration_s": counts["duration_s"],
            "falls": counts["falls"], "static_actor_controls": contacts["static_contact_actor_frames"],
            "peer_pair_controls": contacts["peer_contact_unordered_pair_frames"]}
    proof = {"original_fixture_math_file": plan["math"]["fixture_raw"], "verify_sources_only_replaced_after_actual_source_admission": True,
             "arrays_force_COM_contacts_falls_counters_math_changed": False}
    require(gate["controls"] == 6000 and gate["actors"] == case["actors"] and gate["duration_s"] == 200.0,
            "Complete registered200s denominator required")
    report.update(schema="default_newDP.independent_physical_CPU_audit.v1", status="passed", phase=phase,
                  actual_capture_independently_audited=True, physical_gate_counts=gate,
                  physical_zero_gate_passed=all(gate[key] == 0 for key in ("falls", "static_actor_controls", "peer_pair_controls")),
                  actual_source_admission=admission, registration=record(HERE / "registration.json"),
                  execution_reader=record(__file__), mathematics_adapter_proof=proof,
                  all_failed_contact_idle_reset_samples_retained=True, style_score_used_for_admission=False,
                  CPU_only_reader=True, GPU_used_by_reader=False, SDK_started_by_reader=False,
                  no_default_promotion_claimed=True, source_post_observation=source_binding(plan))
    require(report["source_post_observation"] == {key: admission[key] for key in report["source_post_observation"]}, "Source changed during raw read")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "Raw reader imported model/SDK")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixture_old", "fixture_new"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve earlier actual audit")
    try:
        report = audit(registration(), args.phase)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"schema": "default_newDP.independent_physical_CPU_audit.v1", "status": "failed", "phase": args.phase,
                       "failure": repr(error), "actual_capture_independently_audited": False}, stream, indent=2)
            stream.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "physical_gate_counts": report["physical_gate_counts"],
                      "physical_zero_gate_passed": report["physical_zero_gate_passed"]}))


if __name__ == "__main__":
    main()
