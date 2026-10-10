"""Thin state1950 fixture 90D/45D I/O; original metric function objects only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG_SHA256 = "fab0ae231f8d37ad877bf4811d2a18708886601a61352a87b5a0e2173b8c3159"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/style_improvement_20261003/state_only_matched_q_physical_preparation_v1"))
import audit_fixture as admission


def catalog():
    admission.require(admission.record(HERE / "commands.json")["sha256"] == CATALOG_SHA256,
                      "Exact fixed original metric parameters required")
    value = admission.read(HERE / "commands.json")
    admission.require(value["schema"] == "state_only_matched_q.prepared_fixture_original_90D45D_IO.v1"
                      and value["status"] == "prepared_source_only", "Fixed metric I/O catalog required")
    for row in value["sources"].values():
        admission.require(admission.record(row["path"]) == row, "Original source/reference identity changed")
    return value


def actual_source(value, args):
    registered = admission.plan()
    source = admission.admit_capture(registered, args.owner, args.owner_sha256)
    path = Path(value["raw_report"])
    admission.require(path.is_file() and not path.stat().st_mode & 0o222, "Raw report must be complete and sealed")
    raw = admission.read(path)
    admission.require(raw["schema"] == "state_only_matched_q.actual_fixture_raw_audit.v1"
                      and raw["status"] == "passed" and raw["actual_capture_independently_audited"] is True
                      and raw["actual_source_admission"] == source
                      and raw["original_math"] == registered["fixture_raw_math"]
                      and raw["only_verify_sources_hook_replaced"] is True
                      and raw["counts"]["controls"] == 6000 and raw["counts"]["actors"] == 4
                      and raw["counts"]["duration_s"] == 200.
                      and all(row["passed"] is True for row in raw["checks"]), "Complete original raw audit/source required")
    owner_path = Path(value["future_raw_CPU_owner"])
    admission.require(admission.record(owner_path)["sha256"] == args.raw_owner_sha256
                      and not owner_path.stat().st_mode & 0o222, "Actual raw CPU owner SHA/readonly required")
    owner = admission.read(owner_path)
    command = value["raw_CPU_command_prefix"] + [args.owner_sha256] + value["raw_CPU_command_suffix"]
    admission.require(owner["schema"] == "state_only_matched_q.actual_fixture_raw_CPU_owner.v1"
                      and owner["status"] == "closed" and type(owner["actual_exit_code"]) is int
                      and owner["actual_exit_code"] == 0 and owner["child_exit_observed"] is True
                      and isinstance(owner["actual_root_completion_chunk"], str) and owner["actual_root_completion_chunk"]
                      and owner["command"] == command
                      and owner["raw_report"] == admission.record(path), "Real raw CPU closure/command/report differs")
    return {"actual_capture_source_admission": source, "raw_report": admission.record(path),
            "actual_raw_CPU_owner": admission.record(owner_path),
            "raw_physical_zero_gate": raw["physical_zero_gate_passed"],
            "raw_physical_zero_used_as_style_filter": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dimension", type=int, choices=(90, 45))
    parser.add_argument("--owner", type=Path)
    parser.add_argument("--owner-sha256")
    parser.add_argument("--raw-owner-sha256")
    args = parser.parse_args()
    admission.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Metric I/O requires hidden CUDA")
    value = catalog()
    if args.prepare_only:
        admission.require(not any((args.dimension, args.owner, args.owner_sha256, args.raw_owner_sha256)),
                          "Preparation cannot read actual captures")
        print(json.dumps({"status": "source_only_original_metric_identity_passed", "actual_capture_read": False,
                          "NN_constructed": False, "actual_metrics_computed": False}))
        return 0
    admission.require(args.dimension is not None and args.owner is not None and args.owner_sha256 is not None
                      and args.raw_owner_sha256 is not None, "Actual owner/CPU closure and dimension required")
    row = value["metrics"][str(args.dimension)]
    output = Path(row["output"])
    admission.require(not output.exists() and not output.is_symlink(), "New registered metric output required")
    source = actual_source(value, args)  # Before opening any capture arrays.
    if args.dimension == 90:
        tool = admission.load("state1950_original_style90_entry", value["sources"]["style90_entry"]["path"])
        tool.main(row["metric_argv"])
    else:
        original_path = Path(value["sources"]["original_pose45_runner"]["path"])
        # Original runner imports its original stdlib helper at module load.
        # The numerical pose45 function object/globals are never recompiled.
        sys.path.insert(0, str(original_path.parent))
        original = admission.load("state1950_original_pose45_runner", original_path)
        original.pose45(row, value)
    import torch
    admission.require(not torch.cuda.is_initialized() and not any(name.split(".")[0] in
                      {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
                      "CPU metric loaded SDK or initialized CUDA")
    report = admission.read(output / "evaluation.json")
    admission.require(report["status"] == "passed", "Original metric execution did not complete")
    evidence = {"schema": "state_only_matched_q.actual_fixture_metric_source_admission.v1", "status": "passed",
                "dimension": args.dimension, "original_function_objects_not_recompiled": True,
                "metric_report": admission.record(output / "evaluation.json"), "metric_math": value["sources"],
                "runner": admission.record(__file__), "catalog": admission.record(HERE / "commands.json"),
                "metric_argv": row["metric_argv"], **source,
                "all_failed_idle_and_fall_intervals_retained": True, "basis_refit_performed": False,
                "metric_pass_is_quality_pass": False, "MMD_launch_filter": False,
                "complete_runtime_import_ledger": False, "default_promotion": False}
    with (output / "actual_source_admission.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed_original_metric_execution", "dimension": args.dimension,
                      "physical_zero_gate_as_published": source["raw_physical_zero_gate"],
                      "physical_zero_used_as_metric_filter": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
