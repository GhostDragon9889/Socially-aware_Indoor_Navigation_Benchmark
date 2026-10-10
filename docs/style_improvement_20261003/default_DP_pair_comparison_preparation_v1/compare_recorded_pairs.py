"""Compare already closed defaultDP fixture captures without filtering or new metric mathematics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREP = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1"
REGISTRATION_SHA = "467d26061206eb4a6927010d8b782bcd08c9e941ffb644c87e4006e13ee57494"
HELPER_SHA = "30202b34a13b5756ffa5c9b41110103a913c3883ed1d88d8389546322a8fc05a"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular recorded artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    if record(PREP / "registration.json")["sha256"] != REGISTRATION_SHA or record(PREP / "physical_evidence.py")["sha256"] != HELPER_SHA:
        raise ValueError("Exact registered physical source bridge required")
    spec = importlib.util.spec_from_file_location("recorded_DP_pair_frozen_source_bridge", PREP / "physical_evidence.py")
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    plan = evidence.registration()
    binding = evidence.source_binding(plan)
    return evidence, plan, binding


def current_audit(evidence, plan, phase, path):
    case = plan["cases"][phase]
    evidence.require(str(Path(path).absolute()) == case["audit"], "Exact already registered capture audit path required")
    document = read(path)
    admission = evidence.admit_capture(plan, phase)
    evidence.require(document["schema"] == "default_newDP.independent_physical_CPU_audit.v1" and document["status"] == "passed"
                     and document["phase"] == phase and document["actual_capture_independently_audited"] is True
                     and document["actual_source_admission"] == admission,
                     "Complete independent audit/source admission required; physical failures are retained")
    owner = read(case["CPU_receipt"])
    evidence.require(owner["schema"] == "default_newDP.actual_CPU_audit_launch.v1" and owner["phase"] == phase
                     and owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
                     and owner["child_exit_observed"] is True and owner["driver_completed"] is True
                     and owner["command"] == case["CPU_command"] and owner["audit_artifact"] == record(path)
                     and owner["owner"] == record(PREP / "launch_registered.py")
                     and owner["binding_before"] == owner["binding_after"] == evidence.source_binding(plan),
                     "Actual closed CPU producer/report required")
    evidence.require(not Path(path).stat().st_mode & 0o222, "Actual audit must be sealed readonly")
    return document, read(admission["result"]["path"])


def compare_arrays(left, right):
    import numpy as np
    left_record, right_record = record(left), record(right)
    rows = {}
    with np.load(left, allow_pickle=False) as first, np.load(right, allow_pickle=False) as second:
        left_keys, right_keys = set(first.files), set(second.files)
        for key in sorted(left_keys | right_keys):
            row = {"left_present": key in left_keys, "right_present": key in right_keys}
            values = []
            for side, loaded, keys in (("left", first, left_keys), ("right", second, right_keys)):
                if key in keys:
                    value = loaded[key]
                    if value.dtype.kind == "O":
                        raise ValueError("Object dtype is not a recorded numeric/string array")
                    data = value.tobytes(order="C")
                    row[side] = {"dtype": value.dtype.str, "shape": list(value.shape), "array_bytes": len(data),
                                 "C_order_bytes_sha256": hashlib.sha256(data).hexdigest()}
                    values.append(value)
            if len(values) == 2:
                row.update(dtype_equal=values[0].dtype == values[1].dtype,
                           shape_equal=values[0].shape == values[1].shape,
                           C_order_bytes_equal=row["left"]["array_bytes"] == row["right"]["array_bytes"] and
                                               row["left"]["C_order_bytes_sha256"] == row["right"]["C_order_bytes_sha256"],
                           values_exact_equal=bool(np.array_equal(values[0], values[1])))
            rows[key] = row
    if record(left) != left_record or record(right) != right_record:
        raise ValueError("Recorded NPZ changed during read")
    return {"left_NPZ": left_record, "right_NPZ": right_record, "all_array_keys_equal": left_keys == right_keys,
            "all_unpacked_arrays_dtype_shape_bytes_exact": left_keys == right_keys and all(
                row["dtype_equal"] and row["shape_equal"] and row["C_order_bytes_equal"] for row in rows.values()),
            "arrays": rows, "rows_filtered_or_resampled": False}


def progress(audit, result):
    return {"raw_audit_physical_counts": audit["physical_gate_counts"], "complete_raw_counts": audit["counts"],
            "original_static_peer_channels": audit["original_obstacle_peer_channels"],
            "runtime_progress": {key: result[key] for key in ("actual_control_steps", "fall_resets", "time_limit_resets",
                "route_vertices_reached", "diffusion_plans", "global_path_failures", "successful_start_recoveries", "planning_clock")},
            "all_failed_contact_idle_reset_samples_retained": True}


def compare(old_audit_path, new_audit_path=None):
    evidence, plan, binding = sources()
    old, old_result = current_audit(evidence, plan, "fixture_old", old_audit_path)
    historical_row = plan["inputs"]["historical_fixture_raw_audit"]
    evidence.require(record(historical_row["path"]) == historical_row, "Selected historical frozen raw report changed")
    historical = read(historical_row["path"])
    evidence.require(historical["status"] == "passed" and historical["counts"]["controls"] == 6000
                     and historical["counts"]["actors"] == 4, "Exact selected historical full fixture required")
    old_path = Path(plan["cases"]["fixture_old"]["output"])
    physical = historical["artifacts"]["physical_states"]
    evidence.require(record(physical["path"]) == physical, "Selected historical physical capture changed")
    # Exact selected reference NPZ is published here, not guessed from a nearby run name.
    comparison_row = read(HERE / "source_binding.json")["historical_selected_style_comparison"]
    evidence.require(record(comparison_row["path"]) == comparison_row, "Frozen selected comparison changed")
    saved_plans = read(comparison_row["path"])["stagnation"]["saved_plans"]
    evidence.require(Path(saved_plans["path"]).parent == Path(historical["input"])
                     and record(saved_plans["path"]) == saved_plans, "Selected historical reference plans identity differs")
    report = {"schema": "default_DP_pair.actual_recorded_comparison.v1", "status": "passed", "execution_pass_is_physical_quality_pass": False,
              "OLD_current185": progress(old, old_result), "OLD_audit": record(old_audit_path),
              "OLD_current185_vs_historical_selected": {"historical_raw_audit": historical_row,
                  "historical_comparison": comparison_row, "physical_states": compare_arrays(old_path / "physical_states.npz", physical["path"]),
                  "reference_plans": compare_arrays(old_path / "reference_plans.npz", saved_plans["path"]),
                  "scope": "Exact equality is an observation of these retained captures; it is not a future bitwise physics/RNG guarantee."},
              "current185_OLD_NEW": None, "NEW_capture_read": False,
              "source_before": binding, "reader": record(__file__), "CPU_only": True, "GPU_used": False,
              "SDK_started": False, "models_loaded": False, "style_scores_used": False, "default_promotion": False,
              "sealed34_outputs_read": False}
    if new_audit_path is not None:
        new, new_result = current_audit(evidence, plan, "fixture_new", new_audit_path)
        new_path = Path(plan["cases"]["fixture_new"]["output"])
        report["current185_OLD_NEW"] = {"NEW_audit": record(new_audit_path), "NEW_current185": progress(new, new_result),
                                         "same_current185_source_and_control_parameters": True,
                                         "only_DP_checkpoint_and_output_identity_changed": plan["strict_current185_pair_recipe_delta"],
                                         "arrays": {name: compare_arrays(old_path / name, new_path / name)
                                                    for name in ("physical_states.npz", "reference_plans.npz", "ground_support.npz")},
                                         "NEW_full_physical_zero": new["physical_zero_gate_passed"],
                                         "OLD_failures_did_not_exclude_NEW": True}
        report["NEW_capture_read"] = True
    report["source_after"] = evidence.source_binding(plan)
    evidence.require(report["source_after"] == binding, "Source changed during recorded comparison")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-audit", type=Path)
    parser.add_argument("--new-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        _, _, binding = sources()
        print(json.dumps({"status": "prepared_source_only", "binding": binding, "future_effect": False,
                          "capture_arrays_read": False, "GPU_used": False, "SDK_started": False}))
        return 0
    if args.old_audit is None or args.output is None or args.output.exists():
        raise ValueError("Closed OLD audit and exclusive output are required")
    report = compare(args.old_audit, args.new_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "NEW_capture_read": report["NEW_capture_read"],
                      "historical_physical_arrays_exact": report["OLD_current185_vs_historical_selected"]["physical_states"]["all_unpacked_arrays_dtype_shape_bytes_exact"],
                      "historical_reference_arrays_exact": report["OLD_current185_vs_historical_selected"]["reference_plans"]["all_unpacked_arrays_dtype_shape_bytes_exact"],
                      "output": record(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
