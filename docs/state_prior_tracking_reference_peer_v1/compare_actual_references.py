"""Admit actual new full22 capture, then pair only reference inputs/reset."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVALUATION = ROOT / "docs/style_improvement_20261003/state_prior_evaluation_registration_v1"
FREEZE_SHA = "0042d4d1a2905541194f5718dc67dc9a12587e0989dc1c1932de015c55976669"
NEW_RESULT = ROOT / "output/diffusion_amp/state_prior_tracking2000_v1/result.json"
PREVIOUS_COMPARISON = ROOT / "docs/style_improvement_20261002/terrain_protected_style_v1/full22_step2000_comparison_v1/comparison.json"
PREVIOUS_COMPARISON_SHA = "5ce74c22d7381d36b91a4c976828984caa642e71d08bd64966489bed2d21b392"
COMPARATORS = ("backward_penalty1000", "state_action2000", "protected2000")
CLOCK = ("reference_frame", "reference_time_s", "metric_frame", "native_reset_frame", "episode_id",
         "trajectory_xy", "waypoint_query_phase_frame", "waypoint_held_at_clip_end")
INITIAL = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
           "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_new(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pair(left, right):
    left_bytes, right_bytes = left.tobytes(order="C"), right.tobytes(order="C")
    return {"shape": list(left.shape), "right_shape": list(right.shape),
            "dtype": left.dtype.str, "right_dtype": right.dtype.str,
            "shape_equal": left.shape == right.shape,
            "dtype_equal": left.dtype == right.dtype and left.dtype.str == right.dtype.str,
            "C_order_bytes_equal": left_bytes == right_bytes,
            "left_C_order_sha256": hashlib.sha256(left_bytes).hexdigest(),
            "right_C_order_sha256": hashlib.sha256(right_bytes).hexdigest()}


def check_pair(left, right, label):
    check = pair(left, right)
    require(all(check[name] for name in ("shape_equal", "dtype_equal", "C_order_bytes_equal")),
            "Actual reference input differs: " + label)
    return check


def complete_run(document):
    require(document["status"] == "passed" and document.get("error") is None,
            "Actual completed tracking result required")
    rows = document["sequences"]
    require(len(rows) == len({row["motion_id"] for row in rows}) == 22,
            "Full unique22 tracking clips required")
    require(document["aggregate"]["observed_metric_steps"] == 11982 and
            document["aggregate"]["complete_success_denominator"] == 22 and
            sum(row["observed_metric_steps"] for row in rows) == 11982,
            "All22/11982 failed-inclusive controls required")
    require(all(row["complete_clip"] is True and row["eligible"] is True and
                row["in_full_sequence_success_denominator"] is True and
                row["app_or_simulation_stopped"] is False and
                row["observed_metric_steps"] == row["expected_control_steps"] for row in rows),
            "A partial tracking clip cannot be paired")


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only reference review required")
    freeze_path = EVALUATION / "source_freeze_v1.json"
    require(record(freeze_path)["sha256"] == FREEZE_SHA, "Frozen evaluation guard changed")
    freeze = json.loads(freeze_path.read_text())
    for row in freeze["source_files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen evaluation dependency differs")
    require(record(freeze["archive"]["path"]) == freeze["archive"], "Evaluation source archive differs")
    registration = json.loads((EVALUATION / "registration.json").read_text())
    new_record = record(NEW_RESULT)
    new = json.loads(NEW_RESULT.read_text())
    spec = importlib.util.spec_from_file_location("actual_full22_state_prior_source_bridge", EVALUATION / "evaluation_evidence.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    # This is the actual terminal/full-payload/new934/registered argument and
    # executed owned-import admission. Reference archive processing starts only
    # after this call succeeds.
    source = bridge.admit_actual_evaluation(new, "tracking",
            registration["future_checkpoint"]["terminal_admission_path"],
            ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1/state_prior_tracking2000_v1_GPU_imports.json")
    ledger_bytes = source.pop("ledger_bytes")
    event_bytes = source.pop("event_prefix_bytes")
    write_new(HERE / "actual_source_admission.json", source)
    with (HERE / "GPU_import_ledger_as_observed.json").open("xb") as stream:
        stream.write(ledger_bytes)
    with (HERE / "GPU_import_events_as_observed.jsonl").open("xb") as stream:
        stream.write(event_bytes)
    complete_run(new)
    require(record(PREVIOUS_COMPARISON)["sha256"] == PREVIOUS_COMPARISON_SHA,
            "Previously sealed comparator records changed")
    previous = json.loads(PREVIOUS_COMPARISON.read_text())
    require(previous["runs"]["backward_penalty1000"]["checkpoint"]["sha256"].startswith("62c4edc195d5"),
            "Selected62c4 baseline required")
    old_runs, run_records = {}, {"state_prior2000": new_record}
    new_ids = [row["motion_id"] for row in new["sequences"]]
    for name in COMPARATORS:
        expected = previous["runs"][name]["result"]
        require(record(expected["path"]) == expected, "Actual old result differs: " + name)
        document = json.loads(Path(expected["path"]).read_text())
        complete_run(document)
        require([row["motion_id"] for row in document["sequences"]] == new_ids,
                "Exact full22 ordered motion IDs differ: " + name)
        old_runs[name] = {row["motion_id"]: row for row in document["sequences"]}
        run_records[name] = expected

    import numpy as np
    rows = {name: [] for name in COMPARATORS}
    captures = {"state_prior2000": [], **{name: [] for name in COMPARATORS}}
    negatives = []
    reference_count = reset_count = 0
    for new_row in new["sequences"]:
        new_capture = record(new_row["capture"])
        require(new_capture["sha256"] == new_row["capture_sha256"], "New actual capture checksum differs")
        captures["state_prior2000"].append(new_capture)
        with np.load(new_row["capture"], allow_pickle=False) as right:
            require(right["reference_frame"].shape == (new_row["observed_metric_steps"] + 1,),
                    "Actual reference clock needs reset plus every observed metric frame")
            for name in COMPARATORS:
                old_row = old_runs[name][new_row["motion_id"]]
                require(old_row["source_path"] == new_row["source_path"] and
                        old_row["tracking_window"] == new_row["tracking_window"] and
                        old_row["initial_translation_xyz_m"] == new_row["initial_translation_xyz_m"],
                        "Actual source/window/translation differs: " + name)
                old_capture = record(old_row["capture"])
                require(old_capture["sha256"] == old_row["capture_sha256"], "Old actual capture checksum differs")
                captures[name].append(old_capture)
                with np.load(old_row["capture"], allow_pickle=False) as left:
                    left_reference = {key for key in left.files if key.startswith("reference_")}
                    right_reference = {key for key in right.files if key.startswith("reference_")}
                    require(left_reference == right_reference, "Reference field schema differs")
                    keys = sorted(left_reference | set(CLOCK))
                    full = {key: check_pair(left[key], right[key], name + "/" + str(new_row["motion_id"]) + "/" + key)
                            for key in keys}
                    # These reset records are diagnostic initial conditions, not
                    # a requirement that subsequent policy-generated outputs match.
                    reset = {key: pair(left[key][0], right[key][0]) for key in INITIAL}
                    reference_count += len(full)
                    reset_count += len(reset)
                rows[name].append({"motion_id": new_row["motion_id"], "source_path": new_row["source_path"],
                                   "tracking_window": new_row["tracking_window"],
                                   "full_reference_and_clock": full, "initial_reset_diagnostic": reset})
            if not negatives:
                sample = right["reference_rigid_body_pos"]
                changed = sample.copy()
                changed.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
                for name, modified in (("one_C_byte_changed", changed), ("dtype_changed", sample.astype(np.float64)),
                                       ("shape_changed", sample[:-1])):
                    try:
                        check_pair(sample, modified, name)
                    except ValueError:
                        negatives.append({"case": name, "rejected": True, "derived_from_actual_motion_id": new_row["motion_id"],
                                          "field": "reference_rigid_body_pos", "check": pair(sample, modified)})
                    else:
                        raise ValueError("Reference mismatch negative unexpectedly accepted")
    require(record(NEW_RESULT) == new_record, "New result changed during CPU review")
    for name, records in captures.items():
        require(all(record(row["path"]) == row for row in records), "Capture changed during CPU review: " + name)
    heavy = sorted(name for name in sys.modules if name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "warp", "pxr"})
    require(not heavy, "CPU reference review imported Torch or SDK")
    reset_equal = all(all(check[k] for k in ("shape_equal", "dtype_equal", "C_order_bytes_equal"))
                      for grouped in rows.values() for item in grouped for check in item["initial_reset_diagnostic"].values())
    proof = {"schema": "public_state_prior.actual_full22_reference_peer_review.v1", "status": "passed",
             "checked_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False,
             "SDK_started": False, "sealed34_outputs_read": False, "summarizer": record(__file__),
             "actual_source_admission": record(HERE / "actual_source_admission.json"),
             "evaluation_source_freeze": record(freeze_path), "previous_comparator_source": record(PREVIOUS_COMPARISON),
             "run_results": run_records, "actual_capture_records": captures, "ordered_motion_ids": new_ids,
             "paired_actual_clips": 66, "failed_inclusive_new_clips": 22, "new_metric_frames": 11982,
             "actual_new_tracking_aggregate": new["aggregate"],
             "required_reference_and_clock_array_comparisons": reference_count,
             "diagnostic_initial_reset_array_comparisons": reset_count,
             "all_initial_reset_diagnostics_equal": reset_equal,
             "total_actual_array_pairs": reference_count + reset_count,
             "all_required_reference_shape_dtype_C_bytes_exact": True,
             "generated_post_reset_array_equality_required": False,
             "generated_performance_arrays_compared": False,
             "actual_dtype_and_C_bytes_pairings": rows, "negative_cases": negatives,
             "heavy_modules_imported": heavy,
             "scope": "Actual source/terminal/checkpoint-SHA/import-ledger admission first; actual full22 references against selected62c4, old matched state_action2000 and terrain_protected2000. Ground-truth body/root/DOF pose, quaternion/velocity and all reference clock/phase/commands compare in original dtype, shape and C-order bytes. Initial reset is a separate diagnostic; generated trajectories/actions/errors/falls after reset are never required to match. This does not establish matched training budget, style improvement or author reproduction."}
    write_new(HERE / "actual_reference_pairing.json", proof)
    print(json.dumps({key: proof[key] for key in ("status", "paired_actual_clips", "required_reference_and_clock_array_comparisons",
          "diagnostic_initial_reset_array_comparisons", "all_initial_reset_diagnostics_equal", "total_actual_array_pairs",
          "all_required_reference_shape_dtype_C_bytes_exact", "heavy_modules_imported")}, indent=2))


if __name__ == "__main__":
    main()
