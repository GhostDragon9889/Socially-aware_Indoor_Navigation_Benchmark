"""Expose reductions of saved full-clip validation scalars without new inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.result.is_symlink() or not args.result.is_file() or args.result.stat().st_mode & 0o222:
        parser.error("An actual sealed regular result is required")
    if args.output.exists():
        parser.error("Preserve existing outputs; choose a fresh output")
    content = args.result.read_bytes()
    result = json.loads(content)
    if result["status"] != "passed" or result["preflight"]["split"] != "validation":
        parser.error("Completed validation result required")
    rows = result["sequences"]
    denominator = result["aggregate"]["complete_success_denominator"]
    if len(rows) != denominator or not rows or any(
            not row["complete_clip"] or not row["eligible"]
            or not row["in_full_sequence_success_denominator"] for row in rows):
        raise ValueError("All eligible full clips must remain in the denominator")
    scalar_fields = ("mean_body_position_error_m", "mean_body_rotation_error_rad",
                     "maximum_step_mean_body_position_error_m")
    for row in rows:
        if row["observed_metric_steps"] != row["expected_control_steps"] or row["observed_metric_steps"] <= 0:
            raise ValueError("Full positive metric-frame count required")
        for key in scalar_fields:
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Finite nonnegative saved error required")
    frames = sum(row["observed_metric_steps"] for row in rows)
    primary = result["aggregate"]
    for source_key, primary_key in (
            ("mean_body_position_error_m", "frame_weighted_mean_body_position_error_m"),
            ("mean_body_rotation_error_rad", "frame_weighted_mean_body_rotation_error_rad")):
        reconstructed = sum(row[source_key] * row["observed_metric_steps"] for row in rows) / frames
        if not math.isclose(reconstructed, primary[primary_key], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Sequence scalar weighting disagrees with actual primary")
    failures = [row for row in rows if row["tracking_failed"]]
    report = {
        "schema": "tracking.saved_validation_statistic_scopes.v1", "status": "completed",
        "source_result": {"path": str(args.result.resolve()), "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()},
        "primary_unchanged": primary,
        "secondary_from_saved_sequence_scalars": {
            "clip_equal_mean_position_error_m": fmean(row[scalar_fields[0]] for row in rows),
            "clip_equal_mean_rotation_error_rad": fmean(row[scalar_fields[1]] for row in rows),
            "maximum_over_clips_step_mean_position_error_m": max(row[scalar_fields[2]] for row in rows),
            "maximum_over_clips_clip_mean_position_error_m": max(row[scalar_fields[0]] for row in rows),
            "individual_body_frame_global_max_m": None,
            "individual_body_frame_global_max_reason": "not derivable from saved sequence scalars; no capture reread",
        },
        "scope": {
            "population": "all completed eligible full validation clips, including every failed clip",
            "clips": len(rows), "metric_frames": frames,
            "body_count": 24, "position": "same-frame mean world body-origin Euclidean error",
            "rotation": "same-frame mean24 world quaternion SO3 geodesic error",
            "primary_weighting": "metric-frame weighted; every body has equal weight within frame",
            "secondary_clip_weighting": "each clip has equal weight regardless of duration",
            "maximum_step_mean_reduction": "max over all clips and metric frames of per-frame mean24 error",
            "maximum_clip_mean_reduction": "max over clips of time-mean per-frame mean24 error",
            "reset_phase_and_input_protocol": result["protocol"],
        },
        "failed_clip_scalars": [{"motion_id": row["motion_id"],
                                 "maximum_step_mean_body_position_error_m": row[scalar_fields[2]],
                                 "mean_body_position_error_m": row[scalar_fields[0]]} for row in failures],
        "author_maximum_reduction_and_population_known": False,
        "author_body_set_rotation_formula_and_weighting_known": False,
        "paper_metric_equivalence_claimed": False,
        "metric_capture_recomputed": False, "model_loaded": False,
        "SDK_started": False, "sealed34_outputs_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "clips": len(rows), "metric_frames": frames,
                      **report["secondary_from_saved_sequence_scalars"]}, allow_nan=False))


if __name__ == "__main__":
    main()
