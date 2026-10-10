"""Summarize completed frozen metrics; independently pair actual capture bytes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LABELS = ("backward_penalty1000", "mixed2000", "state_action2000", "protected2000")
INITIAL = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
           "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history")
CLOCK = ("reference_frame", "reference_time_s", "metric_frame", "native_reset_frame", "episode_id",
         "trajectory_xy", "waypoint_query_phase_frame", "waypoint_held_at_clip_end")
DECOMPOSITION = ("root_xy_drift_m", "root_yaw_abs_wrapped_error_rad",
                 "frame_mean_rootcenter_heading_body_position_error_m",
                 "frame_mean_heading_body_orientation_error_rad",
                 "frame_mean_parent_relative_joint_orientation_error_rad", "actual_planar_speed_m_s")


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def exact(left, right):
    return {"shape_equal": left.shape == right.shape, "dtype_equal": left.dtype == right.dtype,
            "C_order_bytes_equal": left.tobytes(order="C") == right.tobytes(order="C")}


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only capture reader required")
    comparison = json.loads((HERE / "comparison.json").read_text())
    runs = {}
    for name in LABELS:
        source = comparison["runs"][name]["result"]
        if record(source["path"]) != source:
            raise ValueError("Actual complete result changed:" + name)
        value = json.loads(Path(source["path"]).read_text())
        if (value["status"] != "passed" or len(value["sequences"]) != 22 or
                value["aggregate"]["observed_metric_steps"] != 11982):
            raise ValueError("All22/11982 completed physical records required")
        runs[name] = value
    new = runs["protected2000"]
    pairings = {}
    for name in LABELS[:-1]:
        previous = {row["motion_id"]: row for row in runs[name]["sequences"]}
        rows = []
        for row in new["sequences"]:
            old = previous[row["motion_id"]]
            if old["source_path"] != row["source_path"]:
                raise ValueError("Reference source identity differs")
            with np.load(old["capture"], allow_pickle=False) as left, np.load(row["capture"], allow_pickle=False) as right:
                reference = sorted({key for key in left.files if key.startswith("reference_")} | set(CLOCK))
                full = {key: exact(left[key], right[key]) for key in reference}
                initial = {key: exact(left[key][0], right[key][0]) for key in INITIAL}
                if not all(all(check.values()) for check in [*full.values(), *initial.values()]):
                    raise ValueError(f"Actual initial/reference dtype/bytes differ:{name}/{row['motion_id']}")
            rows.append({"motion_id": row["motion_id"], "source_path": row["source_path"],
                         "full_reference_and_clock": full, "initial_physical_and_AMP_history": initial})
        pairings[name] = rows
    tables = {}
    for name in LABELS:
        value = comparison["runs"][name]
        tables[name] = {"aggregate": value["aggregate"], "inclusive90D_MMD2": value["inclusive_mmd2"],
                        "speed_Wasserstein_mps": value["speed_wasserstein_m_s"],
                        "decomposition_means": {key: value["same_frame_decomposition"][key]["mean"] for key in DECOMPOSITION},
                        "speed_bins": value["speed_bins"], "capture_validation": value["capture_validation"],
                        "checkpoint": value["checkpoint"], "result": value["result"],
                        "decomposition_source": value["decomposition"]}
    success = {name: {row["motion_id"] for row in run["sequences"] if row["tracking_success"]}
               for name, run in runs.items()}
    changes = {}
    for name in LABELS[:-1]:
        changes[name] = {"gained_success_motion_ids": sorted(success["protected2000"] - success[name]),
                         "lost_success_motion_ids": sorted(success[name] - success["protected2000"])}
    failed = [{key: row[key] for key in ("motion_id", "source_path", "observed_metric_steps",
              "first_tracking_failure_capture_frame", "mean_body_position_error_m",
              "maximum_step_mean_body_position_error_m", "mean_body_rotation_error_rad", "sequence_had_fall")}
              for row in new["sequences"] if not row["tracking_success"]]
    # Secondary summaries do not replace the all22 metrics or remove failed frames.
    secondary = {}
    for comparator in LABELS[:-1]:
        common = success[comparator] & success["protected2000"]
        secondary[comparator] = {}
        for name in (comparator, "protected2000"):
            rows = [row for row in runs[name]["sequences"] if row["motion_id"] in common]
            frames = sum(row["observed_metric_steps"] for row in rows)
            secondary[comparator][name] = {"common_success_motion_ids": sorted(common),
                "clips": len(rows), "metric_frames": frames,
                "frame_weighted_position_m": sum(row["mean_body_position_error_m"] * row["observed_metric_steps"] for row in rows) / frames,
                "frame_weighted_rotation_rad": sum(row["mean_body_rotation_error_rad"] * row["observed_metric_steps"] for row in rows) / frames,
                "per_clip_mean_position_m": sum(row["mean_body_position_error_m"] for row in rows) / len(rows),
                "per_clip_mean_rotation_rad": sum(row["mean_body_rotation_error_rad"] for row in rows) / len(rows)}
    report = {"schema": "protected_terrain.actual_full22_tracking_summary.v1", "status": "passed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
        "metric_evaluator_or_registration_modified": False, "primary_failed_inclusive_sequences": 22,
        "primary_metric_frames": 11982, "all_failed_sequences_and_frames_retained": True,
        "summarizer": record(__file__), "frozen_score_output": record(HERE / "evaluation.json"),
        "frozen_comparison_output": record(HERE / "comparison.json"), "tables": tables,
        "actual_dtype_and_C_order_bytes_pairings_vs_three_comparators": pairings,
        "success_changes": changes, "failed_sequences": failed,
        "common_success_secondary_only_not_primary": secondary,
        "historical13_dicts_unchanged": {"evaluation": comparison["prior13_evaluation_dicts_exact"],
                                         "comparison": comparison["prior13_summary_dicts_exact"]},
        "source_admission": comparison["append_provenance"]["actual_evaluation_source_admission"],
        "interpretation": ["90D style aggregate improves against old state-action2000, while all22 tracking worsens; no universal quality improvement.",
                           "Root XY/yaw errors increase versus old state-action2000; heading-local positions and parent-relative joint rotations are close. This is descriptive decomposition, not a proof of cause.",
                           "This single-seed frozen engineering experiment does not establish author parameter/data equivalence, statistical significance, or navigation deployment safety.",
                           "Native external SDK bytes and unobserved import/finalization events are outside the931 owned-source ledger; no evidence is fabricated.",
                           "90D MMD is the mean of supported fixed speed-bin Gaussian MMD2, sigma1 with velocity scale0.1s;2..3m/s remains unsupported.45D is not evaluated on this tracking run."]}
    write_new(HERE / "actual_summary.json", report)
    print(json.dumps({"status": "passed", "paired_actual_clips": sum(len(rows) for rows in pairings.values()),
                      "all_primary_clips": 22, "all_primary_frames": 11982, "failed_clips_retained": len(failed)}))


if __name__ == "__main__":
    main()
