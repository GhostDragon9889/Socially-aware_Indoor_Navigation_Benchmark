"""Independent NumPy SO(3)/paired-frame review of actual full22 captures.

No Torch/SciPy/SDK imports or metric changes. Matrix trace/skew atan2 verifies
the recorded quaternion-dot primary errors and frozen local decomposition.
All11982 metric frames, including failed clips, remain in every aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BODY = ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
        "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
        "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    require(q.shape[-1] == 4 and np.isfinite(q).all(), "Finite XYZW quaternions required")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.allclose(norm, 1, atol=1e-3, rtol=0), "Unit XYZW rotations required")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                    2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                    2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1] + (3, 3))


def geodesic(actual, reference):
    relative = np.swapaxes(reference, -1, -2) @ actual
    skew = np.stack((relative[..., 2, 1]-relative[..., 1, 2],
                     relative[..., 0, 2]-relative[..., 2, 0],
                     relative[..., 1, 0]-relative[..., 0, 1]), -1) / 2
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.arctan2(np.linalg.norm(skew, axis=-1), cosine)


def heading(rot):
    yaw = np.arctan2(rot[..., 1, 0], rot[..., 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    zero, one = np.zeros_like(c), np.ones_like(c)
    h = np.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(yaw.shape+(3, 3))
    return yaw, h


def diagnostics(capture, parents):
    p, r = [capture[k][:, 0].astype(np.float64) for k in ("rigid_body_pos", "reference_rigid_body_pos")]
    ar, rr = [rotation(capture[k][:, 0]) for k in ("rigid_body_rot", "reference_rigid_body_rot")]
    ay, ah = heading(ar[:, 0])
    ry, rh = heading(rr[:, 0])
    ac, rc = p-p[:, :1], r-r[:, :1]
    alocal = np.einsum("tji,tbj->tbi", ah, ac)
    rlocal = np.einsum("tji,tbj->tbi", rh, rc)
    world_pos = np.linalg.norm(p-r, axis=-1)
    world_rot = geodesic(ar, rr)
    heading_rot = geodesic(np.swapaxes(ah, -1, -2)[:, None] @ ar,
                          np.swapaxes(rh, -1, -2)[:, None] @ rr)
    root_rot = geodesic(np.swapaxes(ar[:, :1], -1, -2) @ ar,
                       np.swapaxes(rr[:, :1], -1, -2) @ rr)
    joint_rot = geodesic(np.swapaxes(ar[:, parents], -1, -2) @ ar,
                        np.swapaxes(rr[:, parents], -1, -2) @ rr)
    return {"world_body_position_error_m": world_pos,
            "world_body_rotation_error_rad": world_rot,
            "root_xy_drift_m": np.linalg.norm(p[:, 0, :2]-r[:, 0, :2], axis=-1),
            "root_yaw_abs_wrapped_error_rad": np.abs(np.arctan2(np.sin(ay-ry), np.cos(ay-ry))),
            "rootcenter_heading_body_position_error_m": np.linalg.norm(alocal-rlocal, axis=-1),
            "heading_body_orientation_error_rad": heading_rot,
            "full_root_relative_body_orientation_error_rad": root_rot,
            "parent_relative_joint_orientation_error_rad": joint_rot}


def stats(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"count": len(a), "mean": float(a.mean()), "rms": float(np.sqrt(np.mean(a*a))),
            "median": float(np.median(a)), "p90": float(np.quantile(a, .9)), "max": float(a.max())}


def parent_indices(path):
    found = {}

    def visit(body, parent=None):
        name = body.attrib["name"]
        require(name not in found, "Duplicate rig body")
        found[name] = parent
        for child in body.findall("body"):
            visit(child, name)
    for body in ET.parse(path).getroot().find("worldbody").findall("body"):
        visit(body)
    require(set(found) == set(BODY), "Current canonical24 parent tree required")
    return [BODY.index(found[name]) if found[name] else 0 for name in BODY]


def analytic_check(parents):
    q = np.zeros((2, 1, 24, 4))
    q[..., 3] = 1
    p = np.arange(2*24*3, dtype=np.float64).reshape(2, 1, 24, 3) / 20
    angle = .8
    aq = q.copy()
    aq[..., 2], aq[..., 3] = np.sin(angle/2), np.cos(angle/2)
    h = rotation(aq[:, 0, 0])
    ap = np.einsum("tij,tbj->tbi", h, p[:, 0])[:, None] + np.array([4, -3, 2])
    diag = diagnostics({"rigid_body_pos": ap, "reference_rigid_body_pos": p,
                        "rigid_body_rot": aq, "reference_rigid_body_rot": q}, parents)
    np.testing.assert_allclose(diag["world_body_rotation_error_rad"], angle, atol=1e-12, rtol=0)
    for key in ("rootcenter_heading_body_position_error_m", "heading_body_orientation_error_rad",
                "full_root_relative_body_orientation_error_rad", "parent_relative_joint_orientation_error_rad"):
        np.testing.assert_allclose(diag[key], 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(geodesic(rotation(q), rotation(-q)), 0, atol=1e-12, rtol=0)
    halfturn = q.copy()
    halfturn[..., 2], halfturn[..., 3] = 1, 0
    np.testing.assert_allclose(geodesic(rotation(halfturn), rotation(q)), np.pi, atol=1e-12, rtol=0)


def load_actual(result_record, parents, candidate=None):
    path = Path(result_record["path"])
    require(record(path) == result_record, "Actual result bytes changed")
    result = json.loads(path.read_text())
    require(result["status"] == "passed" and result["arguments"]["split"] == "validation" and
            len(result["sequences"]) == 22 and result["aggregate"]["observed_metric_steps"] == 11982,
            "Only genuine full22 validation captures allowed")
    require(tuple(result["physics"]["body_names"]) == BODY, "Capture canonical body mapping differs")
    if candidate:
        require([row["motion_id"] for row in result["sequences"]] ==
                [row["motion_id"] for row in candidate["sequences"]], "Paired concatenation must preserve identical clip order")
    arrays, per_clip, pairs, artifacts, max_error = [], [], [], [], 0.
    other = {row["motion_id"]: row for row in candidate["sequences"]} if candidate else {}
    for row in result["sequences"]:
        cap_path = Path(row["capture"])
        artifact = record(cap_path)
        require(cap_path.resolve().parent == (path.parent/"captures").resolve() and
                artifact["sha256"] == row["capture_sha256"], "Actual capture source/hash differs")
        with np.load(cap_path, allow_pickle=False) as archive:
            capture = {key: archive[key] for key in archive.files}
        frames = len(capture["reference_frame"])
        for key, shape in (("rigid_body_pos", (frames, 1, 24, 3)), ("reference_rigid_body_pos", (frames, 1, 24, 3)),
                           ("rigid_body_rot", (frames, 1, 24, 4)), ("reference_rigid_body_rot", (frames, 1, 24, 4))):
            require(capture[key].shape == shape and capture[key].dtype == np.float32 and
                    np.isfinite(capture[key]).all(), "Actual finite float32 paired body shape:"+key)
        metric = capture["metric_frame"]
        require(metric.dtype == np.bool_ and not metric[0] and metric[1:].all() and
                int(metric.sum()) == row["observed_metric_steps"], "Only original initial-reset exclusion allowed")
        require(np.array_equal(capture["native_reset_frame"], ~metric) and
                (np.diff(capture["reference_frame"]) == 1).all() and np.unique(capture["episode_id"]).size == 1,
                "Original continuous same-episode phase required")
        np.testing.assert_allclose(np.diff(capture["reference_time_s"]), 1/30, atol=1e-10, rtol=0)
        diagnostic = diagnostics(capture, parents)
        for key, source in (("world_body_position_error_m", "body_position_error_m"),
                            ("world_body_rotation_error_rad", "body_rotation_error_rad")):
            residual = float(np.max(np.abs(diagnostic[key][metric]-capture[source][metric])))
            max_error = max(max_error, residual)
            np.testing.assert_allclose(diagnostic[key][metric], capture[source][metric], atol=1e-7, rtol=0)
        chosen = {key: value[metric] for key, value in diagnostic.items()}
        arrays.append(chosen)
        per_clip.append({"motion_id": row["motion_id"], "metric_frames": int(metric.sum()),
                         "tracking_success_primary": row["tracking_success"], "fell": row["sequence_had_fall"]})
        artifacts.append(artifact)
        if candidate:
            other_row = other[row["motion_id"]]
            require(row["source_path"] == other_row["source_path"], "Paired reference source changed")
            with np.load(other_row["capture"], allow_pickle=False) as original:
                keys = sorted({key for key in capture if key.startswith("reference_")} | {
                    "metric_frame", "native_reset_frame", "episode_id", "trajectory_xy",
                    "waypoint_query_phase_frame", "waypoint_held_at_clip_end"})
                full = {key: np.array_equal(capture[key], original[key]) for key in keys}
                initial = {key: np.array_equal(capture[key][0], original[key][0]) for key in (
                    "root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
                    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history")}
            require(all(full.values()) and all(initial.values()), "Actual paired initial/reference inputs differ")
            pairs.append({"motion_id": row["motion_id"], "all_reference_and_clock_arrays_exact": full,
                          "all_initial_pose_velocity_history_arrays_exact": initial})
    combined = {key: np.concatenate([value[key] for value in arrays]) for key in arrays[0]}
    require(len(combined["root_xy_drift_m"]) == 11982, "All11982 metric frames must remain")
    for key, report_key in (("world_body_position_error_m", "frame_weighted_mean_body_position_error_m"),
                            ("world_body_rotation_error_rad", "frame_weighted_mean_body_rotation_error_rad")):
        np.testing.assert_allclose(combined[key].mean(), result["aggregate"][report_key], atol=1e-9, rtol=0)
    return result, combined, {"result": result_record, "captures": artifacts, "frames":11982,
        "clips": per_clip, "paired_reference_and_initial": pairs,
        "maximum_independent_matrix_vs_published_primary_residual": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(), "CPU-only fresh output directory required")
    comparison = json.loads(args.comparison.read_text())
    require(comparison["status"] == "passed" and all(comparison["prior12_evaluation_dicts_exact"].values()) and
            all(comparison["prior12_summary_dicts_exact"].values()), "Frozen13-run append must pass")
    mjcf = ROOT / "output/isolated_training/mixed_state_action_v1/code/protomotions/data/assets/mjcf/smpl_humanoid.xml"
    parents = parent_indices(mjcf)
    analytic_check(parents)
    runs, raw = {}, {}
    candidate, values, capture_review = load_actual(comparison["runs"]["state_action2000"]["result"], parents)
    raw["state_action2000"] = values
    for name in ("state_action2000", "mixed2000", "mixed1000", "backward_penalty1000"):
        row = comparison["runs"][name]
        if name != "state_action2000":
            _, values, capture_review = load_actual(row["result"], parents, candidate)
            raw[name] = values
        summary = {key: stats(value.mean(-1) if value.ndim == 2 else value) for key, value in values.items()}
        decomp = row["same_frame_decomposition"]
        for key, value in summary.items():
            decomp_key = "frame_mean_"+key if values[key].ndim == 2 else key
            np.testing.assert_allclose(value["mean"], decomp[decomp_key]["mean"], atol=1e-9, rtol=0)
        wrists = [{"body": body, **{key: stats(value[:, BODY.index(body)]) for key, value in values.items()
                    if value.ndim == 2}} for body in ("L_Wrist", "R_Wrist", "L_Hand", "R_Hand")]
        runs[name] = {"unchanged_primary_and_style_summary": row, "independent_raw_capture_review": capture_review,
                      "independent_diagnostics": summary, "independent_wrists_and_hands": wrists}
    differences = {}
    for name in ("mixed2000", "mixed1000", "backward_penalty1000"):
        delta = {key: raw["state_action2000"][key]-raw[name][key] for key in raw[name]}
        old_sequences = runs[name]["independent_raw_capture_review"]["clips"]
        new_sequences = {row["motion_id"]: row for row in runs["state_action2000"]["independent_raw_capture_review"]["clips"]}
        differences[name] = {"meaning": "candidate-minus-baseline on identical reference phases; negative error difference is smaller, not causal decomposition",
            "paired_frame_mean_error_difference": {key: stats(value.mean(-1) if value.ndim == 2 else value) for key,value in delta.items()},
            "new_successes": [row["motion_id"] for row in old_sequences if not row["tracking_success_primary"] and new_sequences[row["motion_id"]]["tracking_success_primary"]],
            "lost_successes": [row["motion_id"] for row in old_sequences if row["tracking_success_primary"] and not new_sequences[row["motion_id"]]["tracking_success_primary"]]}
    report = {"schema":"state_action.independent_full22_paired_review.v1", "status":"passed",
        "CPU_only":True,"SDK_started":False,"GPU_used":False,"sealed34_outputs_read":False,
        "source_comparison":record(args.comparison), "diagnostic_script":record(__file__), "actual_MJCF":record(mjcf),
        "independent_rotation_formula":"atan2(norm(skew(R_ref^T R_actual))/2,(trace(R_ref^T R_actual)-1)/2), normalized XYZW matrix conversion; no SciPy/primary metric helper",
        "metric_frames_per_run":11982,"all_failed_frames_kept":True,"primary_scores_modified":False,
        "analytic_rigid_transform_antipodal_halfturn_checks_passed":True, "runs":runs,"paired_difference":differences,
        "limits":["No fitted time/pelvis/yaw alignment replaces primary world-space errors; explicitly labelled local diagnostics remove own heading/root only.",
                  "Matrix audit proves consistency of actual paired recordings, not style significance or navigation stability.",
                  "State-action vs state-only bundles differ in D input, D initialization and AMP moments transfer; not a single-variable action-input causal test.",
                  "Budget16.384M matches mixed2000, exceeds mixed1000/negative1000 8.192M; one seed and repeatedly exposed validation, no34 or author equality."]}
    args.output.mkdir(parents=True)
    with (args.output/"audit.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":"passed","frames_per_run":11982,"compared_runs":list(runs),"output":str(args.output)}))


if __name__ == "__main__":
    main()
