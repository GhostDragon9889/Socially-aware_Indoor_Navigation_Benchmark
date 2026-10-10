"""Read-only CPU same-frame diagnostics; preserves the primary tracking metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from CrowdSim.humanoid.motion_tracking import SMPL_BODY_NAMES, body_tracking_errors  # noqa: E402

SPEED_EDGES = (0., .5, 1., 1.5, 2., 3., float("inf"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return None
    return {"count": len(values), "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values**2))), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def yaw_matrices(quaternions):
    matrix = Rotation.from_quat(quaternions).as_matrix()
    yaw = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    projection_norm = np.linalg.norm(matrix[:, :2, 0], axis=-1)
    heading = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    return yaw, heading, projection_norm


def orientation_errors(actual, reference):
    relative = np.einsum("...ji,...jk->...ik", reference, actual)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(relative.shape[:-2])


def parent_indices(mjcf):
    parents = {}

    def walk(body, parent):
        name = body.attrib["name"]
        parents[name] = parent
        for child in body.findall("body"):
            walk(child, name)

    for body in ET.parse(mjcf).getroot().find("worldbody").findall("body"):
        walk(body, None)
    if set(parents) != set(SMPL_BODY_NAMES):
        raise ValueError("Diagnostic MJCF body names differ from canonical SMPL")
    return tuple(0 if parents[name] is None else SMPL_BODY_NAMES.index(parents[name]) for name in SMPL_BODY_NAMES)


def diagnostic_arrays(capture, parents):
    actual, reference = (capture[key][:, 0].astype(np.float64) for key in ("rigid_body_pos", "reference_rigid_body_pos"))
    aq, rq = (capture[key][:, 0].astype(np.float64) for key in ("rigid_body_rot", "reference_rigid_body_rot"))
    primary = body_tracking_errors(actual, aq, reference, rq)
    ayaw, aheading, anorm = yaw_matrices(aq[:, 0])
    ryaw, rheading, rnorm = yaw_matrices(rq[:, 0])
    root_delta = actual[:, 0]-reference[:, 0]
    acentered, rcentered = actual-actual[:, 0, None], reference-reference[:, 0, None]
    alocal = np.einsum("tji,tbj->tbi", aheading, acentered)
    rlocal = np.einsum("tji,tbj->tbi", rheading, rcentered)
    ar = Rotation.from_quat(aq.reshape(-1, 4)).as_matrix().reshape(*aq.shape[:-1], 3, 3)
    rr = Rotation.from_quat(rq.reshape(-1, 4)).as_matrix().reshape(*rq.shape[:-1], 3, 3)
    aheadingrot = np.einsum("tji,tbjk->tbik", aheading, ar)
    rheadingrot = np.einsum("tji,tbjk->tbik", rheading, rr)
    arootrelative = np.einsum("tji,tbjk->tbik", ar[:, 0], ar)
    rrootrelative = np.einsum("tji,tbjk->tbik", rr[:, 0], rr)
    ajoint = np.einsum("tbji,tbjk->tbik", ar[:, parents], ar)
    rjoint = np.einsum("tbji,tbjk->tbik", rr[:, parents], rr)
    body_relative_difference = acentered-rcentered
    return {
        "root_xy_drift_m": np.linalg.norm(root_delta[:, :2], axis=-1),
        "root_xyz_drift_m": np.linalg.norm(root_delta, axis=-1),
        "root_height_abs_error_m": np.abs(root_delta[:, 2]),
        "root_height_signed_error_m": root_delta[:, 2],
        "root_yaw_abs_wrapped_error_rad": np.abs(np.arctan2(np.sin(ayaw-ryaw), np.cos(ayaw-ryaw))),
        "root_rotation_geodesic_error_rad": primary["body_rotation_error_rad"][:, 0],
        "world_body_position_error_m": primary["body_position_error_m"],
        "world_body_rotation_error_rad": primary["body_rotation_error_rad"],
        "rootcenter_only_body_position_error_m": np.linalg.norm(body_relative_difference, axis=-1),
        "rootcenter_heading_body_position_error_m": np.linalg.norm(alocal-rlocal, axis=-1),
        "heading_body_orientation_error_rad": orientation_errors(aheadingrot, rheadingrot),
        "full_root_relative_body_orientation_error_rad": orientation_errors(arootrelative, rrootrelative),
        "parent_relative_joint_orientation_error_rad": orientation_errors(ajoint, rjoint),
        "reference_planar_speed_m_s": np.linalg.norm(capture["reference_root_vel"][:, 0, :2].astype(np.float64), axis=-1),
        "actual_planar_speed_m_s": np.linalg.norm(capture["root_vel"][:, 0, :2].astype(np.float64), axis=-1),
        "world_body_position_squared_error_m2": primary["body_position_error_m"]**2,
        "root_xyz_squared_drift_m2": np.sum(root_delta**2, axis=-1),
        "rootcenter_only_body_squared_error_m2": np.sum(body_relative_difference**2, axis=-1),
        "translation_pose_cross_term_m2": 2*np.sum(root_delta[:, None]*body_relative_difference, axis=-1),
        "heading_projection_min_norm": np.minimum(anorm, rnorm),
    }, primary


def summarize(arrays):
    summary = {}
    for key, value in arrays.items():
        if key.endswith("squared_error_m2") or key.endswith("squared_drift_m2") or key.endswith("cross_term_m2"):
            continue
        if np.asarray(value).ndim == 2:
            summary[f"frame_mean_{key}"] = stats(value.mean(-1))
        else:
            summary[key] = stats(value)
    world = float(arrays["world_body_position_squared_error_m2"].mean())
    root = float(arrays["root_xyz_squared_drift_m2"].mean())
    relative = float(arrays["rootcenter_only_body_squared_error_m2"].mean())
    cross = float(arrays["translation_pose_cross_term_m2"].mean())
    summary["exact_world_position_mse_identity"] = {
        "world_body_mean_squared_error_m2": world, "root_xyz_mean_squared_drift_m2": root,
        "rootcenter_pose_mean_squared_error_m2": relative, "translation_pose_cross_term_m2": cross,
        "closure_error_m2": world-root-relative-cross,
        "terms_are_not_independent_variance_percentages": True,
    }
    summary["heading_removal_descriptive_rotation_mean_reduction_fraction"] = (
        1-float(arrays["heading_body_orientation_error_rad"].mean())/float(arrays["world_body_rotation_error_rad"].mean())
        if arrays["world_body_rotation_error_rad"].mean() > 0 else None)
    summary["near_vertical_root_heading_projection_frames_lt_point1"] = int((arrays["heading_projection_min_norm"] < .1).sum())
    return summary


def check_analytic_invariance(parents):
    """Independent rigid transform fixture; no rollout time alignment fitting."""
    generator = np.random.default_rng(17)
    reference = generator.normal(size=(2, 1, 24, 3))
    rq = Rotation.random(48, random_state=generator).as_quat().reshape(2, 1, 24, 4)
    # Known rotation on the entire world state must disappear only from the
    # declared local diagnostics, while remaining in primary world errors.
    heading = Rotation.from_euler("z", .8)
    actual = heading.apply(reference.reshape(-1, 3)).reshape(reference.shape)+np.array([4., -3., 2.])
    aq = (heading*Rotation.from_quat(rq.reshape(-1, 4))).as_quat().reshape(rq.shape)
    velocity = np.zeros((2, 1, 3))
    arrays, primary = diagnostic_arrays({"rigid_body_pos": actual, "reference_rigid_body_pos": reference,
        "rigid_body_rot": aq, "reference_rigid_body_rot": rq, "root_vel": velocity, "reference_root_vel": velocity}, parents)
    assert primary["body_position_error_m"].mean() > 1.
    np.testing.assert_allclose(primary["body_rotation_error_rad"], .8, atol=1e-12)
    for key in ("rootcenter_heading_body_position_error_m", "heading_body_orientation_error_rad",
                "full_root_relative_body_orientation_error_rad", "parent_relative_joint_orientation_error_rad"):
        assert np.max(arrays[key]) < 1e-12, key
    assert abs(summarize(arrays)["exact_world_position_mse_identity"]["closure_error_m2"]) < 1e-12


def decompose(input_dir):
    result_path = input_dir/"result.json"
    source = json.loads(result_path.read_text())
    if source["status"] != "passed":
        raise ValueError("The captured rollout result must be passed")
    mjcf = ROOT/"protomotions/data/assets/mjcf/smpl_humanoid.xml"
    parents = parent_indices(mjcf)
    check_analytic_invariance(parents)
    rows, all_arrays, initial, capture_sources = [], [], [], []
    for sequence in source["sequences"]:
        path = Path(sequence["capture"])
        if path.parent.resolve() != (input_dir/"captures").resolve():
            raise ValueError("Capture path is not bound to the supplied source result")
        digest = sha256(path)
        if digest != sequence["capture_sha256"]:
            raise ValueError("Capture SHA differs from source result")
        with np.load(path, allow_pickle=False) as archive:
            capture = {key: archive[key] for key in archive.files}
        frames = len(capture["reference_frame"])
        for key, shape in (("rigid_body_pos", (frames, 1, 24, 3)), ("reference_rigid_body_pos", (frames, 1, 24, 3)),
                           ("rigid_body_rot", (frames, 1, 24, 4)), ("reference_rigid_body_rot", (frames, 1, 24, 4)),
                           ("root_vel", (frames, 1, 3)), ("reference_root_vel", (frames, 1, 3))):
            if capture[key].shape != shape or not np.isfinite(capture[key]).all():
                raise ValueError(f"Invalid {key} capture shape/finite values")
        metric = capture["metric_frame"]
        if metric.dtype != np.bool_ or metric.shape != (frames,) or int(metric.sum()) != sequence["observed_metric_steps"]:
            raise ValueError("Invalid metric mask/count")
        if metric[0] or not np.all(metric[1:]) or not np.array_equal(capture["native_reset_frame"], ~metric):
            raise ValueError("Expected one initial reset followed by same-episode metric frames")
        if np.any(np.diff(capture["reference_frame"]) != 1) or np.unique(capture["episode_id"]).size != 1:
            raise ValueError("Capture contains a phase skip or episode transition")
        np.testing.assert_allclose(np.diff(capture["reference_time_s"]), 1/30, atol=1e-10, rtol=0)
        arrays, primary = diagnostic_arrays(capture, parents)
        for key in primary:
            np.testing.assert_allclose(primary[key], capture[key], atol=1e-12, rtol=0)
        selected = {key: value[metric] for key, value in arrays.items()}
        np.testing.assert_allclose(selected["world_body_position_error_m"].mean(), sequence["mean_body_position_error_m"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(selected["world_body_rotation_error_rad"].mean(), sequence["mean_body_rotation_error_rad"], atol=1e-12, rtol=0)
        init = {"motion_id": sequence["motion_id"],
                "mean_body_position_error_m": float(primary["body_position_error_m"][0].mean()),
                "max_body_position_error_m": float(primary["body_position_error_m"][0].max()),
                "mean_body_rotation_error_rad": float(primary["body_rotation_error_rad"][0].mean()),
                "max_body_rotation_error_rad": float(primary["body_rotation_error_rad"][0].max())}
        row = {"motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
               "metric_frames": int(metric.sum()), "tracking_success_primary": sequence["tracking_success"],
               "sequence_had_fall": sequence["sequence_had_fall"], "initial": init,
               "diagnostic": summarize(selected)}
        rows.append(row)
        initial.append(init)
        all_arrays.append(selected)
        capture_sources.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    aggregate = {key: np.concatenate([arrays[key] for arrays in all_arrays]) for key in all_arrays[0]}
    if len(aggregate["root_xy_drift_m"]) != source["aggregate"]["observed_metric_steps"]:
        raise ValueError("Aggregate observed metric frame count changed")
    for key, diagnostic_key in (("frame_weighted_mean_body_position_error_m", "world_body_position_error_m"),
                                ("frame_weighted_mean_body_rotation_error_rad", "world_body_rotation_error_rad")):
        np.testing.assert_allclose(source["aggregate"][key], aggregate[diagnostic_key].mean(), atol=1e-12, rtol=0)
    bodies = [{"body": name, **{key: stats(aggregate[key][:, index]) for key in (
        "world_body_position_error_m", "world_body_rotation_error_rad", "rootcenter_only_body_position_error_m",
        "rootcenter_heading_body_position_error_m", "heading_body_orientation_error_rad",
        "full_root_relative_body_orientation_error_rad", "parent_relative_joint_orientation_error_rad")}}
        for index, name in enumerate(SMPL_BODY_NAMES)]
    speed_bins = []
    for low, high in zip(SPEED_EDGES, SPEED_EDGES[1:]):
        mask = (aggregate["reference_planar_speed_m_s"] >= low) & (aggregate["reference_planar_speed_m_s"] < high)
        speed_bins.append({"reference_speed_range_m_s_half_open": [low, high if np.isfinite(high) else None],
                           "metric_frames": int(mask.sum()),
                           "diagnostic": summarize({key: value[mask] for key, value in aggregate.items()}) if mask.any() else None})
    return {"schema": "navisaaclab.tracking_decomposition.v1", "status": "passed",
        "CPU_only": True, "Kit_started": False, "benchmark_code_or_metric_modified": False,
        "source_result": {"path": str(result_path), "sha256": sha256(result_path)},
        "source_captures": capture_sources,
        "source_protocol": source["protocol"], "reference_velocity_contract": source.get("reference_velocity_contract"),
        "checkpoint_sha256": source["arguments"]["checkpoint_sha256"],
        "selected_rollout_source_sha256": source["selected_source_sha256"],
        "diagnostic_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "diagnostic_mjcf": {"path": str(mjcf), "sha256": sha256(mjcf)},
        "primary_aggregate_unchanged": source["aggregate"],
        "initial_mapping_checks": {
            "max_over_clips_mean_body_position_error_m": max(row["mean_body_position_error_m"] for row in initial),
            "max_over_clips_mean_body_rotation_error_rad": max(row["mean_body_rotation_error_rad"] for row in initial),
            "max_over_all_initial_body_position_error_m": max(row["max_body_position_error_m"] for row in initial),
            "max_over_all_initial_body_rotation_error_rad": max(row["max_body_rotation_error_rad"] for row in initial),
            "clip_mean_both_lt_1point5e_minus6": all(row["mean_body_position_error_m"] < 1.5e-6 and row["mean_body_rotation_error_rad"] < 1.5e-6 for row in initial),
            "threshold_does_not_claim_each_individual_body_error_below_1point5e_minus6": True},
        "aggregate_diagnostic": summarize(aggregate), "by_body": bodies, "by_clip": rows,
        "by_reference_speed_bin": speed_bins,
        "definitions": {
            "metric_frames": "All original post-step metric frames; initial excluded; fall frames retained; no besttime/phase alignment",
            "heading": "atan2(projected local+X worldY, worldX) from current root actor rotation, independently foractual/reference",
            "local_positions": "Subtract own root XYZ then rotate by own inverse root heading; height/tilt are not fitted",
            "heading_orientation": "Rz(-own rootyaw) @ body rotation; SO3geodesic compared atsameframe",
            "full_root_relative_orientation": "Rroot^-1 @ Rbody extra diagnostic; root tilt removed only inthis separately labelledfeature",
            "parent_relative_joint_orientation": "Rparent^-1 @ Rchild fromactualMJCF namedtree; Pelvis identity included in24mean",
            "reference_speed_bins": "Captured reference_root_vel XY norm; legacy smoothed reference estimate retained",
            "decomposition_not_additive": "Heading removal mean reduction isdescriptive, not explained variance or causal share; positionMSE identity includes cross term",
            "not_author_metric_replacement": True,
            "author_exact_pose_error_protocol_unknown": True,
            "mapping_evidence_scope": "Tiny initial pose errors confirm namedrig/reset numerical agreement, notlateractor trackingquality"},
        "analytic_rigid_transform_check_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decompose(args.input.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "initial": result["initial_mapping_checks"],
                      "primary": result["primary_aggregate_unchanged"], "diagnostic": result["aggregate_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
