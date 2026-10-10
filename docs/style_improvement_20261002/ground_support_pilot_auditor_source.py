#!/usr/bin/env python3
"""Read-only NumPy audit of a published live four-substep support capture.

This does not import Torch, USD, Kit, the controller, or the capture producer.
It independently reconstructs the published force, COM and velocity summaries.
The result certifies consistency of recorded evidence, not gait quality or
contact-patch slip. Output paths must be new; source artifacts are never written.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

COMMON = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)
FEET = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
PEER_FIELDS = ("static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n")
REPO = Path(__file__).resolve().parents[2]


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


class Audit:
    def __init__(self):
        self.checks = []

    def require(self, condition, name, **evidence):
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, **evidence})
        if not passed:
            raise ValueError(f"Capture audit failed: {name}")

    def equal(self, left, right, name):
        self.require(np.array_equal(left, right), name)

    def close(self, left, right, name, *, atol=1e-4, rtol=1e-5):
        left, right = np.asarray(left), np.asarray(right)
        error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        self.require(np.allclose(left, right, atol=atol, rtol=rtol), name,
                     maximum_absolute_error=error, atol=atol, rtol=rtol)


def rotation_matrix_xyzw(quat):
    """Independent SO(3) matrix construction, evaluated in float64."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), axis=-1).reshape((*q.shape[:-1], 3, 3))


def verify_shapes(audit, ground, physical, t, b, floors):
    floats = {
        "foot_floor_normal_force_w": (t, 4, b, 4, floors, 3),
        "foot_normal_force_vector_mean_w": (t, b, 4, 3),
        "foot_normal_load_mean_n": (t, b, 4), "foot_normal_load_peak_n": (t, b, 4),
        "side_normal_load_substeps_n": (t, 4, b, 2),
        "side_normal_load_mean_n": (t, b, 2), "side_normal_load_peak_n": (t, b, 2),
        "double_support_fraction": (t, b), "no_support_fraction": (t, b),
        "foot_body_origin_pos_w": (t, 4, b, 4, 3), "foot_body_origin_vel_w": (t, 4, b, 4, 3),
        "body_origin_tangential_speed_proxy_mps": (t, 4, b, 4),
        "supported_body_origin_tangential_speed_proxy_mean_mps": (t, b, 4),
        "body_link_pos_w": (t, 4, b, 24, 3), "body_link_quat_w": (t, 4, b, 24, 4),
        "body_link_lin_vel_w": (t, 4, b, 24, 3), "body_link_ang_vel_w": (t, 4, b, 24, 3),
        "body_com_pos_w": (t, 4, b, 24, 3), "local_com_pose_b": (t, 4, b, 24, 7),
        "body_masses_kg": (t, 4, b, 24), "com_composition_max_error_m": (t, 4),
        "mass_weighted_world_com": (t, 4, b, 3),
    }
    booleans = {
        "foot_contact_substeps": (t, 4, b, 4), "side_contact_substeps": (t, 4, b, 2),
        "side_contact_any": (t, b, 2), "side_contact_all": (t, b, 2),
        "supported_speed_proxy_valid": (t, b, 4), "reset_mask": (t, b),
        "terminated": (t, b), "truncated": (t, b), "valid": (t,),
    }
    integers = {
        "control_step": (t,), "control_step_after": (t,), "episode_ids": (t, b),
        "physics_step_end": (t,), "substep_valid_count": (t,), "monitor_reset_count": (t,),
        "actor_substep_valid_count": (t, b), "physics_step": (t, 4), "physics_step_before": (t,),
    }
    names = {"common_body_names": (24,), "foot_names": (4,)}
    audit.require(set(ground) == set(floats) | set(booleans) | set(integers) | set(names),
                  "exact ground-support v1 field set")
    for specification, kind in ((floats, "f"), (booleans, "b"), (integers, "iu"), (names, "US")):
        for key, shape in specification.items():
            value = ground[key]
            audit.require(value.shape == shape and value.dtype.kind in kind, f"ground shape/dtype: {key}")
            if kind == "f":
                audit.require(value.dtype == np.float32 and np.isfinite(value).all(), f"ground finite float32: {key}")
            if kind == "iu":
                audit.require((value >= 0).all(), f"ground nonnegative integer: {key}")
    physical_shapes = {
        "root_pos": (t, b, 3), "body_pos": (t, b, 24, 3), "root_rot": (t, b, 4),
        "dof_pos": (t, b, 69), "dof_vel": (t, b, 69), "dof_forces": (t, b, 69),
        "signed_power_w": (t, b), "episode_ids": (t, b), "reset_mask": (t, b),
        "terminated": (t, b), "truncated": (t, b),
        "static_contact": (t, b), "peer_contact": (t, b, b),
        "static_contact_max_force_n": (t, b), "peer_contact_max_force_n": (t, b, b),
    }
    for key, shape in physical_shapes.items():
        audit.require(key in physical and physical[key].shape == shape, f"physical shape: {key}")
    for key, value in physical.items():
        audit.require(value.dtype.kind in "bifu" and (value.dtype.kind != "f" or np.isfinite(value).all()),
                      f"physical numeric/finite: {key}")
    audit.require(not set(physical) & (set(ground) - {"episode_ids", "reset_mask", "terminated", "truncated"}),
                  "separate ground archive; no added foot/COM fields in physical capture")


def audit_capture(directory, audit):
    directory = Path(directory).resolve()
    result_path, ground_path, physical_path = (directory / name for name in
                                             ("result.json", "ground_support.npz", "physical_states.npz"))
    result = json.loads(result_path.read_text())
    report = {"input": str(directory), "CPU_only": True, "Kit_started_by_auditor": False,
              "numpy": np.__version__, "artifacts": {"result": fingerprint(result_path),
              "ground_support": fingerprint(ground_path), "physical_states": fingerprint(physical_path)}}
    audit.require(result["status"] == "passed", "source runtime completed successfully")
    args = result["arguments"]
    t, b = args["evaluate_steps"], args["num_envs"]
    audit.require(type(t) is int and t > 0 and type(b) is int and b > 0, "declared positive evaluation dimensions")
    audit.require(args["measure_ground_support"] is True and args["pairwise_contacts"] is True,
                  "actual foot capture and independent peer capture both requested")
    audit.require(args["power_source"] == "projected_joint_effort_substep_mean", "measured-substep world contract")
    measure = result["ground_support_measurement"]
    contract, published = measure["contract"], measure["capture"]
    floor = contract["floor_measurement"]
    for name, record in (("ground_support", published), ("physical_states", result["state_capture"])):
        audit.require(Path(record["path"]).resolve() == Path(report["artifacts"][name]["path"]) and
                      Path(record["path"]).resolve().parent == directory, f"source-result artifact identity: {name}")
    actual_ground = report["artifacts"]["ground_support"]
    audit.require(published["bytes"] == actual_ground["bytes"] and published["sha256"] == actual_ground["sha256"],
                  "runtime-published ground file bytes and full SHA256")
    audit.require(published["completed_controls"] == t and published["actors"] == b and
                  published["completed_physics_substeps"] == 4*t and published["all_controls_valid"] is True and
                  result["actual_control_steps"] == t, "runtime counts match requested controls")
    audit.require(contract["schema"] == "navisaaclab.live_foot_support_capture.v1" and
                  floor["units"] == "N" and floor["physics_dt_s"] == 1/120 and
                  floor["substeps_per_control"] == 4 and result["state_capture"]["frequency_hz"] == 30 and
                  result["physics"]["physics_hz"] == 120 and result["physics"]["control_hz"] == 30,
                  "explicit 120Hz four-substep / 30Hz control contract")
    audit.require(contract["extra_physics_steps"] == 0 and contract["pose_writes"] == 0 and
                  contract["native_engine_time_observed"] is False,
                  "submitted counter provenance; no claim of native engine time")
    audit.require(contract["reference_true_com_available"] is False and contract["reference_true_com"] is None and
                  contract["friction_force_observed"] is False and contract["contact_patch_slip_observed"] is False and
                  contract["author_metric_equivalence"] is False,
                  "missing reference COM/friction/contact-patch channels remain explicit")
    audit.require(contract["native_capture_probe_passed"] is False and measure["native_capture_probe_passed"] is False and
                  floor["native_physical_probe_passed"] is False, "historical probe flags retained unchanged")
    with np.load(ground_path, allow_pickle=False) as archive:
        ground = {key: archive[key] for key in archive.files}
    with np.load(physical_path, allow_pickle=False) as archive:
        physical = {key: archive[key] for key in archive.files}
    verify_shapes(audit, ground, physical, t, b, len(floor["floor_collider_paths"]))
    audit.require(published["array_shapes"] == {key: list(value.shape) for key, value in ground.items()},
                  "runtime-declared shapes bind exact archive")
    audit.equal(ground["common_body_names"], np.asarray(COMMON), "canonical common body order")
    audit.equal(ground["foot_names"], np.asarray(FEET), "canonical foot order")
    audit.require(tuple(result["state_capture"]["body_names"]) == COMMON == tuple(contract["common_body_names"]) ==
                  tuple(result["physics"]["body_names"]), "capture, physics and COM body order agreement")
    native = contract["native_body_names"]
    audit.require(len(native) == 24 and len(set(native)) == 24 and set(native) == set(COMMON) and
                  native == result["physics"]["live_native_body_names"], "native/common named bijection")
    feet_indices = [COMMON.index(name) for name in FEET]
    peers = result["physical_contact_measurement"]
    actor_paths = peers["actor_body_paths"]
    expected_sensors = [actor_paths[actor][index] for actor in range(b) for index in feet_indices]
    audit.require(floor["sensor_paths"] == expected_sensors and floor["native_sensor_paths"] == expected_sensors and
                  floor["foot_names"] == list(FEET) and contract["foot_names"] == list(FEET) and
                  all(path.rsplit("/", 1)[-1] == FEET[i % 4] for i, path in enumerate(expected_sensors)),
                  "exact foot sensors mapped to independently published common-body paths")
    floor_paths = floor["floor_collider_paths"]
    audit.require(floor_paths == ["/World/ground/terrain/mesh"] and
                  floor["native_filter_paths"] == [floor_paths for _ in range(b*4)] and
                  not set(floor_paths) & set(peers["static_collider_paths"]) and
                  not set(floor_paths) & set(expected_sensors), "floor-only exact collider filter; no obstacle/peer mixing")
    audit.require(contract["quaternion_order"] == "xyzw" and
                  contract["link_velocity_source"] == "robot.data.body_link_lin_vel_w; body-origin, not COM alias",
                  "explicit quaternion and body-origin velocity semantics")
    ownership = contract["physics_view_ownership"]
    audit.require(ownership["active_manager_getter_same_view"] is True and
                  ownership["legacy_isaacsim_SimulationManager_used"] is False and
                  ownership["new_simulation_view_created"] is False and ownership["private_view_lookup_used"] is False,
                  "declared contact views use existing active physics manager")
    audit.equal(ground["control_step"], np.arange(t), "all evaluation before-control counters")
    audit.equal(ground["control_step_after"], np.arange(1, t+1), "all evaluation post-control counters")
    before = int(ground["physics_step_before"][0])
    audit.equal(ground["physics_step_before"], before + np.arange(t)*4, "no extra/missing inter-control physics submissions")
    audit.equal(ground["physics_step"].reshape(-1), np.arange(before+1, before+4*t+1), "every actual completed substep consecutive")
    audit.equal(ground["physics_step_end"], before+(np.arange(t)+1)*4, "exact control end counters")
    audit.require(ground["valid"].all() and (ground["substep_valid_count"] == 4).all() and
                  (ground["actor_substep_valid_count"] == 4).all(), "no partial/padded controls")
    for name in ("episode_ids", "reset_mask", "terminated", "truncated"):
        audit.equal(ground[name], physical[name], f"ground vs physical same-frame {name}")
    audit.require(ground["reset_mask"][0].all() and np.all(np.diff(ground["episode_ids"], axis=0) >= 0),
                  "initial reset marker and monotonic actor episode IDs")
    changed = np.diff(ground["episode_ids"], axis=0) != 0
    audit.equal(changed, ground["reset_mask"][1:], "before-frame reset markers exactly match changed episodes")
    ended = ground["terminated"] | ground["truncated"]
    audit.equal(ended[:-1], changed, "all captured terminations followed by reset on next captured frame")
    reset_controls = changed.any(-1).astype(np.int64)
    audit.equal(ground["monitor_reset_count"], np.r_[0, np.cumsum(reset_controls)], "monitor reset count follows published boundaries")
    audit.equal(ground["body_link_pos_w"][:, -1], physical["body_pos"], "final-substep body positions bitwise physical post-state")
    audit.equal(ground["body_link_quat_w"][:, -1, :, 0], physical["root_rot"], "final-substep Pelvis XYZW bitwise physical root rotation")
    audit.equal(physical["root_pos"], physical["body_pos"][:, :, 0], "physical root position equals named Pelvis origin")
    audit.equal(ground["foot_body_origin_pos_w"], ground["body_link_pos_w"][..., feet_indices, :], "all substep foot origins use named link positions")
    audit.equal(ground["foot_body_origin_vel_w"], ground["body_link_lin_vel_w"][..., feet_indices, :], "all substep foot velocities use named link velocities")
    quaternions = ground["body_link_quat_w"]
    audit.close(np.linalg.norm(quaternions, axis=-1), 1., "unit body XYZW quaternions", atol=1e-3, rtol=0)
    audit.close(np.linalg.norm(ground["local_com_pose_b"][..., 3:], axis=-1), 1., "unit local COM principal-axis quaternions", atol=1e-3, rtol=0)
    com_errors = []
    for start in range(0, t, 128):
        stop = min(t, start+128)
        rotation = rotation_matrix_xyzw(quaternions[start:stop])
        reconstructed = ground["body_link_pos_w"][start:stop] + np.einsum(
            "...ij,...j->...i", rotation, ground["local_com_pose_b"][start:stop, ..., :3])
        com_errors.append(np.max(np.abs(reconstructed-ground["body_com_pos_w"][start:stop]), axis=(2, 3, 4)))
    com_errors = np.concatenate(com_errors)
    audit.require(com_errors.max() <= contract["com_composition_tolerance_m"], "independent SO3 local-to-world COM composition",
                  maximum_component_error_m=float(com_errors.max()), tolerance_m=contract["com_composition_tolerance_m"])
    audit.close(com_errors, ground["com_composition_max_error_m"], "recorded COM residual agrees with independent float64 reconstruction", atol=2e-6, rtol=0)
    masses = ground["body_masses_kg"]
    audit.require((masses > 0).all(), "every measured body mass positive", minimum_mass_kg=float(masses.min()))
    audit.close(masses, np.asarray(result["physics"]["live_body_masses_kg"]), "all measured masses agree with common-order runtime physics manifest", atol=1e-6, rtol=0)
    native_masses = np.asarray(result["physics"]["live_native_body_masses_kg"])
    audit.close(native_masses[[native.index(name) for name in COMMON]], masses[0, 0, 0], "independent native-to-common mass permutation", atol=1e-6, rtol=0)
    weighted_com = (ground["body_com_pos_w"].astype(np.float64)*masses[..., None]).sum(-2)/masses.sum(-1)[..., None]
    audit.close(weighted_com, ground["mass_weighted_world_com"], "independent positive-mass-weighted true world COM", atol=5e-6, rtol=0)
    raw = ground["foot_floor_normal_force_w"]
    foot_vector = raw.sum(-2)
    foot_load = np.linalg.norm(foot_vector, axis=-1)
    side_load = foot_load.reshape(t, 4, b, 2, 2).sum(-1)
    threshold = float(floor["threshold_n"])
    audit.require(np.isfinite(threshold) and threshold > 0 and threshold == args["ground_support_threshold_n"], "positive declared foot contact threshold")
    contact = foot_load > threshold
    side_contact = contact.reshape(t, 4, b, 2, 2).any(-1)
    reconstructed = {
        "foot_normal_force_vector_mean_w": foot_vector.mean(1),
        "foot_normal_load_mean_n": foot_load.mean(1), "foot_normal_load_peak_n": foot_load.max(1),
        "side_normal_load_substeps_n": side_load, "side_normal_load_mean_n": side_load.mean(1),
        "side_normal_load_peak_n": side_load.max(1), "foot_contact_substeps": contact,
        "side_contact_substeps": side_contact, "side_contact_any": side_contact.any(1),
        "side_contact_all": side_contact.all(1), "double_support_fraction": side_contact.all(-1).mean(1),
        "no_support_fraction": (~side_contact.any(-1)).mean(1),
    }
    for key, expected in reconstructed.items():
        if expected.dtype == np.bool_:
            audit.equal(ground[key], expected, f"independent raw-force support reconstruction: {key}")
        else:
            audit.close(ground[key], expected, f"independent raw-force summary: {key}")
    normal = np.asarray(contract["ground_normal_w"], dtype=np.float64)
    audit.close(np.linalg.norm(normal), 1., "declared ground plane unit normal", atol=1e-6, rtol=0)
    velocity = ground["foot_body_origin_vel_w"].astype(np.float64)
    tangent = velocity - np.einsum("...i,i->...", velocity, normal)[..., None]*normal
    speed = np.linalg.norm(tangent, axis=-1)
    supported_count = contact.sum(1)
    supported_mean = (speed*contact).sum(1)/np.maximum(supported_count, 1)
    audit.close(ground["body_origin_tangential_speed_proxy_mps"], speed, "independent foot body-origin tangent speed", atol=2e-6, rtol=1e-6)
    audit.equal(ground["supported_speed_proxy_valid"], supported_count > 0, "supported proxy valid mask: no unsupported zero promoted to measurement")
    audit.close(ground["supported_body_origin_tangential_speed_proxy_mean_mps"], supported_mean, "independent contact-conditioned substep speed mean", atol=2e-6, rtol=1e-6)
    peer_threshold = peers["threshold_n"]
    for kind in ("static", "peer"):
        force = physical[f"{kind}_contact_max_force_n"]
        audit.require((force >= 0).all() and physical[f"{kind}_contact"].dtype == np.bool_, f"original {kind} contact field finite/nonnegative/boolean")
        audit.equal(physical[f"{kind}_contact"], force > peer_threshold, f"original {kind} contact threshold convention")
    audit.require(not np.diagonal(physical["peer_contact"], axis1=1, axis2=2).any() and
                  (np.diagonal(physical["peer_contact_max_force_n"], axis1=1, axis2=2) == 0).all(), "original peer channel excludes self contacts")
    report["counts"] = {"controls": t, "actors": b, "control_actor_frames": t*b, "physics_substeps": 4*t,
                        "substep_actor_frames": 4*t*b, "foot_substep_observations": 16*t*b,
                        "control_hz": 30, "physics_hz": 120, "duration_s": t/30,
                        "first_before_submitted_step": before, "last_completed_submitted_step": before+4*t,
                        "native_engine_time_observed": False, "falls": result["fall_resets"],
                        "terminated_actor_frames": int(ground["terminated"].sum()),
                        "time_limits": result["time_limit_resets"], "reset_actor_frames_after_initial": int(changed.sum()),
                        "route_vertices_reached": result["route_vertices_reached"]}
    audit.require(report["counts"]["falls"] == int(ground["terminated"].sum()) and
                  report["counts"]["time_limits"] == int(ground["truncated"].sum()), "runtime fall/time-limit totals match captured terminal masks")
    per_actor = []
    for actor in range(b):
        sides = side_contact[:, :, actor]
        per_actor.append({"actor": actor, "double_support_fraction": float(sides.all(-1).mean()),
                          "left_only_support_fraction": float((sides[..., 0] & ~sides[..., 1]).mean()),
                          "right_only_support_fraction": float((~sides[..., 0] & sides[..., 1]).mean()),
                          "no_foot_support_fraction": float((~sides.any(-1)).mean()),
                          "supported_body_origin_tangent_speed_proxy_mps": {
                              name: stats(speed[:, :, actor, foot][contact[:, :, actor, foot]])
                              for foot, name in enumerate(FEET)}})
    report["support_statistics"] = {
        "denominator": "completed 120Hz actor-substeps; strict foot resultant normal-vector magnitude > threshold",
        "threshold_n": threshold, "double_support_fraction": float(side_contact.all(-1).mean()),
        "no_foot_support_fraction": float((~side_contact.any(-1)).mean()),
        "supported_foot_substep_count": int(contact.sum()), "foot_normal_load_n": stats(foot_load),
        "total_foot_normal_load_per_actor_n": stats(foot_load.sum(-1)), "per_actor": per_actor,
    }
    report["body_origin_motion_proxy"] = {
        "definition": "norm(v_link_origin - dot(v_link_origin,n)*n), only substeps with foot resultant > threshold",
        "aggregate_supported_mps": stats(speed[contact]), "contact_patch_slip_observed": False,
        "friction_force_observed": False, "interpretation": "Foot origins can move under rolling/rotation; this is not true contact-point slip.",
    }
    report["COM_statistics"] = {
        "maximum_independent_composition_error_m": float(com_errors.max()),
        "runtime_recorded_maximum_composition_error_m": float(ground["com_composition_max_error_m"].max()),
        "total_mass_kg": float(masses[0, 0, 0].astype(np.float64).sum()),
        "mass_weighted_true_world_com_height_m": stats(weighted_com[..., 2]),
        "minimum_pelvis_height_m": float(physical["root_pos"][..., 2].min()),
        "reference_true_com": None,
    }
    report["original_obstacle_peer_channels"] = {
        "field_names": list(PEER_FIELDS), "floor_excluded": True,
        "static_contact_actor_frames": int(physical["static_contact"].sum()),
        "peer_contact_actor_frames": int(physical["peer_contact"].any(-1).sum()),
        "peer_contact_unordered_pair_frames": int(np.triu(physical["peer_contact"] | physical["peer_contact"].swapaxes(1, 2), 1).sum()),
        "maximum_static_force_n": float(physical["static_contact_max_force_n"].max()),
        "maximum_peer_force_n": float(physical["peer_contact_max_force_n"].max()),
        "raw_peer_force_matrix_not_archived": True,
        "boundary": "Independent raw peer-force reconstruction unavailable; published original fields checked for shape, semantics and thresholds.",
    }
    report["sources"] = verify_sources(audit, result)
    report["actual_native_capture_validated"] = True
    report["historical_native_capture_probe_passed"] = False
    report["historical_native_physical_probe_passed"] = False
    report["author_experiment_reproduced"] = False
    report["quality_convergence_claim"] = None
    report["limitations"] = [
        "CPU consistency audit of actual native recording; auditor did not execute Kit or add physics steps.",
        "Time uses exact submitted-step counters and declared dt; native engine time was not captured.",
        "Historical neutral-stance probe failed; its false flags are preserved. Valid learned-policy capture is a separate fact.",
        "Floor matrices are normal-force vector sums, not friction/contact-patch force decomposition.",
        "Body-origin contact-conditioned tangential speed is a motion proxy, not true slip; rolling can increase it.",
        "Reference AMASS true COM/contact/friction not available; no reference-ground-truth comparison fabricated.",
        "No author eight-scene, model-equivalence, stability or training-convergence claim follows from this audit.",
    ]
    return report


def verify_sources(audit, result):
    """Bind actual paths/hashes without pretending inherited metadata is runtime."""
    source_manifest = REPO / "docs/style_improvement_20261002/ground_support_integration_review.json"
    frozen = json.loads(source_manifest.read_text())["source_files"]
    source_records = {}
    for name, expected in frozen.items():
        actual = fingerprint(REPO / name)
        audit.require(actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"],
                      f"current integration source matches frozen CPU-reviewed source: {name}")
        source_records[name] = actual
    tree = ast.parse((REPO / "CrowdSim/humanoid/ground_support_archive.py").read_text())
    peer_assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                           any(isinstance(target, ast.Name) and target.id == "PEER_FIELDS" for target in node.targets))
    audit.require(ast.literal_eval(peer_assignment.value) == PEER_FIELDS, "source peer projection exact four original fields")
    runner = ast.parse((REPO / "scripts/train_paper_amp.py").read_text())
    projection = [node for node in ast.walk(runner) if isinstance(node, ast.DictComp) and
                  ast.unparse(node) == "{key: ground_sample[key] for key in PEER_FIELDS}"]
    audit.require(len(projection) == 1, "single explicit ground-sample-to-original-peer-field projection")
    resources = {}
    for name, key in (("controller_checkpoint", "resume"), ("diffusion_checkpoint", "diffusion_checkpoint"),
                      ("profile", "config"), ("scene", "shared_scene"), ("manifest", "manifest")):
        if result["arguments"].get(key):
            path = Path(result["arguments"][key])
            if not path.is_absolute():
                path = REPO/path
            resources[name] = fingerprint(path)
    manifest_hash = result["manifest_gate"]["manifest_sha256"]
    audit.require(resources["manifest"]["sha256"] == manifest_hash, "actual manifest full SHA matches runtime manifest gate")
    diffusion_declared = result.get("evaluation_planner_change", {}).get("evaluation_sha256", result["provenance"]["diffusion_sha256"])
    audit.require(resources["diffusion_checkpoint"]["sha256"] == diffusion_declared,
                  "actual deployed diffusion full SHA matches runtime planner choice")
    preflight = {entry["name"]: entry for entry in result["resource_preflight"]["records"]}
    for name, actual in resources.items():
        preflight_name = "shared_scene" if name == "scene" else name
        if preflight_name in preflight:
            record = preflight[preflight_name]
            audit.require(Path(record["path"]).resolve() == Path(actual["path"]) and record["bytes"] == actual["bytes"],
                          f"actual source resource path/bytes agree with runtime preflight: {name}")
    return {"frozen_integration_manifest": fingerprint(source_manifest), "integration_files": source_records,
            "source_binding_limit": "Runtime result has no full per-file code attestation; current files match the frozen pre-launch CPU-reviewed manifest.",
            "resources_full_sha256_recomputed": resources, "controller_provenance": "observed hash of exact resume file; no runtime-published controller checksum exists in this result",
            "runtime_torch": result["runtime_execution"]["torch"],
            "discriminator_input": result["training_config"]["discriminator_input"],
            "direction_mode": result["training_config"]["direction_mode"]}


def markdown(report):
    counts = report["counts"]
    support, proxy, com = report["support_statistics"], report["body_origin_motion_proxy"], report["COM_statistics"]
    stats_proxy = proxy["aggregate_supported_mps"]
    lines = ["# 实际足部支持采集独立 CPU 验证", "", f"输入：`{report['input']}`", "",
             f"结果：**{report['status']}**。`actual_native_capture_validated=true` 指实际采集完整性与数值一致性；历史 probe 的两个 `false` 标记未改。", "",
             "## 时间、主状态与来源", "",
             f"- {counts['controls']} 个 30 Hz 控制帧，{counts['actors']} 人，{counts['duration_s']:g} 秒；全部 {counts['physics_substeps']} 个 120 Hz 提交步计数连续。每控制恰好四个子步，无补零/丢步。",
             "- 末子步身体位置、Pelvis XYZW、episode/reset/termination 与独立 `physical_states.npz` 逐元素完全相同；其原四个障碍/同行人接触字段保持独立。",
             f"- 跌倒 {counts['falls']} 次、时间上限 reset {counts['time_limits']} 次、路线顶点 {counts['route_vertices_reached']}。不从短时采集推断稳定收敛。",
             f"- ground 支持文件 SHA256：`{report['artifacts']['ground_support']['sha256']}`。完整原状态/结果/模型/profile/scene/source 指纹见 JSON。",
             "- 来源边界：结果未包含逐业务源码哈希；当前源码与运行前冻结的 CPU 审查清单一致。控制器指纹是本次重算的指定 resume 文件，未伪称结果发布了该指纹。", "",
             "## 独立重算", "",
             f"- 正质量总和 {com['total_mass_kg']:.6f} kg；link 世界姿态 × link-local COM 偏置重构真实 COM，最大坐标误差 {com['maximum_independent_composition_error_m']:.3g} m。",
             "- 足部 sensor 与完整 common-body 名称/路径一一对应；唯一 floor filter 为 `/World/ground/terrain/mesh`，不包含墙、同行人或摩擦。",
             f"- 从 raw 四子步 normal-force vector sums 独立重算每足/左右侧载荷、严格 >{support['threshold_n']:g} N 接触、双支持/无足支持与全部速度有效 mask。",
             f"- 双支持占 actor-substeps {support['double_support_fraction']:.4%}；无足支持 {support['no_foot_support_fraction']:.4%}。这些是给定阈值的传感定义，不是作者未公开指标。", "",
             "## 接触条件下的足部原点运动代理", "",
             f"支持条件样本 {stats_proxy['count']}；切向速度 mean {stats_proxy['mean']:.4f}、median {stats_proxy['median']:.4f}、P95 {stats_proxy['p95']:.4f}、max {stats_proxy['max']:.4f} m/s。", "",
             "**这不是接触点滑动测量。** 足部滚动/旋转也会产生 link 原点速度；没有摩擦力、接触点速度或 AMASS 真实 COM，不能称作真实 slip，也不能把代理阈值当论文验收。", "",
             "| actor | 双支持 | 无足支持 |", "|---|---:|---:|"]
    for row in support["per_actor"]:
        lines.append(f"| {row['actor']} | {row['double_support_fraction']:.4%} | {row['no_foot_support_fraction']:.4%} |")
    lines.extend(["", "## 可复现 CPU 命令", "", "```bash",
                  f"CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_ground_support_capture.py --input {report['input']} --output /tmp/ground_support_verification.json",
                  "```", "", "本检查仅读取现有文件并写新报告，不启动 Kit/GPU，不修改历史结果、业务源码或采集 schema。"])
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Output report paths must be new; refusing overwrite")
    audit = Audit()
    try:
        report = audit_capture(args.input, audit)
        report["status"] = "passed"
    except Exception as error:
        report = {"status": "failed", "input": str(args.input.resolve()),
                  "actual_native_capture_validated": False, "error": f"{type(error).__name__}: {error}"}
    report["checks"] = audit.checks
    report["audit_script"] = fingerprint(__file__)
    report["command"] = sys.argv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if report["status"] == "passed":
        with args.output.with_suffix(".md").open("x") as stream:
            stream.write(markdown(report))
    print(json.dumps({"status": report["status"], "checks": len(audit.checks), "output": str(args.output),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
