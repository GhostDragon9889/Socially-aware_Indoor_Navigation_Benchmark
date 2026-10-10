"""Read retained native pedestrian artifacts on CPU; never launch SDK/physics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def arrays(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def array_records(values):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()} for name, value in values.items()}


def audit(run, source_manifest):
    run = Path(run).resolve()
    result_path = run / "result.json"
    result_record = record(result_path)
    result = json.loads(result_path.read_text())
    source_manifest = Path(source_manifest).resolve()
    epoch = json.loads(source_manifest.read_text())
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    archive_path = ROOT / epoch["archive"]["path"]
    archive_record = record(archive_path)
    check("source archive full bytes and SHA", all(archive_record[key] == epoch["archive"][key] for key in ("bytes", "sha256")))
    check("source manifest declared count", len(epoch["files"]) == epoch["source_count"])
    current_changed = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for row in epoch["files"]:
            member = members[row["path"]]
            check("archive regular source " + row["path"], member.isfile())
            data = archive.extractfile(member).read()
            check("archive exact source " + row["path"], len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
            current = record(ROOT / row["path"])
            if any(current[key] != row[key] for key in ("bytes", "sha256")):
                current_changed.append(current)

    physical = arrays(run / "physical_states.npz")
    ground = arrays(run / "ground_support.npz")
    plans = arrays(run / "reference_plans.npz")
    for prefix, values in (("physical", physical), ("ground", ground), ("plans", plans)):
        for name, value in values.items():
            check(prefix + " no object dtype " + name, value.dtype.kind != "O")
            if value.dtype.kind == "f":
                check(prefix + " finite " + name, np.isfinite(value).all())
    t, b, xyz = physical["root_pos"].shape
    check("complete actual requested controls", xyz == 3 and t == result["arguments"]["evaluate_steps"] == result["actual_control_steps"])
    check("actual matched actor count", b == result["arguments"]["num_envs"])
    check("all physical rows retained", all(len(value) == t for value in physical.values()))
    check("all ground controls retained", len(ground["control_step"]) == t)
    check("control counter starts0 and is consecutive", np.array_equal(ground["control_step"], np.arange(t)))
    check("postcontrol counter equals before+1", np.array_equal(ground["control_step_after"], ground["control_step"] + 1))
    check("four substeps valid every control", np.all(ground["substep_valid_count"] == 4) and ground["valid"].all() and np.all(ground["actor_substep_valid_count"] == 4))
    check("every substep counter consecutive", np.array_equal(ground["physics_step"], ground["physics_step_before"][:, None] + np.arange(1, 5)))
    check("every end counter actual fourth substep", np.array_equal(ground["physics_step_end"], ground["physics_step_before"] + 4))
    check("no missing or extra submitted steps between controls", np.array_equal(ground["physics_step_before"][1:], ground["physics_step_end"][:-1]))
    for name in ("episode_ids", "reset_mask", "terminated", "truncated"):
        check("ground/physical exact " + name, np.array_equal(ground[name], physical[name]) and ground[name].dtype == physical[name].dtype)
    episode_change = np.diff(physical["episode_ids"], axis=0)
    check("episode increment0or1", np.isin(episode_change, (0, 1)).all())
    check("reset masks match episode increments", np.array_equal(episode_change == 1, physical["reset_mask"][1:]))
    check("next reset exactly follows termination/truncation", np.array_equal(physical["reset_mask"][1:], (physical["terminated"] | physical["truncated"])[:-1]))
    check("ground common body order exact", ground["common_body_names"].tolist() == result["physics"]["body_names"])
    check("exact four semantic feet", ground["foot_names"].tolist() == ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"])
    check("ground final substep body origins equal physical common bodies", np.array_equal(ground["body_link_pos_w"][:, -1], physical["body_pos"]))

    pos, quat, offset = ground["body_link_pos_w"], ground["body_link_quat_w"], ground["local_com_pose_b"][..., :3]
    cross = 2 * np.cross(quat[..., :3], offset)
    composed = pos + offset + quat[..., 3:] * cross + np.cross(quat[..., :3], cross)
    com_error = float(np.max(np.abs(composed - ground["body_com_pos_w"])))
    contract = result["ground_support_measurement"]["contract"]
    check("native link rotations unit XYZW", np.allclose(np.linalg.norm(quat, axis=-1), 1., atol=1e-3, rtol=0))
    check("same instant trueCOM independently composed", com_error <= contract["com_composition_tolerance_m"])
    masses = ground["body_masses_kg"]
    check("strictly positive body masses", (masses > 0).all())
    weighted = (ground["body_com_pos_w"] * masses[..., None]).sum(-2) / masses.sum(-1, keepdims=True)
    weighted_error = float(np.abs(weighted - ground["mass_weighted_world_com"]).max())
    check("mass weighted trueCOM recomputed", weighted_error < 1e-5)
    check("common masses match runtime provenance", np.array_equal(masses[0, 0, 0], np.asarray(result["physics"]["live_body_masses_kg"], dtype=masses.dtype)))

    threshold = contract["floor_measurement"]["threshold_n"]
    force = ground["foot_floor_normal_force_w"]
    floor_vector = force.sum(-2)
    load = np.linalg.norm(floor_vector, axis=-1)
    foot_contact = load > threshold
    side_load = load.reshape(t, 4, b, 2, 2).sum(-1)
    side_contact = foot_contact.reshape(t, 4, b, 2, 2).any(-1)
    check("threshold actual1N", threshold == 1.)
    check("true foot floor contacts recomputed", np.array_equal(foot_contact, ground["foot_contact_substeps"]))
    check("true side support recomputed", np.array_equal(side_contact, ground["side_contact_substeps"]))
    check("floor side load means recomputed", np.allclose(side_load.mean(1), ground["side_normal_load_mean_n"], atol=1e-3, rtol=1e-6))
    check("double support fraction recomputed", np.array_equal(side_contact.all(-1).mean(1).astype(np.float32), ground["double_support_fraction"]))
    check("no support fraction recomputed", np.array_equal((~side_contact.any(-1)).mean(1).astype(np.float32), ground["no_support_fraction"]))
    provider = result["native_scene_provider"]
    selected_floor = provider["floor_paths"]
    check("exact floor-only filters", selected_floor == contract["floor_measurement"]["floor_collider_paths"] == ["/__PaperMapProbeReferenceGround"])
    check("each native floor filter row exact", all(row == selected_floor for row in contract["floor_measurement"]["native_filter_paths"]))
    static = set(result["physical_contact_measurement"]["static_collider_paths"])
    check("static filters exclude floor and all disabledflat originals", not static.intersection(selected_floor + provider["disabled_original_flat_ground_paths"]))
    check("static exact declared count", len(static) == provider["static_collider_count"] == 745)
    check("actual current-manager shared physics view", contract["physics_view_ownership"]["active_manager_getter_same_view"] is True and contract["physics_view_ownership"]["new_simulation_view_created"] is False)
    check("bound exact dynamic robot roots", provider["exact_dynamic_robot_roots"] == [f"/World/envs/env_{actor}/Robot" for actor in range(b)])
    check("live query binding and precreate hook", provider["live_query_bound"] is True and provider["pre_create_hook_applied"] is True)
    check("actual strict deterministicTorch requested and executed", result["runtime_execution"]["numerics"]["torch_deterministic_algorithms"] is True and result["runtime_execution"]["numerics"]["torch_deterministic_warn_only"] is False)

    old_physics = result["evaluation_actor"]["full_training_provenance"]["physics"]
    identical_rig_fields = ("engine", "physics_hz", "control_hz", "rig", "body_names", "dof_names", "dof_limits_lower", "dof_limits_upper", "joint_control", "live_native_body_names", "live_body_masses_kg")
    for name in identical_rig_fields:
        check("loaded actor/current physical rig contract " + name, old_physics[name] == result["physics"][name])
    for path, sha in ((result["arguments"]["resume"], result["evaluation_actor"]["checkpoint_sha256"]),
                     (result["arguments"]["diffusion_checkpoint"], result["provenance"]["diffusion_sha256"])):
        check("actual weight SHA " + str(path), record(path)["sha256"] == sha)

    height = physical["root_pos"][..., 2] - provider["ground_height_m"]
    q = physical["root_rot"]
    up_z = 1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2)
    fall_def = result["physics"]["fall_definition"]
    computed_fall = (height < fall_def["pelvis_ground_height_less_than_m"]) | (up_z < fall_def["root_up_z_less_than"])
    check("raw fall definition equals termination", np.array_equal(computed_fall, physical["terminated"]))
    check("all fall count equals result", int(computed_fall.sum()) == result["fall_resets"])
    check("all real reached counts equal result", physical["route_vertex_reached"].sum(0).tolist() == result["route_vertices_reached"])
    check("selected sample actual valid0 for S1", np.all(plans["selected_sample"] == 0))
    check("plan row shape alignment", all(len(value) == len(plans["actor_ids"]) for value in plans.values()))
    check("plan actor/control IDs legal", ((plans["actor_ids"] >= 0) & (plans["actor_ids"] < b) & (plans["control_step"] >= 0) & (plans["control_step"] < t)).all())

    events = []
    for frame, actor in np.argwhere(computed_fall):
        episode = physical["episode_ids"][frame, actor]
        episode_rows = np.flatnonzero(physical["episode_ids"][:frame + 1, actor] == episode)
        contact_frames = episode_rows[physical["static_contact"][episode_rows, actor]]
        events.append({"frame0": int(frame), "control1": int(frame + 1), "time_s": float((frame + 1) / 30),
            "actor0": int(actor), "episode_id": int(episode), "episode_length_controls": len(episode_rows),
            "root_xyz_m": physical["root_pos"][frame, actor].tolist(), "pelvis_above_declared_floor_m": float(height[frame, actor]),
            "root_up_z": float(up_z[frame, actor]), "height_fall": bool(height[frame, actor] < .45), "tilt_fall": bool(up_z[frame, actor] < .1),
            "static_contact_control_count_this_episode": len(contact_frames), "first_static_contact_frame0": int(contact_frames[0]) if len(contact_frames) else None,
            "last_static_contact_frame0": int(contact_frames[-1]) if len(contact_frames) else None,
            "static_contact_current": bool(physical["static_contact"][frame, actor]),
            "static_contact_max_n_this_episode": float(physical["static_contact_max_force_n"][episode_rows, actor].max()),
            "foot_floor_load_peak_n": load[frame, :, actor].max(0).tolist(),
            "reference_target_xy_m": physical["target_xy"][frame, actor].tolist(),
            "raw_diffusion_target_xy_m": physical["raw_target_xy"][frame, actor].tolist()})
    check("fall episode lengths equal result", [event["episode_length_controls"] for event in events] == result["completed_episode_lengths"])
    alive_distance = np.linalg.norm(np.diff(physical["root_pos"][..., :2], axis=0), axis=-1)
    alive_distance[physical["reset_mask"][1:]] = 0
    contact_pairs = physical["peer_contact"][:, np.triu_indices(b, 1)[0], np.triu_indices(b, 1)[1]]
    artifacts = {name: record(run / name) for name in ("result.json", "physical_states.npz", "ground_support.npz", "reference_plans.npz", "deployment_entry.json")}
    check("actual ground archive exact declared SHA", artifacts["ground_support.npz"]["sha256"] == result["ground_support_measurement"]["capture"]["sha256"])
    check("result bytes unchanged during independent read", result_record == record(result_path))
    return {"schema": "native_pedestrian_world_independent_audit.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_only_reader": True, "GPU_used_by_reader": False, "SDK_started_by_reader": False,
        "actual_parent_GPU_run_artifacts": True, "status": ("raw_sensor_source_contracts_passed_physical_stability_failed" if events
            else "raw_sensor_source_contracts_passed_bounded_zero_fall_observed"),
        "checks_passed": len(checks), "checks": checks,
        "run": str(run), "source_epoch": {"manifest": record(source_manifest), "archive": archive_record,
            "source_count": epoch["source_count"], "all_archived_source_bytes_verified": True, "current_changed_since_launch": current_changed},
        "artifacts": artifacts, "array_records": {"physical": array_records(physical), "ground": array_records(ground), "plans": array_records(plans)},
        "counts": {"controls": t, "actors": b, "duration_s": t / 30, "actor_controls": t*b,
            "shared_physics_substeps": t*4, "actor_physics_substeps": t*4*b, "first_counter": int(ground["physics_step_before"][0]),
            "last_counter": int(ground["physics_step_end"][-1]), "falls": len(events), "time_limit_resets": result["time_limit_resets"],
            "route_vertices": physical["route_vertex_reached"].sum(0).tolist(),
            "complete_cycles": [0]*b if not physical["route_vertex_reached"].any() else None,
            "static_actor_controls": int(physical["static_contact"].sum()), "static_denominator_actor_controls": t*b,
            "static_actor_controls_per_actor": physical["static_contact"].sum(0).tolist(),
            "peer_pair_controls": int(contact_pairs.sum()), "peer_denominator_unique_pair_controls": t*(b*(b-1)//2)},
        "support": {"selected_floor_paths": selected_floor, "threshold_n": threshold,
            "side_supported_fraction_per_actor": side_contact.mean((0, 1)).tolist(),
            "double_support_fraction_per_actor": side_contact.all(-1).mean((0, 1)).tolist(),
            "no_support_fraction_per_actor": (~side_contact.any(-1)).mean((0, 1)).tolist(),
            "max_total_foot_load_n_per_actor": side_load.sum(-1).max((0, 1)).tolist(),
            "COM_recomposition_max_error_m": com_error, "mass_weighted_COM_max_error_m": weighted_error,
            "COM_tolerance_m": contract["com_composition_tolerance_m"], "slip_metric": "body-origin tangential speed proxy, not contact-point slip"},
        "physical": {"first_postcontrol_roots_xyz_m": physical["root_pos"][0].tolist(), "last_roots_xyz_m": physical["root_pos"][-1].tolist(),
            "minimum_pelvis_height_above_declared_floor_m": height.min(0).tolist(), "within_episode_distance_m": alive_distance.sum(0).tolist(),
            "mean_target_XY_error_m": np.linalg.norm(physical["root_pos"][..., :2]-physical["target_xy"], axis=-1).mean(0).tolist(),
            "mean_reference_endpoint_goal_error_m": physical["plan_endpoint_goal_error"].mean(0).tolist(),
            "max_static_force_n_per_actor": physical["static_contact_max_force_n"].max(0).tolist(), "fall_events": events},
        "scene_contract": {"configured_origins_xy_m": result["physics"]["scene_geometry"]["origins_xy"],
            "negative_world_coordinates_preserved": True, "reference_floor_bounds": provider["floor_bounds_xy_m"],
            "source_USD": provider["source_USD"], "declared_disabled_flat_ground_paths": provider["disabled_original_flat_ground_paths"],
            "terrain_material_reported": result["physics"]["terrain_material"], "live_body_mass_total_kg": result["physics"]["live_total_mass_kg"],
            "same_loaded_actor_rig_contract_fields": list(identical_rig_fields)},
        "evidence_limits": {"fresh_runtime_floor_inventory_and_all_hit_ownership_observed": None,
            "measured_single_App_or_context_construction_count": None, "per_query_clock_rows_retained": None,
            "native_engine_time_observed": False, "native_engine_time_before_after_delta": None,
            "scope": "Actual floor-filter forces and all raw completed-step/COM channels are observed. Fresh stage inventory, singleton construction counters and query event rows were not saved; those remain source-reviewed and old native-resource-probe-bound, not new measured PASS.",
            "causality": "Actor2 had early native static contacts, then delayed first height fall without contemporaneous/prior16control static contact; subsequent reset episodes collided. This supports spatial/controller/command-domain investigation but does not establish a collision, feature, rig, or diffusion cause.",
            "PhysX_Warp_bitwise_guarantee": False, "five_second_run_is_200second_quality_proof": False,
            "declared_result_status_passed_means": "driver completed requested steps; actual raw falls/vertices above are independently reported, not navigation success"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Preserve existing independent audit: " + str(destination))
    result = audit(args.run, args.source_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"],
        "counts": result["counts"], "COM_max_error_m": result["support"]["COM_recomposition_max_error_m"]}))


if __name__ == "__main__":
    main()
