#!/usr/bin/env python3
"""Independent full-population checks of actual TRACE/PACER motion.

Geometric body envelopes and toe origins are proxies, not measured PhysX
contact. A normal simulator exit by itself does not establish motion quality.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def longest_true(values):
    """Longest consecutive interval, including leading/trailing intervals."""
    indices = np.flatnonzero(np.diff(np.r_[False, values, False]))
    return int(np.max(indices[1::2] - indices[::2], initial=0))


def rectangle_distance(xy, center, size):
    """Signed XY distance to an axis-aligned cuboid footprint."""
    q = np.abs(xy - np.asarray(center)[:2]) - np.asarray(size)[:2] / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_bytes = (args.run / "result.json").read_bytes()
    result = json.loads(result_bytes)
    execution = json.loads(args.execution.read_text())
    if not (
        execution.get("actual_returncode") == 0
        and execution.get("actual_returncode_source") == "subprocess.Popen.wait"
        and execution.get("sources_unchanged") is True
        and execution.get("inputs_unchanged") is not False
        and execution.get("result_sha256") == hashlib.sha256(result_bytes).hexdigest()
        and result.get("cleanup_succeeded") is True
        and result.get("status") == "passed"
        and result["actual_controls"] == result["requested_controls"]
    ):
        raise ValueError("Require the full, source-bound physical run and actual exit0")
    scene = None
    if args.scene is None:
        frozen_scene = execution.get("input_snapshot", {}).get("--scene", {}).get("snapshot")
        if frozen_scene:
            args.scene = Path(frozen_scene)
    if args.scene:
        if (
            hashlib.sha256(args.scene.read_bytes()).hexdigest()
            != result["scene_sha256"]
        ):
            raise ValueError("Physical scene identity differs")
        scene = yaml.safe_load(args.scene.read_text())
    elif result.get("scene_sha256"):
        raise ValueError("Actual obstacle geometry required for a shared run")
    physical_path = args.run / "physical.npz"
    physical_sha = hashlib.sha256(physical_path.read_bytes()).hexdigest()
    bound_sha = execution.get("capture_sha256", {}).get("physical.npz")
    if bound_sha is not None and bound_sha != physical_sha:
        raise ValueError("Physical capture differs from the parent's completed receipt")
    with np.load(physical_path, allow_pickle=False) as data:
        samples = data["samples"].astype(np.float64)
        body = data["body_positions"].astype(np.float64)
        rotation = data["root_rotations_xyzw"].astype(np.float64)
        for name in data.files:
            array = data[name]
            if array.dtype.kind in "fci" and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite captured array: {name}")
    n, actors = samples.shape[:2]
    if n != result["actual_controls"] or body.shape[:2] != (n, actors):
        raise ValueError("Body pose/control counts differ")
    if rotation.shape != (n, actors, 4):
        raise ValueError("Actual root quaternion missing")
    quat_error = float(np.max(np.abs(np.linalg.norm(rotation, axis=-1) - 1)))
    if quat_error > 1e-3:
        raise ValueError("Physical root rotation is not a unit quaternion")
    dt = 1 / result["physics"]["control_hz"]
    speed = np.linalg.norm(samples[:, :, 2:4], axis=-1)
    travel = np.linalg.norm(np.diff(samples[:, :, :2], axis=0), axis=-1).sum(0)
    low = speed < 0.1
    config_record=execution.get('input_snapshot',{}).get('--config',{})
    configuration=json.loads(Path(config_record['snapshot']).read_text()) if config_record else {}
    if config_record and hashlib.sha256(Path(config_record['snapshot']).read_bytes()).hexdigest()!=result['config_sha256']:
        raise ValueError('Actual planner configuration identity differs')
    extent=configuration.get('pedestrian_extent_m',[.5,.5,1.7])
    from CrowdSim.humanoid.runtime.trace_pace import validate_pedestrian_extent
    extent=validate_pedestrian_extent(extent)
    radius=np.linalg.norm(extent[:2])/2 # declared ROOT-disk proxy, not all24 link shapes
    pair_min = None
    pair_overlap_frames = 0
    if actors > 1:
        pairs = np.triu_indices(actors, 1)
        distances = np.linalg.norm(
            samples[:, :, None, :2] - samples[:, None, :, :2], axis=-1
        )
        distances = distances[:, pairs[0], pairs[1]]
        pair_min = float(distances.min())
        pair_overlap_frames = int((distances < 2 * radius).any(1).sum())
    map_min = None
    map_overlap_frames = 0
    reached = None
    arrival_masks = None
    if scene:
        native_scene = None
        if scene.get('schema') == 'crowdsim.native_pedestrian_scene.v1':
            import torch
            from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
            from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
            snap = scene.get('probe_script_snapshot')
            if snap and hashlib.sha256(Path(snap['path']).read_bytes()).hexdigest() != snap['sha256']:
                raise ValueError('Native probe source identity changed')
            provider = DynBenchSceneProvider.from_preflight(scene['preflight']['path'],
                expected_sha256=scene['preflight']['sha256'], probe_script_snapshot=None if snap is None else snap['path'])
            provider.admit_native_map(scene['native_map']['path'],expected_sha256=scene['native_map']['sha256'])
            native_scene = NativeGridScene.from_provider(provider,origins_xy=scene['origins_xy'],routes_xy=scene['routes_xy'],
                clearance_m=scene['clearance_m'],arrival_radius_m=scene.get('arrival_radius_m',.75),
                goal_lookahead_m=scene.get('goal_lookahead_m',5.),device='cpu',cyclic_route_lookahead=scene.get('cyclic_route_lookahead',False),start_recovery_radius_m=scene.get('start_recovery_radius_m',0.),start_recovery_clearance_m=scene.get('start_recovery_clearance_m',.4))
            from CrowdSim.humanoid.runtime.native_visual_scene import prepare_native_visual_scene
            prepare_native_visual_scene(scene,result)
            distances = native_scene.signed_distance(torch.tensor(samples[:,:,:2],dtype=torch.float64)).numpy()-radius
        else:
            distances = (
                np.stack(
                    [
                        rectangle_distance(samples[:, :, :2], item["center"], item["size"])
                        for item in scene["cuboids"]
                    ],
                    -1,
                ).min(-1)
                - radius
            )
        map_min = float(distances.min())
        map_overlap_frames = int((distances < 0).any(1).sum())
        # Entry advances one route vertex from PRE-state at each actual control.
        # Initial XY is exactly the declared reset origins, never a teleport
        # counted as a reached vertex. Independently reproduce the total.
        routes = np.asarray(scene["routes_xy"])
        indices = np.zeros(actors, dtype=int)
        reached = np.zeros(actors, dtype=int)
        pre = np.concatenate(
            (np.asarray(scene["origins_xy"])[None], samples[:-1, :, :2]), 0
        )
        arrival_masks = []
        warmup_controls = result.get("initialization_warmup_controls", 0)
        if type(warmup_controls) is not int or not 0 <= warmup_controls <= min(150, n):
            raise ValueError("Invalid declared physical initialization warmup")
        if warmup_controls != scene.get("initialization_warmup_controls", 0):
            raise ValueError("Physical warmup and frozen scene config differ")
        for control, position in enumerate(pre):
            # Runtime does not advance cyclic route indices during stabilization.
            # Preserve the same PRE clock; all warmup physical states still enter
            # speed, fall, support, envelope and contact quality measurements.
            if control < warmup_controls:
                arrival_masks.append(np.zeros(actors, dtype=bool))
                continue
            if native_scene is not None:
                updated, arrived = native_scene._progress(position,indices,np.arange(actors),True)
            else:
                arrived = (
                    np.linalg.norm(position - routes[np.arange(actors), indices], axis=-1)
                    < scene['arrival_radius_m'])
            arrival_masks.append(arrived)
            reached += arrived
            indices = (indices + arrived) % routes.shape[1]
        arrival_masks = np.asarray(arrival_masks)
        if int(reached.sum()) != result["route_vertices_reached"]:
            raise ValueError("Independent route arrivals differ from physical result")
    names = result["physics"]["body_names"]
    toes = body[:, :, [names.index("L_Toe"), names.index("R_Toe")]]
    # Near-ground origin != foot contact; evaluate active frames separately.
    toe_speed = np.linalg.norm(np.diff(toes[..., :2], axis=0), axis=-1) / dt
    near = (toes[1:, ..., 2] < 0.05) & (toes[:-1, ..., 2] < 0.05)
    walking_near = near & (speed[1:, :, None] >= 0.1)
    max_stationary = [longest_true(low[:, actor]) * dt for actor in range(actors)]
    max_no_arrival = None if arrival_masks is None else [
        longest_true(~arrival_masks[:, actor]) * dt for actor in range(actors)
    ]
    planning = json.loads((args.run / "planning.json").read_text())
    brake_count = sum(row["safety_braking"] for row in planning)
    if brake_count != result["safety_braking_plans"]:
        raise ValueError("Recorded braking count differs")
    checks = {
        "full_control_budget_actual_exit0": True,
        "zero_actual_falls": int(samples[:, :, 8].sum()) == 0 and result["falls"] == 0,
        "actual_body_envelopes_no_peer_overlap": pair_overlap_frames == 0,
        "actual_body_envelopes_no_map_overlap": map_overlap_frames == 0,
        "each_actor_reaches_route_vertex_if_duration_ge30s": reached is None
        or n * dt < 30
        or bool((reached >= 1).all()),
        "no_route_vertex_gap_above60s_in_bounded_fixture": scene is None
        or n * dt <= 60
        or max(max_no_arrival) <= 60,
        "no_stationary_interval_above20s_in_cyclic_route": scene is None
        or max(max_stationary) <= 20,
    }
    measured = None
    support_path = args.run / "ground_support.npz"
    if support_path.exists():
        contact_path = args.run / "contact_measurements.npz"
        for capture in (support_path, contact_path):
            expected = execution.get("capture_sha256", {}).get(capture.name)
            if (
                expected is None
                or hashlib.sha256(capture.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(
                    "Measured-contact archive not bound by the real parent"
                )
        with np.load(support_path, allow_pickle=False) as data:
            for name in data.files:
                if data[name].dtype.kind in "fci" and not np.isfinite(data[name]).all():
                    raise ValueError("Nonfinite measured support: " + name)
            before = data["physics_step_before"]
            end = data["physics_step_end"]
            if not (
                data["foot_floor_normal_force_w"].shape[:4] == (n, 4, actors, 4)
                and np.array_equal(data["control_step"], np.arange(n))
                and np.array_equal(data["control_step_after"], np.arange(1, n + 1))
                and np.array_equal(
                    data["physics_step"], before[:, None] + np.arange(1, 5)
                )
                and np.array_equal(end, before + 4)
                and np.array_equal(before[1:], end[:-1])
                and data["substep_valid_count"].shape == (n,)
                and (data["substep_valid_count"] == 4).all()
                and data["valid"].all()
                and not data["reset_mask"][1:].any()
            ):
                raise ValueError("Incomplete/inconsistent actual contact substep clock")
            pelvis = names.index("Pelvis")
            if not (
                list(data["common_body_names"]) == names
                and np.array_equal(body, data["body_link_pos_w"][:, -1])
                and np.array_equal(rotation, data["body_link_quat_w"][:, -1, :, pelvis])
                and (data["actor_substep_valid_count"] == 4).all()
                and not data["terminated"].any()
                and not data["truncated"].any()
                and (data["episode_ids"] == data["episode_ids"][0]).all()
            ):
                raise ValueError("Body/rotation/episodes disagree with measured final physics substep")
            contact = data["foot_contact_substeps"]
            mass = data["body_masses_kg"].astype(float)
            forces = data["foot_floor_normal_force_w"].astype(float)
            velocities = data["foot_body_origin_vel_w"].astype(float)
            if (
                contact.dtype != np.bool_
                or contact.shape != (n, 4, actors, 4)
                or mass.shape != (n, 4, actors, 24)
                or (mass <= 0).any()
            ):
                raise ValueError("Measured foot or mass channels disagree")
            normal_load = np.linalg.norm(forces.sum(-2), axis=-1).sum(-1)
            load_ratio = normal_load / (mass.sum(-1) * 9.81)
            tangential_speed = np.linalg.norm(velocities[..., :2], axis=-1)
            measured = {
                "completed_controls": n,
                "completed_physics_substeps": n * 4,
                "four_substeps_per_control_verified": True,
                "all24_body_positions_and_root_rotation_equal_native_final_substep": True,
                "floor_normal_load_mean_n_per_actor": normal_load.mean((0, 1)).tolist(),
                "floor_load_over_body_weight_mean_per_actor": load_ratio.mean(
                    (0, 1)
                ).tolist(),
                "supported_substep_fraction_per_actor": contact.any(-1)
                .mean((0, 1))
                .tolist(),
                "measured_supported_foot_body_origin_speed_proxy_m_s": float(
                    tangential_speed[contact].mean()
                )
                if contact.any()
                else None,
                "contact_force_threshold_n": 1.0,
                "friction_force_measured": False,
                "contact_patch_slip_measured": False,
            }
        with np.load(contact_path, allow_pickle=False) as data:
            for name, shape in (
                ("peer_contact", (n, actors, actors)),
                ("static_contact", (n, actors)),
            ):
                if data[name].dtype != np.bool_ or data[name].shape != shape:
                    raise ValueError("Actual contact flags do not align with controls")
            measured.update(
                peer_contact_actor_controls=int(data["peer_contact"].any(-1).sum()),
                static_contact_actor_controls=int(data["static_contact"].sum()),
                peer_contact_max_force_n=float(data["peer_contact_max_force_n"].max()),
                static_contact_max_force_n=float(
                    data["static_contact_max_force_n"].max()
                ),
            )
        checks["zero_measured_peer_contacts_above1N"] = (
            measured["peer_contact_actor_controls"] == 0
        )
        checks["zero_measured_static_contacts_above1N"] = (
            measured["static_contact_actor_controls"] == 0
        )
        checks["measured_floor_support_in_majority_of_substeps"] = (
            min(measured["supported_substep_fraction_per_actor"]) > 0.5
        )
    if scene is not None and scene.get("native_map") is not None:
        proof=result.get("native_static_contact_report")
        if (not isinstance(proof,dict) or proof["reads"]!=n*4 or proof.get("body_identity_verified_each_tick") is not True
                or proof["last_physics_step"]-proof["first_physics_step"]+1!=n*4
                or proof["floor_positive_substeps"]<n*2 or proof["maximum_floor_force_n"]<=1
                or not proof["floor_witnesses"]):
            raise ValueError("Every native scene needs complete independently witnessed all-body residual force reports")
        raw=proof["raw_artifact"]
        if hashlib.sha256(Path(raw["path"]).read_bytes()).hexdigest()!=raw["sha256"]:raise ValueError("Native residual raw capture changed")
        with np.load(raw["path"],allow_pickle=False) as arrays:
            if not np.array_equal(arrays["physics_step"],np.arange(proof["first_physics_step"],proof["last_physics_step"]+1)):
                raise ValueError("Native raw force substep clock differs")
            vectors=[arrays[k].astype(float) for k in ("net","floor","peer","own")]
            if any(v.shape!=(n*4,actors,24,3) or not np.isfinite(v).all() for v in vectors):raise ValueError("Incomplete all-body raw residual forces")
            maximum=float(np.linalg.norm(vectors[0]-vectors[1]-vectors[2]-vectors[3],axis=-1).max())
            if not np.isclose(maximum,proof["maximum_static_force_n"],atol=1e-3,rtol=1e-5):raise ValueError("Native force residual differs from raw vectors")
        measured["native_nonfloor_nonpeer_normal_force_residual"]=proof
        checks["zero_native_nonfloor_nonpeer_force_residual_above1N"]=proof["maximum_static_force_n"]<=1
    report = {
        "schema": "crowdsim.trace_pace_reliability_evaluation.v1",
        "geometric_envelope_scope": "root circumscribed disk proxy; separate actual24-body contact residual mandatory in every native scene",
        "declared_root_extent_m": extent,
        "native_source_scene": None if scene is None else scene.get("scene"),
        "quality_status": "passed_local_proxy_gates"
        if all(checks.values())
        else "failed_local_proxy_gates",
        "paper_result": False,
        "actual_force_contacts_measured": measured is not None,
        "measured_contacts": measured,
        "actual_control_count": n,
        "duration_s": n * dt,
        "actors": actors,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "physical_sha256": physical_sha,
        "capture_parent_hash_available": bound_sha is not None,
        "evaluation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "all_frames_included": True,
        "checks": checks,
        "path_length_m": travel.tolist(),
        "low_speed_fraction_below0p1_m_s": low.mean(0).tolist(),
        "maximum_continuous_low_speed_s": max_stationary,
        "maximum_time_without_route_vertex_s": max_no_arrival,
        "maximum_actual_speed_m_s": speed.max(0).tolist(),
        "actual_speed_p95_m_s": np.percentile(speed, 95, axis=0).tolist(),
        "planning_speed_cap_is_not_a_physical_velocity_guarantee": True,
        "minimum_actual_root_pair_distance_m": pair_min,
        "peer_body_envelope_overlap_control_frames": pair_overlap_frames,
        "minimum_actual_body_envelope_map_clearance_m": map_min,
        "map_body_envelope_overlap_control_frames": map_overlap_frames,
        "route_vertices_reached_per_actor": None
        if reached is None
        else reached.tolist(),
        "safety_braking_plans": brake_count,
        "total_plans": len(planning),
        "toe_origin_below_ground_frames": int((toes[..., 2] < 0).any(2).sum()),
        "walking_near_ground_toe_speed_proxy_m_s": None
        if not walking_near.any()
        else float(toe_speed[walking_near].mean()),
        "root_quaternion_max_norm_error": quat_error,
        "limitations": [
            "Body envelopes approximate geometry; measured normal forces cover declared bodies and obstacles"
            if measured is not None else "Body envelopes approximate geometry; force contacts were not recorded",
            "Near-ground toe motion includes articulation and is not measured contact slip",
            "Local gates and fixture are not the eight-scene author benchmark",
            "A passing proxy report does not alone establish visually natural gait",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "quality_status": report["quality_status"],
                "controls": n,
                "checks": checks,
            }
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
