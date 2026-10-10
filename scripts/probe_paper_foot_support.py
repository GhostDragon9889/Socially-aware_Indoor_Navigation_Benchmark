#!/usr/bin/env python3
"""Finite PhysX feet-floor QA: fixed PD stance, airborne feet, airborne wall.

No policy or diffusion is used. One native reset starts each phase; there are
no pose writes or extra physics steps inside a phase. This is sensor QA, not
a motion-quality or paper locomotion-success benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOOT_NAMES = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--anchor-frame", type=int, default=9)
    parser.add_argument("--pose-anchor", choices=("neutral", "gravity_balanced_neutral", "walking_frame"), default="neutral")
    parser.add_argument("--standing-placement", choices=("fixed_height", "collision_grounded"), default="fixed_height")
    parser.add_argument("--initial-sole-clearance-m", type=float, default=.001)
    parser.add_argument("--standing-root-height-m", type=float, default=1.)
    parser.add_argument("--standing-steps", type=int, default=60)
    parser.add_argument("--stable-window-steps", type=int, default=20)
    parser.add_argument("--stable-support-fraction", type=float, default=.9)
    parser.add_argument("--threshold-n", type=float, default=1.)
    parser.add_argument("--zero-force-tolerance-n", type=float, default=1e-5)
    parser.add_argument("--com-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--wall-penetration-m", type=float, default=.02)
    parser.add_argument("--verify-package-hash", action="store_true")
    parser.add_argument("--deterministic-evaluation", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("standing_steps", "stable_window_steps", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.standing_steps < 2 * args.stable_window_steps:
        parser.error("Standing steps must allow at least one stable-window duration of settling")
    for name in ("standing_root_height_m", "threshold_n", "zero_force_tolerance_n", "com_tolerance_m", "wall_penetration_m"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not .5 <= args.standing_root_height_m <= 1.5:
        parser.error("Standing root height must lie in [0.5,1.5] m")
    if not math.isfinite(args.initial_sole_clearance_m) or not 0 <= args.initial_sole_clearance_m <= .005:
        parser.error("Initial sole clearance must lie in [0,0.005] m")
    if args.standing_placement == "collision_grounded" and args.pose_anchor == "walking_frame":
        parser.error("Collision-grounded placement currently requires a neutral initial pose")
    if args.pose_anchor == "gravity_balanced_neutral" and args.standing_placement != "collision_grounded":
        parser.error("Gravity-balanced neutral stance requires explicit collision-grounded placement")
    if not 0 < args.wall_penetration_m <= .05:
        parser.error("Wall initial penetration must lie in (0,0.05] m")
    if not math.isfinite(args.stable_support_fraction) or not 0 < args.stable_support_fraction <= 1:
        parser.error("Stable support fraction must lie in (0,1]")
    if args.zero_force_tolerance_n >= args.threshold_n:
        parser.error("Zero-force tolerance must be smaller than contact threshold")
    if min(args.motion_id, args.anchor_frame, args.seed) < 0 or args.seed >= 2**32:
        parser.error("Motion/frame/seed must be nonnegative; seed <2**32")
    from CrowdSim.humanoid.application.configuration import validate_deterministic_evaluation_environment

    try:
        validate_deterministic_evaluation_environment(args.deterministic_evaluation)
    except ValueError as error:
        parser.error(str(error))
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def probe_scene_description():
    """Wall exists before native views initialize; stance is six metres away."""
    return {
        "schema": "navisaaclab.reconstruction_scene.v1",
        "name": "standalone_foot_support_qa", "bounds_xy": [0., 0., 24., 24.],
        "origins_xy": [[6., 6.]], "routes_xy": [[[4., 4.], [4., 8.]]],
        "cuboids": [{"center": [12., 6., 3.4], "size": [.2, 4., 3.]}],
    }


def preflight(args):
    """Metadata/header/XML checks only; no Torch deserialization or SDK."""
    from CrowdSim.humanoid.application.resources import file_sha256, inspect_resource, resolve_local_path

    document = json.loads(args.manifest.read_text())
    if document.get("schema") != "navisaaclab.amass_reconstruction.v1":
        raise ValueError("Unknown AMASS manifest schema")
    if set(document.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Manifest must declare all three explicit splits")
    entry = document["splits"][args.split]
    clips = entry.get("clips", [])
    ids = [clip.get("motion_id") for clip in clips]
    if any(type(value) is not int or value < 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest requires unique integer motion IDs")
    selected = [clip for clip in clips if clip["motion_id"] == args.motion_id]
    if len(selected) != 1:
        raise ValueError("Requested pose-anchor motion ID is absent")
    record = selected[0]
    if type(record.get("frames")) is not int or not args.anchor_frame < record["frames"]:
        raise ValueError("Pose-anchor frame must be inside its declared clip")
    dt = record.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not math.isclose(dt, 1 / 30, abs_tol=1e-7):
        raise ValueError("This bounded probe expects an explicitly resampled 30Hz pose anchor")
    if not isinstance(record.get("source_path"), str) or not record["source_path"]:
        raise ValueError("Pose-anchor source identity is missing")
    package_path = resolve_local_path(entry["path"], relative_to=args.manifest.parent)
    package = inspect_resource("pose-anchor AMASS package", package_path, kind="torch",
                               expected_bytes=entry["bytes"], declared_sha256=entry["sha256"],
                               recompute_sha256=args.verify_package_hash)
    asset = ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    inspect_resource("SMPL MJCF", asset)
    xml = ET.parse(asset).getroot()
    body_names = [body.get("name") for body in xml.findall(".//body")]
    if any(body_names.count(name) != 1 for name in FOOT_NAMES):
        raise ValueError("SMPL asset lacks unique semantic ankle/toe bodies")
    for element in xml.iter():
        for name in ("pos", "quat", "size", "fromto"):
            if name in element.attrib:
                try:
                    values = [float(value) for value in element.attrib[name].split()]
                except ValueError as error:
                    raise ValueError(f"Invalid numeric MJCF {name}") from error
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite or empty MJCF {name}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Use a new empty output directory; do not overwrite prior probes")
    return {"status": "SDK-free preflight passed", "sdk_started": False, "torch_imported": False,
            "manifest": str(args.manifest), "manifest_sha256": file_sha256(args.manifest),
            "package": asdict(package), "pose_anchor": record,
            "asset": str(asset), "asset_sha256": file_sha256(asset),
            "scene": probe_scene_description(), "author_gait_success_test": False}


def _native_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp

    return wp.to_torch(value)


def resolve_native_contact_factory(simulation_context):
    """Use the existing registered Lab physics view, never a legacy singleton.

    Pinned Lab3 owns its Warp view in PhysxManager. The unrelated Isaac Sim
    SimulationManager can return None even when the articulation is live.
    No private view attribute or freshly created SimulationView is a fallback.
    """
    manager = getattr(simulation_context, "physics_manager", None)
    getter = getattr(manager, "get_physics_sim_view", None)
    if not callable(getter):
        raise RuntimeError("Actual registered Lab physics manager has no public simulation-view getter")
    view = getattr(simulation_context, "physics_sim_view", None)
    if view is None:
        raise RuntimeError("Actual Lab SimulationContext.physics_sim_view is unavailable after world initialization")
    if getter() is not view:
        raise RuntimeError("Public Lab simulation view differs from its active physics manager view")
    if not callable(getattr(view, "create_rigid_contact_view", None)):
        raise RuntimeError("Actual Lab physics view does not support native rigid-contact creation")
    manager_type = manager if isinstance(manager, type) else type(manager)
    return view, {
        "source": "world.simulator._sim.physics_sim_view",
        "owner": f"{manager_type.__module__}.{manager_type.__qualname__}",
        "active_manager_getter_same_view": True,
        "factory": "existing SimulationView.create_rigid_contact_view(pattern, filter_patterns, max_contact_data_count)",
        "legacy_isaacsim_SimulationManager_used": False,
        "new_simulation_view_created": False,
        "private_view_lookup_used": False,
    }


class ProbeSubstepAdapter:
    """Use the existing PaperPowerSimulator hook without adding a physics step.

    The public Lab counter is a monotonic submitted-step counter. It is not an
    unavailable PhysX native simulation-time getter or a bitwise guarantee.
    """

    def __init__(self, world, floor_monitor, wall_view, wall_path, *, com_tolerance_m):
        from CrowdSim.humanoid.foot_support import body_name_permutation

        self.world, self.floor, self.wall_view = world, floor_monitor, wall_view
        self.robot = world.simulator._robot
        self.counter_getter = getattr(world.simulator._sim, "get_physics_step_count", None)
        if not callable(self.counter_getter):
            raise RuntimeError("Known public Lab physics-step counter is unavailable; no synthetic clock fallback")
        native_names = tuple(self.robot.root_view.shared_metatype.link_names)
        if tuple(self.robot.body_names) != native_names:
            raise RuntimeError("Public robot data order differs from native root view; an explicit verified mapping is required")
        self.native_names, self.common_names = native_names, tuple(world.mapping.body_names)
        self.common_indices = body_name_permutation(native_names, self.common_names)
        self.foot_indices = tuple(self.common_names.index(name) for name in FOOT_NAMES)
        if world.num_envs != 1 or len(native_names) != 24:
            raise ValueError("Standalone QA requires one actual24-body SMPL articulation")
        sensors = tuple(getattr(wall_view, "sensor_paths", ()))
        if (getattr(wall_view, "sensor_count", None) != 4 or getattr(wall_view, "filter_count", None) != 1
                or len(sensors) != 4 or len(set(sensors)) != 4 or set(sensors) != set(floor_monitor.sources)):
            raise RuntimeError("Wall witness sensor metadata does not match the four exact feet")
        filters = list(getattr(wall_view, "filter_paths", ()))
        flattened = [path for row in filters for path in row] if filters and all(not isinstance(row, str) for row in filters) else filters
        if len(flattened) not in (1, 4) or set(flattened) != {wall_path}:
            raise RuntimeError("Wall witness must have one exact wall collider and no floor partner")
        self.wall_indices = tuple(sensors.index(path) for path in floor_monitor.sources)
        self.com_tolerance_m = com_tolerance_m
        self.control_step = -1
        self.pending = []
        self.published = True
        self.last_partial_error = None
        self.floor.to_torch = self._record_native_floor
        self.manifest = {"counter_source": "fixed Lab SimulationContext.get_physics_step_count()",
                         "counter_semantics": "monotonic submitted physics-step counter, not PhysX engine time",
                         "native_engine_time_observed": False, "extra_physics_steps": 0,
                         "sampling_hook": "existing PaperPowerSimulator.pedestrian_contacts after each physics step + scene.update",
                         "native_body_names": native_names, "common_body_names": self.common_names,
                         "wall_sensor_paths": sensors, "wall_filter_path": wall_path,
                         "link_velocity_source": "robot.data.body_link_lin_vel_w (actor origin, not COM alias)",
                         "world_COM_source": "robot.data.body_com_pos_w", "local_COM_source": "root_view.get_coms()[..., :3]",
                         "mass_source": "root_view.get_masses() per native body, reordered to common",
                         "quaternion_order": "xyzw", "com_tolerance_m": com_tolerance_m}

    def counter(self):
        value = self.counter_getter()
        if type(value) is not int or value < 0:
            raise RuntimeError("Invalid actual Lab physics-step counter")
        return value

    def episodes(self):
        return tuple(int(value) for value in self.world.episode_ids.detach().cpu().tolist())

    def reset_phase(self):
        if not self.published and self.pending:
            raise RuntimeError("Archive interrupted phase before discarding any sampled substeps")
        self.floor.reset()
        self.pending = []
        self.published = True

    def _record_native_floor(self, value):
        tensor = _native_tensor(value)
        self.pending[-1]["native_floor_normal_force_w"] = tensor.detach().clone()
        return tensor

    def begin_control_step(self):
        if not self.published:
            raise RuntimeError("Previous sampled control was not archived/published")
        self.control_step += 1
        self.pending = []
        self.published = False
        self.before_counter = self.counter()
        self.floor.begin_control_step(episode_ids=self.episodes(), control_step=self.control_step,
                                      physics_step_before=self.before_counter)

    def sample(self, dt):
        import torch
        from CrowdSim.humanoid.foot_support import body_link_world_com, mass_weighted_world_com

        record = {"physics_step_before_read": self.counter(), "control_step": self.control_step,
                  "episode_id": self.episodes()[0], "physics_dt_s": dt}
        self.pending.append(record)
        try:
            indices = list(self.common_indices)
            for key, attribute in (("body_link_pos_w", "body_link_pos_w"), ("body_link_quat_xyzw", "body_link_quat_w"),
                                   ("body_link_vel_w", "body_link_lin_vel_w"), ("body_link_ang_vel_w", "body_link_ang_vel_w"),
                                   ("body_com_pos_w", "body_com_pos_w")):
                tensor = _native_tensor(getattr(self.robot.data, attribute)).detach().clone()
                width = 4 if key == "body_link_quat_xyzw" else 3
                if tensor.shape != (1, 24, width) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                    record["invalid_native_" + attribute] = tensor
                    raise FloatingPointError(f"Invalid actual native {attribute}")
                record[key] = tensor[:, indices]
            coms = _native_tensor(self.robot.root_view.get_coms()).detach().clone()
            masses = _native_tensor(self.robot.root_view.get_masses()).detach().clone()
            if coms.shape != (1, 24, 7) or masses.shape != (1, 24) or not torch.isfinite(coms).all():
                record["invalid_native_local_com_pose"] = coms
                record["invalid_native_masses"] = masses
                raise RuntimeError("Native local COM/mass shape or finite contract changed")
            record["local_com_pos_b"] = coms[:, indices, :3].to(record["body_com_pos_w"])
            record["body_masses_kg"] = masses[:, indices].to(record["body_com_pos_w"])
            composed = body_link_world_com(record["body_link_pos_w"], record["body_link_quat_xyzw"], record["local_com_pos_b"])
            record["com_composition_max_error_m"] = (composed - record["body_com_pos_w"]).abs().amax()
            record["mass_weighted_world_com"] = mass_weighted_world_com(record["body_com_pos_w"], record["body_masses_kg"])
            feet = list(self.foot_indices)
            self.floor.sample_completed_substep(dt, episode_ids=self.episodes(), physics_step=record["physics_step_before_read"],
                foot_body_origin_pos_w=record["body_link_pos_w"][:, feet],
                foot_body_origin_vel_w=record["body_link_vel_w"][:, feet])
            wall = _native_tensor(self.wall_view.get_contact_force_matrix(dt=dt)).detach().clone()
            record["native_wall_normal_force_w"] = wall
            if wall.shape != (4, 1, 3) or not wall.is_floating_point() or not torch.isfinite(wall).all():
                raise FloatingPointError("Invalid wall witness force matrix")
            record["foot_wall_normal_force_w"] = wall[list(self.wall_indices)].reshape(1, 4, 1, 3)
            record["physics_step_after_read"] = self.counter()
            if record["physics_step_after_read"] != record["physics_step_before_read"]:
                raise RuntimeError("Physical state/contact read unexpectedly advanced the actual Lab counter")
            if float(record["com_composition_max_error_m"]) > self.com_tolerance_m:
                raise RuntimeError("Live world COM differs from actual link/local COM composition")
        except Exception as error:
            self.last_partial_error = f"{type(error).__name__}: {error}"
            raise

    def measurements(self):
        import torch

        if len(self.pending) != self.floor.substeps:
            raise RuntimeError("Probe requires exactly four archived substeps")
        if self.counter() != self.before_counter + self.floor.substeps:
            raise RuntimeError("One control did not advance exactly four known physics steps")
        normal = torch.tensor([0., 0., 1.], device=self.world.device)
        result = self.floor.measurements(ground_normal_w=normal)
        keys = set.intersection(*(set(row) for row in self.pending))
        for key in keys:
            values = [row[key] for row in self.pending]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
        self.published = True
        return result


def save_capture(records, path):
    import numpy as np
    import torch

    if not records:
        return None
    keys = set.intersection(*(set(row) for row in records))
    arrays = {}
    for key in keys:
        values = [row[key] for row in records]
        if values[0] is None:
            continue
        if isinstance(values[0], torch.Tensor):
            arrays[key] = np.stack([value.detach().cpu().numpy() for value in values])
        else:
            arrays[key] = np.asarray(values)
    np.savez_compressed(path, **arrays)
    from CrowdSim.humanoid.application.resources import file_sha256

    return {"path": str(path), "sha256": file_sha256(path), "records": len(records),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}}


def summarize_phase(name, records, args):
    import torch

    if not records:
        raise RuntimeError("No completed controls to summarize")
    raw = torch.stack([row["foot_floor_normal_force_w"] for row in records])
    wall = torch.stack([row["foot_wall_normal_force_w"] for row in records])
    sides = torch.stack([row["side_contact_any"] for row in records])
    position = torch.stack([row["body_link_pos_w"] for row in records])
    feet = [records[0]["common_body_names"].index(foot) for foot in FOOT_NAMES]
    summary = {"phase": name, "observed_controls": len(records), "observed_physics_substeps": len(records) * 4,
               "maximum_abs_floor_force_n": float(raw.abs().max()),
               "maximum_wall_foot_normal_force_n": float(torch.linalg.vector_norm(wall, dim=-1).max()),
               "minimum_foot_body_origin_world_z_m": float(position[..., feet, 2].min()),
               "maximum_com_composition_error_m": max(float(row["com_composition_max_error_m"].max()) for row in records),
               "observed_fall_controls": sum(bool(row["fall_mask"]) for row in records),
               "zero_force_tolerance_n": args.zero_force_tolerance_n, "threshold_n": args.threshold_n,
               "criteria_passed": False}
    com_valid = summary["maximum_com_composition_error_m"] <= args.com_tolerance_m
    if name == "standing":
        stable = sides[-args.stable_window_steps:]
        fraction = stable.float().mean(0)[0]
        summary.update(stable_window_controls=len(stable), stable_side_support_fraction=fraction.tolist())
        sensor_valid = len(stable) == args.stable_window_steps and bool((fraction >= args.stable_support_fraction).all())
        summary["standing_no_fall_criterion_passed"] = summary["observed_fall_controls"] == 0
        valid = sensor_valid and summary["standing_no_fall_criterion_passed"]
    elif name == "airborne":
        sensor_valid = (len(records) == 4 and summary["maximum_abs_floor_force_n"] <= args.zero_force_tolerance_n
                        and summary["minimum_foot_body_origin_world_z_m"] > 1.)
        valid = sensor_valid
    elif name == "airborne_wall":
        sensor_valid = (len(records) == 4 and summary["maximum_abs_floor_force_n"] <= args.zero_force_tolerance_n
                        and summary["maximum_wall_foot_normal_force_n"] > args.threshold_n
                        and summary["minimum_foot_body_origin_world_z_m"] > 1.)
        valid = sensor_valid
    else:
        raise ValueError("Unknown foot QA phase")
    summary["criteria_passed"] = bool(valid and com_valid)
    summary["support_sensor_criterion_passed"] = bool(sensor_valid and com_valid)
    return summary


def _phase_reset(world, adapter, anchor, root_pos):
    import torch
    from CrowdSim.humanoid.amp import build_amp_observation
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

    adapter.reset_phase()
    before = adapter.counter()
    zero_root = torch.zeros_like(root_pos)
    world.simulator.reset_envs(ResetState(root_pos=root_pos.clone(), root_rot=anchor["root_rot"].clone(),
        root_vel=zero_root, root_ang_vel=zero_root.clone(), dof_pos=anchor["reset_dof_pos"].clone(),
        dof_vel=torch.zeros_like(anchor["dof_target"]), state_conversion=StateConversion.COMMON),
        env_ids=torch.zeros(1, device=world.device, dtype=torch.long))
    after = adapter.counter()
    if after != before:
        raise RuntimeError("Phase native reset must not add a physics step")
    actual = world.state()
    neutral_geometry_error = None
    if anchor.get("stance_geometry") is not None:
        bind = torch.as_tensor(anchor["stance_geometry"]["geometry"]["body_pos_relative_pelvis_m"],
                               device=world.device, dtype=actual.rigid_body_pos.dtype)
        expected = root_pos[:, None] + bind[None]
        neutral_geometry_error = float((actual.rigid_body_pos - expected).abs().max())
        if neutral_geometry_error > 5e-5:
            raise RuntimeError("Actual neutral reset differs from verified USD bind geometry; grounded placement is unsafe")
    world.age.zero_()
    world.episode_ids.add_(1)
    world.amp_history.copy_(build_amp_observation(actual, mapping=world.mapping)[:, None].expand_as(world.amp_history))
    world.amp_history_episode_ids.copy_(world.episode_ids[:, None].expand_as(world.amp_history_episode_ids))
    world.planar_history.copy_(world.planar(actual)[:, None].expand_as(world.planar_history))
    world.history_available.zero_()
    world.history_available[:, -1] = True
    return actual, {"counter_before_reset": before, "counter_after_reset": after, "episode_id": adapter.episodes()[0],
                    "root_pos_m": actual.root_pos[0].tolist(), "root_rotation_xyzw": actual.root_rot[0].tolist(),
                    "native_reset_count_this_phase": 1, "dof_velocity_frame": "child_joint", "all_initial_velocities_zero": True,
                    "neutral_reset_bind_geometry_max_error_m": neutral_geometry_error}


def load_neutral_collision_geometry(usd_path, common_names):
    """Read actual bind geometry; nested rigid-body colliders keep their owner.

    This supports the current imported SMPL neutral joint frames only. A mesh's
    authored vertices or a cube's transformed corners determine the sole; body
    origins and untransformed bounding boxes do not determine ground clearance.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import numpy as np

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError("Cannot open actual converted robot USD")
    bodies = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in common_names:
            if prim.GetName() in bodies:
                raise ValueError("Ambiguous neutral rigid-body geometry")
            bodies[prim.GetName()] = prim
    if set(bodies) != set(common_names) or "Pelvis" not in bodies:
        raise ValueError("Actual neutral USD must contain every common rigid body exactly once")
    cache = UsdGeom.XformCache()
    inverse_root = cache.GetLocalToWorldTransform(bodies["Pelvis"]).GetInverse()
    positions, parents = [], []
    for name in common_names:
        prim = bodies[name]
        transform = cache.GetLocalToWorldTransform(prim) * inverse_root
        origin = np.asarray(transform.Transform(Gf.Vec3d(0, 0, 0)))
        basis = np.stack([np.asarray(transform.TransformDir(Gf.Vec3d(*axis)))
                          for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], axis=-1)
        if not np.allclose(basis, np.eye(3), atol=1e-6):
            raise ValueError("Neutral static-balance calculation requires verified identity child-joint axes")
        positions.append(origin.tolist())
        parent = prim.GetParent()
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        parents.append(parent.GetName() if parent else None)
    joints = [UsdPhysics.Joint(prim) for prim in stage.Traverse() if UsdPhysics.Joint(prim)]
    for joint in joints:
        for getter in (joint.GetLocalRot0Attr, joint.GetLocalRot1Attr):
            quat = getter().Get()
            if quat is None or abs(abs(float(quat.GetReal())) - 1.) > 1e-6 or np.linalg.norm(quat.GetImaginary()) > 1e-6:
                raise ValueError("Neutral gravity balance requires verified identity joint attachment rotations")
        pos = joint.GetLocalPos1Attr().Get()
        if pos is None or np.linalg.norm(pos) > 1e-6:
            raise ValueError("Neutral gravity balance requires child joint origin at its rigid-body origin")
    points, colliders = {}, {}
    for name in FOOT_NAMES:
        points[name], colliders[name] = [], []
        for prim in Usd.PrimRange(bodies[name]):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != bodies[name]:
                continue
            if prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                vertices = [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)]
            elif prim.IsA(UsdGeom.Mesh):
                vertices = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            else:
                raise ValueError(f"Unsupported exact foot collision geometry: {prim.GetTypeName()}")
            if vertices is None or len(vertices) < 3:
                raise ValueError("Foot collider has no finite geometry vertices")
            transform = cache.GetLocalToWorldTransform(prim) * inverse_root
            transformed = np.asarray([transform.Transform(Gf.Vec3d(*vertex)) for vertex in vertices])
            if not np.isfinite(transformed).all():
                raise ValueError("Nonfinite transformed foot collider geometry")
            points[name].extend(transformed.tolist())
            colliders[name].append(str(prim.GetPath()))
        if not points[name]:
            raise ValueError(f"No owned enabled collision geometry for {name}")
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite neutral bind body positions")
    return {"body_names": list(common_names), "body_pos_relative_pelvis_m": positions,
            "parent_body_names": parents, "foot_collision_vertices_relative_pelvis_m": points,
            "exact_enabled_foot_collider_paths": colliders,
            "joint_frame_assumptions_verified": "identity neutral body and attachment rotation; child attachment position zero"}


def neutral_support_estimate(geometry, masses, local_com, *, sole_clearance_m):
    """Declared static two-ankle support estimate, not measured contact COP."""
    import torch

    names = tuple(geometry["body_names"])
    if (masses.shape != (len(names),) or local_com.shape != (len(names), 3)
            or masses.device != local_com.device or masses.dtype != local_com.dtype
            or not masses.is_floating_point() or not torch.isfinite(masses).all()
            or not torch.isfinite(local_com).all() or bool((masses <= 0).any())):
        raise ValueError("Neutral estimate requires finite positive actual common-order masses/local COM")
    if not math.isfinite(sole_clearance_m) or not 0 <= sole_clearance_m <= .005:
        raise ValueError("Invalid grounded sole clearance")
    position = torch.as_tensor(geometry["body_pos_relative_pelvis_m"], device=masses.device, dtype=masses.dtype)
    com = position + local_com
    weighted = (com * masses[:, None]).sum(0) / masses.sum()
    vertices = {name: torch.as_tensor(geometry["foot_collision_vertices_relative_pelvis_m"][name],
                                    device=masses.device, dtype=masses.dtype) for name in FOOT_NAMES}
    for points in vertices.values():
        if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
            raise ValueError("Invalid exact neutral foot collision vertices")
    combined = torch.cat(list(vertices.values()))
    root_height = sole_clearance_m - float(combined[:, 2].min())
    if not .5 <= root_height <= 1.5:
        raise ValueError("Geometry-derived standing height is outside the expected SMPL rig range")
    ankle_bounds = [torch.stack((vertices[name].min(0).values, vertices[name].max(0).values))
                    for name in ("L_Ankle", "R_Ankle")]
    # Vertical forces at the COM's X and each ankle sole's mid-Y balance the
    # floating-base wrench. Actual contact pressure is measured, never inferred.
    ys = torch.stack([bounds[:, 1].mean() for bounds in ankle_bounds])
    if float((ys[0] - ys[1]).abs()) < 1e-6:
        raise ValueError("Neutral feet cannot define separated left/right support")
    left_fraction = (weighted[1] - ys[1]) / (ys[0] - ys[1])
    if not 0 < float(left_fraction) < 1:
        raise ValueError("Neutral COM projection lies outside two-foot lateral support estimate")
    for bounds in ankle_bounds:
        if not float(bounds[0, 0]) <= float(weighted[0]) <= float(bounds[1, 0]):
            raise ValueError("Neutral COM X is outside an ankle collision footprint")
    contact_positions = torch.stack([torch.stack((weighted[0], y, bounds[0, 2]))
                                    for y, bounds in zip(ys, ankle_bounds)])
    loads = masses.sum() * 9.81 * torch.stack((left_fraction, 1 - left_fraction))
    forces = torch.zeros(2, 3, device=masses.device, dtype=masses.dtype)
    forces[:, 2] = loads
    return {"body_pos": position, "body_com": com, "weighted_com": weighted, "root_height_m": root_height,
            "support_positions": contact_positions, "support_forces": forces,
            "sole_bottoms_m": {name: float(points[:, 2].min()) + root_height for name, points in vertices.items()},
            "footprint_limitation": "initial projected ankle collision footprint; not observed active contact-patch polygon/COP"}


def neutral_static_joint_torques(geometry, masses, estimate):
    """Analytical child-axis torque for the declared static gravity/support load.

    This is a fixed PD bias candidate, not an online controller or additional
    force. The whole-rig external force/moment must balance before deriving it.
    """
    import torch

    names, parents = tuple(geometry["body_names"]), tuple(geometry["parent_body_names"])
    if len(set(names)) != len(names) or len(parents) != len(names) or names[0] != "Pelvis" or parents[0] is not None:
        raise ValueError("Invalid rooted neutral rigid-body tree")
    descendants = []
    for index, name in enumerate(names):
        chain, parent = {name}, parents[index]
        while parent is not None:
            if parent not in names or parent in chain:
                raise ValueError("Invalid neutral rigid-body ancestry")
            chain.add(parent)
            parent = parents[names.index(parent)]
        descendants.append(chain)
    gravity = torch.zeros(len(names), 3, device=masses.device, dtype=masses.dtype)
    gravity[:, 2] = -9.81 * masses
    support = estimate["support_forces"]
    contact = estimate["support_positions"]
    external_force = gravity.sum(0) + support.sum(0)
    external_moment = (torch.linalg.cross(estimate["body_com"], gravity).sum(0)
                       + torch.linalg.cross(contact, support).sum(0))
    tolerance = float(masses.sum() * 9.81) * 1e-5
    if float(external_force.abs().max()) > tolerance or float(external_moment.abs().max()) > tolerance:
        raise ValueError("Declared static neutral support does not balance floating-base wrench")
    torques = torch.zeros(len(names), 3, device=masses.device, dtype=masses.dtype)
    for index, name in enumerate(names[1:], 1):
        origin = estimate["body_pos"][index]
        selected = [i for i, ancestry in enumerate(descendants) if name in ancestry]
        moment = torch.linalg.cross(estimate["body_com"][selected] - origin, gravity[selected]).sum(0)
        for side, foot in enumerate(("L_Ankle", "R_Ankle")):
            if name in descendants[names.index(foot)]:
                moment = moment + torch.linalg.cross(contact[side] - origin, support[side], dim=-1)
        torques[index] = -moment
    if not torch.isfinite(torques).all():
        raise FloatingPointError("Nonfinite static neutral joint torque")
    return torques, {"floating_base_force_residual_n": external_force.tolist(),
                     "floating_base_moment_residual_nm": external_moment.tolist(),
                     "support_model": "two declared vertical ankle loads; true native pressure distribution not known"}


def aggregate_phase_results(phases):
    expected = ("standing", "airborne", "airborne_wall")
    if tuple(phase["phase"] for phase in phases) != expected:
        raise RuntimeError("All three independent native QA phases must complete in their declared order")
    failed = [phase["phase"] for phase in phases if not phase["summary"]["criteria_passed"]]
    standing = phases[0]["summary"]
    sensor_passed = all(phase["summary"]["support_sensor_criterion_passed"] for phase in phases)
    return {"native_physical_probe_passed": not failed, "sensor_channel_criteria_passed": sensor_passed,
            "standing_no_fall_criterion_passed": standing["standing_no_fall_criterion_passed"],
            "failed_criteria_phases": failed, "all_independent_phases_completed": True}


def execute_independent_phases(phase_specs, execute_phase):
    """A completed criterion failure preserves later independent QA evidence.

    Runtime, nonfinite, or incomplete acquisition exceptions still fail closed.
    """
    results = [execute_phase(name, steps) for name, steps in phase_specs]
    return aggregate_phase_results(results)


def build_anchor(world, corpus, args):
    import torch
    from CrowdSim.humanoid.amp import action_to_pd

    if args.pose_anchor in ("neutral", "gravity_balanced_neutral"):
        raw = torch.zeros(1, 69, device=world.device)
        rot = torch.tensor([[0., 0., 0., 1.]], device=world.device)
        height = args.standing_root_height_m
    else:
        start = int(corpus.data["length_starts"][args.motion_id])
        state = corpus.state_at(torch.tensor([start + args.anchor_frame], dtype=torch.long))
        if not all(torch.isfinite(value).all() for value in (state.root_pos, state.root_rot, state.dof_pos, state.dof_vel)):
            raise FloatingPointError("Nonfinite actual pose anchor")
        scale = action_to_pd(torch.ones(1, 69, device=world.device), world.lower, world.upper,
                             dof_names=world.mapping.dof_names)[:, world.mapping.dof_indices]
        target = state.dof_pos[:, world.mapping.dof_indices]
        raw = torch.where(scale.abs() > 1e-8, target / scale.clamp_min(1e-8), torch.zeros_like(target)).clamp(-1, 1)
        rot, height = state.root_rot.clone(), float(state.root_pos[0, 2])
    target = action_to_pd(raw, world.lower, world.upper, dof_names=world.mapping.dof_names)
    reset_dof_pos = target.clone()
    stance = None
    if args.standing_placement == "collision_grounded":
        from CrowdSim.humanoid.foot_support import body_name_permutation
        from CrowdSim.humanoid.application.resources import file_sha256

        usd_path = world.simulator._robot.cfg.spawn.usd_path
        geometry = load_neutral_collision_geometry(usd_path, world.mapping.body_names)
        root_view = world.simulator._robot.root_view
        indices = body_name_permutation(tuple(root_view.shared_metatype.link_names), world.mapping.body_names)
        masses = _native_tensor(root_view.get_masses())[0, list(indices)].to(world.device)
        local_com = _native_tensor(root_view.get_coms())[0, list(indices), :3].to(device=world.device, dtype=masses.dtype)
        estimate = neutral_support_estimate(geometry, masses, local_com, sole_clearance_m=args.initial_sole_clearance_m)
        height = estimate["root_height_m"]
        stance = {"placement": "collision_grounded", "converted_usd_sha256": file_sha256(usd_path),
                  "root_height_above_ground_m": height, "initial_sole_clearance_m": args.initial_sole_clearance_m,
                  "neutral_mass_weighted_COM_relative_pelvis_m": estimate["weighted_com"].tolist(),
                  "total_actual_mass_kg": float(masses.sum()), "initial_sole_bottoms_above_ground_m": estimate["sole_bottoms_m"],
                  "geometry": geometry, "support_estimate_limitation": estimate["footprint_limitation"],
                  "stability_proven": False, "online_balance_controller": False, "external_support_force_applied": False}
        if args.pose_anchor == "gravity_balanced_neutral":
            torques, wrench = neutral_static_joint_torques(geometry, masses, estimate)
            getters = [getattr(root_view, name, None) for name in ("get_dof_stiffnesses", "get_dof_max_forces")]
            if not all(callable(getter) for getter in getters):
                raise RuntimeError("Actual native PD stiffness/effort-limit getters are required for fixed gravity bias")
            native_names = tuple(world.simulator._get_sim_body_ordering().dof_names)
            if len(native_names) != 69 or len(set(native_names)) != 69 or set(native_names) != set(world.mapping.dof_names):
                raise RuntimeError("Actual native/common joint order is not an exact semantic bijection")
            order = [native_names.index(name) for name in world.mapping.dof_names]
            stiffness, limits = [_native_tensor(getter())[0, order].to(device=world.device, dtype=masses.dtype) for getter in getters]
            if (stiffness.shape != (69,) or limits.shape != (69,) or not torch.isfinite(stiffness).all()
                    or not torch.isfinite(limits).all() or bool((stiffness <= 0).any()) or bool((limits <= 0).any())):
                raise ValueError("Actual native PD gains and effort limits must be finite and positive")
            required = torch.stack([torques[geometry["body_names"].index(name.rsplit("_", 1)[0]),
                                           ("x", "y", "z").index(name.rsplit("_", 1)[1])] for name in world.mapping.dof_names])
            if bool((required.abs() >= limits).any()):
                raise ValueError("Static gravity bias exceeds an actual native joint effort limit")
            requested = required / stiffness
            if float(requested.abs().max()) > .25:
                raise ValueError("Static gravity bias exceeds the declared 0.25rad fixed-target safety bound")
            scale = action_to_pd(torch.ones_like(raw), world.lower, world.upper,
                                 dof_names=world.mapping.dof_names)[:, world.mapping.dof_indices]
            requested_canonical = requested[None, world.mapping.dof_indices]
            raw = torch.where(scale.abs() > 1e-8, requested_canonical / scale.clamp_min(1e-8), torch.zeros_like(raw))
            if not torch.isfinite(raw).all() or float(raw.abs().max()) > 1:
                raise ValueError("Static gravity bias cannot be represented without decoder clipping")
            target = action_to_pd(raw, world.lower, world.upper, dof_names=world.mapping.dof_names)
            stance.update(static_wrench=wrench, declared_support_positions_relative_pelvis_m=estimate["support_positions"].tolist(),
                          declared_vertical_support_forces_n=estimate["support_forces"].tolist(),
                          native_common_dof_stiffness_nm_per_rad=stiffness.tolist(), estimated_required_joint_torque_nm=required.tolist(),
                          applied_fixed_target_bias_rad=target[0].tolist(),
                          frozen_decoder_joint_target_bias_omitted=[name for index, name in enumerate(world.mapping.dof_names)
                              if scale[0, list(world.mapping.dof_indices).index(index)] == 0],
                          fixed_bias_semantics="analytical static-load PD target offset; initial neutral DOF pose stays zero; not exact equilibrium proof")
    if not torch.isfinite(target).all() or not torch.isfinite(rot).all():
        raise FloatingPointError("Invalid fixed PD pose")
    return {"raw_action": raw, "dof_target": target, "reset_dof_pos": reset_dof_pos, "root_rot": rot, "root_height_m": height,
            "stance_geometry": stance, "semantics": "fixed actual-rig PD target; frozen toe/hand joints and decoder clipping preserved"}


def main(argv=None):
    args = parse_args(argv)
    resources = preflight(args)
    if args.preflight_only:
        print(json.dumps(resources, indent=2))
        return 0
    from CrowdSim.humanoid.application.configuration import prepare_deterministic_evaluation_environment
    from CrowdSim.humanoid.application.resources import file_sha256

    prepare_deterministic_evaluation_environment(args.deterministic_evaluation)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "preflight": resources, "phases": [], "native_physical_probe_passed": False,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "protocol": {"purpose": "sensor QA, not paper gait success", "author_metric_equivalence": False,
                           "initial_native_resets": "one per phase after AMPWorld construction reset",
                           "runtime_pose_writes_inside_phases": 0, "extra_physics_steps": 0,
                           "airborne_root_lift_m": 3., "airborne_controls": 4, "airborne_wall_controls": 4,
                           "friction_slip": "body-origin tangential-speed proxy only, not contact-point slip",
                           "native_PhysX_time_observed": False}}
    app = world = adapter = None
    completed = []
    phase = "initialization"
    started = time.monotonic()
    try:
        from protomotions.utils.simulator_imports import import_simulator_before_torch

        launcher = import_simulator_before_torch("isaaclab")
        app = launcher({"headless": True, "device": "cuda:0", "enable_cameras": False,
                        "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
        import torch
        from isaaclab.sim.utils import get_current_stage
        from CrowdSim.humanoid.amass_corpus import AMASSCorpus
        from CrowdSim.humanoid.amp_world import AMPWorld
        from CrowdSim.humanoid.application.configuration import configure_deterministic_evaluation_torch, runtime_numerics
        from CrowdSim.humanoid.contact_monitor import resolve_static_collision_paths
        from CrowdSim.humanoid.foot_support import FootSupportMonitor, resolve_floor_collision_paths
        from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths

        torch.set_num_threads(args.cpu_threads)
        configure_deterministic_evaluation_torch(torch, args.deterministic_evaluation)
        torch.manual_seed(args.seed)
        corpus = AMASSCorpus(args.manifest, args.split, device="cuda:0", seed=args.seed, dof_velocity_frame="child_joint")
        actual_clip_frames = int(corpus.data["motion_num_frames"][args.motion_id])
        actual_clip_source = str(corpus.data["motion_files"][args.motion_id]).replace("\\", "/")
        if (actual_clip_frames != resources["pose_anchor"]["frames"]
                or actual_clip_source != resources["pose_anchor"]["source_path"].replace("\\", "/")
                or file_sha256(args.manifest) != resources["manifest_sha256"]):
            raise ValueError("Actual pose-anchor identities changed since preflight")
        world = AMPWorld(app, corpus, num_envs=1, shared_world=True, contact_reporting=True,
                         scene_description=resources["scene"], episode_steps=args.standing_steps + 10)
        if world.simulator.decimation != 4 or not math.isclose(world.simulator._sim.get_physics_dt(), 1 / 120, abs_tol=1e-12):
            raise RuntimeError("Standalone support QA requires the actual four120Hz substep physics loop")
        stage = get_current_stage()
        floors = resolve_floor_collision_paths(stage, ["/World/ground"])
        wall_paths = resolve_static_collision_paths(stage, ["/World/PaperValidation/Obstacle_0"])
        if len(wall_paths) != 1 or set(wall_paths) & set(floors):
            raise RuntimeError("QA wall must resolve to one collider disjoint from the actual ground")
        relative = resolve_body_prim_paths(world.simulator._robot.cfg.spawn.usd_path, list(FOOT_NAMES))
        actor_paths = [{name: f"/World/envs/env_0/Robot/{relative[name]}" for name in FOOT_NAMES}]
        sources = ("scripts/probe_paper_foot_support.py", "CrowdSim/humanoid/foot_support.py", "CrowdSim/humanoid/amp_world.py",
                   "protomotions/simulator/isaaclab/simulator.py", "CrowdSim/humanoid/amp/controller.py")
        report.update(actual_robot_usd_path=world.simulator._robot.cfg.spawn.usd_path,
                      exact_foot_rigid_paths=actor_paths, exact_floor_collider_paths=floors,
                      exact_wall_collider_paths=wall_paths,
                      selected_source_sha256={path: file_sha256(ROOT / path) for path in sources})
        physics_view, lookup_contract = resolve_native_contact_factory(world.simulator._sim)
        report["native_contact_factory_lookup"] = lookup_contract
        floor = FootSupportMonitor.from_factory(physics_view, actor_paths, floors,
            contact_reporting_enabled=True, threshold_n=args.threshold_n, max_contact_data_count=512)
        wall_view = physics_view.create_rigid_contact_view(list(floor.sources),
            filter_patterns=[wall_paths] * 4, max_contact_data_count=512)
        adapter = ProbeSubstepAdapter(world, floor, wall_view, wall_paths[0], com_tolerance_m=args.com_tolerance_m)
        if getattr(world.simulator, "pedestrian_contacts", None) is not None:
            raise RuntimeError("Do not replace a pre-existing obstacle/peer monitor in this standalone QA world")
        world.simulator.pedestrian_contacts = adapter
        report.update(physics=world.physics_manifest(), floor_monitor_contract=floor.manifest,
                      substep_adapter_contract=adapter.manifest, corpus_provenance=corpus.provenance(),
                      actual_robot_usd_path=world.simulator._robot.cfg.spawn.usd_path,
                      runtime_numerics=runtime_numerics(torch, deterministic_evaluation_requested=args.deterministic_evaluation))
        anchor = build_anchor(world, corpus, args)
        report["fixed_PD_anchor"] = {"mode": args.pose_anchor, "source_motion_id": args.motion_id,
            "source_frame": args.anchor_frame, "semantics": anchor["semantics"],
            "normalized_fixed_action": anchor["raw_action"][0].tolist(), "fixed_dof_target_rad": anchor["dof_target"][0].tolist(),
            "initial_reset_dof_position_rad": anchor["reset_dof_pos"][0].tolist(), "stance_geometry": anchor["stance_geometry"]}
        standing_root = torch.cat((world.origins.clone(), torch.tensor([[anchor["root_height_m"]]], device=world.device)), -1)
        standing_root[:, 2] += world.ground_height_at(standing_root[:, :2])
        initial_feet = None
        def execute_phase(phase_name, steps):
            nonlocal phase, completed, initial_feet
            phase = phase_name
            completed = []
            root = standing_root.clone()
            if phase != "standing":
                root[:, 2] += 3.
            if phase == "airborne_wall":
                if initial_feet is None:
                    raise RuntimeError("No observed initial neutral foot offsets for wall placement")
                wall_left_x = resources["scene"]["cuboids"][0]["center"][0] - resources["scene"]["cuboids"][0]["size"][0] / 2
                relative_max_x = float((initial_feet[:, :, 0] - standing_root[:, None, 0]).max())
                root[:, 0] = wall_left_x - relative_max_x + args.wall_penetration_m
            initial, reset = _phase_reset(world, adapter, anchor, root)
            if phase == "standing":
                feet = [world.mapping.body_names.index(name) for name in FOOT_NAMES]
                initial_feet = initial.rigid_body_pos[:, feet].detach().clone()
            phase_report = {"phase": phase, "requested_controls": steps, "reset": reset, "status": "in_progress"}
            report["phases"].append(phase_report)
            with torch.no_grad():
                for _ in range(steps):
                    if not app.is_running() or not world.simulator.is_simulation_running():
                        raise RuntimeError("Simulator/app stopped before all bounded QA phases completed")
                    state, fell, timed_out = world.step(anchor["raw_action"])
                    row = adapter.measurements()
                    row.update(common_body_names=tuple(world.mapping.body_names), fall_mask=bool(fell[0]),
                               time_limit_mask=bool(timed_out[0]), root_pos=state.root_pos.detach().clone(),
                               root_rot_xyzw=state.root_rot.detach().clone())
                    completed.append(row)
            capture = save_capture(completed, args.output / f"{phase}.npz")
            summary = summarize_phase(phase, completed, args)
            phase_report.update(status="passed" if summary["criteria_passed"] else "failed", capture=capture, summary=summary)
            print(json.dumps(summary), flush=True)
            return phase_report

        aggregate = execute_independent_phases((("standing", args.standing_steps), ("airborne", 4), ("airborne_wall", 4)), execute_phase)
        report.update(aggregate, status="passed" if aggregate["native_physical_probe_passed"] else "failed",
                      elapsed_s=time.monotonic() - started)
        if not aggregate["native_physical_probe_passed"]:
            report["error"] = f"Original native QA criteria failed: {', '.join(aggregate['failed_criteria_phases'])}; all independent captures preserved"
        else:
            report["floor_monitor_contract"]["native_physical_probe_passed"] = True
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      failed_phase=phase, elapsed_s=time.monotonic() - started)
        traceback.print_exc()
        if completed:
            report["failed_phase_completed_controls_capture"] = save_capture(completed, args.output / f"{phase}_completed_prefix.npz")
        if adapter is not None and adapter.pending:
            # Preserve each partially acquired row independently; missing fields stay absent, never zero-filled.
            partial = []
            for index, row in enumerate(adapter.pending):
                partial.append(save_capture([row], args.output / f"{phase}_pending_substep_{index}.npz"))
            report["pending_substep_captures"] = partial
            report["pending_substep_error"] = adapter.last_partial_error
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "native_physical_probe_passed": report["native_physical_probe_passed"],
                          "error": report.get("error")}), flush=True)
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1
            world.close()
        elif app is not None:
            app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
