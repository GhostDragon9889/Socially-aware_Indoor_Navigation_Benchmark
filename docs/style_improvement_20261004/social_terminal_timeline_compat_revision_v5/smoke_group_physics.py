#!/usr/bin/env python3
"""Future single-App 8-actor/2-group PhysX smoke; no policy or checkpoint.

This file is preparation source. Nothing runs or emits physics evidence on import.
Only main starts SDK; invoke later under root's exclusive GPU scheduling.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[3]
STAGES = ROOT / "docs/style_improvement_20261004"
FACTORY = STAGES / "group_articulation_scene_staging_v1/group_scene.py"
BRIDGE = STAGES / "group_articulation_tensor_bridge_staging_v1/tensor_bridge.py"
FACTORY_SHA = "88e5a8d289a3a3d8329107faaca857f03cf38a5c640985af84b7b3658f6916b6"
BRIDGE_SHA = "2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec"
DT = 1 / 120
SUBSTEPS = 4
OFFSETS = ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-actors", type=int, choices=(8,), default=8)
    parser.add_argument("--control-steps", type=int, default=8,
                        help="Controls per phase, each with four 120Hz physics steps")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.control_steps < 1:
        parser.error("--control-steps must be positive")
    return args


def _record(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _stage_module(path, name, expected_sha):
    record = _record(path)
    if record["sha256"] != expected_sha:
        raise ValueError(f"Reviewed stage source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, record


def _clone(proxy, torch):
    # Installed ProxyArray.torch is its cached Warp->Torch bridge. Never retain
    # a live alias or apply a second quaternion/frame/DOF conversion here.
    value = proxy.torch.detach().clone()
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError("Expected finite native float32 physics data")
    return value


def _snapshot(assets, rows, bridge, torch, device):
    properties = {
        "root_link_pos_w": "root_link_pos_w", "root_link_quat_xyzw_w": "root_link_quat_w",
        "root_link_lin_vel_w": "root_link_lin_vel_w", "root_link_ang_vel_w": "root_link_ang_vel_w",
        "root_com_lin_vel_w": "root_com_lin_vel_w", "body_link_pos_w": "body_link_pos_w",
        "body_link_quat_xyzw_w": "body_link_quat_w", "body_com_pos_w": "body_com_pos_w",
        "joint_pos_native": "joint_pos", "joint_vel_native": "joint_vel",
    }
    return {name: bridge.assets_to_canonical(
        tuple(_clone(getattr(asset.data, attr), torch) for asset in assets), rows, device=device)
        for name, attr in properties.items()}


def _write_defaults(assets, poses, torch):
    for asset, pose in zip(assets, poses, strict=True):
        zero_velocity = torch.zeros((2, 6), dtype=torch.float32, device=pose.device)
        joint_pos = _clone(asset.data.default_joint_pos, torch)
        joint_vel = _clone(asset.data.default_joint_vel, torch)
        asset.write_root_link_pose_to_sim_index(root_pose=pose, full_data=True)
        # Root LINK velocity, not COM velocity. Zero for this explicit pose probe.
        asset.write_root_link_velocity_to_sim_index(root_velocity=zero_velocity, full_data=True)
        asset.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, full_data=True)
        asset.set_joint_position_target_index(target=joint_pos, full_data=True)


def _contact_views(sim, assets, rows, body_names):
    # Cached import exception tracebacks can retain this completed helper frame.
    # Keep its aliases harmless before extension shutdown, including on failure.
    view = physics_view = None
    views = []
    returned = False
    try:
        from CrowdSim.humanoid.contact_monitor import contact_filter_layout
        from CrowdSim.humanoid.physics_views import resolve_native_contact_factory
        from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths

        physics_view, ownership = resolve_native_contact_factory(sim)
        relative = resolve_body_prim_paths(assets[0].cfg.spawn.usd_path, body_names)
        actor_paths = [None] * 8
        for group in range(2):
            for slot in range(4):
                actor = int(rows[group, slot])
                actor_paths[actor] = [f"/World/envs/env_{group}/Robot_{slot}/{relative[name]}" for name in body_names]
        layouts = contact_filter_layout(actor_paths, [])  # Explicitly exclude floor and self.
        views = []
        for layout in layouts:
            sources, filters = list(layout["sources"]), list(layout["filters"])
            view = physics_view.create_rigid_contact_view(
                sources, filter_patterns=[filters] * len(sources), max_contact_data_count=8192)
            if view.sensor_count != 24 or view.filter_count != 168:
                raise RuntimeError("Exact-path contact view must have 24 bodies and 168 peer-body columns")
            actual_sensors, actual_filters = list(view.sensor_paths), list(view.filter_paths)
            if actual_sensors != sources:
                raise RuntimeError("Native contact sensor paths/order differ from the declared body rows")
            if actual_filters == filters:
                representation = "shared_filter_columns"
            elif actual_filters == [filters] * len(sources):
                representation = "per_sensor_filter_columns"
            else:
                raise RuntimeError("Native contact filter paths/order differ from the declared peer-body columns")
            layout["actual_sensor_paths"] = actual_sensors
            layout["actual_filter_paths"] = actual_filters
            layout["actual_filter_path_representation"] = representation
            views.append(view)
        manifest = {"ownership": ownership, "body_names_in_sensor_order": body_names,
            "actor_body_paths": actor_paths, "layouts": layouts, "sensor_counts": [v.sensor_count for v in views],
            "filter_counts": [v.filter_count for v in views], "max_contact_data_count_per_view": 8192,
            "units": "N", "components": "body-pair normal force vector",
            "physics_dt_s": DT, "threshold_n": 1., "contact_rule": "strict force magnitude >1",
            "excluded": ["floor", "self articulation"], "author_metric_equivalence": False}
        returned = True
        return views, layouts, manifest
    finally:
        if not returned:
            views.clear()  # No caller received these partial native views.
        view = physics_view = sim = assets = None


def _read_contacts(views, layouts, torch, wp, device):
    raw, pair = [], torch.zeros((8, 8), dtype=torch.float32, device=device)
    for view, layout in zip(views, layouts, strict=True):
        force = wp.to_torch(view.get_contact_force_matrix(dt=DT)).detach().clone()
        if tuple(force.shape) != (24, 168, 3) or force.dtype != torch.float32 or not bool(torch.isfinite(force).all()):
            raise ValueError("Contact matrix must be finite float32[24,168,3]")
        raw.append(force)
        peer_max = torch.linalg.vector_norm(force, dim=-1).amax(0).reshape(7, 24).amax(-1)
        pair[layout["actor"], list(layout["peers"])] = peer_max
    return torch.stack(raw), pair


def _phase(name, sim, scene, assets, views, layouts, rows, bridge, args, torch, wp):
    phase_arrays = {"body_pair_normal_force_vector_w": [], "peer_max_force_n": []}
    start = sim.get_physics_step_count()
    if type(start) is not int or start < 0:
        raise ValueError("Native physics step counter must be a nonnegative int")
    targets = [_clone(asset.data.default_joint_pos, torch) for asset in assets]
    for _ in range(args.control_steps):
        forces, pairs = [], []
        for _ in range(SUBSTEPS):
            for asset, target in zip(assets, targets, strict=True):
                asset.set_joint_position_target_index(target=target, full_data=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            raw, pair = _read_contacts(views, layouts, torch, wp, args.device)
            forces.append(raw)
            pairs.append(pair)
        phase_arrays["body_pair_normal_force_vector_w"].append(torch.stack(forces))
        phase_arrays["peer_max_force_n"].append(torch.stack(pairs))
        for key, value in _snapshot(assets, rows, bridge, torch, args.device).items():
            phase_arrays.setdefault(key, []).append(value)
    end = sim.get_physics_step_count()
    if type(end) is not int or end - start != args.control_steps * SUBSTEPS:
        raise ValueError("Actual physics counter differs from the requested control/substep budget")
    arrays = {key: torch.stack(values).cpu().numpy() for key, values in phase_arrays.items()}
    peak = arrays["peer_max_force_n"].max(axis=(0, 1))
    return arrays, {"name": name, "controls": args.control_steps, "substeps": args.control_steps * SUBSTEPS,
        "actual_absolute_physics_step_start": start, "actual_absolute_physics_step_end": end,
        "actual_physics_step_delta": end - start, "peer_peak_force_n": peak.tolist(),
        "all_peer_body_pair_force_exact_zero": bool((arrays["body_pair_normal_force_vector_w"] == 0).all()),
        "peer_contact_threshold_n": 1., "peer_contacts_above_threshold": int((peak > 1.).sum())}


def _release_terminal_timeline_caches(owned_sim, report):
    """Pinned terminal-process compatibility for retained Lab/Core Timeline caches.

    This deliberately updates two private SDK class attributes after owned
    context cleanup. It is not a public API and does not support another App or
    simulation in this process. SDK files and other interface caches are untouched.
    """
    expected = {'lab': ('isaaclab_physx.physics.physx_manager', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py', 34974, 'd83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5'), 'base': ('isaaclab.physics.physics_manager', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/physics/physics_manager.py', 20828, '3a88544d96df6dc02a17a611529b5fd8bc58c2e251b4323be9cbc7d0bc7a5b47'), 'context': ('isaaclab.sim.simulation_context', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py', 42425, '8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89'), 'core': ('isaacsim.core.simulation_manager.impl.simulation_manager', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py', 79181, 'f95259051bb494ca1cf85ec3cfebdce61b4ec5f24cad49cf0ea295ccc0024f8e'), 'extension': ('isaacsim.core.simulation_manager.impl.extension', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/extension.py', 3364, '8c4ef108ce939ed8a485540e719af2517aba7a2401a1de6d0848a06a457b3ea1')}
    method_origins = {'lab': {'close': ('PhysxManager.close', 424, True)}, 'base': {'close': ('PhysicsManager.close', 436, True)}, 'context': {'instance': ('SimulationContext.instance', 98, True), 'clear_instance': ('SimulationContext.clear_instance', 812, True)}, 'core': {'_shutdown': ('SimulationManager._shutdown', 361, True), '_reset': ('SimulationManager._reset', 380, True), 'enable_all_default_callbacks': ('SimulationManager.enable_all_default_callbacks', 1381, True), 'enable_default_callbacks': ('SimulationManager.enable_default_callbacks', 1329, True), 'invalidate_physics': ('SimulationManager.invalidate_physics', 728, True)}, 'extension': {'on_shutdown': ('Extension.on_shutdown', 80, False)}}
    classes, records = {}, {}
    try:
        for key, (module_name, filename, size, sha) in expected.items():
            module = sys.modules.get(module_name)
            if module is None or not getattr(module, "__file__", None):
                raise RuntimeError("Missing initialized SDK compatibility source: " + key)
            record = _record(Path(module.__file__))
            if record != {"path": filename, "bytes": size, "sha256": sha}:
                raise RuntimeError("Terminal compatibility SDK whole-source identity mismatch: " + key)
            classes[key] = getattr(module, {"lab": "PhysxManager", "base": "PhysicsManager",
                "context": "SimulationContext", "core": "SimulationManager", "extension": "Extension"}[key])
            if classes[key].__module__ != module_name:
                raise RuntimeError("Terminal compatibility SDK class origin mismatch: " + key)
            records[key] = record
            for method_name, (qualname, first_line, classmethod_expected) in method_origins[key].items():
                method = getattr(classes[key], method_name)
                function = method.__func__ if classmethod_expected else method
                descriptor = classes[key].__dict__[method_name]
                if (classmethod_expected and (method.__self__ is not classes[key]
                        or descriptor.__func__ is not function)
                        or not classmethod_expected and descriptor is not function
                        or function.__qualname__ != qualname
                        or Path(function.__code__.co_filename).resolve() != Path(filename)
                        or function.__code__.co_firstlineno != first_line):
                    raise RuntimeError("Terminal compatibility bound-method origin mismatch: " + key + "." + method_name)
        if (type(owned_sim) is not classes["context"] or classes["context"].instance() is not None
                or owned_sim.physics_manager is not classes["lab"]):
            raise RuntimeError("Terminal compatibility requires this entry's cleared exact simulation/manager")
        if (classes["base"]._sim is not None or classes["base"]._cfg is not None
                or classes["base"]._callbacks or classes["lab"]._subscriptions
                or classes["lab"]._view is not None or classes["lab"]._view_warp is not None):
            raise RuntimeError("Lab context callbacks/subscriptions/views were not fully cleared")
        module = sys.modules[expected["extension"][0]]
        if (classes["core"]._simulation_manager_interface is None
                or classes["core"]._simulation_manager_interface is not module._simulation_manager_interface):
            raise RuntimeError("Core extension must retain its live owned interface for repeated shutdown")
        report["terminal_timeline_compat_sources"] = records
        report["terminal_timeline_cache_before"] = {
            "Lab": {"native_module": type(classes["lab"]._timeline).__module__,
                "native_class": type(classes["lab"]._timeline).__name__},
            "Core": {"native_module": type(classes["core"]._timeline).__module__,
                "native_class": type(classes["core"]._timeline).__name__},
            "same_native_python_object": classes["lab"]._timeline is classes["core"]._timeline,
            "private_runtime_compatibility": True, "terminal_process_only": True}
        if any(report["terminal_timeline_cache_before"][key] != {
                "native_module": "omni.timeline._timeline", "native_class": "Timeline"}
                for key in ("Lab", "Core")):
            raise RuntimeError("Expected initialized native Timeline class caches")
        classes["core"]._shutdown()
        report["core_terminal_shutdown_returned"] = True
        subscription_names = ("_default_callback_warm_start", "_default_callback_on_stop",
            "_default_callback_stage_open", "_default_callback_stage_close",
            "_assets_loading_callback", "_assets_loaded_callback", "_simulation_registry_sub")
        if (classes["core"]._callbacks or classes["core"]._physics_scenes
                or classes["core"]._physics_sim_view is not None
                or classes["core"]._physics_sim_view__warp is not None
                or any(getattr(classes["core"], name) is not None for name in subscription_names)
                or set(classes["core"]._default_callbacks_state) != {
                    "warm_start", "on_stop", "stage_open", "stage_close", "post_warm_start"}
                or any(value is not False for value in classes["core"]._default_callbacks_state.values())):
            raise RuntimeError("Core shutdown left active callbacks/subscriptions/scenes/views")
        # Native Timeline aliases are never assigned to helper locals or report values.
        classes["lab"]._timeline = None
        classes["core"]._timeline = None
        if classes["lab"]._timeline is not None or classes["core"]._timeline is not None:
            raise RuntimeError("Terminal native Timeline caches were not dropped")
        report["terminal_timeline_cache_compat_returned"] = True
    finally:
        # An import error traceback may retain this helper frame; leave no class aliases.
        classes.clear()
        owned_sim = module = method = function = descriptor = None


def _run(args, report, archives):
    # Bootstrap is the only SDK exception before App. No physics/asset/Torch imports precede it.
    sys.path.insert(0, str(ROOT))
    if "torch" in sys.modules:
        raise RuntimeError("Run this entry in a fresh process; launcher must precede Torch imports")
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    launcher = import_simulator_before_torch("isaaclab")
    app, sim, scene, asset = None, None, None, None
    assets, views, contact_view_refs = [], [], []
    try:
        app = launcher({"headless": True, "device": args.device, "enable_cameras": False,
            "fast_shutdown": False, "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
        report["driver_app_constructors_returned"] = 1
        import torch
        import warp as wp
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab_physx.physics import PhysxCfg
        from protomotions.robot_configs.smpl import SmplRobotConfig
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
        from protomotions.simulator.isaaclab.utils.actuator_groups import build_isaaclab_joint_name_map
        from protomotions.simulator.isaaclab.utils.scene import SceneCfg

        torch.set_num_threads(4)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        factory, factory_record = _stage_module(FACTORY, "smoke_reviewed_group_scene", FACTORY_SHA)
        bridge, bridge_record = _stage_module(BRIDGE, "smoke_reviewed_actor_bridge", BRIDGE_SHA)
        report["stage_sources"] = {"factory": factory_record, "actor_bridge": bridge_record}
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=4,
            physics=PhysxCfg(max_position_iteration_count=4, max_velocity_iteration_count=4)))
        config = IsaacLabSimulatorConfig(num_envs=2, headless=True, experiment_name="diffusion_amp_reconstruction")
        config.sim.fps, config.sim.decimation = 120, 4
        config.sim.physx.num_position_iterations, config.sim.physx.num_velocity_iterations = 4, 4
        robot = SmplRobotConfig()
        robot.contact_bodies = None  # Do not create old single-Robot sensor configs.
        base = SceneCfg(config=config, robot_config=robot, terrain=None, projectile_config=None,
            num_envs=2, env_spacing=8., filter_collisions=True, replicate_physics=True)
        # Same reviewed SMPL conversion/PD path, contact-report schema authored before spawn.
        base.robot.spawn.activate_contact_sensors = True
        base.ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1,
            spawn=sim_utils.CuboidCfg(size=(64., 64., .2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.1)))
        cfg = factory.create_group_scene_cfg(base.robot, base, num_actors=8,
            actor_offsets_xy=OFFSETS, env_spacing=8.)
        scene = InteractiveScene(cfg)
        sim.reset()
        assets = [scene[name] for name in factory.ACTOR_NAMES]
        rows = bridge.mixed_group_rows(8, device=args.device)
        scene_ids = torch.empty(8, dtype=torch.int64, device=args.device)
        scene_ids[rows] = torch.arange(2, device=args.device)[:, None]
        names = [{"asset_key": name, "num_instances": asset.num_instances,
                  "body_names": list(asset.body_names), "joint_names": list(asset.joint_names)}
                 for name, asset in zip(factory.ACTOR_NAMES, assets, strict=True)]
        common_bodies = list(robot.kinematic_info.body_names)
        joint_map = build_isaaclab_joint_name_map(robot.kinematic_info)
        common_joints = [joint_map.semantic_to_backend[name] for name in robot.kinematic_info.dof_names]
        for asset in assets:
            if (asset.num_instances != 2 or len(asset.body_names) != 24 or len(asset.joint_names) != 69
                    or len(set(asset.body_names)) != 24 or len(set(asset.joint_names)) != 69
                    or set(asset.body_names) != set(common_bodies) or set(asset.joint_names) != set(common_joints)):
                raise ValueError("Actual native SMPL body/joint names differ from the reviewed rig")
        if any(item["body_names"] != names[0]["body_names"] or item["joint_names"] != names[0]["joint_names"] for item in names):
            raise ValueError("The four assets must have identical native body/joint ordering")
        origins = scene.env_origins.detach().clone()
        if origins.shape != (2, 3) or not bool(torch.isfinite(origins).all()):
            raise ValueError("Scene must expose two finite world environment origins")
        poses = [_clone(asset.data.default_root_pose, torch) for asset in assets]
        for pose in poses:
            pose[:, :3] += origins  # Exactly once: default_root_pose is env-local.
        if any(bool((pose[:, :2].abs() > 31).any()) for pose in poses):
            raise ValueError("Reviewed static floor does not cover the initial world poses")
        report.update(num_actors=8, num_groups=2, group_actor_rows=rows.cpu().tolist(),
            scene_ids=scene_ids.cpu().tolist(), actual_env_origins=origins.cpu().tolist(), native_assets=names,
            quaternion_order="xyzw", state_positions="link frames; separate COM fields retained",
            native_body_to_common_indices=[assets[0].body_names.index(name) for name in common_bodies],
            native_joint_to_common_indices=[assets[0].joint_names.index(name) for name in common_joints],
            array_body_joint_order="actual native order; bridge changes actor rows only",
            render_argument_each_physics_step=False, floor_world_bounds_xy=[-32., -32., 32., 32.],
            floor_top_world_z=0., collision_filter_enabled=cfg.filter_collisions)
        views, layouts, manifest = _contact_views(sim, assets, rows, common_bodies)
        contact_view_refs = [weakref.ref(view) for view in views]
        report["owned_contact_view_count"] = len(contact_view_refs)
        report["contact_manifest"] = manifest
        _write_defaults(assets, poses, torch)
        scene.reset()
        initial = _snapshot(assets, rows, bridge, torch, args.device)
        archives["separated_initial"] = {key: value.cpu().numpy() for key, value in initial.items()}
        arrays, separated = _phase("separated_default_pose", sim, scene, assets, views, layouts, rows, bridge, args, torch, wp)
        archives["separated"] = arrays
        report["separated_phase"] = separated
        if separated["peer_contacts_above_threshold"]:
            raise RuntimeError("Separated phase unexpectedly reported peer contact >1N")
        probe = [pose.clone() for pose in poses]
        # Same-group actors 0 and1 have a deliberate 0.2m root offset; actor2
        # (another env) gets actor0's exact world pose. Others remain separated.
        probe[1][0] = probe[0][0]
        probe[1][0, 0] += .2
        probe[0][1] = probe[0][0]
        _write_defaults(assets, probe, torch)
        scene.reset()
        initial = _snapshot(assets, rows, bridge, torch, args.device)
        archives["overlap_probe_initial"] = {key: value.cpu().numpy() for key, value in initial.items()}
        exact_pose = bool(torch.equal(initial["root_link_pos_w"][0], initial["root_link_pos_w"][2])
                          and torch.equal(initial["root_link_quat_xyzw_w"][0], initial["root_link_quat_xyzw_w"][2]))
        if not exact_pose:
            raise RuntimeError("Cross-group negative probe did not actually start at the same world pose")
        arrays, probe_report = _phase("deliberate_overlap_filter_probe", sim, scene, assets, views, layouts, rows, bridge, args, torch, wp)
        archives["overlap_probe"] = arrays
        peak = torch.as_tensor(arrays["peer_max_force_n"].max(axis=(0, 1)), device=args.device)
        cross = scene_ids[:, None] != scene_ids[None, :]
        ids = scene_ids.cpu().tolist()
        raw_cross_max = 0.
        for layout in layouts:
            columns = [column for column, peer in enumerate(layout["partner_ids"])
                       if ids[layout["actor"]] != ids[peer]]
            raw_cross = arrays["body_pair_normal_force_vector_w"][:, :, layout["actor"], :, columns, :]
            raw_cross_max = max(raw_cross_max, float(abs(raw_cross).max()))
        probe_report.update(same_group_probe_pair=[0, 1], cross_group_same_initial_world_pose_pair=[0, 2],
            cross_group_initial_pose_actually_equal=exact_pose,
            same_group_probe_union_peak_force_n=float(torch.maximum(peak[0, 1], peak[1, 0])),
            all_cross_group_peak_force_n=float(peak[cross].max()),
            all_cross_group_raw_force_absolute_max_n=raw_cross_max,
            all_cross_group_raw_force_exact_zero=raw_cross_max == 0.,
            all_cross_group_force_exact_zero=raw_cross_max == 0. and bool((peak[cross] == 0).all()),
            interpretation="deliberate body overlap collision-filter probe, not a training pose or quality result")
        report["overlap_probe_phase"] = probe_report
        if probe_report["same_group_probe_union_peak_force_n"] <= 1.:
            raise RuntimeError("Same-group positive contact probe did not activate force reporting >1N")
        if not probe_report["all_cross_group_force_exact_zero"]:
            raise RuntimeError("Cross-group force must be exactly zero, not merely below the 1N contact threshold")
        report["physics_smoke_checks_passed"] = True
    finally:
        errors = []
        if sim is not None:
            try:
                sim.stop()
                report["simulation_stop_returned"] = True
            except Exception as error:
                errors.append(f"sim.stop: {type(error).__name__}: {error}")
        try:
            # Clear owned containers in place: exception frames can retain their lists.
            # Archives contain detached state/contact clones, never native API objects.
            views.clear()
            assets.clear()
            asset = scene = None  # The explicit asset loop above leaves a final alias.
            report["pre_context_clear_gc_collected"] = gc.collect()
            report["owned_contact_views_live_after_release"] = sum(
                ref() is not None for ref in contact_view_refs)
            if report["owned_contact_views_live_after_release"]:
                raise RuntimeError("Owned native contact views remain alive before SDK teardown")
            report["owned_native_references_release_returned"] = True
        except Exception as error:
            errors.append(f"owned native release: {type(error).__name__}: {error}")
        if sim is not None:
            try:
                if type(sim).instance() is not sim:
                    raise RuntimeError("Refuse to clear a simulation singleton not owned by this entry")
                sim.clear_instance()
                if type(sim).instance() is not None:
                    raise RuntimeError("Simulation singleton remains after clear_instance returned")
                report["simulation_clear_instance_returned"] = True
            except Exception as error:
                errors.append(f"sim.clear_instance: {type(error).__name__}: {error}")
        try:
            if sim is None:
                raise RuntimeError("Terminal Timeline compatibility requires an owned simulation context")
            _release_terminal_timeline_caches(sim, report)
        except Exception as error:
            errors.append(f"terminal Timeline compatibility: {type(error).__name__}: {error}")
        try:
            # Manager/service shutdown can make additional traceback cycles unreachable.
            # Collect them before unloading tensor extensions, even when clear raised.
            report["post_context_clear_gc_collected"] = gc.collect()
            report["post_context_clear_gc_returned"] = True
        except Exception as error:
            errors.append(f"post-context gc: {type(error).__name__}: {error}")
        if app is not None:
            try:
                app.close()
                report["app_close_returned"] = True
            except Exception as error:
                errors.append(f"app.close: {type(error).__name__}: {error}")
        report["cleanup_errors"] = errors


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output must be a new or empty independent directory")
    if output == ROOT or ROOT / "output" == output or ROOT / "docs" == output:
        raise ValueError("Refusing an ambiguous output root")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "local.group_articulation.actual_PhysX_smoke.v1", "status": "failed",
        "arguments": {**vars(args), "output": str(output)}, "execution_source": _record(Path(__file__)),
        "driver_app_constructors_returned": 0, "physics_smoke_checks_passed": False,
        "simulation_stop_returned": False, "simulation_clear_instance_returned": False, "app_close_returned": False,
        "owned_native_references_release_returned": False, "post_context_clear_gc_returned": False,
        "core_terminal_shutdown_returned": False, "terminal_timeline_cache_compat_returned": False,
        "owned_contact_view_count": 0, "owned_contact_views_live_after_release": None,
        "NN_constructed": False, "checkpoint_loaded": False, "training_executed": False,
        "author_experiment_reproduced": False}
    archives = {}
    try:
        _run(args, report, archives)
        if (report["cleanup_errors"] or not report["simulation_clear_instance_returned"]
                or not report["owned_native_references_release_returned"] or not report["post_context_clear_gc_returned"]
                or not report["core_terminal_shutdown_returned"] or not report["terminal_timeline_cache_compat_returned"]):
            raise RuntimeError("SDK cleanup did not return successfully")
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    if archives:
        import numpy as np
        report["artifacts"] = {}
        for phase, arrays in archives.items():
            path = output / f"{phase}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            report["artifacts"][phase] = _record(path)
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "result": str(output / "result.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
