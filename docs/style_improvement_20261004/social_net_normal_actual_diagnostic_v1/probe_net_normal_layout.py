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
OVERLAY = ROOT / "output/native_sdk_overlay/simulation_manager_shutdown_v1/isaacsim.core.simulation_manager"
OVERLAY_MANIFEST = OVERLAY.parent / "build_copy_manifest.json"
OVERLAY_MANIFEST_SHA = "713c477db243b9c72081beec906fc30d67143cfd61ec4ca3eb61f95c0489c980"
OVERLAY_KIT_ARGS = "--/app/exts/devPaths=" + json.dumps([str(OVERLAY)], separators=(",", ":"))
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
    view = physics_view = net_view = None
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
            net_view = physics_view.create_rigid_contact_view(sources, filter_patterns=[], max_contact_data_count=0)
            observed = {"actor": layout["actor"], "sensor_count": net_view.sensor_count,
                "sensor_paths": list(net_view.sensor_paths), "expected_sensors": sources,
                "filter_count": net_view.filter_count, "filter_paths": list(net_view.filter_paths),
                "native_check": bool(net_view.check())}
            print("NET_NORMAL_LAYOUT_OBSERVED " + json.dumps(observed), flush=True)
            layout["observed_no_filter_metadata"] = observed
            net_ref = weakref.ref(net_view)
            net_view = None
            gc.collect()
            if net_ref() is not None:
                raise RuntimeError("Diagnostic net-normal view remains owned before physical phases")
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
        view = physics_view = net_view = sim = assets = None


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


def _release_terminal_timeline_caches(owned_sim, report, *, phase):
    """Pinned owned-manager normalization and terminal Timeline compatibility.

    Before clear, normalize only this owned context's public manager attribute.
    After SDK close, detach the existing scene backend through its public setter,
    then retain the original private terminal Timeline release.
    No SDK files/methods/Base state are patched; another App/context is unsupported.
    """
    expected = {'lab': ('isaaclab_physx.physics.physx_manager', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py', 34974, 'd83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5'), 'base': ('isaaclab.physics.physics_manager', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/physics/physics_manager.py', 20828, '3a88544d96df6dc02a17a611529b5fd8bc58c2e251b4323be9cbc7d0bc7a5b47'), 'context': ('isaaclab.sim.simulation_context', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py', 42425, '8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89'), 'core': ('isaacsim.core.simulation_manager.impl.simulation_manager', '/project/EXP/SangforAndSaut/ProtoMotions/output/native_sdk_overlay/simulation_manager_shutdown_v1/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py', 79181, 'f95259051bb494ca1cf85ec3cfebdce61b4ec5f24cad49cf0ea295ccc0024f8e'), 'extension': ('isaacsim.core.simulation_manager.impl.extension', '/project/EXP/SangforAndSaut/ProtoMotions/output/native_sdk_overlay/simulation_manager_shutdown_v1/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/extension.py', 3364, '8c4ef108ce939ed8a485540e719af2517aba7a2401a1de6d0848a06a457b3ea1')}
    method_origins = {'lab': {'close': ('PhysxManager.close', 424, True)}, 'base': {'close': ('PhysicsManager.close', 436, True)}, 'context': {'instance': ('SimulationContext.instance', 98, True), 'clear_instance': ('SimulationContext.clear_instance', 812, True)}, 'core': {'_shutdown': ('SimulationManager._shutdown', 361, True), '_reset': ('SimulationManager._reset', 380, True), 'enable_all_default_callbacks': ('SimulationManager.enable_all_default_callbacks', 1381, True), 'enable_default_callbacks': ('SimulationManager.enable_default_callbacks', 1329, True), 'invalidate_physics': ('SimulationManager.invalidate_physics', 728, True)}, 'extension': {'on_shutdown': ('Extension.on_shutdown', 80, False)}}
    classes, records = {}, {}
    owned_manager = resolved_manager = manager_proxy = None
    backend = simulation_view = rigid_body_view = tensor_module = validity = None
    scene_backend_view_refs = ()
    try:
        if phase not in ("before_clear", "after_clear"):
            raise ValueError("Unknown owned-manager compatibility phase")
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
        resolver_expected = {'resolver': ('isaaclab.utils.string', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/utils/string.py', 22809, '8943b90128542370fbecdfe4de1561722ff85901dc699b6051b50c1b7e43266f'), 'configclass': ('isaaclab.utils.configclass', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/utils/configclass.py', 26964, '3d782b38e8a0d9d10d9d8ad3bf8e86bcd3731f1f84be4e296a6dfd4d40e644d4'), 'cfg': ('isaaclab_physx.physics.physx_manager_cfg', '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py', 9949, 'a09ae907d6e46cf466e0b764be56753e6e1e554755aa1ea5fa3ac4863d70aed3')}
        resolver_origins = {"resolver": (("string_to_callable", 152),
            ("ResolvableString._resolve", 207), ("ResolvableString.__getattr__", 246)),
            "configclass": (("configclass", 36), ("_field_module_dir", 188),
                ("_wrap_resolvable_strings", 212), ("_custom_post_init", 488)), "cfg": ()}
        for key, (module_name, filename, size, sha) in resolver_expected.items():
            module = sys.modules.get(module_name)
            if module is None or not getattr(module, "__file__", None):
                raise RuntimeError("Missing initialized owned-manager resolver source: " + key)
            record = _record(Path(module.__file__))
            if record != {"path": filename, "bytes": size, "sha256": sha}:
                raise RuntimeError("Owned-manager resolver whole-source identity mismatch: " + key)
            records[key] = record
            for qualname, first_line in resolver_origins[key]:
                function = module
                for attribute in qualname.split("."):
                    function = getattr(function, attribute)
                if (function.__module__ != module_name or function.__qualname__ != qualname
                        or Path(function.__code__.co_filename).resolve() != Path(filename)
                        or function.__code__.co_firstlineno != first_line):
                    raise RuntimeError("Owned-manager resolver method origin mismatch: " + qualname)
            if key in ("resolver", "cfg"):
                classes[key] = getattr(module, {"resolver": "ResolvableString", "cfg": "PhysxCfg"}[key])
                if (classes[key].__module__ != module_name or classes[key].__qualname__ != {
                        "resolver": "ResolvableString", "cfg": "PhysxCfg"}[key]):
                    raise RuntimeError("Owned-manager resolver class origin mismatch: " + key)
        guard = {
            "phase": phase,
            "context_type_is_canonical": type(owned_sim) is classes["context"],
            "context_singleton_is_owned": classes["context"].instance() is owned_sim,
            "context_singleton_is_none": classes["context"].instance() is None,
            "context_type_module": type(owned_sim).__module__,
            "context_type_qualname": type(owned_sim).__qualname__,
            "context_source": records["context"]["path"],
            "physics_cfg_type_is_canonical": type(owned_sim._physics) is classes["cfg"],
            "physics_cfg_is_context_cfg": owned_sim._physics is owned_sim.cfg.physics,
            "base_sim_is_owned": classes["base"]._sim is owned_sim,
            "base_cfg_is_owned_physics": classes["base"]._cfg is owned_sim._physics,
            "base_sim_is_none": classes["base"]._sim is None,
            "base_cfg_is_none": classes["base"]._cfg is None,
            "base_callbacks_empty": not bool(classes["base"]._callbacks),
            "lab_subscriptions_empty": not bool(classes["lab"]._subscriptions),
            "lab_view_is_none": classes["lab"]._view is None,
            "lab_view_warp_is_none": classes["lab"]._view_warp is None,
            "lab_inherited_sim_is_none": classes["lab"]._sim is None}
        report["owned_manager_before_clear_guard" if phase == "before_clear"
            else "terminal_owned_context_guard"] = guard
        report["terminal_timeline_compat_sources"] = records
        if (not guard["context_type_is_canonical"] or not guard["physics_cfg_type_is_canonical"]
                or not guard["physics_cfg_is_context_cfg"]):
            raise RuntimeError("Requires this entry's exact owned SimulationContext and PhysxCfg")
        owned_manager = owned_sim.physics_manager
        manager_proxy = owned_sim._physics.class_type
        guard.update(
            manager_raw_is_canonical=owned_manager is classes["lab"],
            manager_raw_is_config_proxy=owned_manager is manager_proxy,
            manager_proxy_type_is_exact=type(manager_proxy) is classes["resolver"],
            manager_proxy_reference_is_expected=str(manager_proxy) ==
                "isaaclab_physx.physics.physx_manager:PhysxManager",
            manager_raw_type_module=type(owned_manager).__module__,
            manager_raw_type_qualname=type(owned_manager).__qualname__,
            manager_proxy_type_module=type(manager_proxy).__module__,
            manager_proxy_type_qualname=type(manager_proxy).__qualname__,
            manager_proxy_source=records["resolver"]["path"],
            manager_config_source=records["cfg"]["path"],
            canonical_manager_source=records["lab"]["path"],
            manager_resolve_is_canonical=None)
        if (not guard["manager_proxy_type_is_exact"]
                or not guard["manager_proxy_reference_is_expected"]):
            raise RuntimeError("Expected owned PhysxCfg's exact canonical ResolvableString proxy")
        resolved_manager = manager_proxy._resolve()
        guard["manager_resolve_is_canonical"] = resolved_manager is classes["lab"]
        if resolved_manager is not classes["lab"]:
            raise RuntimeError("Owned config proxy did not resolve to the exact pinned Lab class")
        if phase == "before_clear":
            if (not guard["context_singleton_is_owned"] or not guard["base_sim_is_owned"]
                    or not guard["base_cfg_is_owned_physics"]
                    or not guard["manager_raw_is_config_proxy"]):
                raise RuntimeError("Refuse normalization of an inactive/unowned manager context")
            # Public instance attribute only. Preserve PhysxCfg.class_type proxy and
            # every SDK method/private Base field; no initialize/dispatch/step here.
            owned_sim.physics_manager = resolved_manager
            guard.update(
                normalized_public_manager_is_canonical=owned_sim.physics_manager is classes["lab"],
                original_cfg_proxy_preserved=owned_sim._physics.class_type is manager_proxy,
                base_sim_still_owned=classes["base"]._sim is owned_sim,
                base_cfg_still_owned=classes["base"]._cfg is owned_sim._physics)
            if not all(guard[key] for key in ("normalized_public_manager_is_canonical",
                    "original_cfg_proxy_preserved", "base_sim_still_owned", "base_cfg_still_owned")):
                raise RuntimeError("Public owned-manager normalization postconditions failed")
            report["owned_manager_normalization_returned"] = True
            return
        if (report.get("owned_manager_normalization_returned") is not True
                or not guard["context_singleton_is_none"]
                or not guard["manager_raw_is_canonical"]):
            raise RuntimeError("Terminal cleanup requires observed owned normalization and cleared context")
        if not all(guard[key] for key in ("base_sim_is_none", "base_cfg_is_none",
                "base_callbacks_empty", "lab_subscriptions_empty", "lab_view_is_none", "lab_view_warp_is_none")):
            raise RuntimeError("Lab context callbacks/subscriptions/views were not fully cleared")
        # The manager's existing scene backend retains a separate SimulationView
        # alias after close/_invalidate_views. Verify the actual public getter and
        # backend property before dropping it; never call its lazy rigid getter.
        module = sys.modules[expected["lab"][0]]
        classes["scene_backend"] = module.PhysxSceneDataBackend
        if (classes["scene_backend"].__module__ != expected["lab"][0]
                or classes["scene_backend"].__qualname__ != "PhysxSceneDataBackend"):
            raise RuntimeError("Scene backend class must originate from the pinned Lab source")
        method = classes["lab"].get_scene_data_backend
        descriptor = classes["lab"].__dict__["get_scene_data_backend"]
        function = method.__func__
        if (type(descriptor) is not classmethod or method.__self__ is not classes["lab"]
                or descriptor.__func__ is not function
                or function.__module__ != expected["lab"][0]
                or function.__qualname__ != "PhysxManager.get_scene_data_backend"
                or Path(function.__code__.co_filename).resolve() != Path(expected["lab"][1])
                or function.__code__.co_firstlineno != 347):
            raise RuntimeError("Scene backend public getter origin mismatch")
        descriptor = classes["scene_backend"].__dict__["simulation_view"]
        if type(descriptor) is not property:
            raise RuntimeError("Scene backend simulation_view must be the pinned property")
        for function, first_line in ((classes["scene_backend"].__dict__["__init__"], 159),
                (descriptor.fget, 164), (descriptor.fset, 168)):
            if (function.__module__ != expected["lab"][0]
                    or function.__qualname__ != "PhysxSceneDataBackend." +
                        ("__init__" if first_line == 159 else "simulation_view")
                    or Path(function.__code__.co_filename).resolve() != Path(expected["lab"][1])
                    or function.__code__.co_firstlineno != first_line):
                raise RuntimeError("Scene backend property/constructor origin mismatch")
        tensor_expected = ("omni.physics.tensors.api",
            "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py",
            382927, "b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903")
        tensor_module = sys.modules.get(tensor_expected[0])
        if tensor_module is None or not getattr(tensor_module, "__file__", None):
            raise RuntimeError("Missing already initialized tensor API module")
        record = _record(Path(tensor_module.__file__))
        if record != {"path": tensor_expected[1], "bytes": tensor_expected[2], "sha256": tensor_expected[3]}:
            raise RuntimeError("Scene backend tensor API whole-source identity mismatch")
        records["tensor_API"] = record
        for key, name, first_line in (("simulation_view", "SimulationView", 114),
                ("rigid_body_view", "RigidBodyView", 3249)):
            classes[key] = getattr(tensor_module, name)
            function = classes[key].__dict__["__init__"]
            if (classes[key].__module__ != tensor_expected[0] or classes[key].__qualname__ != name
                    or function.__module__ != tensor_expected[0]
                    or function.__qualname__ != name + ".__init__"
                    or Path(function.__code__.co_filename).resolve() != Path(tensor_expected[1])
                    or function.__code__.co_firstlineno != first_line):
                raise RuntimeError("Scene backend tensor class origin mismatch: " + name)
        descriptor = classes["simulation_view"].__dict__["is_valid"]
        if type(descriptor) is not property:
            raise RuntimeError("SimulationView validity must use the pinned public property")
        function = descriptor.fget
        if (function.__module__ != tensor_expected[0] or function.__qualname__ != "SimulationView.is_valid"
                or Path(function.__code__.co_filename).resolve() != Path(tensor_expected[1])
                or function.__code__.co_firstlineno != 136):
            raise RuntimeError("SimulationView validity property origin mismatch")
        backend = classes["lab"].get_scene_data_backend()
        report["terminal_scene_backend_before"] = {
            "backend_is_exact_class": type(backend) is classes["scene_backend"],
            "backend_is_manager_existing_backend": backend is classes["lab"]._scene_data_backend,
            "backend_module": type(backend).__module__, "backend_class": type(backend).__qualname__,
            "simulation_view_present": None, "rigid_body_view_present": None}
        if (type(backend) is not classes["scene_backend"]
                or backend is not classes["lab"]._scene_data_backend):
            raise RuntimeError("Expected this canonical manager's existing exact scene backend")
        simulation_view, rigid_body_view = backend.simulation_view, backend._rigid_body_view
        report["terminal_scene_backend_before"].update(
            simulation_view_present=simulation_view is not None,
            rigid_body_view_present=rigid_body_view is not None,
            simulation_view_module=type(simulation_view).__module__,
            simulation_view_class=type(simulation_view).__qualname__,
            rigid_body_view_module=type(rigid_body_view).__module__ if rigid_body_view is not None else None,
            rigid_body_view_class=type(rigid_body_view).__qualname__ if rigid_body_view is not None else None)
        if type(simulation_view) is not classes["simulation_view"]:
            raise RuntimeError("Expected an actually retained exact backend SimulationView; no view count is invented")
        if rigid_body_view is not None and type(rigid_body_view) is not classes["rigid_body_view"]:
            raise RuntimeError("Existing backend rigid cache must be the exact pinned RigidBodyView")
        validity = simulation_view.is_valid
        if type(validity) is not bool:
            raise RuntimeError("Actual public SimulationView validity must be a boolean")
        report["terminal_scene_backend_before"].update(
            simulation_view_is_valid=validity,
            simulation_view_native_backend_module=type(simulation_view._backend).__module__,
            simulation_view_native_backend_class=type(simulation_view._backend).__qualname__,
            rigid_body_view_native_backend_module=type(rigid_body_view._backend).__module__ if rigid_body_view is not None else None,
            rigid_body_view_native_backend_class=type(rigid_body_view._backend).__qualname__ if rigid_body_view is not None else None,
            rigid_body_view_is_valid=None,
            rigid_body_view_validity_reason="Pinned RigidBodyView exposes no public is_valid property; not queried")
        scene_backend_view_refs = (weakref.ref(simulation_view),)
        if rigid_body_view is not None:
            scene_backend_view_refs += (weakref.ref(rigid_body_view),)
        report["terminal_scene_backend_existing_view_count"] = len(scene_backend_view_refs)
        # The only new SDK write is this public property. Its pinned setter clears
        # both cached fields; no private field assignment or lazy view creation.
        backend.simulation_view = None
        report["terminal_scene_backend_after"] = {
            "simulation_view_is_none": backend.simulation_view is None,
            "rigid_body_view_is_none": backend._rigid_body_view is None}
        if not all(report["terminal_scene_backend_after"].values()):
            raise RuntimeError("Public scene backend setter did not clear both existing view aliases")
        report["terminal_scene_backend_release_returned"] = True
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
        return scene_backend_view_refs
    finally:
        # An import error traceback may retain this helper frame; leave no class aliases.
        classes.clear()
        owned_sim = owned_manager = resolved_manager = manager_proxy = module = method = function = descriptor = None
        backend = simulation_view = rigid_body_view = tensor_module = validity = scene_backend_view_refs = None


def _release_owned_asset_simulation_views(owned_sim, assets, report):
    """Terminal-only instance compatibility release; no SDK source/method patch.

    All four assets and four data objects must independently retain the same
    existing backend view. Validate every owner before assigning any field.
    Root-view presence is observed, never assumed from callback ordering.
    """
    expected = {
        "context": ("isaaclab.sim.simulation_context", "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py", 42425, "8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89", "SimulationContext"),
        "lab": ("isaaclab_physx.physics.physx_manager", "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py", 34974, "d83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5", "PhysxManager"),
        "articulation": ("isaaclab_physx.assets.articulation.articulation", "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py", 224858, "2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87", "Articulation"),
        "data": ("isaaclab_physx.assets.articulation.articulation_data", "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py", 95901, "6579e5b3efcd13d113cda141945cd055a4d0b86a069f0a80fc2c73c4cfe4163b", "ArticulationData"),
        "tensor_API": ("omni.physics.tensors.api", "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py", 382927, "b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903", "SimulationView")}
    origins = {
        "context": (("instance", 98, True), ("stop", 772, False), ("is_playing", 780, False), ("is_stopped", 784, False)),
        "lab": (("get_scene_data_backend", 347, True),),
        "articulation": (("__init__", 116, False), ("_initialize_impl", 3857, False), ("_invalidate_initialize_callback", 3976, False)),
        "data": (("__init__", 68, False),), "tensor_API": (("__init__", 114, False),)}
    classes, records = {}, {}
    module = method = function = descriptor = backend = simulation_view = asset = data = root_view = None
    owned_assets = owned_data = None
    try:
        report["owned_asset_simulation_views_release_returned"] = False
        report["owned_asset_simulation_view_release_counts"] = {"articulation_fields": 0, "data_fields": 0}
        for key, (module_name, filename, size, sha, classname) in expected.items():
            module = sys.modules.get(module_name)
            if module is None or not getattr(module, "__file__", None):
                raise RuntimeError("Missing already initialized owned-asset SDK source: " + key)
            record = _record(Path(module.__file__))
            if record != {"path": filename, "bytes": size, "sha256": sha}:
                raise RuntimeError("Owned-asset release SDK whole-source identity mismatch: " + key)
            classes[key] = getattr(module, classname)
            if classes[key].__module__ != module_name or classes[key].__qualname__ != classname:
                raise RuntimeError("Owned-asset release canonical class origin mismatch: " + key)
            records[key] = record
            for name, first_line, is_classmethod in origins[key]:
                descriptor = classes[key].__dict__[name]
                method = getattr(classes[key], name)
                function = method.__func__ if is_classmethod else method
                if (is_classmethod and (type(descriptor) is not classmethod or method.__self__ is not classes[key]
                        or descriptor.__func__ is not function)
                        or not is_classmethod and descriptor is not function
                        or function.__module__ != module_name or function.__qualname__ != classname + "." + name
                        or Path(function.__code__.co_filename).resolve() != Path(filename)
                        or function.__code__.co_firstlineno != first_line):
                    raise RuntimeError("Owned-asset release method/code origin mismatch: " + key + "." + name)
        module = sys.modules[expected["lab"][0]]
        classes["backend"] = module.PhysxSceneDataBackend
        if (classes["backend"].__module__ != expected["lab"][0]
                or classes["backend"].__qualname__ != "PhysxSceneDataBackend"):
            raise RuntimeError("Owned-asset release backend class origin mismatch")
        descriptor = classes["backend"].__dict__["simulation_view"]
        if type(descriptor) is not property:
            raise RuntimeError("Expected the pinned backend view property")
        function = descriptor.fget
        if (function.__module__ != expected["lab"][0]
                or function.__qualname__ != "PhysxSceneDataBackend.simulation_view"
                or Path(function.__code__.co_filename).resolve() != Path(expected["lab"][1])
                or function.__code__.co_firstlineno != 164):
            raise RuntimeError("Owned-asset release backend property/code origin mismatch")
        report["owned_asset_simulation_view_release_sources"] = records
        guard = {"context_is_exact": type(owned_sim) is classes["context"],
            "context_singleton_is_owned": classes["context"].instance() is owned_sim,
            "stop_returned": report.get("simulation_stop_returned") is True}
        report["owned_asset_simulation_view_release_before"] = guard
        if not all(guard.values()):
            raise RuntimeError("Owned-asset release requires this exact active owned context after returned stop")
        for name in ("stop", "is_playing", "is_stopped"):
            method = getattr(owned_sim, name)
            if method.__self__ is not owned_sim or method.__func__ is not classes["context"].__dict__[name]:
                raise RuntimeError("Owned context instance method binding changed: " + name)
        guard.update(is_playing=owned_sim.is_playing(), is_stopped=owned_sim.is_stopped())
        if type(guard["is_playing"]) is not bool or type(guard["is_stopped"]) is not bool:
            raise RuntimeError("Actual stopped/playing observations must be booleans")
        if guard["is_playing"] is not False or guard["is_stopped"] is not True:
            raise RuntimeError("Refuse asset view release before actual stopped state")
        if type(assets) not in (list, tuple) or len(assets) != 4:
            raise TypeError("Exactly four caller-owned vectorized assets are required")
        owned_assets = tuple(assets)
        assets_are_exact_classes = True
        for asset in owned_assets:
            if type(asset) is not classes["articulation"]:
                assets_are_exact_classes = False
        guard.update(asset_count=len(owned_assets), assets_distinct=len({id(asset) for asset in owned_assets}) == 4,
            assets_are_exact_classes=assets_are_exact_classes)
        if not guard["assets_distinct"] or not guard["assets_are_exact_classes"]:
            raise TypeError("Require four distinct exact loaded Articulation instances")
        owned_data = []
        for asset in owned_assets:
            owned_data.append(asset.__dict__.get("_data"))
        data_are_exact_classes = True
        for data in owned_data:
            if type(data) is not classes["data"]:
                data_are_exact_classes = False
        guard.update(data_count=len(owned_data), data_distinct=len({id(data) for data in owned_data}) == 4,
            data_are_exact_classes=data_are_exact_classes)
        if not guard["data_distinct"] or not guard["data_are_exact_classes"]:
            raise TypeError("Require each owned asset's distinct exact existing ArticulationData")
        backend = classes["lab"].get_scene_data_backend()
        guard.update(backend_is_exact=type(backend) is classes["backend"],
            backend_is_existing_manager_backend=backend is classes["lab"]._scene_data_backend)
        if not guard["backend_is_exact"] or not guard["backend_is_existing_manager_backend"]:
            raise RuntimeError("Require the exact canonical manager's existing backend")
        simulation_view = backend.simulation_view
        guard.update(backend_view_present=simulation_view is not None,
            backend_view_is_exact=type(simulation_view) is classes["tensor_API"],
            backend_view_type_module=type(simulation_view).__module__,
            backend_view_type_qualname=type(simulation_view).__qualname__)
        if not guard["backend_view_is_exact"]:
            raise RuntimeError("Require the actual existing exact SimulationView before backend release")
        guard["owners"] = []
        for index, (asset, data) in enumerate(zip(owned_assets, owned_data, strict=True)):
            root_view = asset.__dict__.get("_root_view")
            guard["owners"].append({"slot": index,
                "asset_type_module": type(asset).__module__, "asset_type_qualname": type(asset).__qualname__,
                "data_type_module": type(data).__module__, "data_type_qualname": type(data).__qualname__,
                "asset_has_instance_field": "_physics_sim_view" in asset.__dict__,
                "data_has_instance_field": "_physics_sim_view" in data.__dict__,
                "asset_alias_is_backend_view": asset.__dict__.get("_physics_sim_view") is simulation_view,
                "data_alias_is_backend_view": data.__dict__.get("_physics_sim_view") is simulation_view,
                "root_view_present": root_view is not None,
                "root_view_type_module": type(root_view).__module__ if root_view is not None else None,
                "root_view_type_qualname": type(root_view).__qualname__ if root_view is not None else None,
                "root_view_native_validity_queried": False})
            root_view = None
        if not all(owner[key] for owner in guard["owners"] for key in (
                "asset_has_instance_field", "data_has_instance_field",
                "asset_alias_is_backend_view", "data_alias_is_backend_view")):
            raise RuntimeError("All eight existing instance aliases must match the same backend view before any write")
        # Terminal-only PRIVATE instance-field compatibility. SDK files/classes,
        # Base state, _root_view and native buffers are never patched or queried.
        for asset, data in zip(owned_assets, owned_data, strict=True):
            asset._physics_sim_view = None
            report["owned_asset_simulation_view_release_counts"]["articulation_fields"] += 1
            data._physics_sim_view = None
            report["owned_asset_simulation_view_release_counts"]["data_fields"] += 1
        all_articulation_fields_none = all_data_fields_none = True
        for asset, data in zip(owned_assets, owned_data, strict=True):
            if asset.__dict__.get("_physics_sim_view") is not None:
                all_articulation_fields_none = False
            if data.__dict__.get("_physics_sim_view") is not None:
                all_data_fields_none = False
        report["owned_asset_simulation_view_release_after"] = {
            "all_articulation_fields_none": all_articulation_fields_none,
            "all_data_fields_none": all_data_fields_none,
            "backend_view_preserved_for_original_public_release": backend.simulation_view is simulation_view}
        if (report["owned_asset_simulation_view_release_counts"] != {"articulation_fields": 4, "data_fields": 4}
                or not all(report["owned_asset_simulation_view_release_after"].values())):
            raise RuntimeError("Owned instance release postconditions failed")
        report["owned_asset_simulation_views_release_returned"] = True
    finally:
        classes.clear()
        if owned_data is not None:
            owned_data.clear()
        owned_sim = assets = owned_assets = owned_data = asset = data = root_view = simulation_view = backend = None
        module = method = function = descriptor = None


def _verify_overlay_inputs(report):
    """Admit only the root-built complete copy; this starts no SDK or model."""
    manifest_record = _record(OVERLAY_MANIFEST)
    if manifest_record["sha256"] != OVERLAY_MANIFEST_SHA:
        raise RuntimeError("Root-built native overlay manifest changed")
    manifest = json.loads(OVERLAY_MANIFEST.read_text())
    if (manifest["extension_path"] != str(OVERLAY)
            or manifest["different_files"] != ["bin/libisaacsim.core.simulation_manager.plugin.so"]
            or manifest["status"] != "built_not_runtime_verified"):
        raise RuntimeError("Unexpected independently built overlay contract")
    expected_paths = set()
    for expected in manifest["copied_file_records"]:
        path = OVERLAY / expected["relative_path"]
        if OVERLAY.resolve() not in path.resolve().parents:
            raise RuntimeError("Overlay input escapes its owned directory")
        actual = _record(path)
        if actual != {key: expected[key] for key in ("path", "bytes", "sha256")}:
            raise RuntimeError("Overlay copy or plugin changed: " + expected["relative_path"])
        if path.stat().st_mode & 0o222 or path.stat().st_nlink != 1:
            raise RuntimeError("Overlay inputs must be independent read-only files")
        expected_paths.add(path.resolve())
    if expected_paths != {path.resolve() for path in OVERLAY.rglob("*") if path.is_file()}:
        raise RuntimeError("Unexpected additional or missing overlay input")
    report["native_overlay_input_manifest"] = manifest_record
    report["native_overlay_inputs_verified"] = True


def _verify_loaded_overlay(report):
    """Record exact public loader origins; release local native service aliases."""
    kit_app = extension_manager = settings = module = None
    try:
        import carb.settings
        import omni.kit.app
        settings = carb.settings.get_settings()
        dev_paths = settings.get("/app/exts/devPaths")
        if list(dev_paths or ()) != [str(OVERLAY)]:
            raise RuntimeError("Kit did not apply the explicit overlay development path")
        kit_app = omni.kit.app.get_app()
        extension_manager = kit_app.get_extension_manager()
        extension_id = extension_manager.get_enabled_extension_id("isaacsim.core.simulation_manager")
        if not extension_id:
            raise RuntimeError("Simulation-manager extension is not enabled")
        actual_path = Path(extension_manager.get_extension_path(extension_id)).resolve()
        if actual_path != OVERLAY.resolve():
            raise RuntimeError("SDK selected another simulation-manager extension")
        origins = {}
        for name, relative in (
                ("isaacsim.core.simulation_manager.impl.simulation_manager", "isaacsim/core/simulation_manager/impl/simulation_manager.py"),
                ("isaacsim.core.simulation_manager.impl.extension", "isaacsim/core/simulation_manager/impl/extension.py"),
                ("isaacsim.core.simulation_manager.bindings._simulation_manager", "isaacsim/core/simulation_manager/bindings/_simulation_manager.cpython-312-x86_64-linux-gnu.so")):
            module = sys.modules.get(name)
            if module is None or Path(module.__file__).resolve() != OVERLAY / relative:
                raise RuntimeError("Simulation-manager module origin differs: " + name)
            origins[name] = _record(Path(module.__file__))
        mapped_paths = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].endswith("libisaacsim.core.simulation_manager.plugin.so"):
                mapped_paths.add(parts[5])
        plugin = OVERLAY / "bin/libisaacsim.core.simulation_manager.plugin.so"
        if mapped_paths != {str(plugin)}:
            raise RuntimeError("Expected only the independently built native plugin mapping")
        report["native_overlay_loaded"] = {"extension_id": extension_id,
            "extension_path": str(actual_path), "dev_paths": list(dev_paths),
            "module_origins": origins, "plugin_mapped_paths": sorted(mapped_paths),
            "plugin": _record(plugin)}
        report["native_overlay_loaded_verified"] = True
    finally:
        kit_app = extension_manager = settings = module = None


def _run(args, report, archives):
    # Bootstrap is the only SDK exception before App. No physics/asset/Torch imports precede it.
    _verify_overlay_inputs(report)
    sys.path.insert(0, str(ROOT))
    if "torch" in sys.modules:
        raise RuntimeError("Run this entry in a fresh process; launcher must precede Torch imports")
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    launcher = import_simulator_before_torch("isaaclab")
    app, sim, scene, asset = None, None, None, None
    assets, views, contact_view_refs = [], [], []
    scene_backend_view_refs = ()
    try:
        app = launcher({"headless": True, "device": args.device, "enable_cameras": False,
            "fast_shutdown": False, "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false " + OVERLAY_KIT_ARGS}).app
        report["driver_app_constructors_returned"] = 1
        _verify_loaded_overlay(report)
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
                _release_owned_asset_simulation_views(sim, assets, report)
            except Exception as error:
                errors.append(f"sim.stop/owned asset view release: {type(error).__name__}: {error}")
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
                _release_terminal_timeline_caches(sim, report, phase="before_clear")
                sim.clear_instance()
                if type(sim).instance() is not None:
                    raise RuntimeError("Simulation singleton remains after clear_instance returned")
                report["simulation_clear_instance_returned"] = True
            except Exception as error:
                errors.append(f"sim.clear_instance: {type(error).__name__}: {error}")
        try:
            if sim is None:
                raise RuntimeError("Terminal Timeline compatibility requires an owned simulation context")
            scene_backend_view_refs = _release_terminal_timeline_caches(sim, report, phase="after_clear")
        except Exception as error:
            errors.append(f"terminal Timeline compatibility: {type(error).__name__}: {error}")
        try:
            # Manager/service shutdown can make additional traceback cycles unreachable.
            # Collect them before unloading tensor extensions, even when clear raised.
            report["post_context_clear_gc_collected"] = gc.collect()
            report["post_context_clear_gc_returned"] = True
            if (report["terminal_scene_backend_release_returned"] is not True
                    or type(scene_backend_view_refs) is not tuple
                    or len(scene_backend_view_refs) != report["terminal_scene_backend_existing_view_count"]
                    or not scene_backend_view_refs
                    or any(type(ref) is not weakref.ReferenceType for ref in scene_backend_view_refs)):
                raise RuntimeError("Expected the actual released backend view weakrefs before App unload")
            report["scene_backend_views_live_after_post_context_gc"] = sum(
                ref() is not None for ref in scene_backend_view_refs)
            if report["scene_backend_views_live_after_post_context_gc"]:
                raise RuntimeError("Scene backend native tensor wrappers remain alive before App unload")
            report["terminal_scene_backend_weakrefs_released"] = True
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
        "owned_manager_normalization_returned": False,
        "terminal_scene_backend_release_returned": False, "terminal_scene_backend_weakrefs_released": False,
        "owned_asset_simulation_views_release_returned": False,
        "owned_asset_simulation_view_release_counts": {"articulation_fields": 0, "data_fields": 0},
        "terminal_scene_backend_existing_view_count": None, "scene_backend_views_live_after_post_context_gc": None,
        "owned_contact_view_count": 0, "owned_contact_views_live_after_release": None,
        "NN_constructed": False, "checkpoint_loaded": False, "training_executed": False,
        "author_experiment_reproduced": False}
    archives = {}
    try:
        _run(args, report, archives)
        if (report["cleanup_errors"] or not report["owned_manager_normalization_returned"]
                or not report["simulation_clear_instance_returned"]
                or not report["owned_native_references_release_returned"] or not report["post_context_clear_gc_returned"]
                or not report["core_terminal_shutdown_returned"] or not report["terminal_timeline_cache_compat_returned"]
                or not report["owned_asset_simulation_views_release_returned"]
                or report["owned_asset_simulation_view_release_counts"] != {"articulation_fields": 4, "data_fields": 4}
                or not report["terminal_scene_backend_release_returned"]
                or not report["terminal_scene_backend_weakrefs_released"]
                or report["scene_backend_views_live_after_post_context_gc"] != 0):
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
