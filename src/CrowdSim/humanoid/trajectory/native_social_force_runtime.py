"""Source-bound N3..15 SFM bridge for the existing native navigation World.

Force equations and joint integration retain social_force.py. Only the map
source, actor count and native callback admission extend the N4 fixture bridge.
CPU snapshots validate prediction inputs; they do not validate SDK navigation.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np

from .social_force import SFMParameters, SocialForceTrajectoryGenerator, SourceBinding, StaticDistanceGrid
from .social_force_runtime import SocialForceSharedPlanner, file_identity

ROOT = repository_root(__file__)
HISTORICAL_PINS = {
    "fixture_bridge": ("CrowdSim/humanoid/trajectory/social_force_runtime.py", "2a476693543d871c678cbb4cc56b35aff9331199a34b97232332974cac116676"),
    "force_math": ("CrowdSim/humanoid/trajectory/social_force.py", "936b1c9ed2d0328b5b3dc4799a2607ce385d8061a6e48c4c2b063e8c47ef191e"),
    "deployment": ("CrowdSim/humanoid/application/native_deployment.py", "af708cc18873424ed57b497473f35d17cecc2c3910b15f4c67ef4d8cd5ab49f0"),
    "provider": ("CrowdSim/humanoid/runtime/dynbench_scene.py", "c49282b44d38e65b92cbf71fa1909c3e87f551e8494793aa31c5f68246ac7a73"),
    "grid": ("CrowdSim/humanoid/runtime/native_grid_scene.py", "d77e9b5702d47765e045a03f6e79f770c313dc8d4d0ad9e969aead81d40024f5"),
}


# Version the live source boundary separately from sealed October 6 evidence.
# These are exact approved current implementation hashes, never permissive
# fallbacks. Old reports retain HISTORICAL_PINS and cannot certify this version.
SOURCE_VERSION = "native_sfm.current_scene_sources.v2"
PINS = {
    **HISTORICAL_PINS,
    "deployment": (HISTORICAL_PINS["deployment"][0], "a3a8097e5d63b03e5f96f24e6246c333b2103f3628c6209139e2aff2c564cd96"),
    "fixture_bridge": (HISTORICAL_PINS["fixture_bridge"][0], "10f403e35ed717cd755313a4b9411304d4b58fa52786761eed9afafd55a8469f"),
    "provider": (HISTORICAL_PINS["provider"][0], "68bd2171ad9707c3f280c374d73538b8c532764a42e1f90335c6375090595d28"),
    "grid": (HISTORICAL_PINS["grid"][0], "b5563bd5a509cdb02ae33e57ad34ab7a0524825ad16e481ef631531d59308165"),
}


def _restore_fixture_location(source):
    """Prove that the N4 parent changed only its repository lookup location."""
    replacements = (
        ("from protomotions.repository import repository_root\n\n", ""),
        ("ROOT = repository_root(__file__)", "ROOT = Path(__file__).resolve().parents[3]"),
    )
    original = source
    for current, historical in replacements:
        if original.count(current) != 1:
            raise ValueError("Ambiguous source-bound fixture location derivation")
        original = original.replace(current, historical, 1)
    if hashlib.sha256(original.encode()).hexdigest() != HISTORICAL_PINS["fixture_bridge"][1]:
        raise ValueError("Fixture location inverse did not restore exact historical source")
    return original


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    array = np.asarray(value)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes(order="C")).hexdigest()


def source_bindings():
    result = {}
    for role, (name, sha) in PINS.items():
        selected = file_identity(ROOT / name)
        if selected["sha256"] != sha:
            raise ValueError("Native SFM dependency changed: " + role)
        result[role] = selected
    _restore_fixture_location((ROOT / PINS["fixture_bridge"][0]).read_text())
    result["implementation"] = file_identity(__file__)
    result["source_version"] = SOURCE_VERSION
    result["historical_sha256"] = {role: row[1] for role, row in HISTORICAL_PINS.items()}
    return result


def derive_actor_methods(source):
    """Reversible count-only edits of the exact N4 input/output methods.

    The extra accepted option is the native scene's existing collision_cost
    callback; this SFM arm validates its ownership but applies no DP objective.
    """
    if hashlib.sha256(source.encode()).hexdigest() != PINS["fixture_bridge"][1]:
        raise ValueError("Require the exact source-bound N4 planner bridge")
    historical = _restore_fixture_location(source)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "SocialForceSharedPlanner")
    methods = [textwrap.dedent(ast.get_source_segment(source, node)) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in {"predict_snapshot", "plan"}]
    original = "\n\n".join(methods) + "\n"
    historical_cls = next(node for node in ast.parse(historical).body
                          if isinstance(node, ast.ClassDef) and node.name == "SocialForceSharedPlanner")
    historical_methods = [textwrap.dedent(ast.get_source_segment(historical, node))
                          for node in historical_cls.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"predict_snapshot", "plan"}]
    if original != "\n\n".join(historical_methods) + "\n":
        raise ValueError("Fixture location migration changed exact actor method bytes")
    replacements = [
        ("states.shape != (4, 5)", "states.shape != (self.actor_count, 5)"),
        ("waypoints.shape != (4, 10, 2)", "waypoints.shape != (self.actor_count, 10, 2)"),
        ("Full ordered four-actor FP32 current states", "Full ordered native-scene FP32 current states"),
        ("torch.zeros(4, dtype=torch.int64", "torch.zeros(self.actor_count, dtype=torch.int64"),
        ("history.shape != (4, 31, 5)", "history.shape != (self.actor_count, 31, 5)"),
        ("neighbors.shape != (4, 3, 31, 5)", "neighbors.shape != (self.actor_count, self.actor_count - 1, 31, 5)"),
        ("raster.shape != (4, 2, 224, 224)", "raster.shape != (self.actor_count, 2, 224, 224)"),
        ("history_mask.shape != (4, 31)", "history_mask.shape != (self.actor_count, 31)"),
        ("neighbor_mask.shape != (4, 3, 31)", "neighbor_mask.shape != (self.actor_count, self.actor_count - 1, 31)"),
        ("range(4) if peer != actor] for actor in range(4)",
         "range(self.actor_count) if peer != actor] for actor in range(self.actor_count)"),
        ('"obstacle_centers_xy", "obstacle_sizes_xy"}',
         '"obstacle_centers_xy", "obstacle_sizes_xy", "scene_collision_cost_fn"}'),
        ("scene_ids.shape != (4,)", "scene_ids.shape != (self.actor_count,)"),
    ]
    derived = original
    for old, new in replacements:
        if derived.count(old) != 1:
            raise ValueError("Ambiguous native actor-count derivation")
        derived = derived.replace(old, new, 1)
    inverse = derived
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous native actor-count inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != original:
        raise ValueError("Native actor methods did not restore complete original method bytes")
    compile(derived, "source_bound_native_SFM_actor_methods", "exec")
    return derived, {"parent": source_bindings()["fixture_bridge"], "replacement_count": len(replacements),
                     "exact_inverse": True, "changed_methods": ["predict_snapshot", "plan"],
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "force_integration_reference_math_changed": False, "SDK_executed": False,
                     "source_version": SOURCE_VERSION, "fixture_location_exact_inverse": True,
                     "historical_fixture_sha256": HISTORICAL_PINS["fixture_bridge"][1],
                     "historical_actor_method_bytes_equal": True}


def native_distance_grid(labels, *, origin_xy, bounds_xy, resolution_m, source):
    """Metric EDT of exact admitted ROS labels; unknown and exterior blocked.

    This preserves the public SFM distance-to-occupied-cell-center convention.
    NativeGridScene's separate conservative signed-distance proxy is used only
    by the DP arm. Neither field proves full-body contact clearance.
    """
    from scipy.ndimage import distance_transform_edt

    origin, bounds = np.asarray(origin_xy, np.float64), np.asarray(bounds_xy, np.float64)
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8 or labels.ndim != 2
            or min(labels.shape) < 2 or not np.isin(labels, (0, 205, 254)).all()
            or type(resolution_m) is not float or resolution_m != .1
            or origin.shape != (2,) or bounds.shape != (4,)
            or not np.isfinite(origin).all() or not np.isfinite(bounds).all()
            or not np.allclose(bounds[:2], origin, rtol=0, atol=2e-4)
            or not np.allclose(bounds[2:] - origin, np.array(labels.shape[::-1]) * resolution_m,
                               rtol=0, atol=2e-4)):
        raise ValueError("Require the exact finite native .1m ROS frame and occupied/unknown/free labels")
    distance = distance_transform_edt(np.pad(labels == 254, 1, constant_values=False),
                                      sampling=resolution_m)[1:-1, 1:-1]
    return StaticDistanceGrid(distance, labels, origin_xy=origin, resolution_m=resolution_m,
                              source_id=source["path"], source_sha256=source["sha256"])


_method_source, _method_proof = derive_actor_methods((ROOT / PINS["fixture_bridge"][0]).read_text())
_method_namespace = {"np": np, "SimpleNamespace": SimpleNamespace}
exec(compile(_method_source, "source_bound_native_SFM_actor_methods", "exec"), _method_namespace)


class NativeSocialForcePlanner(SocialForceSharedPlanner):
    """One complete native scene snapshot; no weight/model/World ownership."""

    predict_snapshot = _method_namespace["predict_snapshot"]
    _count_admitted_plan = _method_namespace["plan"]

    def __init__(self, *, world, scene_source, seed):
        from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
        if (type(world.num_envs) is not int or not 3 <= world.num_envs <= 15
                or world.shared_world is not True or world.scene_provider is None
                or world.validation_scene is None or world.validation_scene.provider is not world.scene_provider):
            raise ValueError("Require one original shared native N3..15 navigation World")
        request = preflight_native_scene(scene_source, actors=world.num_envs)
        self._initialize_native(world.validation_scene, request, seed, str(world.device))
        self._world = world
        self.actual_world_boundary_checked = True
        self._guard()

    @classmethod
    def for_CPU_snapshot(cls, *, scene_source, actors, seed):
        from CrowdSim.humanoid.application.native_deployment import make_native_scene_factory, preflight_native_scene
        if type(actors) is not int or not 3 <= actors <= 15:
            raise ValueError("Native SFM protocol requires N3..15")
        request = preflight_native_scene(scene_source, actors=actors)
        scene = make_native_scene_factory(request)(request["configuration"], device="cpu")
        instance = cls.__new__(cls)
        instance._initialize_native(scene, request, seed, "cpu")
        instance._world = None
        instance.actual_world_boundary_checked = False
        instance._guard()
        return instance

    def _initialize_native(self, scene, request, seed, device):
        from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
        from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
        if type(scene) is not NativeGridScene or type(scene.provider) is not DynBenchSceneProvider:
            raise ValueError("Require concrete admitted native scene/provider; no geometry projection")
        self.actor_count = request["actors"]
        self.device = device
        self._scene, self._provider = scene, scene.provider
        self._request = deepcopy(request)
        self._request_hash = _json_hash(self._request)
        self._sources = source_bindings()
        self._scene_hashes = self._scene_arrays()
        self.scene_identity = deepcopy(request["configuration_file"])
        self.generator = SocialForceTrajectoryGenerator(native_distance_grid(scene._labels,
            origin_xy=scene._origin, bounds_xy=scene._bounds, resolution_m=scene.resolution,
            source=request["native_map"]), parameters=SFMParameters(), seed=seed,
            sources=(SourceBinding.bind_sfm(ROOT / "CrowdSim/world/sfm.py"),))
        self.compiled_forward_enabled = False
        self.config = SimpleNamespace(diffusion_steps=0)
        self.last_metadata = None

    def _scene_arrays(self):
        return {name: _array_hash(getattr(self._scene, name))
                for name in ("_labels", "_origin", "_bounds", "_origins", "_routes")}

    def _guard(self):
        if (self._request_hash != _json_hash(self._request) or source_bindings() != self._sources
                or self._scene.provider is not self._provider or self._scene_arrays() != self._scene_hashes
                or self.actor_count != self._request["actors"] or self.device != str(self._scene.device)
                or self._scene.resolution != .1):
            raise ValueError("Native SFM source/count/frame/scene changed")
        for role in ("configuration_file", "preflight", "native_map", "probe_script_snapshot"):
            selected = self._request[role]
            if selected is not None and file_identity(selected["path"]) != selected:
                raise ValueError("Native SFM selected resource changed: " + role)
        self._provider.verify_sources()
        description = self._provider.describe()
        config = self._request["configuration"]
        if (description["preflight"] != self._request["preflight"]
                or description["native_map_evidence"] != self._request["native_map"]
                or description["scene"] != config["scene"]
                or not np.array_equal(self._provider._map, self._scene._labels)
                or not np.array_equal(np.asarray(self._provider._map_origin_xy), self._scene._origin)
                or not np.array_equal(np.asarray(self._provider.bounds_xy).reshape(4), self._scene._bounds)
                or self._scene.clearance_m != config["clearance_m"]
                or not np.array_equal(self._scene._origins, np.asarray(config["origins_xy"], np.float64))
                or not np.array_equal(self._scene._routes, np.asarray(config["routes_xy"], np.float64))):
            raise ValueError("Native SFM grid differs from selected scene configuration")
        if self.actual_world_boundary_checked:
            import torch
            world = self._world
            if (world is None or world.num_envs != self.actor_count or world.shared_world is not True
                    or world.scene_provider is not self._provider or world.validation_scene is not self._scene
                    or description["live_query_bound"] is not True
                    or getattr(self._provider._counter, "__self__", None) is not world.simulator._sim
                    or not torch.equal(world.origins, self._scene.origins)):
                raise ValueError("Native SFM lost the live original shared World/provider/frame boundary")
        elif self._world is not None or description["live_query_bound"] is not False:
            raise ValueError("CPU snapshot must not claim a live SDK boundary")
        self.generator.verify()

    def describe(self):
        self._guard()
        return {"schema": "local.public_sfm.shared_AMP_native_bridge.v1", "scene": self.scene_identity,
                "scene_name": self._request["configuration"]["scene"], "actors": self.actor_count,
                "sources": deepcopy(self._sources), "method_derivation": deepcopy(_method_proof),
                "generator": self.generator.describe(), "native_resources": deepcopy(self._request),
                "map": "Exact admitted ROS labels; metric EDT; unknown205/occupied0/exterior blocked; no AABB projection",
                "raw_output": "51XY knots at0,.1,..,5s; original AMP driver applies common causal reference",
                "goal": "Registered static A* five-second endpoint, constant during each predicted horizon",
                "diffusion_weights_loaded": False, "diffusion_guidance_applied": False, "RNG_consumed": False,
                "selected_sample_zero_semantics": "Compatibility sentinel, not a Diffusion sample",
                "actual_world_boundary_checked": self.actual_world_boundary_checked,
                "physical_validation_established": False, "author_private_protocol": False}

    def plan(self, **inputs):
        self._guard()
        callback = inputs.get("scene_collision_cost_fn")
        if (getattr(callback, "__self__", None) is not self._scene
                or getattr(callback, "__func__", None) is not type(self._scene).collision_cost
                or any(inputs.get(name) is not None for name in
                       ("scene_bounds_xy", "obstacle_centers_xy", "obstacle_sizes_xy"))):
            raise ValueError("Require the same native scene callback and no substitute AABB geometry")
        return self._count_admitted_plan(**inputs)
