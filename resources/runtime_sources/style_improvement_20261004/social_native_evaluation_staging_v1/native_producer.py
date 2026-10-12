"""Thin native-world boundary around the immutable raw social producer.

Stdlib only at import. Construction requires an already existing original
AMPWorld; this module never creates, steps or resets a simulator or policy.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import stat
import sys
from types import ModuleType

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
BASE = ROOT / "resources/runtime_sources/style_improvement_20261004/social_evaluation_observation_staging_v1/observation_producer.py"
BASE_BYTES = 17810
BASE_SHA = 'e1c0eba40a895c4bcfe06c3fb1823ad961f29ca4e29cd922b34df021d715d4f0'
SCHEMA = "local.neutral_smpl.raw_social_native_evaluation_observations.v1"
PINS = {
    "native_deployment": ("CrowdSim/humanoid/application/native_deployment.py", 5610, "af708cc18873424ed57b497473f35d17cecc2c3910b15f4c67ef4d8cd5ab49f0"),
    "provider": ("CrowdSim/humanoid/runtime/dynbench_scene.py", 53151, "c49282b44d38e65b92cbf71fa1909c3e87f551e8494793aa31c5f68246ac7a73"),
    "grid": ("CrowdSim/humanoid/runtime/native_grid_scene.py", 41778, "d77e9b5702d47765e045a03f6e79f770c313dc8d4d0ad9e969aead81d40024f5"),
    "live": ("CrowdSim/humanoid/runtime/dynbench_live.py", 13565, "3d9db0f3ee00dacb458950cdc8551887b599510c06a0cc93d4922c7738ead692"),
}
OLD_LIFECYCLE = '''        if (type(world.num_envs) is not int or world.num_envs not in (1, 4)
                or type(world.shared_world) is not bool or world.shared_world != (world.num_envs == 4)
                or world.dt != 1 / 30):
            raise ValueError("Require N1 isolated tracking or N4 shared evaluation at30Hz")
'''
NEW_LIFECYCLE = '''        if (type(world.num_envs) is not int or not 3 <= world.num_envs <= 15
                or world.shared_world is not True or world.dt != 1 / 30
                or world.scene_provider is None or world.validation_scene is None
                or world.validation_scene.provider is not world.scene_provider):
            raise ValueError("Require N3..15 in one original shared native AMPWorld at30Hz")
'''


def _record(path, *, readonly=False):
    path = Path(path).absolute()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or readonly and before.st_mode & 0o222:
        raise ValueError("Expected ordinary single-link source/resource: " + str(path))
    data = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
    if any(getattr(before, name) != getattr(after, name) for name in fields) or len(data) != before.st_size:
        raise ValueError("File changed while reading: " + str(path))
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_EXECUTION_RECORD = _record(Path(__file__))


def _sources():
    base = _record(BASE, readonly=True)
    if base["bytes"] != BASE_BYTES or base["sha256"] != BASE_SHA:
        raise ValueError("Immutable social producer base changed")
    records = {"original_producer": base}
    for key, (relative, size, sha) in PINS.items():
        row = _record(ROOT / relative)
        if row["bytes"] != size or row["sha256"] != sha:
            raise ValueError("Native production source changed: " + key)
        records[key] = row
    if _record(Path(__file__)) != _EXECUTION_RECORD:
        raise ValueError("Native execution adapter changed after import")
    records["execution_adapter"] = dict(_EXECUTION_RECORD)
    return records


def _derive(source):
    if source.count(OLD_LIFECYCLE) != 1:
        raise ValueError("Expected exactly one original lifecycle fragment")
    derived = source.replace(OLD_LIFECYCLE, NEW_LIFECYCLE)
    if derived.count(NEW_LIFECYCLE) != 1 or derived.replace(NEW_LIFECYCLE, OLD_LIFECYCLE) != source:
        raise ValueError("Native producer must reverse to every original byte")
    before, after = ast.parse(source), ast.parse(derived)
    def methods(tree):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RawSocialEvaluationObservations")
        return {node.name: ast.dump(node, include_attributes=False) for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name != "_capture"}
    if methods(before) != methods(after):
        raise ValueError("An original observation/reset/clock method changed")
    return derived, {"replacement_count": 1, "whole_source_reverse_equal": True,
                     "all_other_method_AST_equal": True, "derived_sha256": hashlib.sha256(derived.encode()).hexdigest()}


def _actual_class(value, key, name):
    module = sys.modules.get(type(value).__module__)
    if (module is None or Path(getattr(module, "__file__", "")).resolve() != ROOT / PINS[key][0]
            or type(value) is not getattr(module, name, None)):
        raise TypeError("Actual native class origin differs: " + key)


class RawSocialNativeEvaluationObservations:
    """Native binding facade; original producer owns every observation/episode."""

    @classmethod
    def from_world(cls, world, actor_ids, scene_ids, *, native_request,
                   terrain_sampler, terrain_sampler_semantics, policy_description,
                   grid_center="root", grid_heading="root", height_reference="static_ground", include_ego=False):
        instance = cls.__new__(cls)
        instance.sources = _sources()
        request_keys = {"schema", "configuration", "configuration_file", "preflight", "native_map", "probe_script_snapshot",
                        "actors", "tensor_arrays_validated", "SDK_started", "human_navigation_validated"}
        if (type(native_request) is not dict or set(native_request) != request_keys
                or native_request["schema"] != "native_scene_deployment_request.v1"
                or any(native_request[key] is not False for key in
                       ("tensor_arrays_validated", "SDK_started", "human_navigation_validated"))
                or type(native_request["actors"]) is not int or not 3 <= native_request["actors"] <= 15
                or type(world.num_envs) is not int or world.num_envs != native_request["actors"]
                or world.shared_world is not True):
            raise ValueError("Use the exact original native preflight for N3..15 shared actors")
        instance._request = deepcopy(native_request)
        instance._resources = [instance._request[key] for key in ("configuration_file", "preflight", "native_map")]
        if instance._request["probe_script_snapshot"] is not None:
            instance._resources.append(instance._request["probe_script_snapshot"])
        for row in instance._resources:
            if _record(row["path"]) != row:
                raise ValueError("Registered native resource changed")
        import yaml  # Existing native preflight/driver dependency; no SDK/tensor import.
        if yaml.safe_load(Path(instance._request["configuration_file"]["path"]).read_text()) != instance._request["configuration"]:
            raise ValueError("Native request configuration differs from its bound actual scene file")
        provider, scene = world.scene_provider, world.validation_scene
        _actual_class(provider, "provider", "DynBenchSceneProvider")
        _actual_class(scene, "grid", "NativeGridScene")
        if scene.provider is not provider:
            raise ValueError("Native navigation scene and world must own the same provider")
        world_module = sys.modules.get(type(world).__module__)
        if (world_module is None or getattr(terrain_sampler, "__self__", None) is not world
                or getattr(terrain_sampler, "__func__", None) is not world_module.AMPWorld.height_at):
            raise ValueError("Native features must use the original world.height_at proxy exactly once")
        instance.world, instance._provider, instance._scene = world, provider, scene
        instance._dynamic_roots = [f"/World/envs/env_{actor}/Robot" for actor in range(world.num_envs)]
        source = BASE.read_bytes()
        if len(source) != BASE_BYTES or hashlib.sha256(source).hexdigest() != BASE_SHA:
            raise ValueError("Producer base changed before derivation")
        derived, instance.derivation = _derive(source.decode())
        backend = ModuleType("_raw_social_native_derived_producer")
        backend.__file__ = str(Path(__file__).resolve())  # The executed adapter owns _EXECUTION_RECORD.
        exec(compile(derived, backend.__file__, "exec"), backend.__dict__)
        instance._core = backend.RawSocialEvaluationObservations.from_world(world, actor_ids, scene_ids,
            terrain_sampler=terrain_sampler, terrain_sampler_semantics=terrain_sampler_semantics,
            policy_description=policy_description, grid_center=grid_center, grid_heading=grid_heading,
            height_reference=height_reference, include_ego=include_ego)
        instance._base_describe = instance._core.describe
        instance._origins = world.origins.detach().clone()
        expected = instance._core.torch.tensor(native_request["configuration"]["origins_xy"],
            dtype=instance._origins.dtype, device=instance._origins.device)
        if (tuple(expected.shape) != (world.num_envs, 2) or not instance._core.torch.equal(expected, instance._origins)
                or not instance._core.torch.equal(scene.origins, instance._origins)):
            raise ValueError("Use exactly registered absolute native spawn origins; no env-origin addition")
        instance._guard()
        instance._core.describe = instance.describe  # Metadata only; original observe invokes its unchanged method.
        return instance

    def _guard(self):
        before = self._core._clock()
        if (self.world.scene_provider is not self._provider or self.world.validation_scene is not self._scene
                or self._scene.provider is not self._provider
                or not self._core.torch.equal(self.world.origins, self._origins)):
            raise ValueError("Caller replaced native provider/scene/absolute origins")
        description = self._provider.describe()
        live = self.world.native_scene_live_observation
        if (description["scene"] != self._request["configuration"]["scene"]
                or description["preflight"] != self._request["preflight"]
                or description["native_map_evidence"] != self._request["native_map"]
                or description["live_query_bound"] is not True
                or description["exact_dynamic_robot_roots"] != self._dynamic_roots
                or type(live) is not dict or live.get("status") != "passed_live_stage_and_sampled_native_floor_ownership"
                or live.get("observed_current_singleton_context_identity") is not True
                or live.get("observed_active_stage_matches_context") is not True
                or live.get("reset_origins_tested") != self.world.num_envs
                or live.get("exact_dynamic_robot_roots") != self._dynamic_roots
                or getattr(self._provider._counter, "__self__", None) is not self._core.sim):
            raise ValueError("Actual native map/query/live-world binding differs from the selected request")
        if _sources() != self.sources:
            raise ValueError("Native execution sources changed")
        self._core._unchanged(before)

    def begin_episode(self, actor_rows=None):
        try:
            self._guard()
            self._core.begin_episode(actor_rows)
            self._guard()
        except BaseException:
            self._core._faulted = True
            raise

    def observe(self, trajectory_world_xy):
        try:
            self._guard()
            packet = self._core.observe(trajectory_world_xy)
            self._guard()
            return packet
        except BaseException:
            self._core._faulted = True
            raise

    def describe(self):
        original = self._base_describe()
        return {**original, "schema": SCHEMA, "original_observation_schema": original["schema"],
            "native_adapter_sources": deepcopy(self.sources), "native_derivation": dict(self.derivation),
            "native_request": deepcopy(self._request),
            "native_lifecycle": "original AMPWorld per-actor resets; no b608 grouped-training World",
            "absolute_origins": "registered world coordinates; no additional env_origins translation",
            "static_height_feature": "absolute admitted-grid floor/floor+1.8m blocked/unknown/exterior proxy, not measured tops",
            "ground_reference": "original current native floor-only ray query; finite-floor exterior fails",
            "native_physics_or_new_model_quality_validated": False}

    def verify_resources(self):
        """Final caller gate; no tensor reads, stepping or resource writes."""
        for row in self._resources:
            if _record(row["path"]) != row:
                raise ValueError("Registered native resource changed during evaluation")
        self._guard()
