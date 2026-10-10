"""Prepared DynBench USD provider; importing it starts no SDK or tensor runtime.

Only an explicit native full-grid resource probe admits a map. Scene injection
and the pre-SimulationContext query hook are not integrated into AMPWorld yet.
The finite reference floor is an engineering choice, not an authored USD floor.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys


SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")
FLOOR_PATH = "/__PaperMapProbeReferenceGround"
SCENE_MARKER_PATH = "/__PaperDynBenchSceneProvider"
HEIGHT_BAND = (.15, 1.8)


def _finite(value, label):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _xy(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two finite world coordinates")
    return tuple(_finite(item, label) for item in value)


def _paths(values, label):
    if (not isinstance(values, list)
            or any(type(value) is not str or not value.startswith("/") or value == "/"
                   or "//" in value or ".." in value or any(c in value for c in "*?{}") for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} requires unique exact absolute USD paths")
    return tuple(values)


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _check_record(record):
    if (type(record) is not dict or set(record) != {"path", "bytes", "sha256"}
            or type(record["bytes"]) is not int or record["bytes"] < 1
            or type(record["sha256"]) is not str or len(record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sha256"])):
        raise ValueError("Source record requires exact path/positive bytes/SHA256")
    if file_record(record["path"]) != record:
        raise ValueError(f"Scene resource bytes changed: {record['path']}")


def _read_bound_json(path, expected_sha256):
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Explicit lower-case full SHA256 is required")
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError("Scene evidence SHA256 differs from the selected source")
    value = json.loads(Path(record["path"]).read_text())
    if type(value) is not dict:
        raise ValueError("Scene evidence must be a JSON object")
    return value, record


def reference_ground_geometry(bounds, ground, margin=.2):
    """Exact reviewed probe geometry: one finite two-triangle reference plane."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("Reference ground bounds require low/high XY")
    low, high = _xy(bounds[0], "low bounds"), _xy(bounds[1], "high bounds")
    ground, margin = _finite(ground, "ground height"), _finite(margin, "floor margin")
    if any(a >= b for a, b in zip(low, high)) or not .1 <= margin <= 1.:
        raise ValueError("Reference ground requires increasing bounds and .1..1m margin")
    low, high = tuple(x - margin for x in low), tuple(x + margin for x in high)
    return [[low[0], low[1], ground], [high[0], low[1], ground],
            [high[0], high[1], ground], [low[0], high[1], ground]], [3, 3], [0, 1, 2, 0, 2, 3]


def _editable_paths(stage, paths):
    """Session-only deinstancing copied from the reviewed scene-map probe."""
    changed = []
    for path in paths:
        for _ in range(32):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Missing collision edit path: {path}")
            if not prim.IsInstanceProxy():
                break
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.IsInstance() and not ancestor.IsInstanceProxy():
                    changed.append(str(ancestor.GetPath()))
                    ancestor.SetInstanceable(False)
                    break
                ancestor = ancestor.GetParent()
            else:
                raise ValueError(f"Cannot resolve editable instance root for {path}")
        else:
            raise ValueError(f"Unresolved nested instance path: {path}")
    return sorted(set(changed))


def author_reference_ground(stage, bounds, ground, disabled_paths, margin=.2):
    """Lazy USD authoring, same material/schema/triangle recipe as source probe.

    A caller must launch Kit first; this helper neither opens an App nor saves
    the source USD. Standalone USD contract tests do not prove PhysX cooking.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    points, counts, indices = reference_ground_geometry(bounds, ground, margin)
    disabled_paths = _paths(list(disabled_paths), "disabled floor paths")
    if stage.GetPrimAtPath(FLOOR_PATH):
        raise ValueError("Reference ground path already exists; no stacked floor")
    stage.SetEditTarget(stage.GetSessionLayer())
    changed = _editable_paths(stage, disabled_paths)
    for path in disabled_paths:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    mesh = UsdGeom.Mesh.Define(stage, FLOOR_PATH)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr("rightHanded")
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.002)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
    material = UsdShade.Material.Define(stage, FLOOR_PATH + "/PhysicsMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.)
    physics.CreateDynamicFrictionAttr(1.)
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    material.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    return {"path": FLOOR_PATH, "points": points, "face_counts": counts, "face_indices": indices,
            "disabled_original_floor_paths": list(disabled_paths), "deinstanced_roots": changed,
            "native_cooking_verified": False, "source_USD_saved": False}


class DynBenchSceneProvider:
    """A single shared source USD, a finite floor, and exact native hit filters.

    Construction admits only reviewed preflight bytes. It does not admit their
    projection for navigation. Native resource/map admission and a live query
    binding are explicit separate operations; no visual bounding boxes are used.
    """

    @classmethod
    def from_preflight(cls, path, *, expected_sha256):
        data, record = _read_bound_json(path, expected_sha256)
        return cls(data, record)

    def __init__(self, data, preflight_record):
        data = deepcopy(data)
        if (data.get("schema") != "dynbench_physx_map_preflight.v1" or data.get("scene") not in SCENES
                or data.get("status") != "passed" or data.get("SDK_started") is not False
                or data.get("floor_owner") != "reference_flat_terrain"
                or data.get("floor_overlay_mode") != "none" or data.get("up_axis") != "Z"
                or data.get("meters_per_unit") != 1.0 or data.get("original_composition_unchanged") is not True):
            raise ValueError("Expected a source-bound reference_flat_terrain SDK-free preflight")
        if data.get("time_sampled_xform_paths") or any(
                body.get("enabled") and not body.get("kinematic") for body in data["authored_rigid_bodies"]):
            raise ValueError("A static navigation provider cannot admit animated or dynamic source bodies")
        _check_record(preflight_record)
        resources = [data["source"], *data["layers"], *data["input_projection"].values(),
                     data["reference_ground_recipe"], data["reference_ground"]["overlay_source"], data["script"]]
        for resource in resources:
            _check_record(resource)
        metadata = data["projection_metadata"]
        bounds, ground = metadata["bounds"], metadata["ground"]
        recipe = data["reference_ground"]
        points, counts, indices = reference_ground_geometry(bounds, ground, recipe["margin_m"])
        dims = metadata["dims"]
        if (not isinstance(dims, list) or len(dims) != 2
                or any(type(value) is not int or value < 1 for value in dims)
                or any(not math.isclose(bounds[1][axis] - bounds[0][axis], dims[axis] * .1,
                                        abs_tol=1e-8, rel_tol=0) for axis in (0, 1))):
            raise ValueError("Map bounds and .1m raster dimensions disagree")
        if (recipe["path"] != FLOOR_PATH or recipe["floor_roots"] != [FLOOR_PATH]
                or data["exact_floor_collider_paths"] != [FLOOR_PATH] or recipe["points"] != points
                or recipe["face_counts"] != counts or recipe["face_indices"] != indices
                or recipe["height_m"] != ground or recipe["floor_owner"] != "reference_flat_terrain"
                or recipe["holes_steps_and_other_floors_inferred"] is not False
                or recipe["material"] != {"static_friction": 1., "dynamic_friction": 1., "restitution": 0., "combine_mode": "average"}):
            raise ValueError("Finite floor recipe differs from the reviewed geometry")
        original_floors = _paths(data["original_enabled_floor_collision_paths"], "original floor colliders")
        if tuple(recipe["disabled_visual_floor_collision_paths"]) != original_floors:
            raise ValueError("All original floor colliders must be disabled; no stacked floor")
        collider_paths = _paths([row["path"] for row in data["authored_colliders"]], "authored colliders")
        enabled = {row["path"] for row in data["authored_colliders"] if row["enabled"] is True}
        if not set(original_floors) <= enabled or FLOOR_PATH in collider_paths:
            raise ValueError("Original floor/static collision sets are inconsistent")
        self._data, self._preflight_record, self._resources = data, deepcopy(preflight_record), resources
        self.scene_name = data["scene"]
        self.floor_paths = (FLOOR_PATH,)
        self.static_paths = tuple(sorted(enabled - set(original_floors)))
        if not self.static_paths or set(self.static_paths) & set(self.floor_paths):
            raise ValueError("Exact floor and static obstacles must be nonempty and disjoint")
        self.bounds_xy = tuple(_xy(value, "map bounds") for value in bounds)
        self.ground_height_m = _finite(ground, "ground height")
        self.floor_bounds_xy = ((points[0][0], points[0][1]), (points[2][0], points[2][1]))
        self._native = self._map = self._map_origin_xy = self._query = self._counter = None
        self._spawned = self._query_hook_applied = False

    def verify_sources(self):
        for record in (self._preflight_record, *self._resources):
            _check_record(record)
        return True

    def prepare_simulation_cfg(self, configuration):
        """Required pre-create hook; the current AMPWorld does not invoke it."""
        if not hasattr(configuration, "enable_scene_query_support"):
            raise ValueError("SimulationCfg lacks its explicit scene-query support field")
        result = deepcopy(configuration)
        result.enable_scene_query_support = True
        self._query_hook_applied = True
        return result

    def prepare_scene_cfg(self, configuration):
        """Lazy post-Kit SceneCfg injection without creating a SimulationContext."""
        if "isaaclab.sim" not in sys.modules:
            raise RuntimeError("Launch Kit and import isaaclab.sim before scene injection")
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import UsdFileCfg
        if not self._query_hook_applied or getattr(configuration, "filter_collisions", None) is not False:
            raise ValueError("Explicit pre-create query hook and shared-world collision config are required")
        if hasattr(configuration, "dynbench_source"):
            raise ValueError("Scene configuration already owns a DynBench source")
        result = deepcopy(configuration)
        result.terrain = None  # One provider plane owns physics; no generated Terrain mesh.
        result.replicate_physics = False
        # The cloned robot contact sensors must not keep a deleted Terrain
        # filter. Exact native floor/static monitors remain a separate binding.
        for name, item in vars(result).items():
            if name.startswith("contact_sensor_") and hasattr(item, "filter_prim_paths_expr"):
                paths = list(item.filter_prim_paths_expr)
                item.filter_prim_paths_expr = [FLOOR_PATH if path == "/World/ground/terrain/mesh" else path for path in paths]
        result.dynbench_source = AssetBaseCfg(prim_path=SCENE_MARKER_PATH,
            spawn=UsdFileCfg(usd_path=self._data["source"]["path"], func=self._spawn))
        return result

    def _spawn(self, prim_path, cfg, translation=None, orientation=None, **kwargs):
        """Compose entire source as a session sublayer, preserving original /Root paths."""
        del cfg, kwargs
        if (prim_path != SCENE_MARKER_PATH or self._spawned or not self._query_hook_applied
                or translation not in (None, (0., 0., 0.))
                or orientation not in (None, (1., 0., 0., 0.), (0., 0., 0., 1.))):
            raise ValueError("Provider requires one untransformed global scene spawn")
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        self.verify_sources()
        source = self._data["source"]["path"]
        layer = Sdf.Layer.FindOrOpen(source)
        if layer is None or any(stage.GetPrimAtPath(str(root.path)) for root in layer.rootPrims):
            raise ValueError("Source layer roots conflict with the active stage")
        if stage.GetPrimAtPath(FLOOR_PATH) or stage.GetPrimAtPath(SCENE_MARKER_PATH):
            raise ValueError("Existing provider/floor would stack a second scene")
        source_stage = Usd.Stage.Open(layer)
        physics_paths = [str(prim.GetPath()) for prim in source_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        session = stage.GetSessionLayer()
        stage.SetEditTarget(session)
        if source in session.subLayerPaths:
            raise ValueError("DynBench source already composed")
        session.subLayerPaths.append(source)
        if stage.GetCompositionErrors():
            raise ValueError("Composed DynBench layer has USD composition errors")
        if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
            raise ValueError("Active stage is not source-bound Z-up/metres=1")
        for path in physics_paths:
            stage.GetPrimAtPath(path).SetActive(False)
        author_reference_ground(stage, self.bounds_xy, self.ground_height_m,
            self._data["original_enabled_floor_collision_paths"], self._data["reference_ground"]["margin_m"])
        self._spawned = True
        return stage.DefinePrim(SCENE_MARKER_PATH, "Xform")

    def admit_native_map(self, path, *, expected_sha256):
        """Admit only the real original probe's full static native resource gate.

        This gate proves native grid/support at its recorded snapshot, not human
        navigation quality or the new provider's future GPU physics cooking.
        """
        data, record = _read_bound_json(path, expected_sha256)
        self.verify_sources()
        if (data.get("schema") != "dynbench_physx_map_probe.v1"
                or data.get("scene") != self.scene_name or data.get("source") != self._data["source"]
                or data.get("preflight") != self._preflight_record or data.get("script") != self._data["script"]
                or data.get("status") != "scene_resource_gate_passed" or data.get("scene_resource_gate_passed") is not True
                or data.get("native_started") is not True or data.get("native_probe_completed") is not True
                or data.get("native_mapping_verified") is not True or data.get("fallback_used") is not False
                or data.get("original_source_layers_unchanged") is not True
                or data.get("projection_unknown_cells_preserved") is not True
                or data.get("floor_collider_paths") != list(self.floor_paths)
                or data.get("selected_floor_height_m") != self.ground_height_m
                or data.get("sampled_free_cells_with_overlap") != 0 or data.get("foreign_native_hit_paths") != []):
            raise ValueError("Map lacks matching real native full-grid resource admission; no projection fallback")
        support, physics, activity = data["floor_support"], data["physics"], data["dynamic_activity"]
        if (support["full_grid"] is not True or type(support["native_free_cells"]) is not int
                or support["native_free_cells"] < 1 or support["tested"] != support["native_free_cells"]
                or support["supported"] != support["tested"] or support["unsupported"] != 0
                or physics["scene_query_support"] is not True or physics["no_query_time_advance"] is not True
                or physics["timeline_at_query_s"] != physics["timeline_after_queries_s"]
                or not math.isfinite(physics["timeline_at_query_s"])
                or data["origin_verification"]["overlap_hits"] != []
                or data["origin_verification"]["floor_support"] is None
                or activity["static_snapshot_gate"] is not True):
            raise ValueError("Native map/support/static snapshot gate is incomplete")
        for artifact in (*data["exports"].values(), support["artifact"], data["native_omap"]):
            _check_record(artifact)
        map_record = data["exports"]["occupancy.yaml"]
        from CrowdSim.utils.map_metadata import load_occupancy_map_metadata
        from PIL import Image
        import numpy as np
        metadata = load_occupancy_map_metadata(map_record["path"])
        image = np.asarray(Image.open(metadata.image_path))
        if (image.ndim != 2 or image.dtype != np.uint8 or not np.isin(image, [0, 205, 254]).all()
                or metadata.resolution != .1
                or not np.allclose(metadata.origin_xy, self.bounds_xy[0], atol=2e-4, rtol=0)
                or metadata.origin_yaw != 0. or metadata.negate != 0
                or list(image.shape[::-1]) != self._data["projection_metadata"]["dims"]
                or file_record(metadata.image_path) != data["exports"]["occupancy.png"]):
            raise ValueError("Native occupancy image/coordinates differ from source-bound map")
        projection_path = self._data["input_projection"]["occupancy.png"]["path"]
        projection = np.asarray(Image.open(projection_path))
        if projection.shape != image.shape or not np.all(image[projection == 205] == 205):
            raise ValueError("Native map promoted unverified projection support to walkable")
        with np.load(support["artifact"]["path"], allow_pickle=False) as rays:
            n = support["tested"]
            if (rays["supported"].shape != (n,) or rays["supported"].dtype != np.bool_
                    or not rays["supported"].all() or not np.array_equal(rays["native_free_index"], np.arange(n))
                    or rays["native_free_index"].dtype.kind not in "iu"
                    or rays["free_world_positions"].shape != (n, 3) or not np.isfinite(rays["free_world_positions"]).all()
                    or rays["all_hit_count"].shape != (n,) or rays["all_hit_count"].dtype.kind not in "iu"
                    or not np.all(rays["all_hit_count"] >= 1)
                    or rays["hit_position"].shape != (n, 3) or rays["hit_normal"].shape != (n, 3)
                    or not np.isfinite(rays["hit_position"]).all() or not np.isfinite(rays["hit_normal"]).all()
                    or not np.all(rays["hit_normal"][:, 2] >= .9)
                    or not np.all(np.abs(rays["hit_position"][:, 2] - self.ground_height_m) <= .08)
                    or rays["collision"].shape != (n,) or not np.isin(rays["collision"], self.floor_paths).all()):
                raise ValueError("Raw full-grid support rays do not verify the exact selected floor")
        self._native = {"record": record, "data": deepcopy(data)}
        self._map = image.copy()
        self._map_origin_xy = metadata.origin_xy
        return self.describe()

    def require_navigation_map(self):
        if self._native is None:
            raise RuntimeError("DynBench navigation requires native full-grid map admission; projection/preflight is insufficient")

    def bind_native_query(self, query, *, physics_step_count):
        self.require_navigation_map()
        if not callable(getattr(query, "raycast_all", None)) or not callable(physics_step_count):
            raise ValueError("Live native raycast_all and completed physics-step counter are required")
        self._query, self._counter = query, physics_step_count

    def _heights(self, xy, *, ground_only):
        self.require_navigation_map()
        if self._query is None:
            raise RuntimeError("Height queries require explicit actual native query binding")
        import numpy as np
        torch_module = sys.modules.get("torch")
        tensor_input = torch_module is not None and isinstance(xy, torch_module.Tensor)
        if tensor_input and xy.dtype not in (torch_module.float32, torch_module.float64):
            raise ValueError("Native height tensor coordinates require float32/float64")
        points = np.asarray(xy.detach().cpu().numpy() if tensor_input else xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError("Height query requires finite world[...,2] coordinates")
        low, high = np.asarray(self.floor_bounds_xy)
        if np.any(points < low) or np.any(points > high):
            raise ValueError("Height query outside finite reference floor; no clamping")
        before = self._counter()
        if type(before) is not int or before < 0:
            raise ValueError("Actual completed physics-step counter must be a nonnegative integer")
        heights = []
        allowed = set(self.floor_paths if ground_only else (*self.floor_paths, *self.static_paths))
        z = self.ground_height_m
        for x, y in points.reshape(-1, 2):
            hits = []
            floor_hits = []
            hit_count = 0

            def collect(hit):
                nonlocal hit_count
                hit_count += 1
                collision = str(hit.collision)
                position, normal = tuple(hit.position), tuple(hit.normal)
                if (len(position) != 3 or len(normal) != 3 or not all(math.isfinite(v) for v in (*position, *normal))):
                    raise ValueError("Native height query returned nonfinite geometry")
                if collision in allowed and normal[2] >= .9:
                    if not ground_only or abs(position[2] - z) <= .08:
                        hits.append(float(position[2]))
                    if collision in self.floor_paths and abs(position[2] - z) <= .08:
                        floor_hits.append(float(position[2]))
                return True

            any_hit = self._query.raycast_all((float(x), float(y), z + 1.8), (0., 0., -1.), 2.1, collect, False)
            if bool(any_hit) != bool(hit_count):
                raise ValueError("Native ray boolean disagrees with callbacks")
            if not hits or not floor_hits:
                raise ValueError("Actual native query found no admissible exact-floor support")
            heights.append(max(hits))
        if self._counter() != before:
            raise RuntimeError("Physics advanced during native height batch")
        result = np.asarray(heights).reshape(points.shape[:-1])
        return torch_module.as_tensor(result, dtype=xy.dtype, device=xy.device) if tensor_input else result

    def ground_height_at(self, xy):
        return self._heights(xy, ground_only=True)

    def height_at(self, xy):
        return self._heights(xy, ground_only=False)

    def raster(self, states, *, device="cpu"):
        """TRACE two-channel convention, retaining 205 and outside as unknown=.5."""
        self.require_navigation_map()
        import numpy as np
        import torch
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 5 or not np.isfinite(states).all():
            raise ValueError("Diffusion map states require finite world[x,y,vx,vy,yaw] [B,5]")
        if np.any(np.abs(states[:, :2]) > 1e6):
            raise ValueError("World XY exceeds safe native-map indexing range; no trajectory clamping")
        row, col = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
        lx, ly = (col - 56) / 12, (row - 112) / 12
        cosine, sine = np.cos(states[:, 4, None, None]), np.sin(states[:, 4, None, None])
        x = states[:, 0, None, None] + cosine * lx - sine * ly
        y = states[:, 1, None, None] + sine * lx + cosine * ly
        px = np.floor((x - self._map_origin_xy[0]) / .1).astype(np.int64)
        py = self._map.shape[0] - 1 - np.floor((y - self._map_origin_xy[1]) / .1).astype(np.int64)
        valid = (px >= 0) & (px < self._map.shape[1]) & (py >= 0) & (py < self._map.shape[0])
        values = np.full(px.shape, 205, dtype=np.uint8)
        values[valid] = self._map[py[valid], px[valid]]
        free = np.where(values == 254, 1., np.where(values == 0, 0., .5)).astype(np.float32)
        return torch.as_tensor(np.stack((free, 1 - free), 1), device=device)

    def describe(self):
        return {"schema": "dynbench_reference_flat_scene_provider.v1", "status": "prepared_not_runtime_integrated",
            "scene": self.scene_name, "preflight": deepcopy(self._preflight_record),
            "source_USD": deepcopy(self._data["source"]), "floor_owner": "reference_flat_terrain",
            "floor_paths": list(self.floor_paths), "static_collider_count": len(self.static_paths),
            "floor_static_disjoint": True, "map_bounds_xy_m": self.bounds_xy, "floor_bounds_xy_m": self.floor_bounds_xy,
            "ground_height_m": self.ground_height_m, "negative_world_XY_preserved": True,
            "projection_is_navigation_admitted": False, "native_resource_map_admitted": self._native is not None,
            "native_map_evidence": None if self._native is None else deepcopy(self._native["record"]),
            "unknown_and_outside_raster_value": .5, "live_query_bound": self._query is not None,
            "pre_create_scene_query_hook_required": True, "pre_create_hook_applied": self._query_hook_applied,
            "new_provider_PhysX_cooking_verified": False, "human_navigation_quality_validated": False,
            "App_created": False, "SimulationContext_created": False, "source_USD_saved": False}
