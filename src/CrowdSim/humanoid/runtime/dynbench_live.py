"""Explicit read-only observations of an already initialized native USD world."""
from __future__ import annotations

import math
import sys


def _matches_authored_points(actual, expected):
    """Compare with the point3f values actually authored by UsdGeom.Mesh."""
    import numpy as np
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float32).astype(np.float64)
    return actual.shape == expected.shape and np.allclose(actual, expected, atol=1e-6, rtol=0)


def verify_live_runtime(provider, *, origins_xy, simulation_context, free_cell_limit=0):
    """Verify source/floor/context and all native ray hits; no SDK construction."""
    import numpy as np
    from .dynbench_scene import FLOOR_PATH, SCENE_MARKER_PATH, _active_spawn_stage, reference_ground_geometry

    provider.require_navigation_map()
    if provider._query is None or not provider._query_hook_applied:
        raise RuntimeError("Live runtime verification requires the precreate hook and actual query binding")
    if type(free_cell_limit) is not int or not 0 <= free_cell_limit <= 256:
        raise ValueError("Optional deterministic admitted-free sample limit must be an integer0..256")
    torch_module = sys.modules.get("torch")
    if torch_module is not None and isinstance(origins_xy, torch_module.Tensor):
        origins_xy = origins_xy.detach().cpu().numpy()
    origins = np.asarray(origins_xy, dtype=np.float64)
    if (origins.ndim != 2 or not 1 <= len(origins) <= 15 or origins.shape[1] != 2
            or not np.isfinite(origins).all() or np.any(np.abs(origins) > 1e6)):
        raise ValueError("Live origins require1..15 finite world[B,2] coordinates")
    cell = np.floor((origins - provider._map_origin_xy) / .1).astype(np.int64)
    rows, cols = provider._map.shape[0] - 1 - cell[:, 1], cell[:, 0]
    if (np.any(rows < 0) or np.any(rows >= provider._map.shape[0]) or np.any(cols < 0)
            or np.any(cols >= provider._map.shape[1]) or np.any(provider._map[rows, cols] != 254)):
        raise ValueError("Every live reset origin must be an admitted known-free map cell; no relocation")

    provider.verify_sources()
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics

    context_class = getattr(sim_utils, "SimulationContext", None)
    if context_class is None or not callable(getattr(context_class, "instance", None)):
        raise RuntimeError("Pinned public SimulationContext.instance is required")
    if context_class.instance() is not simulation_context:
        raise ValueError("Requested world is not the existing SDK singleton SimulationContext")
    stage = _active_spawn_stage()
    if getattr(simulation_context, "stage", None) is not stage:
        raise ValueError("Active SDK stage differs from the actual singleton context stage")
    if getattr(getattr(simulation_context, "cfg", None), "enable_scene_query_support", None) is not True:
        raise ValueError("Scene query support was not enabled on the actual context config")
    counter_getter = getattr(simulation_context, "get_physics_step_count", None)
    if not callable(counter_getter) or getattr(provider._counter, "__self__", None) is not simulation_context:
        raise ValueError("Bound native query counter must belong to the same actual SimulationContext")
    before = provider._counter()
    if type(before) is not int or before < 0 or counter_getter() != before:
        raise ValueError("Actual submitted-step counter must be a consistent nonnegative integer")
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("Live native world is not source-bound Z-up/metres1")
    from social_navigation.portable_usd import verify_owned_layers
    verify_owned_layers(provider._portable_layers, provider._portable_usd_receipt)
    source_layer = provider._runtime_source_layer
    source_path = source_layer.identifier
    session = stage.GetSessionLayer()
    if list(session.subLayerPaths).count(source_path) != 1 or not stage.GetPrimAtPath(SCENE_MARKER_PATH):
        raise ValueError("Live stage requires exactly one source session layer and provider marker")
    if stage.GetCompositionErrors():
        raise ValueError("Live source composition has USD errors")
    source_stage = Usd.Stage.Open(source_layer)
    if source_stage is None:
        raise ValueError("Cannot inspect unchanged authored source stage")
    source_roots = [str(prim.GetPath()) for prim in source_stage.GetPseudoRoot().GetChildren()]
    if not source_roots or any(not stage.GetPrimAtPath(path) for path in source_roots):
        raise ValueError("Authored source roots are missing in the actual stage")
    authored_physics = [str(prim.GetPath()) for prim in source_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if any(stage.GetPrimAtPath(path).IsActive() for path in authored_physics):
        raise ValueError("Authored source PhysicsScenes remain active alongside the singleton context")

    def dynamic(path):
        return any(path == root or path.startswith(root + "/") for root in provider._dynamic_robot_roots)

    enabled = []
    live_scenes = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene):
            live_scenes.append(path)
        if (prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() and not dynamic(path)):
            enabled.append(path)
    expected = {*provider.floor_roots, *provider.static_collision_paths}
    if set(enabled) != expected or len(enabled) != len(set(enabled)):
        raise ValueError("Actual enabled static/floor colliders differ from exact admitted paths; duplicate/foreign Terrain or floor")
    if len(live_scenes) != 1:
        raise ValueError("Live stage must have exactly one active PhysicsScene")
    for path in provider._disabled_flat_ground_paths:
        prim = stage.GetPrimAtPath(path)
        if (not prim or (prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())):
            raise ValueError("An original flat-ground Plane/Mesh is missing or still collision enabled")
    floor = stage.GetPrimAtPath(FLOOR_PATH)
    if (not floor or not floor.IsA(UsdGeom.Mesh) or not floor.HasAPI(UsdPhysics.CollisionAPI)
            or not UsdPhysics.CollisionAPI(floor).GetCollisionEnabledAttr().Get()):
        raise ValueError("Single reference floor is not an active native-collision Mesh")
    points, counts, indices = reference_ground_geometry(provider.bounds_xy, provider.ground_height_m,
        provider._data["reference_ground"]["margin_m"])
    floor_mesh = UsdGeom.Mesh(floor)
    if (not _matches_authored_points(floor_mesh.GetPointsAttr().Get(), points)
            or list(floor_mesh.GetFaceVertexCountsAttr().Get()) != counts
            or list(floor_mesh.GetFaceVertexIndicesAttr().Get()) != indices
            or not np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(floor)), np.eye(4), atol=1e-8, rtol=0)):
        raise ValueError("Runtime floor geometry/transform differs from the source-bound finite reference")
    material_path = FLOOR_PATH + "/PhysicsMaterial"
    if [str(path) for path in floor.GetRelationship("material:binding:physics").GetTargets()] != [material_path]:
        raise ValueError("Reference floor is not bound to its exact physics material")
    material = stage.GetPrimAtPath(material_path)
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Live reference-floor material is missing its physics schema")
    material_api = UsdPhysics.MaterialAPI(material)
    material_values = {"static_friction": material_api.GetStaticFrictionAttr().Get(),
        "dynamic_friction": material_api.GetDynamicFrictionAttr().Get(), "restitution": material_api.GetRestitutionAttr().Get(),
        "combine_mode": str(material.GetAttribute("physxMaterial:frictionCombineMode").Get())}
    if material_values != provider._data["reference_ground"]["material"]:
        raise ValueError("Actual floor material differs from the admitted reference-ground recipe")

    samples = origins.tolist()
    roles = ["reset_origin"] * len(samples)
    if free_cell_limit:
        free = np.argwhere(provider._map == 254)
        chosen = free[np.linspace(0, len(free) - 1, min(free_cell_limit, len(free)), dtype=np.int64)]
        xy = np.stack((provider._map_origin_xy[0] + (chosen[:, 1] + .5)*.1,
            provider._map_origin_xy[1] + (provider._map.shape[0] - chosen[:, 0] - .5)*.1), -1)
        samples.extend(xy.tolist())
        roles.extend(["deterministic_admitted_free_sample"] * len(chosen))
    ray_rows = []
    for xy, role in zip(samples, roles):
        hits = []
        start_counter = provider._counter()

        def collect(hit):
            position, normal, distance = tuple(hit.position), tuple(hit.normal), float(hit.distance)
            path = str(hit.collision)
            if (len(position) != 3 or len(normal) != 3 or not all(math.isfinite(value) for value in (*position, *normal, distance))
                    or distance < 0 or not math.isclose(np.linalg.norm(normal), 1., abs_tol=1e-3)):
                raise ValueError("Live native ownership ray returned invalid/nonfinite hit geometry")
            near = normal[2] >= .9 and abs(position[2] - provider.ground_height_m) <= .08
            if dynamic(path):
                classification = "excluded_exact_dynamic_robot"
            elif path not in expected:
                raise ValueError("Live ownership ray returned a foreign collider")
            elif path in provider.floor_roots:
                classification = "selected_floor_support" if near else "selected_floor_other_surface"
            elif near:
                raise ValueError("Undeclared coplanar upward support in the live world")
            else:
                classification = "known_static_nonfloor_surface"
            hits.append({"collision": path, "rigid_body": str(hit.rigid_body), "position": list(position),
                "normal": list(normal), "distance": distance, "classification": classification})
            return True

        any_hit = provider._query.raycast_all((float(xy[0]), float(xy[1]), provider.ground_height_m + .3),
            (0., 0., -1.), .6, collect, False)
        after = provider._counter()
        if start_counter != before or after != before:
            raise RuntimeError("Physics advanced during read-only native live-runtime verification")
        if bool(any_hit) != bool(hits) or not any(row["classification"] == "selected_floor_support" for row in hits):
            raise ValueError("Live all-hit ray has no exact selected-floor support or inconsistent callback count")
        ray_rows.append({"world_xy_m": xy, "role": role, "counter_before": start_counter,
            "counter_after": after, "all_hits": hits})
    provider.verify_sources()
    after = provider._counter()
    if after != before or counter_getter() != before:
        raise RuntimeError("Read-only stage/source verification advanced the submitted physics counter")
    return {"schema": "dynbench_live_runtime_observation.v1", "status": "passed_live_stage_and_sampled_native_floor_ownership",
        "scene": provider.scene_name, "observed_current_singleton_context_identity": True,
        "observed_active_stage_matches_context": True,
        "stage_root_identifier": str(stage.GetRootLayer().identifier), "stage_session_identifier": str(session.identifier),
        "App_creation_count_observed": None, "SimulationContext_creation_count_observed": None,
        "new_App_or_Context_created_by_verifier": False, "source_session_occurrences": 1,
        "source_USD": dict(provider._data["source"]), "source_layer_records": [dict(row) for row in provider._data["layers"]],
        "source_layers_rehashed_before_after_unchanged": True, "source_root_paths": source_roots,
        "authored_source_physics_scenes_inactive": authored_physics, "active_physics_scene_paths": live_scenes,
        "enabled_static_and_floor_collision_paths": sorted(enabled), "floor_paths": list(provider.floor_roots),
        "disabled_original_flat_ground_paths": list(provider._disabled_flat_ground_paths),
        "generated_or_foreign_terrain_collision_paths": [], "observed_floor_geometry_matches_recipe": True,
        "observed_floor_material": material_values, "exact_dynamic_robot_roots": list(provider._dynamic_robot_roots),
        "native_floor_ray_rows": ray_rows, "reset_origins_tested": len(origins), "optional_admitted_free_samples": len(samples)-len(origins),
        "submitted_step_counter_before": before, "submitted_step_counter_after": after, "query_physics_advance": 0,
        "counter_semantics": "actual Lab submitted-step counter; no claimed native engine timestamp",
        "native_engine_time_observed": False, "native_engine_time_before_after_delta": None,
        "fresh_full_grid_native_reprobe_performed": False, "navigation_or_stability_quality_verified": False,
        "limitations": ["Finite all-hit native ray samples verify current support at reset/sample points, not every world point or swept body.",
            "Current singleton/context and stage identity are observed; App/context historical constructor counts are not measured.",
            "No physical steps, pose writes, new world, query view, or source USD save is performed."]}
