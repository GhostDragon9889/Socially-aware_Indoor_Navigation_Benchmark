"""Source-bound native USD visuals for recorded poses, never native physics."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .dynbench_scene import DynBenchSceneProvider, FLOOR_PATH, _check_record, file_record, reference_ground_geometry

SCHEMA = "crowdsim.native_pedestrian_scene.v1"


def prepare_native_visual_scene(scene, result):
    """CPU-only bind original USD/native map to a completed physical capture."""
    if (not isinstance(scene, dict) or scene.get("schema") != SCHEMA
            or result.get("status") != "passed" or result.get("evaluation_scene") != scene):
        raise ValueError("Native recorded visuals require the exact completed source evaluation_scene")
    actual = result.get("native_scene_provider", {})
    if (actual.get("native_resource_map_admitted") is not True or actual.get("live_query_bound") is not True
            or actual.get("floor_owner") != "reference_flat_terrain" or actual.get("scene") != scene.get("scene")):
        raise ValueError("Source recorded run lacks its admitted and bound actual native scene provider")
    try:
        preflight_record, native_record = actual["preflight"], actual["native_map_evidence"]
        for name, record in (("preflight", preflight_record), ("native_map", native_record)):
            resource = scene[name]
            if (Path(resource["path"]).resolve() != Path(record["path"]).resolve()
                    or resource["sha256"] != record["sha256"] or file_record(record["path"]) != record):
                raise ValueError("Native scene configuration/result resource identity changed: " + name)
        snapshot = actual.get("probe_script_verified_historical_snapshot")
        declared_snapshot = scene.get("probe_script_snapshot")
        if snapshot is not None:
            if (not isinstance(declared_snapshot, dict) or file_record(snapshot["path"]) != snapshot
                    or Path(declared_snapshot["path"]).resolve() != Path(snapshot["path"]).resolve()
                    or declared_snapshot["sha256"] != snapshot["sha256"]):
                raise ValueError("Recorded native visual probe source snapshot changed")
        elif declared_snapshot is not None:
            raise ValueError("Scene snapshot declaration differs from the actual recorded provider")
        provider = DynBenchSceneProvider.from_preflight(preflight_record["path"],
            expected_sha256=preflight_record["sha256"], probe_script_snapshot=None if snapshot is None else snapshot["path"])
        provider.admit_native_map(native_record["path"], expected_sha256=native_record["sha256"])
        if (provider._data["source"] != actual["source_USD"] or provider.scene_name != scene["scene"]
                or list(provider.floor_roots) != actual["floor_paths"]):
            raise ValueError("Native recorded source USD or selected floor identity changed")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete actual native recorded-scene provenance") from error
    return {"schema": "native_recorded_visual_scene.v1", "scene": provider.scene_name,
        "source_USD": deepcopy(provider._data["source"]), "source_layers": deepcopy(provider._data["layers"]),
        "preflight": deepcopy(preflight_record), "native_map": deepcopy(native_record),
        "probe_script_snapshot": deepcopy(snapshot), "evaluation_scene": deepcopy(scene),
        "finite_reference_floor": {"path": FLOOR_PATH, "bounds_xy_m": provider.bounds_xy,
            "height_m": provider.ground_height_m, "margin_m": provider._data["reference_ground"]["margin_m"]},
        "world_coordinates": "original source world XY including negative coordinates; no visual fitting/reanchor",
        "recorded_pose_visualization_only": True, "native_navigation_verified_by_renderer": False,
        "source_falls": result.get("fall_resets"), "source_control_steps": result.get("actual_control_steps")}


def verify_native_visual_sources(contract):
    """Rebind the original prepared native evidence immediately before rendering."""
    for key in ("recorded_source_result", "recorded_states"):
        if key in contract:
            _check_record(contract[key])
    # Reconstruct only previously admitted resource fields for repeat hashing;
    # this wrapper never supplies a new physical rollout or live observation.
    bound_resources = {"status": "passed", "evaluation_scene": contract["evaluation_scene"],
        "native_scene_provider": {"scene": contract["scene"], "native_resource_map_admitted": True,
            "live_query_bound": True, "floor_owner": "reference_flat_terrain", "preflight": contract["preflight"],
            "native_map_evidence": contract["native_map"], "probe_script_verified_historical_snapshot": contract["probe_script_snapshot"],
            "source_USD": contract["source_USD"], "floor_paths": [contract["finite_reference_floor"]["path"]]}}
    rebound = prepare_native_visual_scene(contract["evaluation_scene"], bound_resources)
    if rebound["source_layers"] != contract["source_layers"] or rebound["finite_reference_floor"] != contract["finite_reference_floor"]:
        raise ValueError("Prepared native visual source layers/reference floor changed")
    return True


def compose_native_visual_scene(stage, contract):
    """Compose original source into the caller's existing render-only USD stage.

    Unlike the physical provider spawner, this requires no SimulationContext
    or SDK thread-local stage. The caller supplies its sole existing visual
    stage. All physics is disabled in its session before render app updates.
    """
    verify_native_visual_sources(contract)
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("Recorded native visual stage requires Z-up/metres1")
    source = contract["source_USD"]["path"]
    layer = Sdf.Layer.FindOrOpen(source)
    if layer is None:
        raise ValueError("Recorded native source USD cannot be opened")
    source_roots = [str(prim.path) for prim in layer.rootPrims]
    if not source_roots or any(stage.GetPrimAtPath(path) for path in source_roots) or stage.GetPrimAtPath(FLOOR_PATH):
        raise ValueError("Existing visual stage conflicts with original native source/floor roots")
    if any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        raise ValueError("Recorded visual replay cannot share an existing active PhysicsScene")
    session = stage.GetSessionLayer()
    if source in session.subLayerPaths:
        raise ValueError("Native source is already composed in visual stage")
    original_edit = stage.GetEditTarget()
    disabled_scenes, disabled_collisions, disabled_bodies = [], [], []
    try:
        stage.SetEditTarget(session)
        session.subLayerPaths.append(source)
        if stage.GetCompositionErrors():
            raise ValueError("Native recorded visual source has USD composition errors")
        # Source admission requires no animated/dynamic geometry. Deinstance
        # only source ancestors needed for session physics overrides; retain
        # authored paths/transforms/materials and never save original layers.
        source_prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        physics_paths = [str(prim.GetPath()) for prim in source_prims if prim.IsA(UsdPhysics.Scene)]
        from .dynbench_scene import _editable_paths
        collider_paths = [str(prim.GetPath()) for prim in source_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
        body_paths = [str(prim.GetPath()) for prim in source_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        _editable_paths(stage, (*collider_paths, *body_paths))
        for path in physics_paths:
            stage.GetPrimAtPath(path).SetActive(False)
            disabled_scenes.append(path)
        for path in collider_paths:
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
            disabled_collisions.append(path)
        for path in body_paths:
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateRigidBodyEnabledAttr(False)
            disabled_bodies.append(path)
        floor = contract["finite_reference_floor"]
        points, counts, indices = reference_ground_geometry(floor["bounds_xy_m"], floor["height_m"], floor["margin_m"])
        mesh = UsdGeom.Mesh.Define(stage, floor["path"])
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateOrientationAttr("rightHanded")
        if mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("Recorded reference-floor visual unexpectedly has physics collision")
    finally:
        stage.SetEditTarget(original_edit)
    if (any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse())
            or any(prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))):
        raise ValueError("Active source physics remains in the recorded-only visual stage")
    verify_native_visual_sources(contract)
    return {"schema": "native_recorded_visual_stage.v1", "source_USD": deepcopy(contract["source_USD"]),
        "source_root_paths": source_roots, "source_session_occurrences": list(session.subLayerPaths).count(source),
        "stage_root_identifier": str(stage.GetRootLayer().identifier), "disabled_authored_PhysicsScene_paths": disabled_scenes,
        "disabled_collision_paths": disabled_collisions, "disabled_rigid_body_paths": disabled_bodies,
        "visual_reference_floor": deepcopy(contract["finite_reference_floor"]), "visual_floor_has_collision": False,
        "original_global_coordinates_preserved": True, "source_USD_layers_rehashed_unchanged": True,
        "SimulationContext_created": False, "physics_steps": 0, "policy_forward_calls": 0,
        "contacts_or_controller_quality_verified": False, "source_USD_saved": False}
