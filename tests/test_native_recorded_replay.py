"""Native recorded-pose CPU source gates; no renderer/SDK execution claim."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from CrowdSim.humanoid.runtime import native_visual_scene
from CrowdSim.humanoid.runtime.dynbench_scene import file_record
from scripts.replay_paper_pedestrians import validate_arrays, validate_scene
import test_dynbench_scene_provider as contracts
from test_paper_recorded_replay import fixture

evidence = contracts.evidence
ROOT = Path(__file__).resolve().parents[1]


def recorded_result(evidence):
    arrays, result = fixture()
    provider = contracts.admitted_fixture(evidence)
    scene = {"schema": native_visual_scene.SCHEMA, "scene": "Office",
        "preflight": {"path": str(evidence.preflight_path), "sha256": evidence.preflight_record["sha256"]},
        "native_map": {"path": str(evidence.native_path), "sha256": evidence.native_record["sha256"]}}
    result["evaluation_scene"] = scene
    result["native_scene_provider"] = {**provider.describe(), "live_query_bound": True}
    return arrays, result


def test_native_visuals_require_explicit_flag_and_actual_completed_provider(evidence):
    arrays, result = recorded_result(evidence)
    with pytest.raises(ValueError, match="native-scene-visuals"):
        validate_arrays(arrays, result, steps=8)
    prepared = validate_arrays(arrays, result, steps=8, native_scene_visuals=True)
    visual = prepared["scene"]
    assert visual["source_USD"] == evidence.preflight["source"]
    assert visual["recorded_pose_visualization_only"] and not visual["native_navigation_verified_by_renderer"]
    assert visual["finite_reference_floor"]["height_m"] == 0
    assert visual["world_coordinates"].startswith("original source world XY")
    with pytest.raises(ValueError):
        validate_scene({"cuboids": []}, native_scene_visuals=True, source_result=result)
    with pytest.raises(ValueError):
        validate_scene(result["evaluation_scene"], native_scene_visuals=True)


@pytest.mark.parametrize("change", ["not_completed", "cold_provider", "unadmitted", "different_scene", "wrong_sourceUSD",
    "wrong_preflight", "wrong_map", "changed_source", "missing_provider", "changed_bound_result"])
def test_native_visual_admission_rejects_stale_missing_or_mixed_provenance(evidence, change):
    _, result = recorded_result(evidence)
    scene = result["evaluation_scene"]
    if change == "not_completed":
        result["status"] = "preflight_passed"
    elif change == "cold_provider":
        result["native_scene_provider"]["live_query_bound"] = False
    elif change == "unadmitted":
        result["native_scene_provider"]["native_resource_map_admitted"] = False
    elif change == "different_scene":
        result["native_scene_provider"]["scene"] = "Warehouse"
    elif change == "wrong_sourceUSD":
        result["native_scene_provider"]["source_USD"]["sha256"] = "0"*64
    elif change == "wrong_preflight":
        scene["preflight"]["sha256"] = "0"*64
    elif change == "wrong_map":
        scene["native_map"]["sha256"] = "0"*64
    elif change == "changed_source":
        Path(evidence.preflight["source"]["path"]).write_text("changed")
    elif change == "missing_provider":
        result.pop("native_scene_provider")
    elif change == "changed_bound_result":
        visual = native_visual_scene.prepare_native_visual_scene(scene, result)
        resource = evidence.tmp / "result.json"
        resource.write_text("{}")
        visual["recorded_source_result"] = file_record(resource)
        resource.write_text('{"status":"changed"}')
        with pytest.raises(ValueError):
            native_visual_scene.verify_native_visual_sources(visual)
        return
    with pytest.raises(ValueError):
        native_visual_scene.prepare_native_visual_scene(scene, result)


def test_old_cuboid_default_preflight_exactly_matches_frozen_before_bytes():
    before = ROOT / "docs/style_improvement_20261002/native_recorded_replay_v1/before/scripts/replay_paper_pedestrians.py"
    spec = importlib.util.spec_from_file_location("before_recorded_replay", before)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The historical source computes ROOT from its file path; rebind only the
    # actual rig path, not validation/capture algorithms, for this comparison.
    module.RIG = ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    arrays, result = fixture()
    result["evaluation_scene"] = {"cuboids": [{"center": [1., 2., 3.], "size": [1., 1., 1.]}]}
    old = module.validate_arrays(arrays, result, start_step=2, steps=6, capture_every_steps=2)
    current = validate_arrays(arrays, result, start_step=2, steps=6, capture_every_steps=2)
    assert current == old


def test_importing_new_helper_keeps_SDK_and_torch_unloaded():
    code = """import sys
from CrowdSim.humanoid.runtime import native_visual_scene
import scripts.replay_paper_pedestrians
assert not any(n.split('.')[0] in {'torch','omni','pxr','isaaclab','isaacsim'} for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_visual_composition_uses_passed_stage_without_context_or_physics(evidence, monkeypatch):
    from test_dynbench_live_runtime import Attribute, Prim, Stage

    _, result = recorded_result(evidence)
    contract = native_visual_scene.prepare_native_visual_scene(result["evaluation_scene"], result)
    root = Prim("/Root", kind="Xform", collision=False)
    collision = Prim("/Root/Wall")
    physics_scene = Prim("/Root/SourcePhysics", kind="Scene", collision=False)
    stage = Stage([])
    stage.edit = "original-edit"
    session = stage.GetSessionLayer()

    class Sublayers(list):
        def append(self, path):
            super().append(path)
            stage.prims.update({prim.path: prim for prim in (root, collision, physics_scene)})

    session.subLayerPaths = []
    stage.GetRootLayer().subLayerPaths = Sublayers()
    stage.GetEditTarget = lambda: stage.edit
    stage.SetEditTarget = lambda value: setattr(stage, "edit", value)
    physics_scene.SetActive = lambda value: setattr(physics_scene, "active", value)

    class Mesh:
        @classmethod
        def Define(cls, value, path):
            assert value is stage
            visual = Prim(path, kind="VisualMesh", collision=False)
            stage.prims[path] = visual
            return type("Visual", (), {"GetPrim": lambda self: visual,
                **{name: lambda self, data: None for name in ("CreatePointsAttr", "CreateFaceVertexCountsAttr", "CreateFaceVertexIndicesAttr", "CreateSubdivisionSchemeAttr", "CreateOrientationAttr")}})()

    class CollisionAPI:
        def __new__(cls, prim):
            return type("Collision", (), {"GetCollisionEnabledAttr": lambda self: Attribute(prim.collision),
                "CreateCollisionEnabledAttr": lambda self, value: setattr(prim, "collision", value)})()

    old_has_api = Prim.HasAPI
    monkeypatch.setattr(Prim, "HasAPI", lambda prim, schema: old_has_api(prim, "CollisionAPI" if schema is CollisionAPI else schema))
    from types import SimpleNamespace
    pxr = SimpleNamespace(Sdf=SimpleNamespace(Layer=SimpleNamespace(FindOrOpen=lambda path: SimpleNamespace(rootPrims=[SimpleNamespace(path="/Root")]))),
        Usd=SimpleNamespace(PrimRange=SimpleNamespace(Stage=lambda value, predicate: value.Traverse()), TraverseInstanceProxies=lambda: None),
        UsdGeom=SimpleNamespace(Mesh=Mesh, GetStageUpAxis=lambda value: "Z", GetStageMetersPerUnit=lambda value: 1.),
        UsdPhysics=SimpleNamespace(Scene="Scene", CollisionAPI=CollisionAPI, RigidBodyAPI="RigidBodyAPI"))
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setattr("CrowdSim.humanoid.runtime.dynbench_scene._editable_paths", lambda value, paths: list(paths))
    report = native_visual_scene.compose_native_visual_scene(stage, contract)
    assert stage.edit == "original-edit" and not collision.collision and not physics_scene.active
    assert report["disabled_authored_PhysicsScene_paths"] == ["/Root/SourcePhysics"]
    assert report["source_session_occurrences"] == 0
    assert report["source_root_layer_occurrences"] == 1
    assert report["source_is_weak_root_sublayer_not_session_sublayer"]
    assert session.subLayerPaths == []
    assert not report["SimulationContext_created"] and report["physics_steps"] == 0
    assert not report["contacts_or_controller_quality_verified"] and not report["visual_floor_has_collision"]
    with pytest.raises(ValueError, match="conflicts"):
        native_visual_scene.compose_native_visual_scene(stage, contract)
