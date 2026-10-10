"""Actual USD composition tests, not a render/physics SDK verification."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = ROOT / "docs/style_improvement_20261002/native_visual_placeholder_sdk_probe_v1/actual_empty_stage.json"

spec = importlib.util.spec_from_file_location("CrowdSim.humanoid.runtime._native_visual_candidate", HERE / "native_visual_scene_candidate.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def runtime_stage():
    stage = Usd.Stage.CreateInMemory("SDK_root_text_reconstruction.usda")
    assert stage.GetRootLayer().ImportFromString(json.loads(PROBE.read_text())["root_layer_text"])
    return stage


def source_stage(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "source.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Root")
    root.AddTranslateOp().Set((3., -2., 1.))
    nested = UsdGeom.Xform.Define(stage, "/Root/Nested")
    nested.AddRotateZOp().Set(30.)
    mesh = UsdGeom.Cube.Define(stage, "/Root/Nested/SourceCube")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    # Matches actual source's no-content over, including its registered
    # metadata. Copying the actual Sdf spec avoids recreating an API guess.
    actual = Sdf.Layer.FindOrOpen(json.loads(PROBE.read_text())["source_USD"]["path"])
    Sdf.CopySpec(actual, "/Render", stage.GetRootLayer(), "/Render")
    stage.GetRootLayer().Save()
    return stage


def snapshot_runtime_properties(stage):
    values = {}
    for prim in Usd.PrimRange.AllPrims(stage.GetPrimAtPath("/Render")):
        for prop in prim.GetAuthoredProperties():
            if isinstance(prop, Usd.Attribute):
                values[str(prop.GetPath())] = (prop.GetTypeName(), prop.Get(), prop.GetTimeSamples())
            else:
                values[str(prop.GetPath())] = tuple(prop.GetTargets())
    return values


def test_actual_sdk_root_text_accepts_only_actual_source_render_metadata(tmp_path):
    stage, source = runtime_stage(), source_stage(tmp_path)
    record = candidate.admit_sdk_render_metadata_overlap(stage, source.GetRootLayer())
    assert len(record["SDK_owned_paths"]) == 7
    assert record["source_Render_products_or_geometry_added"] == []
    assert not record["other_source_roots_or_reference_floor_collision_allowed"]


def test_weak_source_composition_preserves_runtime_and_exact_geometry(tmp_path, monkeypatch):
    stage, source = runtime_stage(), source_stage(tmp_path)
    original = source.GetRootLayer().ExportToString()
    properties = snapshot_runtime_properties(stage)
    path = "/Root/Nested/SourceCube"
    expected = [UsdGeom.XformCache(time).GetLocalToWorldTransform(source.GetPrimAtPath(path)) for time in (Usd.TimeCode.Default(), Usd.TimeCode(0))]
    # This synthetic test isolates composition, not provider/resource admission.
    monkeypatch.setattr(candidate, "verify_native_visual_sources", lambda contract: True)
    report = candidate.compose_native_visual_scene(stage, {"source_USD": {"path": source.GetRootLayer().identifier},
        "finite_reference_floor": {"path": candidate.FLOOR_PATH, "bounds_xy_m": [[-5, -5], [5, 5]], "height_m": 0., "margin_m": .2}})
    actual = [UsdGeom.XformCache(time).GetLocalToWorldTransform(stage.GetPrimAtPath(path)) for time in (Usd.TimeCode.Default(), Usd.TimeCode(0))]
    assert expected == actual and properties == snapshot_runtime_properties(stage)
    assert source.GetRootLayer().ExportToString() == original
    assert report["source_root_layer_occurrences"] == 1 and report["source_session_occurrences"] == 0
    assert not stage.GetPrimAtPath("/PhysicsScene").IsActive()
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get()
    assert not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetRigidBodyEnabledAttr().Get()
    assert not stage.GetPrimAtPath(candidate.FLOOR_PATH).HasAPI(UsdPhysics.CollisionAPI)


@pytest.mark.parametrize("bad", ["spatial_child", "inactive_child", "purpose", "foreign_layer", "references", "unknown_API", "unknown_metadata"])
def test_runtime_render_rejects_geometry_inactive_or_unowned_opinions(tmp_path, bad):
    stage, source = runtime_stage(), source_stage(tmp_path)
    foreign = None
    if bad == "spatial_child":
        UsdGeom.Xform.Define(stage, "/Render/Spatial")
    elif bad == "inactive_child":
        stage.DefinePrim("/Render/Hidden", "Scope").SetActive(False)
    elif bad == "purpose":
        UsdGeom.Imageable(stage.GetPrimAtPath("/Render")).CreatePurposeAttr("guide")
    elif bad == "foreign_layer":
        foreign = Usd.Stage.CreateInMemory()
        foreign.OverridePrim("/Render")
        stage.GetRootLayer().subLayerPaths.append(foreign.GetRootLayer().identifier)
    elif bad == "references":
        stage.GetPrimAtPath("/Render").GetReferences().AddInternalReference("/OmniverseKit_Persp")
    elif bad == "unknown_API":
        stage.GetPrimAtPath("/Render").SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["UnknownPhysicsAPI"]))
    elif bad == "unknown_metadata":
        stage.GetPrimAtPath("/Render").SetMetadata("kind", "component")
    with pytest.raises(ValueError):
        candidate.admit_sdk_render_metadata_overlap(stage, source.GetRootLayer())


@pytest.mark.parametrize("bad", ["source_product", "source_spatial", "source_visibility", "source_inactive", "source_reference", "source_API"])
def test_source_render_content_cannot_be_admitted_as_metadata_over(tmp_path, bad):
    stage, source = runtime_stage(), source_stage(tmp_path)
    prim = source.GetPrimAtPath("/Render")
    if bad == "source_product":
        source.DefinePrim("/Render/Product", "RenderProduct")
    elif bad == "source_spatial":
        UsdGeom.Xform.Define(source, "/Render")
    elif bad == "source_visibility":
        UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
    elif bad == "source_inactive":
        prim.SetActive(False)
    elif bad == "source_reference":
        prim.GetReferences().AddInternalReference("/Root")
    elif bad == "source_API":
        UsdPhysics.CollisionAPI.Apply(prim)
    with pytest.raises(ValueError):
        candidate.admit_sdk_render_metadata_overlap(stage, source.GetRootLayer())


@pytest.mark.parametrize("path", ["/Root", "/PhysicsScene", "/NavMesh", candidate.FLOOR_PATH])
def test_other_source_or_floor_root_conflicts_remain_rejected(tmp_path, monkeypatch, path):
    stage, source = runtime_stage(), source_stage(tmp_path)
    if path == "/NavMesh":
        source.DefinePrim(path, "Scope")
    stage.DefinePrim(path, "Xform")
    monkeypatch.setattr(candidate, "verify_native_visual_sources", lambda contract: True)
    with pytest.raises(ValueError, match="conflicts"):
        candidate.compose_native_visual_scene(stage, {"source_USD": {"path": source.GetRootLayer().identifier}})


@pytest.mark.parametrize("marker", ["hide_in_stage_window", "no_delete"])
def test_explicit_false_sdk_marker_cannot_use_metadata_truthiness(tmp_path, marker):
    stage, source = runtime_stage(), source_stage(tmp_path)
    # Import retains literal unregistered false in standalone USD too.
    text = stage.GetRootLayer().ExportToString()
    text = text.replace(marker + " = true", marker + " = false", 1)
    assert stage.GetRootLayer().ImportFromString(text)
    with pytest.raises(ValueError, match="observed SDK-owned"):
        candidate.admit_sdk_render_metadata_overlap(stage, source.GetRootLayer())


def test_existing_physics_outside_source_roots_remains_forbidden(tmp_path, monkeypatch):
    stage, source = runtime_stage(), source_stage(tmp_path)
    UsdPhysics.Scene.Define(stage, "/AlreadyRunningPhysics")
    monkeypatch.setattr(candidate, "verify_native_visual_sources", lambda contract: True)
    with pytest.raises(ValueError, match="active PhysicsScene"):
        candidate.compose_native_visual_scene(stage, {"source_USD": {"path": source.GetRootLayer().identifier}})
