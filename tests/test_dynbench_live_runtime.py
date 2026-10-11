"""Fake-interface CPU contracts, not SDK native execution or physics evidence."""
from types import SimpleNamespace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.runtime import dynbench_scene
from CrowdSim.humanoid.runtime.dynbench_scene import FLOOR_PATH, SCENE_MARKER_PATH, reference_ground_geometry
import test_dynbench_scene_provider as source_contracts

# Reuse the explicit source-bound contract fixture, without importing SDK.
evidence = source_contracts.evidence


def test_floor_points_compare_the_actual_USD_float32_representation():
    from pxr import Usd, UsdGeom
    from CrowdSim.humanoid.runtime.dynbench_live import _matches_authored_points
    # Hospital's 37.6 metre boundary rounds by 1.53e-6 in a point3f array.
    points, _, _ = reference_ground_geometry(((-48.8, -4.9), (27.4, 37.4)), 0., .2)
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/Floor')
    mesh.CreatePointsAttr(points)
    actual = np.asarray(mesh.GetPointsAttr().Get())
    assert actual.dtype == np.float32
    assert np.max(np.abs(actual.astype(np.float64) - points)) > 1e-6
    assert _matches_authored_points(actual, points)
    changed = actual.copy()
    changed[0, 0] += .001
    assert not _matches_authored_points(changed, points)
    assert not _matches_authored_points(actual[:1], points)
    changed[0, 0] = np.nan
    assert not _matches_authored_points(changed, points)


class Attribute:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value


class Prim:
    def __init__(self, path, *, kind="Mesh", active=True, collision=True):
        self.path, self.kind, self.active, self.collision = path, kind, active, collision
        self.attrs = {}
        self.targets = []
        self.points = self.counts = self.indices = None

    def GetPath(self):
        return self.path

    def IsActive(self):
        return self.active

    def IsA(self, schema):
        return self.kind == schema

    def HasAPI(self, api):
        return api in ("CollisionAPI", "MaterialAPI") and (self.kind != "Xform") and (
            (api == "CollisionAPI" and self.kind == "Mesh") or (api == "MaterialAPI" and self.kind == "Material"))

    def GetRelationship(self, name):
        assert name == "material:binding:physics"
        return SimpleNamespace(GetTargets=lambda: self.targets)

    def GetAttribute(self, name):
        return Attribute(self.attrs[name])

    def __bool__(self):
        return True


class Stage:
    def __init__(self, prims, sublayers=()):
        self.prims = {prim.path: prim for prim in prims}
        self.session = SimpleNamespace(subLayerPaths=list(sublayers), identifier="anon:cpu_contract_session")
        self.root = SimpleNamespace(identifier="anon:cpu_contract_root")

    def GetPrimAtPath(self, path):
        return self.prims.get(path)

    def GetSessionLayer(self):
        return self.session

    def GetRootLayer(self):
        return self.root

    def GetCompositionErrors(self):
        return []

    def GetPseudoRoot(self):
        return SimpleNamespace(GetChildren=lambda: [self.prims["/Root"]])

    def Traverse(self):
        return [prim for prim in self.prims.values() if prim.active]


class Context:
    def __init__(self, stage):
        self.stage, self.step = stage, 2
        self.cfg = SimpleNamespace(enable_scene_query_support=True)

    def get_physics_step_count(self):
        return self.step


@pytest.fixture
def live(evidence, monkeypatch):
    provider = source_contracts.admitted_fixture(evidence)
    provider.prepare_simulation_cfg(SimpleNamespace(enable_scene_query_support=False))
    floor = Prim(FLOOR_PATH)
    floor.points, floor.counts, floor.indices = reference_ground_geometry(provider.bounds_xy, 0., .2)
    floor.targets = [FLOOR_PATH + "/PhysicsMaterial"]
    material = Prim(floor.targets[0], kind="Material", collision=False)
    material.attrs["physxMaterial:frictionCombineMode"] = "average"
    material.attrs.update(static_friction=1., dynamic_friction=1., restitution=0.)
    original = Prim("/Root/Floor", collision=False)
    source_physics = Prim("/Root/SourcePhysics", kind="Scene", active=False, collision=False)
    stage = Stage([floor, material, original, Prim("/Root", kind="Xform", collision=False),
        Prim("/Root/Wall"), source_physics, Prim(SCENE_MARKER_PATH, kind="Xform", collision=False),
        Prim("/World/physicsScene", kind="Scene", collision=False),
        Prim("/World/envs/env_0/Robot/Foot")], sublayers=[provider._data["source"]["path"]])
    source_stage = Stage([Prim("/Root", kind="Xform", collision=False), Prim("/Root/SourcePhysics", kind="Scene", collision=False)])
    # This fixture constructs a fake spawned owner, so supply the same layer
    # closure fields that the real provider._spawn publishes. The actual
    # verify_owned_layers function still validates every declared memory hash.
    source_path = provider._data["source"]["path"]
    source_text = Path(source_path).read_text()
    owned = SimpleNamespace(identifier=source_path, ExportToString=lambda: source_text)
    provider._runtime_source_layer = owned
    provider._portable_layers = (owned,)
    provider._portable_usd_receipt = [{"owned_layer": source_path,
        "owned_ram_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "exact_inverse": True, "geometry_or_physics_opinions_changed": False}]
    context = Context(stage)
    hit_rows = [SimpleNamespace(collision=FLOOR_PATH, rigid_body="", position=(0., 0., 0.), normal=(0., 0., 1.), distance=.3)]
    calls = []

    def raycast(origin, direction, distance, callback, any_hit):
        calls.append(origin)
        assert direction == (0., 0., -1.) and distance == .6 and any_hit is False
        for hit in hit_rows:
            callback(hit)
        return bool(hit_rows)

    class Mesh:
        def __new__(cls, prim):
            return SimpleNamespace(GetPointsAttr=lambda: Attribute(prim.points),
                GetFaceVertexCountsAttr=lambda: Attribute(prim.counts), GetFaceVertexIndicesAttr=lambda: Attribute(prim.indices))

    class CollisionAPI:
        def __new__(cls, prim):
            return SimpleNamespace(GetCollisionEnabledAttr=lambda: Attribute(prim.collision))

    class MaterialAPI:
        def __new__(cls, prim):
            return SimpleNamespace(GetStaticFrictionAttr=lambda: Attribute(prim.attrs["static_friction"]),
                GetDynamicFrictionAttr=lambda: Attribute(prim.attrs["dynamic_friction"]),
                GetRestitutionAttr=lambda: Attribute(prim.attrs["restitution"]))

    usd = SimpleNamespace(Stage=SimpleNamespace(Open=lambda path: source_stage),
        PrimRange=SimpleNamespace(Stage=lambda value, predicate: value.Traverse()), TraverseInstanceProxies=lambda: None)
    geometry = SimpleNamespace(Mesh=Mesh, XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda prim: np.eye(4)),
        GetStageUpAxis=lambda value: "Z", GetStageMetersPerUnit=lambda value: 1.)
    physics = SimpleNamespace(Scene="Scene", CollisionAPI=CollisionAPI, MaterialAPI=MaterialAPI)
    # Schema identity is the class itself, as in actual USD bindings.
    old_is_a, old_has_api = Prim.IsA, Prim.HasAPI

    def is_a(prim, schema):
        return prim.kind == "Mesh" if schema is Mesh else old_is_a(prim, schema)

    def has_api(prim, schema):
        mapped = "CollisionAPI" if schema is CollisionAPI else "MaterialAPI" if schema is MaterialAPI else schema
        return old_has_api(prim, mapped)

    monkeypatch.setattr(Prim, "IsA", is_a)
    monkeypatch.setattr(Prim, "HasAPI", has_api)
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Usd=usd, UsdGeom=geometry, UsdPhysics=physics))
    simulated_sdk = SimpleNamespace(SimulationContext=SimpleNamespace(instance=lambda: context))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", simulated_sdk)
    # Python may resolve an already imported submodule through its parent attr.
    if "isaaclab" in sys.modules:
        monkeypatch.setattr(sys.modules["isaaclab"], "sim", simulated_sdk, raising=False)
    monkeypatch.setattr(dynbench_scene, "_active_spawn_stage", lambda: stage)
    provider.bind_native_query(SimpleNamespace(raycast_all=raycast), physics_step_count=context.get_physics_step_count,
        dynamic_robot_roots=["/World/envs/env_0/Robot"])
    return SimpleNamespace(provider=provider, stage=stage, context=context, hits=hit_rows, calls=calls,
        floor=floor, material=material, original=original, source_physics=source_physics, geometry=geometry,
        origin=[[-.15, -.05]])


def test_observed_live_gate_preserves_clock_sources_and_records_all_hits(live):
    live.hits.append(SimpleNamespace(collision="/World/envs/env_0/Robot/Foot", rigid_body="robot", position=(0., 0., .1),
        normal=(0., 0., 1.), distance=.2))
    report = live.provider.verify_live_runtime(origins_xy=live.origin, simulation_context=live.context, free_cell_limit=2)
    assert len(live.calls) == 3 and live.context.step == 2
    assert report["observed_active_stage_matches_context"] and report["query_physics_advance"] == 0
    assert report["native_engine_time_before_after_delta"] is None
    assert report["App_creation_count_observed"] is None and not report["new_App_or_Context_created_by_verifier"]
    assert report["floor_paths"] == [FLOOR_PATH] and report["optional_admitted_free_samples"] == 2
    assert len(report["native_floor_ray_rows"][0]["all_hits"]) == 2
    assert report["native_floor_ray_rows"][0]["all_hits"][1]["classification"] == "excluded_exact_dynamic_robot"
    assert not report["fresh_full_grid_native_reprobe_performed"] and not report["navigation_or_stability_quality_verified"]


@pytest.mark.parametrize("change", ["duplicate_source", "original_floor_enabled", "foreign_terrain", "source_physics_active",
    "two_physics_scenes", "material", "geometry", "missing_marker", "wrong_stage", "wrong_singleton", "wrong_counter"])
def test_live_gate_rejects_wrong_stage_context_floor_source_or_material(live, monkeypatch, change):
    if change == "duplicate_source":
        live.stage.session.subLayerPaths *= 2
    elif change == "original_floor_enabled":
        live.original.collision = True
    elif change == "foreign_terrain":
        live.stage.prims["/World/ground/terrain/mesh"] = Prim("/World/ground/terrain/mesh")
    elif change == "source_physics_active":
        live.source_physics.active = True
    elif change == "two_physics_scenes":
        live.stage.prims["/AnotherPhysics"] = Prim("/AnotherPhysics", kind="Scene", collision=False)
    elif change == "material":
        live.material.attrs["static_friction"] = .5
    elif change == "geometry":
        live.floor.points[0][0] += .02
    elif change == "missing_marker":
        del live.stage.prims[SCENE_MARKER_PATH]
    elif change == "wrong_stage":
        live.context.stage = object()
    elif change == "wrong_singleton":
        wrong_sdk = SimpleNamespace(SimulationContext=SimpleNamespace(instance=lambda: object()))
        monkeypatch.setitem(sys.modules, "isaaclab.sim", wrong_sdk)
        if "isaaclab" in sys.modules:
            monkeypatch.setattr(sys.modules["isaaclab"], "sim", wrong_sdk, raising=False)
    elif change == "wrong_counter":
        live.provider._counter = lambda: 2
    with pytest.raises((ValueError, RuntimeError)):
        live.provider.verify_live_runtime(origins_xy=live.origin, simulation_context=live.context)
    assert not live.calls


@pytest.mark.parametrize("change", ["coplanar", "Robot2", "nonfinite", "no_floor", "bad_normal"])
def test_all_hit_ownership_rejects_hidden_support_unknown_robot_prefix_or_invalid_geometry(live, change):
    if change == "no_floor":
        live.hits.clear()
    else:
        path = "/Root/Wall" if change == "coplanar" else "/World/envs/env_0/Robot2/Foot" if change == "Robot2" else FLOOR_PATH
        position = (0., 0., float("nan")) if change == "nonfinite" else (0., 0., 0.)
        normal = (0., 0., 0.) if change == "bad_normal" else (0., 0., 1.)
        live.hits.append(SimpleNamespace(collision=path, rigid_body="", position=position, normal=normal, distance=.3))
    with pytest.raises(ValueError):
        live.provider.verify_live_runtime(origins_xy=live.origin, simulation_context=live.context)


def test_native_query_advancing_counter_fails_closed(live):
    old = live.provider._query.raycast_all

    def advancing(*args):
        live.context.step += 1
        return old(*args)

    live.provider._query.raycast_all = advancing
    with pytest.raises(RuntimeError, match="advanced"):
        live.provider.verify_live_runtime(origins_xy=live.origin, simulation_context=live.context)


@pytest.mark.parametrize("origin,limit", [([[float("nan"), 0]], 0), ([[50., 0]], 0), ([[-.15, -.05]], True),
    ([[-.15, -.05]], 257), ([[-.15, -.05]], -1)])
def test_invalid_live_origins_and_sample_bounds_fail_before_queries(live, origin, limit):
    with pytest.raises(ValueError):
        live.provider.verify_live_runtime(origins_xy=origin, simulation_context=live.context, free_cell_limit=limit)
    assert not live.calls
