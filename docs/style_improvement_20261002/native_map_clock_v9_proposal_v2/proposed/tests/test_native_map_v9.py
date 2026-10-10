"""Synthetic v9 geometry/session contracts; these tests never claim native PASS."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import CrowdSim

from CrowdSim.humanoid.runtime.dynbench_scene import file_record
import test_native_map_v7 as legacy
import test_native_map_v8 as prior
from CrowdSim.humanoid.runtime import native_map_v8 as prior_verifier
from CrowdSim.humanoid.runtime import dynbench_scene as prior_provider
from CrowdSim.humanoid.runtime import native_grid_scene as prior_grid

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(CrowdSim.__file__).resolve().parents[1]


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate_modules(monkeypatch):
    # Source-loaded modules permit CPU evaluation of a reviewable docs proposal.
    # No business files, App, SDK or GPU are changed/started by this fixture.
    monkeypatch.setattr(legacy, "ROOT", WORKSPACE)
    monkeypatch.setattr(prior, "ROOT", WORKSPACE)
    package = "CrowdSim.humanoid.runtime."
    v8 = source_module(package + "native_map_v9", ROOT / "CrowdSim/humanoid/runtime/native_map_v9.py")
    monkeypatch.setitem(sys.modules, package + "native_map_v9", v8)
    provider = source_module(package + "dynbench_scene", ROOT / "CrowdSim/humanoid/runtime/dynbench_scene.py")
    monkeypatch.setitem(sys.modules, package + "dynbench_scene", provider)
    grid = source_module(package + "v8_CPU_grid", ROOT / "CrowdSim/humanoid/runtime/native_grid_scene.py")
    return v8, provider, grid


def fixture_v9(folder, modules):
    verifier, provider_module, _ = modules
    old, data, low_row = prior.fixture_v8(folder,(prior_verifier,prior_provider,prior_grid))
    preflight=deepcopy(old._data)
    script=folder/"probe_v9.py"
    shutil.copyfile(ROOT/"scripts/probe_paper_scene_maps.py",script)
    assert file_record(script)["sha256"]==verifier.PROBE_SHA256
    preflight["script"]=file_record(script)
    data["script"]=preflight["script"]
    data["preflight"]=legacy.legacy.write_json(folder/"preflight.json",preflight)
    runtime="anon:synthetic_runtime_root.usda"
    session="anon:synthetic_runtime_session.usda"
    source=preflight["source"]["path"]
    from pxr import Sdf, Usd, UsdGeom
    import hashlib
    original_stage=Usd.Stage.CreateInMemory()
    original_stage.SetStartTimeCode(-1.)
    original_stage.SetEndTimeCode(0.)
    original_stage.SetTimeCodesPerSecond(60.)
    original_stage.SetFramesPerSecond(60.)
    UsdGeom.SetStageUpAxis(original_stage,"Z")
    UsdGeom.SetStageMetersPerUnit(original_stage,1.)
    UsdGeom.Xform.Define(original_stage,"/Root")
    UsdGeom.Mesh.Define(original_stage,"/Root/Floor")
    UsdGeom.Mesh.Define(original_stage,"/Root/Wall")
    original_stage.SetDefaultPrim(original_stage.GetPrimAtPath("/Root"))
    original_stage.GetRootLayer().Export(source)
    original_ram=hashlib.sha256(Sdf.Layer.FindOrOpen(source).ExportToString().encode()).hexdigest()
    preflight["source"]=file_record(Path(source))
    preflight["layers"]=[preflight["source"]]
    data["source"]=preflight["source"]
    data["preflight"]=legacy.legacy.write_json(folder/"preflight.json",preflight)
    info={"schema":"dynbench.native_anonymous_runtime_root.v9","absolute_original_source_sublayers":[source],
        "original_source_layer_identifier":source,"owned_runtime_root_identifier":runtime,
        "owned_runtime_root_anonymous":True,"runtime_session_identifier":session,
        "original_source_time_range":deepcopy(data["source_time_range"]),
        "source_stage_metadata":{"up_axis":"Z","meters_per_unit":1.,"frames_per_second":60.,"default_prim_path":"/Root"},
        "original_source_root_in_memory_sha256":original_ram,"original_source_root_in_memory_sha256_at_query":original_ram,
        "original_source_root_in_memory_unchanged_after_composition":True,"original_source_root_in_memory_unchanged_at_query":True,
        "original_source_layers_saved":False,"extra_App_or_Context_created":False,"source_prim_paths_relocated":False}
    data["runtime_stage_composition"]=info
    data["runtime_stage_attachment"]={"schema":"dynbench.native_runtime_stage_attachment.v9","expected_stage_cache_id":123,
        "observed_context_stage_cache_id":123,"active_context_root_identifier":runtime,"active_context_session_identifier":session,
        "observed_context_owns_exact_runtime_stage":True,"app_updates_waiting_for_async_attachment":1,
        "submitted_step_counter_before":0,"submitted_step_counter_after":0,"explicit_physics_steps_issued_by_wait":0,
        "native_engine_time_observed":None,"extra_App_or_Context_created":False,"manual_context_or_PhysX_attach_called":False}
    record=data["runtime_time_range_override"]
    record.update(schema="dynbench.native_frozen_timeline_session.v9",source_root_layer_identifier=runtime,
        session_layer_identifier=session,owned_runtime_root_identifier=runtime,owned_runtime_root_anonymous=True,
        original_source_layer_identifier=source,original_source_root_in_memory_sha256=original_ram,
        original_source_root_in_memory_sha256_at_query=original_ram,original_source_root_in_memory_unchanged_after_setup=True,
        original_source_root_in_memory_unchanged_at_query=True,owned_runtime_root_sublayers_at_query=[source],owned_runtime_root_anonymous_at_query=True,
        source_root_in_memory_sha256="1"*64,source_root_in_memory_sha256_at_query="2"*64,source_root_in_memory_unchanged_at_query=False)
    artifact=folder/"runtime_root.usda"
    artifact.write_text('#usda 1.0\n(\n    defaultPrim = "Root"\n    endTimeCode = 60\n    framesPerSecond = 60\n    metersPerUnit = 1\n    startTimeCode = 0\n    subLayers = [@'+source+'@]\n    timeCodesPerSecond = 60\n    upAxis = "Z"\n)\n')
    data["exports"]["runtime_root.usda"]=file_record(artifact)
    record["source_root_in_memory_sha256_at_query"]=data["exports"]["runtime_root.usda"]["sha256"]
    fresh=provider_module.DynBenchSceneProvider.from_preflight(data["preflight"]["path"],expected_sha256=data["preflight"]["sha256"])
    return fresh,data,low_row


def admit(provider, data):
    path = Path(provider._preflight_record["path"]).parent / "result_v8.json"
    record = legacy.legacy.write_json(path, data)
    return provider.admit_native_map(path, expected_sha256=record["sha256"])


def test_v9_synthetic_dispatch_retains_crop_missing_API_unknown_and_low_obstacles(tmp_path, candidate_modules):
    v8, _, grid_module = candidate_modules
    provider, data, _ = fixture_v9(tmp_path,candidate_modules)
    admit(provider, data)
    grid = grid_module.NativeGridScene.from_provider(provider, origins_xy=[[-4.45, -3.45]], routes_xy=[[[-4.45, -3.45], [-3.45, -2.45]]], clearance_m=0.)
    checked = v8.verify_v9(provider, data, admitted=True)
    assert checked["admission_protocol"] == "v9_exact_probe_SHA"
    assert checked["missing_known5_API_count"] == 3
    assert checked["known5_API_complete"] is False
    assert checked["known_static_nonfloor_near_ground_blocked_known5_cells"] == 1
    assert grid._labels.shape == (20, 24)
    assert not grid.is_walkable([[-1e3, -1e3]])[0]
    assert len(grid.astar_path([-4.45, -3.45], [-3.45, -2.45])) > 1
    assert data["CPU_contract_fixture_only"] is True
    assert checked["new_physical_execution_verified"] is False


@pytest.mark.parametrize("group,key,value",[
    ("runtime_time_range_override","session_edit_target_at_query",False),
    ("runtime_time_range_override","original_source_root_in_memory_sha256_at_query","f"*64),
    ("runtime_time_range_override","original_source_root_in_memory_unchanged_at_query",False),
    ("runtime_time_range_override","owned_runtime_root_sublayers_at_query",["/foreign/source.usd"]),
    ("runtime_time_range_override","owned_runtime_root_anonymous_at_query",False),
    ("runtime_time_range_override","observed_timeline_start_at_query_s",-1/60),
    ("runtime_stage_attachment","observed_context_stage_cache_id",124),
    ("runtime_stage_attachment","active_context_root_identifier","anon:wrong"),
    ("runtime_stage_attachment","submitted_step_counter_after",1),
    ("runtime_stage_attachment","manual_context_or_PhysX_attach_called",True),
    ("runtime_stage_attachment","extra_App_or_Context_created",True),
    ("runtime_stage_attachment","native_engine_time_observed",0.),
    ("runtime_stage_composition","original_source_root_in_memory_sha256_at_query","f"*64),
    ("runtime_stage_composition","source_prim_paths_relocated",True),
])
def test_v9_rejects_original_RAM_cache_ownership_or_query_clock(tmp_path,candidate_modules,group,key,value):
    provider,data,_=fixture_v9(tmp_path,candidate_modules)
    data[group][key]=value
    with pytest.raises(ValueError):
        admit(provider,data)


def test_v9_actual_session_export_cannot_contradict_report(tmp_path, candidate_modules):
    provider, data, _ = fixture_v9(tmp_path,candidate_modules)
    path = Path(data["exports"]["runtime_session.usda"]["path"])
    path.write_text(path.read_text().replace("startTimeCode = 0", "startTimeCode = -1"))
    data["exports"]["runtime_session.usda"] = file_record(path)
    with pytest.raises(ValueError, match="Owned runtime export contradicts startTimeCode"):
        admit(provider, data)


@pytest.mark.parametrize("corruption", ["original_range", "missing_override", "foreign_floor", "actual_negative_clock", "unknown_promoted", "failed_status", "unsupported_source"])
def test_v9_still_rejects_old_geometry_and_negative_actual_clock(tmp_path, candidate_modules, corruption):
    provider, data, _ = fixture_v9(tmp_path,candidate_modules)
    if corruption == "original_range":
        data["source_time_range"]["start_time_code"] = 0.
    elif corruption == "missing_override":
        del data["runtime_time_range_override"]
    elif corruption == "foreign_floor":
        legacy.edit_archive(data, "floor_support", lambda a: a["all_hit_collision_paths"].__setitem__(0, "/Foreign/Floor"))
    elif corruption == "actual_negative_clock":
        clock = data["actual_query_clock_initial"]
        clock["observation"]["timeline_s"] = -1/60
        clock["artifact"] = legacy.legacy.write_json(Path(clock["artifact"]["path"]), clock["observation"])
    elif corruption == "unknown_promoted":
        legacy.edit_archive(data, "canonical", lambda a: a["canonical_known5_native_buffer_indices"].__setitem__(-1, 0))
    elif corruption == "failed_status":
        data["status"] = "failed"
    else:
        script = Path(data["script"]["path"])
        script.write_text("# unsupported producer revision; synthetic only\n")
        data["script"] = file_record(script)
    with pytest.raises(ValueError):
        admit(provider, data)


@pytest.mark.parametrize("protocol", ["v5", "v7"])
def test_v9_dispatch_keeps_old_provider_grid_astar_and_SDF_Jacobian_exact(tmp_path, candidate_modules, protocol):
    import numpy as np
    import torch
    _, provider_module, grid_module = candidate_modules
    original = legacy.legacy.fixture_provider(tmp_path) if protocol == "v5" else legacy.fixture_v7(tmp_path)[0]
    current = provider_module.DynBenchSceneProvider.from_preflight(original._preflight_record["path"], expected_sha256=original._preflight_record["sha256"])
    current.admit_native_map(original._native["record"]["path"], expected_sha256=original._native["record"]["sha256"])
    assert current.describe() == original.describe()
    if protocol == "v5":
        start, end = [-3.45, -.05], [2.55, -.05]
    else:
        start, end = [-4.45, -3.45], [-3.45, -2.45]
    kwargs = dict(origins_xy=[start], routes_xy=[[start, end]], clearance_m=0.)
    old = legacy.NativeGridScene.from_provider(original, **kwargs)
    new = grid_module.NativeGridScene.from_provider(current, **kwargs)
    assert old.describe() == new.describe()
    assert np.array_equal(old.astar_path(start, end), new.astar_path(start, end))
    for dtype in (torch.float32, torch.float64):
        a = torch.tensor([start, end, [-100., -100.]], dtype=dtype, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        va, vb = old.signed_distance(a), new.signed_distance(b)
        assert torch.equal(va, vb)
        assert torch.equal(torch.autograd.grad(va.sum(), a)[0], torch.autograd.grad(vb.sum(), b)[0])



@pytest.mark.parametrize("key,old,new",[("sublayer","@","@/foreign"),("fps","framesPerSecond = 60","framesPerSecond = 24"),("axis","upAxis = \"Z\"","upAxis = \"Y\""),("default","defaultPrim = \"Root\"","defaultPrim = \"Other\"")])
def test_v9_exported_owned_root_metadata_cannot_drift(tmp_path,candidate_modules,key,old,new):
    provider,data,_=fixture_v9(tmp_path,candidate_modules)
    path=Path(data["exports"]["runtime_root.usda"]["path"])
    path.write_text(path.read_text().replace(old,new,1))
    data["exports"]["runtime_root.usda"]=file_record(path)
    with pytest.raises(ValueError):
        admit(provider,data)


@pytest.mark.parametrize("where",["root_prim","source_xform","source_points","root_digest"])
def test_v9_rejects_spatial_layer_content_even_after_artifact_rehash(tmp_path,candidate_modules,where):
    provider,data,_=fixture_v9(tmp_path,candidate_modules)
    from pxr import Sdf, Usd, UsdGeom, Gf
    key="runtime_root.usda" if where in ("root_prim","root_digest") else "runtime_session.usda"
    path=Path(data["exports"][key]["path"])
    layer=Sdf.Layer.FindOrOpen(str(path))
    stage=Usd.Stage.Open(layer)
    if where=="root_prim":
        UsdGeom.Xform.Define(stage,"/Extra")
    elif where=="source_xform":
        UsdGeom.Xformable(stage.OverridePrim("/Root")).AddTranslateOp().Set(Gf.Vec3d(1.,2.,3.))
    elif where=="source_points":
        UsdGeom.Mesh(stage.OverridePrim("/Root/Wall")).CreatePointsAttr([Gf.Vec3f(1.,2.,3.)])
    else:
        stage.SetMetadata("comment","modified after actual query")
    layer.Save()
    data["exports"][key]=file_record(path)
    with pytest.raises(ValueError):
        admit(provider,data)


def test_v9_keeps_v8_provider_grid_values_and_jacobians_exact(tmp_path,candidate_modules):
    import torch
    import numpy as np
    _,provider_module,grid_module=candidate_modules
    original,data,_=prior.fixture_v8(tmp_path,(prior_verifier,prior_provider,prior_grid))
    prior.admit(original,data)
    current=provider_module.DynBenchSceneProvider.from_preflight(original._preflight_record["path"],expected_sha256=original._preflight_record["sha256"])
    current.admit_native_map(original._native["record"]["path"],expected_sha256=original._native["record"]["sha256"])
    assert current.describe()==original.describe()
    start,end=[-4.45,-3.45],[-3.45,-2.45]
    kwargs=dict(origins_xy=[start],routes_xy=[[start,end]],clearance_m=0.)
    old,new=prior_grid.NativeGridScene.from_provider(original,**kwargs),grid_module.NativeGridScene.from_provider(current,**kwargs)
    assert old.describe()==new.describe()
    assert np.array_equal(old.astar_path(start,end),new.astar_path(start,end))
    for dtype in (torch.float32,torch.float64):
        a=torch.tensor([start,end,[-100.,-100.]],dtype=dtype,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        va,vb=old.signed_distance(a),new.signed_distance(b)
        assert torch.equal(va,vb)
        assert torch.equal(torch.autograd.grad(va.sum(),a)[0],torch.autograd.grad(vb.sum(),b)[0])


def test_v9_rechecks_mutated_export_bytes_instead_of_reusing_cached_Sdf_layer(tmp_path,candidate_modules):
    verifier,_,_=candidate_modules
    provider,data,_=fixture_v9(tmp_path,candidate_modules)
    from pxr import Sdf
    path=Path(data["exports"]["runtime_session.usda"]["path"])
    old=Sdf.Layer.FindOrOpen(str(path))
    verifier._session_range_evidence(provider,data)
    original_RAM=old.ExportToString()
    path.write_text(path.read_text()+'\nover "Root"\n{\n    double3 xformOp:translate = (100, 0, 0)\n}\n')
    data["exports"]["runtime_session.usda"]=file_record(path)
    assert old.ExportToString()==original_RAM
    with pytest.raises(ValueError,match="source geometry or properties"):
        verifier._session_range_evidence(provider,data)
