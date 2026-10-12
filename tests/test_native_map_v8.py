"""Synthetic v8 geometry/session contracts; these tests never claim native PASS."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import CrowdSim
from protomotions.repository import repository_root

from CrowdSim.humanoid.runtime.dynbench_scene import file_record
import test_native_map_v7 as legacy

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = repository_root(CrowdSim.__file__)


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
    package = "CrowdSim.humanoid.runtime."
    v8 = source_module(package + "native_map_v8", ROOT / "CrowdSim/humanoid/runtime/native_map_v8.py")
    monkeypatch.setitem(sys.modules, package + "native_map_v8", v8)
    provider = source_module(package + "dynbench_scene", ROOT / "CrowdSim/humanoid/runtime/dynbench_scene.py")
    monkeypatch.setitem(sys.modules, package + "dynbench_scene", provider)
    grid = source_module(package + "v8_CPU_grid", ROOT / "CrowdSim/humanoid/runtime/native_grid_scene.py")
    return v8, provider, grid


def fixture_v8(folder, modules):
    v8, provider_module, _ = modules
    old, low_row = legacy.fixture_v7(folder)
    data, preflight = deepcopy(old._native["data"]), deepcopy(old._data)
    script = folder / "probe_v8.py"
    shutil.copyfile(WORKSPACE / "resources/runtime_sources/style_improvement_20261002/native_map_clock_v8_proposal_v2/proposed/scripts/probe_paper_scene_maps.py", script)
    assert file_record(script)["sha256"] == v8.PROBE_SHA256
    preflight["script"] = file_record(script)
    data["script"] = preflight["script"]
    data["preflight"] = legacy.legacy.write_json(folder / "preflight.json", preflight)
    data["source_time_range"] = {
        "start_time_code": -1., "end_time_code": 0., "time_codes_per_second": 60.,
        "authored_time_range": True, "USD_time_range_modified": False,
    }
    ram = "0" * 64
    data["runtime_time_range_override"] = {
        "schema": "dynbench.native_frozen_timeline_session.v8",
        "original_source_composed_range": {"start_time_code": -1., "end_time_code": 0., "time_codes_per_second": 60.},
        "requested_runtime_start_time_code": 0., "requested_runtime_end_time_code": 60.,
        "requested_runtime_start_s": 0., "requested_runtime_end_s": 1.,
        "composed_stage_start_time_code": 0., "composed_stage_end_time_code": 60.,
        "composed_stage_start_at_query_time_code": 0., "composed_stage_end_at_query_time_code": 60.,
        "composed_stage_time_codes_per_second_at_query": 60.,
        "observed_timeline_start_after_setup_s": 0., "observed_timeline_end_after_setup_s": 1.,
        "observed_timeline_start_at_query_s": 0., "observed_timeline_end_at_query_s": 1.,
        "session_layer_identifier": "anon:synthetic_v8.usda",
        "source_root_layer_identifier": preflight["source"]["path"],
        "source_root_in_memory_unchanged": True, "source_root_in_memory_unchanged_at_query": True,
        "source_root_in_memory_sha256": ram, "source_root_in_memory_sha256_at_query": ram,
        "session_edit_target_at_query": True,
        "runtime_session_time_range_modified": True, "time_codes_per_second_modified": False,
        "source_layers_saved": False, "actual_query_clock_pass_claim": False,
    }
    session = folder / "runtime_session_v8.usda"
    session.write_text("#usda 1.0\n(\n    endTimeCode = 60\n    startTimeCode = 0\n)\n")
    data["exports"]["runtime_session.usda"] = file_record(session)
    fresh = provider_module.DynBenchSceneProvider.from_preflight(data["preflight"]["path"], expected_sha256=data["preflight"]["sha256"])
    return fresh, data, low_row


def admit(provider, data):
    path = Path(provider._preflight_record["path"]).parent / "result_v8.json"
    record = legacy.legacy.write_json(path, data)
    return provider.admit_native_map(path, expected_sha256=record["sha256"])


def test_v8_synthetic_dispatch_retains_crop_missing_API_unknown_and_low_obstacles(tmp_path, candidate_modules):
    v8, _, grid_module = candidate_modules
    provider, data, _ = fixture_v8(tmp_path, candidate_modules)
    admit(provider, data)
    grid = grid_module.NativeGridScene.from_provider(provider, origins_xy=[[-4.45, -3.45]], routes_xy=[[[-4.45, -3.45], [-3.45, -2.45]]], clearance_m=0.)
    checked = v8.verify_v8(provider, data, admitted=True)
    assert checked["admission_protocol"] == "v8_exact_probe_SHA"
    assert checked["missing_known5_API_count"] == 3
    assert checked["known5_API_complete"] is False
    assert checked["known_static_nonfloor_near_ground_blocked_known5_cells"] == 1
    assert grid._labels.shape == (20, 24)
    assert not grid.is_walkable([[-1e3, -1e3]])[0]
    assert len(grid.astar_path([-4.45, -3.45], [-3.45, -2.45])) > 1
    assert data["CPU_contract_fixture_only"] is True
    assert checked["new_physical_execution_verified"] is False


@pytest.mark.parametrize("key,value", [
    ("schema", "dynbench.native_frozen_timeline_session.v7"),
    ("requested_runtime_start_time_code", -1.),
    ("requested_runtime_end_time_code", 0.),
    ("requested_runtime_end_s", 0.),
    ("composed_stage_start_at_query_time_code", -1.),
    ("composed_stage_time_codes_per_second_at_query", 24.),
    ("observed_timeline_start_at_query_s", -1/60),
    ("observed_timeline_end_at_query_s", 0.),
    ("session_edit_target_at_query", False),
    ("source_root_in_memory_unchanged_at_query", False),
    ("source_root_in_memory_sha256_at_query", "1" * 64),
    ("source_root_in_memory_sha256", "malformed"),
    ("source_root_layer_identifier", "/wrong/source.usd"),
    ("session_layer_identifier", "/persistent/session.usda"),
    ("source_layers_saved", True),
    ("time_codes_per_second_modified", True),
    ("actual_query_clock_pass_claim", True),
])
def test_v8_rejects_range_ownership_RAM_or_invented_clock_corruption(tmp_path, candidate_modules, key, value):
    provider, data, _ = fixture_v8(tmp_path, candidate_modules)
    data["runtime_time_range_override"][key] = value
    with pytest.raises(ValueError):
        admit(provider, data)


def test_v8_actual_session_export_cannot_contradict_report(tmp_path, candidate_modules):
    provider, data, _ = fixture_v8(tmp_path, candidate_modules)
    path = Path(data["exports"]["runtime_session.usda"]["path"])
    path.write_text(path.read_text().replace("startTimeCode = 0", "startTimeCode = -1"))
    data["exports"]["runtime_session.usda"] = file_record(path)
    with pytest.raises(ValueError, match="session time metadata"):
        admit(provider, data)


@pytest.mark.parametrize("corruption", ["original_range", "missing_override", "foreign_floor", "actual_negative_clock", "unknown_promoted", "failed_status", "unsupported_source"])
def test_v8_still_rejects_old_geometry_and_negative_actual_clock(tmp_path, candidate_modules, corruption):
    provider, data, _ = fixture_v8(tmp_path, candidate_modules)
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
def test_v8_dispatch_keeps_old_provider_grid_astar_and_SDF_Jacobian_exact(tmp_path, candidate_modules, protocol):
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
