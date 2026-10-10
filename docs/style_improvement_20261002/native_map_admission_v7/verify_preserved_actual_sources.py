"""CPU v5 bytewise regression and v6 raw-layout proof, never v7 execution."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
from CrowdSim.humanoid.runtime.native_map_v7 import PROBE_SHA256, _indices
from scripts import probe_paper_scene_maps as official

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent


def before(name):
    path = DOC / "before/CrowdSim/humanoid/runtime" / (name + ".py")
    spec = importlib.util.spec_from_file_location("CrowdSim.humanoid.runtime.v7_before_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert not torch.cuda.is_initialized()
    result_path = ROOT / "output/scene_maps/native_reference_warehouse_floor_owner_v5/Warehouse/result.json"
    result = json.loads(result_path.read_text())
    snapshot = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/probe_paper_scene_maps.py"
    kwargs = {"expected_sha256": result["preflight"]["sha256"], "probe_script_snapshot": snapshot}
    provider = DynBenchSceneProvider.from_preflight(result["preflight"]["path"], **kwargs)
    old_provider = before("dynbench_scene").DynBenchSceneProvider.from_preflight(result["preflight"]["path"], **kwargs)
    record = file_record(result_path)
    current_description = provider.admit_native_map(result_path, expected_sha256=record["sha256"])
    old_description = old_provider.admit_native_map(result_path, expected_sha256=record["sha256"])
    assert json.dumps(current_description, sort_keys=True) == json.dumps(old_description, sort_keys=True)
    physical = json.loads((ROOT / "output/diffusion_amp/native_warehouse_clearance08_30s_v2/result.json").read_text())
    grid_kwargs = {"origins_xy": physical["evaluation_scene"]["origins_xy"], "routes_xy": physical["evaluation_scene"]["routes_xy"], "clearance_m": .8}
    current = NativeGridScene.from_provider(provider, **grid_kwargs)
    original = before("native_grid_scene").NativeGridScene.from_provider(provider, **grid_kwargs)
    assert json.dumps(current.describe(), sort_keys=True) == json.dumps(original.describe(), sort_keys=True)
    paths = []
    for start, route in zip(grid_kwargs["origins_xy"], grid_kwargs["routes_xy"]):
        a, b = current.astar_path(start, route[0]), original.astar_path(start, route[0])
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        paths.append({"vertices": len(a), "length_m": float(np.linalg.norm(np.diff(a, axis=0), axis=-1).sum())})
    comparisons = []
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[-12.51, -18.51], [0.371, 6.423], [-6.91, -8.95], [12.9, 18.8]], dtype=dtype, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        values, old_values = current.signed_distance(x), original.signed_distance(y)
        gradient = torch.autograd.grad(values.sum(), x)[0]
        old_gradient = torch.autograd.grad(old_values.sum(), y)[0]
        assert values.detach().numpy().tobytes() == old_values.detach().numpy().tobytes()
        assert gradient.numpy().tobytes() == old_gradient.numpy().tobytes()
        comparisons.append({"dtype": str(dtype), "values_bytewise_equal": True, "Jacobians_bytewise_equal": True})
    v5 = {"actual_resource": record, "provider_description_exact": True, "grid_description_exact": True,
          "four_actual_fixture_routes_Astar_bytewise_equal": paths, "SDF_comparisons": comparisons,
          "new_GPU_execution": False}
    readiness = json.loads((ROOT / "docs/style_improvement_20261002/paper_scene_maps_probe_readiness_v7.json").read_text())
    rows = []
    for evidence in readiness["actual_v6_evidence_preserved"]:
        record = evidence["native_omap"]
        assert file_record(record["path"]) == record
        native_result = json.loads(Path(evidence["result"]["path"]).read_text())
        assert file_record(evidence["result"]["path"]) == evidence["result"]
        with np.load(record["path"], allow_pickle=False) as raw:
            buffer, low, high, dims = (raw[key] for key in ("buffer", "min_bound", "max_bound", "dimensions"))
            occupied, api = raw["occupied_positions"], raw["free_positions"]
            oi, fi = _indices(occupied, low, dims), _indices(api, low, dims)
            assert np.array_equal(np.sort(oi), np.flatnonzero(buffer == 4))
            assert np.isin(buffer[fi], [5, 6]).all()
            canonical = official.canonical_known5_positions(buffer, api, low, high, dims, source="official_buffer_centers")
            actual_indices = _indices(canonical["positions"], low, dims)
            assert np.array_equal(actual_indices, canonical["native_buffer_indices"])
            assert np.array_equal(np.sort(actual_indices), np.flatnonzero(buffer == 5))
            data = json.loads(Path(native_result["preflight"]["path"]).read_text())
            window = official.projection_grid_window(low, high, dims, data["projection_metadata"]["bounds"], np.asarray(data["projection_metadata"]["dims"]))
            assert window == native_result["derived_export_grid"]
            rows.append({"scene": evidence["scene"], "actual_v6_result": evidence["result"], "unchanged_original_status": native_result["status"],
                "raw": record, "all_occupied_API_world_labels_equal4": True, "all_API_free_world_labels_5_or6": True,
                "all_canonical_world_centers_exact_fixed_native_indices": True, "native_known5": len(actual_indices),
                "API_missing_known5": len(canonical["missing_known5_buffer_indices"]), "lower_padding_cells_xy": window["lower_padding_cells_xy"],
                "world_crop_matches_preserved_result": True, "new_v7_native_PASS_claim": False, "runtime_map_admission_attempted": False})
    assert not torch.cuda.is_initialized()
    report = {"schema": "native_map_v7_preserved_actual_CPU_proof.v1", "CPU_only": True, "SDK_started": False, "GPU_used": False,
              "v5_actual_resource_exact_before_after": v5, "v6_actual_raw_fixed_layout_only": rows,
              "v7_probe_SHA": PROBE_SHA256, "native_v7_executed": False, "new_map_resource_admitted_from_v6": False,
              "reader": file_record(__file__)}
    (DOC / "actual_v5_and_four_scene_v6_raw_proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v5_before_after_exact": True, "actual_v6_raw_scenes": len(rows), "native_v7_executed": False}))


if __name__ == "__main__":
    main()
