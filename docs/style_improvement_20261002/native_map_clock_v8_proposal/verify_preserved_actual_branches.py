"""CPU actual-v5/v7 regression of staged v8 dispatch; no native-v8 execution."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    assert not torch.cuda.is_initialized()
    prefix = "CrowdSim.humanoid.runtime."
    sys.modules[prefix + "native_map_v8"] = module(prefix + "native_map_v8", DOC / "proposed/CrowdSim/humanoid/runtime/native_map_v8.py")
    candidate_provider = module(prefix + "dynbench_scene", DOC / "proposed/CrowdSim/humanoid/runtime/dynbench_scene.py")
    sys.modules[prefix + "dynbench_scene"] = candidate_provider
    candidate_grid = module(prefix + "v8_actual_CPU_grid", DOC / "proposed/CrowdSim/humanoid/runtime/native_grid_scene.py")
    cases = [
        ("v5", "Warehouse", ROOT / "output/scene_maps/native_reference_warehouse_floor_owner_v5/Warehouse/result.json", ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/probe_paper_scene_maps.py"),
        ("v7", "Warehouse", ROOT / "output/scene_maps/native_reference_four_scenes_clock_obstacles_v7/Warehouse/result.json", ROOT / "docs/style_improvement_20261002/native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py"),
        ("v7", "Full_Warehouse", ROOT / "output/scene_maps/native_reference_four_scenes_clock_obstacles_v7/Full_Warehouse/result.json", ROOT / "docs/style_improvement_20261002/native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py"),
    ]
    rows = []
    for protocol, scene, path, snapshot in cases:
        source = file_record(path)
        data = json.loads(path.read_text())
        kwargs = dict(expected_sha256=data["preflight"]["sha256"], probe_script_snapshot=snapshot)
        old = DynBenchSceneProvider.from_preflight(data["preflight"]["path"], **kwargs)
        new = candidate_provider.DynBenchSceneProvider.from_preflight(data["preflight"]["path"], **kwargs)
        assert old.admit_native_map(path, expected_sha256=source["sha256"]) == new.admit_native_map(path, expected_sha256=source["sha256"])
        labels = old._map
        from scipy.ndimage import distance_transform_edt
        free_distance = distance_transform_edt(np.pad(labels == 254, 1, constant_values=False))[1:-1, 1:-1] * .1
        walkable = (labels == 254) & (free_distance >= 2**.5 * .1)
        pixels = np.argwhere(walkable[:, :-1] & walkable[:, 1:])[0]
        centers = np.array([[pixels[0], pixels[1]], [pixels[0], pixels[1] + 1]])
        low = np.array(old.bounds_xy[0])
        xy = np.column_stack((low[0] + (centers[:, 1] + .5)*.1, low[1] + (labels.shape[0] - centers[:, 0] - .5)*.1))
        options = dict(origins_xy=[xy[0].tolist()], routes_xy=[xy.tolist()], clearance_m=0.)
        original = NativeGridScene.from_provider(old, **options)
        current = candidate_grid.NativeGridScene.from_provider(new, **options)
        assert original.describe() == current.describe()
        a, b = original.astar_path(xy[0], xy[1]), current.astar_path(xy[0], xy[1])
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        comparisons = []
        for dtype in (torch.float32, torch.float64):
            points = np.r_[xy, np.asarray(old.bounds_xy), np.array([[-1000., -1000.]])]
            x = torch.tensor(points, dtype=dtype, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            before, after = original.signed_distance(x), current.signed_distance(y)
            before_gradient = torch.autograd.grad(before.sum(), x)[0]
            after_gradient = torch.autograd.grad(after.sum(), y)[0]
            assert before.detach().numpy().tobytes() == after.detach().numpy().tobytes()
            assert before_gradient.numpy().tobytes() == after_gradient.numpy().tobytes()
            comparisons.append({"dtype": str(dtype), "values_bytewise_equal": True, "Jacobians_bytewise_equal": True})
        assert file_record(path) == source
        rows.append({"protocol": protocol, "scene": scene, "actual_historical_result": source,
                     "provider_and_grid_description_exact": True, "Astar_dtype_shape_and_bytes_equal": True,
                     "Astar_path_C_order_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                     "SDF_value_and_Jacobian_comparisons": comparisons,
                     "original_historical_result_unchanged": True})
    assert not torch.cuda.is_initialized()
    report = {"schema": "dynbench.v8.staged_dispatch_actual_v5_v7_CPU_regression.v1", "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "native_v8_executed": False,
              "current_production_files_modified": False, "rows": rows,
              "reader": file_record(__file__)}
    with (DOC / "preserved_actual_v5_v7_branches.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"actual_historical_cases": len(rows), "all_old_descriptions_paths_values_Jacobians_bytewise": True, "native_v8_executed": False}))


if __name__ == "__main__":
    main()
