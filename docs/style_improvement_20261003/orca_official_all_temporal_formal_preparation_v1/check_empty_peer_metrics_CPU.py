"""One bounded NumPy proof of known-empty peers and original multi-agent math."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "orca_official_mixed_training_staging_v2"
sys.path.insert(0, str(HERE))
sys.path.append(str(FORMAL))

from metrics import offline_plan_metrics as old_metrics  # noqa: E402
from metrics_all_temporal import offline_plan_metrics  # noqa: E402


def record(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def fixture(domain, indices):
    single = np.zeros((1, 100, 5), dtype=np.float32)
    single[0, :, 0] = np.arange(100, dtype=np.float32) / 10
    multi = np.repeat(single, 2, axis=0)
    multi[1, :, 1] = .5
    episodes = [{"states": single, "scene_id": "single", "agent_radii_m": [.25]},
                {"states": multi, "scene_id": "multi", "agent_radii_m": [.25, .35]}]
    windows = np.asarray([[0, 0, 30], [1, 0, 30], [1, 1, 30]], dtype=np.int32)
    scenes = [{"scene_id": "single", "obstacles": []},
              {"scene_id": "multi", "obstacles": [[[0., 3.], [12., 3.], [12., 4.], [0., 4.]]]}]
    core = SimpleNamespace(episodes=episodes, windows=windows, split="test",
                           manifest_path=HERE / "analytic_only_no_manifest.json",
                           manifest={"splits": {"test": {"scenes": scenes}}})
    future = np.stack([episodes[windows[index, 0]]["states"][windows[index, 1], 31:83]
                       for index in indices])
    # Deliberately no conditioning peers: metrics must still use recorded peers.
    sample = {"states": future[:, 0].copy(), "future": future,
              "window_indices": np.asarray(indices, dtype=np.int64),
              "neighbor_mask": np.zeros((len(indices), 19, 31), dtype=bool)}
    return SimpleNamespace(domain=domain, corpus=core), sample, future[:, :50, :2].copy()


def main():
    results = []
    for domain in ("maps", "interact"):
        multi, sample, predicted = fixture(domain, [1, 2])
        old = old_metrics(multi, sample, predicted)
        actual = offline_plan_metrics(multi, sample, predicted)
        assert all(key in actual and actual[key] == value for key, value in old.items())
        assert actual["neighbor_collision_point_fraction"] == 1.
        assert actual["neighbor_collision_trajectory_fraction"] == 1.
        assert actual["recorded_peer_observations"]["recorded_other_agent_point_comparisons"] == 100
        results.append({"domain": domain, "case": "multi_zero_conditioning_still_recorded_peers",
                        "original_existing_keys_exact": True, "result": actual})

        single, sample, predicted = fixture(domain, [0])
        try:
            old_metrics(single, sample, predicted)
        except ValueError:
            original_empty_reduce_failure_observed = True
        else:
            raise AssertionError("Original empty-peer reduction should fail for this real function")
        actual = offline_plan_metrics(single, sample, predicted)
        assert actual["ADE_m"] == actual["FDE_m"] == 0.
        assert actual["neighbor_collision_point_fraction"] == 0.
        assert actual["neighbor_collision_trajectory_fraction"] == 0.
        assert actual["min_neighbor_surface_gap_m"] is None
        assert actual["min_neighbor_clearance_reason"] == "known_no_other_recorded_agents"
        counts = actual["recorded_peer_observations"]
        assert counts["all_ego_future_points_collision_denominator"] == 50
        assert counts["known_no_other_agent_windows"] == 1
        assert counts["recorded_other_agent_point_comparisons"] == 0
        if domain == "maps":
            assert actual["min_obstacle_surface_gap_m"] is None
            assert actual["obstacle_collision_point_fraction"] == 0.
        else:
            assert actual["obstacle_collision_point_fraction"] is None
        results.append({"domain": domain, "case": "single_known_absence_retained",
                        "original_empty_reduce_failure_observed": original_empty_reduce_failure_observed,
                        "result": actual})

        mixed, sample, predicted = fixture(domain, [0, 1])
        actual = offline_plan_metrics(mixed, sample, predicted)
        assert actual["windows"] == 2
        assert actual["ADE_m"] == actual["FDE_m"] == 0.
        assert actual["neighbor_collision_point_fraction"] == .5
        assert actual["neighbor_collision_trajectory_fraction"] == .5
        assert math.isfinite(actual["min_neighbor_surface_gap_m"])
        counts = actual["recorded_peer_observations"]
        assert counts["all_ego_future_points_collision_denominator"] == 100
        assert counts["windows_with_recorded_other_agents"] == 1
        assert counts["known_no_other_agent_windows"] == 1
        assert counts["recorded_other_agent_point_comparisons"] == 50
        results.append({"domain": domain, "case": "mixed_empty_and_collision_all_windows_retained",
                        "result": actual})
    assert not any(name.split(".")[0] in {"torch", "isaaclab", "isaacsim", "omni", "pxr", "warp"}
                   for name in sys.modules)
    document = {"schema": "official_orca.actual_empty_peer_metric_NumPy_proof.v1", "status": "passed",
                "cases": results, "existing_multi_agent_keys_exact": True,
                "all_single_and_mixed_cases_allow_nan_false_JSON": True,
                "source": [record(HERE / "metrics_all_temporal.py"), record(__file__)],
                "synthetic_analytic_arrays_not_dataset_or_model_inference": True,
                "Torch_NN_SDK_GPU_or_dataset_loaded": False,
                "paper_or_physical_effect_proven": False}
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path = HERE / "actual_empty_peer_metrics_CPU.json"
    with path.open("x") as stream:
        stream.write(text)
    print(json.dumps({"status": "passed", "cases": len(results), "report": record(path)}))


if __name__ == "__main__":
    main()
