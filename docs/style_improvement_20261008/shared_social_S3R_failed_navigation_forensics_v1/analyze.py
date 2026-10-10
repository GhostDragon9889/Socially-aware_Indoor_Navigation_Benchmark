"""Describe saved arrays from the actual -11 navigation; never admit success."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = ROOT / "output/evaluation/shared_social_S3R_budget12500_v2"
NAV = RUN / "navigation200s"
OWNER = ROOT / "docs/style_improvement_20261008/shared_social_S3R_evaluation_v2/completion.json"
paths = [RUN / "execution.json", OWNER, NAV / "result.json", NAV / "physical_states.npz",
         NAV / "ground_support.npz", NAV / "reference_plans.npz"]
inputs = [record(path) for path in paths]
pipeline = json.loads(paths[0].read_text())
completion = json.loads(OWNER.read_text())
result = json.loads(paths[2].read_text())
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules
assert pipeline["status"] == "failed" and pipeline["steps"][-1]["name"] == "navigation200s"
assert type(pipeline["steps"][-1]["actual_exit_code"]) is int and pipeline["steps"][-1]["actual_exit_code"] == -11
assert completion["status"] == "failed_preserved" and completion["steps"][0]["actual_exit_code"] == 1
assert result["actual_control_steps"] == 6000
finite_arrays = 0
for path in paths[3:]:
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.isfinite(arrays[key]).all(), (path, key)
                finite_arrays += 1
with np.load(paths[3], allow_pickle=False) as arrays:
    root = arrays["root_pos"]
    assert root.shape == (6000, 4, 3)
    falls = arrays["terminated"]
    peer = arrays["peer_contact"]
    static = arrays["static_contact"]
    route = arrays["route_vertex_reached"]
    assert falls.shape == static.shape == route.shape == (6000, 4)
    assert peer.shape == (6000, 4, 4)
    assert all(flags.dtype == np.bool_ for flags in (falls, peer, static, route))
    assert not np.diagonal(peer, axis1=1, axis2=2).any()
    assert result["physical_contact_measurement"]["threshold_n"] == 1.0
    assert np.array_equal(peer, arrays["peer_contact_max_force_n"] > 1.0)
    assert np.array_equal(static, arrays["static_contact_max_force_n"] > 1.0)
    assert int(falls.sum()) == result["fall_resets"]
    assert route.sum(axis=0).tolist() == result["route_vertices_reached"]
    episode = arrays["episode_ids"]
    same = np.diff(episode, axis=0) == 0
    speed = np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1) * 30
    unordered = np.logical_or(peer, peer.swapaxes(1, 2))[:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]]
    quarters = []
    for start in (0, 1500, 3000, 4500):
        selection = same[max(start, 1) - 1:start + 1499]
        values = speed[max(start, 1) - 1:start + 1499][selection]
        quarters.append({"seconds": [start / 30, (start + 1500) / 30],
                         "same_episode_intervals": int(values.size), "mean_root_XY_speed_mps": float(values.mean()),
                         "root_XY_speed_under_0_1_fraction": float((values < .1).mean())})
    measured = {"captured_controls": 6000, "captured_actor_controls": 24000,
                "fall_termination_flags": int(falls.sum()), "route_vertices_by_actor": route.sum(axis=0).tolist(),
                "static_contact_actor_controls": int(static.sum()), "unordered_peer_pair_controls": int(unordered.sum()),
                "same_episode_intervals": int(same.sum()), "mean_root_XY_speed_mps": float(speed[same].mean()),
                "root_XY_speed_under_0_1_fraction": float((speed[same] < .1).mean()), "quarters": quarters}
assert all(record(row["path"]) == row for row in inputs)
assert "torch" not in sys.modules
report = {"schema": "shared_social.failed_navigation_raw_array_forensics.v1", "status": "completed_CPU_forensics",
          "source": record(__file__), "inputs": inputs, "finite_float_arrays_checked": finite_arrays,
          "actual_navigation_exit_code": -11, "actual_parent_pipeline_exit_code": 1,
          "original_raw_CPU_success_audit_executed": False, "measurements": measured,
          "SDK_process_success_admitted": False, "locomotion_quality_or_paper_equivalence_certified": False,
          "Torch_imported": False, "SDK_started": False, "default_model_promoted": False}
target = HERE / "result.json"
with target.open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
target.chmod(0o444)
print(json.dumps({"status": report["status"], "result": record(target), "measurements": measured,
                  "actual_navigation_exit_code": -11, "SDK_process_success_admitted": False}))
