"""Read actual paired NPZs and original metric functions; no model/DDPM/SDK."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_paired_evaluation_v1"
ACTUAL = PREP / "actual_comparison_v1"
RECEIPT = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_paired_actual_runs_v1/comparison_actual_launch.json"


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(b)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def main():
    started = time.monotonic()
    receipt = json.loads(RECEIPT.read_text())
    assert record(RECEIPT)["sha256"] == "ac6a2b7fb5e1b0b1b3c1b9d9ea0aa0d3c769253459a17e8ffa1df8befa4450f7"
    assert receipt["status"] == "closed" and type(receipt["exit_code"]) is int and receipt["exit_code"] == 0
    assert receipt["child_exit_observed"] is receipt["driver_completed"] is True
    for row in receipt["artifacts"]:
        assert record(row["path"]) == row and stat.S_IMODE(Path(row["path"]).stat().st_mode) == 0o444
    registration = json.loads((PREP / "registration.json").read_text())
    selected = registration["window_selection"]
    inputs = json.loads((ACTUAL / "actual_inputs.json").read_text())
    assert record(inputs["new_checkpoint"]["path"]) == inputs["new_checkpoint"]
    assert inputs["new_checkpoint"]["sha256"] == "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90"
    common = arrays(ACTUAL / "common_inputs.npz")
    assert len(selected) == 100 and len(set(common["scene_ids"])) == 100
    corpus_path = Path(registration["sources"]["original_metrics_entry"]["path"]).parents[1] / "CrowdSim/humanoid/orca_corpus.py"
    specification = importlib.util.spec_from_file_location("actual935_original_numpy_corpus", corpus_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    corpus = module.ORCACorpus(registration["sources"]["manifest"]["path"], "validation", device="cpu", seed=registration["seed"])
    indices = [r["global_window_index"] for r in selected]
    reconstructed = corpus.numpy_batch(100, indices=indices)
    assert common.keys() == reconstructed.keys()
    for key in common:
        assert np.array_equal(common[key], np.asarray(reconstructed[key])), key
    metric_entry = Path(registration["sources"]["original_metrics_entry"]["path"])
    assert record(metric_entry) == registration["sources"]["original_metrics_entry"]
    tree = ast.parse(metric_entry.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"polygon_surface_gap", "offline_plan_metrics"}]
    assert len(functions) == 2
    original_metrics = {"np": np, "json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(metric_entry), "exec"), original_metrics)
    metric = original_metrics["offline_plan_metrics"]
    batches, scenes = [], []
    for start in range(0, 100, registration["batch_size"]):
        stop = min(start + registration["batch_size"], 100)
        old = arrays(ACTUAL / f"old63e7_batch_{start:03d}.npz")
        new = arrays(ACTUAL / f"expanded_final5000_batch_{start:03d}.npz")
        for key in ("window_indices", "timesteps", "noise"):
            assert old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
            assert np.array_equal(old[key], new[key]) and old[key].tobytes() == new[key].tobytes(), key
        assert np.array_equal(old["window_indices"], common["window_indices"][start:stop])
        assert old["xy"].shape == new["xy"].shape == (stop - start, 51, 2)
        old_meta = json.loads((ACTUAL / f"old63e7_batch_{start:03d}.json").read_text())
        new_meta = json.loads((ACTUAL / f"expanded_final5000_batch_{start:03d}.json").read_text())
        for key in ("start", "stop", "loss_seed", "plan_seed"):
            assert old_meta[key] == new_meta[key]
        sample = {k: v[start:stop] for k, v in common.items()}
        for name, prediction, metadata in (("old", old, old_meta), ("new", new, new_meta)):
            measured = metric(corpus, sample, prediction["xy"][:, 1:])
            for key, value in measured.items():
                if isinstance(value, (int, float)):
                    assert np.isclose(value, metadata["original_open_loop_metrics"][key], atol=1e-12, rtol=0), (start, name, key)
        batches.append({"start": start, "stop": stop, "loss_noise_timesteps_index_byte_equal": True,
                        "common_GT_and_context_matches_original_numpy_corpus": True,
                        "common_plan_seed_metadata_equal": True, "saved_original_batch_metrics_reproduced_exact": True})
        for offset in range(stop - start):
            i = start + offset
            one = {k: v[i:i + 1] for k, v in common.items()}
            a = metric(corpus, one, old["xy"][offset:offset + 1, 1:])
            b = metric(corpus, one, new["xy"][offset:offset + 1, 1:])
            row = {"scene_id": str(common["scene_ids"][i]), "episode_id": str(common["episode_ids"][i]),
                   "agent_id": int(common["agent_ids"][i]), "frame": int(common["frame_indices"][i]),
                   "global_window_index": int(common["window_indices"][i])}
            for key in ("ADE_m", "FDE_m", "min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m",
                        "neighbor_collision_point_fraction", "obstacle_collision_point_fraction",
                        "neighbor_collision_trajectory_fraction", "obstacle_collision_trajectory_fraction"):
                row["old_" + key], row["new_" + key], row["new_minus_old_" + key] = a[key], b[key], b[key] - a[key]
            scenes.append(row)
    distributions = {}
    for key in ("ADE_m", "FDE_m"):
        delta = np.asarray([r["new_minus_old_" + key] for r in scenes])
        distributions[key] = {"improved": int((delta < 0).sum()), "worsened": int((delta > 0).sum()), "exact_tie": int((delta == 0).sum()),
                              "mean_old": float(np.mean([r["old_" + key] for r in scenes])),
                              "mean_new": float(np.mean([r["new_" + key] for r in scenes])),
                              "delta_quantiles_0_25_50_75_100": np.quantile(delta, [0, .25, .5, .75, 1]).tolist()}
    worst = {key: {name: min(scenes, key=lambda r: r[name + "_" + key]) for name in ("old", "new")}
             for key in ("min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m")}
    csv_path = HERE / "all100_scene_metrics.csv"
    with csv_path.open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scenes[0]))
        writer.writeheader()
        writer.writerows(scenes)
    assert not any(n.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for n in sys.modules)
    report = {"schema": "local_orca_maps.independent_actual_array_and_scene_review.v1", "status": "passed_saved_arrays_and_original_metrics",
              "actual_parent_receipt": record(RECEIPT), "actual_inputs": record(ACTUAL / "actual_inputs.json"),
              "new_checkpoint": inputs["new_checkpoint"], "old_checkpoint": inputs["old_checkpoint"],
              "common_inputs": record(ACTUAL / "common_inputs.npz"), "original_metrics_entry": record(metric_entry),
              "original_numpy_corpus_entry": record(corpus_path), "actual_batch_checks": batches,
              "all100_scene_metrics": record(csv_path), "distribution": distributions, "worst_surface_gap_scenes": worst,
              "noise_scope": "Saved arrays are loss-noise and loss timesteps. DDPM initial/reverse noise tensors were not separately captured; plan-seed equality is metadata/source evidence, not a saved-DDPM-noise byte comparison.",
              "GT_scope": "Single common GT/context archive, exactly rebuilt from the original NumPy corpus and index selection; per-model prediction archives do not duplicate GT. Same common sample slicing for both models is explicit in the registered evaluator.",
              "all100_scenes_and_worst_collision_failures_retained": True,
              "model_run_or_deserialization": False, "DDPM_rerun": False, "SDK_started": False, "GPU_used": False,
              "sealed34_outputs_read": False, "offline_is_closed_loop_physical_quality_proof": False,
              "elapsed_seconds": time.monotonic() - started}
    with (HERE / "actual_array_peer.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "windows": len(scenes), "batches": len(batches),
                      "distribution": distributions, "seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
