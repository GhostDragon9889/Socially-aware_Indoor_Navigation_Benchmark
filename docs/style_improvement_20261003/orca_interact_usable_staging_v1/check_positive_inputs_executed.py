"""Analytic plus three real scene probes; no full conversion or NN construction."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from convert_official_raw import convert_selected, interaction_windows, record, resample_raw
from interaction_corpus import InteractionORCACorpus


def main():
    started = time.perf_counter()
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append({"name": name, "passed": True})

    qc_path = HERE.parent / "orca_interact_raw_QC_v1/actual_QC.json"
    qc = json.loads(qc_path.read_text())
    selected = [next(row for row in qc["all_scene_records"] if row["split"] == split and row["agents"] == 20)
                for split in ("train", "val", "test")]
    with tempfile.TemporaryDirectory(prefix="orca_interact_positive_", dir="/tmp") as temporary:
        folder = Path(temporary)
        # Analytic irregular times retain the true initial state and straight velocity.
        times = np.r_[1 / 60 + np.arange(100) / 10, 10.]
        initial = np.asarray([[0., 0.], [0., 1.]])
        velocity = np.asarray([[1., 0.], [0., 0.]])
        raw = folder / "analytic.npz"
        np.savez(raw, trackTime=times, trackPos=initial[:, None] + velocity[:, None] * times[None, :, None],
                 trackVel=np.broadcast_to(velocity[:, None], (2, 101, 2)), initPos=initial,
                 initVelocity=velocity, radius=np.full((2, 1), .4), numAgents=2)
        analytic = resample_raw(raw)
        check("analytic100points_true_t0_no_extrap", np.array_equal(analytic["times"], np.arange(100) / 10)
              and np.array_equal(analytic["states"][:, 0, :2], initial.astype(np.float32))
              and np.allclose(analytic["states"][0, :, 0], np.arange(100) / 10, rtol=0, atol=5e-7))
        check("stationary_heading_explicit_zero_invalid", not analytic["velocity_heading_valid"][1].any()
              and (analytic["states"][1, :, 4] == 0).all())
        check("one_agent_retained_without_fake_neighbors", interaction_windows(analytic["states"][:1]).shape == (0, 2))
        dense = np.zeros((20, 100, 5), dtype=np.float32)
        dense[:, :, 0] = np.arange(20)[:, None] / 10
        check("all20_agents18valid_windows_each", len(interaction_windows(dense)) == 360)
        manifest = convert_selected(selected, folder / "partial_real_probe", {"actual_QC": record(qc_path), "scope": "3 real20-agent scenes only"})
        document = json.loads(manifest.read_text())
        check("three_registered_scene_splits_preserved", all(len(document["splits"][split]["scenes"]) == 1
              for split in ("train", "validation", "test")))
        layouts = [document["splits"][split]["scenes"][0]["layout_sha256"] for split in document["splits"]]
        check("identical_true_empty_layout_across_split_legal", len(set(layouts)) == 1)
        probes = []
        for split in document["splits"]:
            corpus = InteractionORCACorpus(manifest, split, seed=7)
            sample = corpus.numpy_batch(2, indices=np.asarray([0, len(corpus.windows) - 1]))
            check("actual_numpy_training_ABI:" + split, sample["states"].shape == (2, 5)
                  and sample["history"].shape == (2, 31, 5) and sample["future"].shape == (2, 52, 5)
                  and sample["neighbors"].shape == (2, 19, 31, 5) and sample["neighbor_mask"][:, :, -1].any(axis=1).all())
            check("explicit_no_map_inputs:" + split, (sample["raster"] == .5).all() and not sample["map_condition_mask"].any())
            probes.append({"split": split, "scene_id": corpus.episodes[0]["scene_id"], "windows": len(corpus.windows),
                           "first_and_last_selected": sample["frame_indices"].tolist(), "max_real_neighbor_slots_in_probe": int(sample["neighbor_mask"][:, :, -1].sum(1).max())})
        # Exercise the existing NumPy batching math at the maximum19-neighbor boundary.
        corpus.episodes[0]["states"] = dense
        corpus.windows = np.c_[np.zeros(360, dtype=np.int32), interaction_windows(dense)]
        dense_sample = corpus.numpy_batch(1, indices=np.asarray([0]))
        check("19_real_neighbors_not_truncated", dense_sample["neighbor_mask"].all() and len(set(dense_sample["neighbor_ids"][0])) == 19)

        # Actual staged context and encode_context functions, with explicit fake encoders.
        # No PaperDiffusionPlanner/model/optimizer constructor or checkpoint read occurs.
        spec = importlib.util.spec_from_file_location("interact_positive_paper", HERE / "paper_interact.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import torch
        torch.set_num_threads(4)
        planner = object.__new__(module.PaperDiffusionPlanner)
        planner.device = torch.device("cpu")
        batch = module.PaperDiffusionTrainBatch.from_world(planner, dense_sample["states"], dense_sample["history"],
            dense_sample["neighbors"], dense_sample["raster"], dense_sample["future"],
            history_mask=dense_sample["history_mask"], neighbor_mask=dense_sample["neighbor_mask"],
            map_condition_mask=dense_sample["map_condition_mask"])
        check("actual_train_batch_retains_false_map_mask", batch.clean_trajectory.shape == (1, 52, 5)
              and batch.context["map_condition_mask"].shape == (1,) and not batch.context["map_condition_mask"].any())
        fake = SimpleNamespace(config=module.PaperDiffusionConfig(), normalize=lambda value: value,
            ego_encoder=lambda value, mask: value.new_zeros((len(value), 128)),
            neighbor_encoder=lambda value, mask: value.new_zeros((len(value), 128)), conditioning=lambda value: value,
            map_encoder=lambda raster: (None, raster.new_ones((len(raster), 32, 56, 56))))
        masked = module.PaperDiffusionModel.encode_context(fake, batch.context)
        check("map_grid_actually_zero_with_false_mask", (masked["map_grid"] == 0).all())
        context = {**batch.context, "map_condition_mask": torch.ones(1, dtype=torch.bool)}
        present = module.PaperDiffusionModel.encode_context(fake, context)
        check("present_map_grid_preserved", (present["map_grid"] == 1).all())
        check("no_CUDA_or_SDK", not torch.cuda.is_initialized()
              and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
    report = {"schema": "local_orca_interact.usable_stage_positive_inputs.v1", "status": "passed_small_input_probes_only",
              "checks": checks, "selected_original_raw_scenes": selected, "actual_scene_probes": probes,
              "source_records": [record(HERE / name) for name in ("convert_official_raw.py", "interaction_corpus.py", "paper_interact.py", "train_interact_diffusion.py")],
              "temporary_partial_conversion_removed": True, "all1000_converted": False, "published_usable_dataset": False,
              "NN_model_constructed": False, "optimizer_or_DDPM_executed": False, "weights_or_models_read": False,
              "fake_encoders_explicit": True, "GPU_used": False, "SDK_started": False,
              "production185_or_q935_modified": False, "CPU_elapsed_s": time.perf_counter() - started}
    with (HERE / "actual_positive_inputs.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "report": record(HERE / "actual_positive_inputs.json"),
                      "CPU_elapsed_s": report["CPU_elapsed_s"]}))


if __name__ == "__main__":
    main()
