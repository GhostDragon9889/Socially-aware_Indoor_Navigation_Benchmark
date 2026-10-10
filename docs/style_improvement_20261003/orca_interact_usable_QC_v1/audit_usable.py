"""All1000 Interact dataset QC; NumPy/loader only, no models or resampling rerun."""
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = HERE.parent
STAGE = DOC / "orca_interact_usable_staging_v1"


def main():
    started = time.perf_counter()
    sys.path[:0] = [str(ROOT), str(STAGE)]
    from convert_official_raw import EMPTY_LAYOUT_SHA, HEADING_SPEED_EPSILON, record
    from interaction_corpus import InteractionORCACorpus

    owner_path = DOC / "orca_interact_usable_actual_runs_v1/conversion_actual_launch.json"
    owner = json.loads(owner_path.read_text())
    assert record(owner_path)["sha256"] == "8e54ff32fbbb449bc126d0e22a83afb9d3653fc02e63c667c484dda958aed9f8"
    assert owner["schema"] == "local_orca_interact.actual_usable_conversion.v1" and owner["status"] == "closed"
    assert type(owner["exit_code"]) is int and owner["exit_code"] == 0
    assert record(owner["converter"]["path"]) == owner["converter"]
    assert record(owner["source_binding"]["path"]) == owner["source_binding"]
    manifest_path = Path(owner["manifest"]["path"])
    assert record(manifest_path) == owner["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "navisaaclab.orca_interaction_adapter.v1" and manifest["condition_mode"] == "interaction_only"
    assert manifest["max_neighbors"] == 19 and manifest["history_frames"] == 31 and manifest["future_frames"] == 52
    assert manifest["map_condition"]["map_condition_mask"] is False
    for row in manifest["sources"].values():
        assert record(row["path"]) == row
    raw_qc = json.loads(Path(manifest["sources"]["actual_QC"]["path"]).read_text())
    splits = json.loads(Path(manifest["sources"]["registered_scene_splits"]["path"]).read_text())
    raw_seal = json.loads(Path(manifest["sources"]["raw_root_seal"]["path"]).read_text())
    original = {row["scene_id"]: row for row in raw_qc["all_scene_records"]}
    sealed_raw = {row["path"]: row for row in raw_seal["raw_files"]}
    assert len(original) == len(sealed_raw) == 1000
    expected_splits = {"train": 800, "validation": 100, "test": 100}
    assert set(manifest["splits"]) == set(expected_splits)
    seen_scenes, seen_episodes, seen_raw_paths = set(), set(), set()
    counts, window_counts, zero_scenes = Counter(), Counter(), Counter()
    agent_histogram = Counter()
    total_agents = total_agent_frames = heading_valid_count = 0
    radius_min, radius_max = float("inf"), float("-inf")
    dt_min, dt_max = float("inf"), float("-inf")
    max_real_neighbors = 0
    max_neighbor_window = None
    episode_rows = []
    expected_files = {manifest_path}
    for split, entry in manifest["splits"].items():
        assert len(entry["scenes"]) == expected_splits[split]
        for scene in entry["scenes"]:
            original_name = scene["original_scene_id"]
            row = original[original_name]
            original_split = "val" if split == "validation" else split
            assert splits["scene_split_map"][original_name] == row["split"] == scene["original_registered_split"] == original_split
            assert scene["scene_id"] == "Interact_" + original_name and scene["scene_id"] not in seen_scenes
            sealed_identity = {key: sealed_raw[row["sim_npz"]["path"]][key] for key in ("path", "bytes", "sha256")}
            assert scene["source_raw"] == row["sim_npz"] == sealed_identity
            raw_path = Path(scene["source_raw"]["path"])
            assert raw_path not in seen_raw_paths and not raw_path.is_symlink() and raw_path.stat().st_nlink == 1 and raw_path.stat().st_mode & 0o222 == 0
            assert record(raw_path) == scene["source_raw"]
            seen_raw_paths.add(raw_path)
            seen_scenes.add(scene["scene_id"])
            assert scene["map"] is None and scene["map_condition_mask"] is False and scene["obstacles"] == []
            assert scene["layout_sha256"] == EMPTY_LAYOUT_SHA and len(scene["episodes"]) == 1
            episode = scene["episodes"][0]
            assert episode["episode_id"] not in seen_episodes and episode["episode_id"] == scene["scene_id"] + "_episode0"
            seen_episodes.add(episode["episode_id"])
            path = manifest_path.parent / scene["folder"] / episode["path"]
            assert path not in expected_files and not path.is_symlink() and path.stat().st_nlink == 1
            identity = record(path)
            assert (identity["bytes"], identity["sha256"]) == (episode["bytes"], episode["sha256"])
            expected_files.add(path)
            with np.load(path, allow_pickle=False) as archive:
                assert set(archive.files) == {"states", "times", "conditional_windows", "velocity_heading_valid"}
                states, times, windows, heading = (archive[key] for key in ("states", "times", "conditional_windows", "velocity_heading_valid"))
            agents = episode["agents"]
            assert agents == row["agents"] and 1 <= agents <= 20 and states.shape == (agents, 100, 5) and states.dtype == np.float32
            assert np.isfinite(states).all() and times.dtype == np.float64 and np.array_equal(times, np.arange(100, dtype=np.float64) / 10)
            dt_min, dt_max = min(dt_min, float(np.diff(times).min())), max(dt_max, float(np.diff(times).max()))
            assert heading.shape == (agents, 100) and heading.dtype == bool
            speed_valid = np.linalg.norm(states[..., 2:4], axis=-1) > HEADING_SPEED_EPSILON
            assert np.array_equal(heading, speed_valid)
            expected_yaw = np.where(heading, np.arctan2(states[..., 3], states[..., 2]), 0.)
            assert np.array_equal(states[..., 4], expected_yaw)
            radii = np.asarray(episode["agent_radii_m"])
            assert radii.shape == (agents,) and np.isfinite(radii).all() and (radii > 0).all()
            with np.load(raw_path, allow_pickle=False) as raw:
                assert np.array_equal(radii, raw["radius"].reshape(-1))
                initial = np.concatenate((raw["initPos"], raw["initVelocity"]), axis=1).astype(np.float32)
                assert np.array_equal(states[:, 0, :4], initial)
            assert row["first_time_s"] > 0. and row["last_time_s"] >= times[-1]
            assert windows.dtype == np.int32 and windows.ndim == 2 and windows.shape[1] == 2 and len(windows) == episode["conditional_window_count"]
            assert len(np.unique(windows, axis=0)) == len(windows)
            assert not ((windows[:, 0] < 0) | (windows[:, 0] >= agents) | (windows[:, 1] < 30) | (windows[:, 1] + 52 >= 100)).any()
            if len(windows):
                ego_ids, anchors = windows.T
                distances = np.linalg.norm(states[:, anchors, :2].transpose(1, 0, 2) - states[ego_ids, anchors, :2][:, None], axis=-1)
                distances[np.arange(len(windows)), ego_ids] = np.inf
                neighbor_counts = (distances <= 15.).sum(axis=1)
                assert ((neighbor_counts >= 1) & (neighbor_counts <= 19)).all()
                maximum = int(neighbor_counts.max())
                if maximum > max_real_neighbors:
                    max_real_neighbors = maximum
                    selected = int(np.argmax(neighbor_counts))
                    max_neighbor_window = {"split": split, "scene_id": scene["scene_id"], "agent": int(ego_ids[selected]), "frame": int(anchors[selected])}
            counts[split] += 1
            window_counts[split] += len(windows)
            zero_scenes[split] += int(len(windows) == 0)
            agent_histogram[agents] += 1
            total_agents += agents
            total_agent_frames += agents * 100
            heading_valid_count += int(heading.sum())
            radius_min, radius_max = min(radius_min, float(radii.min())), max(radius_max, float(radii.max()))
            episode_rows.append({"scene_id": scene["scene_id"], "split": split, "original_scene_id": original_name,
                                 "episode_id": episode["episode_id"], "agents": agents, "frames": 100,
                                 "windows": len(windows), "converted_npz": identity, "original_raw": scene["source_raw"]})
    assert len(seen_scenes) == len(seen_episodes) == len(seen_raw_paths) == 1000 and dict(counts) == expected_splits
    assert {path for path in manifest_path.parent.rglob("*") if path.is_file()} == expected_files
    batches = []
    for split in manifest["splits"]:
        corpus = InteractionORCACorpus(manifest_path, split, seed=7)
        assert len(corpus.windows) == window_counts[split] and len(corpus.episodes) == expected_splits[split]
        assert len(corpus.maps) == expected_splits[split]
        selected_indices = np.asarray([0, len(corpus.windows) - 1])
        sample = corpus.numpy_batch(2, indices=selected_indices)
        assert sample["states"].shape == (2, 5) and sample["history"].shape == (2, 31, 5) and sample["future"].shape == (2, 52, 5)
        assert sample["neighbors"].shape == (2, 19, 31, 5) and sample["neighbor_mask"].shape == (2, 19, 31)
        assert sample["neighbor_mask"][:, :, -1].any(axis=1).all()
        assert sample["raster"].shape == (2, 2, 224, 224) and (sample["raster"] == .5).all()
        assert sample["map_condition_mask"].dtype == bool and not sample["map_condition_mask"].any()
        batches.append({"split": split, "loader_windows": len(corpus.windows), "retained_episodes": len(corpus.episodes),
                       "batch_shapes": {key: list(value.shape) for key, value in sample.items() if isinstance(value, np.ndarray)},
                       "selected_window_indices": selected_indices.tolist(), "selected_frame_indices": sample["frame_indices"].tolist(),
                       "map_mask": sample["map_condition_mask"].tolist(), "current_real_neighbor_slots": sample["neighbor_mask"][:, :, -1].sum(1).tolist()})
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules)
    report = {"schema": "local_orca_interact.actual_usable_dataset_QC.v1", "status": "passed_all1000_numpy_dataset_and_loader_contract",
              "execution_source": record(__file__), "conversion_owner": record(owner_path), "source_binding": owner["source_binding"],
              "manifest": record(manifest_path), "original_raw_seal": manifest["sources"]["raw_root_seal"],
              "registered_splits": manifest["sources"]["registered_scene_splits"], "scene_counts": dict(counts), "episodes": 1000,
              "total_agents": total_agents, "total_agent_frames": total_agent_frames, "files": len(expected_files),
              "eligible_window_counts": dict(window_counts), "zero_window_scene_counts": dict(zero_scenes),
              "agent_count_histogram": dict(sorted(agent_histogram.items())), "actual_radius_m_range": [radius_min, radius_max],
              "times": {"frames": 100, "first_s": 0., "last_s": 9.9, "declared_dt_s": .1, "actual_float64_interval_range_s": [dt_min, dt_max]},
              "velocity_heading_valid_agent_frames": heading_valid_count, "stationary_proxy_zero_agent_frames": total_agent_frames - heading_valid_count,
              "all_initial_states_match_official_init": True, "all_heading_valid_and_yaw_match_actual_velocity_proxy": True,
              "all_empty_maps_and_false_masks": True, "real_max_neighbor_slots": max_real_neighbors, "max_neighbor_example": max_neighbor_window,
              "actual_numpy_batches": batches, "all_episode_records": episode_rows,
              "all_registered_splits_and_raw_SHA_mapping_exact": True, "all_scene_and_episode_ids_unique": True,
              "original_raw_files_unchanged_readonly_nlink1": True, "converted_file_modes_changed_by_auditor": False,
              "resampling_converter_reexecuted": False, "rendering_or_archive_created": False, "project_NN_constructed": False,
              "models_or_weights_read": False, "training_executed": False, "SDK_started": False, "GPU_used": False,
              "author_dataset": False, "paper_or_physical_quality_confirmed": False, "CPU_elapsed_s": time.perf_counter() - started}
    with (HERE / "actual_QC.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "scene_counts": dict(counts), "windows": dict(window_counts),
                      "zero_windows": dict(zero_scenes), "files": len(expected_files), "total_agent_frames": total_agent_frames,
                      "max_real_neighbors": max_real_neighbors, "radius_range_m": [radius_min, radius_max],
                      "report": record(HERE / "actual_QC.json"), "CPU_elapsed_s": report["CPU_elapsed_s"]}))


if __name__ == "__main__":
    main()
