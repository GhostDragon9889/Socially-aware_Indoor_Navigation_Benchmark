"""Once-only full official Maps file QC; no NN, conversion or old raw QC rerun."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = HERE.parent
STAGE = DOC / "orca_maps_official_usable_staging_v1"
OWNER = DOC / "orca_maps_official_usable_actual_runs_v1/conversion_actual_launch.json"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    started = time.perf_counter()
    owner_record = record(OWNER)
    assert owner_record["sha256"] == "df61fe07c71c3eede8e229675b9ccbafe08f20f8344cbee3b8b7d966e2a12a5a"
    owner = json.loads(OWNER.read_text())
    assert owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
    assert owner["actual_child_exit_observed"] is True and owner["actual_root_session_id"] == 45405
    assert owner["actual_launch_chunk"] == "65d430" and owner["actual_completion_chunk"] == "e60c2a"
    assert owner["converter"] == record(STAGE / "convert_official_maps.py")
    manifest_path = Path(owner["manifest"]["path"])
    assert record(manifest_path) == owner["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "navisaaclab.orca_official_maps_adapter.v1" and manifest["condition_mode"] == "official_maps"
    assert (manifest["history_frames"], manifest["future_frames"], manifest["max_neighbors"], manifest["source_max_agents"]) == (31, 52, 19, 10)
    assert manifest["map_condition"]["map_condition_mask"] is True
    for row in manifest["sources"].values():
        assert record(row["path"]) == row
    specification = importlib.util.spec_from_file_location("actual_official_Maps_converter_source_only", STAGE / "convert_official_maps.py")
    converter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(converter)
    assert Path(converter.resample_raw.__code__.co_filename).resolve() == converter.INTERACT_CONVERTER.resolve()
    assert Path(converter.interaction_windows.__code__.co_filename).resolve() == converter.INTERACT_CONVERTER.resolve()
    raw_qc = json.loads(Path(manifest["sources"]["actual_QC"]["path"]).read_text())
    raw_seal = json.loads(Path(manifest["sources"]["raw_root_seal"]["path"]).read_text())
    splits = json.loads(Path(manifest["sources"]["registered_scene_splits"]["path"]).read_text())
    raw_rows = {row["scene_id"]: row for row in raw_qc["all_scene_records"]}
    raw_sealed = {row["path"]: row for row in raw_seal["files"]}
    assert len(raw_rows) == 1000 and len(raw_sealed) == 2000
    expected_counts = {"train": 800, "validation": 100, "test": 100}
    assert set(manifest["splits"]) == set(expected_counts)
    counts, windows_counts, zero_counts, empty_counts = Counter(), Counter(), Counter(), Counter()
    totals = Counter()
    agent_counts, layouts, occupied_counts = Counter(), Counter(), []
    seen_scenes, seen_episodes, seen_raw = set(), set(), set()
    files = {manifest_path: record(manifest_path)}
    scene_records = []
    maximum_neighbors = total_map_geometry_probe_pixels = 0
    axis = np.linspace(0, 1079, 33).round().astype(np.int64)
    probe_y, probe_x = np.meshgrid(axis, axis, indexing="ij")
    for split, entry in manifest["splits"].items():
        assert len(entry["scenes"]) == expected_counts[split]
        for scene in entry["scenes"]:
            name = scene["original_scene_id"]
            row = raw_rows[name]
            raw_split = "val" if split == "validation" else split
            assert row["split"] == splits["scene_split_map"][name] == scene["original_registered_split"] == raw_split
            assert scene["scene_id"] == "Maps_" + name and scene["scene_id"] not in seen_scenes
            assert scene["map_present"] is True and scene["map_condition_mask"] is True and scene["map"] is not None
            seen_scenes.add(scene["scene_id"])
            for field, original_key in (("source_raw", "sim_npz"), ("source_map_pkl", "map_pkl")):
                raw_record = row[original_key]
                assert scene[field] == raw_record == raw_sealed[raw_record["path"]] == record(raw_record["path"])
                p = Path(raw_record["path"])
                assert p not in seen_raw and not p.is_symlink() and p.stat().st_nlink == 1 and p.stat().st_mode & 0o222 == 0
                seen_raw.add(p)
            obstacles, _ = converter.geometry.decode_map(scene["source_map_pkl"]["path"])
            # JSON preserves coordinates but converts the original point tuples to lists.
            assert json.dumps(scene["obstacles"], separators=(",", ":")) == json.dumps(obstacles, separators=(",", ":"))
            actual_layout_sha = hashlib.sha256(json.dumps(obstacles, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            assert scene["layout_sha256"] == actual_layout_sha == row["layout_sha256"]
            layouts[actual_layout_sha] += 1
            folder = manifest_path.parent / scene["folder"]
            map_path = folder / scene["map"]["path"]
            map_record = record(map_path)
            assert (map_record["bytes"], map_record["sha256"]) == (scene["map"]["bytes"], scene["map"]["sha256"])
            assert map_path not in files and not map_path.is_symlink() and map_path.stat().st_nlink == 1
            files[map_path] = map_record
            with np.load(map_path, allow_pickle=False) as archive:
                assert archive.files == ["raster"]
                raster = archive["raster"]
            assert raster.shape == (2, 1080, 1080) and raster.dtype == np.uint8 and ((raster == 0) | (raster == 1)).all()
            assert np.array_equal(raster[0], 1 - raster[1])
            occupied = int(raster[1].sum())
            occupied_counts.append(occupied)
            if not obstacles:
                assert occupied == 0 and (raster[0] == 1).all()
                empty_counts[split] += 1
            else:
                assert occupied > 0
                union = shapely.union_all([Polygon(poly) for poly in obstacles])
                vertex_pixels = np.rint(np.concatenate([np.asarray(poly) for poly in obstacles]) * 12 + 540).astype(np.int64)
                valid_pixels = vertex_pixels[(vertex_pixels >= 0).all(1) & (vertex_pixels < 1080).all(1)]
                x = np.r_[probe_x.ravel(), valid_pixels[:, 0]]
                y = np.r_[probe_y.ravel(), valid_pixels[:, 1]]
                expected = shapely.intersects_xy(union, (x - 540) / 12, (y - 540) / 12)
                assert np.array_equal(raster[1, y, x].astype(bool), expected)
                total_map_geometry_probe_pixels += len(x)
            assert len(scene["episodes"]) == 1
            episode = scene["episodes"][0]
            assert episode["episode_id"] == scene["scene_id"] + "_episode0" and episode["episode_id"] not in seen_episodes
            seen_episodes.add(episode["episode_id"])
            path = folder / episode["path"]
            identity = record(path)
            assert path not in files and not path.is_symlink() and path.stat().st_nlink == 1
            assert (identity["bytes"], identity["sha256"]) == (episode["bytes"], episode["sha256"])
            files[path] = identity
            with np.load(path, allow_pickle=False) as archive:
                assert set(archive.files) == {"states", "times", "conditional_windows", "velocity_heading_valid", "recorded_times", "radii"}
                states, times, windows, heading, recorded_times, radii = (archive[key] for key in
                    ("states", "times", "conditional_windows", "velocity_heading_valid", "recorded_times", "radii"))
            agents = episode["agents"]
            assert agents == row["agents"] and 1 <= agents <= 10 and states.shape == (agents, 100, 5) and states.dtype == np.float32
            assert np.isfinite(states).all() and times.dtype == np.float64 and np.array_equal(times, np.arange(100, dtype=np.float64) / 10)
            assert heading.shape == (agents, 100) and heading.dtype == bool
            valid_heading = np.linalg.norm(states[..., 2:4], axis=-1) > converter.interact.HEADING_SPEED_EPSILON
            assert np.array_equal(heading, valid_heading)
            assert np.array_equal(states[..., 4], np.where(heading, np.arctan2(states[..., 3], states[..., 2]), 0.))
            assert radii.dtype == np.float64 and np.array_equal(radii, np.asarray(episode["agent_radii_m"]))
            with np.load(scene["source_raw"]["path"], allow_pickle=False) as raw:
                assert np.array_equal(radii, raw["radius"].reshape(-1)) and np.array_equal(radii, np.full(agents, .4))
                assert np.array_equal(recorded_times, raw["trackTime"]) and recorded_times.dtype == np.float64
                assert np.array_equal(states[:, 0, :4], np.concatenate((raw["initPos"], raw["initVelocity"]), axis=1).astype(np.float32))
            assert recorded_times[0] > 0 and (np.diff(recorded_times) > 0).all() and times[-1] <= recorded_times[-1]
            assert windows.dtype == np.int32 and windows.ndim == 2 and windows.shape[1] == 2 and len(windows) == episode["conditional_window_count"]
            assert len(np.unique(windows, axis=0)) == len(windows)
            assert not ((windows[:, 0] < 0) | (windows[:, 0] >= agents) | (windows[:, 1] < 30) | (windows[:, 1] + 52 >= 100)).any()
            all_agents, all_frames = np.meshgrid(np.arange(agents), np.arange(30, 48), indexing="ij")
            all_agents, all_frames = all_agents.ravel(), all_frames.ravel()
            distances = np.linalg.norm(states[:, all_frames, :2].transpose(1, 0, 2) - states[all_agents, all_frames, :2][:, None], axis=-1)
            distances[np.arange(len(all_agents)), all_agents] = np.inf
            neighbor_counts = (distances <= 15.).sum(axis=1)
            eligible = neighbor_counts > 0
            expected_windows = np.stack((all_agents[eligible], all_frames[eligible]), axis=-1).astype(np.int32)
            assert np.array_equal(windows, expected_windows)
            maximum_neighbors = max(maximum_neighbors, int(neighbor_counts.max()))
            counts[split] += 1
            windows_counts[split] += len(windows)
            zero_counts[split] += int(not len(windows))
            agent_counts[agents] += 1
            totals["agents"] += agents
            totals["agent_frames"] += agents * 100
            totals["heading_valid_agent_frames"] += int(heading.sum())
            scene_records.append({"scene_id": scene["scene_id"], "original_scene_id": name, "split": split,
                "agents": agents, "frames": 100, "windows": len(windows), "obstacles": len(obstacles), "occupied_global_pixels": occupied,
                "map_presence": True, "layout_sha256": actual_layout_sha, "converted_episode": identity, "converted_map": map_record})
    assert dict(counts) == expected_counts and len(seen_scenes) == len(seen_episodes) == 1000 and len(seen_raw) == 2000
    assert sum(empty_counts.values()) == raw_qc["layout_counts"]["empty_maps"] == 45
    assert {p for p in manifest_path.parent.rglob("*") if p.is_file()} == set(files) and len(files) == 2001
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "local_orca_maps.actual_usable_dataset_QC.v1", "status": "passed_all1000_actual_files_time_window_map_contract",
        "execution_source": record(__file__), "conversion_owner": owner_record, "manifest": record(manifest_path),
        "scene_counts": dict(counts), "files": len(files), "all_converted_file_records": list(files.values()),
        "eligible_window_counts": dict(windows_counts), "zero_window_scene_counts": dict(zero_counts),
        "empty_map_counts": dict(empty_counts), "unique_geometry": len(layouts), "geometry_unique_required": False,
        "agent_count_histogram": dict(sorted(agent_counts.items())), "totals": dict(totals),
        "map_raster": {"shape": [2, 1080, 1080], "dtype": "uint8", "all_maps_present_true_including_empty": True,
            "all_complementary_binary_semantics": True, "occupied_pixel_count_range": [min(occupied_counts), max(occupied_counts)],
            "actual_polygon_pixel_probes": total_map_geometry_probe_pixels, "all_nonempty_maps_geometry_probed": True,
            "full_all_pixel_geometry_rerasterization": False, "official_cv2_boundary_pixel_identical_claimed": False},
        "all_true_initial_states_radii_and_original_timestamps_match": True,
        "all_uniform_times_t0_through9_9_match": True, "all_velocity_heading_proxies_match": True,
        "all_eligible_real_neighbor_windows_present_exact": True, "max_real_neighbors": maximum_neighbors,
        "all_original_scene_splits_preserved": True, "no_visible_obstacle_or_collision_filter": True,
        "all_raw_files_unchanged_readonly": True, "converted_files_modified_or_chmod_by_auditor": False,
        "original_Interact_resample_and_window_function_source_objects": True, "resampling_converter_reexecuted": False,
        "NN_or_loader_model_constructed": False, "training_executed": False, "weights_or_models_read": False,
        "SDK_started": False, "GPU_used": False, "sealed34_outputs_read": False, "author_dataset": False,
        "trajectory_collision_free_or_paper_quality_claimed": False, "all_scene_records": scene_records,
        "CPU_elapsed_s": time.perf_counter() - started}
    with (HERE / "actual_QC.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "scene_counts", "files", "eligible_window_counts", "zero_window_scene_counts", "empty_map_counts", "totals", "max_real_neighbors", "CPU_elapsed_s")}))


if __name__ == "__main__":
    main()
