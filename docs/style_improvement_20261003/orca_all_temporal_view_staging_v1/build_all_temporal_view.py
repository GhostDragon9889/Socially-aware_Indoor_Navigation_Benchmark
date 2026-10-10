"""Build independent window indices over sealed original states/maps; no NN."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

from all_temporal_corpus import (HERE, ROOT, SELECTION, VIEW_SCHEMA, record,
                                 verified_readonly, DERIVATION)

ORIGINALS = {"maps": ROOT / "data/diffusion_amp/orca_maps_official_usable_1000_v1/manifest.json",
             "interact": ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json"}
TARGETS = {name: ROOT / f"data/diffusion_amp/orca_{name}_all_temporal_1000_v1" for name in ORIGINALS}


def write_json(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o444)


def build_domain(name, source, destination):
    original_record = record(source)
    verified_readonly(original_record)
    original = json.loads(source.read_text())
    for field in ("history_frames", "future_frames", "data_hz"):
        if type(original.get(field)) is not int or original[field] != SELECTION[field]:
            raise ValueError("Actual source temporal/dt contract differs")
    destination.mkdir(parents=True, exist_ok=False)
    view = copy.deepcopy(original)
    view.update(view_schema=VIEW_SCHEMA, source_manifest=original_record, window_selection=SELECTION)
    view["window_rule"] = {"criterion": SELECTION["criterion"], "minimum_real_neighbors": 0,
                           "visible_obstacle_required": False, "geometry_or_collision_filter": False,
                           "source_scene_split_preserved": True}
    totals, scenes_report = {}, []
    all_scene_ids, all_episode_ids, all_raw_paths = set(), set(), set()
    for split, entry in view["splits"].items():
        totals[split] = {"scenes": len(entry["scenes"]), "agents": 0, "windows": 0,
                         "old_windows": 0, "zero_neighbor_windows": 0, "single_agent_scenes": 0,
                         "empty_geometry_scenes": 0, "neighbor_histogram": [0] * 20,
                         "anchor_counts": {str(anchor): 0 for anchor in range(original["history_frames"] - 1,
                                                                              100 - original["future_frames"])}}
        for scene in entry["scenes"]:
            if (scene["scene_id"] in all_scene_ids or scene["source_raw"]["path"] in all_raw_paths
                    or scene["original_registered_split"] != split):
                raise ValueError("Original scene/raw/split identity leakage")
            all_scene_ids.add(scene["scene_id"])
            all_raw_paths.add(scene["source_raw"]["path"])
            folder = (source.parent / scene["folder"]).resolve()
            scene["folder"] = str(folder)
            if scene["map"] is not None:
                artifact = {**scene["map"], "path": str(folder / scene["map"]["path"])}
                verified_readonly(artifact)
                with np.load(artifact["path"], allow_pickle=False) as data:
                    raster = data["raster"]
                if raster.dtype != np.uint8 or raster.shape != (2, 1080, 1080):
                    raise ValueError("Original semantic map ABI differs")
                for polygon in scene["obstacles"]:
                    polygon = np.asarray(polygon)
                    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
                        raise ValueError("Original polygon geometry is invalid")
            totals[split]["empty_geometry_scenes"] += int(not scene["obstacles"])
            for episode in scene["episodes"]:
                if episode["episode_id"] in all_episode_ids:
                    raise ValueError("Original episode identity leakage")
                all_episode_ids.add(episode["episode_id"])
                artifact = {"path": str(folder / episode["path"]),
                            "bytes": episode["bytes"], "sha256": episode["sha256"]}
                verified_readonly(artifact)
                with np.load(artifact["path"], allow_pickle=False) as data:
                    states, times, old_windows = data["states"], data["times"], data["conditional_windows"]
                count, frames = episode["agents"], episode["frames"]
                if (states.shape != (count, frames, 5) or frames != 100 or states.dtype != np.float32
                        or not np.isfinite(states).all() or not np.array_equal(times, np.arange(frames) / original["data_hz"])):
                    raise ValueError("Original states/time ABI differs")
                agents, anchors = np.meshgrid(np.arange(count),
                    np.arange(original["history_frames"] - 1, frames - original["future_frames"]), indexing="ij")
                windows = np.column_stack((agents.ravel(), anchors.ravel())).astype(np.int32)
                current = states[windows[:, 0], windows[:, 1], :2]
                distance = np.linalg.norm(states[:, windows[:, 1], :2].transpose(1, 0, 2) - current[:, None], axis=-1)
                distance[np.arange(len(windows)), windows[:, 0]] = np.inf
                neighbors = (distance <= SELECTION["current_neighbor_radius_m"]).sum(1)
                if ((neighbors > SELECTION["maximum_real_neighbors"]).any()
                        or not np.array_equal(old_windows, windows[neighbors > 0])
                        or len(old_windows) != episode["conditional_window_count"]):
                    raise ValueError("Original filtered windows do not match the retained subset")
                window_path = destination / split / scene["original_scene_id"] / (episode["episode_id"] + "_windows.npy")
                window_path.parent.mkdir(parents=True, exist_ok=True)
                with window_path.open("xb") as stream:
                    np.save(stream, windows, allow_pickle=False)
                window_path.chmod(0o444)
                episode["window_index"] = record(window_path)
                episode["conditional_window_count"] = len(windows)
                histogram = np.bincount(neighbors, minlength=20)
                totals[split]["agents"] += count
                totals[split]["windows"] += len(windows)
                totals[split]["old_windows"] += len(old_windows)
                totals[split]["zero_neighbor_windows"] += int(histogram[0])
                totals[split]["single_agent_scenes"] += int(count == 1)
                totals[split]["neighbor_histogram"] = (np.asarray(totals[split]["neighbor_histogram"]) + histogram).tolist()
                for anchor in totals[split]["anchor_counts"]:
                    totals[split]["anchor_counts"][anchor] += count
                scenes_report.append({"split": split, "scene_id": scene["scene_id"], "episode_id": episode["episode_id"],
                    "source_episode": artifact, "window_index": episode["window_index"],
                    "agents": count, "windows": len(windows), "old_windows": len(old_windows),
                    "neighbor_histogram": histogram.tolist(), "per_agent_windows": [int((windows[:, 0] == agent).sum()) for agent in range(count)],
                    "per_agent_zero_neighbor_windows": [int(((windows[:, 0] == agent) & (neighbors == 0)).sum()) for agent in range(count)],
                    "anchors": np.unique(windows[:, 1]).tolist(), "original_subset_exact": True})
    view["conversion_counts"] = {key: value for split, counts in totals.items()
                                for key, value in ((split, counts["scenes"]), (split + "_windows", counts["windows"]))}
    write_json(destination / "manifest.json", view)
    if record(source) != original_record:
        raise ValueError("Original manifest changed during view build")
    return {"manifest": record(destination / "manifest.json"), "source_manifest": original_record,
            "split_counts": totals, "episodes": scenes_report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE / "actual_build_QC.json")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.report.exists() or any(path.exists() for path in TARGETS.values()):
        raise ValueError("Fresh independent outputs required; preserve any previous failure")
    domains = {name: build_domain(name, ORIGINALS[name], TARGETS[name]) for name in ORIGINALS}
    assert not any(name.split(".")[0] in {"torch", "isaaclab", "isaacsim", "omni", "pxr", "warp"} for name in sys.modules)
    document = {"schema": "local_orca.all_temporal_view_actual_build_QC.v1", "status": "passed",
                "domains": domains, "source_derivation": DERIVATION,
                "source_files": [record(HERE / filename) for filename in ("build_all_temporal_view.py", "all_temporal_corpus.py")],
                "original_states_maps_rewritten": False, "NN_SDK_GPU_used": False,
                "elapsed_seconds": time.monotonic() - started}
    write_json(args.report, document)
    print(json.dumps({"status": "passed", "report": record(args.report), "elapsed_seconds": document["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
