"""Thin official Maps adapter: original time resampling, real maps, neighbor windows."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCHEMA = "navisaaclab.orca_official_maps_adapter.v1"
INTERACT_CONVERTER = HERE.parent / "orca_interact_usable_staging_v1/convert_official_raw.py"
RAW_DECODER = HERE.parent / "orca_maps_official_raw_QC_v1/audit_raw.py"
RASTER_SOURCE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1/code/CrowdSim/humanoid/orca_corpus.py"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_source(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# Reuse these actual function objects rather than reimplement time/window math.
interact = load_source("official_maps_shared_Interact_resampling", INTERACT_CONVERTER)
resample_raw, interaction_windows = interact.resample_raw, interact.interaction_windows
geometry = load_source("official_maps_raw_primitive_decoder", RAW_DECODER)
maps = load_source("official_maps_existing_global_raster", RASTER_SOURCE)
HISTORY, FUTURE, MAX_NEIGHBORS = interact.HISTORY, interact.FUTURE, interact.MAX_NEIGHBORS


def registered_inputs(actual_qc, registered_splits, raw_root_seal):
    if any(stat.S_IMODE(Path(p).stat().st_mode) != 0o444 for p in (actual_qc, registered_splits, raw_root_seal)):
        raise ValueError("Registered raw QC/split/root seal must be readonly")
    qc, splits, seal = (json.loads(Path(p).read_text()) for p in (actual_qc, registered_splits, raw_root_seal))
    if (seal["schema"] != "local_orca_maps.actual_raw_data_root_seal.v1"
            or seal["status"] != "all2000_actual_QC_files_verified_and_sealed"
            or seal["file_count"] != 2000 or record(actual_qc) != seal["QC"]):
        raise ValueError("Actual completed Maps raw seal and QC differ")
    for field in ("actual_CPU_closure", "actual_generation_owner"):
        entry = seal[field]
        if record(entry["path"]) != entry or stat.S_IMODE(Path(entry["path"]).stat().st_mode) != 0o444:
            raise ValueError("Closed CPU generation/QC identity differs")
    if qc["status"] != "passed_raw_numeric_map_and_split_contract" or qc["raw_file_count"] != 2000:
        raise ValueError("Passed full1000 numeric/maps QC required; no collision-free claim")
    if record(registered_splits) != qc["registered_scene_split"]:
        raise ValueError("Preassigned Random0 split differs")
    rows = qc["all_scene_records"]
    if len(rows) != 1000 or {r["scene_id"]: r["split"] for r in rows} != splits["scene_split_map"]:
        raise ValueError("Preserve all registered scenes and split identities")
    expected = {r[k]["path"]: r[k] for r in rows for k in ("sim_npz", "map_pkl")}
    actual = {r["path"]: r for r in seal["files"]}
    if len(expected) != 2000 or len(actual) != 2000 or actual != expected:
        raise ValueError("Raw seal must cover every sim and map, including empty maps")
    if any(stat.S_IMODE(Path(p).stat().st_mode) != 0o444 or Path(p).is_symlink() for p in expected):
        raise ValueError("Sealed raw files must be readonly real files")
    return rows, {"actual_QC": record(actual_qc), "registered_scene_splits": record(registered_splits),
                  "raw_root_seal": record(raw_root_seal), "actual_generation_owner": seal["actual_generation_owner"]}


def convert_selected(rows, output, source_records):
    """No scene filtering: zero-window/empty/overlap scenes stay in the manifest."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "author_dataset": False, "condition_mode": "official_maps",
        "state_order": ["world_x", "world_y", "world_vx", "world_vy", "velocity_heading_yaw"],
        "history_frames": HISTORY, "future_frames": FUTURE, "data_hz": 10,
        "max_neighbors": MAX_NEIGHBORS, "source_max_agents": 10,
        "resampling": {"target_frames": 100, "target_time_s": [0., 9.9], "actual_init_state_at_t0": True,
            "position_and_velocity": "original Interact linear interpolation at true recorded timestamps",
            "extrapolation": False, "heading": "atan2 interpolated velocity; stationary=0 proxy, not observed orientation",
            "stationary_speed_threshold_mps": interact.HEADING_SPEED_EPSILON},
        "map_condition": {"map_condition_mask": True, "empty_maps_are_present": True,
            "empty_layouts_may_repeat": True, "raster_unknown_fill": .5,
            "global_shape": [2, 1080, 1080], "pixels_per_m": 12, "global_width_m": 90,
            "rasterization": "existing pinned ORCACorpus Shapely polygon occupancy at global pixel coordinates",
            "official_cv2_fillPoly_boundary_pixels_identical_claimed": False},
        "window_rule": {"criterion": "original Interact current real neighbor within15m",
            "visible_obstacle_required": False, "geometry_or_collision_filter": False,
            "source_scene_split_preserved": True},
        "sources": {**source_records, "converter": record(__file__), "shared_Interact_resampling_windows": record(INTERACT_CONVERTER),
                    "raw_map_decoder": record(RAW_DECODER), "shared_local_raster_source": record(RASTER_SOURCE)},
        "splits": {name: {"scenes": []} for name in ("train", "validation", "test")}}
    seen, totals = set(), Counter()
    for row in rows:
        name = row["scene_id"]
        if name in seen or row["split"] not in ("train", "val", "test") or not 1 <= row["agents"] <= 10:
            raise ValueError("Repeated source scene or wrong official split/agent count")
        seen.add(name)
        raw, raw_map = row["sim_npz"], row["map_pkl"]
        if record(raw["path"]) != raw or record(raw_map["path"]) != raw_map:
            raise ValueError("Raw data identity changed after actual QC")
        sampled = resample_raw(raw["path"])
        obstacles, _ = geometry.decode_map(raw_map["path"])
        layout_sha = hashlib.sha256(json.dumps(obstacles, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if layout_sha != row["layout_sha256"] or len(obstacles) != row["obstacle_count"]:
            raise ValueError("Real map geometry differs from actual QC")
        windows = interaction_windows(sampled["states"])
        split = "validation" if row["split"] == "val" else row["split"]
        folder = output / split / name
        folder.mkdir(parents=True)
        target = folder / "episode.npz"
        np.savez_compressed(target, states=sampled["states"], times=sampled["times"], conditional_windows=windows,
                            velocity_heading_valid=sampled["velocity_heading_valid"], recorded_times=sampled["recorded_times"],
                            radii=sampled["radii"])
        target_record = record(target)
        target_record["path"] = target.name
        map_path = folder / "map.npz"
        np.savez_compressed(map_path, raster=maps.rasterize_obstacles(obstacles))
        map_record = record(map_path)
        map_record["path"] = map_path.name
        episode = {**target_record, "episode_id": "Maps_" + name + "_episode0", "agents": len(sampled["states"]),
                   "frames": 100, "conditional_window_count": len(windows), "agent_radii_m": sampled["radii"].tolist()}
        manifest["splits"][split]["scenes"].append({"scene_id": "Maps_" + name, "original_scene_id": name,
            "original_registered_split": row["split"], "folder": str(folder.relative_to(output)),
            "layout_sha256": layout_sha, "obstacles": obstacles, "map": map_record,
            "map_present": True, "map_condition_mask": True,
            "source_raw": raw, "source_map_pkl": raw_map, "episodes": [episode]})
        totals[split] += 1
        totals[split + "_windows"] += len(windows)
        totals["zero_window_scenes"] += int(len(windows) == 0)
        totals["empty_map_scenes"] += int(not obstacles)
    manifest["conversion_counts"] = dict(totals)
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output / "manifest.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-QC", type=Path, required=True)
    parser.add_argument("--registered-splits", type=Path, required=True)
    parser.add_argument("--raw-root-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows, sources = registered_inputs(args.actual_QC, args.registered_splits, args.raw_root_seal)
    if args.output.is_symlink() or args.output.exists():
        raise FileExistsError("New usable output must be absent; never overwrite original data")
    if args.prepare_only:
        print(json.dumps({"status": "prepared_source_and_actual_raw_identity_only", "raw_scenes": len(rows),
                          "schema": SCHEMA, "output": str(args.output.resolve()), "sources": sources,
                          "conversion_executed": False, "no_torch_SDK_GPU": True}))
    else:
        path = convert_selected(rows, args.output, sources)
        print(json.dumps({"manifest": record(path), "all_scene_conversion": True}))
    if any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules):
        raise RuntimeError("Maps conversion must not import Torch/SDK/GPU")


if __name__ == "__main__":
    main()
