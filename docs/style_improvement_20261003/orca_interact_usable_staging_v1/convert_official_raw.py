"""Independent Interact conversion. Import/probe is safe; full CLI is unexecuted."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "navisaaclab.orca_interaction_adapter.v1"
HISTORY, FUTURE, MAX_NEIGHBORS = 31, 52, 19
EMPTY_LAYOUT_SHA = hashlib.sha256(b"[]").hexdigest()
HEADING_SPEED_EPSILON = 1e-8


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resample_raw(path):
    """True initial state + recorded time interpolation; exactly t=0..9.9s."""
    with np.load(path, allow_pickle=False) as raw:
        times = np.asarray(raw["trackTime"], dtype=np.float64)
        position, velocity = np.asarray(raw["trackPos"]), np.asarray(raw["trackVel"])
        initial_position, initial_velocity = np.asarray(raw["initPos"]), np.asarray(raw["initVelocity"])
        radii = np.asarray(raw["radius"], dtype=np.float64).reshape(-1)
        count = int(raw["numAgents"])
    if not 1 <= count <= 20 or position.shape != velocity.shape or position.shape != (count, len(times), 2):
        raise ValueError("Expected 1..20 synchronized official agents")
    if initial_position.shape != initial_velocity.shape or initial_position.shape != (count, 2) or radii.shape != (count,):
        raise ValueError("Official initial positions, velocities and radii are required")
    if times.ndim != 1 or not np.isfinite(times).all() or times[0] <= 0 or not (np.diff(times) > 0).all():
        raise ValueError("Recorded time must increase strictly after real t=0")
    if not all(np.isfinite(value).all() for value in (position, velocity, initial_position, initial_velocity, radii)) or (radii <= 0).any():
        raise ValueError("Finite trajectories and positive real radii are required")
    support = np.r_[0., times]
    samples = np.arange(100, dtype=np.float64) / 10
    if samples[0] < support[0] or samples[-1] > support[-1]:
        raise ValueError("Uniform target must lie inside true recorded support; no extrapolation")
    source = np.concatenate((np.concatenate((initial_position[:, None], position), axis=1),
                             np.concatenate((initial_velocity[:, None], velocity), axis=1)), axis=-1)
    states = np.zeros((count, 100, 5), dtype=np.float32)
    for agent in range(count):
        for coordinate in range(4):
            states[agent, :, coordinate] = np.interp(samples, support, source[agent, :, coordinate])
    heading_valid = np.linalg.norm(states[..., 2:4], axis=-1) > HEADING_SPEED_EPSILON
    states[..., 4] = np.where(heading_valid, np.arctan2(states[..., 3], states[..., 2]), 0.)
    return {"states": states, "times": samples, "velocity_heading_valid": heading_valid,
            "radii": radii, "recorded_times": times}


def interaction_windows(states):
    """Real current neighbor within15m; no obstacle/future behavior filtering."""
    states = np.asarray(states)
    if states.ndim != 3 or states.shape[2] != 5 or not np.isfinite(states).all():
        raise ValueError("Finite five-state trajectories are required")
    agents, frames = np.meshgrid(np.arange(len(states)), np.arange(HISTORY - 1, states.shape[1] - FUTURE), indexing="ij")
    agents, frames = agents.ravel(), frames.ravel()
    if not len(agents):
        return np.empty((0, 2), dtype=np.int32)
    current = states[agents, frames, :2]
    distances = np.linalg.norm(states[:, frames, :2].transpose(1, 0, 2) - current[:, None], axis=-1)
    distances[np.arange(len(agents)), agents] = np.inf
    eligible = (distances <= 15.).any(axis=1)
    return np.stack((agents[eligible], frames[eligible]), axis=-1).astype(np.int32)


def convert_selected(rows, output, source_records):
    """Convert explicit registered rows; full publication remains root-owned."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "author_dataset": False, "condition_mode": "interaction_only",
                "state_order": ["world_x", "world_y", "world_vx", "world_vy", "velocity_heading_yaw"],
                "history_frames": HISTORY, "future_frames": FUTURE, "data_hz": 10, "max_neighbors": MAX_NEIGHBORS,
                "resampling": {"target_frames": 100, "target_time_s": [0., 9.9], "actual_init_state_at_t0": True,
                               "position_and_velocity": "linear interpolation at recorded float timestamps", "extrapolation": False,
                               "heading": "atan2 of current interpolated velocity; stationary=0 proxy, not observed physical orientation",
                               "stationary_speed_threshold_mps": HEADING_SPEED_EPSILON},
                "map_condition": {"raster_unknown_fill": .5, "map_condition_mask": False, "grid_features_masked_to_zero": True,
                                  "empty_layouts_may_repeat": True, "no_fake_obstacles": True},
                "sources": {**source_records, "converter": record(__file__)},
                "splits": {name: {"scenes": []} for name in ("train", "validation", "test")}}
    totals = Counter()
    seen = set()
    for row in rows:
        scene_name = row["scene_id"]
        if scene_name in seen or row["split"] not in ("train", "val", "test"):
            raise ValueError("Repeated scene or invalid registered split")
        seen.add(scene_name)
        raw = row["sim_npz"]
        if record(raw["path"]) != raw:
            raise ValueError("Official raw identity differs from actual QC record")
        sampled = resample_raw(raw["path"])
        windows = interaction_windows(sampled["states"])
        split = "validation" if row["split"] == "val" else row["split"]
        folder = output / split / scene_name
        folder.mkdir(parents=True)
        target = folder / "episode.npz"
        np.savez_compressed(target, states=sampled["states"], times=sampled["times"], conditional_windows=windows,
                            velocity_heading_valid=sampled["velocity_heading_valid"])
        target_record = record(target)
        target_record["path"] = target.name
        episode = {**target_record, "episode_id": "Interact_" + scene_name + "_episode0", "agents": len(sampled["states"]),
                   "frames": 100, "conditional_window_count": len(windows), "agent_radii_m": sampled["radii"].tolist()}
        manifest["splits"][split]["scenes"].append({"scene_id": "Interact_" + scene_name, "original_scene_id": scene_name,
            "original_registered_split": row["split"], "folder": str(folder.relative_to(output)),
            "layout_sha256": EMPTY_LAYOUT_SHA, "obstacles": [], "map": None, "map_condition_mask": False,
            "source_raw": raw, "episodes": [episode]})
        totals[split] += 1
        totals[split + "_windows"] += len(windows)
        totals["zero_window_scenes"] += int(len(windows) == 0)
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
    args = parser.parse_args()
    qc, splits, seal = (json.loads(path.read_text()) for path in (args.actual_QC, args.registered_splits, args.raw_root_seal))
    if record(args.actual_QC) != seal["actual_QC"] or seal["raw_file_count"] != 1000:
        raise ValueError("Actual root raw seal and QC differ")
    rows = qc["all_scene_records"]
    if len(rows) != 1000 or {row["scene_id"]: row["split"] for row in rows} != splits["scene_split_map"]:
        raise ValueError("Exact preassigned1000 scene split must be preserved")
    path = convert_selected(rows, args.output, {"actual_QC": record(args.actual_QC), "registered_scene_splits": record(args.registered_splits),
                                             "raw_root_seal": record(args.raw_root_seal)})
    print(json.dumps({"manifest": record(path), "all_scene_conversion": True}))


if __name__ == "__main__":
    main()
