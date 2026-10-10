"""Independent CPU audit of newly published local ORCA Maps data.

Only saved 10Hz states are measured. The producer's 60Hz QA is bound to
source/log/closed execution and never represented as an independent replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/diffusion_amp/orca_maps_expanded_1000_v1"
LAUNCH = ROOT / "docs/style_improvement_20261003/orca_maps_expansion_v1/generation_actual_launch.json"
MANIFEST_SHA = "97e1d6390b15aa9e72adddcb6cf57a843d4a1c8440d1fb0ed88d7daf1bf97bdd"
GEOMETRY_TOL = 1e-4


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular file required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    require(all(not value.dtype.hasobject and value.dtype.kind in "fiu" and np.isfinite(value).all()
                for value in values.values()), "Unsafe/nonfinite NPZ: " + str(path))
    return values


def disc_gaps(states, radii, union):
    """All saved 10Hz circle-circle and signed circle-polygon clearances."""
    xy = states[..., :2].astype(np.float64)
    count = len(xy)
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    distance -= radii[:, None, None].astype(np.float64) + radii[None, :, None].astype(np.float64)
    pair = distance[np.triu_indices(count, 1)]
    points = shapely.points(xy)
    signed = shapely.distance(points, union)
    inside = shapely.intersects(union, points)
    signed[inside] = -shapely.distance(points[inside], union.boundary)
    obstacle = signed - radii[:, None].astype(np.float64)
    return {"checked_hz": 10, "saved_frames": xy.shape[1],
            "min_pair_surface_gap_m": float(pair.min()),
            "min_obstacle_surface_gap_m": float(obstacle.min()),
            "pair_overlap_sample_count": int((pair < -GEOMETRY_TOL).sum()),
            "obstacle_overlap_sample_count": int((obstacle < -GEOMETRY_TOL).sum())}


def visible_raster_samples(raster, current, stride):
    """Check genuine crop lattice points; coarse witnesses then exact fallback."""
    row, col = np.meshgrid(np.arange(0, 224, stride), np.arange(0, 224, stride), indexing="ij")
    x, y = (col.ravel() - 56) / 12., (row.ravel() - 112) / 12.
    cosine, sine = np.cos(current[:, 4]).astype(np.float64), np.sin(current[:, 4]).astype(np.float64)
    wx = current[:, 0, None] + cosine[:, None] * x - sine[:, None] * y
    wy = current[:, 1, None] + sine[:, None] * x + cosine[:, None] * y
    px = np.floor(wx * 12 + 540 + .5).astype(np.int64)
    py = np.floor(wy * 12 + 540 + .5).astype(np.int64)
    inside = (px >= 0) & (py >= 0) & (px < 1080) & (py < 1080)
    return ((raster[1, np.clip(py, 0, 1079), np.clip(px, 0, 1079)] == 1) & inside).any(axis=1)


def check_windows(states, windows, raster, union):
    """Recompute every legal neighbor/map condition, without importing producer."""
    count, frames, _ = states.shape
    agent, frame = np.meshgrid(np.arange(count), np.arange(30, frames - 52), indexing="ij")
    agent, frame = agent.ravel(), frame.ravel()
    current = states[agent, frame]
    distance = np.linalg.norm(states[:, frame, :2].transpose(1, 0, 2) - current[:, None, :2], axis=-1)
    distance[np.arange(len(agent)), agent] = np.inf
    has_neighbor = (distance <= 15.).any(axis=-1)
    corners = np.array([[-56, -112], [167, -112], [167, 111], [-56, 111]], dtype=np.float64) / 12
    cosine, sine = np.cos(current[:, 4]), np.sin(current[:, 4])
    # Preserve the producer's float32 yaw trigonometry followed by float64
    # crop corners so exact boundary selection is checked, not retuned.
    rotation = np.stack((cosine, -sine, sine, cosine), axis=-1).reshape(-1, 2, 2)
    world_corners = np.einsum("bij,kj->bki", rotation, corners) + current[:, None, :2]
    area = shapely.area(shapely.intersection(shapely.polygons(world_corners), union))
    expected = np.column_stack((agent[has_neighbor & (area >= .25)], frame[has_neighbor & (area >= .25)])).astype(np.int32)
    require(np.array_equal(windows, expected), "Stored windows differ from every legal neighbor/map-conditioned window")
    selected = states[windows[:, 0], windows[:, 1]]
    witnessed = visible_raster_samples(raster, selected, 8)
    missing = np.flatnonzero(~witnessed)
    for chunk in np.array_split(missing, max(1, int(np.ceil(len(missing) / 8)))):
        if len(chunk):
            witnessed[chunk] = visible_raster_samples(raster, selected[chunk], 1)
    require(witnessed.all(), "A declared map-conditioned crop has no genuine occupied raster sample")
    return {"all_legal_candidates": len(agent), "stored_conditional_windows": len(windows),
            "real_neighbor_and_visible_geometry_recomputed": True,
            "actual_crop_obstacle_witnesses": int(witnessed.sum()),
            "coarse_actual_lattice_witnesses": int(len(witnessed) - len(missing)),
            "full_crop_fallbacks": len(missing), "complete_window_selection_exact": True}


def audit():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only environment required")
    fixture = np.zeros((2, 3, 5), dtype=np.float32)
    fixture[0, :, :2] = [2., .5]
    fixture[1, :, :2] = [.5, .5]
    known = disc_gaps(fixture, np.full(2, .4), Polygon([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]))
    require(np.isclose(known["min_pair_surface_gap_m"], .7) and np.isclose(known["min_obstacle_surface_gap_m"], -.9)
            and known["pair_overlap_sample_count"] == 0 and known["obstacle_overlap_sample_count"] == 3,
            "Signed true-circle analytic fixture failed")
    launch_record = record(LAUNCH)
    launch = json.loads(LAUNCH.read_text())
    require(launch["status"] == "closed" and launch["exit_code"] == 0 and launch["child_exit_observed"] is True,
            "Generation must have a real closed successful process")
    manifest_record = record(DATA / "manifest.json")
    require(manifest_record == launch["manifest"] and manifest_record["sha256"] == MANIFEST_SHA, "Actual published manifest differs")
    manifest = json.loads((DATA / "manifest.json").read_text())
    require(manifest["schema"] == "navisaaclab.orca_social_reconstruction.v1" and manifest["seed"] == 20261003
            and manifest["author_dataset"] is False and manifest["simulation_hz"] == 60 and manifest["data_hz"] == 10
            and manifest["history_frames"] == 31 and manifest["future_frames"] == 52
            and manifest["state_order"] == ["world_x", "world_y", "world_vx", "world_vy", "velocity_heading_yaw"], "Dataset convention differs")
    require(launch["source_manifest"]["sha256"] == "1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21", "Generator must use the unchanged935 source")
    for row in launch["owned_sources"] + [launch["driver"], launch["source_manifest"]]:
        require(record(row["path"]) == row, "Generation source identity differs")
    require(manifest["sources"]["wrapper_sha256"] == launch["owned_sources"][1]["sha256"], "Actual generator wrapper identity differs")
    require(manifest["sources"]["trace_commit"] == "594b2af0824c5853e1a1884e167cd5ed097dd81c"
            and manifest["sources"]["rvo2_commit"] == "c2c46ba8d59556aa10faf03479293236efea154d", "Pinned source commits differ")
    files = [manifest_record]
    expected_paths = {DATA / "manifest.json"}
    for name, origin in (("TRACE-NVIDIA-License-NC.txt", ROOT / "third_party/trace/LICENSE"),
                         ("RVO2-Apache-2.0.txt", ROOT / "data/tools/orca_generator/Python-RVO2/LICENSE")):
        path = DATA / "LICENSES" / name
        require(path.read_bytes() == origin.read_bytes(), "Dataset license bytes differ")
        files.append(record(path))
        expected_paths.add(path)
    seeds, ids, layouts = set(), set(), set()
    split_rows, scene_rows = {}, []
    scene_index = 0
    for split, expected_count in (("train", 800), ("validation", 100), ("test", 100)):
        entry = manifest["splits"][split]
        require(entry["scene_count"] == len(entry["scenes"]) == entry["episode_count"] == expected_count, "Scene split count differs")
        actual_windows = 0
        for scene in entry["scenes"]:
            sid = f"scene_{scene_index:06d}"
            require(scene["scene_id"] == sid and scene["folder"] == f"{split}/{sid}" and len(scene["episodes"]) == 1, "Scene/folder split leakage")
            require(scene["scene_id"] not in ids and scene["layout_sha256"] not in layouts, "Duplicate scene or layout")
            ids.add(scene["scene_id"])
            layouts.add(scene["layout_sha256"])
            folder = DATA / scene["folder"]
            obstacle_path = folder / "obstacles.json"
            obstacle_record = record(obstacle_path)
            require(obstacle_record["sha256"] == scene["obstacles_sha256"], "Obstacle file SHA differs")
            obstacles = json.loads(obstacle_path.read_text())
            canonical = json.dumps(obstacles, separators=(",", ":"))
            require(hashlib.sha256(canonical.encode()).hexdigest() == scene["layout_sha256"]
                    and 1 <= len(obstacles) == scene["obstacle_count"] <= 20, "Nonempty unique layout differs")
            polygons = [Polygon(poly) for poly in obstacles]
            require(all(poly.is_valid and not poly.is_empty and poly.area > 0 for poly in polygons), "Invalid obstacle polygon")
            union = shapely.union_all(polygons)
            map_path = folder / scene["map"]["path"]
            map_record = record(map_path)
            require(map_record["bytes"] == scene["map"]["bytes"] and map_record["sha256"] == scene["map"]["sha256"], "Map file identity differs")
            map_values = read_npz(map_path)
            require(set(map_values) == {"raster"}, "Unexpected map NPZ field")
            raster = map_values["raster"]
            require(raster.dtype == np.uint8 and raster.shape == (2, 1080, 1080)
                    and (raster <= 1).all() and np.all(raster[0] + raster[1] == 1) and raster[1].any(), "Invalid semantic map")
            # Check map/geometry coordinates on a declared deterministic global
            # lattice; this is not called a full per-pixel raster regeneration.
            yy, xx = np.meshgrid(np.arange(0, 1080, 16), np.arange(0, 1080, 16), indexing="ij")
            geometry = shapely.intersects_xy(union, (xx - 540) / 12., (yy - 540) / 12.)
            require(np.array_equal(raster[1, yy, xx] == 1, geometry), "Global raster/world geometry convention differs")
            episode = scene["episodes"][0]
            require(episode["episode_id"] == sid + "_episode_000" and episode["path"] == "episode_000.npz"
                    and episode["agents"] == 4 + scene_index % 12 and episode["frames"] == 101, "Episode/count/cadence identity differs")
            for value in (scene["scene_seed"], episode["episode_seed"]):
                require(type(value) is int and 0 <= value < 2**32 - 1 and value not in seeds, "Duplicate or invalid published seed")
                seeds.add(value)
            path = folder / episode["path"]
            episode_record = record(path)
            require(episode_record["bytes"] == episode["bytes"] and episode_record["sha256"] == episode["sha256"], "Episode file identity differs")
            values = read_npz(path)
            prop_keys = {"prop_numAgents", "prop_neighborDist", "prop_maxNeighbors", "prop_timeHorizon", "prop_timeHorizonObst",
                         "prop_radius", "prop_maxSpeed", "prop_initVelocity", "prop_prefVelocity", "prop_initPos"}
            require(set(values) == {"states", "times", "radii", "conditional_windows"} | prop_keys, "Unexpected or absent numeric episode fields")
            n = episode["agents"]
            states, times, radii, windows = (values[key] for key in ("states", "times", "radii", "conditional_windows"))
            require(states.dtype == np.float32 and states.shape == (n, 101, 5) and times.dtype == np.float64
                    and np.array_equal(times, np.arange(101, dtype=np.float64) / 10)
                    and radii.dtype == np.float32 and radii.shape == (n,) and np.allclose(radii, .4, atol=1e-7, rtol=0), "Saved 10Hz states/time/radii differ")
            require(windows.dtype == np.int32 and windows.ndim == 2 and windows.shape[1] == 2
                    and len(windows) == episode["conditional_window_count"] > 0
                    and ((windows[:, 0] >= 0) & (windows[:, 0] < n) & (windows[:, 1] >= 30) & (windows[:, 1] + 52 < 101)).all()
                    and len(np.unique(windows, axis=0)) == len(windows) and episode["all_window_count"] == n * 19, "Window crosses an episode or has invalid identity")
            require(int(values["prop_numAgents"]) == n and values["prop_initPos"].shape == (n, 2)
                    and np.array_equal(states[:, 0, :2], values["prop_initPos"].astype(np.float32))
                    and np.array_equal(states[:, 0, 2:4], values["prop_initVelocity"].astype(np.float32)), "Initial saved physical state differs from properties")
            require(np.array_equal(radii, values["prop_radius"][:, 0].astype(np.float32)), "Actual disc radius property differs")
            yaw = np.arctan2(states[..., 3], states[..., 2]).astype(np.float64)
            delta = yaw - states[..., 4].astype(np.float64)
            require(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))) < 2e-6, "Velocity-derived yaw convention differs")
            max_speed = float(np.linalg.norm(states[..., 2:4].astype(np.float64), axis=-1).max())
            require(max_speed <= 2.00001 and np.max(np.linalg.norm(np.diff(states[..., :2].astype(np.float64), axis=1), axis=-1)) <= .20001,
                    "Saved trajectories violate disc max-speed displacement bounds")
            gaps = disc_gaps(states, radii, union)
            require(gaps["pair_overlap_sample_count"] == gaps["obstacle_overlap_sample_count"] == 0, "Independent saved10Hz true-disc clearance failed")
            qa = episode["qa"]
            require(qa["checked_hz"] == 60 and qa["simulation_steps"] == 600 and qa["frames"] == 101
                    and qa["pair_overlap_sample_count"] == qa["obstacle_overlap_sample_count"] == 0
                    and qa["min_pair_surface_gap_m"] <= gaps["min_pair_surface_gap_m"] + 2e-5
                    and qa["min_obstacle_surface_gap_m"] <= gaps["min_obstacle_surface_gap_m"] + 2e-5, "Producer60Hz metadata inconsistent with saved subset")
            window_info = check_windows(states, windows, raster, union)
            actual_windows += len(windows)
            scene_rows.append({"split": split, "scene_id": sid, "scene_seed": scene["scene_seed"], "episode_seed": episode["episode_seed"],
                               "agents": n, "obstacles": len(obstacles), "layout_sha256": scene["layout_sha256"],
                               "saved10Hz_clearance": gaps, "windows": window_info,
                               "max_saved10Hz_speed_m_s": max_speed, "global_geometry_lattice_samples": int(xx.size),
                               "producer60Hz_QA_bound_not_independently_recomputed": qa})
            files.extend((obstacle_record, map_record, episode_record))
            expected_paths.update((obstacle_path, map_path, path))
            scene_index += 1
            if scene_index % 100 == 0:
                print(json.dumps({"completed_scenes": scene_index, "CPU_only": True}), flush=True)
        require(actual_windows == entry["window_count"], "Split window denominator differs")
        split_rows[split] = {"scenes": expected_count, "episodes": expected_count, "conditional_windows": actual_windows}
    rejected = manifest["rejected_episodes"]
    for row in rejected:
        require(row["scene_id"] in ids and row["episode_index"] == 0 and type(row["seed"]) is int
                and row["seed"] not in seeds and isinstance(row["reason"], str) and row["reason"], "Rejected attempt identity/reason missing or duplicated")
        seeds.add(row["seed"])
    observed = {path for path in DATA.rglob("*") if path.is_file()}
    require(observed == expected_paths and not any(path.is_symlink() for path in DATA.rglob("*")), "Unexpected/omitted/symlink corpus files")
    logs = [json.loads(line) for line in Path(launch["log"]).read_text().splitlines() if line.strip()]
    progress = [row for row in logs if "qa" in row]
    require(len(progress) == 1000 and {row["scene"] for row in progress} == ids, "Producer closed log does not cover1000 actual scenes")
    for row, checked in zip(progress, scene_rows, strict=True):
        require(row["scene"] == checked["scene_id"] and row["split"] == checked["split"]
                and row["qa"] == checked["producer60Hz_QA_bound_not_independently_recomputed"], "Real producer QA log differs from published manifest")
    require(record(LAUNCH) == launch_record and record(DATA / "manifest.json") == manifest_record, "Source metadata changed during audit")
    require(not any(name == "torch" or name == "rvo2" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules), "QC imported forbidden simulator/model runtime")
    return {"schema": "local_orca_maps.independent_saved10Hz_QC.v1", "status": "passed", "CPU_only": True,
            "GPU_used": False, "SDK_started": False, "RVO2_simulation_rerun": False, "models_read": False,
            "sealed34_outputs_read": False, "author_dataset": False, "quality_effect_proven": False,
            "manifest": manifest_record, "closed_generation": launch_record, "producer_log": record(launch["log"]),
            "reader": record(__file__), "split_statistics": split_rows, "unique_scenes": len(ids),
            "analytic_true_circle_inside_polygon_fixture_passed": True,
            "unique_layouts": len(layouts), "recorded_distinct_seeds": len(seeds), "rejected_attempts_preserved": len(rejected),
            "all_declared_files_SHA_recomputed": files, "expected_file_count": len(files), "scene_measurements": scene_rows,
            "independently_measured_hz": 10, "independent60Hz_recomputation": False,
            "measurement_scope": "Every saved10Hz actor/frame circle gap and every legal window; source/log-bound producer60Hz QA. Raster geometry checked on explicit step16 global lattice; every selected crop has actual occupied lattice witness, with full224x224 fallback when coarse witness absent.",
            "limitations": ["No unsaved60Hz states exist in published NPZ; between-sample motion clearance cannot be independently measured.",
                            "New data test split is inspected for integrity, never used for model selection/training; sealed34 pedestrian outputs not read.",
                            "Planar disc experts do not supply SMPL poses, torques or NavIsaacLab author data."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "actual_QC.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh exclusive QC output required")
    report = audit()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": "passed", "output": record(args.output), "splits": report["split_statistics"]}))


if __name__ == "__main__":
    main()
