"""Read every completed official Maps scene; keep empty maps and all diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import stat
import sys
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "docs/style_improvement_20261003/orca_maps_official_raw_actual_runs_v1/generation_actual_launch.json"
OWNER_SHA = "ff800d7253cf92c752e023a908dc42d865ec5062c129dadc1f15d9c1d80ff753"
KEYS = {"trackTime", "trackPos", "trackVel", "numAgents", "radius", "initPos", "neighborDist",
        "maxNeighbors", "timeHorizon", "timeHorizonObst", "maxSpeed", "initVelocity", "prefVelocity"}


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class PrimitiveUnpickler(pickle.Unpickler):
    """The registered generator saves only nested lists/tuples of numeric points."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Map cannot load a global: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Map cannot load persistent objects")


def decode_map(path):
    obstacles = PrimitiveUnpickler(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(obstacles, list) or not 0 <= len(obstacles) <= 20:
        raise ValueError("Official map is a list of0..20 polygons, including valid empty maps")
    polygons = []
    for points in obstacles:
        if not isinstance(points, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
            raise ValueError("Expected original polygon coordinate lists")
        if not all(type(x) in (float, int) for p in points for x in p):
            raise ValueError("Map coordinates must be primitive numeric values")
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 4 or not np.isfinite(array).all():
            raise ValueError("Finite closed polygon vertices required")
        if not np.array_equal(array[0], array[-1]) or len(np.unique(array[:-1], axis=0)) < 3:
            raise ValueError("Official map polygon closure or distinct vertices differ")
        polygon = Polygon(array)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise ValueError("Invalid original polygon")
        polygons.append(polygon)
    return obstacles, polygons


def geometry_diagnostics(positions, radii, polygons):
    """True disk gaps at the saved samples; these never filter official scenes."""
    positions = np.asarray(positions)
    count, frames = positions.shape[:2]
    pair_gap = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    pair_gap -= radii[:, None, None] + radii[None, :, None]
    valid = np.triu(np.ones((count, count), dtype=bool), k=1)
    selected = pair_gap[valid]
    result = {"frames": frames, "unordered_pair_observations": int(len(selected) * frames),
              "min_pair_disk_gap_m": float(selected.min()) if selected.size else None,
              "pair_overlap_observations_below_minus1e4": int((selected < -1e-4).sum())}
    if polygons:
        union = shapely.union_all(polygons)
        points = shapely.points(positions)
        distance = shapely.distance(points, union)
        inside = shapely.intersects(union, points)
        distance[inside] = -shapely.distance(points[inside], union.boundary)
        gap = distance - radii[:, None]
        result.update(min_obstacle_disk_gap_m=float(gap.min()),
                      obstacle_overlap_observations_below_minus1e4=int((gap < -1e-4).sum()),
                      obstacle_agent_observations=int(gap.size))
    else:
        result.update(min_obstacle_disk_gap_m=None, obstacle_overlap_observations_below_minus1e4=0,
                      obstacle_agent_observations=0)
    return result


def main():
    started = time.monotonic()
    owner_identity = record(OWNER)
    assert owner_identity["sha256"] == OWNER_SHA
    owner = json.loads(OWNER.read_text())
    assert owner["schema"] == "local_orca_maps.actual_CPU_generation_root_receipt.v1"
    assert owner["status"] == "observed_closed0" and owner["actual_exit_code"] == 0
    assert owner["actual_child_exit_observed"] is True and owner["actual_tool_session_id"] == 96267
    assert owner["launch_chunk_id"] == "c6b873" and owner["GPU_used"] is False
    for key in ("registration", "delegate", "actual_log"):
        row = owner[key]
        assert record(row["path"]) == row and stat.S_IMODE(Path(row["path"]).stat().st_mode) == 0o444
    plan = json.loads(Path(owner["registration"]["path"]).read_text())
    assert plan["generation_seed"] == 20261005 and plan["nominal_agent_range"] == [1, 10]
    assert owner["output"] == plan["output"]
    split_record = plan["scene_splits"]
    assert record(split_record["path"]) == split_record
    splits = json.loads(Path(split_record["path"]).read_text())
    labels = ["train"] * 800 + ["val"] * 100 + ["test"] * 100
    random.Random(0).shuffle(labels)
    split_map = {f"scene_{i:06d}": label for i, label in enumerate(labels)}
    assert split_map == splits["scene_split_map"]
    assert Counter(labels) == splits["split_counts"] == {"train": 800, "val": 100, "test": 100}
    generator_row = next(row for row in plan["trace_files"] if row["path"].endswith("/gen_dataset.py"))
    assert record(generator_row["path"]) == generator_row
    assert plan["official_raw_argv"][-1] == "--resample_goal"
    directory = Path(plan["output"])
    assert not directory.is_symlink() and {p.name for p in directory.iterdir()} == set(split_map)
    issues, rows, fields = [], [], {}
    agent_counts, obstacle_counts, vertex_counts = Counter(), Counter(), Counter()
    layouts, raw_hashes = defaultdict(list), defaultdict(list)
    time_first, time_last, time_min_dt, time_max_dt, nominal_error = [], [], [], [], []
    pair_min, obstacle_min, init_pair_min, init_obstacle_min = [], [], [], []
    totals = Counter()
    max_track_abs_position = max_track_speed = 0.
    expected_times = np.r_[np.arange(100, dtype=np.float64) / 10 + 1 / 60, 10.]
    for scene_id in sorted(split_map):
        folder = directory / scene_id
        if folder.is_symlink() or {p.name for p in folder.iterdir()} != {"sim.npz", "map.pkl"}:
            issues.append({"scene": scene_id, "issue": "Expected only sim.npz and map.pkl"})
        row = {"scene_id": scene_id, "split": split_map[scene_id],
               "sim_npz": record(folder / "sim.npz"), "map_pkl": record(folder / "map.pkl")}
        rows.append(row)
        try:
            assert not (folder / "sim.npz").is_symlink() and not (folder / "map.pkl").is_symlink()
            with np.load(folder / "sim.npz", allow_pickle=False) as saved:
                values = {key: saved[key] for key in saved.files}
            assert set(values) == KEYS
            number = values["numAgents"]
            assert number.shape == () and number.dtype.kind in "iu" and 1 <= int(number) <= 10
            count = int(number)
            shapes = {"trackTime": (101,), "trackPos": (count, 101, 2), "trackVel": (count, 101, 2), "numAgents": ()}
            shapes.update({k: (count, 2) for k in ("initPos", "initVelocity", "prefVelocity")})
            shapes.update({k: (count, 1) for k in ("radius", "neighborDist", "maxNeighbors", "timeHorizon", "timeHorizonObst", "maxSpeed")})
            for key, value in values.items():
                assert value.shape == shapes[key] and not value.dtype.hasobject and value.dtype.kind in "fiu" and np.isfinite(value).all(), key
                fields.setdefault(key, {"dtypes": set(), "shapes": set()})
                fields[key]["dtypes"].add(str(value.dtype))
                fields[key]["shapes"].add(str(value.shape))
            assert np.array_equal(values["radius"], np.full((count, 1), .4))
            assert np.array_equal(values["initVelocity"], np.zeros((count, 2)))
            assert np.array_equal(values["maxSpeed"], np.full((count, 1), 2.))
            assert np.array_equal(values["maxNeighbors"], np.full((count, 1), 10))
            assert (np.abs(values["initPos"]) <= 15.).all()
            for key, low, high in (("neighborDist", 10., 20.), ("timeHorizon", 2., 5.), ("timeHorizonObst", 2., 5.)):
                assert ((values[key] >= low) & (values[key] <= high)).all()
            preferred_speed = np.linalg.norm(values["prefVelocity"], axis=-1)
            assert ((preferred_speed >= 1. - 1e-12) & (preferred_speed <= 2. + 1e-12)).all()
            times = values["trackTime"]
            intervals = np.diff(times)
            assert (intervals > 0).all() and times[0] > 0
            assert np.allclose(times, expected_times, atol=1e-4, rtol=0)
            obstacles, polygons = decode_map(folder / "map.pkl")
            layout_sha = hashlib.sha256(json.dumps(obstacles, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            row.update(agents=count, obstacle_count=len(obstacles), polygon_vertex_counts=[len(p) for p in obstacles],
                       frames=len(times), first_time_s=float(times[0]), last_time_s=float(times[-1]),
                       layout_sha256=layout_sha, agent_radii_m=values["radius"][:, 0].tolist())
            initial_geometry = geometry_diagnostics(values["initPos"][:, None], values["radius"][:, 0], polygons)
            saved_geometry = geometry_diagnostics(values["trackPos"], values["radius"][:, 0], polygons)
            row.update(initial_geometry=initial_geometry, saved_sample_geometry=saved_geometry)
            for name, target in (("min_pair_disk_gap_m", pair_min), ("min_obstacle_disk_gap_m", obstacle_min)):
                if saved_geometry[name] is not None:
                    target.append(saved_geometry[name])
            for name, target in (("min_pair_disk_gap_m", init_pair_min), ("min_obstacle_disk_gap_m", init_obstacle_min)):
                if initial_geometry[name] is not None:
                    target.append(initial_geometry[name])
            for key in ("pair_overlap_observations_below_minus1e4", "obstacle_overlap_observations_below_minus1e4"):
                totals["initial_" + key] += initial_geometry[key]
                totals["saved_" + key] += saved_geometry[key]
            layouts[layout_sha].append({"scene_id": scene_id, "split": split_map[scene_id]})
            raw_hashes[row["sim_npz"]["sha256"]].append(scene_id)
            agent_counts[str(count)] += 1
            obstacle_counts[str(len(obstacles))] += 1
            vertex_counts.update(str(len(p)) for p in obstacles)
            totals["agents"] += count
            totals["agent_frames"] += count * len(times)
            time_first.append(float(times[0]))
            time_last.append(float(times[-1]))
            time_min_dt.append(float(intervals.min()))
            time_max_dt.append(float(intervals.max()))
            nominal_error.append(float(np.abs(times - expected_times).max()))
            max_track_abs_position = max(max_track_abs_position, float(np.abs(values["trackPos"]).max()))
            max_track_speed = max(max_track_speed, float(np.linalg.norm(values["trackVel"], axis=-1).max()))
        except (AssertionError, ValueError, pickle.UnpicklingError) as error:
            issues.append({"scene": scene_id, "exception": type(error).__name__, "issue": str(error)})
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    status = "passed_raw_numeric_map_and_split_contract" if not issues else "failed_raw_contract_all_scene_records_retained"
    report = {"schema": "local_orca_maps.actual_raw_QC.v1", "status": status,
              "execution_source": record(__file__), "generation_owner": owner_identity,
              "registered_source": owner["registration"], "registered_scene_split": split_record,
              "scene_count": len(rows), "raw_file_count": len(rows) * 2,
              "split_counts": dict(Counter(labels)), "source_generation_seed": plan["generation_seed"],
              "per_scene_seed_stored_or_independently_recovered": False,
              "agent_count_histogram": dict(sorted(agent_counts.items(), key=lambda x: int(x[0]))),
              "obstacle_count_histogram": dict(sorted(obstacle_counts.items(), key=lambda x: int(x[0]))),
              "polygon_vertex_count_histogram": dict(vertex_counts), "totals": dict(totals),
              "field_shapes_and_dtypes": {k: {kind: sorted(value) for kind, value in row.items()} for k, row in fields.items()},
              "actual_time": {"nominal_first_time_s": 1 / 60, "nominal_last_time_s": 10.,
                 "first_time_s_range": [min(time_first), max(time_first)], "last_time_s_range": [min(time_last), max(time_last)],
                 "interval_s_range": [min(time_min_dt), max(time_max_dt)], "nominal_timestamp_max_abs_error_s": max(nominal_error),
                 "first99_nominal_interval_s": .1, "last_nominal_interval_s": 5 / 60,
                 "true_initial_state_separately_saved": True, "strict_uniform_10Hz": False},
              "geometry_diagnostics_not_filter": {"sampled_states_only": True, "independent60Hz_recomputed": False,
                 "min_initial_pair_disk_gap_m": min(init_pair_min), "min_initial_obstacle_disk_gap_m": min(init_obstacle_min),
                 "min_saved_pair_disk_gap_m": min(pair_min), "min_saved_obstacle_disk_gap_m": min(obstacle_min),
                 "max_track_abs_position_m": max_track_abs_position, "max_saved_speed_mps": max_track_speed,
                 "motion_outside15m_is_legal_source_USE_BOUNDARY_false": True},
              "layout_counts": {"unique_geometry": len(layouts), "empty_maps": obstacle_counts["0"],
                 "repeated_geometries": {sha: entries for sha, entries in layouts.items() if len(entries) > 1},
                 "empty_or_duplicate_geometry_rejected": False},
              "repeated_sim_content_hashes": {sha: entries for sha, entries in raw_hashes.items() if len(entries) > 1},
              "all_scene_records": rows, "issues_retained": issues,
              "all2000_raw_file_hashes_recorded": len(rows) == 1000,
              "unsafe_pickle_globals_allowed": False, "raw_files_modified_or_chmod": False,
              "ORCACorpus_connected": False, "author_dataset": False, "GPU_used": False, "SDK_started": False,
              "weights_or_models_read": False, "sealed34_outputs_read": False,
              "CPU_elapsed_s": time.monotonic() - started}
    with (HERE / "actual_QC.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "scene_count", "raw_file_count", "split_counts", "agent_count_histogram", "obstacle_count_histogram", "totals", "actual_time", "geometry_diagnostics_not_filter", "CPU_elapsed_s")}))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
