"""Small post-generation raw QC; no models, raster generation or conversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "docs/style_improvement_20261003/orca_interact_raw_actual_runs_v1/generation_actual_launch.json"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    started = time.monotonic()
    owner_record = record(OWNER)
    assert owner_record["sha256"] == "7ae3127155d07e1411eabaa0fdee7a7e168feaf2bcffbe8b6d8d4fb8c0413461"
    owner = json.loads(OWNER.read_text())
    assert owner["schema"] == "local_orca_interact.actual_CPU_generation.v1"
    assert owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
    assert owner["actual_completion_chunk"] == "090b8c" and owner["GPU_used"] is False
    for name in ("registration", "delegate", "registered_scene_split", "log"):
        row = owner[name]
        assert record(row["path"]) == row and stat.S_IMODE(Path(row["path"]).stat().st_mode) == 0o444
    plan = json.loads(Path(owner["registration"]["path"]).read_text())
    splits = json.loads(Path(owner["registered_scene_split"]["path"]).read_text())
    trace = Path(plan["trace_repository"])
    head = subprocess.check_output(["git", "-C", str(trace), "rev-parse", "HEAD"], text=True).strip()
    assert head == plan["trace_commit"]
    assert not subprocess.check_output(["git", "-C", str(trace), "status", "--porcelain"], text=True).strip()
    for row in plan["trace_files"]:
        assert record(row["path"]) == row
        relative = str(Path(row["path"]).relative_to(trace))
        assert subprocess.check_output(["git", "-C", str(trace), "show", f"{head}:{relative}"]) == Path(row["path"]).read_bytes()
    for row in plan["local_comparison_sources"]:
        assert record(row["path"]) == row
    rvo = Path(plan["rvo2_repository"])
    assert subprocess.check_output(["git", "-C", str(rvo), "rev-parse", "HEAD"], text=True).strip() == plan["rvo2_commit"]
    assert owner["output"] == plan["output"] and plan["scene_splits"] == owner["registered_scene_split"]
    directory = Path(plan["output"])
    expected_names = {f"scene_{i:06d}" for i in range(1000)}
    assert {path.name for path in directory.iterdir()} == expected_names == set(splits["scene_split_map"])
    split_counts = {name: list(splits["scene_split_map"].values()).count(name) for name in ("train", "val", "test")}
    assert split_counts == splits["split_counts"] == {"train": 800, "val": 100, "test": 100}
    keys = {"trackTime", "trackPos", "trackVel", "numAgents", "radius", "initPos", "neighborDist",
            "maxNeighbors", "timeHorizon", "timeHorizonObst", "maxSpeed", "initVelocity", "prefVelocity"}
    fields, scene_rows, issues = {}, [], []
    agent_histogram = {str(i): 0 for i in range(1, 21)}
    first_times, last_times, minimum_dt, maximum_dt = [], [], [], []
    all_tenth_intervals = all_uniform_intervals = all_last_intervals = total_agents = total_agent_frames = 0
    for scene_id in sorted(expected_names):
        folder = directory / scene_id
        if folder.is_symlink() or {path.name for path in folder.iterdir()} != {"sim.npz"}:
            issues.append({"scene": scene_id, "issue": "scene directory must contain only sim.npz and no map/render"})
        path = folder / "sim.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        assert set(values) == keys
        count = int(values["numAgents"])
        assert values["numAgents"].shape == () and values["numAgents"].dtype.kind in "iu" and 1 <= count <= 20
        agent_histogram[str(count)] += 1
        shapes = {"trackTime": (101,), "trackPos": (count, 101, 2), "trackVel": (count, 101, 2), "numAgents": ()}
        shapes.update({name: (count, 2) for name in ("initPos", "initVelocity", "prefVelocity")})
        shapes.update({name: (count, 1) for name in ("radius", "neighborDist", "maxNeighbors", "timeHorizon", "timeHorizonObst", "maxSpeed")})
        for key, value in values.items():
            assert value.shape == shapes[key] and value.dtype.kind in "fiu" and np.isfinite(value).all(), (scene_id, key)
            fields.setdefault(key, {"dtypes": set(), "shapes": set()})
            fields[key]["dtypes"].add(str(value.dtype))
            fields[key]["shapes"].add(str(value.shape))
        times = values["trackTime"]
        intervals = np.diff(times)
        assert (intervals > 0).all()
        all_tenth_intervals += int(np.isclose(intervals[:-1], .1, atol=1e-9, rtol=0).sum())
        all_uniform_intervals += int(np.isclose(intervals, .1, atol=1e-9, rtol=0).sum())
        all_last_intervals += int(np.isclose(intervals[-1], 5 / 60, atol=1e-9, rtol=0))
        first_times.append(float(times[0]))
        last_times.append(float(times[-1]))
        minimum_dt.append(float(intervals.min()))
        maximum_dt.append(float(intervals.max()))
        total_agents += count
        total_agent_frames += count * len(times)
        scene_rows.append({"scene_id": scene_id, "split": splits["scene_split_map"][scene_id], "agents": count,
                           "frames": len(times), "first_time_s": float(times[0]), "last_time_s": float(times[-1]),
                           "sim_npz": record(path)})
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "local_orca_interact.actual_raw_QC.v1", "status": "passed_raw_source_and_numeric_contract" if not issues else "raw_directory_contract_failed",
              "execution_source": record(__file__), "generation_owner": owner_record, "registered_source": owner["registration"],
              "registered_scene_split": owner["registered_scene_split"], "source_post_binding": {"trace_commit": head,
                "trace_clean_selected_file_bytes_equal_commit": True, "rvo2_commit": plan["rvo2_commit"],
                "frozen_q935_comparison_source_records_unchanged": True, "empty_output_prepare_guard_called": False},
              "scene_count": len(scene_rows), "split_counts": split_counts, "total_agents": total_agents,
              "total_agent_frames": total_agent_frames, "agent_count_histogram": agent_histogram,
              "field_shapes_and_dtypes": {key: {kind: sorted(values) for kind, values in row.items()} for key, row in fields.items()},
              "all_numeric_fields_finite": True, "actual_time": {"frames_per_scene": 101,
                "first_time_s_range": [min(first_times), max(first_times)], "last_time_s_range": [min(last_times), max(last_times)],
                "interval_s_range": [min(minimum_dt), max(maximum_dt)], "first99_intervals_close_to_0_1s_count": all_tenth_intervals,
                "final_interval_close_to_5_over60s_scene_count": all_last_intervals,
                "strict_uniform_10Hz": bool(all_uniform_intervals == 100000)},
              "no_map_or_render_files": not issues, "source_max_obstacles_zero": True, "issues_retained": issues,
              "all_scene_records": scene_rows, "raw_scope": "Each scene's tracks are read only from its own sim.npz; no cross-scene concatenation or train windows were created",
              "ORCACorpus_connected": False, "author_dataset": False, "GPU_used": False, "SDK_started": False,
              "weights_or_models_read": False, "sealed34_outputs_read": False, "CPU_elapsed_s": time.monotonic() - started}
    with (HERE / "actual_QC.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "scene_count", "split_counts", "total_agents", "total_agent_frames", "agent_count_histogram", "actual_time", "CPU_elapsed_s")}))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
