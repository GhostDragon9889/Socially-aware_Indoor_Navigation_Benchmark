#!/usr/bin/env python3
"""Analyze recorded route events on CPU without loading a policy or simulator.

Indices are captured after the 30 Hz arrival check and before physical reset.
Each new episode starts a route attempt at index zero. A complete cycle means
four ordered arrivals within one episode, not four arrivals summed over falls.
Old captures lacking route event/index fields produce null route metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = {"route_vertex_reached", "route_indices", "episode_ids", "reset_mask"}


def _validate_array(value, name, shape, kind):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in kind or not np.isfinite(value).all():
        raise ValueError(f"{name} requires finite {kind} array with shape {shape}")
    return value


def _distance(root, episode, reset):
    """No displacement connects two episodes or crosses a reset marker."""
    if root is None:
        return None, None
    legal = (episode[1:] == episode[:-1]) & ~reset[1:]
    distance = np.linalg.norm(np.diff(root[..., :2].astype(np.float64), axis=0), axis=-1)
    return np.where(legal, distance, 0), legal


def analyze_arrays(arrays, runner, *, vertices_per_cycle=4):
    if isinstance(vertices_per_cycle, bool) or not isinstance(vertices_per_cycle, int) or vertices_per_cycle < 2:
        raise ValueError("vertices_per_cycle must be an integer >=2")
    missing = sorted(REQUIRED - arrays.keys())
    result = {
        "schema": "navisaaclab.route_completion_analysis.v1",
        "vertices_per_cycle": vertices_per_cycle,
        "definition": "ordered arrivals from index zero through wrap within one physical episode",
        "capture_index_timing": "post-arrival check, before physical reset",
        "reset_policy": "restart route at zero; never accumulate a cycle across episodes",
        "missing_fields": missing,
        "route_reached_total_per_actor": None,
        "complete_cycles_total": None,
        "complete_cycles_per_actor": None,
        "reported_count_match": None,
        "actors": None,
        "issues": [],
    }
    if missing:
        result["status"] = "unavailable"
        return result
    reached = np.asarray(arrays["route_vertex_reached"])
    if reached.ndim != 2 or min(reached.shape) < 1:
        raise ValueError("route_vertex_reached requires [T>=1,B>=1]")
    shape = reached.shape
    reached = _validate_array(reached, "route_vertex_reached", shape, "b")
    indices = _validate_array(arrays["route_indices"], "route_indices", shape, "iu")
    episode = _validate_array(arrays["episode_ids"], "episode_ids", shape, "iu")
    reset = _validate_array(arrays["reset_mask"], "reset_mask", shape, "b")
    if (indices >= vertices_per_cycle).any() or (indices < 0).any():
        raise ValueError("route_indices outside the declared cycle")
    terminal = {key: _validate_array(arrays[key], key, shape, "b")
                for key in ("terminated", "truncated") if key in arrays}
    root = None
    if "root_pos" in arrays:
        root = _validate_array(arrays["root_pos"], "root_pos", (*shape, 3), "fiu")
    distance, legal_distance = _distance(root, episode, reset)
    result.update(frames=shape[0], actors_count=shape[1], actors=[])
    contract = runner.get("route_completion_contract")
    result["runner_route_contract"] = contract
    if not isinstance(contract, dict) or contract.get("reset_position_counts_as_progress") is not False:
        result["issues"].append({"kind": "missing_or_incompatible_reset_progress_contract"})
    scene = runner.get("evaluation_scene") or runner.get("physics", {}).get("scene_geometry")
    if isinstance(scene, dict) and "routes_xy" in scene:
        routes = np.asarray(scene["routes_xy"])
        if routes.shape != (shape[1], vertices_per_cycle, 2):
            raise ValueError("Runner scene route dimensions differ from the declared cycle")
    for actor in range(shape[1]):
        changes = np.flatnonzero((episode[1:, actor] != episode[:-1, actor]) | reset[1:, actor]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, shape[0]]
        row = {"actor": actor, "route_vertices_reached": int(reached[:, actor].sum()),
               "complete_cycles": 0, "verified_complete_cycles_lower_bound": 0,
               "walked_distance_xy_m_excluding_resets": None if distance is None else float(distance[:, actor].sum()),
               "excluded_reset_displacements": None if legal_distance is None else int((~legal_distance[:, actor]).sum()),
               "episodes": []}
        previous_episode = None
        for start, end in zip(starts, ends):
            start, end = int(start), int(end)
            episode_id = int(episode[start, actor])
            known_start = bool(reset[start, actor])
            issues = []
            if previous_episode is not None and episode_id <= previous_episode:
                issues.append({"frame": start, "kind": "episode_id_not_increasing"})
            if start > 0 and known_start != (episode[start, actor] != episode[start-1, actor]):
                issues.append({"frame": start, "kind": "episode_reset_boundary_disagreement"})
            previous_episode = episode_id
            expected = 0 if known_start else None
            count = 0
            events = []
            for frame in range(start, end):
                observed = int(indices[frame, actor])
                arrived = bool(reached[frame, actor])
                if arrived and "terminated" in terminal and bool(terminal["terminated"][frame, actor]):
                    issues.append({"frame": frame, "kind": "arrival_on_fall_frame"})
                if expected is not None:
                    expected_post = (expected + int(arrived)) % vertices_per_cycle
                    if observed != expected_post:
                        issues.append({"frame": frame, "kind": "index_event_disagreement",
                                       "before_index": expected, "reached": arrived,
                                       "expected_post_index": expected_post, "observed_post_index": observed})
                    if arrived:
                        events.append({"frame": frame, "vertex_index": expected, "post_index": observed})
                        if expected == vertices_per_cycle - 1 and observed == 0:
                            count += 1
                elif arrived:
                    events.append({"frame": frame, "vertex_index": None, "post_index": observed})
                expected = observed
            if any(terminal[key][start:end-1, actor].any() for key in terminal):
                issues.append({"kind": "terminal_frame_followed_by_same_episode"})
            cycles = count if known_start and not issues else None
            actor_episode = {
                "episode_id": episode_id, "first_frame": start, "last_frame": end-1,
                "observed_frames": end-start, "route_attempt_start_known": known_start,
                "route_vertices_reached": int(reached[start:end, actor].sum()),
                "reached_events": events, "final_route_index": int(indices[end-1, actor]),
                "index_continuity_valid": not issues, "complete_cycles": cycles,
                "ended_by_fall": bool(terminal["terminated"][end-1, actor]) if "terminated" in terminal else None,
                "ended_by_time_limit": bool(terminal["truncated"][end-1, actor]) if "truncated" in terminal else None,
                "right_censored": (not any(bool(value[end-1, actor]) for value in terminal.values())) if terminal else None,
                "walked_distance_xy_m_excluding_resets": None if distance is None else float(distance[start:end-1, actor].sum()),
                "issues": issues,
            }
            row["episodes"].append(actor_episode)
            if cycles is None:
                row["complete_cycles"] = None
            else:
                row["verified_complete_cycles_lower_bound"] += cycles
                if row["complete_cycles"] is not None:
                    row["complete_cycles"] += cycles
            result["issues"].extend({"actor": actor, "episode_id": episode_id, **issue} for issue in issues)
        result["actors"].append(row)
    result["route_reached_total_per_actor"] = [row["route_vertices_reached"] for row in result["actors"]]
    result["complete_cycles_per_actor"] = [row["complete_cycles"] for row in result["actors"]]
    if all(count is not None for count in result["complete_cycles_per_actor"]):
        result["complete_cycles_total"] = sum(result["complete_cycles_per_actor"])
    reported = runner.get("route_vertices_reached")
    result["runner_reported_route_vertices_reached"] = reported
    if reported is not None:
        reported_array = np.asarray(reported)
        if reported_array.shape != (shape[1],) or reported_array.dtype.kind not in "iu" or (reported_array < 0).any():
            raise ValueError("Runner route_vertices_reached requires nonnegative integer [B]")
        result["reported_count_match"] = reported == result["route_reached_total_per_actor"]
        if not result["reported_count_match"]:
            result["issues"].append({"kind": "runner_count_disagreement", "captured": result["route_reached_total_per_actor"], "reported": reported})
    result["status"] = "invalid" if result["issues"] else "checked"
    return result


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def analyze_run(run_dir, *, vertices_per_cycle=4):
    run_dir = Path(run_dir)
    result_path, states_path = run_dir / "result.json", run_dir / "physical_states.npz"
    runner = json.loads(result_path.read_text())
    if runner.get("status") != "passed":
        raise ValueError("Runner has not completed successfully; do not label an incomplete capture as a completed evaluation")
    with np.load(states_path, allow_pickle=False) as archive:
        selected = REQUIRED | {"root_pos", "terminated", "truncated"}
        arrays = {key: archive[key] for key in selected & set(archive.files)}
    result = analyze_arrays(arrays, runner, vertices_per_cycle=vertices_per_cycle)
    result["sources"] = {"capture": file_record(states_path), "runner": file_record(result_path), "analyzer": file_record(__file__)}
    result["analyzed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def markdown_report(report):
    lines = ["# Recorded route completion", "", f"Analysis status: `{report['status']}`.", "",
             "A cycle is an ordered traversal of all vertices within one episode. Fall/reset restarts cannot be combined.", ""]
    if report["status"] == "unavailable":
        lines += [f"Required fields missing: {', '.join(report['missing_fields'])}. Route metrics are `null`.", ""]
    else:
        lines += [f"Captured arrival counts match runner: `{report['reported_count_match']}`.", "",
                  "| Actor | Actual arrivals | Complete cycles | Episodes | XY distance excluding resets (m) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in report["actors"]:
            distance = row["walked_distance_xy_m_excluding_resets"]
            distance_text = "null" if distance is None else f"{distance:.3f}"
            lines.append(f"| {row['actor']} | {row['route_vertices_reached']} | {row['complete_cycles']} | {len(row['episodes'])} | {distance_text} |")
        lines += ["", f"Complete cycles across actors: `{report['complete_cycles_total']}`.", "",
                  f"Continuity/contract issues: `{len(report['issues'])}`. Episode event lists and issues are in the JSON.", ""]
    lines += ["This analysis reads recorded physical positions and route flags only. It does not rerun physics, infer missing routes, or establish paper benchmark success.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vertices-per-cycle", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path, help="default: run directory/route_completion_analysis")
    args = parser.parse_args(argv)
    report = analyze_run(args.run_dir, vertices_per_cycle=args.vertices_per_cycle)
    prefix = args.output_prefix or args.run_dir / "route_completion_analysis"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    prefix.with_suffix(".md").write_text(markdown_report(report))
    print(json.dumps({key: report[key] for key in ("status", "route_reached_total_per_actor", "complete_cycles_per_actor", "reported_count_match")}))
    return int(report["status"] == "invalid")


if __name__ == "__main__":
    raise SystemExit(main())
