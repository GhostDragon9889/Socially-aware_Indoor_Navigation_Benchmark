#!/usr/bin/env python3
"""CPU temporal/contact analysis of an already evaluated physical capture.

Never starts Kit, loads a policy, changes a checkpoint or decides stability.
Outputs are separate from the runner and pose-evaluator reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--phase-seconds", type=int, default=20)
    args = parser.parse_args()
    if args.phase_seconds < 1:
        parser.error("phase-seconds must be positive")
    source = args.capture_dir.resolve()
    output = source / "pedestrian_metrics"
    evaluation_path = output / "evaluation.json"
    states_path, result_path = source / "physical_states.npz", source / "result.json"
    evaluation = json.loads(evaluation_path.read_text())
    runner = json.loads(result_path.read_text())
    if evaluation["status"] != "evaluated" or evaluation["input_kind"] != "physical_rollout":
        raise ValueError("A completed actual physical-rollout evaluator report is required")
    if evaluation["sources"]["physical_states"]["sha256"] != digest(states_path):
        raise ValueError("Capture changed after its evaluator contract validation")
    if evaluation["rollout_metrics"]["actual_contacts"] is None:
        raise ValueError("This supplemental analysis requires validated recorded contacts")
    with np.load(states_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    root = arrays["root_pos"].astype(np.float64)
    frames, actors = root.shape[:2]
    frequency = runner["state_capture"]["frequency_hz"]
    if frequency != 30:
        raise ValueError("Requires the declared 30Hz physical capture")
    times = (np.arange(frames) + 1) / frequency
    same_episode = arrays["episode_ids"][1:] == arrays["episode_ids"][:-1]
    comparable = same_episode & ~arrays["reset_mask"][1:]
    distance = np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1)
    tracking = np.linalg.norm(root[..., :2] - arrays["target_xy"], axis=-1)
    route = arrays.get("route_vertex_xy")
    if route is None or route.shape != (frames, actors, 2) or not np.isfinite(route).all():
        raise ValueError("Requires finite recorded route_vertex_xy[T,B,2]")
    route_distance = np.linalg.norm(root[..., :2] - route, axis=-1)
    same_vertex = np.max(np.abs(np.diff(route, axis=0)), axis=-1) <= 1e-7
    comparable_vertex = comparable & same_vertex
    reduction = route_distance[:-1] - route_distance[1:]
    root_quaternion = arrays["root_rot"].astype(np.float64)
    up_z = 1 - 2 * np.sum(root_quaternion[..., :2] ** 2, axis=-1)
    windows_s = (.5, 1., 2.)
    terminal_rows = []
    for frame, actor in np.argwhere(arrays["terminated"]):
        episode = arrays["episode_ids"][frame, actor]
        mask = (arrays["episode_ids"][:frame + 1, actor] == episode)
        contacts = np.flatnonzero(arrays["static_contact"][:frame + 1, actor] & mask)
        peers = np.flatnonzero(arrays["peer_contact"][:frame + 1, actor].any(-1) & mask)
        window_report = {}
        for seconds in windows_s:
            begin = max(0, frame - int(round(seconds * frequency)))
            in_episode = arrays["episode_ids"][begin:frame + 1, actor] == episode
            window_report[str(seconds)] = {
                "static_contact_actor_frames": int((arrays["static_contact"][begin:frame + 1, actor] & in_episode).sum()),
                "peer_contact_actor_frames": int((arrays["peer_contact"][begin:frame + 1, actor].any(-1) & in_episode).sum())}
        terminal_rows.append({
            "frame_index": int(frame), "physical_time_s": float(times[frame]), "actor": int(actor),
            "episode_id": int(episode), "root_pos_m": root[frame, actor].tolist(),
            "pelvis_height_m": float(root[frame, actor, 2]), "root_up_z": float(up_z[frame, actor]),
            "static_contact_same_frame": bool(arrays["static_contact"][frame, actor]),
            "peer_contact_same_frame": bool(arrays["peer_contact"][frame, actor].any()),
            "previous_static_contact_lag_s": float((frame - contacts[-1]) / frequency) if contacts.size else None,
            "previous_peer_contact_lag_s": float((frame - peers[-1]) / frequency) if peers.size else None,
            "same_episode_lookback_inclusive_s": window_report})
    phase_rows = []
    phase_frames = args.phase_seconds * frequency
    for begin in range(0, frames, phase_frames):
        end = min(frames, begin + phase_frames)
        transitions = slice(begin, end - 1)
        legal = comparable[transitions]
        pair_mask = arrays["peer_contact"][begin:end]
        paired = (pair_mask | pair_mask.swapaxes(-1, -2))[:, np.triu_indices(actors, 1)[0], np.triu_indices(actors, 1)[1]]
        phase_rows.append({
            "frame_range_half_open": [begin, end], "physical_time_range_s": [begin / frequency, end / frequency],
            "actor_frames": (end - begin) * actors,
            "walked_xy_m_per_actor": np.where(legal, distance[transitions], 0).sum(0).tolist(),
            "mean_planar_speed_m_s": float(distance[transitions][legal].mean() * frequency) if legal.any() else None,
            "tracking_mean_m": float(tracking[begin:end].mean()),
            "tracking_rms_m": float(np.sqrt(np.mean(tracking[begin:end] ** 2))),
            "plan_endpoint_goal_residual_mean_m": float(arrays["plan_endpoint_goal_error"][begin:end].mean()),
            "minimum_pelvis_height_m": float(root[begin:end, :, 2].min()),
            "terminal_frames": int(arrays["terminated"][begin:end].sum()),
            "static_contact_actor_frames": int(arrays["static_contact"][begin:end].sum()),
            "peer_contact_unordered_pair_frames": int(paired.sum()),
            "route_vertex_distance_mean_m": float(route_distance[begin:end].mean())})
    static_rows = [{"frame_index": int(frame), "physical_time_s": float(times[frame]), "actor": int(actor),
                    "episode_id": int(arrays["episode_ids"][frame, actor]),
                    "root_pos_m": root[frame, actor].tolist(),
                    "maximum_body_pair_normal_force_n": float(arrays["static_contact_max_force_n"][frame, actor])}
                   for frame, actor in np.argwhere(arrays["static_contact"])]
    summary = {
        "schema": "temporal_contact_physical_capture_v1", "status": "analyzed", "device": "cpu",
        "source_sha256": {str(path): digest(path) for path in (states_path, result_path, evaluation_path, Path(__file__))},
        "timing": "frame i is the post-control physical sample at (i+1)/30 seconds; initial pre-step pose is not recorded",
        "phase_protocol": "fixed consecutive intervals; within-phase paths exclude reset/episode discontinuities",
        "phases": phase_rows, "terminal_contact_relation": terminal_rows, "static_contact_frames": static_rows,
        "recorded_route_vertex_progress": {
            "first_recorded_distance_m_per_actor": route_distance[0].tolist(),
            "last_recorded_distance_m_per_actor": route_distance[-1].tolist(),
            "minimum_recorded_distance_m_per_actor": route_distance.min(0).tolist(),
            "same_vertex_same_episode_transitions_per_actor": comparable_vertex.sum(0).tolist(),
            "signed_distance_reduction_sum_m_per_actor": np.where(comparable_vertex, reduction, 0).sum(0).tolist(),
            "route_vertex_changes_within_episode_per_actor": (same_episode & ~same_vertex).sum(0).tolist(),
            "route_vertices_reached": runner["route_vertices_reached"],
            "meaning": "Recorded full next route vertex, separate from the moving 5m A* conditioning endpoint; reset jumps excluded from reductions"},
        "limitations": ["Temporal association does not establish the cause of a fall.",
                       "Floor, self contacts and tangential friction forces are excluded by the measurement contract.",
                       "Maximum body-pair normal force is not a total body impulse or summed force.",
                       "Phase summaries are descriptive; there is no invented quality, stability or author-equivalence threshold.",
                       "A single capture cannot establish an improvement between actor or planner stages."]}
    (output / "phase_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for actor in range(actors):
        height_line = root[:, actor, 2].copy()
        route_line = route_distance[:, actor].copy()
        reset_after_start = arrays["reset_mask"][:, actor].copy()
        reset_after_start[0] = False
        height_line[reset_after_start] = np.nan
        route_line[reset_after_start] = np.nan
        axes[0].plot(times, height_line, label=f"Actor {actor}", linewidth=.8)
        axes[1].plot(times, route_line, linewidth=.8)
    axes[0].axhline(.45, color="black", linewidth=.8, linestyle="--", label="Declared fall height")
    for row in terminal_rows:
        axes[0].scatter(row["physical_time_s"], row["pelvis_height_m"], marker="x", color="black", s=55)
    for row in static_rows:
        axes[0].scatter(row["physical_time_s"], row["root_pos_m"][2], marker="o", facecolors="none", edgecolors="red", s=40)
    axes[0].set_ylabel("Pelvis height (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Recorded physical states: x = fall terminal; red circle = static normal contact")
    axes[1].set_ylabel("Full next route vertex\ndistance (m)")
    axes[2].plot(times, tracking.mean(1), linewidth=.8, label="Physical root to same-time target")
    axes[2].plot(times, arrays["plan_endpoint_goal_error"].mean(1), linewidth=.8, label="Plan endpoint to conditioning goal")
    axes[2].set_ylabel("Actor mean error (m)")
    axes[2].set_xlabel("Physical control time (s)")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output / "phase_quality.png", dpi=160)
    plt.close(fig)
    progress = summary["recorded_route_vertex_progress"]
    lines = [f"# {frames / frequency:g} 秒真实记录：时段、接触与路线进度", "",
             "本文件是 CPU 描述统计；不将时间相关性认定为跌倒原因，不判断稳定或论文一致。", "",
             f"## 每 {args.phase_seconds} 秒时段", "",
             "| 时段(s) | terminal | static actor-frames | peer pair-frames | tracking mean(m) | endpoint residual mean(m) | min pelvis(m) | mean speed(m/s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in phase_rows:
        lines.append(f"| {row['physical_time_range_s']} | {row['terminal_frames']} | {row['static_contact_actor_frames']} | {row['peer_contact_unordered_pair_frames']} | {row['tracking_mean_m']:.4f} | {row['plan_endpoint_goal_residual_mean_m']:.4f} | {row['minimum_pelvis_height_m']:.4f} | {row['mean_planar_speed_m_s']:.4f} |")
    same_frame_static = sum(row["static_contact_same_frame"] for row in terminal_rows)
    same_frame_peer = sum(row["peer_contact_same_frame"] for row in terminal_rows)
    lines += ["", "## Terminal 与接触", "",
              f"{len(terminal_rows)} 次 terminal 中，当帧 static contact={same_frame_static} 次，peer contact={same_frame_peer} 次。逐帧证据：", ""]
    for row in terminal_rows:
        static_lag = f"{row['previous_static_contact_lag_s']:g}s" if row['previous_static_contact_lag_s'] is not None else "无已记录接触"
        peer_lag = f"{row['previous_peer_contact_lag_s']:g}s" if row['previous_peer_contact_lag_s'] is not None else "无已记录接触"
        lines.append(f"- Actor {row['actor']}：frame {row['frame_index']} / {row['physical_time_s']:.4f}s，height={row['pelvis_height_m']:.4f}m，最近同 episode static contact 时间差={static_lag}，peer={peer_lag}；0.5/1/2s 前窗计数见 JSON。")
    lines += ["", "## 完整路线顶点", "",
              f"首帧距离：{progress['first_recorded_distance_m_per_actor']}m；末帧：{progress['last_recorded_distance_m_per_actor']}m。",
              f"同顶点同 episode 的距离减少累计：{progress['signed_distance_reduction_sum_m_per_actor']}m（带符号，排除 reset）。",
              f"最小距离：{progress['minimum_recorded_distance_m_per_actor']}m；route vertices reached={progress['route_vertices_reached']}。",
              "5m A* condition goal 与这里的完整 next route vertex 分开统计。有限路程和低 tracking error 不等于路线到达。", "",
              "![逐帧物理曲线](phase_quality.png)", "", "输入／脚本 SHA、每次接触的 root/force、严格时间窗口与计数均见 [phase_quality.json](phase_quality.json)。"]
    (output / "phase_quality.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "analyzed", "output_dir": str(output), "phases": len(phase_rows),
                      "terminal_rows": len(terminal_rows), "static_contact_frames": len(static_rows)}))


if __name__ == "__main__":
    main()
