#!/usr/bin/env python3
"""Export reproducible diffusion/AMP training snapshots and shareable PNGs.

Run again while training progresses. All plots use recorded, unsmoothed values.
An incomplete final JSONL record is ignored; other corrupt records are errors.
No simulator, checkpoint loading, convergence threshold or paper-match claim is
part of this tool. Defaults are resolved from the repository, not the shell cwd.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def _input_snapshot(path: Path):
    data = path.read_bytes()
    return data, {"path": str(path.resolve()), "snapshot_bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest()}


def _json(path: Path):
    data, provenance = _input_snapshot(path)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, provenance


def read_metrics(path: Path):
    data, provenance = _input_snapshot(path)
    lines = data.splitlines(keepends=True)
    records = []
    incomplete = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                incomplete = True
                break
            raise ValueError(f"Malformed complete JSONL record at {path}:{index + 1}") from None
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSONL object at {path}:{index + 1}")
        update = record.get("update_count")
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError(f"Missing positive integer update_count at {path}:{index + 1}")
        records.append(record)
    if not records:
        raise ValueError(f"No complete AMP metric records: {path}")
    provenance.update(records=len(records), incomplete_final_record_ignored=incomplete)
    return records, provenance


def _number(record, name):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float("nan")
    return float(value)


def split_sessions(records):
    """Counters and elapsed wall time restart at each current runner invocation."""
    sessions = [[]]
    boundaries = []
    for record in records:
        if sessions[-1]:
            previous = sessions[-1][-1]
            reasons = []
            if record["update_count"] <= previous["update_count"]:
                reasons.append("non-increasing update_count")
            for field in ("elapsed_s", "resets", "fall_resets", "time_limit_resets"):
                if _number(record, field) < _number(previous, field):
                    reasons.append(f"{field} restarted")
            if reasons:
                boundaries.append({"before_update": record["update_count"], "reasons": reasons})
                sessions.append([])
        sessions[-1].append(record)
    return sessions, boundaries


def _line(ax, sessions, field, label, *, style="-", color=None):
    labeled = False
    for session in sessions:
        if not any(math.isfinite(_number(row, field)) for row in session):
            continue
        ax.plot([row["update_count"] for row in session], [_number(row, field) for row in session],
                style, marker="." if len(session) < 3 else None,
                label=None if labeled else label, color=color, linewidth=1.4)
        labeled = True


def _axes(ax, title, ylabel, xlabel="Optimizer update"):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=.25)


def _legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def _save_figure(figure, path):
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160, facecolor="white", bbox_inches="tight")
    os.replace(temporary, path)


def plot_diffusion(plt, result, output):
    metrics = result.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Diffusion result has no recorded metric list")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    steps = [row["step"] for row in metrics]
    for field, label in (("loss", "Training x0 loss"), ("validation_loss", "Validation x0 loss")):
        selected = [row for row in metrics if math.isfinite(_number(row, field))]
        axes[0].plot([row["step"] for row in selected], [_number(row, field) for row in selected],
                     marker="o" if field == "validation_loss" else None, label=label, linewidth=1.4)
    for field, step, label, marker in (
        ("initial_validation_loss", result.get("start_step", 0), "Initial validation", "o"),
        ("final_validation_loss", result.get("completed_step", steps[-1]), "Final validation", "s"),
        ("test_loss", result.get("completed_step", steps[-1]), "Held-out test (single evaluation)", "D"),
    ):
        if math.isfinite(_number(result, field)):
            axes[0].scatter([step], [_number(result, field)], label=label, marker=marker, s=36, zorder=4)
    for field, label in (("state_mse", "State MSE"), ("yaw_mse", "Yaw MSE")):
        axes[1].plot(steps, [_number(row, field) for row in metrics], label=label, linewidth=1.4)
    _axes(axes[0], "Diffusion objective", "Recorded loss", "Diffusion optimizer step")
    _axes(axes[1], "Logged diffusion error terms", "Recorded MSE", "Diffusion optimizer step")
    for ax in axes:
        _legend(ax)
    figure.suptitle("AMASS five-state reconstruction: raw training metrics")
    _save_figure(figure, output / "diffusion_training.png")
    plt.close(figure)


def plot_amp(plt, sessions, output):
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for field, label, color in zip(("ppo_loss", "policy_loss", "value_loss"),
                                   ("PPO aggregate", "Policy loss", "Value loss"), colors):
        _line(axes[0], sessions, field, label, color=color)
    for field, label, color in zip(("disc_loss", "disc_classification", "disc_gradient_penalty"),
                                   ("Discriminator aggregate", "Classification", "Gradient penalty"), colors):
        _line(axes[1], sessions, field, label, color=color)
    _axes(axes[0], "PPO losses", "Recorded objective")
    _axes(axes[1], "Action-conditioned AMP discriminator", "Recorded objective")
    for ax in axes:
        _legend(ax)
    figure.suptitle("AMP optimization: raw metrics; lines break at detected process restarts")
    _save_figure(figure, output / "amp_optimization.png")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for field, label, color in zip(
        ("reward_mean", "amp_rewards_mean", "trajectory_rewards_mean", "energy_rewards_mean", "direction_rewards_mean"),
        ("Total reward", "AMP component", "Trajectory component", "Energy component", "Direction component"), colors,
    ):
        _line(axes[0, 0], sessions, field, label, color=color)
    _axes(axes[0, 0], "Logged mean reward and components", "Recorded rollout mean")
    # Explicit falls are not inferred from aggregate resets, height or PPO loss.
    for field, label, color, axis in zip(("fall_resets", "time_limit_resets", "resets"),
                                        ("Fall terminations", "Time limits only", "All resets"),
                                        colors, (axes[0, 1], axes[0, 2], axes[0, 2])):
        labeled = False
        for session in sessions:
            previous = 0.0
            xs, deltas = [], []
            for index, row in enumerate(session):
                value = _number(row, field)
                # First log may already resume an unrecorded process. Only update=1
                # unambiguously establishes a zero baseline for per-update deltas.
                delta = value - previous if index or row["update_count"] == 1 else float("nan")
                xs.append(row["update_count"])
                deltas.append(delta if math.isfinite(value) and math.isfinite(previous) else float("nan"))
                previous = value
            if any(math.isfinite(value) for value in deltas):
                axis.plot(xs, deltas, label=None if labeled else label, color=color, linewidth=1.4)
                labeled = True
    _axes(axes[0, 1], "Fall terminations per logged rollout", "Recorded fall count")
    _axes(axes[0, 2], "Time limits and all resets per logged rollout", "Recorded event count")
    axes[0, 1].set_ylim(bottom=0)
    if not axes[0, 1].get_legend_handles_labels()[0]:
        axes[0, 1].text(.5, .5, "No comparable explicit fall counters", ha="center", transform=axes[0, 1].transAxes)
    _line(axes[1, 0], sessions, "minimum_height_m", "Running minimum pelvis height", color=colors[0])
    _axes(axes[1, 0], "Minimum height through each process's current update", "Height (m)")
    _line(axes[1, 1], sessions, "mean_completed_episode_s", "Mean of completed episodes in this process", color=colors[1])
    _axes(axes[1, 1], "Completed episode duration (includes falls and time limits)", "Duration (s)")
    if not axes[1, 1].get_legend_handles_labels()[0]:
        axes[1, 1].text(.5, .5, "No completed episodes reported", ha="center", transform=axes[1, 1].transAxes)
    _line(axes[1, 2], sessions, "forward_velocity_mean", "Mean forward velocity", color=colors[2])
    _axes(axes[1, 2], "Logged root forward velocity", "Velocity (m/s)")
    for ax in axes.flat:
        _legend(ax)
    figure.suptitle("AMP physical rollout statistics; no stability or paper-equivalence verdict")
    _save_figure(figure, output / "amp_rollouts.png")
    plt.close(figure)


def _write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-result", type=Path, default=ROOT / "output/diffusion_amp/diffusion_training/training_result.json")
    parser.add_argument("--amp-metrics", type=Path, default=ROOT / "output/diffusion_amp/amp_training/metrics.jsonl")
    parser.add_argument("--amp-result", type=Path, default=ROOT / "output/diffusion_amp/amp_training/result.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/diffusion_amp_20261001/training_analysis")
    args = parser.parse_args(argv)
    diffusion, diffusion_source = _json(args.diffusion_result)
    metrics, metrics_source = read_metrics(args.amp_metrics)
    sessions, boundaries = split_sessions(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the matplotlib cache within the permitted temporary directory.
    with tempfile.TemporaryDirectory(prefix="navisaaclab-matplotlib-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_diffusion(plt, diffusion, args.output_dir)
        plot_amp(plt, sessions, args.output_dir)
    report = {
        "schema": "navisaaclab.training_analysis.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {"diffusion_result": diffusion_source, "amp_metrics": metrics_source},
        "diffusion": {key: diffusion.get(key) for key in (
            "status", "start_step", "completed_step", "initial_validation_loss", "final_validation_loss", "test_loss", "elapsed_seconds", "assumptions")},
        "amp": {"records": len(metrics), "latest_record": metrics[-1],
                "detected_process_sessions": len(sessions), "session_boundaries": boundaries,
                "session_summaries": [{"first_update": rows[0]["update_count"],
                    "last_update": rows[-1]["update_count"], "records": len(rows),
                    "last_recorded_counters": {key: rows[-1].get(key) for key in (
                        "resets", "fall_resets", "time_limit_resets", "mean_completed_episode_s", "minimum_height_m")},
                    "rows_without_explicit_fall_counter": sum("fall_resets" not in row for row in rows)} for rows in sessions]},
        "limitations": [
            "Snapshot of completed log records; a still-running process may append more.",
            "No smoothing, stability threshold, convergence decision or author-experiment equivalence is inferred.",
            "Process boundaries are inferred from update/counter/elapsed-time restarts, not an explicit run identifier.",
            "Reset counters and running height/episode statistics restart with the current runner process.",
            "Missing fall counters are unknown, never substituted with total resets.",
            "Only consecutive records within a process give per-rollout reset differences; an ambiguous first record has no inferred baseline.",
            "The diffusion dataset is individual AMASS with empty neighbors and uniform walkable maps; this is not a social avoidance benchmark.",
        ],
        "author_experiment_reproduced": False,
        "convergence_claim": False,
    }
    if args.amp_result.is_file():
        result, source = _json(args.amp_result)
        report["sources"]["amp_result"] = source
        report["amp"]["last_completed_result"] = {key: result.get(key) for key in (
            "status", "update_count", "physical_transition_count", "fall_resets", "time_limit_resets", "convergence_claim", "initialization", "author_experiment_reproduced")}
        report["amp"]["result_update_matches_latest_metrics"] = result.get("update_count") == metrics[-1]["update_count"]
    else:
        report["amp"]["last_completed_result"] = None
    report["artifacts"] = [{"path": name, "bytes": (args.output_dir / name).stat().st_size,
                            "sha256": hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()}
                           for name in ("diffusion_training.png", "amp_optimization.png", "amp_rollouts.png")]
    _write_text(args.output_dir / "summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    last = metrics[-1]
    text = f"""# Diffusion + AMP 训练指标快照

生成时间：{report['generated_at_utc']}。这是读取日志时的快照，脚本可在训练期间及结束后重复运行。

## Diffusion

已记录步数：{diffusion.get('completed_step')}；初始验证 loss：{diffusion.get('initial_validation_loss')}；最终验证 loss：{diffusion.get('final_validation_loss')}；独立测试 loss：{diffusion.get('test_loss')}。

![Diffusion 训练与验证](diffusion_training.png)

## AMP

共读取 {len(metrics)} 条完整日志，最新 update 为 {last['update_count']}，累计物理 transitions 字段为 {last.get('physical_transitions')}；检测到 {len(sessions)} 个进程片段。

最新 reward mean：{last.get('reward_mean')}；该进程累计 fall resets：{last.get('fall_resets', '未记录')}；time-limit resets：{last.get('time_limit_resets', '未记录')}；该进程运行最低 pelvis height：{last.get('minimum_height_m')} m。

![AMP 优化目标](amp_optimization.png)

![AMP 奖励与物理 rollout](amp_rollouts.png)

## 解释边界

- 图中均为原始记录，不平滑，不依据 loss/reward 自动判断稳定或收敛。
- 恢复训练后当前 runner 的累计 reset、最低高度、完成 episode 平均时长和 elapsed 会从新进程重新开始；曲线在检测到的边界断开。
- fall 与 time-limit 分别采用明确字段；旧日志没有 fall 字段时保持未知。reset 差值仅比较同一进程的相邻记录。
- AMP result.json 可能是上一轮已完成进程的结果；summary.json 单独记录其 update 是否与最新日志一致。
- 当前 AMASS diffusion 输入没有同步行人邻居与真实地图；这些图不能证明论文八场景导航评估、作者 2,848 动作清单或作者权重的一致性。
- 精确输入快照字节数、SHA-256、进程边界与最后完整记录见 [summary.json](summary.json)。
"""
    _write_text(args.output_dir / "report.md", text)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "amp_records": len(metrics),
                      "latest_amp_update": last["update_count"], "detected_process_sessions": len(sessions),
                      "incomplete_final_record_ignored": metrics_source["incomplete_final_record_ignored"]}))


if __name__ == "__main__":
    main()
