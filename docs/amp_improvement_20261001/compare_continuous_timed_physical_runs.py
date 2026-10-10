#!/usr/bin/env python3
"""CPU comparison of the timed causal-reference deployment and raw baseline.

Reference constraints and measured physical motion are reported separately.
No simulator, policy inference, checkpoint mutation or CUDA use is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np

from analyze_route_completion import analyze_run as analyze_routes
from compare_independent_amp_physical_runs import analyze, contact_summary, source


ROOT = Path(__file__).resolve().parents[2]


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return {"samples": int(values.size), "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
            "p95": float(np.quantile(values, .95)), "maximum": float(values.max())}


def reference_analysis(path, runner):
    with np.load(path / "physical_states.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    for name in ("raw_target_xy", "reference_target_velocity_xy"):
        assert arrays[name].shape == (6000, 4, 2) and arrays[name].dtype.kind == "f" and np.isfinite(arrays[name]).all()
    for name in ("reference_deviation_m", "raw_plan_endpoint_goal_error"):
        assert arrays[name].shape == (6000, 4) and arrays[name].dtype.kind == "f" and np.isfinite(arrays[name]).all() and (arrays[name] >= 0).all()
    root = arrays["root_pos"][..., :2].astype(np.float64)
    raw = arrays["raw_target_xy"].astype(np.float64)
    reference = arrays["target_xy"].astype(np.float64)
    velocity = arrays["reference_target_velocity_xy"].astype(np.float64)
    deviation = np.linalg.norm(reference - raw, axis=-1)
    assert np.allclose(deviation, arrays["reference_deviation_m"], atol=1e-5, rtol=1e-5)
    legal = (arrays["episode_ids"][1:] == arrays["episode_ids"][:-1]) & ~arrays["reset_mask"][1:]
    physical_velocity = np.diff(root, axis=0) * 30
    legal_acceleration = legal[1:] & legal[:-1]
    physical_acceleration = np.diff(physical_velocity, axis=0) * 30
    reference_acceleration = np.diff(velocity, axis=0) * 30
    integration_residual = np.linalg.norm(np.diff(reference, axis=0) - velocity[1:] / 30, axis=-1)[legal]
    with np.load(path / "reference_plans.npz", allow_pickle=False) as handle:
        plans = {key: handle[key] for key in handle.files}
    count = len(plans["actor_ids"])
    shapes = {"control_step": (count,), "actor_ids": (count,), "raw_xy": (count, 51, 2), "reference_xy": (count, 151, 2),
              "reference_velocity_xy": (count, 151, 2), "initial_overspeed": (count,), "speed_violation": (count, 151), "deviation_m": (count, 151)}
    for name, shape in shapes.items():
        assert plans[name].shape == shape and np.isfinite(plans[name]).all()
        assert plans[name].dtype.kind == ("b" if name in {"initial_overspeed", "speed_violation"} else "i" if name in {"control_step", "actor_ids"} else "f")
    assert (plans["actor_ids"] >= 0).all() and (plans["actor_ids"] < 4).all()
    assert (plans["control_step"] >= 0).all() and (plans["control_step"] < 6000).all()
    configuration = runner["reference_adaptation"]["configuration"]
    cap = configuration["max_speed_mps"]
    acceleration_cap = configuration["max_acceleration_mps2"]
    tolerance = configuration["speed_tolerance_mps"]
    planned_velocity = plans["reference_velocity_xy"].astype(np.float64)
    planned_reference = plans["reference_xy"].astype(np.float64)
    speed = np.linalg.norm(planned_velocity, axis=-1)
    acceleration = np.linalg.norm(np.diff(planned_velocity, axis=1) * 30, axis=-1)
    assert np.array_equal(plans["initial_overspeed"], speed[:, 0] > cap + tolerance)
    assert np.array_equal(plans["speed_violation"], speed > cap + tolerance)
    assert float(acceleration.max()) <= acceleration_cap + 1e-4
    times = np.arange(151) / 30 / .1
    lower = np.minimum(np.floor(times).astype(int), 50)
    upper = np.minimum(lower + 1, 50)
    alpha = times - lower
    raw_dense = plans["raw_xy"][:, lower].astype(np.float64) * (1 - alpha)[None, :, None] + plans["raw_xy"][:, upper].astype(np.float64) * alpha[None, :, None]
    planned_deviation = np.linalg.norm(planned_reference - raw_dense, axis=-1)
    assert np.allclose(planned_deviation, plans["deviation_m"], atol=1e-5, rtol=1e-5)
    planned_integration = np.linalg.norm(np.diff(planned_reference, axis=1) - planned_velocity[:, 1:] / 30, axis=-1)
    assert float(planned_integration.max()) < 1e-5 and float(integration_residual.max()) < 1e-5
    assert float(np.linalg.norm(reference_acceleration[legal], axis=-1).max()) <= acceleration_cap + 1e-4
    unique_controls = np.unique(plans["control_step"])
    assert count == len(unique_controls) * 4 == runner["trajectory_plans"]
    assert len(unique_controls) == runner["planning_clock"]["forward_calls"]
    for step in unique_controls:
        assert sorted(plans["actor_ids"][plans["control_step"] == step].tolist()) == [0, 1, 2, 3]
    per_actor = [{"actor": actor, "actor_plans": int((plans["actor_ids"] == actor).sum()),
                  "initial_overspeed": int(plans["initial_overspeed"][plans["actor_ids"] == actor].sum()),
                  "future_speed_violation_samples": int(plans["speed_violation"][plans["actor_ids"] == actor].sum())} for actor in range(4)]
    return {"sources": {"reference_plans": source(path / "reference_plans.npz")}, "plan_array_shapes": {k: list(v.shape) for k, v in plans.items()},
            "configuration": configuration, "actor_plans": count, "joint_planning_calls": len(unique_controls), "per_actor": per_actor,
            "initial_overspeed_plan_count": int(plans["initial_overspeed"].sum()), "braking_activation_indicator": "initial_overspeed; explicit brake policy selected, no separate brake flag",
            "planned_speed_violation_mask_count": int(plans["speed_violation"].sum()), "planned_speed_mask_denominator": int(plans["speed_violation"].size),
            "planned_reference_speed_mps": statistics(speed), "planned_reference_vector_acceleration_mps2": statistics(acceleration),
            "planned_reference_deviation_m": statistics(planned_deviation), "planned_reference_integration_residual_m_max": float(planned_integration.max()),
            "executed_reference_integration_residual_m_max": float(integration_residual.max()),
            "root_to_raw_target_error_m": statistics(np.linalg.norm(root - raw, axis=-1)),
            "root_to_executed_reference_error_m": statistics(np.linalg.norm(root - reference, axis=-1)),
            "executed_reference_to_raw_target_deviation_m": statistics(deviation),
            "raw_plan_endpoint_goal_error_m": statistics(arrays["raw_plan_endpoint_goal_error"]),
            "executed_reference_speed_mps": statistics(np.linalg.norm(velocity, axis=-1)),
            "executed_reference_vector_acceleration_mps2": statistics(np.linalg.norm(reference_acceleration[legal], axis=-1)),
            "physical_root_control_interval_displacement_speed_mps": statistics(np.linalg.norm(physical_velocity[legal], axis=-1)),
            "physical_root_control_interval_second_difference_acceleration_mps2": statistics(np.linalg.norm(physical_acceleration[legal_acceleration], axis=-1)),
            "physical_speed_greater_than_reference_cap_transitions": int((np.linalg.norm(physical_velocity[legal], axis=-1) > cap + tolerance).sum()),
            "physical_speed_transition_denominator": int(legal.sum()),
            "speed_masks_recomputed_and_exact": True, "deviations_recomputed_and_close": True,
            "reference_constraints_do_not_bound_physical_body": True,
            "limitations": "Reference statistics cover all future plan knots, including unexecuted futures. Physical speed/acceleration are 30Hz root-position differences with reset/episode boundaries excluded, not instantaneous PhysX velocity or full-body bounds."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/diffusion_amp/shared_independent_amp_raw_v1")
    parser.add_argument("--new", type=Path, default=ROOT / "output/diffusion_amp/shared_independent_amp_continuous_timed_v1")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/amp_improvement_20261001/continuous_timed_physical_comparison")
    args = parser.parse_args()
    old, old_eval, old_runner = analyze(args.baseline.resolve())
    new, new_eval, new_runner = analyze(args.new.resolve())
    protocol_keys = ("embedding_choices", "reference_sampling", "generated_sampling", "cluster_choices", "train_vs_test_gaussian_mmd_reference")
    protocols = {key: old_eval[key] == new_eval[key] for key in protocol_keys}
    cluster_equal = {str(k): (args.baseline / f"pedestrian_metrics/train_pose_clusters_k{k}.json").read_bytes() == (args.new / f"pedestrian_metrics/train_pose_clusters_k{k}.json").read_bytes() for k in (50, 100)}
    assert all(protocols.values()) and all(cluster_equal.values())
    weights = {name: source(runner["arguments"][field]) for name, runner, field in (("baseline_controller", old_runner, "resume"), ("new_controller", new_runner, "resume"), ("baseline_diffusion", old_runner, "diffusion_checkpoint"), ("new_diffusion", new_runner, "diffusion_checkpoint"))}
    assert weights["baseline_controller"] == weights["new_controller"] and weights["baseline_diffusion"] == weights["new_diffusion"]
    assert old["physics"] == new["physics"]
    assert old["actual_contacts"]["measurement_contract"] == new["actual_contacts"]["measurement_contract"]
    route_old, route_new = analyze_routes(args.baseline), analyze_routes(args.new)
    assert route_old["status"] == route_new["status"] == "checked"
    with np.load(args.new / "physical_states.npz", allow_pickle=False) as handle:
        reached = handle["route_vertex_reached"]
        route_distance = np.linalg.norm(handle["root_pos"][..., :2].astype(np.float64) - handle["route_vertex_xy"], axis=-1)[reached]
        routes = np.asarray(new_runner["evaluation_scene"]["routes_xy"], dtype=np.float64)
        actors = np.broadcast_to(np.arange(4), reached.shape)[reached]
        previous_indices = (handle["route_indices"][reached] - 1) % 4
        assert np.allclose(handle["route_vertex_xy"][reached], routes[actors, previous_indices], atol=1e-7, rtol=0)
        assert (route_distance <= new_runner["evaluation_scene"]["arrival_radius_m"] + 1e-5).all()
        assert not (reached & handle["terminated"]).any() and not (reached & handle["reset_mask"]).any()
        route_geometry = {"actual_arrivals": int(reached.sum()), "all_recorded_targets_match_declared_route": True,
                          "maximum_arrival_distance_m": float(route_distance.max()),
                          "arrival_radius_m": new_runner["evaluation_scene"]["arrival_radius_m"],
                          "arrivals_on_fall_or_first_reset_frame": 0}
    reference = reference_analysis(args.new, new_runner)
    manifest_path = ROOT / "docs/amp_improvement_20261001/shared_continuous_timed_source.json"
    archive_path = manifest_path.with_suffix(".tar.gz")
    manifest = json.loads(manifest_path.read_text())
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {row["path"] for row in manifest["files"]}
        for row in manifest["files"]:
            archived = archive.extractfile(row["path"]).read()
            assert len(archived) == row["bytes"] and hashlib.sha256(archived).hexdigest() == row["sha256"]
    differences = [{**row, "current_sha256": source(ROOT / row["path"])["sha256"]} for row in manifest["files"] if source(ROOT / row["path"])["sha256"] != row["sha256"]]
    contacts = {"baseline": contact_summary(old), "new": contact_summary(new)}
    argument_changes = {key: {"baseline": old_runner["arguments"].get(key), "new": new_runner["arguments"].get(key)} for key in old_runner["arguments"].keys() | new_runner["arguments"].keys() if old_runner["arguments"].get(key) != new_runner["arguments"].get(key)}
    report = {"schema": "continuous_timed_physical_comparison_v1", "status": "cpu_analyzed", "baseline": old, "new": new, "weights": weights,
              "pose_protocol_equal": protocols, "train_cluster_artifacts_byte_identical": cluster_equal, "physics_manifest_equal": True, "contact_contract_equal": True,
              "argument_changes": argument_changes, "reference_analysis": reference, "contacts_compact": contacts,
              "route_analysis": {"baseline": route_old, "new": route_new, "new_physical_arrival_geometry": route_geometry},
              "source_archive": source(archive_path), "source_manifest": source(manifest_path), "archived_files_match_manifest": len(manifest["files"]), "post_launch_source_differences": differences,
              "simultaneous_changes": ["endpoint -> ten timed A* points at0.5..5s", "explicit invalid-clearance-start recovery radius0.5m; no physical root teleport", "peer radius0.4 ->0.6m; scene radius0.5 ->0.8m", "raw XY -> causal continuous30Hz reference, speed1.5/vector acceleration2, explicit brake policy"],
              "reference_adaptation": new_runner["reference_adaptation"], "successful_start_recoveries": new_runner["successful_start_recoveries"],
              "single_factor_causal_claim": False, "paper_exact_claim": False, "pose_convergence_established": False,
              "limitations": ["One seed7 200s four-actor fixture, not eight benchmark scenes or a significance test.", "Zero falls does not establish realistic gait, no contact, whole-body kinematic bounds or style convergence.", "Raw and executed-reference tracking have different target semantics and must not be substituted.", "Normal contacts exclude floor/self and tangential friction; local incidence counts are not the paper's unknown aggregate.", "Per-frame pose MMD/coverage do not evaluate temporal motion.", "Current source changes are distinct from the preserved launch archive."], "analysis_script": source(__file__)}
    rows = []
    for label, key in (("跌倒", "falls"), ("每1000行人秒跌倒", "falls_per1000_actor_seconds"), ("XY路程m（排除reset）", "walked_xy_m_total"), ("平均XY速度m/s", "mean_xy_speed_excluding_resets_mps"), ("最低pelvis m", "minimum_pelvis_height_m")):
        rows.append(f"| {label} | {old[key]:.6g} | {new[key]:.6g} |")
    rows += [f"| 同episode完整四顶点循环 | {route_old['complete_cycles_total']} | {route_new['complete_cycles_total']} |",
             f"| 实际到达vertices | {old['goal_metrics']['route_vertices_reached_total']} | {new['goal_metrics']['route_vertices_reached_total']} |",
             f"| Gaussian MMD²，σ1 | {old['gaussian_mmd']['mmd_squared']:.9f} | {new['gaussian_mmd']['mmd_squared']:.9f} |"]
    for k in ("50", "100"):
        rows.append(f"| K{k} coverage/归一化entropy | {old['pose_generated'][k]['coverage']:.2f}/{old['pose_generated'][k]['normalized_pose_entropy']:.6f} | {new['pose_generated'][k]['coverage']:.2f}/{new['pose_generated'][k]['normalized_pose_entropy']:.6f} |")
    for label, key in (("Static actor-frames/24000", "static_actor_frames"), ("Static连续actor事件", "static_actor_incidences"), ("Peer pair-frames/36000", "peer_unordered_pair_frames")):
        rows.append(f"| {label} | {contacts['baseline'][key]} | {contacts['new'][key]} |")
    rows.append(f"| A*失败查询 | {old['goal_metrics']['global_path_failures']} | {new['goal_metrics']['global_path_failures']} |")
    raw_error = reference["root_to_raw_target_error_m"]
    ref_error = reference["root_to_executed_reference_error_m"]
    dev = reference["executed_reference_to_raw_target_deviation_m"]
    text = f"""# Timed + causal reference：真实四行人200秒对照

## 实测效果与边界

两份6000帧×4人、30Hz真实记录各800行人秒；相同独立课程AMP1000和Diffusion5000 checkpoint、物理与真实contact合同。新run **0次跌倒**、pelvis最低{new['minimum_pelvis_height_m']:.6f}m，26次vertex到达、5个同episode完整四顶点循环；不是八场景benchmark，也不表示pose收敛。

| 指标 | raw endpoint | timed + causal reference |
|---|---:|---:|
{chr(10).join(rows)}

新每人vertices={route_new['route_reached_total_per_actor']}、完整cycles={route_new['complete_cycles_per_actor']}。26个记录到达点全部匹配声明route，最大root到vertex距离{route_geometry['maximum_arrival_distance_m']:.6f}m≤.75m，无fall或首个reset帧计数。所有四人200秒episode均未跌倒/超时、结束右删失；不是无限时稳定性证明。Static actor-frame比例7%→{100*contacts['new']['static_actor_frames']/24000:.5f}%，新仍有{contacts['new']['static_actor_incidences']}个连续actor接触事件，不能说无碰撞。新peer正常力阈值事件为0。训练与部署权重未修改。

## 原始模型目标、参考和物理root分别报告

旧raw run root→raw目标mean/RMS={old['tracking']['mean_xy_error_m']:.6f}/{old['tracking']['rmse_xy_m']:.6f}m。新 **root→raw模型目标** mean/RMS/p95/max={raw_error['mean']:.6f}/{raw_error['rms']:.6f}/{raw_error['p95']:.6f}/{raw_error['maximum']:.6f}m；新 **root→执行reference**={ref_error['mean']:.6f}/{ref_error['rms']:.6f}/{ref_error['p95']:.6f}/{ref_error['maximum']:.6f}m。evaluator中的`plan_tracking`在新run指执行reference，不能拿它直接当旧raw目标指标。

执行reference与raw模型目标偏差mean/p95/max={dev['mean']:.6f}/{dev['p95']:.6f}/{dev['maximum']:.6f}m。适配保留原51个10Hz模型knots，生成151个30Hz参考knots；普通重规划从已执行参考的XY/速度延续，仅episode初始化/reset从物理root建立起点，不将物理root搬到参考位置。

## 参考约束与brake标记的CPU复核

保存{reference['actor_plans']}份actor-plan、{reference['joint_planning_calls']}次四人联合规划；全部future reference样本分母{reference['planned_speed_mask_denominator']}，含未执行future。显式`initial_speed_policy=brake`，**initial_overspeed={reference['initial_overspeed_plan_count']}、speed_violation mask={reference['planned_speed_violation_mask_count']}**；本run没有触发初始超速制动，不将“配置brake”写成实际执行了制动。mask按保存的速度/1.5m/s+1e-6 tolerance逐项重算完全相等。

计划reference max speed={reference['planned_reference_speed_mps']['maximum']:.9f}m/s、max vector acceleration={reference['planned_reference_vector_acceleration_mps2']['maximum']:.9f}m/s²（float32误差，声明cap1.5/2）；执行reference同样复核。计划/执行积分残差最大{reference['planned_reference_integration_residual_m_max']:.3g}/{reference['executed_reference_integration_residual_m_max']:.3g}m；保存deviation重算通过。

**参考cap不等于物理body bound。**实际root的30Hz位移差分速度max={reference['physical_root_control_interval_displacement_speed_mps']['maximum']:.6f}m/s，{reference['physical_speed_greater_than_reference_cap_transitions']}/{reference['physical_speed_transition_denominator']}合法区间高于参考1.5m/s；二阶位置差分acceleration max={reference['physical_root_control_interval_second_difference_acceleration_mps2']['maximum']:.6f}m/s²。它们是控制区间离散root指标，不是SDK瞬时速度、子步acceleration或所有body约束。适配也没有障碍/接触/路线到达保证。

## 同时变化的部署参数

保持weight20总guidance，endpoint改为同一次静态A*路径的10个0.5..5秒点，每点weight2；start recovery显式0.5m。恢复仅在实际root有效但clearance cell被占用时连接近邻可达起点，不teleport，初始连接段clearance未认证；successful recovery={new_runner['successful_start_recoveries']}次查询，不是不同episode数或物理reset。peer radius.4→.6、scene radius.5→.8；新causal reference速度1.5/向量加速度2。**多项同时改变，不能把效果单独归给timed或filter。**两run的route到达/reset/forced-replan合同一致。

DDPM100、FP32、replan15、batch8、seed7与XY消费保持；执行状态和额外route触发改变后续随机调用，不是逐位配对轨迹。速度/加速度guidance penalties仍为0，真实约束只对新增reference声明。

## 固定姿态协议与provenance

train/test各2048、generated2048 time-major索引、seed7、15节点45D root/heading/height规范化、σ1以及全部参考IDs相同。K50/100 train-only cluster文件逐字节相同。MMD从.09943到.09215，coverage/entropy有所变化，仍不能判定风格通过；该协议未衡量temporal gait，论文节点/尺度/带宽未知，不声称exact paper。

AMP SHA256 `{weights['new_controller']['sha256']}`；Diffusion SHA256 `{weights['new_diffusion']['sha256']}`。启动源归档{len(manifest['files'])}文件逐bytes/hash匹配manifest；后续CLI sample-count变化单列，不覆盖原源hash。这次200秒不含后续2-sample演示。墙钟raw{old['elapsed_wall_s']:.2f}s、新{new['elapsed_wall_s']:.2f}s，不能当純planner benchmark。

![分段比较](continuous_timed_physical_comparison.png)

完整tracking/reference masks/几何/route/collision与source证据：[JSON](continuous_timed_physical_comparison.json)。[新逐帧phase](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_independent_amp_continuous_timed_v1/pedestrian_metrics/phase_quality.md)。[可重跑CPU脚本](compare_continuous_timed_physical_runs.py)。本轮未修改训练、部署业务代码或权重。
"""
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/protomotions-matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    with np.load(args.new / "physical_states.npz", allow_pickle=False) as handle:
        raw_tracking = np.linalg.norm(handle["root_pos"][..., :2].astype(np.float64) - handle["raw_target_xy"], axis=-1)
    for run, label in ((old, "Raw endpoint"), (new, "Timed + causal reference")):
        x = [np.mean(p["physical_time_range_s"]) for p in run["phases_20s"]]
        for axis, field in zip(axes, ("terminal_frames", "tracking_mean_m", "mean_planar_speed_m_s", "static_contact_actor_frames"), strict=True):
            axis.plot(x, [p[field] for p in run["phases_20s"]], marker="o", label=label, linewidth=1.3)
    axes[1].plot(x, [raw_tracking[i:i+600].mean() for i in range(0, 6000, 600)], marker="x", label="Timed physical root to raw model target", linewidth=1.3)
    for axis, ylabel in zip(axes, ("Falls / 20s", "Target error mean (m)", "Physical root XY speed (m/s)", "Static actor-frames / 20s"), strict=True):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_title("Same checkpoints + physics; timed path, recovery, radii and causal reference change together")
    axes[-1].set_xlabel("Physical interval midpoint (s)")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)
    report["plot"] = source(args.output_prefix.with_suffix(".png"))
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "cpu_analyzed", "falls": [old["falls"], new["falls"]], "cycles": [route_old["complete_cycles_total"], route_new["complete_cycles_total"]], "mmd_squared": [old["gaussian_mmd"]["mmd_squared"], new["gaussian_mmd"]["mmd_squared"]], "reference_overspeed_flags": reference["initial_overspeed_plan_count"], "output_prefix": str(args.output_prefix)}))


if __name__ == "__main__":
    main()
