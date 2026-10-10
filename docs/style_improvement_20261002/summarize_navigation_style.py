#!/usr/bin/env python3
"""CPU-only comparison and explicitly defined stagnation diagnostics.

Consumes existing frozen style and ground-support reports; it neither changes
their metrics nor fits a new reference. Displacement idle is <0.1 m/s on valid
same-episode 30Hz intervals. AABB distance is a pelvis-point geometry diagnostic,
not collider/contact clearance. Source paths are taken from recorded artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def summary(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if not value.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "min": None}
    return {"count": int(value.size), "mean": float(value.mean()), "median": float(np.median(value)),
            "p95": float(np.quantile(value, .95)), "max": float(value.max()), "min": float(value.min())}


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def aabb_point_distance(points, cuboids):
    """XY signed distance of root point to each declared box; not body clearance."""
    distances = []
    for box in cuboids:
        delta = np.abs(points-np.asarray(box["center"][:2]))-np.asarray(box["size"][:2])/2
        distance = np.linalg.norm(np.maximum(delta, 0), axis=-1)+np.minimum(np.max(delta, axis=-1), 0)
        distances.append(distance)
    return np.stack(distances, axis=-1)


def stagnation(directory, ground_audit):
    directory = Path(directory)
    run = json.loads((directory/"result.json").read_text())
    result_record = record(directory/"result.json")
    if result_record != ground_audit["artifacts"]["result"]:
        raise ValueError("Runtime result changed after the audited native capture")
    physical_record = record(directory/"physical_states.npz")
    ground_record = record(directory/"ground_support.npz")
    if physical_record != ground_audit["artifacts"]["physical_states"] or ground_record != ground_audit["artifacts"]["ground_support"]:
        raise ValueError("Captured state changed after independent audit")
    with np.load(directory/"physical_states.npz", allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    with np.load(directory/"ground_support.npz", allow_pickle=False) as archive:
        ground = {key: archive[key] for key in ("body_link_lin_vel_w", "double_support_fraction", "no_support_fraction")}
    plan_path = directory/"reference_plans.npz"
    if Path(run["reference_adaptation"]["plan_capture"]).resolve() != plan_path.resolve():
        raise ValueError("Result binds another reference plan artifact")
    with np.load(plan_path, allow_pickle=False) as archive:
        plans = {key: archive[key] for key in archive.files}
    t, b = state["root_pos"].shape[:2]
    for name, shape in (("control_step", (len(plans["actor_ids"]),)), ("raw_xy", (len(plans["actor_ids"]), 51, 2)),
                        ("reference_xy", (len(plans["actor_ids"]), 151, 2)),
                        ("reference_velocity_xy", (len(plans["actor_ids"]), 151, 2))):
        if plans[name].shape != shape or not np.isfinite(plans[name]).all():
            raise ValueError("Invalid saved plan shape/finite contract")
    root = state["root_pos"][..., :2].astype(np.float64)
    speed = np.linalg.norm(np.diff(root, axis=0), axis=-1)*30
    valid = ((state["episode_ids"][1:] == state["episode_ids"][:-1]) & ~state["reset_mask"][1:])
    idle = (speed < .1) & valid
    reference_speed = np.linalg.norm(state["reference_target_velocity_xy"], axis=-1)
    native_speed = np.linalg.norm(ground["body_link_lin_vel_w"][:, -1, :, 0, :2], axis=-1)
    box_distances = aabb_point_distance(root, run["evaluation_scene"]["cuboids"])
    per_actor = []
    for actor in range(b):
        actor_records = np.flatnonzero(plans["actor_ids"] == actor)
        controls = plans["control_step"][actor_records]
        if not len(controls) or controls[0] != 0 or (np.diff(controls) <= 0).any() or controls[-1] >= t:
            raise ValueError("Each actor needs exact ordered saved replans from control zero")
        selected = actor_records[np.searchsorted(controls, np.arange(t), side="right")-1]
        ages = np.arange(t)-plans["control_step"][selected]+1
        if (ages < 1).any() or (ages > run["planning_clock"]["maximum_interval_steps"]).any():
            raise ValueError("Actual plan age exceeds declared replanning interval")
        ref_from_plans = plans["reference_xy"][selected, ages]
        velocity_from_plans = plans["reference_velocity_xy"][selected, ages]
        lower = ages//3
        fraction = (ages%3)/3
        raw_from_plans = plans["raw_xy"][selected, lower]*(1-fraction[:, None])+plans["raw_xy"][selected, lower+1]*fraction[:, None]
        for left, right in ((ref_from_plans, state["target_xy"][:, actor]),
                            (velocity_from_plans, state["reference_target_velocity_xy"][:, actor]),
                            (raw_from_plans, state["raw_target_xy"][:, actor])):
            if not np.allclose(left, right, atol=1e-5, rtol=0):
                raise ValueError("Post-step targets do not match saved before-control plan plus age+1")
        sections = spans(idle[:, actor])
        sections = sorted(sections, key=lambda item: item[1]-item[0], reverse=True)
        longest = None
        if sections:
            start, stop = sections[0]
            frames = slice(int(start), int(stop)+1)
            # A plan control c generates source post-frame c; XY at the exact
            # pre-plan instant is frame c-1. c=0 has no recorded pre-state.
            rows = actor_records[(controls >= start) & (controls <= stop) & (controls > 0)]
            planned_at = plans["control_step"][rows]
            anchor = root[planned_at-1, actor]
            raw = plans["raw_xy"][rows].astype(np.float64)
            reference = plans["reference_xy"][rows].astype(np.float64)
            target = state["global_goal_xy"][planned_at, actor]
            to_goal = target-anchor
            goal_unit = to_goal/np.maximum(np.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-12)
            raw_half = raw[:, 5]-anchor
            raw_end = raw[:, -1]-anchor
            longest = {
                "before_interval_source_frame": int(start), "last_interval_source_frame": int(stop),
                "interval_count": int(stop-start), "duration_s": float((stop-start)/30),
                "start_time_s": float((start+1)/30), "end_time_s": float((stop+1)/30),
                "start_xy": root[start, actor].tolist(), "end_xy": root[stop, actor].tolist(),
                "net_displacement_m": float(np.linalg.norm(root[stop, actor]-root[start, actor])),
                "physical_displacement_speed_mps": summary(speed[start:stop, actor]),
                "native_end_substep_root_speed_mps": summary(native_speed[frames, actor]),
                "executed_target_error_m": summary(np.linalg.norm(state["target_xy"][frames, actor]-root[frames, actor], axis=-1)),
                "raw_next_step_target_error_m": summary(np.linalg.norm(state["raw_target_xy"][frames, actor]-root[frames, actor], axis=-1)),
                "executed_reference_speed_mps": summary(reference_speed[frames, actor]),
                "A_star_lookahead_endpoint_distance_m": summary(np.linalg.norm(state["global_goal_xy"][frames, actor]-root[frames, actor], axis=-1)),
                "actual_route_vertex_distance_m": summary(np.linalg.norm(state["route_vertex_xy"][frames, actor]-root[frames, actor], axis=-1)),
                "double_support_fraction_120Hz": float(ground["double_support_fraction"][frames, actor].mean()),
                "no_foot_support_fraction_120Hz": float(ground["no_support_fraction"][frames, actor].mean()),
                "minimum_root_point_to_any_AABB_distance_m": float(box_distances[frames, actor].min()),
                "minimum_root_point_to_central_AABB_distance_m": float(box_distances[frames, actor, 0].min()),
                "saved_replans_in_segment": int(len(rows)),
                "raw_plan_half_second_displacement_m": summary(np.linalg.norm(raw_half, axis=-1)),
                "raw_plan_five_second_displacement_m": summary(np.linalg.norm(raw_end, axis=-1)),
                "raw_plan_half_second_progress_toward_lookahead_m": summary(np.einsum("ij,ij->i", raw_half, goal_unit)),
                "raw_plan_five_second_progress_toward_lookahead_m": summary(np.einsum("ij,ij->i", raw_end, goal_unit)),
                "raw_plan_five_second_endpoint_residual_to_lookahead_m": summary(np.linalg.norm(raw[:, -1]-target, axis=-1)),
                "raw_plan_five_second_path_length_m": summary(np.linalg.norm(np.diff(raw, axis=1), axis=-1).sum(-1)),
                "adapted_reference_five_second_net_displacement_m": summary(np.linalg.norm(reference[:, -1]-reference[:, 0], axis=-1)),
            }
        per_actor.append({"actor": actor, "valid_same_episode_intervals": int(valid[:, actor].sum()),
                          "idle_intervals": int(idle[:, actor].sum()), "idle_duration_s": float(idle[:, actor].sum()/30),
                          "idle_fraction": float(idle[:, actor].sum()/valid[:, actor].sum()),
                          "strict_idle_segments": len(sections), "longest_idle": longest})
    snapshots = []
    for seconds in (90, 105, 120):
        frame = int(seconds*30)-1
        snapshots.append({"time_s": seconds, "source_frame": frame, "actor": 0,
                          "root_xy": root[frame, 0].tolist(), "executed_target_xy": state["target_xy"][frame, 0].tolist(),
                          "raw_next_step_target_xy": state["raw_target_xy"][frame, 0].tolist(),
                          "executed_reference_velocity_xy": state["reference_target_velocity_xy"][frame, 0].tolist(),
                          "A_star_lookahead_endpoint_xy": state["global_goal_xy"][frame, 0].tolist(),
                          "actual_route_vertex_xy": state["route_vertex_xy"][frame, 0].tolist(),
                          "root_point_AABB_distances_m": box_distances[frame, 0].tolist(),
                          "double_support_fraction": float(ground["double_support_fraction"][frame, 0])})
    return {"idle_definition": "same-episode frame-to-frame root XY displacement speed <0.1 m/s; reset intervals excluded",
            "time_semantics": "source frame i is post control i, t=(i+1)/30; interval [i,i+1] spans 1/30 seconds",
            "global_goal_field_semantics": "in timed_path, global_goal_xy is current five-metre A* lookahead endpoint; route_vertex_xy is the actual route vertex",
            "all_saved_plan_target_phase_checks_passed": True,
            "geometry_definition": "signed root-XY point distance to declared cuboid AABBs; not body/capsule clearance or measured contact",
            "terrain_normalizer_causality_established": False, "per_actor": per_actor,
            "requested_actor0_snapshots": snapshots, "saved_plans": record(plan_path)}


def build(evaluation_path, ground_path, output, *, previous="previous_best", new="backward_ground200", previous_ground_path=None):
    evaluation = json.loads(evaluation_path.read_text())
    ground = json.loads(ground_path.read_text())
    if evaluation["status"] != "passed" or ground["status"] != "passed":
        raise ValueError("Both actual source evaluations must pass")
    directory = Path(ground["input"])
    outcomes = {}
    for name, run in evaluation["runs"].items():
        result = json.loads(Path(run["sources"]["result"]["path"]).read_text())
        with np.load(run["sources"]["states"]["path"], allow_pickle=False) as archive:
            static = archive["static_contact"]
            peer = archive["peer_contact"]
        outcomes[name] = {
            "full_run_sources": run["sources"], "training_transitions": result["physical_transition_count"],
            "training_updates": result["update_count"], "training_config": result["training_config"],
            "runtime_arguments": result["arguments"], "root_speed": run["generated"]["physical_root_speed_m_s"],
            "same_episode_intervals": run["generated"]["same_episode_transition_count"],
            "valid_acceleration_triplets": run["generated"]["valid_acceleration_triplets"],
            "inclusive_pose_velocity_mmd2": run["inclusive_speed_conditioned"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "root_speed_distribution_W1_mps": run["inclusive_speed_conditioned"]["physical_speed_distribution_wasserstein_m_s"],
            "displacement_idle_fraction": run["inclusive_speed_conditioned"]["speed_bins"][0]["generated_count"]/run["generated"]["same_episode_transition_count"],
            "upright_retained_fraction": run["upright_subset"]["generated_fraction_retained"],
            "falls": result["fall_resets"], "route_vertices_reached": result["route_vertices_reached"],
            "total_vertices": sum(result["route_vertices_reached"]),
            "completed_four_vertex_cycles": sum(value//4 for value in result["route_vertices_reached"]),
            "static_contact_actor_frames": int(static.sum()), "static_contact_actor_frame_fraction": float(static.mean()),
            "peer_contact_actor_frames": int(peer.any(-1).sum()),
            "speed_bins": [{"lower": item["lower_m_s"], "upper": item["upper_m_s"],
                            "generated_intervals": item["generated_count"], "reference_intervals": item["reference_count"],
                            "mmd2": None if item["pose_velocity_gaussian_mmd2"] is None else item["pose_velocity_gaussian_mmd2"]["value"]}
                           for item in run["inclusive_speed_conditioned"]["speed_bins"]],
        }
    old, current = outcomes[previous], outcomes[new]
    differences = {key: {"previous": old["training_config"].get(key), "new": current["training_config"].get(key),
                         "previous_field_present": key in old["training_config"], "new_field_present": key in current["training_config"]}
                   for key in sorted(set(old["training_config"]) | set(current["training_config"]))
                   if old["training_config"].get(key) != current["training_config"].get(key)
                   or (key in old["training_config"]) != (key in current["training_config"])}
    source_reference = evaluation_path.parent/"heldout_reference.json"
    original_reference = Path(__file__).parent/"tracking_style_v1/heldout_reference.json"
    if source_reference.read_bytes() != original_reference.read_bytes():
        raise ValueError("Frozen reference object was changed")
    document = {"status": "passed", "CPU_only": True, "Kit_started": False,
                "source_evaluation": record(evaluation_path), "source_ground_audit": record(ground_path),
                "heldout_reference": record(source_reference), "outcomes": outcomes,
                "comparison_run_names": {"previous": previous, "new": new},
                "training_configuration_differences": differences,
                "runtime_argument_differences": {key: {"previous": old["runtime_arguments"].get(key), "new": current["runtime_arguments"].get(key)}
                    for key in sorted(set(old["runtime_arguments"]) | set(current["runtime_arguments"]))
                    if old["runtime_arguments"].get(key) != current["runtime_arguments"].get(key)},
                "same_author_experiment": False, "single_factor_causal_comparison": False,
                "reference_true_com": None, "reference_actual_foot_contact": None,
                "new_native_ground_support": {key: ground[key] for key in
                     ("sources", "support_statistics", "body_origin_motion_proxy", "COM_statistics")},
                "new_native_vs_old_near_ground_toe_proxy_comparable_as_same_metric": False,
                "stagnation": stagnation(directory, ground),
                "clarification_of_frozen_style_null_channels": "Frozen style CLI reads the common physical-state channels only. Its null actual_foot_ground_contact/true_com_world do not deny the separately measured new ground_support.npz. Reference equivalents remain unknown.",
                "limitations": ["Single seed and one reconstruction fixture; not author eight-scene replication.",
                    "Controller, corpus, training budget, D/normalizers/reward and deterministic evaluation settings differ; no single-factor attribution.",
                    "Speed conditioning retains different turn/social/context distributions. Sparse unsupported bins are unknown, not zero error.",
                    "Balanced MMD uses recorded per-run sample counts/IDs, so sparse-bin sample count differences remain disclosed.",
                    "Ground origin-speed proxy includes rolling motion; friction and true contact-point slip are absent.",
                    "Idle diagnosis establishes observed command/progress relations; it does not establish terrain-normalizer causality."]}
    if previous_ground_path is not None:
        previous_ground = json.loads(previous_ground_path.read_text())
        if previous_ground["status"] != "passed":
            raise ValueError("Previous actual ground capture must also be audited")
        document["previous_native_ground_support"] = {key: previous_ground[key] for key in
                    ("sources", "support_statistics", "body_origin_motion_proxy", "COM_statistics")}
        document["previous_stagnation"] = stagnation(Path(previous_ground["input"]), previous_ground)
        document["source_previous_ground_audit"] = record(previous_ground_path)
        document["declared_single_parameter_deployment_ablation"] = (
            set(document["runtime_argument_differences"]) == {"guidance_weight", "output"}
            and old["full_run_sources"]["models_resolved_from_evaluation_arguments"]
            == current["full_run_sources"]["models_resolved_from_evaluation_arguments"]
        )
        document["limitations"][1] = (
            "Declared active models, common training-config values and runtime arguments are equal except guidance_weight/output; "
            "the newly serialized expert_sampling=None is disabled. This is one declared deployment-parameter ablation, "
            "not replicated population-level causality or terrain-normalizer diagnosis. Current source equality is not assumed."
        )
    with output.with_suffix(".json").open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    write_markdown(document, output.with_suffix(".md"))
    plot(document, output.with_suffix(".png"))
    return document


def write_markdown(report, path):
    if report["comparison_run_names"] != {"previous": "previous_best", "new": "backward_ground200"}:
        write_ablation_markdown(report, path)
        return
    old, new = (report["outcomes"][key] for key in ("previous_best", "backward_ground200"))
    lines = ["# 两份真实 200 秒导航采集：冻结风格与停滞诊断", "",
             "两份采集均为 4 人、6000 个 30Hz 控制帧，保持全时段 inclusive 指标，不删跌倒/低速/不直立片段。冻结 validation 参考、窗口、特征与指标均未修改；reference SHA256 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。", "",
             "| 指标 | previous_best | backward_ground200 |", "|---|---:|---:|"]
    for label, key in (("训练 transitions", "training_transitions"), ("跌倒", "falls"),
                       ("到达路线顶点", "total_vertices"), ("完整四顶点 cycles", "completed_four_vertex_cycles"),
                       ("inclusive 速度分层姿态+速度 MMD²", "inclusive_pose_velocity_mmd2"),
                       ("物理速度分布 W1 (m/s)", "root_speed_distribution_W1_mps"),
                       ("直立敏感性保留比例", "upright_retained_fraction"),
                       ("实际静态接触 actor-frames", "static_contact_actor_frames")):
        lines.append(f"| {label} | {old[key]} | {new[key]} |")
    lines.extend(["", "局部 MMD² 下降，但速度分布差距增大、路线顶点 26→21、完整 cycles 5→4。不能判定所有方面都改善或论文风格收敛。旧 controller SHA `e0d6ed4…`，新 `62c4edc…`，Diffusion 同为 `63e7cad…`；完整 source/config 字典见 JSON。", "",
                  "旧池 319 sources，新池 2848 非重叠 segments / 1853 parents；训练量 2.048M→8.192M，同时 D/RMS/reward/profile 与确定性设置不同。因此这不是单一方向奖励消融；真正 sign-only fulltracking 对照在另一个报告。", "",
                  "## 停滞诊断", "",
                  "定义：同 episode 连续位置差分速度 <0.1 m/s；排除跨 reset 跃迁。严格连续段不合并中途小速度波动。", "",
                  "| actor | idle intervals / valid | idle比例 | 最长连续段 (s) | 该段执行 target误差均值 (m) | 双支持 |", "|---|---:|---:|---:|---:|---:|"])
    for row in report["stagnation"]["per_actor"]:
        longest = row["longest_idle"]
        lines.append(f"| {row['actor']} | {row['idle_intervals']} / {row['valid_same_episode_intervals']} | {row['idle_fraction']:.2%} | {longest['duration_s']:.3f} | {longest['executed_target_error_m']['mean']:.5f} | {longest['double_support_fraction_120Hz']:.2%} |")
    longest = report["stagnation"]["per_actor"][0]["longest_idle"]
    lines.extend(["", f"Actor 0 最长严格 idle 为 {longest['start_time_s']:.3f}–{longest['end_time_s']:.3f}s，净位移 {longest['net_displacement_m']:.4f}m；执行 target 误差均值 {longest['executed_target_error_m']['mean']:.4f}m，native 末子步 root速度均值 {longest['native_end_substep_root_speed_mps']['mean']:.4f}m/s。",
                  f"该段原始 Diffusion .5s 位移均值 {longest['raw_plan_half_second_displacement_m']['mean']:.4f}m，5s 位移 {longest['raw_plan_five_second_displacement_m']['mean']:.4f}m；实际路线顶点仍相距约 {longest['actual_route_vertex_distance_m']['mean']:.2f}m。",
                  "原始模型的近段命令已缺乏前进，远端 5 秒移动不等于每 .5 秒重规划会执行该移动。执行 reference 与 root 很近，不能仅据 idle 将原因归于低层 terrain RMS。这里只定位闭环命令/运动关系，未建立归一化因果。",
                  f"该段 root 点到最近 box 的几何距离最小 {longest['minimum_root_point_to_any_AABB_distance_m']:.4f}m；它不是全身体碰撞距离。90/105/120s 的 root/原始与执行 target/路线顶点/box距离原值在 JSON。",
                  "`global_goal_xy` 在 timed_path 中是 A* 五米 lookahead 端点；实际完整路线顶点使用 `route_vertex_xy`，不能将前者误报全局到达。", "",
                  "## 新增真实足力 / COM 与旧代理的边界", "",
                  "冻结风格 CLI 只读取原 physical_states 通道，因此其中 actual-foot/true-COM 保持 null；新 raw ground_support 的独立核验另列，不伪改旧指标。AMASS 对应真实 COM、接触力和摩擦仍未知。",
                  f"新四子步支持占比：双支持 {report['new_native_ground_support']['support_statistics']['double_support_fraction']:.4%}，无足支持 {report['new_native_ground_support']['support_statistics']['no_foot_support_fraction']:.4%}。支持条件下足部 link原点切向速度均值 {report['new_native_ground_support']['body_origin_motion_proxy']['aggregate_supported_mps']['mean']:.5f}m/s。",
                  "后者不是 contact-point slip；滚动/旋转可产生原点运动，不能与旧 near-ground toe 差分代理硬比，也不生成 AMASS 摩擦/接触真值。", "",
                  "![冻结风格分层与速度混合](comparison.png)", "",
                  "完整证据：[冻结原评估](evaluation.json)、[本比较 JSON](comparison.json)、[独立 ground 核验](../ground_support_200s_actual_verification.json)。同 source-parent 与 heldout 无交集已验证；subject/公开 PACER 预训练身份重叠仍未知。"])
    with path.open("x") as stream:
        stream.write("\n".join(lines)+"\n")


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    colors = ("#5b78a6", "#cb7957")
    for offset, (name, outcome) in enumerate(report["outcomes"].items()):
        bins = outcome["speed_bins"]
        mmd = [np.nan if item["mmd2"] is None else item["mmd2"] for item in bins]
        x = np.arange(len(bins))+(offset-.5)*.35
        axes[0].bar(x, mmd, width=.35, color=colors[offset], label=name)
        counts = np.asarray([item["generated_intervals"] for item in bins])
        axes[1].bar(x, counts/counts.sum()*100, width=.35, color=colors[offset], label=name)
    labels = ("0–.1", ".1–.5", ".5–1", "1–1.5", "1.5–2", "2–3", "≥3")
    for axis in axes:
        axis.set_xticks(np.arange(7), labels)
        axis.set_xlabel("Physical root speed bin (m/s)")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Pose + velocity MMD²")
    axes[0].set_title("Frozen heldout reference; unsupported bins are unknown")
    axes[1].set_ylabel("Generated same-episode intervals (%)")
    axes[1].set_title("Physical speed mixture beside conditional style scores")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_ablation_markdown(report, path):
    names = report["comparison_run_names"]
    old, new = (report["outcomes"][names[key]] for key in ("previous", "new"))
    lines = ["# Guidance 20→40：完整 200 秒真实导航对照", "",
             "同一固定 controller、Diffusion、共同训练配置值与 reconstruction fixture；新序列化 metadata 仅增加未启用的 expert_sampling=None。部署 argv 只改变 guidance_weight 20→40（另有输出路径）；不宣称后续工作树全部源码相同或重复运行位级一致。保留全部终止帧、跌倒及 episode reset，冻结原 validation reference/特征/采样协议。完整每-run source/config/runtime 字典在 comparison.json。", "",
             f"| 指标 | {names['previous']} | {names['new']} |", "|---|---:|---:|"]
    for label, key in (("训练 transitions", "training_transitions"), ("跌倒", "falls"),
                       ("路线顶点", "total_vertices"), ("四顶点 cycles累计", "completed_four_vertex_cycles"),
                       ("inclusive 分层姿态+速度 MMD²", "inclusive_pose_velocity_mmd2"),
                       ("物理速度分布 W1 (m/s)", "root_speed_distribution_W1_mps"),
                       ("位移速度 <0.1m/s 的有效 interval比例", "displacement_idle_fraction"),
                       ("直立敏感性保留比例（非主指标）", "upright_retained_fraction"),
                       ("实际静态接触 actor-frames", "static_contact_actor_frames")):
        lines.append(f"| {label} | {old[key]} | {new[key]} |")
    lines.extend(["", "Cycles 为各 actor 累计顶点数整除4再求和；reset 与首次终止的原始时刻均保留，不以该累计数替代无中断 episode 完成率。", "",
                  "## 停滞与命令关系", "",
                  "仅同 episode rootXY差分速度 <0.1m/s 为 idle；不合并短暂速度波动。global_goal_xy 是 timed A* lookahead endpoint，最终路线顶点另看 route_vertex_xy。", "",
                  "| actor | guidance20 idle | guidance40 idle | 20最长(s) | 40最长(s) | 40段 .5s raw位移(m) | 40执行target误差(m) |", "|---|---:|---:|---:|---:|---:|---:|"])
    for previous, current in zip(report["previous_stagnation"]["per_actor"], report["stagnation"]["per_actor"], strict=True):
        a, b = previous["longest_idle"], current["longest_idle"]
        lines.append(f"| {current['actor']} | {previous['idle_fraction']:.2%} | {current['idle_fraction']:.2%} | {a['duration_s']:.3f} | {b['duration_s']:.3f} | {b['raw_plan_half_second_displacement_m']['mean']:.4f} | {b['executed_target_error_m']['mean']:.4f} |")
    lines.extend(["", "每个最长段包含 native 同瞬时 root速度、执行reference速度与误差、原始 .5s/5s 计划的位移/向lookahead投影、双支持和 root点至AABB几何距离。它们定位命令与物理响应关系，不建立 terrain RMS 因果；几何点距离不是全身体碰撞距离。", "",
                  "## 真正足力/COM通道", "",
                  "两份 normal-force sensor 均由独立 CPU audit 重算四子步聚合、COM和时钟；历史 neutral-stance probe=False 不改。AMASS 真 COM/接触/摩擦仍未知。真实接触条件下的 link原点速度代理可按同定义比较，但不能叫接触点滑动；另有 frozen style 的近地 toe差分代理，二者不混用。", "",
                  "![同冻结协议分层风格和物理速度分布](comparison.png)", "",
                  "来源：本目录 ground_support_actual_verification.json 校验实际运行前70文件归档，未用后续工作树宣称全源码不变。style实际输出在 frozen_style/evaluation.json；reference SHA保持 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。单个 seed / fixture 的控制部署对照不是论文八场景或风格验收。"])
    with path.open("x") as stream:
        stream.write("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--ground-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh comparison prefix")
    parser.add_argument("--previous", default="previous_best")
    parser.add_argument("--new", default="backward_ground200")
    parser.add_argument("--previous-ground-audit", type=Path)
    args = parser.parse_args()
    if any(args.output.with_suffix(suffix).exists() for suffix in (".json", ".md", ".png")):
        parser.error("Refusing to overwrite historical reports")
    report = build(args.evaluation.resolve(), args.ground_audit.resolve(), args.output,
                   previous=args.previous, new=args.new, previous_ground_path=args.previous_ground_audit)
    print(json.dumps({"status": report["status"], "output": str(args.output), "idle_actors": len(report["stagnation"]["per_actor"])}))


if __name__ == "__main__":
    main()
