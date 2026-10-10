#!/usr/bin/env python3
"""CPU comparison of independent AMP versus the preserved Diff5000 capture.

Both learned planners and physics are fixed. Controller and route handling are
explicit changes, so this produces a descriptive comparison, not causal proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_paper_pedestrians import read_capture  # noqa: E402


def source(path):
    path = Path(path)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def analyze(path):
    arrays, runner, validated = read_capture(path / "physical_states.npz", path / "result.json")
    evaluation = json.loads((path / "pedestrian_metrics/evaluation.json").read_text())
    phase = json.loads((path / "pedestrian_metrics/phase_quality.json").read_text())
    capture_source = source(path / "physical_states.npz")
    assert evaluation["sources"]["physical_states"]["sha256"] == capture_source["sha256"]
    assert phase["source_sha256"][str((path / "physical_states.npz").resolve())] == capture_source["sha256"]
    root = arrays["root_pos"].astype(np.float64)
    tracking = np.linalg.norm(root[..., :2] - arrays["target_xy"], axis=-1)
    legal = (arrays["episode_ids"][1:] == arrays["episode_ids"][:-1]) & ~arrays["reset_mask"][1:]
    distances = np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1)
    frequency = validated["frequency_hz"]
    falls = int(arrays["terminated"].sum())
    assert frequency == 30 and root.shape[:2] == (6000, 4)
    assert falls == runner["fall_resets"]
    contacts = evaluation["rollout_metrics"]["actual_contacts"]
    assert contacts is not None
    longest_static = []
    for actor in range(4):
        longest = current = 0
        for frame, flagged in enumerate(arrays["static_contact"][:, actor]):
            if frame and (arrays["reset_mask"][frame, actor] or arrays["episode_ids"][frame, actor] != arrays["episode_ids"][frame-1, actor]):
                current = 0
            current = current + 1 if flagged else 0
            longest = max(current, longest)
        longest_static.append(longest / frequency)
    data = {"sources": {"runner": source(path / "result.json"), "capture": capture_source,
                "evaluation": source(path / "pedestrian_metrics/evaluation.json"), "phase": source(path / "pedestrian_metrics/phase_quality.json")},
            "capture_validation": validated, "actor_seconds": 800., "falls": falls,
            "falls_per_actor": arrays["terminated"].sum(0).tolist(), "falls_per1000_actor_seconds": falls / .8,
            "timeouts": int(arrays["truncated"].sum()), "tracking": {**evaluation["rollout_metrics"]["plan_tracking"], "p95_xy_error_m": float(np.quantile(tracking, .95))},
            "walked_xy_m_per_actor": evaluation["rollout_metrics"]["walked_distance_xy_m_per_agent"],
            "walked_xy_m_total": evaluation["rollout_metrics"]["walked_distance_xy_m_total"],
            "mean_xy_speed_excluding_resets_mps": float(distances[legal].mean() * frequency),
            "minimum_pelvis_height_m": evaluation["rollout_metrics"]["minimum_pelvis_ground_height_m"],
            "actual_contacts": contacts, "longest_contiguous_static_contact_s_per_actor": longest_static,
            "pose_generated": {k: v["generated"] for k, v in evaluation["cluster_scores"].items()},
            "gaussian_mmd": evaluation["generated_vs_test_gaussian_mmd"], "goal_metrics": evaluation["goal_condition_metrics"],
            "terminal_events": phase["terminal_contact_relation"], "phases_20s": phase["phases"],
            "route_vertex_progress": phase["recorded_route_vertex_progress"], "arguments": runner["arguments"],
            "physics": runner["physics"], "runtime_execution": runner["runtime_execution"],
            "navigation_guidance": runner["navigation_guidance"], "diffusion_guidance": runner["diffusion_guidance"],
            "route_completion_contract": runner.get("route_completion_contract"),
            "elapsed_wall_s": runner["elapsed_s"], "timing": runner["timing"]}
    return data, evaluation, runner


def contact_summary(data):
    contacts = data["actual_contacts"]
    static = contacts["static"]["actor_frame_statistics"]
    peer = contacts["peer"]["unordered_pair_statistics"]
    peer_actor = contacts["peer"]["actor_frame_statistics"]
    return {"static_actor_frames": static["observed_flagged_frames"], "static_actor_frame_denominator": static["observed_frames"],
            "static_actor_incidences": static["observed_event_starts"], "static_maximum_body_pair_force_n": contacts["static"]["maximum_body_pair_force_n"],
            "peer_unordered_pair_frames": peer["observed_contact_pair_frames"], "peer_pair_frame_denominator": peer["observed_pair_frames"],
            "peer_pair_incidences": peer["contiguous_pair_contact_incidences"], "peer_actor_frames": peer_actor["observed_flagged_frames"],
            "peer_actor_incidences": peer_actor["observed_event_starts"], "peer_maximum_body_pair_force_n": contacts["peer"]["maximum_body_pair_force_n"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2")
    parser.add_argument("--new", type=Path, default=ROOT / "output/diffusion_amp/shared_independent_amp_raw_v1")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/amp_improvement_20261001/independent_amp_physical_comparison")
    args = parser.parse_args()
    old, old_eval, old_runner = analyze(args.baseline.resolve())
    new, new_eval, new_runner = analyze(args.new.resolve())
    protocol_keys = ("embedding_choices", "reference_sampling", "generated_sampling", "cluster_choices", "train_vs_test_gaussian_mmd_reference")
    protocols = {key: old_eval[key] == new_eval[key] for key in protocol_keys}
    clusters = {str(k): (args.baseline / f"pedestrian_metrics/train_pose_clusters_k{k}.json").read_bytes() ==
                      (args.new / f"pedestrian_metrics/train_pose_clusters_k{k}.json").read_bytes() for k in (50, 100)}
    assert all(protocols.values()) and all(clusters.values())
    weights = {"baseline_controller": source(old_runner["arguments"]["resume"]), "new_controller": source(new_runner["arguments"]["resume"]),
               "baseline_diffusion": source(old_runner["arguments"]["diffusion_checkpoint"]), "new_diffusion": source(new_runner["arguments"]["diffusion_checkpoint"])}
    assert weights["baseline_diffusion"] == weights["new_diffusion"]
    assert weights["new_diffusion"]["sha256"] == new_runner["provenance"]["diffusion_sha256"]
    assert weights["baseline_diffusion"]["sha256"] == old_runner["evaluation_planner_change"]["evaluation_sha256"]
    assert old["physics"] == new["physics"] and old["navigation_guidance"] == new["navigation_guidance"] and old["diffusion_guidance"] == new["diffusion_guidance"]
    contact_contract_equal = old["actual_contacts"]["measurement_contract"] == new["actual_contacts"]["measurement_contract"]
    assert contact_contract_equal
    changes = {key: {"baseline": old_runner["arguments"].get(key), "new": new_runner["arguments"].get(key)}
               for key in old_runner["arguments"].keys() | new_runner["arguments"].keys()
               if old_runner["arguments"].get(key) != new_runner["arguments"].get(key)}
    manifest_path = ROOT / "docs/amp_improvement_20261001/shared_independent_amp_raw_source.json"
    archive_path = manifest_path.with_suffix(".tar.gz")
    manifest = json.loads(manifest_path.read_text())
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {row["path"] for row in manifest["files"]}
        for row in manifest["files"]:
            data = archive.extractfile(row["path"]).read()
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    source_differences = [{**row, "current_sha256": source(ROOT / row["path"])["sha256"]} for row in manifest["files"] if source(ROOT / row["path"])["sha256"] != row["sha256"]]
    route_verification = json.loads((ROOT / "docs/amp_improvement_20261001/raw_route_completion_verification.json").read_text())
    assert route_verification["capture_sha256"] == new["sources"]["capture"]["sha256"]
    contacts = {"baseline": contact_summary(old), "new": contact_summary(new)}
    report = {"schema": "independent_amp_physical_comparison_v1", "status": "cpu_analyzed", "baseline": old, "new": new,
              "weights": weights, "argument_changes": changes, "pose_protocol_equal": protocols, "train_cluster_artifacts_byte_identical": clusters,
              "physics_manifest_equal": True, "effective_diffusion_and_navigation_guidance_equal": True, "contact_contract_equal": contact_contract_equal,
              "contacts_compact": contacts, "raw_route_verification": route_verification,
              "source_archive": source(archive_path), "source_manifest": source(manifest_path), "archived_files_match_manifest": len(manifest["files"]),
              "post_launch_working_tree_differences": source_differences,
              "explicit_changes": ["Controller changed: prior Diffusion-command-trained corrected1000 -> independent smooth-command AMP1000, both public actor/critic transfer plus random state_action D.",
              "New runner advances vertices on every alive 30Hz post-step physical pose, resets route index to0, does not count reset position as arrival, and forces a next-control full-scene replan after a reached vertex.",
              "Old reported route counts were collected under earlier arrival/reset handling; old capture lacks route_indices/route_vertex_reached. Full-cycle change cannot be reconstructed.",
              "DDPM input batch/shape, seed7, FP32,100denoising steps,15-control replanning, raw XY and endpoint guidance are retained. State evolution and extra arrival-triggered replans can change later RNG consumption; trajectories are not bitwise paired.",
              "New reference_adapter=none. Added CLI defaults do not imply this run applied causal filtering or timed waypoints.",
              "Source archives retain launch hashes; later working-tree changes are recorded separately."],
              "single_factor_causal_claim": False, "author_metric_equivalence": False,
              "limitations": ["One 200-second realization per configuration; no multi-seed significance claim.", "Few falls do not establish realistic style, unobstructed motion, or a full route success rate.",
              "Per-frame 45D pose MMD/coverage do not evaluate temporal gait; all local embedding/bandwidth/node choices are explicitly matched.",
              "Contact is strict normal force>1N, maximum body-pair magnitude over four substeps; floor/self/tangential friction excluded. No author aggregation claimed.",
              "Contact preceding a fall is temporal association, not proof of cause."], "analysis_script": source(__file__)}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for label, key in (("跌倒终止", "falls"), ("每1000行人秒跌倒", "falls_per1000_actor_seconds"), ("排除reset的总XY路程m", "walked_xy_m_total"), ("平均XY速度m/s", "mean_xy_speed_excluding_resets_mps")):
        table.append(f"| {label} | {old[key]:.6g} | {new[key]:.6g} |")
    for label, key in (("Tracking mean m", "mean_xy_error_m"), ("Tracking RMS m", "rmse_xy_m"), ("Tracking p95 m", "p95_xy_error_m")):
        table.append(f"| {label} | {old['tracking'][key]:.6g} | {new['tracking'][key]:.6g} |")
    table.append(f"| Gaussian MMD²，σ1 | {old['gaussian_mmd']['mmd_squared']:.9f} | {new['gaussian_mmd']['mmd_squared']:.9f} |")
    for k in ("50", "100"):
        table.append(f"| K{k} coverage/归一化entropy | {old['pose_generated'][k]['coverage']:.2f}/{old['pose_generated'][k]['normalized_pose_entropy']:.6f} | {new['pose_generated'][k]['coverage']:.2f}/{new['pose_generated'][k]['normalized_pose_entropy']:.6f} |")
    for label, key in (("Static actor-frames/24000", "static_actor_frames"), ("Static连续actor事件", "static_actor_incidences"), ("Peer unordered pair-frames/36000", "peer_unordered_pair_frames"), ("Peer连续pair事件", "peer_pair_incidences")):
        table.append(f"| {label} | {contacts['baseline'][key]} | {contacts['new'][key]} |")
    table.append(f"| A*失败查询 | {old['goal_metrics']['global_path_failures']} | {new['goal_metrics']['global_path_failures']} |")
    table.append(f"| plan endpoint到condition goal均差m | {old['goal_metrics']['recorded_plan_endpoint_goal_residual']['mean_m']:.6g} | {new['goal_metrics']['recorded_plan_endpoint_goal_residual']['mean_m']:.6g} |")
    phases = []
    for left, right in zip(old["phases_20s"], new["phases_20s"], strict=True):
        phases.append(f"| {right['physical_time_range_s']} | {left['terminal_frames']}/{right['terminal_frames']} | {left['tracking_mean_m']:.4f}/{right['tracking_mean_m']:.4f} | {right['static_contact_actor_frames']} | {right['peer_contact_unordered_pair_frames']} | {right['mean_planar_speed_m_s']:.4f} |")
    text = f"""# 独立课程 AMP1000：真实四行人200秒对照

## 同暴露结果

CPU读取两份6000帧×4人、30Hz真实物理记录，每组800行人秒。保持同Diffusion5000、clean guidance20、peer5/radius.4、signed-AABB10/radius.5、endpoint、FP32、DDPM100、seed7与物理manifest。**控制器和route到达/reset处理同时变化，不能单因归因。**

| 指标 | previous clean_sdf_5000_v2 | independent_amp_raw_v1 |
|---|---:|---:|
{chr(10).join(table)}

每人跌倒：{old['falls_per_actor']} → {new['falls_per_actor']}。总运动350.954→355.604m近似持平。跌倒减少，但static接触显著增加、tracking误差和姿态MMD增大、覆盖/entropy下降，**不能称综合质量或风格已经改善**。两run都没有超时终止。

## 采样和模型证据

train/test各2048姿态、seed7、15节点45D heading/root/height embedding、Gaussian σ1以及2048个time-major生成索引逐项完全相同。K50/100均只在train fit，两套cluster JSON逐字节相同；train/test参考MMD²={old_eval['train_vs_test_gaussian_mmd_reference']['mmd_squared']:.9f}。没有val/generated重新fit中心，指标只代表本地显式协议，非论文未知超参数的精确值。

共同Diffusion SHA256：`{weights['new_diffusion']['sha256']}`。

旧AMP SHA256：`{weights['baseline_controller']['sha256']}`；新独立课程AMP SHA256：`{weights['new_controller']['sha256']}`。新低层1000训练仅消费独立smooth课程，公开PACER actor/critic/task与moments warm start + 随机state_action2019 D；旧低层1000训练消费当时的Diffusion命令。新训练完整性见[训练审查](independent_amp_training_review.md)。

两run真实contact合同逐字段相等：normal force>1N，每30Hz控制帧取4个120Hz子步body-pair最大值，排除floor/self，不包含切向摩擦。旧→新static接触比例1.29583%→7.00000%；peer pair比例1.63333%→0.00278%，并非同一种碰撞量。新static per-actor帧数{new['actual_contacts']['static']['contact_actor_frames_per_agent']}；最长连续static接触秒数{new['longest_contiguous_static_contact_s_per_actor']}。

## Route合同变化与完整循环

旧记录累计vertices={old['goal_metrics']['route_vertices_reached']}，新={new['goal_metrics']['route_vertices_reached']}；这两个累计计数合同不同。新runner在每个活着的30Hz post-step姿态检查到达，物理reset把route index归0且reset位置不算进度，到达vertex后下一control立即整组强制replan；旧在更稀疏的规划查询中推进与统计，reset行为也不同。新capture保存route_indices/reached，可逐episode验证12次实际到达，全都位于.75m半径内、无fall/reset帧伪计数。新actor2/3各完成1个四顶点完整循环，共2个；actor3之后仍于156.6s跌倒。旧capture缺这两个数组，旧完整循环数为None，不能宣称从0升到2或作者benchmark成功。

最初DDPM噪声形状、batch设置保持；物理状态和额外到达触发replan会改变后续调用/RNG消费，不能说整个200秒轨迹逐位配对。配置文件虽声明timed_path/recovery.5，本run实际CLI endpoint、无start recovery，`reference_adapter=none`，未使用causal reference限速/限加速度。实际physics manifest完全相同；旧arguments未显式写terrain_extent，新写64m，不将缺失字段当物理改变。

## 跌倒时间关系与滞留

3个fall：66.6s actor0，static前.0667s且近2秒49接触帧；102.1667s actor1，该episode没有已记录static/peer；156.6s actor3，当帧static。全部无近2秒peer。只描述时间关联，不推断接触必然导致跌倒。

新actor0在140–200秒每20秒仅走.0783/.0747/.0811m；少跌倒与滞留可以同时出现。新A*失败135次（旧23），goal可按无路径合同返回current root，因此小tracking/condition-goal误差不能替代全局进展。新actor0末60秒static帧为0，不能把所有停滞归为持续墙contact。

| 时段s | 旧/新falls | 旧/新tracking mean m | 新static actor-frames | 新peer pair-frames | 新均速m/s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(phases)}

![同暴露分段对照](independent_amp_physical_comparison.png)

## Provenance与下一对照

新run源归档{len(manifest['files'])}文件逐bytes/hash匹配启动manifest；当前树后续改变单列于JSON，不替代启动hash。墙钟旧{old['elapsed_wall_s']:.2f}s、新{new['elapsed_wall_s']:.2f}s，接触/规划/物理耗时不是纯算法性能benchmark。只有单seed，episode删失和route合同变更限制结论。

下一步保持这些权重与真实contact协议，单列timed-path/causal-reference部署消融，继续报告falls、完整同episode循环、A*失败、static滞留、目标跟踪与固定pose指标。不能把当前少跌倒作为风格收敛、社交导航或严格论文复现的验收。

完整fields/hashes/argument changes：[JSON](independent_amp_physical_comparison.json)。新逐帧与contact证据：[phase_quality.md](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_independent_amp_raw_v1/pedestrian_metrics/phase_quality.md)。CPU脚本：[compare_independent_amp_physical_runs.py](compare_independent_amp_physical_runs.py)。
"""
    args.output_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/protomotions-matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for run, label in ((old, "Previous AMP1000 / raw Diff5000"), (new, "Independent AMP1000 / raw Diff5000")):
        x = [np.mean(p["physical_time_range_s"]) for p in run["phases_20s"]]
        for axis, field in zip(axes, ("terminal_frames", "tracking_mean_m", "mean_planar_speed_m_s", "static_contact_actor_frames"), strict=True):
            axis.plot(x, [p[field] for p in run["phases_20s"]], marker="o", label=label, linewidth=1.4)
    for axis, ylabel in zip(axes, ("Falls / 20s", "Tracking mean (m)", "Root XY speed (m/s)", "Static actor-frames / 20s"), strict=True):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Same planner + physics; controller and route handling changed; one-seed descriptive comparison")
    axes[-1].set_xlabel("Physical interval midpoint (s)")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)
    report["plot"] = source(args.output_prefix.with_suffix(".png"))
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "cpu_analyzed", "falls": [old["falls"], new["falls"]], "mmd_squared": [old["gaussian_mmd"]["mmd_squared"], new["gaussian_mmd"]["mmd_squared"]], "all_pose_protocols_equal": True, "output_prefix": str(args.output_prefix)}))


if __name__ == "__main__":
    main()
