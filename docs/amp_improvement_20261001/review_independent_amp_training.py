#!/usr/bin/env python3
"""CPU audit of a completed independent-curriculum AMP run and fixed snapshot.

This never starts a simulator or restores CUDA RNG. It reads immutable step_1000,
not the mutable last.pt, and reuses the repository's JSONL snapshot reader.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tarfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.summarize_paper_training import read_metrics, split_sessions  # noqa: E402


def snapshot(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def tensor_leaves(value, prefix=""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{prefix}[{index}]")


def assert_exact_tensors(saved, loaded):
    import torch

    before, after = dict(tensor_leaves(saved)), dict(tensor_leaves(loaded))
    assert before.keys() == after.keys(), "Tensor schema changed during strict CPU load"
    for name, value in before.items():
        restored = after[name]
        assert value.dtype == restored.dtype and value.shape == restored.shape and torch.equal(value, restored), name
    return len(before)


def audit_checkpoint(path):
    import numpy as np
    import torch

    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum

    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    assert unsafe == [], unsafe
    state = torch.load(path, map_location="cpu", weights_only=True)
    leaves = list(tensor_leaves(state))
    assert leaves and all(t.device.type == "cpu" and bool(torch.isfinite(t).all()) for _, t in leaves)
    trainer, runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
    assert state["update_count"] == trainer.update_count == 1000
    assert state["physical_transition_count"] == trainer.physical_transition_count == 2_048_000
    assert trainer.model.discriminator[0].in_features == 2019
    assert state["config"]["discriminator_input"] == "state_action"
    exact = {}
    exact["model"] = assert_exact_tensors(state["model"], trainer.model.state_dict())
    exact["optimizer"] = assert_exact_tensors(state["optimizer"], trainer.optimizer.state_dict())
    exact["discriminator_optimizer"] = assert_exact_tensors(state["discriminator_optimizer"], trainer.discriminator_optimizer.state_dict())
    exact["normalizers"] = assert_exact_tensors(state["normalizers"], {k: getattr(trainer, k).state_dict() for k in state["normalizers"]})
    assert_exact_tensors(state["runner_state"], runner)
    assert_exact_tensors(state["rng"], trainer._pending_rng_state)
    optimizers = {}
    for name, live in (("optimizer", trainer.optimizer), ("discriminator_optimizer", trainer.discriminator_optimizer)):
        saved = state[name]
        ids = [i for group in saved["param_groups"] for i in group["params"]]
        params = [p for group in live.param_groups for p in group["params"]]
        assert set(ids) == set(saved["state"]) and len(ids) == len(params)
        steps = []
        for identifier, parameter in zip(ids, params, strict=True):
            entry = saved["state"][identifier]
            assert set(entry) == {"step", "exp_avg", "exp_avg_sq"}
            assert entry["exp_avg"].shape == entry["exp_avg_sq"].shape == parameter.shape
            assert bool((entry["exp_avg_sq"] >= 0).all())
            steps.append(float(entry["step"]))
        assert all(step == 6000 for step in steps), steps
        optimizers[name] = {"parameter_states": len(ids), "adam_step_min": min(steps), "adam_step_max": max(steps), "moment_shapes_match_parameters": True}
    normalizers = {name: {"shape": list(v["mean"].shape), "count": float(v["count"]), "mean_abs_max": float(v["mean"].abs().max()), "variance_min": float(v["variance"].min()), "variance_max": float(v["variance"].max())} for name, v in state["normalizers"].items()}
    rng = state["rng"]
    random.Random().setstate(rng["python"])
    ns = rng["numpy"]
    np.random.RandomState().set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    for value in (rng["torch_cpu"], runner["corpus_rng"]):
        assert value.dtype == torch.uint8 and value.device.type == "cpu"
        torch.Generator(device="cpu").set_state(value)
    assert all(t.dtype == torch.uint8 and t.ndim == 1 and t.numel() > 0 for t in rng["torch_cuda"])
    curriculum_state = runner["curriculum_state"]
    config = PacerTrajectoryConfig(**curriculum_state["config"])
    first, second = PacerTrajectoryCurriculum(config, seed=1), PacerTrajectoryCurriculum(config, seed=2)
    first.load_state_dict(curriculum_state)
    second.load_state_dict(curriculum_state)
    assert_exact_tensors(curriculum_state, first.state_dict())
    physical = SimpleNamespace(**runner["physical"])
    plan_a, plan_b = first.build_plan(physical), second.build_plan(physical)
    for name in ("xy", "velocities_xy", "times_s", "speed_goals_mps", "sharp_turn_mask", "boundary_reflection_mask", "initial_velocity_clipped"):
        assert torch.equal(getattr(plan_a, name), getattr(plan_b, name)), name
    assert first.plans_generated == second.plans_generated == curriculum_state["plans_generated"] + 64
    assert torch.equal(first.generator.get_state(), second.generator.get_state())
    assert runner["control_step"] == 32000
    assert runner["runtime_execution"]["diffusion_used_for_rollouts"] is False
    assert runner["runtime_execution"]["consumed_planner_output"] is None
    assert torch.cuda._initialized is False
    return {"status": "passed", "source": snapshot(path), "weights_only": True, "unsafe_globals": unsafe,
            "all_tensor_leaves": len(leaves), "tensor_elements": sum(t.numel() for _, t in leaves),
            "all_tensor_leaves_finite_and_cpu": True, "strict_load_exact_tensor_leaves": exact,
            "optimizers": optimizers, "normalizers": normalizers, "discriminator_input_width": 2019,
            "action_width": 69, "effective_action_coordinates": sum(state["config"]["action_mask"]),
            "update_count": trainer.update_count, "physical_transition_count": trainer.physical_transition_count,
            "runner_control_step": runner["control_step"], "runner_state_keys": list(runner),
            "curriculum_plans_generated": curriculum_state["plans_generated"],
            "private_curriculum_rng_dtype": str(curriculum_state["generator_state"].dtype),
            "private_curriculum_rng_shape": list(curriculum_state["generator_state"].shape),
            "private_curriculum_exact_continuation_from_saved_physical_state": True,
            "python_numpy_torch_cpu_corpus_rng_validated_with_private_instances": True,
            "saved_cuda_rng_states": len(rng["torch_cuda"]), "cuda_rng_validation": "finite CPU uint8 structure only; not restored or executed",
            "cuda_initialized": False, "global_rng_restored": False,
            "limitations": "CPU strict load and private command continuation do not prove bitwise PhysX solver/contact-cache resume."}


MEANS = ("reward_mean", "amp_rewards_mean", "trajectory_rewards_mean", "energy_rewards_mean", "direction_rewards_mean",
         "signed_power_w_mean", "forward_velocity_mean", "backwards_fraction", "value_loss", "approximate_kl", "clip_fraction",
         "disc_expert_probability", "disc_policy_probability", "disc_classification", "disc_gradient_penalty")


def segments(records, control_hz):
    result = []
    for start in range(0, 1000, 250):
        rows = records[start:start + 250]
        before = records[start - 1] if start else {"fall_resets": 0, "time_limit_resets": 0, "physical_transitions": 0}
        transitions = rows[-1]["physical_transitions"] - before["physical_transitions"]
        falls = rows[-1]["fall_resets"] - before["fall_resets"]
        timeouts = rows[-1]["time_limit_resets"] - before["time_limit_resets"]
        result.append({"first_update": rows[0]["update_count"], "last_update": rows[-1]["update_count"],
                       "physical_transitions": transitions, "actor_seconds": transitions / control_hz,
                       "fall_resets": falls, "time_limit_resets": timeouts,
                       "falls_per_1000_actor_seconds": falls * 1000 * control_hz / transitions,
                       "means_of_250_equal_size_rollout_means": {key: statistics.mean(row[key] for row in rows) for key in MEANS}})
    return result


def plot(records, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/protomotions-matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = [r["update_count"] for r in records]
    for key, label in (("reward_mean", "Total"), ("amp_rewards_mean", "AMP"), ("trajectory_rewards_mean", "Trajectory (unweighted)"), ("energy_rewards_mean", "Energy (unweighted)"), ("direction_rewards_mean", "Direction (unweighted)")):
        axes[0, 0].plot(x, [r[key] for r in records], label=label, linewidth=.9)
    for key, label in (("disc_expert_probability", "Expert D probability"), ("disc_policy_probability", "Policy D probability")):
        axes[0, 1].plot(x, [r[key] for r in records], label=label, linewidth=1)
    for key, label in (("fall_resets", "Falls"), ("time_limit_resets", "Time limits")):
        values = [r[key] for r in records]
        axes[0, 2].plot(x, [values[0]] + [b - a for a, b in zip(values, values[1:])], label=label, linewidth=.9)
    for key, label in (("forward_velocity_mean", "Root forward velocity (m/s)"), ("backwards_fraction", "Backwards fraction")):
        axes[1, 0].plot(x, [r[key] for r in records], label=label, linewidth=.9)
    for key, label in (("value_loss", "Value loss"), ("approximate_kl", "Approximate KL")):
        axes[1, 1].plot(x, [r[key] for r in records], label=label, linewidth=.9)
    axes[1, 2].plot(x, [r["signed_power_w_mean"] for r in records], label="Signed projected joint power (W)", linewidth=.9)
    titles = ("Reward components", "State + action discriminator", "Per-update reset events", "Physical rollout velocity", "PPO diagnostics", "Synchronized 120 Hz power")
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set(title=title, xlabel="Optimizer update")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.suptitle("Independent smooth curriculum AMP: 1000 updates, raw metrics; convergence unverified")
    figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/diffusion_amp/amp_independent_curriculum_v1")
    parser.add_argument("--probe", type=Path, default=ROOT / "output/diffusion_amp/amp_independent_curriculum_resume_probe_v1")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/amp_improvement_20261001/independent_amp_training_review")
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    records, metric_source = read_metrics(args.run / "metrics.jsonl")
    sessions, restarts = split_sessions(records)
    assert len(sessions) == 1 and not restarts and len(records) == 1000
    assert [r["update_count"] for r in records] == list(range(1, 1001))
    assert [r["physical_transitions"] for r in records] == [2048 * i for i in range(1, 1001)]
    assert all(math.isfinite(v) for row in records for v in row.values() if type(v) in (int, float))
    assert result["status"] == "passed" and result["update_count"] == 1000
    assert result["physical_transition_count"] == 2_048_000
    assert result["diffusion_plans"] == result["planning_clock"]["forward_calls"] == 0
    assert result["trajectory_source"] == "pacer_curriculum" and result["convergence_claim"] is False
    for key in ("fall_resets", "time_limit_resets", "resets", "minimum_height_m"):
        assert result[key] == records[-1][key]
    checkpoint = audit_checkpoint(args.run / "step_1000.pt")
    immutable = [json.loads(line) for line in (args.run / "immutable_checkpoints.jsonl").read_text().splitlines()]
    assert immutable[-1]["sha256"] == checkpoint["source"]["sha256"]
    assert immutable[-1]["bytes"] == checkpoint["source"]["bytes"]
    source_manifest_path = ROOT / "docs/amp_improvement_20261001/independent_curriculum_training_source.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_archive_path = source_manifest_path.with_suffix(".tar.gz")
    with tarfile.open(source_archive_path) as archive:
        assert set(archive.getnames()) == {entry["path"] for entry in source_manifest["files"]}
        for entry in source_manifest["files"]:
            archived = archive.extractfile(entry["path"]).read()
            assert len(archived) == entry["bytes"] and hashlib.sha256(archived).hexdigest() == entry["sha256"]
    current_sources = [{**entry, "current_sha256": snapshot(ROOT / entry["path"])["sha256"]} for entry in source_manifest["files"]]
    source_differences = [entry for entry in current_sources if entry["sha256"] != entry["current_sha256"]]
    hz = result["physics"]["control_hz"]
    transitions = result["physical_transition_count"]
    actor_seconds = transitions / hz
    counters = {key: result[key] for key in ("fall_resets", "time_limit_resets", "resets", "minimum_height_m")}
    counters.update(actor_seconds=actor_seconds, simultaneous_simulated_seconds=result["actual_control_steps"] / hz,
                    falls_per_1000_actor_seconds=result["fall_resets"] * 1000 / actor_seconds,
                    time_limits_per_1000_actor_seconds=result["time_limit_resets"] * 1000 / actor_seconds,
                    fall_fraction_of_completed_episodes=result["fall_resets"] / result["resets"],
                    transitions_per_wall_second=transitions / result["elapsed_s"],
                    updates_per_wall_second=result["update_count"] / result["elapsed_s"])
    probe_result = json.loads((args.probe / "result.json").read_text())
    probe_rows, probe_source = read_metrics(args.probe / "metrics.jsonl")
    probe_sessions, probe_restarts = split_sessions(probe_rows)
    assert [row["update_count"] for row in probe_rows] == [1, 2, 3, 4]
    assert probe_result["physical_transition_count"] == 8192 and probe_result["planning_clock"]["control_step"] == 128
    assert len(probe_sessions) == 2
    probe = {"source": snapshot(args.probe / "result.json"), "metrics_source": probe_source,
             "status": probe_result["status"], "logical_updates": 4, "physical_transitions": 8192,
             "restored_global_control_step": 128, "actor_seconds": 8192 / hz,
             "simultaneous_simulated_seconds": 128 / hz, "process_restarts": probe_restarts,
             "sessions": [{"first_update": rows[0]["update_count"], "last_update": rows[-1]["update_count"],
                           "fall_resets": rows[-1]["fall_resets"], "time_limit_resets": rows[-1]["time_limit_resets"],
                           "process_elapsed_s": rows[-1]["elapsed_s"]} for rows in probe_sessions],
             "falls_across_sessions": sum(rows[-1]["fall_resets"] for rows in probe_sessions),
             "note": "Two genuine 2-update processes, fresh then resumed; final result counters/actual_control_steps=64 are resumed-process-local, global control_step=128. Not a claim of solver-cache bitwise continuation."}
    report = {"schema": "independent_amp_training_review.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "cpu_integrity_audit_passed", "sources": {"result": snapshot(args.run / "result.json"), "metrics": metric_source,
              "source_manifest": snapshot(source_manifest_path), "source_archive": snapshot(source_archive_path)},
              "fixed_checkpoint": checkpoint, "immutable_checkpoint_manifest": immutable,
              "source_snapshot_files": len(current_sources), "source_archive_matches_all_manifest_hashes": True,
              "post_launch_source_differences": source_differences,
              "source_snapshot_interpretation": "The frozen archive, not the later working-tree hash, defines this run. At this audit scripts/train_paper_amp.py and application/configuration.py add causal_accel deployment branches/CLI; this completed curriculum run recorded no reference-adapter option and zero diffusion forwards. No later source hash is substituted into launch provenance.",
              "run": {key: result[key] for key in ("method", "arguments", "trajectory_source", "curriculum", "initialization", "runtime_execution", "amp_reward_timestamp", "power_measurement", "training_config", "update_count", "physical_transition_count", "actual_control_steps", "elapsed_s", "curriculum_plans", "planning_clock", "timing")},
              "training_exposure_and_events": counters, "segments": segments(records, hz), "first_update": records[0], "last_update": records[-1], "resume_probe": probe,
              "sampling_and_formula_choices": {"environments": 64, "rollout_steps": 32, "samples_per_update": 2048,
              "expert_train_clips": result["manifest_gate"]["counts"]["train"], "corpus_manifest_sha256": result["manifest_gate"]["manifest_sha256"],
              "state_action_input": "10 newest-first 195-dimensional frames + applied canonical 69 actions = 2019",
              "expert_actions": "next-frame mocap joint-position inverse-PD targets; no observed human expert commands",
              "reward": "AMP + 0.5*(trajectory + signed projected joint power energy term + published literal direction term)",
              "segment_mean_definition": "Arithmetic mean of 250 logged rollout means, each based on 64*32 actor-controls; episode events are cumulative differences at window endpoints.",
              "trajectory_reward_not_deployment_tracking": True},
              "conclusions": {"complete_finite_resume_bundle": True, "diffusion_used_for_rollouts": False,
              "style_convergence_established": False, "obstacle_social_or_deployment_convergence_established": False,
              "author_experiment_reproduced": False,
              "remaining_gate": "Evaluate this new controller on fixed analytic commands and identical Diffusion5D/shared-scene captures, with physical contacts, falls, target errors and train-only pose clusters/MMD. Training task reward alone does not test deployment."}}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot(records, args.output_prefix.with_suffix(".png"))
    report["plot"] = snapshot(args.output_prefix.with_suffix(".png"))
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    table = []
    for window in report["segments"]:
        means = window["means_of_250_equal_size_rollout_means"]
        table.append(f"| {window['first_update']}–{window['last_update']} | {window['fall_resets']} | {window['time_limit_resets']} | {means['reward_mean']:.4f} | {means['amp_rewards_mean']:.4f} | {means['trajectory_rewards_mean']:.4f} | {means['disc_expert_probability']:.4f}/{means['disc_policy_probability']:.4f} | {means['forward_velocity_mean']:.4f} |")
    text = f"""# 独立低层课程 AMP 训练审查

## 完成与边界

真实 PhysX 训练完成 **1000 次 PPO/AMP 更新、2,048,000 个物理转移**。64 个隔离环境，每次32控制步；120Hz物理、30Hz控制，共32,000控制步、1066.667秒并行仿真、68,266.667行人秒。运行耗时 {result['elapsed_s']:.3f} 秒，{counters['transitions_per_wall_second']:.2f} 转移/秒（含采样、物理、优化和保存；并非纯物理速度）。原始日志1000条完整且全部数值有限。

这是独立低层命令课程训练；**Diffusion 在训练 rollout 中调用0次**。`step_5000.ckpt` 被绑定用于另外的部署合同，没有消费其规划输出。地面128m、spawn margin48m、episode300步=10秒，5秒命令窗，默认不反射、不修改物理 root。公开 PACER 参数基础上显式使用 smooth 适配：目标速度独立均匀采样0–3m/s，标量/向量加速度上限2m/s²、heading速率2rad/s；不是公开尖转生成器逐位复现。[TRACE/PACER §3.3](https://arxiv.org/html/2304.01893v1#S3.SS3) 明确两级独立训练。

初始化移植公开 PACER actor、critic、task encoder、观测和value归一化矩；判别器为**独立随机初始化 state_action**，2019输入=10×195姿态历史+69应用动作，57个有效动作坐标。没有移植公开1950维state-only判别器。公开权重SHA256：`{result['initialization']['sha256']}`。专家动作为下一帧 mocap 姿态的 inverse-PD target 代理；不能解释为真实专家控制指令。

## 跌倒与超时分母

训练全程 **{result['fall_resets']} 次跌倒、{result['time_limit_resets']} 次10秒超时、{result['resets']} 次完成episode/reset**。跌倒定义为pelvis相对地面高度<0.45m或root up-z<0.1；运行最低pelvis {result['minimum_height_m']:.6f}m。跌倒率 **{counters['falls_per_1000_actor_seconds']:.4f}/1000行人秒**；完成episode中跌倒占 **{100*counters['fall_fraction_of_completed_episodes']:.4f}%**。超时不计入跌倒。尚在运行的64个episode未作为完成episode分母。未按“87/2,048,000”冒充独立样本失败概率。

## 分段奖励与风格

每格为250次等大小rollout的均值；事件计数取分段首尾累计差。轨迹组件是对齐的单步位置奖励，不能当作部署路径到达率。

| 更新 | 跌倒 | 超时 | 总奖励 | AMP奖励 | 轨迹奖励 | D expert/policy | 前向速度m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

末次更新 D expert/policy={records[-1]['disc_expert_probability']:.4f}/{records[-1]['disc_policy_probability']:.4f}，AMP均奖={records[-1]['amp_rewards_mean']:.4f}，轨迹均奖={records[-1]['trajectory_rewards_mean']:.4f}。判别器仍明显区分专家和策略，**尚不能确认风格收敛**。总奖励使用AMP+0.5×任务和；signed power 为同一120Hz子步的 projected active joint effort×末端关节速度，4子步时间平均，非隔离motor torque。保留论文打印的direction符号，未在本轮静默替换。

![原始训练曲线](independent_amp_training_review.png)

## 固定 checkpoint 与恢复完整性

审核对象为不可变 `step_1000.pt`，**{checkpoint['source']['bytes']:,} bytes**，SHA256 **`{checkpoint['source']['sha256']}`**；与immutable manifest一致。CPU `weights_only=True` 读取成功，unsupported globals为空；{checkpoint['all_tensor_leaves']}个Tensor叶子全部有限。strict加载模型、两个Adam optimizer和四套normalizer后逐Tensor精确相等；policy/D optimizer分别{checkpoint['optimizers']['optimizer']['parameter_states']}/{checkpoint['optimizers']['discriminator_optimizer']['parameter_states']}份完整参数状态，Adam step均6000，moment形状匹配参数。更新/转移/runner control counter分别1000/2,048,000/32,000。

课程私有CPU RNG uint8[5056]、plans_generated={checkpoint['curriculum_plans_generated']}与物理/历史/计划/episode状态已保存。两个独立课程实例从同一保存状态生成下一批64人命令，所有输出和下一RNG状态逐Tensor相等。Python、NumPy、Torch CPU和corpus RNG用私有实例验证；CUDA RNG只验证保存结构，没有恢复或初始化CUDA。CPU校验不证明PhysX接触缓存/solver的逐位续跑等价。

源码归档的{len(current_sources)}个文件逐一与冻结manifest的bytes/hash匹配。审查时当前树有{len(source_differences)}项后续变化（runner与CLI新增causal_accel部署实验）；本次result没有该适配选项，Diffusion调用为0。当前树hash不是运行时hash，本报告保留原启动归档，不追溯改写provenance。

## 4更新实际续训探针

探针共8192物理转移：先2更新再从checkpoint续训2更新，逻辑control_step128、4.2667秒并行仿真、273.0667行人秒。第一进程跌倒0，第二进程跌倒2，超时0。末result的`actual_control_steps=64`及计数描述最后进程；不是8192转移的单次不中断物理过程。课程计划/RNG/训练计数随恢复保留。

## 下一验收

完成且状态有限不等于低层风格收敛，也未测试共享障碍/社交部署。本轮没有scene、没有Diffusion rollout；需要用固定解析命令和相同Diffusion5D/共享场景评估新controller，联合物理contact、跌倒、目标误差与train-only姿态K/MMD。没有论文原始2848条动作ID、作者权重及八场景完整manifest，不能声明作者实验复现。

完整字段/分段component/value/KL、normalizer矩和证据hash：[JSON](independent_amp_training_review.json)。可重跑脚本：[review_independent_amp_training.py](review_independent_amp_training.py)；本次只创建CPU审查工件，没有修改训练代码或checkpoint。
"""
    args.output_prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "sha256": checkpoint["source"]["sha256"], "tensor_leaves": checkpoint["all_tensor_leaves"], "source_differences": len(source_differences), "output_prefix": str(args.output_prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
