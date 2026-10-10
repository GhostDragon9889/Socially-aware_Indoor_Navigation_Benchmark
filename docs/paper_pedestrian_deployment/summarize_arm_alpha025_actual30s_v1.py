"""Describe paired actual 30s runs with the existing frozen style formulas.

Only the optional stagnation diagnostic snapshots change from90/105/120s to
10/20/30s so they stay inside the actual capture. No score/idle/phase formula
changes. This does not import or run a simulator.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "experimental_arm_alpha025_actual30s_comparison_v1"
FROZEN = {
    "scripts/evaluate_pedestrian_style.py": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "CrowdSim/humanoid/style_evaluation.py": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
    "docs/style_improvement_20261002/summarize_navigation_style.py": "ee0d13c3f345cf11aa2981002c21b070bf53eb4b6129d86d4a6f851443ac56d7",
    "docs/style_improvement_20261002/audit_ground_support_capture.py": "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f",
    "docs/style_improvement_20261002/mixed_curriculum_navigation_style_v1/summarize.py": "b0b593930362a525abb37f685abacb7a2b5e7d5f05758a6a9ce499ad467d55d2",
}
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or (OUT / "comparison.json").exists():
        raise ValueError("CPU-only and fresh report required")
    sys.path.insert(0, str(ROOT))
    for path, expected in FROZEN.items():
        if record(ROOT / path)["sha256"] != expected:
            raise ValueError("Frozen formula/source changed:" + path)
    evaluation = json.loads((OUT / "frozen_style/evaluation.json").read_text())
    reference = OUT / "frozen_style/heldout_reference.json"
    if evaluation["status"] != "passed" or record(reference)["sha256"] != REFERENCE_SHA:
        raise ValueError("Frozen style/reference contract changed")
    native = json.loads((OUT / "native_ground_comparison.json").read_text())
    if native["status"] != "passed":
        raise ValueError("Native ground admission failed")
    idle_source = ROOT / "docs/style_improvement_20261002/summarize_navigation_style.py"
    idle_module = load("alpha025_frozen_idle_module", idle_source)
    # Preserve the complete frozen function except bounded presentation times.
    tree = ast.parse(idle_source.read_text())
    function = deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "stagnation"))
    snapshots = [node for node in ast.walk(function) if isinstance(node, ast.For) and
                 isinstance(node.target, ast.Name) and node.target.id == "seconds"]
    if len(snapshots) != 1 or ast.literal_eval(snapshots[0].iter) != (90, 105, 120):
        raise ValueError("Optional diagnostic timestamp loop changed")
    snapshots[0].iter = ast.parse("(10, 20, 30)", mode="eval").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(__file__), "exec"), idle_module.__dict__)
    articulation = load("alpha025_frozen_articulation_module",
        ROOT / "docs/style_improvement_20261002/mixed_curriculum_navigation_style_v1/summarize.py")
    rows, idle, observations = {}, {}, {}
    for name, item in evaluation["runs"].items():
        directory = Path(item["sources"]["result"]["path"]).parent
        result = json.loads((directory / "result.json").read_text())
        ground = native["baseline" if name == "baseline_pure30" else "candidate"]
        idle[name] = idle_module.stagnation(directory, ground)
        observations[name] = articulation.observed_articulation(directory)
        inclusive = item["inclusive_speed_conditioned"]
        rows[name] = {
            "capture_controls": 900, "capture_actor_frames": 3600, "duration_s": 30,
            "same_episode_intervals": item["generated"]["same_episode_transition_count"],
            "valid_acceleration_triplets": item["generated"]["valid_acceleration_triplets"],
            "falls": result["fall_resets"], "time_limits": result["time_limit_resets"],
            "route_vertices_reached": result["route_vertices_reached"],
            "total_vertices": sum(result["route_vertices_reached"]),
            "inclusive_pose_velocity_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
            "speed_distribution_W1_mps": inclusive["physical_speed_distribution_wasserstein_m_s"],
            "physical_speed_mean_mps": item["generated"]["physical_root_speed_m_s"]["mean"],
            "upright_retained_fraction": item["upright_subset"]["generated_fraction_retained"],
            "upright_secondary_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "idle_actor_seconds": sum(actor["idle_duration_s"] for actor in idle[name]["per_actor"]),
            "idle_interval_fraction": sum(actor["idle_intervals"] for actor in idle[name]["per_actor"]) / item["generated"]["same_episode_transition_count"],
            "support_statistics": ground["support_statistics"],
            "supported_foot_origin_tangential_speed_proxy": ground["body_origin_motion_proxy"]["aggregate_supported_mps"],
            "COM_statistics": ground["COM_statistics"],
            "obstacle_peer_contacts": ground["original_obstacle_peer_channels"],
            "toe_origin_near_ground_proxy": item["generated"]["toe_slide_near_ground_proxy"],
            "speed_bins": [{"lower_mps": entry["lower_m_s"], "upper_mps": entry["upper_m_s"],
                            "generated_count": entry["generated_count"], "reference_count": entry["reference_count"],
                            "mmd2": None if entry["pose_velocity_gaussian_mmd2"] is None else entry["pose_velocity_gaussian_mmd2"]["value"]}
                           for entry in inclusive["speed_bins"]],
        }
    write(OUT / "stagnation.json", {"status": "passed", "runs": idle,
        "frozen_idle_threshold_and_plan_phase_checks_changed": False,
        "optional_snapshot_times_only_changed_seconds": [10, 20, 30], "original_snapshot_times_seconds": [90, 105, 120]})
    write(OUT / "descriptive_yaw_wrist.json", {"status": "passed", "runs": observations,
        "primary_style_formula_changed": False, "no_paired_reference_rotation_error": True})
    report = {"schema": "arm_style.actual_paired30_style.v1", "status": "passed", "CPU_only": True,
        "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
        "outcomes": rows, "sources": {"evaluation": record(OUT / "frozen_style/evaluation.json"),
        "native_measurement": record(OUT / "native_ground_comparison.json"), "heldout_reference": record(reference),
        "frozen_implementations": {key: record(ROOT / key) for key in FROZEN}},
        "all900_controls_and_failures_retained": True, "comparison_with200s_MMD_as_same_window": False,
        "default_controller_promoted": False, "author_effect_reproduced": False,
        "limitations": native["limits"] + ["MMD .1-.5m/s worsens despite improved unconditional speedW1; no universal style improvement.",
            "Only optional idle diagnostic timestamps are bounded to30s; thresholds/phase/score formulas unchanged.",
            "One seed/short trajectory, no population uncertainty or paper threshold.",
            "Yaw and wrists are actual descriptive features, not paired body-orientation ground-truth errors."]}
    write(OUT / "comparison.json", report)
    for origin, name in (("/tmp/arm_alpha025_style_actual30s_v1.log", "style.log"),
                         ("/tmp/arm_alpha025_native_actual30s_v1.log", "native.log")):
        with (OUT / name).open("xb") as stream:
            stream.write(Path(origin).read_bytes())
    old, new = rows["baseline_pure30"], rows["arm_alpha025_30"]
    lines = ["# 实际30秒 alpha=.25 手臂混合：同长度冻结对照", "",
        "两份900步/30秒/4人全部保留。CPU评分/原生采集完整性通过；短时效果不替代200秒稳定性，不提升默认控制器。", "",
        "|指标|pure baseline|alpha=.25|", "|---|---:|---:|"]
    for label, key in (("fall", "falls"), ("路线顶点", "total_vertices"),
        ("inclusive MMD²", "inclusive_pose_velocity_mmd2"), ("速度W1 m/s", "speed_distribution_W1_mps"),
        ("实际差分速度mean m/s", "physical_speed_mean_mps"), ("idle总actor秒 / 120秒", "idle_actor_seconds")):
        lines.append(f"|{label}|{old[key]:.9g}|{new[key]:.9g}|")
    lines += ["", "主MMD略退，.1–.5m/s bin .032663878→.107676361变差；<.1m/s bin .160563718→.117759450改善。unsupported bins保持unknown。速度分布接近参考、路线2→3不等于风格整体改善。", "",
        "每份3600 actor-frame保留；合法same-episode导数3596、加速度triplet3592，没有fall/reset帧过滤。c274参考与82ee/b09原90D/sigma1/速度bins/min30/max256不改。",
        "", "## 原生支持、接触与足部代理", "",
        "原测量auditor函数未改，alpha真实231检查通过，baseline复用此前实际900步验收并重新绑定result/NPZ完整SHA。均无静态/peer接触；这只是此次30秒，不是普遍collision-free。", "",
        f"双支持 {old['support_statistics']['double_support_fraction']:.6%}→{new['support_statistics']['double_support_fraction']:.6%}；无足支持 {old['support_statistics']['no_foot_support_fraction']:.6%}→{new['support_statistics']['no_foot_support_fraction']:.6%}。支持条件足部原点切向速度mean {old['supported_foot_origin_tangential_speed_proxy']['mean']:.6f}→{new['supported_foot_origin_tangential_speed_proxy']['mean']:.6f}m/s。它包含滚动/旋转，不能当contact-point slip。COM真实重构最大误差约1e-6米；AMASS真实COM/contact参考未知。", "",
        "## 实验与来源边界", "",
        "baseline62c4…与donor28aed…，仅24个Thorax/Shoulder/Elbow/Wrist归一化raw mean坐标按alpha混合，再统一clamp/57mask，单次物理PD解码。未执行critic/D/训练transition。跨不同观测下不能声称其余raw坐标逐位相同。", "",
        "源146 baseline与源154 alpha采用各自实际预启动archive，完整控制For AST相同但world/contact/simulator/可选native等文件不同。native branch未启用；不拿后来的provider编辑或current tree替代历史，不宣称完整运行时GPU import map。", "",
        "停留定义和plan phase沿用原函数；只将可选90/105/120秒诊断snapshot改为10/20/30秒，避免越界，不改指标。yaw/wrist仅描述性，没有逐帧身体旋转真值。", "",
        "证据：[比较JSON](comparison.json)、[冻结原评分](frozen_style/evaluation.json)、[原生验收](native_ground_comparison.json)、[停留/phase](stagnation.json)、[yaw/wrist](descriptive_yaw_wrist.json)、[参考](frozen_style/heldout_reference.json)。CPU-only；未读取sealed34、改业务/weights或启动Kit/GPU。"]
    with (OUT / "README.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "outcomes": {k: {n: v for n, v in row.items() if n in
        {"falls", "total_vertices", "inclusive_pose_velocity_mmd2", "speed_distribution_W1_mps", "idle_actor_seconds"}}
        for k, row in rows.items()}}))


if __name__ == "__main__":
    run()
