"""Bind frozen full22-reader output to terminal CPU admission, no SDK/model load."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def record(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.comparison_dir
    comparison_path = folder / "comparison.json"
    terminal_path = Path(__file__).parent / "mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json"
    comparison = json.loads(comparison_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    assert comparison["status"] == "passed" and comparison["CPU_only"] and comparison["paired_clip_count"] == 22
    assert terminal["status"] == "passed" and terminal["CPU_only"] and not terminal["GPU_used"] and not terminal["SDK_started"]
    assert terminal["check_count"] == 220 and len(terminal["checks"]) == 220 and all(row["passed"] for row in terminal["checks"])
    matches = [row for row in terminal["checkpoints"] if row["checkpoint"] == comparison["checkpoints"]["mixed"]]
    assert len(matches) == 1
    matched = matches[0]
    budget = comparison["saved_mixed_admission"]["budget"]
    assert matched["update_count"] == budget["updates"]
    assert matched["physical_transitions"] == budget["total_transitions"]
    assert matched["group_statistics"]["amass_actor_transitions"] == budget["amass_transitions"]
    assert matched["group_statistics"]["procedural_actor_transitions"] == budget["procedural_transitions"]
    assert all(row["reference_and_clock_fields_bitwise_equal"] and row["initial_physical_state_and_history_bitwise_equal"]
               for row in comparison["paired_clips"])
    assert all(run["observed_metric_steps"] == 11982 and run["measured_sequences"] == 22
               for run in comparison["aggregate"].values())
    binding = {"schema": "mixed_tracking_terminal_evidence_binding.v1", "status": "passed", "CPU_only": True,
        "script": record(Path(__file__)), "comparison": record(comparison_path), "terminal_independent_audit": record(terminal_path),
        "terminal_check_count": 220, "matching_terminal_checkpoint": matched["checkpoint"],
        "primary_payload_admission_retained": comparison["independent_payload_audit"],
        "primary_check_count": comparison["independent_payload_admission"]["independent_CPU_check_count"],
        "all22_reference_clock_initial_physical_state_history_bitwise_equal": True,
        "reference_clips": 22, "all_metric_frames": 11982,
        "limitations": ["Terminal execution counts do not substitute for strict raw full22 quality checks.",
            "Terminal result fall totals apply to step2000, not earlier checkpoint stages.",
            "No GPU/Kit, 34test, or official style recomputation."]}
    with (folder / "terminal_audit_binding.json").open("x") as stream:
        json.dump(binding, stream, indent=2)
        stream.write("\n")
    baseline, candidate = (comparison["aggregate"][name] for name in ("baseline", "mixed"))
    diagnostics = comparison["same_frame_diagnostics"]
    text = f"# step{budget['updates']} 误差分解补充\n\n"
    text += "完整22 / 11982帧主分母，reference/clock和初始physical/history全部dtype及bitwise配对通过。原独立payload门禁保持原样，另以terminal_audit_binding.json绑定终点220项审计中的同一checkpoint字节身份与分组实际转换。\n\n"
    text += "| 同帧描述性指标 | negative1000 | mixed |\n| --- | --- | --- |\n"
    for label, key in (("root XY drift (m)", "root_xy_drift_m"),
                       ("root wrapped yaw error (rad)", "root_yaw_abs_wrapped_error_rad"),
                       ("root-centered / heading-aligned body position (m)", "frame_mean_rootcenter_heading_body_position_error_m"),
                       ("heading-relative body rotation (rad)", "frame_mean_heading_body_orientation_error_rad"),
                       ("parent-relative joint rotation (rad)", "frame_mean_parent_relative_joint_orientation_error_rad"),
                       ("actual planar speed (m/s)", "actual_planar_speed_m_s")):
        text += f"| {label} | {diagnostics['baseline'][key]['mean']:.9g} | {diagnostics['mixed'][key]['mean']:.9g} |\n"
    position_change = (candidate["frame_weighted_mean_body_position_error_m"] / baseline["frame_weighted_mean_body_position_error_m"] - 1) * 100
    rotation_change = (candidate["frame_weighted_mean_body_rotation_error_rad"] / baseline["frame_weighted_mean_body_rotation_error_rad"] - 1) * 100
    text += f"\nALL22世界body位置误差变化 {position_change:.3f}%；世界body旋转误差变化 {rotation_change:.3f}%。分解仅描述性，不是独立因果贡献百分比。\n\n"
    subsets = comparison["secondary_success_subsets"]
    for name, subset in subsets["each_run_success_only"].items():
        text += f"{name} success-only：{subset['clips']}条 / {subset['metric_frames']}帧。\n\n"
    common = subsets["common_success_intersection"]["baseline"]
    text += f"common-success：相同{common['clips']}条 / {common['metric_frames']}帧。各自与共同子集的per-clip/frame两类均值保留在comparison JSON，不替代ALL22或未知论文分母。\n\n"
    text += f"累计{budget['updates']}updates，总{budget['total_transitions']}转换，每组{budget['amass_transitions']}；baseline总8.192M。equal_total_budget_to_baseline={budget['equal_total_budget_to_baseline']}；each_group_has_baseline_total_exposure={budget['each_group_has_baseline_total_exposure']}。\n\n"
    text += "本工具没有运行200秒导航、官方风格重算或读取34test，也没有改变部署默认、权重或业务源码。\n"
    with (folder / "decomposition_notes.md").open("x") as stream:
        stream.write(text)
    print({"status": "passed", "stage": budget["updates"], "terminal_check_count": 220,
           "position_change_percent": position_change, "rotation_change_percent": rotation_change})


if __name__ == "__main__":
    main()
