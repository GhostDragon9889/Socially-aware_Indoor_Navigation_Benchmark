"""Seal the already selected final34 test; CPU evidence only, no model selection."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path

import numpy as np

import audit_amass_root_tracking as audit


SELECTION_SHA = "429bbe40425d43736a1d62434af6cd7cdb0fa7ce4b0e5be3a396056e9a8861b9"
CANDIDATE_SHA = "dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e"


def full_test(result):
    audit.require(result["status"] == "passed" and result["arguments"]["split"] == "test", "Completed test required")
    audit.require(all(result["arguments"][key] is None for key in ("motion_ids", "max_clips", "max_steps")), "No caps/subsets")
    rows = result["sequences"]
    audit.require(len(rows) == 34 and len({row["motion_id"] for row in rows}) == 34, "All34 unique test clips required")
    audit.require(all(row["complete_clip"] and row["in_full_sequence_success_denominator"]
                      and row["observed_metric_steps"] == row["expected_control_steps"] for row in rows), "Full real clip denominators required")
    return {row["motion_id"]: row for row in rows}


def independent_sequence(capture, row, threshold):
    measured = capture["metric_frame"]
    error = np.linalg.norm(capture["rigid_body_pos"].astype(np.float64)
                           - capture["reference_rigid_body_pos"].astype(np.float64), axis=-1).mean((-1, -2))
    failed = np.flatnonzero(measured & (error > threshold))
    audit.require(bool(len(failed)) == row["tracking_failed"], "Failure threshold disagrees with raw body positions")
    audit.require((int(failed[0]) if len(failed) else None) == row["first_tracking_failure_capture_frame"], "First failure frame differs")
    audit.require(row["tracking_success"] == (row["complete_clip"] and not len(failed)), "Success rule differs")
    return {"failure_rule_recomputed": True, "metric_frames": int(measured.sum()),
            "failure_capture_frames": failed.tolist(), "falls": audit.fall_intervals(capture, row)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only environment required")
    audit.require(not args.output.exists(), "Use a fresh evidence directory")
    admission_path = audit.DOCS / "amass_root_final34_pretest_admission.json"
    admission = audit.read(admission_path)
    selection_path = Path(admission["selection"]["path"])
    selection = audit.read(selection_path)
    audit.require(audit.sha256(selection_path) == SELECTION_SHA == admission["selection"]["sha256"], "Pre-test selection changed")
    audit.require(admission["selected_checkpoint_sha256"] == CANDIDATE_SHA
                  and selection["sealed_full34_test_read_for_selection"] is False
                  and datetime.fromisoformat(admission["admission_utc"]) > datetime.fromisoformat(admission["selection_utc"]), "Selection must precede test admission")
    paths = {"baseline": audit.RUNS / "motion_tracking_amass2848_backward_penalty_full_test_v1/result.json",
             "amass_root": audit.RUNS / "motion_tracking_amass_root_curriculum_full_test_v1/result.json"}
    results = {name: audit.read(path) for name, path in paths.items()}
    rows = {name: full_test(value) for name, value in results.items()}
    audit.require(rows["baseline"].keys() == rows["amass_root"].keys(), "Test IDs differ")
    audit.require(results["amass_root"]["arguments"]["checkpoint_sha256"] == CANDIDATE_SHA, "Candidate differs from sealed pre-test selection")
    flag_diff = audit.differences(results["baseline"]["arguments"], results["amass_root"]["arguments"])
    audit.require(set(flag_diff) <= {"checkpoint", "checkpoint_sha256", "output"}, "Evaluation flags changed")
    for key in ("protocol", "physics", "layout", "reference_velocity_contract", "runtime_numerics", "training_config"):
        audit.require(results["baseline"][key] == results["amass_root"][key], f"Paired evaluation contract differs:{key}")
    checkpoints = {name: audit.verified_checkpoint(value) for name, value in results.items()}
    sources = {"baseline": audit.DOCS / "backward_penalty_training_source.json",
               "amass_root": audit.DOCS / "amass_root_linear_training_source_v1.json"}
    archives, entries, runtime_binding = {}, {}, {}
    for name, path in sources.items():
        archives[name], entries[name] = audit.verify_archive(path)
        runtime_binding[name] = {}
        for relative, digest in results[name]["selected_source_sha256"].items():
            matched = relative in entries[name] and entries[name][relative]["sha256"] == digest
            if not matched:
                # This older test ran after its training archive. Preserve the
                # exact missing-byte limitation rather than label the archive
                # as the evaluation source or invent reconstructed source.
                audit.require(name == "baseline" and relative == "CrowdSim/humanoid/amp_world.py"
                              and digest == "ba10eeb4c15277ab8b44f243844973930caf6448ab086d6e2fd93461b7a8ca29",
                              "Unexpected captured source mismatch")
            runtime_binding[name][relative] = {"captured_sha256": digest,
                "training_archive_sha256": entries[name].get(relative, {}).get("sha256"),
                "runtime_bytes_bound_to_training_archive": matched,
                "runtime_source_bytes_available": matched}
    spec = importlib.util.spec_from_file_location("frozen_tracking_gate", audit.ROOT / "scripts/benchmark_paper_motion_tracking.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    import torch
    checkpoint_gates = {}
    for name, result in results.items():
        saved = torch.load(checkpoints[name]["path"], weights_only=True, map_location="cpu", mmap=True)
        audit.require(saved["update_count"] == 1000 and saved["physical_transition_count"] == 8192000, "Actual saved training budget differs")
        audit.require(saved["provenance"] == result["controller_provenance"], "Captured actor provenance differs from actual checkpoint")
        gate = benchmark.verify_actor_training_split(saved["provenance"], result["arguments"]["manifest"], "test")
        audit.require(gate == result["actor_training_heldout_parent_gate"], "Actual normalized parent gate differs")
        checkpoint_gates[name] = {"actual_saved_updates": saved["update_count"],
                                  "actual_saved_transitions": saved["physical_transition_count"], "parent_gate": gate}
        del saved
    decomposition = audit.helper()
    decomposed = {name: decomposition.decompose(path.parent) for name, path in paths.items()}
    parents = decomposition.parent_indices(audit.ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    pairs = []
    for mid, baseline in rows["baseline"].items():
        candidate = rows["amass_root"][mid]
        audit.require(all(baseline[key] == candidate[key] for key in ("source_path", "tracking_window", "observed_metric_steps", "initial_translation_xyz_m")), "Clip/source initialization differs")
        with np.load(baseline["capture"], allow_pickle=False) as old, np.load(candidate["capture"], allow_pickle=False) as new:
            exact = audit.exact_capture_pair(old, new)
            independent = {name: independent_sequence(data, row, results[name]["arguments"]["failure_threshold_m"])
                           for name, data, row in (("baseline", old, baseline), ("amass_root", new, candidate))}
        pairs.append({"motion_id": mid, **exact, "independent_raw_checks": independent,
                      "fall_diagnostics": {name: audit.capture_diagnostics(row["capture"], row, decomposition, parents)
                                           for name, row in (("baseline", baseline), ("amass_root", candidate))}})
    success = {name: {mid for mid, row in values.items() if row["tracking_success"]} for name, values in rows.items()}
    output = {"schema": "amass_root.once_only_final34_pair.v1", "status": "passed", "CPU_only": True,
              "GPU_used": False, "Kit_started": False, "style_metrics_recomputed": False,
              "selection": audit.record(selection_path), "pretest_admission": audit.record(admission_path),
              "selected_checkpoint_sha256": CANDIDATE_SHA, "test_outcomes_used_to_select_or_tune": False,
              "script": audit.record(__file__), "results": {name: audit.record(path) for name, path in paths.items()},
              "checkpoints": checkpoints, "checkpoint_training_and_parent_gates": checkpoint_gates,
              "source_archives": archives, "paired_clip_count": len(pairs), "paired_clips": pairs,
              "runtime_source_binding": runtime_binding,
              "all_historical_runtime_source_bytes_verified": False,
              "evaluation_argument_differences": flag_diff,
              "aggregate": {name: value["aggregate"] for name, value in results.items()},
              "same_frame_diagnostics": {name: value["aggregate_diagnostic"] for name, value in decomposed.items()},
              "secondary_success_subsets": {"each_run_success_only": {name: audit.subset_primary(values, success[name]) for name, values in rows.items()},
                    "common_success_intersection": {name: audit.subset_primary(values, success["baseline"] & success["amass_root"]) for name, values in rows.items()}},
              "limits": ["All34 clips/all7739 physical metric frames retained, including fall frames; no time fitting.",
                         "Secondary successful subsets are descriptive and do not replace all34 metrics.",
                         "One fixed seed and local manifest; not the unpublished author population or paper metric equivalence.",
                         "AMASS XY tracking is separate from learned Diffusion deployment navigation.",
                         "No test-informed recipe/hyperparameter changes; next curriculum was motivated by validation/navigation before formal test."]}
    output["limits"].append("Baseline test amp_world.py captured ba10eeb4 differs from training archive70914a2; exact ba10 bytes are not preserved in available before/tar archives. Five other baseline runtime sources and all six candidate sources bind; raw pairing does not prove omitted source-byte identity.")
    args.output.mkdir(parents=True)
    for name, value in decomposed.items():
        audit.write_new(args.output / f"{name}_decomposition.json", value)
    audit.write_new(args.output / "comparison.json", output)
    lines = ["# 封存的34段最终测试配对", "", "先固定候选，再读取test；本审计仅CPU重算既有捕获，不调参、不改变检查点。", "",
             "| 完整34段 /7739帧 | negative1000 | AMASS-root1000 |", "| --- | --- | --- |"]
    for key in ("tracking_successes", "sequences_with_fall", "frame_weighted_mean_body_position_error_m", "frame_weighted_mean_body_rotation_error_rad"):
        lines.append(f"| {key} | {results['baseline']['aggregate'][key]:.9g} | {results['amass_root']['aggregate'][key]:.9g} |")
    lines += ["", "全部34初始物理状态/AMP history以及参考目标/phase/time通过dtype+bitwise配对；原始位置错误阈值、跌倒连续时段、全文主指标已独立重算。",
              "实际权重内1000updates/8.192M与完整parent-disjoint门禁通过；training源码tar逐member核验。baseline test的world ba10eeb4与训练70914a2不同，现有before/tar无对应bytes，因此该单文件完整来源绑定标未知；另5个baseline与6个candidate捕获source均匹配。", "",
              "本结果支持这一固定候选在本地test的root跟踪位置和成功数改善，旋转均值略差；不证明style质量、200秒导航或作者64.6%协议等价。",
              "单seed、固定本地样本，不以success-only子集替代完整分母。部署导航失败的历史结果继续保留，未提升默认模型。", "",
              f"预选SHA `{SELECTION_SHA}`；candidate SHA `{CANDIDATE_SHA}`。", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print({"status": "passed", "paired_clips": len(pairs), "aggregate": output["aggregate"]})


if __name__ == "__main__":
    main()
