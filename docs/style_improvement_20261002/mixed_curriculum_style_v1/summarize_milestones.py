"""Append three completed frozen validation reports without metric recomputation.

Only JSON and small evidence files are read. No Torch, SDK, training data bank,
sealed test capture, checkpoint restore, or new simulator is used.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "amass_root_style_v3"
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
TOOL_SHA = "d19c9cea62839792c85fccc92cdefd5bf9a3f1d27c3c39c7b0b7f6d0d4783612"


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compact(row):
    diagnostic = row["same_frame_decomposition"]
    return {"aggregate": row["aggregate"], "inclusive_mmd2": row["inclusive_mmd2"],
            "upright_mmd2": row["upright_mmd2"], "upright_fraction": row["upright_fraction"],
            "speed_wasserstein_m_s": row["speed_wasserstein_m_s"], "speed_bins": row["speed_bins"],
            "body_origin_toe_slide_proxy_fraction": row["toe_near_ground_body_origin_proxy"]["sliding_above_threshold_fraction"],
            "mean_same_frame": {key: diagnostic[key]["mean"] for key in (
                "root_xy_drift_m", "root_yaw_abs_wrapped_error_rad", "actual_planar_speed_m_s",
                "frame_mean_rootcenter_heading_body_position_error_m",
                "frame_mean_parent_relative_joint_orientation_error_rad")},
            "wrist_parent_rotation_mean_rad": {body["body"]: body["parent_relative_joint_orientation_error_rad"]["mean"]
                for body in row["wrists_and_hands"] if body["body"] in ("L_Wrist", "R_Wrist")}}


def main():
    prior, prior_evaluation = load(PRIOR / "comparison.json"), load(PRIOR / "evaluation.json")
    require(len(prior["runs"]) == len(prior_evaluation["runs"]) == 9, "Expected frozen nine-model predecessor")
    require(record(HERE / "compare.py")["sha256"] == TOOL_SHA, "Frozen metric invocation tool changed")
    combined, evaluated = deepcopy(prior["runs"]), deepcopy(prior_evaluation)
    upstream, milestones, outcomes = {}, {}, {}
    for step in (500, 1000, 2000):
        name, directory = f"mixed{step}", HERE / f"step{step}_comparison_v1"
        comparison, evaluation, delivery = (load(directory / filename) for filename in (
            "comparison.json", "evaluation.json", "delivery.json"))
        require(comparison["status"] == delivery["status"] == "passed", "Upstream CPU gate failed")
        require(set(comparison["runs"]) == set(prior["runs"]) | {name}, "Unexpected milestone identities")
        require(all(comparison["runs"][key] == row for key, row in prior["runs"].items()), "Historical comparison changed")
        require(all(evaluation["runs"][key] == row for key, row in prior_evaluation["runs"].items()), "Historical evaluation changed")
        require(evaluation["reference"] == prior_evaluation["reference"]
                and evaluation["sources"] == prior_evaluation["sources"], "Reference or metric implementation changed")
        require(record(directory / "heldout_reference.json")["sha256"] == REFERENCE_SHA, "Reference bytes changed")
        pairs = comparison["same_initial_reference_pairs_vs_negative_baseline"][name]
        require(len(pairs) == 22 and all(all(row[group].values()) for row in pairs for group in (
            "reference_and_clock_arrays_exact", "initial_physical_state_and_amp_history_exact")), "Paired reference or initial state differs")
        contract = comparison["mixed_checkpoint_contracts"][name]
        require(contract["updates"] == step and contract["physical_transitions"] == step * 256 * 32, "Milestone budget differs")
        row = comparison["runs"][name]
        actual = load(ROOT / row["result"]["path"])
        require(record(row["result"]["path"])["sha256"] == row["result"]["sha256"], "Capture result changed")
        require(actual["status"] == "passed" and actual["arguments"]["split"] == "validation", "Only terminal validation capture admitted")
        require(len(actual["sequences"]) == 22 and actual["aggregate"] == row["aggregate"], "Incomplete sequence set or changed metric aggregate")
        require(actual["aggregate"]["observed_metric_steps"] == 11982, "Capture frame count differs")
        combined[name], evaluated["runs"][name] = row, evaluation["runs"][name]
        upstream[name] = {filename: record(directory / filename) for filename in (
            "comparison.json", "evaluation.json", "delivery.json")}
        milestones[name] = {**compact(row), "checkpoint": contract["checkpoint"], "updates": step,
            "physical_transitions": contract["physical_transitions"], "actual_group_transitions": contract["actual_group_transitions"],
            "training_statistics": contract["curriculum_statistics"]}
        outcomes[name] = {sequence["motion_id"]: {key: sequence[key] for key in (
            "source_path", "tracking_success", "sequence_had_fall", "observed_metric_steps",
            "mean_body_position_error_m", "mean_body_rotation_error_rad")}
            for sequence in actual["sequences"]}
    require(all(set(rows) == set(outcomes["mixed500"]) for rows in outcomes.values()), "Validation identities differ")
    require(all(combined[key] == row for key, row in prior["runs"].items()), "Combined historical dictionary changed")
    now = datetime.now(timezone.utc).isoformat()
    output = HERE / "milestone_comparison_v1"
    output.mkdir(exist_ok=False)
    evaluated["generated_at_utc"] = now
    evaluated["append_provenance"] = {"upstream_frozen_actual_reports": upstream, "old_nine_dicts_exact": True,
                                    "new_metrics_recomputed_by_aggregation": False}
    write_new(output / "evaluation.json", evaluated)
    write_new(output / "comparison.json", {"schema": "mixed_curriculum_three_milestone_comparison.v1",
        "status": "passed", "generated_at_utc": now, "runs": combined, "upstream_actual_reports": upstream,
        "prior_nine_comparison_dicts_exact": True, "prior_nine_evaluation_dicts_exact": True,
        "metric_kernel_reference_parameters_changed": False, "metrics_recomputed_by_aggregation": False})
    shutil.copyfile(PRIOR / "heldout_reference.json", output / "heldout_reference.json")
    sequence_rows = [{"motion_id": mid, "source_path": outcomes["mixed500"][mid]["source_path"],
        "milestones": {name: rows[mid] for name, rows in outcomes.items()}} for mid in sorted(outcomes["mixed500"])]
    failures = {name: [mid for mid, row in rows.items() if not row["tracking_success"]] for name, rows in outcomes.items()}
    summary = {"schema": "mixed_curriculum_validation_selection_evidence.v1", "status": "passed",
        "created_utc": now, "milestones": milestones, "reference_baselines": {name: compact(prior["runs"][name])
            for name in ("backward_penalty1000", "amass_root1000")}, "by_sequence": sequence_rows,
        "failed_motion_ids": failures, "same_initial_reference_clock_all22_exact": True,
        "prior_nine_comparison_dicts_exact": True, "prior_nine_evaluation_dicts_exact": True,
        "frozen_reference": record(output / "heldout_reference.json"), "frozen_metric_invocation_tool": record(HERE / "compare.py"),
        "CPU_aggregation_tool": record(__file__), "upstream_actual_reports": upstream,
        "selected_navigation_checkpoint": None, "selection_status": "evidence_only_root_selection_not_performed",
        "tradeoff": "mixed1000 has lower primary inclusive MMD2; mixed2000 has lower rotation/pose-relative errors and speed Wasserstein; both19/22 zero falls, with different failed identities",
        "GPU_used": False, "Kit_started": False, "sealed34_test_outputs_read": False,
        "limitations": ["Repeatedly used validation, not blind test or independent-seed significance",
            "Unequal update budgets, correlated milestones of one fresh training run",
            "True future-XY tracking is not learned Diffusion closed-loop navigation",
            "Toe body-origin proxy is not native sole/contact or true COM evidence",
            "Execution status passed does not mean paper quality passed"]}
    write_new(output / "summary.json", summary)
    print(json.dumps({"status": "passed", "output": str(output), "old_nine_exact": True,
                      "merged_models": len(combined), "navigation_checkpoint_selected": False}))


if __name__ == "__main__":
    main()
