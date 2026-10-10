"""Append the genuine500-resumed750 experiment to the frozen eight-model review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from summarize_stratified_tracking import (
    DOCS,
    REFERENCE_SHA,
    ROOT,
    exact_initial_and_reference_pairs,
    load,
    record,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    args = parser.parse_args()
    here = args.evaluation.resolve().parent
    style = load(args.evaluation)
    prior = load(DOCS / "amass_root_style_v2/evaluation.json")
    prior_comparison = load(DOCS / "amass_root_style_v2/comparison.json")
    reference = record(here / "heldout_reference.json")
    assert style["status"] == "passed" and style["device"] == "CPU" and not style["kit_started"]
    assert reference["sha256"] == REFERENCE_SHA and style["sources"] == prior["sources"]
    old_equality = {name: item == style["runs"][name] for name, item in prior["runs"].items()}
    assert all(old_equality.values()) and set(style["runs"]) - set(prior["runs"]) == {"amass_root750"}
    runs = dict(prior_comparison["runs"])
    item = style["runs"]["amass_root750"]
    result_source = item["sources"]["result"]
    assert record(result_source["path"]) == result_source
    result = load(result_source["path"])
    decomposition = load(args.decomposition)
    assert decomposition["status"] == "passed" and decomposition["analytic_rigid_transform_check_passed"]
    assert decomposition["primary_aggregate_unchanged"] == result["aggregate"]
    assert decomposition["source_result"]["sha256"] == result_source["sha256"]
    inclusive = item["inclusive_speed_conditioned"]
    runs["amass_root750"] = {
        "result": result_source,
        "checkpoint": item["sources"]["models_resolved_from_evaluation_arguments"]["controller"],
        "aggregate": result["aggregate"],
        "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
        "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
        "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
        "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"],
        "speed_bins": [
            {key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
            | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
            for row in inclusive["speed_bins"]
        ],
        "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
        "mass_weighted_joint_origin_height_proxy_m": item["generated"]["metrics"]["mass_weighted_joint_origin_height_proxy_m"],
        "capture_validation": {key: item["capture_validation"][key] for key in (
            "metric_frames", "clips", "initial_reset_frames_discarded", "valid_same_clip_style_intervals", "fall_metric_frames_kept")},
        "decomposition": record(args.decomposition),
        "same_frame_decomposition": decomposition["aggregate_diagnostic"],
        "wrists_and_hands": [row for row in decomposition["by_body"] if row["body"] in ("L_Wrist", "R_Wrist", "L_Hand", "R_Hand")],
        "by_reference_speed_bin": decomposition["by_reference_speed_bin"],
        "heldout_parent_disjointness": item["reference_not_in_declared_training"],
    }
    pairs, changes = {}, {}
    for name in ("backward_penalty1000", "amass_root500", "amass_root1000"):
        baseline = load(runs[name]["result"]["path"])
        assert all(baseline[key] == result[key] for key in ("protocol", "physics", "layout", "reference_velocity_contract"))
        pairs[name] = exact_initial_and_reference_pairs(baseline, result)
        previous_rows = {row["motion_id"]: row for row in baseline["sequences"]}
        previous_success = {row["motion_id"] for row in baseline["sequences"] if row["tracking_success"]}
        new_success = {row["motion_id"] for row in result["sequences"] if row["tracking_success"]}
        changes[name] = {
            "gained_success_motion_ids": sorted(new_success - previous_success),
            "lost_success_motion_ids": sorted(previous_success - new_success),
            "clips_with_position_error_improved": sum(row["mean_body_position_error_m"] < previous_rows[row["motion_id"]]["mean_body_position_error_m"] for row in result["sequences"]),
            "clips_with_rotation_error_improved": sum(row["mean_body_rotation_error_rad"] < previous_rows[row["motion_id"]]["mean_body_rotation_error_rad"] for row in result["sequences"]),
            "fall_sources": [row["source_path"] for row in result["sequences"] if row["sequence_had_fall"]],
        }
    train_path = ROOT / "output/diffusion_amp/amp_style_amass_root_from500_to750_v1/result.json"
    train = load(train_path)
    assert train["status"] == "passed" and train["update_count"] == 750 and train["physical_transition_count"] == 6144000
    assert train["arguments"]["updates"] == 250 and train["arguments"]["num_envs"] == 256
    assert train["arguments"]["rollout_steps"] == 32 and train["actual_control_steps"] == 8000
    source500 = record(ROOT / train["arguments"]["resume"])
    assert source500["sha256"] == "8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314"
    original = load(ROOT / "output/diffusion_amp/amp_style_amass_root_curriculum_v1/result.json")
    assert train["training_config"] == original["training_config"]
    summary = {
        "schema": "navisaaclab.amass_root_tracking_style_review.v3",
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_only": True,
        "Kit_started": False,
        "business_data_profiles_and_evaluator_modified": False,
        "style_evaluation": record(args.evaluation),
        "frozen_reference": reference,
        "prior_eight_run_dicts_exactly_unchanged": old_equality,
        "prior_eight_comparison_run_dicts_exactly_preserved": {name: runs[name] == value for name, value in prior_comparison["runs"].items()},
        "frozen_implementations": style["sources"],
        "summary_script": record(__file__),
        "style_spec": style["reference"]["metric_spec"],
        "runs": runs,
        "real500_resumed750_branch": {
            "training_result": record(train_path),
            "source500": source500,
            "additional_updates": 250,
            "additional_physical_transitions": 2048000,
            "cumulative_updates": 750,
            "cumulative_physical_transitions": 6144000,
            "segment_only_training_fall_resets": train["fall_resets"],
            "segment_only_training_time_limit_resets": train["time_limit_resets"],
            "configuration_same_as_original1000": True,
            "qualification": "A genuine new500-resumed250-update experiment, not the original1000 run's missed750 checkpoint or bitwise continuous physics reproduction.",
        },
        "exact_initial_and_reference_pairs": pairs,
        "paired_clip_changes": changes,
        "limitations": prior_comparison["limitations"] + [
            "The750 branch was proposed after repeatedly inspecting22 validation/style tradeoffs; no sealed34 test captures were read.",
            "Selection, repeated validation and unequal budgets prevent a blind-test or multi-seed significance claim.",
        ],
    }
    with (here / "comparison.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "old8_exact": True, "new750": runs["amass_root750"]["aggregate"],
                      "inclusive_mmd2": runs["amass_root750"]["inclusive_mmd2"]}, indent=2))


if __name__ == "__main__":
    main()
