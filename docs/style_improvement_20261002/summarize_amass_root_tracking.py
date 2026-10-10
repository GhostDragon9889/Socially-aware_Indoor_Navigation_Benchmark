"""CPU-only comparison of completed AMASS-root tracking and frozen style metrics."""

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


def differences(left, right):
    return {
        key: {"baseline": left.get(key), "amass_root": right.get(key),
              "baseline_key_present": key in left, "amass_root_key_present": key in right}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key) or (key in left) != (key in right)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--decomposition", action="append", required=True)
    args = parser.parse_args()
    here = args.evaluation.resolve().parent
    style = load(args.evaluation)
    prior = load(DOCS / "stratified_experts_style_v2/evaluation.json")
    prior_comparison = load(DOCS / "stratified_experts_style_v2/comparison.json")
    reference = record(here / "heldout_reference.json")
    assert style["status"] == "passed" and style["device"] == "CPU" and not style["kit_started"]
    assert reference["sha256"] == REFERENCE_SHA
    assert style["sources"] == prior["sources"], "Frozen evaluator source changed"
    prior_equality = {name: value == style["runs"][name] for name, value in prior["runs"].items()}
    assert all(prior_equality.values()), "Historical six run dictionaries changed"
    new_names = set(style["runs"]) - set(prior["runs"])
    assert new_names and new_names <= {"amass_root1000", "amass_root500"}
    decomposition_paths = {}
    for value in args.decomposition:
        name, separator, path = value.partition("=")
        assert separator and name and name not in decomposition_paths
        decomposition_paths[name] = Path(path)
    assert set(decomposition_paths) == new_names
    runs = dict(prior_comparison["runs"])
    baseline = load(prior_comparison["runs"]["backward_penalty1000"]["result"]["path"])
    paired = {}
    changes = {}
    baseline_rows = {row["motion_id"]: row for row in baseline["sequences"]}
    baseline_success = {row["motion_id"] for row in baseline["sequences"] if row["tracking_success"]}
    for name in style["runs"]:
        if name not in new_names:
            continue
        item = style["runs"][name]
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        decomposition = load(decomposition_paths[name])
        assert decomposition["status"] == "passed" and decomposition["analytic_rigid_transform_check_passed"]
        assert decomposition["primary_aggregate_unchanged"] == result["aggregate"]
        assert decomposition["source_result"]["sha256"] == result_source["sha256"]
        assert all(baseline[key] == result[key] for key in ("protocol", "physics", "layout", "reference_velocity_contract"))
        paired[name] = exact_initial_and_reference_pairs(baseline, result)
        inclusive = item["inclusive_speed_conditioned"]
        bins = [
            {key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
            | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
            for row in inclusive["speed_bins"]
        ]
        runs[name] = {
            "result": result_source,
            "checkpoint": item["sources"]["models_resolved_from_evaluation_arguments"]["controller"],
            "aggregate": result["aggregate"],
            "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
            "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"],
            "speed_bins": bins,
            "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
            "mass_weighted_joint_origin_height_proxy_m": item["generated"]["metrics"]["mass_weighted_joint_origin_height_proxy_m"],
            "capture_validation": {key: item["capture_validation"][key] for key in (
                "metric_frames", "clips", "initial_reset_frames_discarded", "valid_same_clip_style_intervals", "fall_metric_frames_kept")},
            "decomposition": record(decomposition_paths[name]),
            "same_frame_decomposition": decomposition["aggregate_diagnostic"],
            "wrists_and_hands": [row for row in decomposition["by_body"] if row["body"] in ("L_Wrist", "R_Wrist", "L_Hand", "R_Hand")],
            "by_reference_speed_bin": decomposition["by_reference_speed_bin"],
            "heldout_parent_disjointness": item["reference_not_in_declared_training"],
        }
        success = {row["motion_id"] for row in result["sequences"] if row["tracking_success"]}
        changes[name] = {
            "gained_success_motion_ids": sorted(success - baseline_success),
            "lost_success_motion_ids": sorted(baseline_success - success),
            "clips_with_position_error_improved": sum(row["mean_body_position_error_m"] < baseline_rows[row["motion_id"]]["mean_body_position_error_m"] for row in result["sequences"]),
            "clips_with_rotation_error_improved": sum(row["mean_body_rotation_error_rad"] < baseline_rows[row["motion_id"]]["mean_body_rotation_error_rad"] for row in result["sequences"]),
            "fall_sources": [row["source_path"] for row in result["sequences"] if row["sequence_had_fall"]],
        }
    old_train_path = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/result.json"
    new_train_path = ROOT / "output/diffusion_amp/amp_style_amass_root_curriculum_v1/result.json"
    old_train, new_train = load(old_train_path), load(new_train_path)
    old_config, new_config = old_train["training_config"], new_train["training_config"]
    # Older disabled checkpoints omit this later optional field; preserve raw
    # evidence and compare only its documented disabled default explicitly.
    assert old_config.get("expert_sampling") is None and new_config.get("expert_sampling") is None
    assert dict(old_config, expert_sampling=None) == dict(new_config, expert_sampling=None)
    comparable_training_keys = (
        "physical_transition_count", "actual_control_steps", "update_count", "initialization",
        "runtime_numerics", "physics", "runtime_execution", "power_measurement",
    )
    training_equality = {key: old_train[key] == new_train[key] for key in comparable_training_keys}
    assert all(training_equality.values())
    summary = {
        "schema": "navisaaclab.amass_root_tracking_style_review.v1",
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_only": True,
        "Kit_started": False,
        "business_data_profiles_and_evaluator_modified": False,
        "style_evaluation": record(args.evaluation),
        "frozen_reference": reference,
        "prior_six_run_dicts_exactly_unchanged": prior_equality,
        "prior_six_comparison_run_dicts_exactly_preserved": {name: runs[name] == value for name, value in prior_comparison["runs"].items()},
        "frozen_implementations": style["sources"],
        "summary_script": record(__file__),
        "style_spec": style["reference"]["metric_spec"],
        "runs": runs,
        "AMASS_root_training_intervention": {
            "training_results": {"baseline": record(old_train_path), "amass_root": record(new_train_path)},
            "training_config_raw_exact": old_config == new_config,
            "training_config_raw_differences": differences(old_config, new_config),
            "training_config_legacy_optional_disabled_sampler_default_equivalent": True,
            "equal_training_setup_fields": training_equality,
            "training_cli_differences": differences(old_train["arguments"], new_train["arguments"]),
            "curriculum_changes": {"baseline": old_train["curriculum"], "amass_root": new_train["curriculum"]},
            "root_command_contract": new_train["amass_root_command_contract"],
            "full1000_physical_transitions_each": 8192000,
            "fall_resets": {"baseline": old_train["fall_resets"], "amass_root": new_train["fall_resets"]},
            "qualification": "Root-XY command, legal reset frame and episode truncation distribution change together as a declared curriculum intervention; not an expert-only or isolated loss ablation.",
        },
        "exact_initial_and_reference_pairs": paired,
        "paired_clip_changes_vs_backward_penalty1000": changes,
        "limitations": [
            "All22 clips and post-step fall frames are retained; upright-only is secondary and cross-source derivative intervals alone are excluded.",
            "The frozen90D local pose/velocity biased Gaussian MMD squared uses sigma1/speed conditioning; author statistic and acceptance threshold are unknown.",
            "Toe near-floor body-origin speed and mass-weighted body origins are proxies, not measured foot support, actual COM or peer/static contacts.",
            "Matched evaluation initial/reference arrays do not imply equal training reset trajectories or bitwise physical reproduction.",
            "Validation has been repeatedly exposed; this is single-seed iterative validation, not an untouched test or multi-seed significance result.",
            "The500 checkpoint has half the1000 training budget and is an exploratory checkpoint diagnostic, not a posthoc replacement with equal-budget proof.",
            "AMASS-root truefutureXY isolated tracking does not establish learned Diffusion social navigation or the author eight-scene experiment.",
        ],
    }
    with (here / "comparison.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"tracking": run["aggregate"], "inclusive_mmd2": run["inclusive_mmd2"]} for name, run in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
