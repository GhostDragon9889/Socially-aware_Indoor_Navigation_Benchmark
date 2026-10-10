"""Read completed, frozen evaluations; do not recompute or change their metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOCS = HERE.parent


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def differences(left, right):
    return {key: {"printed_equation": left.get(key), "backward_penalty": right.get(key)}
            for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}


def main():
    style_path = HERE / "evaluation.json"
    style = load(style_path)
    previous = load(DOCS / "tracking_style_v1/three_checkpoints/evaluation.json")
    reference = record(HERE / "heldout_reference.json")
    expected_reference_sha = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
    assert style["status"] == "passed" and not style["kit_started"] and style["device"] == "CPU"
    assert reference["sha256"] == expected_reference_sha
    baseline_equality = {name: value == style["runs"][name] for name, value in previous["runs"].items()}
    assert all(baseline_equality.values())
    for source in style["sources"].values():
        assert record(source["path"]) == source

    decomposition_paths = {
        "legacy1000": DOCS / "legacy_tracking_decomposition.json",
        "amass2848_child500": DOCS / "amass2848_child_step500_tracking_decomposition.json",
        "amass2848_child1000": DOCS / "amass2848_child_tracking_decomposition.json",
        "backward_penalty1000": DOCS / "backward_penalty_tracking_decomposition.json",
    }
    runs = {}
    results = {}
    for name, item in style["runs"].items():
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        results[name] = result
        decomposition = load(decomposition_paths[name])
        assert decomposition["status"] == "passed" and decomposition["analytic_rigid_transform_check_passed"]
        assert decomposition["primary_aggregate_unchanged"] == result["aggregate"]
        assert decomposition["source_result"]["sha256"] == result_source["sha256"]
        bins = []
        for row in item["inclusive_speed_conditioned"]["speed_bins"]:
            bins.append({key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                        | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]})
        generated = item["generated"]
        runs[name] = {
            "result": result_source,
            "checkpoint": item["sources"]["models_resolved_from_evaluation_arguments"]["controller"],
            "decomposition": record(decomposition_paths[name]),
            "aggregate": result["aggregate"],
            "inclusive_mmd2": item["inclusive_speed_conditioned"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_diagnostic_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
            "speed_wasserstein_m_s": item["inclusive_speed_conditioned"]["physical_speed_distribution_wasserstein_m_s"],
            "speed_bins": bins,
            "capture_validation": {key: item["capture_validation"][key] for key in
                ("metric_frames", "clips", "initial_reset_frames_discarded", "valid_same_clip_style_intervals", "fall_metric_frames_kept")},
            "toe_near_ground_proxy": generated["toe_slide_near_ground_proxy"],
            "toe_per_state_proxy": generated["metrics"]["toe_slide_near_ground_proxy_m_s"],
            "mass_weighted_joint_origin_height_proxy_m": generated["metrics"]["mass_weighted_joint_origin_height_proxy_m"],
            "actual_foot_ground_contact": item["actual_foot_ground_contact"],
            "true_com_world": item["true_com_world"],
            "same_frame_decomposition": decomposition["aggregate_diagnostic"],
            "wrists_and_hands": [row for row in decomposition["by_body"] if row["body"] in
                                  ("L_Wrist", "R_Wrist", "L_Hand", "R_Hand")],
            "by_reference_speed_bin": decomposition["by_reference_speed_bin"],
            "reference_training_parent_disjointness": item["reference_not_in_declared_training"],
        }

    old = results["amass2848_child1000"]
    new = results["backward_penalty1000"]
    train_old_path = ROOT / "output/diffusion_amp/amp_style_amass2848_child_v1/result.json"
    train_new_path = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/result.json"
    train_old = load(train_old_path)
    train_new = load(train_new_path)
    config_diff = differences(train_old["training_config"], train_new["training_config"])
    assert config_diff == {"direction_mode": {"printed_equation": "paper_equation", "backward_penalty": "backwards_penalty"}}
    training_equal = {key: train_old[key] == train_new[key] for key in
                      ("physical_transition_count", "actual_control_steps", "update_count", "curriculum", "initialization",
                       "runtime_numerics", "physics", "runtime_execution", "provenance", "power_measurement")}
    assert all(training_equal.values())
    evaluation_equal = {key: old[key] == new[key] for key in ("protocol", "physics", "layout", "reference_velocity_contract")}
    assert all(evaluation_equal.values())

    # Verify exact oracle inputs and resets, without comparing later physical trajectories.
    initial_keys = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
                    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history")
    clock_keys = ("reference_frame", "reference_time_s", "metric_frame", "native_reset_frame", "episode_id",
                  "trajectory_xy", "waypoint_query_phase_frame", "waypoint_held_at_clip_end")
    pair_checks = []
    old_rows = {row["motion_id"]: row for row in old["sequences"]}
    assert set(old_rows) == {row["motion_id"] for row in new["sequences"]}
    for row in new["sequences"]:
        prior = old_rows[row["motion_id"]]
        assert prior["source_path"] == row["source_path"]
        with np.load(prior["capture"], allow_pickle=False) as left, np.load(row["capture"], allow_pickle=False) as right:
            reference_keys = sorted(key for key in left.files if key.startswith("reference_"))
            full_keys = sorted(set(reference_keys) | set(clock_keys))
            full_equal = {key: np.array_equal(left[key], right[key]) for key in full_keys}
            initial_equal = {key: np.array_equal(left[key][0], right[key][0]) for key in initial_keys}
            assert all(full_equal.values()) and all(initial_equal.values())
        pair_checks.append({"motion_id": row["motion_id"], "source_path": row["source_path"],
                            "all_reference_and_clock_arrays_exactly_equal": full_equal,
                            "initial_physical_state_and_amp_history_exactly_equal": initial_equal})

    last_metrics = {}
    for label, result_path in (("printed_equation", train_old_path), ("backward_penalty", train_new_path)):
        metrics_path = result_path.parent / "metrics.jsonl"
        latest = json.loads(metrics_path.read_text().splitlines()[-1])
        last_metrics[label] = {"source": record(metrics_path), "last_update": {key: latest[key] for key in
            ("update_count", "physical_transitions", "backwards_fraction", "forward_velocity_mean",
             "direction_rewards_mean", "amp_rewards_mean", "trajectory_rewards_mean", "energy_rewards_mean", "reward_mean")}}
    old_success = {row["motion_id"] for row in old["sequences"] if row["tracking_success"]}
    new_success = {row["motion_id"] for row in new["sequences"] if row["tracking_success"]}
    clip_change = {"gained_success_motion_ids": sorted(new_success - old_success),
                   "lost_success_motion_ids": sorted(old_success - new_success),
                   "clips_with_position_error_improved": sum(row["mean_body_position_error_m"] < old_rows[row["motion_id"]]["mean_body_position_error_m"] for row in new["sequences"]),
                   "clips_with_rotation_error_improved": sum(row["mean_body_rotation_error_rad"] < old_rows[row["motion_id"]]["mean_body_rotation_error_rad"] for row in new["sequences"]),
                   "lost_success_sources": [row["source_path"] for row in new["sequences"] if row["motion_id"] in old_success - new_success]}
    summary = {
        "schema": "navisaaclab.backward_penalty_tracking_style_review.v1", "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "Kit_started": False,
        "business_code_data_profiles_or_frozen_evaluator_modified": False,
        "style_evaluation": record(style_path), "frozen_reference": reference,
        "prior_three_run_dicts_exactly_unchanged": baseline_equality,
        "frozen_implementations": style["sources"], "summary_script": record(__file__),
        "style_spec": style["reference"]["metric_spec"], "runs": runs,
        "direction_only_training_comparison": {
            "training_results": {"printed_equation": record(train_old_path), "backward_penalty": record(train_new_path)},
            "training_config_differences": config_diff,
            "training_cli_differences": differences(train_old["arguments"], train_new["arguments"]),
            "equal_training_setup_fields": training_equal, "equal_tracking_protocol_fields": evaluation_equal,
            "updates_each": train_new["update_count"], "physical_transitions_each": train_new["physical_transition_count"],
            "fall_resets_during_training": {"printed_equation": train_old["fall_resets"], "backward_penalty": train_new["fall_resets"]},
            "tracking_initial_and_oracle_inputs": pair_checks, "last_update_metrics": last_metrics,
            "not_identical_realized_training_rollouts": True, "no_multiseed_confidence_or_PhysX_bitwise_claim": True,
        },
        "paired_tracking_clip_changes": clip_change,
        "limitations": [
            "All22 original primary sequences and all fall frames are retained; upright is a secondary diagnostic only.",
            "Style uses speed-conditioned90D local-position/velocity Gaussian biased MMD squared, sigma1; author exact statistic/threshold is unknown.",
            "Style speed bins use physical root displacement/dt; decomposition bins use same-frame captured reference velocity.",
            "Low-speed-bin improvement includes removal of fallen states; upright-only MMD of the new model is slightly worse than prior positive-sign model.",
            "Toe support and sliding use body-origin height/vertical-speed thresholds; no actual foot contact, support-point velocity or true COM was captured in these runs.",
            "Reference has been inspected during development, so it is not an untouched final test set.",
            "This is isolated low-level tracking of truefutureXY oracle commands, not learned Diffusion crowd navigation or the author eight-scene experiment.",
            "The legacy model used2.048M transitions/319clips and the new500 model4.096M; only the positive1000/newpenalty1000 pair share8.192M budget and one config difference.",
            "Backward penalty is an explicit semantic correction/ablation of the printed positive-backward equation, not an exact reproduction of that printed equation.",
        ],
    }
    (HERE / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("PASS: unchanged frozen reference/baselines, exact22 reset/oracle pairs, direction-only config, four-model summary")
    print(json.dumps({name: {"tracking": value["aggregate"], "inclusive_mmd2": value["inclusive_mmd2"]}
                      for name, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
