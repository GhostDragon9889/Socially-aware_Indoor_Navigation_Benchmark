"""CPU-only summary of completed, unchanged tracking/style evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"


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
    return {key: {"baseline": left.get(key), "stratified": right.get(key)}
            for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}


def exact_initial_and_reference_pairs(left, right):
    initial_keys = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
                    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history")
    clock_keys = ("reference_frame", "reference_time_s", "metric_frame", "native_reset_frame", "episode_id",
                  "trajectory_xy", "waypoint_query_phase_frame", "waypoint_held_at_clip_end")
    left_rows = {row["motion_id"]: row for row in left["sequences"]}
    assert set(left_rows) == {row["motion_id"] for row in right["sequences"]}
    checks = []
    for row in right["sequences"]:
        prior = left_rows[row["motion_id"]]
        assert prior["source_path"] == row["source_path"]
        with np.load(prior["capture"], allow_pickle=False) as a, np.load(row["capture"], allow_pickle=False) as b:
            keys = sorted(set(key for key in a.files if key.startswith("reference_")) | set(clock_keys))
            full = {key: np.array_equal(a[key], b[key]) for key in keys}
            initial = {key: np.array_equal(a[key][0], b[key][0]) for key in initial_keys}
            assert all(full.values()) and all(initial.values())
        checks.append({"motion_id": row["motion_id"], "source_path": row["source_path"],
                       "reference_and_clock_arrays_exact": full, "initial_physical_state_and_amp_history_exact": initial})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--decomposition", action="append", required=True, help="NAME=existing diagnostic JSON")
    args = parser.parse_args()
    here = args.evaluation.resolve().parent
    style = load(args.evaluation)
    prior = load(DOCS / "backward_penalty_style_v1/evaluation.json")
    reference = record(here / "heldout_reference.json")
    assert style["status"] == "passed" and style["device"] == "CPU" and not style["kit_started"]
    assert reference["sha256"] == REFERENCE_SHA
    baseline_equality = {name: value == style["runs"][name] for name, value in prior["runs"].items()}
    assert all(baseline_equality.values())
    decomposition_paths = {}
    for value in args.decomposition:
        name, separator, path = value.partition("=")
        assert separator and name not in decomposition_paths
        decomposition_paths[name] = Path(path)
    assert set(decomposition_paths) == set(style["runs"])
    results = {}
    runs = {}
    for name, item in style["runs"].items():
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        decomposition = load(decomposition_paths[name])
        assert decomposition["status"] == "passed" and decomposition["analytic_rigid_transform_check_passed"]
        assert decomposition["primary_aggregate_unchanged"] == result["aggregate"]
        assert decomposition["source_result"]["sha256"] == result_source["sha256"]
        results[name] = result
        inclusive = item["inclusive_speed_conditioned"]
        bins = [{key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
                for row in inclusive["speed_bins"]]
        runs[name] = {
            "result": result_source, "checkpoint": item["sources"]["models_resolved_from_evaluation_arguments"]["controller"],
            "aggregate": result["aggregate"], "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
            "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"], "speed_bins": bins,
            "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
            "mass_weighted_joint_origin_height_proxy_m": item["generated"]["metrics"]["mass_weighted_joint_origin_height_proxy_m"],
            "capture_validation": {key: item["capture_validation"][key] for key in
                                   ("metric_frames", "clips", "initial_reset_frames_discarded", "valid_same_clip_style_intervals", "fall_metric_frames_kept")},
            "decomposition": record(decomposition_paths[name]), "same_frame_decomposition": decomposition["aggregate_diagnostic"],
            "wrists_and_hands": [row for row in decomposition["by_body"] if row["body"] in ("L_Wrist", "R_Wrist", "L_Hand", "R_Hand")],
            "by_reference_speed_bin": decomposition["by_reference_speed_bin"],
            "heldout_parent_disjointness": item["reference_not_in_declared_training"],
        }
    train_old_path = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/result.json"
    train_new_path = ROOT / "output/diffusion_amp/amp_style_amass2848_stratified_experts_v1/result.json"
    train_old, train_new = load(train_old_path), load(train_new_path)
    config_diff = differences(train_old["training_config"], train_new["training_config"])
    assert set(config_diff) == {"expert_sampling"} and config_diff["expert_sampling"]["baseline"] is None
    setup_keys = ("physical_transition_count", "actual_control_steps", "update_count", "curriculum", "initialization",
                  "runtime_numerics", "physics", "runtime_execution", "power_measurement")
    equal_setup = {key: train_old[key] == train_new[key] for key in setup_keys}
    assert all(equal_setup.values())
    baseline = results["backward_penalty1000"]
    primary = results["stratified_experts1000"]
    equal_protocol = {key: baseline[key] == primary[key] for key in ("protocol", "physics", "layout", "reference_velocity_contract")}
    assert all(equal_protocol.values())
    primary_pair_checks = exact_initial_and_reference_pairs(baseline, primary)
    old_rows = {row["motion_id"]: row for row in baseline["sequences"]}
    old_success = {row["motion_id"] for row in baseline["sequences"] if row["tracking_success"]}
    new_success = {row["motion_id"] for row in primary["sequences"] if row["tracking_success"]}
    changes = {"gained_success_motion_ids": sorted(new_success - old_success), "lost_success_motion_ids": sorted(old_success - new_success),
               "clips_with_position_error_improved": sum(row["mean_body_position_error_m"] < old_rows[row["motion_id"]]["mean_body_position_error_m"] for row in primary["sequences"]),
               "clips_with_rotation_error_improved": sum(row["mean_body_rotation_error_rad"] < old_rows[row["motion_id"]]["mean_body_rotation_error_rad"] for row in primary["sequences"]),
               "fall_sources": [row["source_path"] for row in primary["sequences"] if row["sequence_had_fall"]]}
    summary = {
        "schema": "navisaaclab.stratified_expert_tracking_style_review.v1", "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_only": True, "Kit_started": False, "business_data_profiles_and_evaluator_modified": False,
        "style_evaluation": record(args.evaluation), "frozen_reference": reference,
        "prior_four_run_dicts_exactly_unchanged": baseline_equality, "frozen_implementations": style["sources"],
        "summary_script": record(__file__), "style_spec": style["reference"]["metric_spec"], "runs": runs,
        "sampling_only_training_comparison": {
            "training_results": {"baseline": record(train_old_path), "stratified": record(train_new_path)},
            "training_config_differences": config_diff, "training_cli_differences": differences(train_old["arguments"], train_new["arguments"]),
            "equal_training_setup_fields": equal_setup, "provenance_differences": differences(train_old["provenance"], train_new["provenance"]),
            "equal_tracking_protocol_fields": equal_protocol, "exact_initial_and_reference_pairs": primary_pair_checks,
            "updates_each": 1000, "physical_transitions_each": 8192000,
            "training_fall_resets": {"baseline": train_old["fall_resets"], "stratified": train_new["fall_resets"]},
            "matched_call_shared_RNG_preservation_is_not_equal_realized_fall_reset_sequences": True,
        },
        "primary_paired_clip_changes": changes,
        "limitations": [
            "All22 sequences and all post-step fall frames remain in the inclusive metrics; upright-only is secondary.",
            "90D local position/velocity biased Gaussian MMD squared has sigma1 and speed-bin conditioning; exact author statistic/threshold is unknown.",
            "Foot sliding is a near-ground body-origin/vertical-speed proxy, not actual support-point contact or true COM.",
            "Shared corpus RNG is preserved under matching calls; changed learned dynamics/fall/reset schedules need not produce identical training rollouts.",
            "A single seeded training run is not multi-seed significance or physical bitwise equivalence evidence.",
            "The exposed validation reference is used iteratively; it is not an untouched final test set.",
            "This truefutureXY isolated low-level benchmark does not establish learned Diffusion crowd navigation or the author's eight-scene experiment.",
            "The500 stratified checkpoint, when supplied, has half the1000 budget and is a post-training exploratory diagnostic, not a tuned replacement.",
        ],
    }
    if "stratified_experts500" in results:
        summary["step500_exact_initial_and_reference_pairs"] = exact_initial_and_reference_pairs(baseline, results["stratified_experts500"])
    with (here / "comparison.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"tracking": value["aggregate"], "inclusive_mmd2": value["inclusive_mmd2"]} for name, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
