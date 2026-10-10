"""Read completed navigation/style/unchanged-phase reports; no new metric kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from summarize_near_horizon_style import DOCS, REFERENCE_SHA, differences, load, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--stagnation", type=Path, required=True)
    args = parser.parse_args()
    here = args.evaluation.resolve().parent
    evaluation = load(args.evaluation)
    previous_path = DOCS / "near_horizon_eta025_style_v1/evaluation.json"
    previous = load(previous_path)
    assert evaluation["status"] == "passed" and evaluation["device"] == "CPU" and not evaluation["kit_started"]
    assert set(evaluation["runs"]) == {"guidance20", "amass_root1000"}
    assert evaluation["runs"]["guidance20"] == previous["runs"]["guidance20"]
    assert evaluation["sources"] == previous["sources"]
    reference = record(here / "heldout_reference.json")
    assert reference["sha256"] == REFERENCE_SHA
    stagnation = load(args.stagnation)
    assert stagnation["status"] == "passed" and not stagnation["feature_time_or_threshold_modified"]
    assert not stagnation["old_stagnation_recomputed"]
    old_phase = load(stagnation["old_phase_report"]["path"])["stagnation"]
    assert old_phase == stagnation["stagnation"]["guidance20"]
    runs, results = {}, {}
    for name, item in evaluation["runs"].items():
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        assert result["status"] == "passed" and result.get("error") is None
        assert result["actual_control_steps"] == 6000 and result["arguments"]["num_envs"] == 4
        assert result["trajectory_source"] == "diffusion" and result["curriculum"] is None
        assert result["runtime_execution"]["consumed_planner_output"] == "XY only"
        assert result["runtime_execution"]["diffusion_used_for_rollouts"]
        inclusive = item["inclusive_speed_conditioned"]
        idle = stagnation["stagnation"][name]
        per_actor = idle["per_actor"]
        valid_intervals = sum(actor["valid_same_episode_intervals"] for actor in per_actor)
        idle_intervals = sum(actor["idle_intervals"] for actor in per_actor)
        assert valid_intervals == item["generated"]["same_episode_transition_count"]
        assert idle_intervals == inclusive["speed_bins"][0]["generated_count"]
        results[name] = result
        runs[name] = {
            "result": result_source,
            "states": item["sources"]["states"],
            "models": item["sources"]["models_resolved_from_evaluation_arguments"],
            "capture_validation": item["capture_validation"],
            "heldout_parent_disjointness": item["reference_not_in_declared_training"],
            "same_episode_intervals": valid_intervals,
            "valid_acceleration_triplets": item["generated"]["valid_acceleration_triplets"],
            "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
            "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"],
            "physical_root_speed_m_s": item["generated"]["physical_root_speed_m_s"],
            "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
            "idle_total_actor_seconds": idle_intervals / 30,
            "idle_same_episode_interval_fraction": idle_intervals / valid_intervals,
            "idle_per_actor": per_actor,
            "total_route_vertices": sum(result["route_vertices_reached"]),
            "route_vertices_per_actor": result["route_vertices_reached"],
            "recorded_outcomes": {key: result.get(key) for key in (
                "fall_resets", "time_limit_resets", "resets", "minimum_height_m", "global_path_failures",
                "successful_start_recoveries", "actual_control_steps")},
            "speed_bins": [
                {key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
                for row in inclusive["speed_bins"]
            ],
        }
    baseline, candidate = results["guidance20"], results["amass_root1000"]
    assert runs["guidance20"]["models"]["diffusion"] == runs["amass_root1000"]["models"]["diffusion"]
    assert runs["amass_root1000"]["models"]["controller"]["sha256"] == "dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e"
    argument_diff = differences(baseline["arguments"], candidate["arguments"])
    assert set(argument_diff) == {"config", "output", "resume", "near_horizon_waypoint_mass"}
    assert baseline["arguments"].get("near_horizon_waypoint_mass", 0) == candidate["arguments"]["near_horizon_waypoint_mass"] == 0
    old_config, new_config = baseline["training_config"], candidate["training_config"]
    assert dict(old_config, expert_sampling=None) == dict(new_config, expert_sampling=None)
    equal_fields = {key: baseline[key] == candidate[key] for key in (
        "physics", "evaluation_scene", "runtime_numerics", "runtime_execution", "reference_adaptation",
        "physical_contact_measurement", "power_measurement", "route_completion_contract", "diffusion_guidance")}
    # Raw reference_adaptation contains each run's output plan path. That is
    # an artifact identity, not a changed adaptation parameter.
    adaptation_parameters_equal = (
        {key: value for key, value in baseline["reference_adaptation"].items() if key != "plan_capture"}
        == {key: value for key, value in candidate["reference_adaptation"].items() if key != "plan_capture"}
    )
    assert adaptation_parameters_equal
    assert all(value for key, value in equal_fields.items() if key != "reference_adaptation")
    summary = {
        "schema": "navisaaclab.amass_root_closed_loop_style_comparison.v1",
        "status": "passed",
        "status_meaning": "CPU measurements and artifact contracts passed; selected controller's navigation quality failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_only": True,
        "Kit_started": False,
        "GPU_used": False,
        "sealed34_test_capture_read": False,
        "business_profiles_resources_models_or_metric_kernels_modified": False,
        "style_evaluation": record(args.evaluation),
        "reference": reference,
        "previous_style_report": record(previous_path),
        "previous_guidance20_run_dictionary_exactly_unchanged": True,
        "frozen_implementations": evaluation["sources"],
        "summary_script": record(__file__),
        "unchanged_stagnation": record(args.stagnation),
        "unchanged_phase_helper": stagnation["unchanged_helper"],
        "baseline_stagnation_object_exact_and_not_recomputed": True,
        "style_spec": evaluation["reference"]["metric_spec"],
        "runs": runs,
        "runtime_argument_differences": argument_diff,
        "training_config_raw_differences": differences(old_config, new_config),
        "training_config_legacy_disabled_sampler_equivalent": True,
        "equal_recorded_protocol_fields": equal_fields,
        "reference_adaptation_parameters_equal_except_output_plan_artifact_path": adaptation_parameters_equal,
        "planning_clock_protocol_equal": {key: baseline["planning_clock"][key] == candidate["planning_clock"][key]
                                          for key in ("mode", "maximum_interval_steps")},
        "model_transfer_interpretation": "Different trained weights/normalizers under explicit AMASS-root curriculum; identical deployment protocol and Diffusion weight. This is not a single-normalizer causal ablation.",
        "quality_recommendation": "Keep existing0fall/21vertex deployment baseline; selected1000 did not transfer its full22 tracking improvement to closed-loop navigation.",
        "limitations": [
            "Every fall/reset is retained; only cross-reset derivative intervals are excluded. Upright-only is secondary.",
            "Idle uses the existing same-episode30Hz displacement-speed<0.1m/s formula and source-time convention; no threshold tuning.",
            "Route vertices are recorded cumulative progress, not uninterrupted success or eight-scene author performance.",
            "Toe near-floor body-origin speed is not actual contact-point slip; native ground/COM/contact audit is separate.",
            "Static task terrain RMS is bit-identical across checkpoints; other weights/statistics differ and no cause was isolated.",
            "The exposed c274 validation reference and one seed are iterative evidence, not untouched test or population significance.",
        ],
    }
    with (here / "comparison.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {key: value[key] for key in (
        "inclusive_mmd2", "speed_wasserstein_m_s", "idle_total_actor_seconds", "total_route_vertices", "recorded_outcomes")}
        for name, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
