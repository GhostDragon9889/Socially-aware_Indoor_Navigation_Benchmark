"""Append actual state-action200s evidence without recomputing old4 scores.

Keeps frozen c274 features/MMD, f851 measurement and ee0 idle/phase outputs.
No Torch, Kit, CUDA, training or evaluation retuning is performed here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
HERE = DOCS / "state_action_navigation_style_v1"
OLD = DOCS / "mixed_curriculum_navigation_style_v1"
RAW = DOCS / "state_action_navigation_ground2000_review_v1"
NAME = "state_action2000_nav"
REF_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only summary required")
    prior_eval, prior = load(OLD / "evaluation.json"), load(OLD / "comparison.json")
    fresh = load(HERE / "new_evaluation/evaluation.json")
    raw, idle = load(RAW / "audit.json"), load(RAW / "stagnation.json")
    require(prior_eval["status"] == prior["status"] == fresh["status"] == raw["status"] == "passed",
            "Completed old/new frozen style and actual raw audit required")
    require(set(prior_eval["runs"]) == set(prior["runs"]) and len(prior["runs"]) == 4 and
            set(fresh["runs"]) == {NAME} and fresh["sources"] == prior_eval["sources"] and
            fresh["reference"] == prior_eval["reference"], "Frozen4 predecessor/metric source differs")
    reference = HERE / "new_evaluation/heldout_reference.json"
    require(record(reference)["sha256"] == REF_SHA, "Frozen reference bytes changed")
    merged, comparison = deepcopy(prior_eval), deepcopy(prior)
    merged["runs"][NAME] = fresh["runs"][NAME]
    item = fresh["runs"][NAME]
    result_record = item["sources"]["result"]
    require(record(result_record["path"]) == result_record == raw["artifacts"]["result"], "Actual complete result identity differs")
    result = load(result_record["path"])
    require(result["actual_control_steps"] == 6000 and result["arguments"]["num_envs"] == 4 and
            result["trajectory_source"] == "diffusion" and result["curriculum"] is None,
            "Actual full4x200s deployment required, not training curriculum")
    valid = sum(row["valid_same_episode_intervals"] for row in idle["per_actor"])
    idle_count = sum(row["idle_intervals"] for row in idle["per_actor"])
    inclusive = item["inclusive_speed_conditioned"]
    require(valid == item["generated"]["same_episode_transition_count"] and
            idle_count == inclusive["speed_bins"][0]["generated_count"], "Original idle/style derivative denominator differs")
    comparison["runs"][NAME] = {
        "result": result_record, "states": item["sources"]["states"],
        "models": item["sources"]["models_resolved_from_evaluation_arguments"],
        "capture_validation": item["capture_validation"], "heldout_parent_disjointness": item["reference_not_in_declared_training"],
        "same_episode_intervals": valid, "valid_acceleration_triplets": item["generated"]["valid_acceleration_triplets"],
        "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
        "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
        "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
        "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"],
        "physical_root_speed_m_s": item["generated"]["physical_root_speed_m_s"],
        "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
        "idle_total_actor_seconds": idle_count / 30, "idle_same_episode_interval_fraction": idle_count / valid,
        "idle_per_actor": idle["per_actor"], "total_route_vertices": sum(result["route_vertices_reached"]),
        "route_vertices_per_actor": result["route_vertices_reached"],
        "recorded_outcomes": {key: result.get(key) for key in ("fall_resets", "time_limit_resets", "resets", "minimum_height_m",
            "global_path_failures", "successful_start_recoveries", "actual_control_steps")},
        "speed_bins": [{key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
            | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
            for row in inclusive["speed_bins"]]}
    exact_eval = {name: row == merged["runs"][name] for name, row in prior_eval["runs"].items()}
    exact_comp = {name: row == comparison["runs"][name] for name, row in prior["runs"].items()}
    require(all(exact_eval.values()) and all(exact_comp.values()), "Old4 evaluation/comparison dictionaries changed")
    old_phase = load(OLD / "stagnation.json")
    phase = deepcopy(old_phase)
    phase["stagnation"][NAME] = idle
    phase["old_stagnation_recomputed"] = False
    phase["independent_native_ground_reports"][NAME] = record(RAW / "audit.json")
    spec = importlib.util.spec_from_file_location("frozen_navigation_articulation_reader", OLD / "summarize.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    articulation = load(OLD / "descriptive_yaw_wrist.json")
    articulation["runs"][NAME] = helper.observed_articulation(Path(result_record["path"]).parent)
    support_paths = {
        "guidance20": DOCS / "ground_support_200s_actual_verification.json",
        "mixed2000": DOCS / "mixed_step2000_navigation_ground_actual_verification_v1.json",
        NAME: RAW / "audit.json"}
    ground = {}
    for name, path in support_paths.items():
        report = load(path)
        require(report["status"] == "passed" and report["artifacts"]["result"] == comparison["runs"][name]["result"],
                "Existing baseline/new native source binding differs:" + name)
        ground[name] = {"original_audit": record(path),
            **{key: deepcopy(report[key]) for key in ("counts", "support_statistics", "body_origin_motion_proxy",
                                                     "COM_statistics", "original_obstacle_peer_channels", "sources")}}
    supported = {name: [(row["lower_m_s"], row["upper_m_s"]) for row in comparison["runs"][name]["speed_bins"]
                       if row["mmd2"] is not None] for name in support_paths}
    config_diff = {name: {key: {"baseline": load(comparison["runs"][name]["result"]["path"])["training_config"].get(key),
                                "candidate": result["training_config"].get(key)}
        for key in set(load(comparison["runs"][name]["result"]["path"])["training_config"]) | set(result["training_config"])
        if load(comparison["runs"][name]["result"]["path"])["training_config"].get(key) != result["training_config"].get(key)}
        for name in ("guidance20", "mixed2000")}
    now = datetime.now(timezone.utc).isoformat()
    provenance = {"previous_evaluation": record(OLD / "evaluation.json"), "previous_comparison": record(OLD / "comparison.json"),
        "new_only_evaluation": record(HERE / "new_evaluation/evaluation.json"), "actual_native_ground_audit": record(RAW / "audit.json"),
        "actual_source_admission": raw["sources"], "frozen_style_reference": record(reference),
        "old4_scores_or_denominators_recomputed": False, "script": record(__file__)}
    merged.update(generated_at_utc=now, append_provenance=provenance)
    comparison.update(schema="state_action.closed_loop200s_frozen_comparison.v1", generated_at_utc=now,
        old4_evaluation_dictionaries_exact=exact_eval, old4_comparison_dictionaries_exact=exact_comp,
        append_provenance=provenance, actual_new_source_epoch="isolatedv3/old146, separately bound from old baseline127/mixed138",
        actual_new_discriminator="2019 state_action independent D; expert next-frame normalizedPD position proxy, not measured expert control",
        supported_primary_speed_bins=supported,
        same_supported_speed_bins_vs_selected_baseline=supported[NAME] == supported["guidance20"],
        explicit_training_configuration_differences=config_diff,
        recommendation="Keep selected negative-guide20 default (0F/21 vertices). State-action2F/2 vertices with73.985% idle fails navigation acceptance; tracking17/22 is not navigation success.",
        no_author_equivalence_or_model_promotion_claim=True)
    comparison["limitations"] += [
        "New model majority idle: conditional speed-bin MMD alone cannot demonstrate locomotion; support-bin subsets may differ, kept explicit without retuning.",
        "State-actionD input, independent initialization/AMP-moment transfer and learned weight/statistics differ; selected baseline also has half training budget/different command curriculum.",
        "Isolated146 actual entry/argv/importledger differs from historical source127/138; default policy/clock/force protocol retained, no all-source equality claim."]
    write(HERE / "evaluation.json", merged)
    write(HERE / "comparison.json", comparison)
    write(HERE / "stagnation.json", phase)
    write(HERE / "descriptive_yaw_wrist.json", articulation)
    write(HERE / "native_ground_comparison.json", {"status":"passed","CPU_only":True,"measurement_formulas_changed":False,
        "baseline_native_arrays_reaudited":False,"runs":ground,"comparison_source":provenance,
        "friction_or_reference_true_COM_available":False,"body_origin_proxy_is_contact_patch_slip":False})
    shutil.copyfile(reference, HERE / "heldout_reference.json")
    print(json.dumps({"status":"passed","old4_dicts_exact":True,"new_run":NAME,
        "idle_actor_seconds":idle_count/30,"idle_fraction":idle_count/valid,"supported_bins_same":supported[NAME]==supported["guidance20"]}))


if __name__ == "__main__":
    main()
