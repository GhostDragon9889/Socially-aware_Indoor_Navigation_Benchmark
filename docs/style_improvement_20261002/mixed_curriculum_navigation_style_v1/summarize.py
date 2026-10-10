"""Append two real mixed navigation captures; preserve historical dictionaries."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/style_improvement_20261002"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
FROZEN = {
    "scripts/evaluate_pedestrian_style.py": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "CrowdSim/humanoid/style_evaluation.py": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
    "docs/style_improvement_20261002/summarize_navigation_style.py": "ee0d13c3f345cf11aa2981002c21b070bf53eb4b6129d86d4a6f851443ac56d7",
}


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def observed_articulation(directory):
    """Descriptive actual yaw/wrists; no paired AMASS pose/orientation error."""
    from CrowdSim.humanoid.style_evaluation import KEYPOINTS, sequence_style_samples, summary
    from scripts.evaluate_paper_pedestrians import read_capture

    arrays, runner, capture = read_capture(directory / "physical_states.npz", directory / "result.json")
    sample = sequence_style_samples(arrays["body_pos"], arrays["root_rot"],
        body_names=capture["body_names"], episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"])
    q = arrays["root_rot"].astype(np.float64)
    yaw = np.arctan2(2*(q[..., 3]*q[..., 2] + q[..., 0]*q[..., 1]), 1-2*(q[..., 1]**2 + q[..., 2]**2))
    valid = ((arrays["episode_ids"][1:] == arrays["episode_ids"][:-1]) & ~arrays["reset_mask"][1:])
    yaw_delta = np.diff(yaw, axis=0)
    yaw_rate = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)) * 30
    root = arrays["body_pos"][..., capture["body_names"].index("Pelvis"), :2].astype(np.float64)
    displacement = np.diff(root, axis=0)
    speed = np.linalg.norm(displacement, axis=-1) * 30
    moving = valid & (speed >= .1)
    velocity_yaw = np.arctan2(displacement[..., 1], displacement[..., 0])
    mismatch = velocity_yaw - yaw[1:]
    mismatch = np.arctan2(np.sin(mismatch), np.cos(mismatch))
    pose = sample["local_pose"].reshape(-1, len(KEYPOINTS), 3)
    velocity = sample["local_pose_velocity"].reshape(-1, len(KEYPOINTS), 3)
    wrists = {}
    for name in ("L_Wrist", "R_Wrist"):
        index = KEYPOINTS.index(name)
        wrists[name] = {"heading_local_position_chain_units": {
            axis: summary(pose[:, index, i]) for i, axis in enumerate("xyz")},
            "heading_local_speed_chain_s": summary(np.linalg.norm(velocity[:, index], axis=-1))}
    return {"same_episode_intervals": int(valid.sum()),
        "absolute_root_yaw_rate_rad_s": summary(np.abs(yaw_rate[valid])),
        "moving_displacement_intervals_speed_at_least_0_1": int(moving.sum()),
        "moving_heading_vs_displacement_abs_wrapped_rad": summary(np.abs(mismatch[moving])),
        "moving_opposite_heading_fraction": float((np.cos(mismatch[moving]) < 0).mean()),
        "wrists": wrists,
        "definition": "Same-episode actual root yaw and existing heading-local wrist feature coordinates; all failure frames retained. No paired reference body rotation error exists in this navigation capture.",
        "primary_MMD_feature_kernel_or_threshold_changed": False}


def main():
    frozen = {name: record(ROOT/name) for name in FROZEN}
    assert all(value["sha256"] == FROZEN[name] for name, value in frozen.items())
    old_path = DOCS / "amass_root_navigation_style_v1/evaluation.json"
    prior = load(old_path)
    fresh_path = HERE / "new_evaluation/evaluation.json"
    fresh = load(fresh_path)
    assert fresh["status"] == "passed" and fresh["device"] == "CPU" and not fresh["kit_started"]
    assert set(fresh["runs"]) == {"mixed1000", "mixed2000"}
    assert fresh["sources"] == prior["sources"] and fresh["reference"] == prior["reference"]
    reference_path = HERE / "new_evaluation/heldout_reference.json"
    assert record(reference_path)["sha256"] == REFERENCE_SHA
    merged = deepcopy(prior)
    merged["runs"].update(fresh["runs"])
    merged["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    merged["append_provenance"] = {"previous": record(old_path), "new_only": record(fresh_path),
        "historical_run_dicts_recomputed": False}
    exact_old = {name: value == merged["runs"][name] for name, value in prior["runs"].items()}
    assert all(exact_old.values())
    write(HERE / "evaluation.json", merged)
    shutil.copyfile(reference_path, HERE / "heldout_reference.json")
    spec = importlib.util.spec_from_file_location("unchanged_navigation_phase", DOCS / "summarize_navigation_style.py")
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    old_phase_path = DOCS / "amass_root_navigation_stagnation_v1.json"
    old_phase = load(old_phase_path)
    stagnation = deepcopy(old_phase["stagnation"])
    ground = {}
    for name, count in (("mixed1000", 1000), ("mixed2000", 2000)):
        audit_path = DOCS / f"mixed_step{count}_navigation_ground_actual_verification_v1.json"
        audit = load(audit_path)
        assert audit["status"] == "passed" and audit["actual_native_capture_validated"]
        directory = Path(audit["input"])
        stagnation[name] = phase.stagnation(directory, audit)
        ground[name] = record(audit_path)
    write(HERE / "stagnation.json", {"status": "passed", "CPU_only": True,
        "unchanged_helper": frozen["docs/style_improvement_20261002/summarize_navigation_style.py"],
        "old_phase_report": record(old_phase_path), "old_stagnation_recomputed": False,
        "feature_time_or_threshold_modified": False, "stagnation": stagnation,
        "independent_native_ground_reports": ground})
    runs, results = {}, {}
    prior_summary = load(DOCS / "amass_root_navigation_style_v1/comparison.json")
    for name, item in merged["runs"].items():
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        assert result["status"] == "passed" and result.get("error") is None
        assert result["actual_control_steps"] == 6000 and result["arguments"]["num_envs"] == 4
        assert result["trajectory_source"] == "diffusion" and result["curriculum"] is None
        results[name] = result
        if name in prior_summary["runs"]:
            runs[name] = deepcopy(prior_summary["runs"][name])
            continue
        idle = stagnation[name]["per_actor"]
        valid = sum(row["valid_same_episode_intervals"] for row in idle)
        idle_count = sum(row["idle_intervals"] for row in idle)
        inclusive = item["inclusive_speed_conditioned"]
        assert valid == item["generated"]["same_episode_transition_count"]
        assert idle_count == inclusive["speed_bins"][0]["generated_count"]
        runs[name] = {"result": result_source, "states": item["sources"]["states"],
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
            "idle_per_actor": idle, "total_route_vertices": sum(result["route_vertices_reached"]),
            "route_vertices_per_actor": result["route_vertices_reached"],
            "recorded_outcomes": {key: result.get(key) for key in ("fall_resets", "time_limit_resets", "resets",
                "minimum_height_m", "global_path_failures", "successful_start_recoveries", "actual_control_steps")},
            "speed_bins": [{key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
                for row in inclusive["speed_bins"]]}
    baseline = results["guidance20"]
    protocols = {}
    for name in fresh["runs"]:
        result = results[name]
        assert merged["runs"][name]["sources"]["models_resolved_from_evaluation_arguments"]["diffusion"] == merged["runs"]["guidance20"]["sources"]["models_resolved_from_evaluation_arguments"]["diffusion"]
        fields = {key: baseline[key] == result[key] for key in ("physics", "runtime_numerics", "runtime_execution",
            "physical_contact_measurement", "power_measurement", "route_completion_contract", "diffusion_guidance", "evaluation_scene")}
        assert all(fields.values())
        assert {key: value for key, value in baseline["reference_adaptation"].items() if key != "plan_capture"} == {
            key: value for key, value in result["reference_adaptation"].items() if key != "plan_capture"}
        assert baseline["arguments"].get("near_horizon_waypoint_mass", 0) == result["arguments"]["near_horizon_waypoint_mass"] == 0
        assert dict(baseline["training_config"], expert_sampling=None) == dict(result["training_config"], expert_sampling=None)
        protocols[name] = {"identical_recorded_protocol_fields": fields,
            "training_config_equal_with_legacy_disabled_expert_sampler": True,
            "planning_mode_and_maximum_interval_equal": all(baseline["planning_clock"][key] == result["planning_clock"][key]
                for key in ("mode", "maximum_interval_steps")),
            "actual_planner_forward_calls": result["planning_clock"]["forward_calls"],
            "runtime_argument_differences": {key: {"baseline": baseline["arguments"].get(key), "candidate": result["arguments"].get(key)}
                for key in set(baseline["arguments"]) | set(result["arguments"])
                if baseline["arguments"].get(key) != result["arguments"].get(key) or (key in baseline["arguments"]) != (key in result["arguments"])}}
    descriptive = {name: observed_articulation(Path(item["sources"]["result"]["path"]).parent)
        for name, item in merged["runs"].items()}
    write(HERE / "descriptive_yaw_wrist.json", {"status": "passed", "primary_features_kernel_or_threshold_modified": False,
        "no_paired_reference_rotation_error": True, "runs": descriptive,
        "qualification": "Descriptive actual yaw and existing local wrist coordinates, not a new style score or reference-pose tracking error."})
    comparison = {"schema": "mixed_curriculum_closed_loop_style_review.v1", "status": "passed",
        "status_meaning": "CPU artifact/measurement contracts pass, mixed navigation stability does not",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False,
        "SDK_or_Kit_started": False, "sealed34_outputs_read": False,
        "old_evaluation_dictionaries_exact": exact_old,
        "old_comparison_dictionaries_exact": {name: row == runs[name] for name, row in prior_summary["runs"].items()},
        "frozen_implementations": frozen, "reference": record(HERE / "heldout_reference.json"),
        "new_only_evaluation": record(fresh_path), "previous_evaluation": record(old_path),
        "independent_native_ground_reports": ground, "protocols": protocols, "runs": runs,
        "descriptive_articulation": record(HERE / "descriptive_yaw_wrist.json"),
        "recommendation": "Keep negative1000 guide20 0-fall baseline; neither mixed milestone passes navigation stability despite better full22 tracking.",
        "limitations": ["Inclusive all failure/low/upright/idle frames; only cross-reset derivative intervals are excluded.",
            "One seed and an engineering four-person fixture, not author eight-scene or multi-seed proof.",
            "Mixed1000 has equal8.192M training transitions to negative1000 but a different curriculum and learned statistics/weights; mixed2000 has twice the budget.",
            "Repeatedly exposed fixed validation reference; no author metric threshold or untouched-test claim.",
            "Toe body-origin motion is a proxy, not measured contact-point slip; native ground/contact/COM audit remains separate.",
            "No time-aligned AMASS body orientations in navigation, so yaw/wrists are descriptive only.",
            "No data, weights, runtime business, fixed evaluator/kernel/bins/reference or historical reports were modified."]}
    assert all(comparison["old_comparison_dictionaries_exact"].values())
    write(HERE / "comparison.json", comparison)
    print(json.dumps({name: {key: row[key] for key in ("inclusive_mmd2", "speed_wasserstein_m_s",
        "idle_total_actor_seconds", "total_route_vertices", "recorded_outcomes")} for name, row in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
