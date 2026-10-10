"""Read-only CPU QC of sealed navigation reports; never rerun a metric/model."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METRICS = ROOT / "docs/style_improvement_20261003/state_prior_metrics_v1"
ACTUAL = ROOT / "docs/style_improvement_20261003/state_prior_metrics_actual_runs_v1"
PREP = ROOT / "docs/style_improvement_20261003/state_prior_CPU_metrics_preparation_v1"
OLD = ROOT / "docs/style_improvement_20261002"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    receipt_path = ACTUAL / "navigation_CPU_completion_v1.json"
    require(record(receipt_path)["sha256"] == "e63423d4473cd1d55652d46199727c9865995f84a122cf355d1151d2b0bbe292", "Completion receipt changed")
    receipt = read(receipt_path)
    require(receipt["status"] == "passed" and receipt["CPU_only"] is True and receipt["GPU_used"] is False and receipt["sealed34_outputs_read"] is False, "Actual closed CPU receipt required")
    for row in receipt["artifact_records"]:
        require(record(row["path"]) == row and Path(row["path"]).stat().st_mode & 0o222 == 0, "Sealed actual artifact changed")
    for key in ("actual_navigation_capture", "actual_terminal_admission", "source_manifest", "frozen_catalog", "metrics_source_freeze"):
        require(record(receipt[key]["path"]) == receipt[key], "Bound actual source/input changed")
    catalog = read(receipt["frozen_catalog"]["path"])
    source = read(receipt["source_manifest"]["path"])
    declared = {row["path"]: row for row in source["files"]}
    code = Path(catalog["bundle"]) / "code"
    independent = read(HERE / "actual_source_admission.json")
    prior_peer = read(HERE / "actual_navigation_source_review.json")
    require(prior_peer["status"] == "passed" and prior_peer["actual_result"] == receipt["actual_navigation_capture"] and prior_peer["actual_source_admission"] == record(HERE / "actual_source_admission.json"), "Independent actual934 admission binding differs")
    require(independent["actual_terminal_admission"] == receipt["actual_terminal_admission"] and independent["copied_source_files_full_SHA_recomputed"] == 934, "Independent terminal/source boundary differs")
    observed = {}
    for name in ("navigation_raw", "navigation90", "navigation45"):
        admission = read(METRICS / name / "actual_source_admission.json")
        require(admission["status"] == "passed" and admission["CPU_only"] is True and admission["GPU_used"] is False and admission["Kit_started"] is False and admission["actual_source_admission"] == independent and admission["runner"] == record(catalog["runner"]) and admission["catalog"] == receipt["frozen_catalog"] and admission["fixed_metric_argv"] == catalog["metrics"][name]["metric_argv"], "Actual metric admission differs")
        for filename in ("GPU_import_ledger_as_observed.json", "GPU_import_events_as_observed.jsonl"):
            require((METRICS / name / filename).read_bytes() == (HERE / filename).read_bytes(), "Actual GPU committed-prefix bytes differ")
        launch = receipt["actual_closed_CPU_launches"][name]
        require(launch["actual_started"] is True and launch["actual_closed"] is True and launch["exit_code"] == 0 and launch["argv"] == catalog["metrics"][name]["command"] and launch["environment"] == catalog["CPU_environment"], "Actual closed CPU command differs")
        ledger_path = METRICS / (name + "_CPU_imports.json")
        ledger = read(ledger_path)
        events = [json.loads(line) for line in Path(ledger["event_ledger"]).read_text().splitlines()]
        require(ledger["status"] == "finalized" and ledger["phase"] == "passed" and ledger["source_manifest_sha256"] == receipt["source_manifest"]["sha256"] and ledger["all_observed_owned_modules_isolated"] is True and ledger["runtime_finalization_observed"] is True and len(events) == ledger["event_count"] and [event["sequence"] for event in events] == list(range(1, len(events) + 1)) and events[-1]["event"] == "final" and events[-1]["final_status"] == "passed", "Actual CPU ledger incomplete")
        executed = {event["module"]: event for event in events if event["event"] == "owned_execution"}
        require(len(executed) == len(ledger["owned_modules"]), "CPU event/module counts differ")
        for module in ledger["owned_modules"]:
            path = Path(module["path"])
            require(path.resolve().is_relative_to(code), "CPU owned module escaped934CODE")
            frozen = declared["code/" + str(path.relative_to(code))]
            actual = record(path)
            require(all(actual[key] == frozen[key] == module[key] == executed[module["module"]][key] for key in ("bytes", "sha256")) and executed[module["module"]]["execution_observed"] is True, "Actual executed CPU bytes differ")
        for stage in admission["staged_exact_external_tools"].values():
            require(record(stage["original"]["path"]) == stage["original"] and record(stage["staged_execution"]["path"]) == stage["staged_execution"] and stage["original"]["sha256"] == stage["staged_execution"]["sha256"], "Original exact staged tool changed")
        observed[name] = {"ledger": record(ledger_path), "events": record(ledger["event_ledger"]), "owned_executions": len(executed), "committed_events": len(events), "runtime_finalization_actually_observed": True}
    physical = read(HERE / "independent_input_counts.json")
    require(record(physical["source"]["path"]) == physical["source"], "Independently counted physical capture changed")
    raw = read(METRICS / "navigation_raw/audit.json")
    idle = read(METRICS / "navigation_raw/stagnation.json")
    require(raw["status"] == "passed" and raw["CPU_only"] is True and raw["counts"] == receipt["counts"] and raw["check_count"] == 203 and raw["counts"]["control_actor_frames"] == 24000 and raw["counts"]["foot_substep_observations"] == 384000 and raw["counts"]["falls"] == 0 and raw["counts"]["route_vertices_reached"] == [10] * 4, "Raw physical population differs")
    channels = raw["original_obstacle_peer_channels"]
    require(channels["static_contact_actor_frames"] == physical["static_contact_actor_frames"] == 0 and channels["peer_contact_actor_frames"] == physical["peer_contact_actor_frames"] == 0 and channels["floor_excluded"] is True, "Original contact observations differ")
    require(sum(row["idle_intervals"] for row in idle["per_actor"]) == physical["idle_actor_intervals"] == 4 and sum(row["valid_same_episode_intervals"] for row in idle["per_actor"]) == 23996 and sum(row["idle_duration_s"] for row in idle["per_actor"]) == physical["idle_actor_seconds"] and idle["all_saved_plan_target_phase_checks_passed"] is True, "Original idle/end-frame population differs")
    style = read(METRICS / "navigation90/evaluation.json")
    baseline90_path = OLD / "backward_penalty_navigation_style_v1/evaluation.json"
    require(record(baseline90_path)["sha256"] == "fa2e158e6888fe464f9a916cdbf0789f03858718ade01544c877320ea0db1306", "Selected baseline90 changed")
    baseline90 = read(baseline90_path)
    run = style["runs"]["state_prior_navigation2000"]
    old_run = baseline90["runs"]["backward_ground200"]
    require(style["status"] == "passed" and style["device"] == "CPU" and style["kit_started"] is False and style["reference"]["metric_spec"] == baseline90["reference"]["metric_spec"] and style["reference"]["actual_selected_array_sha256"] == baseline90["reference"]["actual_selected_array_sha256"], "Primary metric/reference recipe differs")
    require(run["generated"]["same_episode_transition_count"] == old_run["generated"]["same_episode_transition_count"] == 23996 and run["generated"]["valid_acceleration_triplets"] == 23992 and run["generated"]["excluded_reset_transitions"] == 0, "Inclusive original derivative population differs")
    metric = run["inclusive_speed_conditioned"]
    old_metric = old_run["inclusive_speed_conditioned"]
    bins = metric["speed_bins"]
    old_bins = old_metric["speed_bins"]
    require(metric["minimum_metric_samples_per_bin"] == old_metric["minimum_metric_samples_per_bin"] == 30 and metric["maximum_balanced_motion_samples_per_bin"] == old_metric["maximum_balanced_motion_samples_per_bin"] == 256 and [row["lower_m_s"] for row in bins] == [0., .1, .5, 1., 1.5, 2., 3.] and sum(row["generated_count"] for row in bins) == 23996, "Original bins/sample rules changed")
    values = [row["pose_velocity_gaussian_mmd2"]["value"] for row in bins if row["pose_velocity_gaussian_mmd2"] is not None]
    score = metric["equal_supported_bin_mean_pose_velocity_mmd2"]
    require(len(values) == metric["motion_supported_bins"] == 3 and score == receipt["inclusive90_MMD2"] and math.isclose(sum(values) / 3, score, abs_tol=1e-15, rel_tol=0) and old_metric["motion_supported_bins"] == 5 and old_metric["equal_supported_bin_mean_pose_velocity_mmd2"] == 0.07238060315284697, "Primary aggregate/support differs")
    require(all(bins[index]["pose_velocity_gaussian_mmd2"] is None for index in (0, 4, 5, 6)) and bins[0]["generated_count"] == 4 and bins[4]["generated_count"] == 19, "Unsupported bins must remain unknown")
    common = []
    for index in (1, 2, 3):
        new, old = bins[index], old_bins[index]
        require(new["lower_m_s"] == old["lower_m_s"] and new["upper_m_s"] == old["upper_m_s"] and new["reference_count"] == old["reference_count"], "Secondary common bins/reference differ")
        common.append({"lower_m_s": new["lower_m_s"], "upper_m_s": new["upper_m_s"], "reference_count": new["reference_count"], "selected_generated_count": old["generated_count"], "new_generated_count": new["generated_count"], "selected_MMD2": old["pose_velocity_gaussian_mmd2"]["value"], "new_MMD2": new["pose_velocity_gaussian_mmd2"]["value"]})
    pose = read(METRICS / "navigation45/evaluation.json")
    pose_run = pose["runs"]["state_prior_navigation2000"]
    old_pose_path = OLD / "pose_diversity_candidate_v1/actual_evaluation_v1/evaluation.json"
    old_pose = read(old_pose_path)
    old_pose_run = old_pose["runs"]["guidance20"]
    require(pose["status"] == "passed" and pose["CPU_only"] is True and pose["Kit_started"] is False and pose["CUDA_initialized"] is False and pose["method"]["fit"]["refit_performed_for_this_new_run"] is False and pose["method"]["fit"]["generated_samples_used_for_fit"] is False, "Saved45 basis no-fit contract differs")
    for key in ("raw_source_frames_read_without_failure_filter", "source_actor_frames", "included_pose_valid_interval_endpoints", "excluded_first_frame_actor_samples", "excluded_cross_reset_interval_endpoints", "same_episode_intervals", "cross_reset_intervals_excluded", "fall_upright_idle_or_outlier_filter_applied"):
        require(pose_run[key] == old_pose_run[key], "Original45 inclusive endpoint/filter rule differs")
    require(pose_run["included_pose_valid_interval_endpoints"] == 23996 and pose_run["fall_upright_idle_or_outlier_filter_applied"] is False, "45 population was filtered")
    basis_path = Path(pose["method"]["fit"]["saved_basis_artifact"]["path"])
    require(record(basis_path)["sha256"] == "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb" and record(METRICS / "navigation45/reference_basis.npz")["sha256"] == record(basis_path)["sha256"], "Fixed b25 basis changed")
    centers = {}
    with np.load(basis_path, allow_pickle=False) as saved, np.load(METRICS / "navigation45/reference_basis.npz", allow_pickle=False) as actual:
        require(set(saved.files) == set(actual.files), "Saved basis keys differ")
        for key in saved.files:
            left, right = saved[key], actual[key]
            require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C"), "Saved basis dtype/shape/C bytes differ")
            centers[key] = {"dtype": str(left.dtype), "shape": list(left.shape), "C_bytes_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest()}
    coverage = {}
    for count in ("50", "100"):
        new = pose_run["poses"]["clusters"][count]
        old = old_pose_run["poses"]["clusters"][count]
        require(new["samples_assigned"] == sum(new["assignment_counts"]) == 23996 and new["occupied_clusters"] == sum(value > 0 for value in new["assignment_counts"]), "45 inclusive assignments differ")
        coverage[count] = {"selected_coverage": old["coverage_occupied_over_K"], "new_coverage": new["coverage_occupied_over_K"], "new_normalized_entropy": new["normalized_entropy"]}
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "QC imported model/SDK")
    report = {"schema": "public_state_prior.actual_navigation_CPU_metrics_fresh_peer.v1", "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False, "metrics_recomputed": False, "reviewer": record(__file__), "actual_completion_receipt": record(receipt_path), "independent_actual_source_review": record(HERE / "actual_navigation_source_review.json"), "actual_terminal_admission": receipt["actual_terminal_admission"], "actual_observed_CPU_ledgers": observed, "actual_GPU_ledger_boundary": independent["actual_evaluation_imports"], "input_counts": record(HERE / "independent_input_counts.json"), "controls": 6000, "duration_s": 200, "falls": 0, "route_vertices_reached": [10] * 4, "static_contact_actor_frames": 0, "peer_contact_actor_frames": 0, "idle_actor_intervals_retained": 4, "idle_actor_seconds": physical["idle_actor_seconds"], "same_episode_intervals_retained": 23996, "first_frame_actor_samples_excluded_by_original_helper": 4, "cross_reset_intervals_excluded": 0, "primary90": {"selected_report": record(baseline90_path), "new_report": record(METRICS / "navigation90/evaluation.json"), "selected_MMD2": old_metric["equal_supported_bin_mean_pose_velocity_mmd2"], "new_MMD2": score, "selected_supported_bins": 5, "new_supported_bins": 3, "registered_numeric_threshold_passed": score <= old_metric["equal_supported_bin_mean_pose_velocity_mmd2"], "unsupported_bins_retained_unknown": True, "same_supported_bin_population": False}, "secondary_common3": {"primary_aggregate_replaced": False, "metrics_recomputed": False, "bins": common, "selected_equal_bin_mean_MMD2": sum(row["selected_MMD2"] for row in common) / 3, "new_equal_bin_mean_MMD2": sum(row["new_MMD2"] for row in common) / 3, "interpretation": "On the same three supported speed bins, selected baseline has lower MMD2 in every bin and in the equal-bin mean. The primary numeric threshold pass does not establish matched-speed style improvement."}, "pose45": {"selected_report": record(old_pose_path), "new_report": record(METRICS / "navigation45/evaluation.json"), "basis": record(basis_path), "saved_center_C_bytes": centers, "inclusive_pose_count": 23996, "reference_refit_performed": False, "generated_sample_fit_performed": False, "coverage": coverage}, "comparison_scope": "Same published public scene/controller/diffusion parameters and frozen metric recipes. Different learned models and initialization shapes consume planner RNG differently; no bitwise noise matching or causal claim. Generated behavior is compared, never required identical.", "metric_execution_pass_is_quality_pass": False}
    with (HERE / "actual_navigation_review.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "common3": report["secondary_common3"], "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
