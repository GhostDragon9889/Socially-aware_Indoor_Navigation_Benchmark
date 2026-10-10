"""Check saved native4/15 comparisons without recomputing style metrics."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from verify_six_metrics import CHECKS, DOC, HERE, OUT, RECORDS, ROOT, RUNS, epoch, exact, read, record, require


def main():
    earlier = read(HERE / "actual_six_metric_QC_v1.json")
    require(earlier["status"] == "passed" and earlier["checks_passed"] == 1742, "Completed independent six-metric QC")
    for row in earlier["evidence_records"]:
        exact(row)
    prep = DOC / "state_prior_native_metrics_preparation_v2"
    out = DOC / "state_prior_native_metrics_v2"
    catalog_path, freeze_path = prep / "commands_native15_CPU_metrics_v2.json", prep / "source_freeze_v2.json"
    require(record(catalog_path)["sha256"] == "5870b6638b14668976c09d2f30a13003b328e4732e540a3c5107737c50dcb000", "Pinned native15 metric catalog")
    require(record(freeze_path)["sha256"] == "6eb5e82075f332ae513c2e32c83a5fb57b7c5fd9473eec5e853df38ac706fc81", "Pinned native15 math source freeze")
    catalog, freeze = read(catalog_path), read(freeze_path)
    for row in freeze["files"]:
        exact(row, readonly=True)
    exact(freeze["archive"], readonly=True)
    for row in catalog["sources"].values():
        exact(row)
    production = read(catalog["sources"]["production_V11"]["path"])
    allowed = {r["path"]: r for r in production["files"] + catalog["engine_extra_sources"]}
    for row in allowed.values():
        exact(row)
    closed = read(RUNS / "native15_v2_first_CPU_completion.json")
    require(closed["status"] == "passed" and closed["CPU_only"] is True and closed["GPU_used"] is closed["Kit_started"] is False, "Native15 actual CPU completion")
    require(closed["catalog"] == record(catalog_path) and closed["prepared_source_freeze"] == record(freeze_path), "Actual native15 receipt binds v2")
    require(set(closed["actual_closed_phases"]) == {"new15_90_v2", "new15_45_v2", "comparison15_v2"}, "Exactly three closed native15 phases")
    for phase, item in closed["actual_closed_phases"].items():
        exact(item["launch"], readonly=True)
        exact(item["log"], readonly=True)
        launch = read(item["launch"]["path"])
        expected = catalog["comparison"]["command"] if phase == "comparison15_v2" else catalog["metrics"]["90" if "90" in phase else "45"]["command"]
        require(item["actual_closed"] is launch["actual_closed"] is launch["actual_started"] is True and item["exit_code"] == launch["exit_code"] == 0 and launch["status"] == "closed", phase + " actual closed0")
        require(launch["argv"] == expected and launch["cwd"] == str(ROOT) and launch["environment"] == catalog["CPU_environment"] and launch["GPU_used"] is False, phase + " fixed CPU launch")
        require(launch["catalog"] == record(catalog_path) and launch["prepared_source_freeze"] == record(freeze_path), phase + " frozen source binding")
    raw = read(catalog["sources"]["actual_native15_raw_audit"]["path"])
    require(raw["status"] == "raw_sensor_source_contracts_passed_bounded_zero_fall_observed" and raw["counts"]["actors"] == 15 and raw["counts"]["controls"] == 6000, "Actual complete native15 raw audit")
    epoch(raw["source_epoch"])
    for row in raw["artifacts"].values():
        exact(row)
    outer = read(catalog["sources"]["raw15_revision_registration"]["path"])
    require(raw["representation_adapter"] == catalog["sources"]["raw15_revision_driver"] and raw["representation_revision_registration"] == outer and raw["immutable_math_core_freeze"] == catalog["sources"]["native_reader_freeze"] and raw["first_actual_CPU_failure"] == outer["first_actual_failure"], "Actual15 canonical JSON adapter identity")
    require(raw["compiled_v1_v2_routes_math_changed"] is False and raw["proof_tuple_list_difference_only"] is True and raw["adapter_proof"] == outer["actual_original_proof"], "Representation-only native15 proof")
    new = {}
    for dimension in (90, 45):
        name = f"new15_{dimension}"
        admission = read(out / name / "actual_native_source_admission.json")
        require(admission["status"] == "passed" and admission["CPU_only"] is True and admission["GPU_used"] is admission["Kit_started"] is admission["sealed34_outputs_read"] is False, name + " actual CPU admission")
        require(admission["catalog"] == record(catalog_path) and admission["fixed_metric_argv"] == catalog["metrics"][str(dimension)]["metric_argv"], name + " exact metric source/arguments")
        source = admission["actual_native_source_admission"]
        require(source["capture_artifacts"] == raw["artifacts"] and source["native_source_admission"] == raw["actual_source_admission"] and source["runtime_source_count"] == 183, name + " exact actual native183 source")
        revision = source["JSON_representation_revision"]
        require(revision["freeze"] == catalog["sources"]["raw15_revision_freeze"] and revision["driver"] == catalog["sources"]["raw15_revision_driver"] and revision["outer_registration"] == catalog["sources"]["raw15_revision_registration"] and revision["compiled_math_changed"] is False and revision["proof_tuple_list_difference_only"] is True, name + " no blind v1 representation admission")
        ledger = read(out / (name + "_CPU_imports.json"))
        require(ledger["status"] == "passed" and ledger["runtime_finalization_observed"] is True and ledger["CPU_only"] is True and ledger["GPU_used"] is False and ledger["native_GPU_import_ledger_claimed"] is False and ledger["catalog"] == record(catalog_path), name + " actual finalized CPU source ledger")
        for module in ledger["owned_modules"]:
            require(module["execution_observed"] is True and {k: module[k] for k in ("path", "bytes", "sha256")} == allowed[module["path"]], name + " owned module " + module["module"])
        report = read(out / name / "evaluation.json")
        require(report["status"] == "passed" and set(report["runs"]) == {"new_native15"}, name + " actual saved report")
        run = report["runs"]["new_native15"]
        require(run["sources"]["states"] == raw["artifacts"]["physical_states.npz"] and run["sources"]["result"] == raw["artifacts"]["result.json"], name + " full capture identity")
        require(record(out / name / "heldout_reference.json")["sha256"] == catalog["sources"]["style_reference"]["sha256"], name + " c274 reference exact")
        if dimension == 90:
            require(run["generated"]["same_episode_transition_count"] == 89985 and run["generated"]["excluded_reset_transitions"] == 0, "All native15 89985 intervals in90D")
            aggregate = run["inclusive_speed_conditioned"]
            require(sum(v["generated_count"] for v in aggregate["speed_bins"]) == 89985, "Native15 all speed bins retain full population")
            require(aggregate["minimum_metric_samples_per_bin"] == 30 and aggregate["maximum_balanced_motion_samples_per_bin"] == 256, "Original native15 bin support/cap")
        else:
            require(run["poses"]["full_inclusive_pose_count"] == run["same_episode_intervals"] == run["included_pose_valid_interval_endpoints"] == 89985 and run["raw_source_frames_read_without_failure_filter"] == 6000 and run["source_actor_frames"] == 90000, "All native15 pose population")
            require(run["excluded_first_frame_actor_samples"] == 15 and run["excluded_cross_reset_interval_endpoints"] == 0 and run["fall_upright_idle_or_outlier_filter_applied"] is False, "Native15 only original first-frame/reset exclusions")
            fit = report["method"]["fit"]
            require(fit["generated_samples_used_for_fit"] is fit["refit_performed_for_this_new_run"] is admission["reference_refit_performed"] is False, "Native15 saved b25 no fit")
            with np.load(catalog["sources"]["pose45_basis"]["path"], allow_pickle=False) as original, np.load(out / name / "reference_basis.npz", allow_pickle=False) as saved:
                require(set(saved.files) == set(original.files), "Native15 center names unchanged")
                for key in original.files:
                    a, b = original[key], saved[key]
                    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C"), "Native15 b25 dtype/shape/Cbytes " + key)
            for k, cluster in run["poses"]["clusters"].items():
                counts = cluster["assignment_counts"]
                require(sum(counts) == cluster["samples_assigned"] == 89985 and sum(v > 0 for v in counts) == cluster["occupied_clusters"] and cluster["occupied_clusters"] / int(k) == cluster["coverage_occupied_over_K"], "Native15 inclusive assignments " + k)
        new[dimension] = run
    p15 = out / "comparison15/comparison.json"
    require(record(p15)["sha256"] == "2ce702fc936542a980df93575d95ba394bdd90508973864f2d68b500ec59e75e", "Pinned actual comparison15")
    comparisons = {4: read(OUT / "comparison4/comparison.json"), 15: read(p15)}
    old15 = {dimension: read(OUT / f"old15_{dimension}/evaluation.json")["runs"]["old_native15"] for dimension in (90, 45)}
    require(comparisons[15]["primary90"]["old"] == old15[90]["inclusive_speed_conditioned"] and comparisons[15]["primary90"]["new"] == new[90]["inclusive_speed_conditioned"] and comparisons[15]["pose45"]["old"] == old15[45]["poses"] and comparisons[15]["pose45"]["new"] == new[45]["poses"], "Comparison15 exact old/new saved values")
    summary = {}
    for actors, comparison in comparisons.items():
        require(comparison["status"] == "passed" and comparison["CPU_only"] is True and comparison["GPU_used"] is False and comparison["all_failed_contact_idle_samples_retained"] is True and comparison["primary90"]["primary_aggregates_replaced"] is False, "Actual comparison inclusive and primary preserved " + str(actors))
        artifacts = comparison["bound_metric_artifacts"]
        if actors == 15:
            for row in artifacts:
                exact(row, readonly=True)
        else:
            for side in artifacts.values():
                for dimension in side.values():
                    for row in dimension.values():
                        exact(row, readonly=True)
        derived = []
        supported = {}
        primary = comparison["primary90"]
        for side in ("old", "new"):
            bins = primary[side]["speed_bins"]
            require([(b["lower_m_s"], b["upper_m_s"]) for b in bins] == [(0, .1), (.1, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, None)], "Original speed edges " + side + str(actors))
            supported[side] = []
            for b in bins:
                ok = min(b["generated_count"], b["reference_count"]) >= 30
                require((b["pose_velocity_gaussian_mmd2"] is not None) == ok, "Unknown bins not zero " + side + str(actors) + str(b["lower_m_s"]))
                if ok:
                    supported[side].append([b["lower_m_s"], b["upper_m_s"]])
            scores = [b["pose_velocity_gaussian_mmd2"]["value"] for b in bins if b["pose_velocity_gaussian_mmd2"]]
            require(len(scores) == primary[side]["motion_supported_bins"] and math.isclose(sum(scores) / len(scores), primary[side]["equal_supported_bin_mean_pose_velocity_mmd2"], abs_tol=1e-15), "Original primary saved arithmetic " + side + str(actors))
        for old, current in zip(primary["old"]["speed_bins"], primary["new"]["speed_bins"]):
            if old["pose_velocity_gaussian_mmd2"] and current["pose_velocity_gaussian_mmd2"]:
                derived.append({"lower_m_s": old["lower_m_s"], "upper_m_s": old["upper_m_s"], "old_MMD2": old["pose_velocity_gaussian_mmd2"]["value"], "new_MMD2": current["pose_velocity_gaussian_mmd2"]["value"], "old_generated_count": old["generated_count"], "new_generated_count": current["generated_count"]})
        common = comparison["secondary_common_speed_bins"]
        require(common["bins"] == derived, "Exact shared bins " + str(actors))
        for side in ("old", "new"):
            require(math.isclose(sum(v[side + "_MMD2"] for v in derived) / len(derived), common[side + "_equal_bin_mean_MMD2"], abs_tol=1e-15), "Shared-bin secondary mean " + side + str(actors))
        counts = {side: comparison["captures"][side]["counts"] if actors == 4 else comparison["counts"][side] for side in ("old", "new")}
        summary[str(actors)] = {"primary90_old_new": [primary[side]["equal_supported_bin_mean_pose_velocity_mmd2"] for side in ("old", "new")], "supported_bins_old_new": supported, "secondary_common_bins": common, "pose45_MMD2_old_new": [comparison["pose45"][side]["pose_only_biased_gaussian_MMD2"]["value"] for side in ("old", "new")], "coverage50_old_new": [comparison["pose45"][side]["clusters"]["50"]["coverage_occupied_over_K"] for side in ("old", "new")], "coverage100_old_new": [comparison["pose45"][side]["clusters"]["100"]["coverage_occupied_over_K"] for side in ("old", "new")], "route_vertices_old_new": [sum(counts[side]["route_vertices"]) for side in ("old", "new")], "complete_cycles_old_new": [sum(counts[side]["complete_cycles"]) for side in ("old", "new")], "raw_counts": counts}
    result = {"schema": "public_state_prior.actual_native4_native15_style_comparison_independent_QC.v1", "status": "passed", "CPU_only": True, "GPU_used": False, "Kit_started": False, "model_deserialized": False, "metric_mathematics_rerun": False, "sealed34_read": False, "source": record(Path(__file__)), "earlier_six_metric_QC": record(HERE / "actual_six_metric_QC_v1.json"), "native15_actual_closed_receipt": record(RUNS / "native15_v2_first_CPU_completion.json"), "native15_raw_audit": catalog["sources"]["actual_native15_raw_audit"], "comparisons": {"4": record(OUT / "comparison4/comparison.json"), "15": record(p15)}, "summary": summary, "checks_passed": len(CHECKS), "limitations": ["Primary90 support differs: four bins versus four different bins for native4, five old versus four new bins for native15. Common-bin means are secondary.", "Native4 coverage improves .34/.31 to .58/.47, but route vertices/cycles decline57/12 to44/8.", "Native15 coverage K50 declines .76 to .70; K100 remains .61. Lower MMD does not imply increased pose diversity.", "No bitwise planner RNG matching, single-factor causal gain, naturalness, or default-promotion claim."], "evidence_records": list(RECORDS.values())}
    path = HERE / "actual_native4_native15_comparison_QC_v1.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(CHECKS), "report": str(path), "summary": {k: {x: v for x, v in value.items() if x not in ("raw_counts", "supported_bins_old_new", "secondary_common_bins")} for k, value in summary.items()}}))


if __name__ == "__main__":
    main()
