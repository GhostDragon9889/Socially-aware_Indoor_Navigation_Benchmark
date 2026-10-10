"""Read sealed native4/old4/old15 evidence; do not run metric mathematics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / "docs/style_improvement_20261003"
PREP = DOC / "state_prior_native_metrics_preparation_v1"
RUNS = DOC / "state_prior_native_metrics_actual_runs_v1"
OUT = DOC / "state_prior_native_metrics_v1"
NAMES = ("new4_90", "new4_45", "old4_90", "old4_45", "old15_90", "old15_45")
CHECKS = []
RECORDS = {}


def require(ok, label):
    if not bool(ok):
        raise ValueError(label)
    CHECKS.append(label)


def record(path):
    path = Path(path).absolute()
    key = str(path)
    if key not in RECORDS:
        require(path.is_file() and not path.is_symlink(), "Regular file: " + key)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        RECORDS[key] = {"path": key, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return RECORDS[key]


def exact(row, readonly=False):
    require(record(row["path"]) == row, "Exact recorded bytes: " + row["path"])
    if readonly:
        require(not Path(row["path"]).stat().st_mode & 0o222, "Readonly: " + row["path"])


def read(path):
    record(path)
    return json.loads(Path(path).read_text())


def epoch(epoch):
    exact(epoch["manifest"])
    exact(epoch["archive"])
    source = read(epoch["manifest"]["path"])
    require(source["source_count"] == len(source["files"]) == epoch["source_count"], "Native source count")
    expected = {}
    for row in source["files"]:
        path = Path(row["path"])
        name = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        expected[name] = row
    with tarfile.open(epoch["archive"]["path"], "r:gz") as archive:
        regular = {item.name: item for item in archive.getmembers() if item.isfile()}
        require(set(regular) == set(expected), "Archive exact member set: " + str(epoch["source_count"]))
        for name, row in expected.items():
            value = archive.extractfile(regular[name]).read()
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"], "Archived source: " + name)
    return source


def main():
    catalog_path = PREP / "commands_native_CPU_metrics_v1.json"
    freeze_path = PREP / "source_freeze_v1.json"
    require(record(catalog_path)["sha256"] == "f3bed1b88466ad009fa44fa7a08ce2352115b328781753e14b5203fafd00c067", "Pinned six-metric catalog")
    require(record(freeze_path)["sha256"] == "42da14b2dc48a01157e29f8bcc901e12a345975f94f7785f2dc51c9666ce15db", "Pinned math preparation freeze")
    catalog, freeze = read(catalog_path), read(freeze_path)
    for row in freeze["files"]:
        exact(row, readonly=True)
    exact(freeze["archive"], readonly=True)
    for row in catalog["sources"].values():
        exact(row)
    production = read(catalog["sources"]["production_V11"]["path"])
    source = epoch({"manifest": catalog["sources"]["production_V11"], "archive": production["archive"], "source_count": 183})
    allowed = {row["path"]: row for row in source["files"] + catalog["engine_extra_sources"]}
    for row in allowed.values():
        exact(row)

    first_path = RUNS / "new4_old4_old15_CPU_completion_v1.json"
    require(record(first_path)["sha256"] == "629eec9a1fb24a449500a0f60a026dc39ccabd2c03c6e1059388233608e924f3", "Pinned first closed six-launch receipt")
    first = read(first_path)
    batch = read(RUNS / "native4_and_old15_CPU_completion_v1.json")
    require(first["status"] == batch["status"] == "passed", "Closed batch passed")
    require(set(first["actual_closed_metrics"]) == set(batch["metrics"]) == set(NAMES), "Only authorized six metrics")
    require(len(batch["artifact_records"]) == 42, "42 sealed batch artifacts")
    for row in batch["artifact_records"]:
        exact(row, readonly=True)
    for item in (first, batch):
        require(item["CPU_only"] is True and item["GPU_used"] is False and item["Kit_started"] is False, "Actual CPU-only receipt")
        require(item["new15_metrics_computed"] is False and item["default_promotion"] is False, "No new15 metrics or promotion in this batch")
        exact(item["prepared_catalog"], readonly=True)
        exact(item["prepared_source_freeze"], readonly=True)
    require(batch["model_deserialized"] is False and batch["sealed34_outputs_read"] is False and batch["new15_capture_read"] is False, "Bounded saved-result consumption")
    require(batch["all_failed_contact_idle_samples_retained"] is True and batch["unsupported_bins_remain_unknown"] is True and batch["primary_aggregates_replaced"] is False, "Inclusive metrics retain unknown support")

    metrics, raw_cache, base_arrays = {}, {}, {}
    with np.load(catalog["sources"]["pose45_basis"]["path"], allow_pickle=False) as basis:
        base_arrays = {key: basis[key] for key in basis.files}
    require(set(base_arrays) == {"centers_50", "centers_100"}, "Fixed b25 center names")
    for name in NAMES:
        row = catalog["metrics"][name]
        closed = first["actual_closed_metrics"][name]
        exact(closed["launch"], readonly=True)
        exact(closed["log"], readonly=True)
        launch = read(closed["launch"]["path"])
        require(closed["exit_code"] == launch["exit_code"] == 0 and closed["actual_closed"] is launch["actual_closed"] is True, name + " actual closed0")
        require(launch["status"] == "closed" and launch["actual_started"] is True and launch["GPU_used"] is False, name + " real CPU launch")
        require(launch["argv"] == row["command"] and launch["cwd"] == str(ROOT) and launch["environment"] == catalog["CPU_environment"], name + " fixed argv/env")
        require(launch["prepared_catalog"] == record(catalog_path) and launch["prepared_source_freeze"] == record(freeze_path), name + " launch source binding")
        admission = read(OUT / name / "actual_native_source_admission.json")
        require(admission["status"] == "passed" and admission["CPU_only"] is True and admission["GPU_used"] is False and admission["Kit_started"] is False, name + " actual source admission")
        require(admission["fixed_metric_argv"] == row["metric_argv"] and admission["catalog"] == record(catalog_path), name + " admitted fixed math arguments")
        evidence = admission["actual_native_source_admission"]
        exact(evidence["raw_audit"], readonly=row["kind"] == "new")
        audit_key = evidence["raw_audit"]["path"]
        if audit_key not in raw_cache:
            raw = read(audit_key)
            epoch(raw["source_epoch"])
            require(raw["counts"]["controls"] == 6000 and raw["counts"]["duration_s"] == 200 and raw["counts"]["actors"] == row["actors"], name + " complete raw capture")
            require(set(raw["artifacts"]) == {"result.json", "physical_states.npz", "ground_support.npz", "reference_plans.npz", "deployment_entry.json"}, name + " complete five artifacts")
            for artifact in raw["artifacts"].values():
                exact(artifact)
            with np.load(raw["artifacts"]["physical_states.npz"]["path"], allow_pickle=False) as states:
                reset, episodes = states["reset_mask"], states["episode_ids"]
                terminated, truncated = states["terminated"], states["truncated"]
                require(reset.shape == episodes.shape == terminated.shape == truncated.shape == (6000, row["actors"]), name + " raw counter shape")
                valid = (episodes[1:] == episodes[:-1]) & ~reset[1:]
                raw["peer_valid_intervals"] = int(valid.sum())
                require(int(terminated.sum()) == raw["counts"]["falls"] and int(truncated.sum()) == raw["counts"]["time_limit_resets"], name + " raw failed samples retained")
            raw_cache[audit_key] = raw
        raw = raw_cache[audit_key]
        require(evidence["capture_artifacts"] == raw["artifacts"] and evidence["runtime_source_count"] == row["source_count"] == raw["source_epoch"]["source_count"], name + " own native epoch")
        if row["kind"] == "old":
            require(evidence["native_source_admission"]["current_production_used_as_historical_runtime_proof"] is False, name + " historical source not conflated with V11")
        else:
            require(raw["actual_source_admission"] == evidence["native_source_admission"], name + " new183 actual admission exact")
        ledger = read(OUT / (name + "_CPU_imports.json"))
        require(ledger["status"] == "passed" and ledger["runtime_finalization_observed"] is True and ledger["math_source_epoch"] == 183 and ledger["CPU_only"] is True and ledger["GPU_used"] is False and ledger["native_GPU_import_ledger_claimed"] is False, name + " actual CPU final ledger without GPU claim")
        require(ledger["catalog"] == record(catalog_path), name + " ledger catalog")
        for module in ledger["owned_modules"]:
            require(module["execution_observed"] is True and {k: module[k] for k in ("path", "bytes", "sha256")} == allowed[module["path"]], name + " actual owned module " + module["module"])
        evaluation = read(OUT / name / "evaluation.json")
        require(evaluation["status"] == "passed" and set(evaluation["runs"]) == {row["run_name"]}, name + " one fixed report run")
        run = evaluation["runs"][row["run_name"]]
        require(run["sources"]["states"] == raw["artifacts"]["physical_states.npz"] and run["sources"]["result"] == raw["artifacts"]["result.json"], name + " report uses complete native artifacts")
        reference = OUT / name / "heldout_reference.json"
        require(record(reference)["sha256"] == catalog["sources"]["style_reference"]["sha256"], name + " fixed c274 reference")
        summary = {"runtime_source_count": row["source_count"], "actors": row["actors"], "inclusive_intervals": raw["peer_valid_intervals"], "owned_CPU_modules": len(ledger["owned_modules"])}
        if row["dimension"] == 90:
            generated, aggregate = run["generated"], run["inclusive_speed_conditioned"]
            require(generated["same_episode_transition_count"] == raw["peer_valid_intervals"] == 5999 * row["actors"] and generated["excluded_reset_transitions"] == 0, name + " all same-episode transitions")
            bins = aggregate["speed_bins"]
            edges = [(b["lower_m_s"], b["upper_m_s"]) for b in bins]
            require(edges == list(zip(catalog["metric_rules"]["speed_edges_m_s"][:-1], catalog["metric_rules"]["speed_edges_m_s"][1:])), name + " original speed edges including .1")
            require(sum(b["generated_count"] for b in bins) == raw["peer_valid_intervals"], name + " every interval belongs to an unfiltered speed bin")
            scores = []
            compact = []
            for b in bins:
                score = b["pose_velocity_gaussian_mmd2"]
                supported = min(b["generated_count"], b["reference_count"]) >= 30
                require((score is not None) == supported, name + " strict bin support " + str(b["lower_m_s"]))
                if supported:
                    scores.append(score["value"])
                    require(score["sample_count_each"] == min(256, b["generated_count"], b["reference_count"]), name + " balanced cap")
                compact.append({"lower_m_s": b["lower_m_s"], "upper_m_s": b["upper_m_s"], "generated_count": b["generated_count"], "reference_count": b["reference_count"], "MMD2": score["value"] if supported else None})
            require(len(scores) == aggregate["motion_supported_bins"] and math.isclose(sum(scores) / len(scores), aggregate["equal_supported_bin_mean_pose_velocity_mmd2"], abs_tol=1e-15), name + " saved primary arithmetic")
            require(run["source_navigation_outcomes"]["route_vertices_reached"] == raw["counts"]["route_vertices"], name + " arrivals remain attached")
            summary.update(primary90_MMD2=aggregate["equal_supported_bin_mean_pose_velocity_mmd2"], bins=compact)
        else:
            require(run["raw_source_frames_read_without_failure_filter"] == 6000 and run["source_actor_frames"] == 6000 * row["actors"] and run["poses"]["full_inclusive_pose_count"] == run["same_episode_intervals"] == raw["peer_valid_intervals"], name + " inclusive pose population")
            require(run["excluded_first_frame_actor_samples"] == row["actors"] and run["excluded_cross_reset_interval_endpoints"] == run["cross_reset_intervals_excluded"] == 0 and run["fall_upright_idle_or_outlier_filter_applied"] is False, name + " only original first-frame/reset exclusions")
            fit = evaluation["method"]["fit"]
            require(fit["saved_basis_artifact"] == catalog["sources"]["pose45_basis"] and fit["generated_samples_used_for_fit"] is fit["refit_performed_for_this_new_run"] is False and admission["pose_reference_refit_performed"] is False, name + " fixed reference-only b25 no refit")
            with np.load(OUT / name / "reference_basis.npz", allow_pickle=False) as saved:
                require(set(saved.files) == set(base_arrays), name + " exact saved center set")
                for key, original in base_arrays.items():
                    actual = saved[key]
                    require(actual.dtype == original.dtype and actual.shape == original.shape and actual.tobytes(order="C") == original.tobytes(order="C"), name + " center dtype/shape/Cbytes " + key)
                    k = key.removeprefix("centers_")
                    require(fit["centers"][k]["float64_sha256"] == hashlib.sha256(actual.tobytes(order="C")).hexdigest(), name + " report center SHA")
            for k, clusters in run["poses"]["clusters"].items():
                counts = clusters["assignment_counts"]
                occupied = sum(value > 0 for value in counts)
                entropy = -sum((v / sum(counts)) * math.log(v / sum(counts)) for v in counts if v) / math.log(int(k))
                require(len(counts) == int(k) and sum(counts) == clusters["samples_assigned"] == raw["peer_valid_intervals"] and occupied == clusters["occupied_clusters"] and occupied / int(k) == clusters["coverage_occupied_over_K"] and math.isclose(entropy, clusters["normalized_entropy"], abs_tol=1e-14), name + " all cluster assignments and entropy " + k)
            summary.update(pose45_MMD2=run["poses"]["pose_only_biased_gaussian_MMD2"]["value"], coverage={k: v["coverage_occupied_over_K"] for k, v in run["poses"]["clusters"].items()})
        metrics[name] = summary

    comparison = read(OUT / "comparison4/comparison.json")
    exact(batch["comparison4"]["record"], readonly=True)
    common = comparison["secondary_common_speed_bins"]
    derived = []
    for old, new in zip(metrics["old4_90"]["bins"], metrics["new4_90"]["bins"]):
        if old["MMD2"] is not None and new["MMD2"] is not None:
            derived.append({"lower_m_s": old["lower_m_s"], "upper_m_s": old["upper_m_s"], "old_MMD2": old["MMD2"], "new_MMD2": new["MMD2"], "old_generated_count": old["generated_count"], "new_generated_count": new["generated_count"]})
    require(common["bins"] == derived and len(derived) == 3, "Actual common three bins exact")
    for side in ("old", "new"):
        require(math.isclose(sum(v[side + "_MMD2"] for v in derived) / 3, common[side + "_equal_bin_mean_MMD2"], abs_tol=1e-15), "Common-bin secondary arithmetic " + side)
    require(comparison["primary90"]["primary_aggregates_replaced"] is False and comparison["all_failed_contact_idle_samples_retained"] is True, "Secondary comparison preserves primary inclusive results")
    for flag in ("same_RNG_or_bitwise_planning_noise_claimed", "causal_single_factor_gain_claimed", "overall_style_gain_claimed", "default_promotion"):
        require(comparison[flag] is False, "No unsupported comparison claim: " + flag)
    for side, metric in (("old", "old4"), ("new", "new4")):
        require(comparison["primary90"][side] == read(OUT / (metric + "_90") / "evaluation.json")["runs"][catalog["metrics"][metric + "_90"]["run_name"]]["inclusive_speed_conditioned"], "Comparison primary exact " + side)
        require(comparison["pose45"][side] == read(OUT / (metric + "_45") / "evaluation.json")["runs"][catalog["metrics"][metric + "_45"]["run_name"]]["poses"], "Comparison pose45 exact " + side)
    result = {"schema": "public_state_prior.actual_native_six_metric_independent_peer_QC.v1", "status": "passed", "CPU_only": True, "GPU_used": False, "Kit_started": False, "model_deserialized": False, "metric_mathematics_rerun": False, "new15_metrics_read": False, "sealed34_read": False, "source": record(Path(__file__)), "closed_batch": record(first_path), "completed_artifact_batch": record(RUNS / "native4_and_old15_CPU_completion_v1.json"), "prepared_catalog": record(catalog_path), "prepared_source_freeze": record(freeze_path), "metrics": metrics, "common_three_speed_bins_secondary": common, "native4_route_vertices_old_new": [sum(comparison["captures"][side]["counts"]["route_vertices"]) for side in ("old", "new")], "native4_cycles_old_new": [sum(comparison["captures"][side]["counts"]["complete_cycles"]) for side in ("old", "new")], "limitations": ["Native4 primary aggregates use four supported bins each, but different supported bins.", "New native4 style proxies improve here while route vertices decline 57 to 44 and complete cycles decline 12 to 8.", "Controller shape and planner RNG consumption differ; no bitwise planning-noise or single-factor causal comparison.", "Metric/source correctness does not prove naturalness or overall style improvement."], "checks_passed": len(CHECKS), "checks": CHECKS, "evidence_records": list(RECORDS.values())}
    target = HERE / "actual_six_metric_QC_v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(CHECKS), "report": str(target)}))


if __name__ == "__main__":
    main()
