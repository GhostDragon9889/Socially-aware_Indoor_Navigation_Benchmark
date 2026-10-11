#!/usr/bin/env python3
"""Supplement completed shared Social measurements with stricter CPU admission.

Checks saved JSON, exits and hashes only. Unequal 90D samples remain unknown;
their physical, tracking and frozen pose45 measurements remain independently usable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts import compare_shared_social_evidence as original  # noqa: E402

FROZEN = {
    "scripts/evaluate_pedestrian_style.py": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "CrowdSim/humanoid/style_evaluation.py": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
    "scripts/evaluate_shared_social_style.py": "d3296011b0e08286ae109daea6a518c95f5f9fa9f24a474a20555cd002baa47c",
    "docs/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json": "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564",
    "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py": "208ff775d765b7d046e7e7e8c331dbb2b8d3bfdc4c1567c1df499a584af30bfa",
    "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz": "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb",
    "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/evaluation.json": "9478afe6b16540f7624867e1d736f30f78399c84177c12d6c9b616bc496c3780",
}
TRACKING_STAGES = ("lifecycle_smoke", "full22", "style")
POST_STAGES = ("actor_CPU_inference", "tracking_style", "navigation_smoke", "navigation200s", "navigation_CPU_array_audit")
MEASUREMENTS = ("tracking", "style90", "pose45", "navigation", "navigation_CPU_audit")


def identity(value):
    """Strip capture/stat metadata while retaining the actual file identity."""
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def successful_stages(pipeline, names):
    original.require(pipeline["status"] == "passed_execution"
                     and [row["name"] for row in pipeline["steps"]] == list(names),
                     "Missing, duplicate or reordered pipeline stages")
    original.require(all(row["status"] == "passed" and type(row["actual_exit_code"]) is int
                         and row["actual_exit_code"] == 0 for row in pipeline["steps"]),
                     "A required child lacks an actual successful exit")
    stages = {row["name"]: row for row in pipeline["steps"]}
    for name, row in stages.items():
        expected = "passed_execution" if name == "tracking_style" else (
            "passed_lifecycle_smoke" if name == "lifecycle_smoke" else "passed")
        original.require(original.bound(row["result"])["status"] == expected,
                         "Successful child exit disagrees with its result: " + name)
    return stages


def tracking_lineage(pipeline, measurements, bundle):
    stages = successful_stages(pipeline, TRACKING_STAGES)
    original.require(stages["full22"]["result"] == measurements["tracking"]
                     and stages["style"]["result"] == measurements["style90"],
                     "Tracking/style measurements differ from successful child outputs")
    original.require(bundle["style90"]["frozen_pose45_result"] == measurements["pose45"],
                     "Pose45 measurement differs from the successful style output")
    admission = bundle["tracking"]["training_execution_admission"]
    original.require(pipeline["training_receipt"] == admission["receipt"],
                     "Tracking pipeline and measurement admit different training receipts")
    if "training_execution_admission" in pipeline:
        original.require(pipeline["training_execution_admission"] == admission,
                         "Tracking child training admission differs from pipeline")


def post_lineage(receipt, pipeline, bundle):
    original.require(receipt["measurements"] == pipeline["measurements"],
                     "Owner measurements differ from the completed pipeline")
    stages = successful_stages(pipeline, POST_STAGES)
    measurements = receipt["measurements"]
    for name, key in (("navigation200s", "navigation"), ("navigation_CPU_array_audit", "navigation_CPU_audit")):
        original.require(stages[name]["result"] == measurements[key],
                         "Measurement differs from successful child output: " + key)
    tracking_pipeline = original.bound(stages["tracking_style"]["result"])
    tracking_lineage(tracking_pipeline, measurements, bundle)
    admission = pipeline["training_execution_admission"]
    original.require(admission == bundle["tracking"]["training_execution_admission"]
                     == original.bound(stages["actor_CPU_inference"]["result"])["training_execution_admission"],
                     "Post-training children admit different training/checkpoint identities")


def admit_post(path):
    receipt = original.read(path)
    bundle = original.admit_candidate(path)
    post_lineage(receipt, original.bound(receipt["pipeline"]), bundle)
    return bundle, receipt["measurements"]


def admit_legacy(tracking_path, navigation_path, audit_path):
    bundle = original.admit_baseline(tracking_path, navigation_path, audit_path)
    t, n, a = map(original.read, (tracking_path, navigation_path, audit_path))
    measurements = {key: t[key] for key in ("tracking", "style90", "pose45")}
    measurements.update(navigation=n["result"], navigation_CPU_audit=a["audit"])
    tracking_lineage(original.bound(t["pipeline"]), measurements, bundle)
    original.require(bundle["navigation_CPU_audit"]["execution_receipt"] == original.record(navigation_path),
                     "Navigation audit consumed another execution receipt")
    return bundle, measurements


def normalized(value, excluded):
    return {key: item for key, item in value.items() if key not in excluded}


def tracking_protocol(before, after):
    for key in ("physics", "reference_velocity_contract"):
        original.require(before[key] == after[key], "Tracking protocol differs: " + key)
    for key, excluded in (("arguments", {"checkpoint", "checkpoint_sha256", "output"}),
                          ("preflight", {"checkpoint", "checkpoint_sha256_recomputed"})):
        original.require(normalized(before[key], excluded) == normalized(after[key], excluded),
                         "Tracking protocol differs: " + key)


def frozen_source(value, relative):
    from social_navigation.artifact_identity import artifact_path
    original.require(artifact_path(value["path"], root=ROOT).resolve() == (ROOT / relative).resolve()
                     and value["sha256"] == FROZEN[relative],
                     "Frozen style source differs: " + relative)
    return identity(value)


def bundle_sources(bundle, measurements):
    """Bind each model's style, pose and saved arrays to its own tracking run."""
    t, s, p, n, a = (bundle[key] for key in MEASUREMENTS)
    bridge, run, prun = s["shared_social_reader_bridge"], s["runs"]["shared2000"], p["runs"]["shared2000"]
    sources = [frozen_source(s["sources"]["implementation"], "scripts/evaluate_pedestrian_style.py"),
               frozen_source(s["sources"]["module"], "CrowdSim/humanoid/style_evaluation.py"),
               frozen_source(bridge["implementation"], "scripts/evaluate_shared_social_style.py")]
    original.require(bridge["original_style_tool"] == s["sources"]["implementation"]
                     and bridge["reference"] == p["reference"], "Style bridge source/reference binding differs")
    for key, relative in (("reference", "docs/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json"),
                          ("basis", "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz"),
                          ("original_basis_evaluation", "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/evaluation.json"),
                          ("metric_implementation", "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py")):
        sources.append(frozen_source(p[key], relative))
    original.require(s["reference"] == original.bound(p["reference"]), "Style and pose references differ")
    original.require(bridge["inputs"]["shared2000"] == run["sources"]["result"]
                     == prun["source"] == measurements["tracking"], "Style/pose consumed another tracking result")
    original.require(s["frozen_pose45_result"] == measurements["pose45"], "Style names another pose45 artifact")
    capture = run["capture_validation"]
    original.require(capture == prun["capture_validation"]
                     and capture["sources"] == run["sources"]["tracking_captures"],
                     "Style and pose capture populations differ")
    original.require(t["cleanup_succeeded"] is True and t["cleanup_errors"] == [], "Tracking cleanup did not pass")
    sequences = t["sequences"]
    original.require(len(sequences) == len(capture["sources"]) == 22
                     and len({row["motion_id"] for row in sequences}) == 22,
                     "Tracking must contain 22 distinct complete motions")
    for row, saved in zip(sequences, capture["sources"], strict=True):
        original.require(row["complete_clip"] is True and row["expected_control_steps"] == row["observed_metric_steps"]
                         == saved["metric_frames"] and row["motion_id"] == saved["motion_id"]
                         and row["source_path"] == saved["source_path"]
                         and Path(row["capture"]).resolve() == Path(saved["path"]).resolve()
                         and row["capture_sha256"] == saved["sha256"]
                         and saved["fall_metric_frames_kept"] == len(row["fall_capture_frames"]),
                         "Tracking sequence and saved style capture disagree")
        sources.append(identity(saved))
    original.require(sum(row["metric_frames"] for row in capture["sources"]) == t["aggregate"]["observed_metric_steps"]
                     and sum(row["fall_metric_frames_kept"] for row in capture["sources"]) == capture["fall_metric_frames_kept"],
                     "Capture aggregate omitted frames or failures")
    task = run["task_contract"]
    original.require(task["benchmark_primary_aggregate_unchanged"] == t["aggregate"]
                     and task["reference_velocity_contract"] == t["reference_velocity_contract"],
                     "Style tracking task contract differs")
    original.require(run["reference_not_in_declared_training"]["original_source_intersection"] == 0,
                     "Style reference overlaps declared training sources")
    checkpoint = t["training_execution_admission"]["checkpoint"]
    original.require(t["training_execution_admission"]["terminal_exit_code"] == 0
                     and run["sources"]["models_resolved_from_evaluation_arguments"]["controller"] == checkpoint
                     and t["arguments"]["checkpoint"] == checkpoint["path"]
                     and t["arguments"]["checkpoint_sha256"] == checkpoint["sha256"]
                     and n["arguments"]["resume"] == checkpoint["path"]
                     and identity(n["evaluation_actor"]["checkpoint"]) == checkpoint,
                     "Tracking/style/navigation used different actor checkpoints")
    sources.extend([checkpoint, run["sources"]["declared_training_manifest"]])
    original.require(a["result"] == measurements["navigation"], "Navigation audit consumed another result")
    nav_receipt = original.bound(a["execution_receipt"])
    original.require(nav_receipt["actual_exit_code"] == 0 and nav_receipt["result"] == measurements["navigation"],
                     "Navigation array audit lacks a matching successful execution receipt")
    expected_arrays = {Path(measurements["navigation"]["path"]).parent / name
                       for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")}
    original.require(len(a["arrays"]) == 3 and {Path(row["path"]) for row in a["arrays"]} == expected_arrays,
                     "Navigation audit arrays belong to another run")
    sources.extend([a["execution_receipt"], *a["arrays"]])
    return sources


def matched_speed_bins(before, after):
    """Retain unknown bins without admitting sample-size or reference-subset drift."""
    safe = [deepcopy(before), deepcopy(after)]
    reasons = []
    original.require(len(before["speed_bins"]) == len(after["speed_bins"]), "Speed-bin count differs")
    for index, (old, new) in enumerate(zip(before["speed_bins"], after["speed_bins"], strict=True)):
        x, y = old["pose_velocity_gaussian_mmd2"], new["pose_velocity_gaussian_mmd2"]
        reason = []
        if x is None or y is None:
            reason.append("measurement_unsupported")
        else:
            for key in ("features", "feature_width", "velocity_time_scale_s", "sigma", "estimator"):
                original.require(x[key] == y[key], "90D estimator differs: " + key)
            if x["sample_count_each"] != y["sample_count_each"]:
                reason.append("sample_counts_differ")
            if x.get("reference_sample_indices") != y.get("reference_sample_indices"):
                reason.append("reference_subsets_differ")
            for metric in (x, y):
                count, indices = metric["sample_count_each"], metric.get("reference_sample_indices")
                if (type(count) is not int or count < before["minimum_metric_samples_per_bin"]
                        or not isinstance(indices, list) or len(indices) != count
                        or any(type(v) is not int or v < 0 for v in indices)
                        or len(set(indices)) != count):
                    reason.append("invalid_reference_sampling_evidence")
                    break
        reasons.append(reason)
        if reason:
            for population in safe:
                population["speed_bins"][index]["pose_velocity_gaussian_mmd2"] = None
    result = original.common_speed_bins(*safe)
    for row, old, new, reason in zip(result["bins"], before["speed_bins"], after["speed_bins"], reasons, strict=True):
        row["directional_comparison_eligible"] = not reason
        row["eligibility"] = "eligible" if not reason else "unknown"
        row["unknown_reasons"] = reason
        row["jointly_supported"] = all(value["pose_velocity_gaussian_mmd2"] is not None for value in (old, new))
        row["actual_balanced_samples_each"] = [None if value["pose_velocity_gaussian_mmd2"] is None
                                               else value["pose_velocity_gaussian_mmd2"]["sample_count_each"] for value in (old, new)]
    result["qualification"] = "Directional 90D comparisons use only matched sample counts and identical saved reference indices. Unknown bins contribute no score; other axes remain separate."
    return result


def supplemental_comparison(before, after):
    tracking_protocol(before["tracking"], after["tracking"])
    bins = matched_speed_bins(*[value["style90"]["runs"]["shared2000"]["inclusive_speed_conditioned"] for value in (before, after)])
    # Reuse the original independent physical/pose checks without calculating an
    # unequal-sampling directional score, then attach the qualified 90D result.
    safe = [deepcopy(before), deepcopy(after)]
    for value in safe:
        for row in value["style90"]["runs"]["shared2000"]["inclusive_speed_conditioned"]["speed_bins"]:
            row["pose_velocity_gaussian_mmd2"] = None
    result = original.compare_metrics(*safe)
    result["style"]["common_speed_bin_secondary"] = bins
    return result


def unique_artifacts(artifacts):
    """Do not let a later identity hide inconsistent evidence for the same file."""
    by_path = {}
    for item in artifacts:
        original.require(item["path"] not in by_path or by_path[item["path"]] == item,
                         "Conflicting identities for admitted file: " + item["path"])
        by_path[item["path"]] = item
    return list(by_path.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-post-receipt", required=True, type=Path)
    parser.add_argument("--baseline-post-receipt", type=Path)
    for name in ("baseline-tracking-receipt", "baseline-navigation-receipt", "baseline-audit-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    original.require(not args.output.exists(), "Preserve previous admission reports; use a new output")
    legacy = [args.baseline_tracking_receipt, args.baseline_navigation_receipt, args.baseline_audit_receipt]
    if args.baseline_post_receipt is not None:
        if any(legacy):
            parser.error("Use one baseline post receipt or all three legacy baseline receipts")
        baseline_paths = [args.baseline_post_receipt]
    else:
        if not all(legacy):
            parser.error("Require a baseline post receipt or all three legacy baseline receipts")
        baseline_paths = legacy
    receipt_records = [original.record(path) for path in [*baseline_paths, args.candidate_post_receipt]]
    before, before_ids = admit_post(args.baseline_post_receipt) if args.baseline_post_receipt else admit_legacy(*legacy)
    after, after_ids = admit_post(args.candidate_post_receipt)
    artifacts = [*before_ids.values(), *after_ids.values(), *bundle_sources(before, before_ids), *bundle_sources(after, after_ids)]
    artifacts = unique_artifacts(artifacts)
    for item in artifacts:
        original.require(original.record(item["path"]) == item, "Admitted source/capture changed: " + item["path"])
    comparison = supplemental_comparison(before, after)
    original.require(receipt_records == [original.record(path) for path in [*baseline_paths, args.candidate_post_receipt]],
                     "Execution receipts changed during admission")
    for item in artifacts:
        original.require(original.record(item["path"]) == item, "Admitted source/capture changed during comparison: " + item["path"])
    original.require("torch" not in sys.modules, "Supplemental admission must remain stdlib-only")
    report = {"schema": "shared_social.supplemental_evidence_admission.v2", "status": "passed_supplemental_admission",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "reader": original.record(__file__),
              "original_reader": original.record(original.__file__), "exit_receipts": receipt_records,
              "measurements": {"baseline": before_ids, "candidate": after_ids}, "verified_artifacts": artifacts,
              "comparison": comparison, "checks": {"child_exit_measurement_lineage": True,
              "tracking_physics_arguments_preflight_velocity": True, "frozen_style_implementation": True,
              "own_tracking_capture_bindings": True, "saved_capture_hashes_verified": True,
              "raw_array_metrics_recomputed": False}, "Torch_imported": False, "SDK_started": False,
              "paper_effect_achieved": False, "default_model_promoted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "output": original.record(args.output),
                      "eligible_common_90D_bins": comparison["style"]["common_speed_bin_secondary"]["common_supported_bin_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
