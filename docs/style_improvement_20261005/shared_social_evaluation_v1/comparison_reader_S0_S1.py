#!/usr/bin/env python3
"""Compare completed shared-AMP measurements on fixed independent axes.

Uses only JSON and SHA256, with real exit receipts required before admission.
Does not run a simulator, fit references, filter failures or certify a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def bound(identity):
    if record(identity["path"]) != identity:
        raise ValueError("Recorded artifact changed: " + identity["path"])
    return read(identity["path"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def change(before, after, lower_is_better=True):
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in (before, after)),
            "Comparison requires measured finite numbers")
    delta = after - before
    return {"baseline": before, "candidate": after, "candidate_minus_baseline": delta,
            "lower_is_better": lower_is_better,
            "improved": delta < 0 if lower_is_better else delta > 0}


def common_speed_bins(before, after):
    """Keep unsupported bins unknown and retain each side's actual population."""
    require(before["minimum_metric_samples_per_bin"] == after["minimum_metric_samples_per_bin"]
            and before["maximum_balanced_motion_samples_per_bin"] == after["maximum_balanced_motion_samples_per_bin"],
            "Speed-bin sampling rules differ")
    rows = []
    left, right = [], []
    a, b = before["speed_bins"], after["speed_bins"]
    require(len(a) == len(b), "Speed-bin count differs")
    for old, new in zip(a, b, strict=True):
        edges = (old["lower_m_s"], old["upper_m_s"])
        require(edges == (new["lower_m_s"], new["upper_m_s"]), "Speed-bin edges differ")
        x, y = old["pose_velocity_gaussian_mmd2"], new["pose_velocity_gaussian_mmd2"]
        row = {"lower_m_s": edges[0], "upper_m_s": edges[1],
               "baseline_generated_count": old["generated_count"],
               "candidate_generated_count": new["generated_count"],
               "reference_count": old["reference_count"],
               "jointly_supported": x is not None and y is not None,
               "MMD2_comparison": None}
        require(old["reference_count"] == new["reference_count"], "Reference speed population differs")
        if row["jointly_supported"]:
            for key in ("features", "feature_width", "velocity_time_scale_s", "sigma", "estimator"):
                require(x[key] == y[key], "90D estimator differs: " + key)
            row["MMD2_comparison"] = change(x["value"], y["value"])
            row["actual_balanced_samples_each"] = [x["sample_count_each"], y["sample_count_each"]]
            left.append(x["value"])
            right.append(y["value"])
        rows.append(row)
    return {"bins": rows, "common_supported_bin_count": len(left),
            "equal_common_bin_mean_MMD2": change(sum(left) / len(left), sum(right) / len(right)) if left else None,
            "qualification": "Common speed edges and estimator; sample counts are observed, not forced equal across models. Unsupported bins are unknown."}


def compare_metrics(before, after):
    """Admitted bundles must contain tracking, style90, pose45, nav and nav audit."""
    tracking = []
    style = []
    poses = []
    for document in (before, after):
        t, s, p = (document[key] for key in ("tracking", "style90", "pose45"))
        require(t["status"] == s["status"] == p["status"] == "passed", "A measurement is incomplete")
        aggregate = t["aggregate"]
        require(aggregate["measured_sequences"] == aggregate["complete_success_denominator"] == 22
                and aggregate["observed_metric_steps"] == 11982, "Incomplete validation22 denominator")
        run = s["runs"]["shared2000"]
        capture = run["capture_validation"]
        bridge = s["shared_social_reader_bridge"]
        require(bridge["failures_filtered"] is False and bridge["Test34_read"] is False
                and bridge["metric_math_changed"] is False, "Style reader changes the protocol")
        require(capture["metric_frames"] == 11982 and capture["clips"] == 22
                and capture["valid_same_clip_style_intervals"] == 11960, "Style frames were omitted")
        prun = p["runs"]["shared2000"]
        require(prun["failure_filter_applied"] is False and p["reference_refit_performed"] is False
                and p["generated_samples_used_for_fit"] is False
                and prun["poses"]["full_inclusive_pose_count"] == 11960, "Pose population or frozen basis changed")
        require(prun["fall_frames_kept"] == capture["fall_metric_frames_kept"], "Failure populations disagree")
        require(prun["tracking_aggregate"] == aggregate, "Pose source does not match tracking aggregate")
        tracking.append(aggregate)
        style.append(run)
        poses.append(prun["poses"])
    require(before["tracking"]["protocol"] == after["tracking"]["protocol"], "Tracking protocols differ")
    sequence_keys = ("motion_id", "source_path", "expected_control_steps", "tracking_window")
    require([{k: row[k] for k in sequence_keys} for row in before["tracking"]["sequences"]]
            == [{k: row[k] for k in sequence_keys} for row in after["tracking"]["sequences"]],
            "Tracking clips or source frame windows differ")
    require(before["style90"]["reference"] == after["style90"]["reference"], "Style reference or metric spec differs")
    for key in ("reference", "basis", "metric_implementation", "MMD_estimator"):
        require(before["pose45"][key] == after["pose45"][key], "Pose basis/metric differs: " + key)
    x, y = [p["pose_only_biased_gaussian_MMD2"] for p in poses]
    for key in ("sigma", "dimensions", "estimator", "maximum_samples_each", "generated_samples",
                "reference_samples", "generated_indices_sha256", "reference_indices_sha256", "sampling"):
        require(x[key] == y[key], "45D estimator or deterministic sample indices differ: " + key)
    tracking_out = {key: change(tracking[0][key], tracking[1][key], key != "tracking_successes")
                    for key in ("tracking_successes", "sequences_with_fall", "fall_onset_events",
                                "frame_weighted_mean_body_position_error_m", "frame_weighted_mean_body_rotation_error_rad")}
    clusters = {k: {"coverage": change(poses[0]["clusters"][k]["coverage_occupied_over_K"],
                                      poses[1]["clusters"][k]["coverage_occupied_over_K"], False),
                    "normalized_entropy": change(poses[0]["clusters"][k]["normalized_entropy"],
                                                 poses[1]["clusters"][k]["normalized_entropy"], False),
                    "nearest_center_mean": change(poses[0]["clusters"][k]["nearest_center_distance"]["mean"],
                                                  poses[1]["clusters"][k]["nearest_center_distance"]["mean"])}
                for k in ("50", "100")}
    # Exclude only declared artifact-specific paths from the physical protocol.
    old_args, new_args = [dict(d["navigation"]["arguments"]) for d in (before, after)]
    for key in ("output", "resume"):
        old_args.pop(key)
        new_args.pop(key)
    require(old_args == new_args, "Navigation arguments differ beyond actor checkpoint/output")
    for key in ("physics", "route_completion_contract", "navigation_guidance", "diffusion_guidance"):
        require(before["navigation"][key] == after["navigation"][key], "Navigation protocol differs: " + key)
    nav = []
    for d in (before, after):
        n, a = d["navigation"], d["navigation_CPU_audit"]
        require(n["status"] == a["status"] == "passed" and n["cleanup_succeeded"] is True
                and not n["cleanup_errors"], "Navigation execution/cleanup/audit did not pass")
        require(a["controls"] == n["actual_control_steps"] == 6000 and a["actor_controls"] == 24000
                and a["physics_substeps"] == 24000, "Incomplete navigation exposure")
        require(a["fall_termination_actor_controls"] == n["fall_resets"]
                and a["route_vertices_reached"] == n["route_vertices_reached"], "Navigation audit disagrees")
        nav.append(a)
    require(nav[0]["native_normal_force_threshold_n"] == nav[1]["native_normal_force_threshold_n"] == 1.0,
            "Contact threshold differs")
    nav_out = {key: change(nav[0][key], nav[1][key]) for key in (
        "fall_termination_actor_controls", "static_contact_actor_controls", "unordered_peer_pair_controls")}
    nav_out["route_vertices_total"] = change(sum(nav[0]["route_vertices_reached"]), sum(nav[1]["route_vertices_reached"]), False)
    nav_out["route_vertices_by_actor"] = [n["route_vertices_reached"] for n in nav]
    return {"tracking": tracking_out, "style": {"pose45_MMD2": change(x["value"], y["value"]),
            "clusters": clusters, "speed_W1_m_s": change(*[s["inclusive_speed_conditioned"]["physical_speed_distribution_wasserstein_m_s"] for s in style]),
            "supported_bins": [s["inclusive_speed_conditioned"]["motion_supported_bins"] for s in style],
            "common_speed_bin_secondary": common_speed_bins(*[s["inclusive_speed_conditioned"] for s in style]),
            "fall_frames_included": [s["capture_validation"]["fall_metric_frames_kept"] for s in style]},
            "navigation": nav_out, "qualification": "Single seed and local obstacle fixture; axes are separate. No paper numeric equivalence or overall superiority."}


def admit_baseline(tracking_receipt, navigation_receipt, audit_receipt):
    t, n, a = map(read, (tracking_receipt, navigation_receipt, audit_receipt))
    require(t["actual_pipeline_exit_code"] == n["actual_exit_code"] == a["actual_exit_code"] == 0,
            "Baseline lacks actual successful exits")
    pipeline = bound(t["pipeline"])
    require(pipeline["status"] == "passed_execution"
            and all(row["actual_exit_code"] == 0 for row in pipeline["steps"]), "Baseline tracking child failed")
    output = {key: bound(t[key]) for key in ("tracking", "style90", "pose45")}
    output.update(navigation=bound(n["result"]), navigation_CPU_audit=bound(a["audit"]))
    require(output["navigation_CPU_audit"]["result"] == n["result"], "Baseline nav audit result binding differs")
    return output


def admit_candidate(receipt):
    document = read(receipt)
    require(document["schema"] == "shared_social.actual_post_training_owner_exit.v1"
            and document["actual_exit_code"] == 0, "Candidate lacks actual pipeline terminal exit")
    pipeline = bound(document["pipeline"])
    require(pipeline["status"] == "passed_execution" and len(pipeline["steps"]) == 5
            and all(row["status"] == "passed" and row["actual_exit_code"] == 0 for row in pipeline["steps"]),
            "Candidate post-training stages did not complete")
    output = {key: bound(document["measurements"][key]) for key in (
        "tracking", "style90", "pose45", "navigation", "navigation_CPU_audit")}
    require(output["navigation_CPU_audit"]["result"] == document["measurements"]["navigation"],
            "Candidate nav audit result binding differs")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-tracking-receipt", "baseline-navigation-receipt", "baseline-audit-receipt",
                 "candidate-post-receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Retain prior comparisons; use a new output")
    paths = [args.baseline_tracking_receipt, args.baseline_navigation_receipt, args.baseline_audit_receipt,
             args.candidate_post_receipt]
    identities = [record(p) for p in paths]
    baseline = admit_baseline(*paths[:3])
    candidate = admit_candidate(paths[3])
    comparison = compare_metrics(baseline, candidate)
    require(identities == [record(p) for p in paths], "Receipts changed during comparison")
    report = {"schema": "shared_social.completed_independent_comparison.v1", "status": "passed_comparison",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "reader": record(__file__),
              "exit_receipts": identities, "comparison": comparison,
              "Torch_imported": "torch" in sys.modules, "SDK_started": False,
              "paper_effect_achieved": False, "default_model_promoted": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "output": record(args.output), "comparison": comparison}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
