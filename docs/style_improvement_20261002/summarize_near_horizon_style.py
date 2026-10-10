"""Read completed frozen style reports; do not tune metrics or audit native contacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


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
    return {key: {"guidance20": left.get(key), "candidate": right.get(key),
                  "guidance20_field_present": key in left, "candidate_field_present": key in right}
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key) or (key in left) != (key in right)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    here = args.evaluation.resolve().parent
    evaluation = load(args.evaluation)
    prior_path = DOCS / "navigation_guidance40_comparison_v1/frozen_style/evaluation.json"
    prior = load(prior_path)
    assert evaluation["status"] == "passed" and evaluation["device"] == "CPU" and not evaluation["kit_started"]
    assert set(evaluation["runs"]) == {"guidance20", "guidance40", "eta025"}
    reference = record(here / "heldout_reference.json")
    assert reference["sha256"] == REFERENCE_SHA
    unchanged = {name: value == evaluation["runs"][name] for name, value in prior["runs"].items()}
    assert set(unchanged) == {"guidance20", "guidance40"} and all(unchanged.values())
    runs = {}
    results = {}
    for name, item in evaluation["runs"].items():
        result_source = item["sources"]["result"]
        assert record(result_source["path"]) == result_source
        result = load(result_source["path"])
        assert result["status"] == "passed" and result.get("error") is None
        results[name] = result
        inclusive = item["inclusive_speed_conditioned"]
        runs[name] = {
            "result": result_source, "states": item["sources"]["states"],
            "models": item["sources"]["models_resolved_from_evaluation_arguments"],
            "heldout_parent_disjointness": item["reference_not_in_declared_training"],
            "capture_validation": item["capture_validation"],
            "same_episode_intervals": item["generated"]["same_episode_transition_count"],
            "valid_acceleration_triplets": item["generated"]["valid_acceleration_triplets"],
            "inclusive_mmd2": inclusive["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_mmd2": item["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": item["upright_subset"]["generated_fraction_retained"],
            "speed_wasserstein_m_s": inclusive["physical_speed_distribution_wasserstein_m_s"],
            "physical_root_speed_m_s": item["generated"]["physical_root_speed_m_s"],
            "toe_near_ground_body_origin_proxy": item["generated"]["toe_slide_near_ground_proxy"],
            "speed_bins": [{key: row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                           | {"mmd2": None if row["pose_velocity_gaussian_mmd2"] is None else row["pose_velocity_gaussian_mmd2"]["value"]}
                           for row in inclusive["speed_bins"]],
            "recorded_runtime_outcomes": {key: result.get(key) for key in
                                          ("fall_resets", "time_limit_resets", "resets", "route_vertices_reached", "minimum_height_m", "actual_control_steps")},
            "native_foot_COM_contact_and_near_stall_analysis": "Owned by the separate ground-support audit; not recomputed here",
        }
    baseline = results["guidance20"]
    comparisons = {}
    for name in ("guidance40", "eta025"):
        candidate = results[name]
        model_equal = {key: runs["guidance20"]["models"][key] == runs[name]["models"][key] for key in runs["guidance20"]["models"]}
        assert all(model_equal.values())
        comparisons[name] = {
            "model_file_identity_exact": model_equal,
            "training_config_differences": differences(baseline["training_config"], candidate["training_config"]),
            "runtime_argument_differences": differences(baseline["arguments"], candidate["arguments"]),
            "equal_recorded_fields": {key: baseline.get(key) == candidate.get(key) for key in
                                      ("training_config", "initialization", "physics", "runtime_numerics", "runtime_execution", "evaluation_scene", "power_measurement", "planning_clock")},
            "same_seed_is_not_physical_trajectory_equality": True,
        }
    summary = {
        "schema": "navisaaclab.near_horizon_three_run_style_review.v1", "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "Kit_started": False,
        "business_data_profiles_or_evaluator_modified": False, "evaluation": record(args.evaluation),
        "reference": reference, "previous_two_run_report": record(prior_path), "previous_two_run_dicts_exact": unchanged,
        "frozen_implementations": evaluation["sources"], "summary_script": record(__file__),
        "style_spec": evaluation["reference"]["metric_spec"], "runs": runs, "baseline_comparisons": comparisons,
        "limitations": [
            "All captured fall frames remain inclusive; only cross-reset derivative intervals are excluded.",
            "Upright-only is a secondary diagnostic and does not replace inclusive acceptance.",
            "90D local pose/velocity Gaussian biased MMD squared uses sigma1; exact author parameters/pass threshold are unknown.",
            "The exposed validation artifact is fixed for iterative comparison, not an untouched final test.",
            "Body-origin toe proxies are not native support-point contact, physical COM or a guarantee of natural gait.",
            "Same model/seed/declared setup does not establish identical realized dynamics or multi-seed confidence.",
            "These runs use a declared flat obstacle layout and deployment adaptation, not the author eight-scene experiment.",
        ],
    }
    with (here / "comparison.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"inclusive_mmd2": item["inclusive_mmd2"], "upright_mmd2": item["upright_mmd2"],
                             "speed_wasserstein_m_s": item["speed_wasserstein_m_s"], "recorded_outcomes": item["recorded_runtime_outcomes"]}
                      for name, item in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
