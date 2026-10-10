"""CPU-only comparison of immutable tracking/style artifacts; never changes metrics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from CrowdSim.humanoid.style_evaluation import sequence_style_samples  # noqa: E402
from CrowdSim.humanoid.application.settings import TrainingConfig  # noqa: E402

STYLE = HERE / "tracking_style_v1/three_checkpoints/evaluation.json"
REFERENCE = HERE / "style_velocity_adapter_legacy_cpu/heldout_reference.json"
DECOMPOSITION = {
    "legacy1000": HERE / "legacy_tracking_decomposition.json",
    "amass2848_child500": HERE / "amass2848_child_step500_tracking_decomposition.json",
    "amass2848_child1000": HERE / "amass2848_child_tracking_decomposition.json",
}


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    specification = importlib.util.spec_from_file_location("tracking_style_reader", ROOT / "scripts/evaluate_pedestrian_style.py")
    reader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reader)
    style = load(STYLE)
    if style["status"] != "passed" or style["kit_started"] or style["device"] != "CPU":
        raise ValueError("Expected actual completed CPU style evaluation")
    frozen = record(REFERENCE)
    if frozen["sha256"] != "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564":
        raise ValueError("Frozen reference changed")
    if record(STYLE.parent / "heldout_reference.json")["sha256"] != frozen["sha256"]:
        raise ValueError("Tracking reference is not byte-identical")
    runs = {}
    results = {}
    decompositions = {}
    import torch  # CPU mmap of known local checkpoints; no Kit import.
    for name, style_run in style["runs"].items():
        result_path = Path(style_run["sources"]["result"]["path"])
        if record(result_path) != style_run["sources"]["result"]:
            raise ValueError("Source result changed since style evaluation")
        result = load(result_path)
        results[name] = result
        decomposition = load(DECOMPOSITION[name])
        decompositions[name] = decomposition
        if any(decomposition["source_result"].get(key) != record(result_path)[key] for key in ("path", "sha256")):
            raise ValueError("Style/decomposition sources differ")
        if decomposition["primary_aggregate_unchanged"] != result["aggregate"]:
            raise ValueError("Primary aggregate changed")
        checkpoint = style_run["sources"]["models_resolved_from_evaluation_arguments"]["controller"]
        bundle = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
        loaded_config = json.loads(json.dumps(asdict(TrainingConfig(**bundle["config"]))))
        if loaded_config != result["training_config"]:
            raise ValueError("Checkpoint/tracking TrainingConfig differ")
        arrays, _, capture = reader.read_tracking_capture(result_path.parent)
        sample = sequence_style_samples(arrays["body_pos"], arrays["root_rot"], episode_ids=arrays["episode_ids"],
                                        reset_mask=arrays["reset_mask"], body_names=capture["body_names"])
        end_fall = arrays["fall_mask"][sample["end_frame_indices"], sample["actor_indices"]]
        low_speed = sample["speed_m_s"] < .1
        upright = (sample["metrics"]["root_up_z"] >= .8) & (sample["metrics"]["pelvis_height_m"] >= .7)
        counts = {"all_same_clip_intervals": len(low_speed), "fall_ending_intervals": int(end_fall.sum()),
                  "physical_speed_below_point1_intervals": int(low_speed.sum()),
                  "low_speed_and_fall_ending_intervals": int((low_speed & end_fall).sum()),
                  "low_speed_and_not_upright_intervals": int((low_speed & ~upright).sum()),
                  "low_speed_upright_intervals": int((low_speed & upright).sum())}
        bins = [{key: bin_row[key] for key in ("lower_m_s", "upper_m_s", "generated_count", "reference_count")}
                | {"mmd2": None if bin_row["pose_velocity_gaussian_mmd2"] is None else bin_row["pose_velocity_gaussian_mmd2"]["value"]}
                for bin_row in style_run["inclusive_speed_conditioned"]["speed_bins"]]
        runs[name] = {
            "tracking_result": record(result_path), "decomposition": record(DECOMPOSITION[name]),
            "checkpoint": checkpoint,
            "actual_saved_update_count": bundle["update_count"],
            "actual_saved_physical_transitions": bundle["physical_transition_count"],
            "tracking_elapsed_wall_s": result["elapsed_s"],
            "dataset": result["controller_provenance"]["dataset"],
            "training_config": result["training_config"],
            "checkpoint_config_omitted_fields_loaded_via_documented_defaults": {
                key: loaded_config[key] for key in loaded_config if key not in bundle["config"]},
            "reference_velocity_contract": result["reference_velocity_contract"],
            "source_training_parent_disjointness": style_run["reference_not_in_declared_training"],
            "primary_aggregate_unchanged": result["aggregate"],
            "inclusive_mmd2": style_run["inclusive_speed_conditioned"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_diagnostic_mmd2": style_run["upright_subset"]["comparison"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "upright_fraction": style_run["upright_subset"]["generated_fraction_retained"],
            "physical_speed_distribution_wasserstein_m_s": style_run["inclusive_speed_conditioned"]["physical_speed_distribution_wasserstein_m_s"],
            "inclusive_physical_speed_bins": bins,
            "fall_and_stop_diagnostic": counts,
            "style_summary": style_run["generated"], "frozen_reference_style_summary": style_run["reference"],
            "same_frame_decomposition": decomposition["aggregate_diagnostic"],
            "initial_mapping_checks": decomposition["initial_mapping_checks"],
            "by_body": decomposition["by_body"], "by_clip": decomposition["by_clip"],
            "by_reference_instantaneous_speed": decomposition["by_reference_speed_bin"],
        }
        del bundle
    old = results["legacy1000"]["training_config"]
    new = results["amass2848_child1000"]["training_config"]
    diff = {key: {"legacy1000": old.get(key), "amass2848_child1000": new.get(key)}
            for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)}
    if results["amass2848_child500"]["training_config"] != new:
        raise ValueError("Intermediate and final checkpoint training configs differ")
    union_fall_ids = sorted({row["motion_id"] for item in decompositions.values() for row in item["by_clip"] if row["sequence_had_fall"]})
    panel = {}
    for name, decomposition in decompositions.items():
        selected = [row for row in decomposition["by_clip"] if row["motion_id"] not in union_fall_ids]
        weights = np.array([row["metric_frames"] for row in selected])
        keys = [key for key, value in selected[0]["diagnostic"].items() if isinstance(value, dict) and "mean" in value]
        panel[name] = {"clips": len(selected), "metric_frames": int(weights.sum()),
                       "same_frame_diagnostic_frame_weighted_means": {
                           key: float(np.average([row["diagnostic"][key]["mean"] for row in selected], weights=weights)) for key in keys}}
    normal = load(HERE / "tracking_style_normal_regression_cpu/evaluation.json")
    old_normal = load(HERE / "style_velocity_adapter_legacy_cpu/evaluation.json")
    stable_fields = ("generated", "reference", "inclusive_speed_conditioned", "upright_subset", "heldout_between_clip_group_calibration")
    normal_equal = {name: {key: value[key] == old_normal["runs"][name][key] for key in stable_fields}
                    for name, value in normal["runs"].items()}
    if not all(all(value.values()) for value in normal_equal.values()):
        raise ValueError("Existing physical-state scores changed")
    test_suite = ET.parse(HERE / "tracking_style_reader_final_tests.xml").getroot().find("testsuite")
    review = {
        "status": "passed", "CPU_only": True, "Kit_started": False,
        "before": load(HERE / "tracking_style_reader_before/source_manifest.json"),
        "after": record(ROOT / "scripts/evaluate_pedestrian_style.py"),
        "style_module": record(ROOT / "CrowdSim/humanoid/style_evaluation.py"),
        "style_module_hash_matches_historical": normal["sources"]["module"] == old_normal["sources"]["module"] == style["sources"]["module"],
        "frozen_reference": frozen,
        "three_checkpoint_reference_byte_identity": record(STYLE.parent / "heldout_reference.json")["sha256"] == frozen["sha256"],
        "actual_previous_physical_reader_scores_exactly_equal": normal_equal,
        "regression_sources": [record(HERE / "tracking_style_normal_regression_cpu/evaluation.json"), record(HERE / "style_velocity_adapter_legacy_cpu/evaluation.json")],
        "tests": {"junit": record(HERE / "tracking_style_reader_final_tests.xml"), "counts": dict(test_suite.attrib)},
        "actual_tracking_frames_each": {name: {key: value["capture_validation"][key] for key in
            ("metric_frames", "clips", "initial_reset_frames_discarded", "valid_same_clip_style_intervals", "fall_metric_frames_kept")} for name, value in style["runs"].items()},
        "source_contracts": ["Only explicitly declared result sequences; no glob or silent missing-file fallback",
            "Each capture SHA, sequence count, source ID/path, consecutive 30Hz reference phase and initial/reset masks checked",
            "Initial reset pose excluded; distinct clips get separate episode IDs and before-frame reset masks",
            "All true post-step frames retained, including falls; no time alignment or source best-match search",
            "Actual controller training dataset is checked separately from oracle heldout input",
            "Parent-source overlap including #frames segments rejects; explicit child/legacy reference velocity preserved",
            "Contacts/true COM absent remain null, not filled with navigation proxies"],
    }
    if not review["style_module_hash_matches_historical"]:
        raise ValueError("Frozen style module changed")
    report = {
        "schema": "navisaaclab.tracking_style_checkpoint_comparison.v1", "status": "passed",
        "CPU_only": True, "Kit_started": False, "primary_metric_modified": False,
        "author_equivalent_metrics": False, "causal_single_factor_comparison": False,
        "comparison_script": record(Path(__file__)), "style_artifact": record(STYLE), "frozen_reference": frozen,
        "style_spec_unchanged": style["reference"]["metric_spec"], "runs": runs,
        "actual_config_difference_count": len(diff), "actual_config_differences": diff,
        "reference_speed_bin_vs_style_speed_bin_distinction": "Decomposition bins use captured reference instantaneous velocity; style bins use actual/generated root displacement divided by dt",
        "common_no_observed_fall_clip_panel_diagnostic_only": {"excluded_union_fall_motion_ids": union_fall_ids,
            "selection_depends_on_observed_outcomes": True, "not_primary_metric_or_acceptance_gate": True, "runs": panel},
        "limitations": ["All 22 primary sequences and all falls remain in the main result; upright and common-no-fall panels are explanatory only",
            "New step500 has 4.096M physical transitions and new1000 has8.192M; old1000 has2.048M, so equal update counts are not equal sample budgets",
            "Training dataset, 14 TrainingConfig fields, parallel actor count and evaluation reset/reference DOF velocity contract differ",
            "TruefutureXY oracle low-level tracking is not diffusion navigation or the paper eight-scene experiment",
            "Style MMD is explicit local90D biased MMD squared/sigma1, not the author's unknown embedding/bandwidth statistic",
            "Validation is repeatedly inspected for model development; no untouched test-set or IID confidence interval claim",
            "Toe-height support/sliding and mass-weighted joint-origin height are proxies; actual foot contact and true body COM are unavailable"],
    }
    for filename, data in (("tracking_style_input_review.json", review), ("tracking_style_comparison.json", report)):
        (HERE / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for name, item in runs.items():
        print(name, item["actual_saved_update_count"], item["actual_saved_physical_transitions"], item["inclusive_mmd2"], item["fall_and_stop_diagnostic"])
    print("Config differences", len(diff), "Common diagnostic panel", panel)


if __name__ == "__main__":
    main()
