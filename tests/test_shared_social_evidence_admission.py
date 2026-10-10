"""Supplemental admission regressions use in-memory evidence and stdlib only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import verify_shared_social_evidence_admission as admission


def file_record(name):
    return {"path": str(admission.ROOT / "fixture" / name), "bytes": 1, "sha256": name}


def stage(name, result):
    return {"name": name, "status": "passed", "actual_exit_code": 0, "result": result}


def pipeline_fixture():
    measurements = {key: file_record(key) for key in admission.MEASUREMENTS}
    training = {"receipt": file_record("training_receipt"), "checkpoint": file_record("actor")}
    tracking = {"status": "passed_execution", "training_receipt": training["receipt"],
                "training_execution_admission": training,
                "steps": [stage("lifecycle_smoke", file_record("lifecycle")),
                          stage("full22", measurements["tracking"]), stage("style", measurements["style90"])]}
    pipeline = {"status": "passed_execution", "measurements": deepcopy(measurements),
                "training_execution_admission": training,
                "steps": [stage(name, file_record(name)) for name in admission.POST_STAGES]}
    pipeline["steps"][3]["result"] = measurements["navigation"]
    pipeline["steps"][4]["result"] = measurements["navigation_CPU_audit"]
    bundle = {"tracking": {"training_execution_admission": training},
              "style90": {"frozen_pose45_result": measurements["pose45"]}}
    graph = {row["result"]["path"]: {"status": "passed"} for row in pipeline["steps"] + tracking["steps"]}
    graph[file_record("lifecycle")["path"]] = {"status": "passed_lifecycle_smoke"}
    graph[file_record("tracking_style")["path"]] = tracking
    graph[file_record("actor_CPU_inference")["path"]]["training_execution_admission"] = training
    return {"measurements": measurements}, pipeline, bundle, graph


def tracking_fixture():
    return {"physics": {"control_hz": 30, "fall_definition": {"height_m": .45}},
            "arguments": {"checkpoint": "old.pt", "checkpoint_sha256": "old", "output": "old-output",
                          "seed": 7, "failure_threshold_m": .5, "deterministic_evaluation": False},
            "preflight": {"checkpoint": "old.pt", "checkpoint_sha256_recomputed": "old",
                          "manifest_sha256": "fixed", "split": "validation", "selected_clips": [0, 1]},
            "reference_velocity_contract": {"reference_dof_velocity_frame": "child_joint"}}


def source_fixture():
    frozen = {relative: {"path": str(admission.ROOT / relative), "bytes": 1, "sha256": sha}
              for relative, sha in admission.FROZEN.items()}
    ids = {key: file_record(key) for key in admission.MEASUREMENTS}
    reference = frozen["docs/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json"]
    checkpoint = file_record("actor.pt")
    captures = [{**file_record("motion_%06d.npz" % index), "motion_id": index, "source_path": "source-%d" % index,
                 "metric_frames": 1, "fall_metric_frames_kept": 0} for index in range(22)]
    capture = {"sources": captures, "fall_metric_frames_kept": 0}
    t = {"cleanup_succeeded": True, "cleanup_errors": [], "aggregate": {"observed_metric_steps": 22},
         "reference_velocity_contract": {"frame": "child_joint"},
         "arguments": {"checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"]},
         "training_execution_admission": {"checkpoint": checkpoint, "terminal_exit_code": 0},
         "sequences": [{"motion_id": row["motion_id"], "source_path": row["source_path"], "complete_clip": True,
                        "expected_control_steps": 1, "observed_metric_steps": 1, "capture": row["path"],
                        "capture_sha256": row["sha256"], "fall_capture_frames": []} for row in captures]}
    reference_document = {"reference": "frozen"}
    p = {"reference": reference,
         "basis": frozen["docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz"],
         "original_basis_evaluation": frozen["docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/evaluation.json"],
         "metric_implementation": frozen["docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py"],
         "runs": {"shared2000": {"source": ids["tracking"], "capture_validation": deepcopy(capture)}}}
    s = {"reference": reference_document, "frozen_pose45_result": ids["pose45"],
         "sources": {"implementation": frozen["scripts/evaluate_pedestrian_style.py"],
                     "module": frozen["CrowdSim/humanoid/style_evaluation.py"]},
         "shared_social_reader_bridge": {"implementation": frozen["scripts/evaluate_shared_social_style.py"],
                                         "original_style_tool": frozen["scripts/evaluate_pedestrian_style.py"],
                                         "reference": reference, "inputs": {"shared2000": ids["tracking"]}},
         "runs": {"shared2000": {"capture_validation": capture,
                                  "sources": {"result": ids["tracking"], "tracking_captures": captures,
                                              "models_resolved_from_evaluation_arguments": {"controller": checkpoint},
                                              "declared_training_manifest": file_record("train.json")},
                                  "task_contract": {"benchmark_primary_aggregate_unchanged": t["aggregate"],
                                                    "reference_velocity_contract": t["reference_velocity_contract"]},
                                  "reference_not_in_declared_training": {"original_source_intersection": 0}}}}
    nav_directory = Path(ids["navigation"]["path"]).parent
    a = {"result": ids["navigation"], "execution_receipt": file_record("nav_receipt.json"),
         "arrays": [dict(file_record(name), path=str(nav_directory / name))
                    for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")]}
    n = {"arguments": {"resume": checkpoint["path"]}, "evaluation_actor": {"checkpoint": checkpoint}}
    graph = {reference["path"]: reference_document,
             a["execution_receipt"]["path"]: {"actual_exit_code": 0, "result": ids["navigation"]}}
    return {"tracking": t, "style90": s, "pose45": p, "navigation": n, "navigation_CPU_audit": a}, ids, graph


def population(values):
    return {"minimum_metric_samples_per_bin": 30, "maximum_balanced_motion_samples_per_bin": 256,
            "speed_bins": [{"lower_m_s": index * .5, "upper_m_s": (index + 1) * .5,
                            "generated_count": 100, "reference_count": 200,
                            "pose_velocity_gaussian_mmd2": None if value is None else {
                                "value": value, "sample_count_each": 30, "reference_sample_indices": list(range(30)),
                                "features": "pose_velocity", "feature_width": 90, "velocity_time_scale_s": .1,
                                "sigma": 1, "estimator": "biased MMD squared including diagonal"}}
                           for index, value in enumerate(values)]}


class AdmissionTests(unittest.TestCase):
    def test_successful_nested_children_admit_their_own_measurements(self):
        receipt, pipeline, bundle, graph = pipeline_fixture()
        with patch.object(admission.original, "bound", side_effect=lambda value: graph[value["path"]]):
            admission.post_lineage(receipt, pipeline, bundle)

    def test_successful_pipeline_cannot_admit_another_measurement_bundle(self):
        receipt, pipeline, bundle, _ = pipeline_fixture()
        receipt["measurements"]["tracking"] = file_record("other-actor-tracking")
        with self.assertRaisesRegex(ValueError, "Owner measurements"):
            admission.post_lineage(receipt, pipeline, bundle)

    def test_child_failure_or_child_result_substitution_cannot_enter(self):
        for kind in ("exit", "nested_result", "stage_order"):
            receipt, pipeline, bundle, graph = pipeline_fixture()
            if kind == "exit":
                pipeline["steps"][0]["actual_exit_code"] = -11
            elif kind == "stage_order":
                pipeline["steps"].reverse()
            else:
                graph[file_record("tracking_style")["path"]]["steps"][1]["result"] = file_record("other-tracking")
                graph[file_record("other-tracking")["path"]] = {"status": "passed"}
            with self.subTest(kind=kind), patch.object(admission.original, "bound", side_effect=lambda value: graph[value["path"]]):
                with self.assertRaises(ValueError):
                    admission.post_lineage(receipt, pipeline, bundle)

    def test_tracking_protocol_drift_is_rejected(self):
        before = tracking_fixture()
        mutations = (("physics", "control_hz", 60), ("arguments", "failure_threshold_m", 100),
                     ("arguments", "seed", 8), ("arguments", "deterministic_evaluation", True),
                     ("preflight", "manifest_sha256", "other"), ("preflight", "selected_clips", [0]),
                     ("reference_velocity_contract", "reference_dof_velocity_frame", "parent_joint"))
        for section, key, value in mutations:
            after = deepcopy(before)
            after[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                admission.tracking_protocol(before, after)

    def test_actor_and_output_paths_can_differ_between_models(self):
        before, after = tracking_fixture(), tracking_fixture()
        for section, keys in (("arguments", ("checkpoint", "checkpoint_sha256", "output")),
                              ("preflight", ("checkpoint", "checkpoint_sha256_recomputed"))):
            for key in keys:
                after[section][key] = "new-" + key
        admission.tracking_protocol(before, after)

    def test_frozen_sources_bind_own_tracking_captures(self):
        bundle, ids, graph = source_fixture()
        with patch.object(admission.original, "bound", side_effect=lambda value: graph[value["path"]]):
            sources = admission.bundle_sources(bundle, ids)
        self.assertIn(file_record("motion_000021.npz"), sources)
        self.assertIn(file_record("actor.pt"), sources)

    def test_conflicting_file_identity_cannot_be_hidden_by_deduplication(self):
        first = file_record("source")
        second = dict(first, sha256="changed")
        with self.assertRaisesRegex(ValueError, "Conflicting identities"):
            admission.unique_artifacts([first, second])

    def test_bottom_metric_or_bridge_source_drift_is_rejected(self):
        for kind in ("module", "bridge", "original_tool"):
            bundle, ids, _ = source_fixture()
            if kind == "module":
                bundle["style90"]["sources"]["module"]["sha256"] = "changed"
            else:
                key = "implementation" if kind == "bridge" else "original_style_tool"
                bundle["style90"]["shared_social_reader_bridge"][key] = file_record("changed")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                admission.bundle_sources(bundle, ids)

    def test_other_tracking_result_or_capture_cannot_be_used(self):
        for kind in ("tracking_result", "capture", "training_overlap"):
            bundle, ids, graph = source_fixture()
            if kind == "tracking_result":
                bundle["pose45"]["runs"]["shared2000"]["source"] = file_record("other-tracking")
            elif kind == "capture":
                bundle["pose45"]["runs"]["shared2000"]["capture_validation"]["sources"][0]["sha256"] = "other"
            else:
                bundle["style90"]["runs"]["shared2000"]["reference_not_in_declared_training"]["original_source_intersection"] = 1
            with self.subTest(kind=kind), patch.object(admission.original, "bound", side_effect=lambda value: graph[value["path"]]):
                with self.assertRaises(ValueError):
                    admission.bundle_sources(bundle, ids)

    def test_unequal_counts_are_unknown_and_do_not_enter_mean_as_zero(self):
        before, after = population([.1, .9]), population([.2, 0.0])
        metric = after["speed_bins"][1]["pose_velocity_gaussian_mmd2"]
        metric.update(sample_count_each=60, reference_sample_indices=list(range(60)))
        result = admission.matched_speed_bins(before, after)
        row = result["bins"][1]
        self.assertFalse(row["directional_comparison_eligible"])
        self.assertEqual(row["eligibility"], "unknown")
        self.assertIsNone(row["MMD2_comparison"])
        self.assertNotIn("improved", row)
        self.assertEqual(result["common_supported_bin_count"], 1)
        self.assertEqual(result["equal_common_bin_mean_MMD2"]["baseline"], .1)
        self.assertEqual(result["equal_common_bin_mean_MMD2"]["candidate"], .2)

    def test_equal_counts_with_different_reference_subset_are_unknown(self):
        before, after = population([.9]), population([0.0])
        after["speed_bins"][0]["pose_velocity_gaussian_mmd2"]["reference_sample_indices"] = list(range(30, 60))
        result = admission.matched_speed_bins(before, after)
        self.assertEqual(result["bins"][0]["unknown_reasons"], ["reference_subsets_differ"])
        self.assertIsNone(result["equal_common_bin_mean_MMD2"])
        self.assertEqual(result["common_supported_bin_count"], 0)

    def test_matched_sampling_is_eligible_and_unsupported_bin_is_unknown(self):
        result = admission.matched_speed_bins(population([.9, None]), population([.1, .3]))
        self.assertTrue(result["bins"][0]["MMD2_comparison"]["improved"])
        self.assertEqual(result["bins"][1]["eligibility"], "unknown")
        self.assertIsNone(result["bins"][1]["MMD2_comparison"])

    def test_unknown_90d_sampling_preserves_other_independent_axes(self):
        before = {"tracking": tracking_fixture(), "style90": {"runs": {"shared2000": {"inclusive_speed_conditioned": population([.9])}}}}
        after = deepcopy(before)
        after["style90"]["runs"]["shared2000"]["inclusive_speed_conditioned"]["speed_bins"][0]["pose_velocity_gaussian_mmd2"]["reference_sample_indices"] = list(range(30, 60))
        physical = {"tracking": {"valid": True}, "navigation": {"valid": True}, "style": {"pose45_MMD2": {"value": .2}}}
        with patch.object(admission.original, "compare_metrics", return_value=deepcopy(physical)):
            result = admission.supplemental_comparison(before, after)
        self.assertEqual(result["tracking"], physical["tracking"])
        self.assertEqual(result["navigation"], physical["navigation"])
        self.assertEqual(result["style"]["pose45_MMD2"], physical["style"]["pose45_MMD2"])
        self.assertIsNone(result["style"]["common_speed_bin_secondary"]["equal_common_bin_mean_MMD2"])


if __name__ == "__main__":
    unittest.main()
