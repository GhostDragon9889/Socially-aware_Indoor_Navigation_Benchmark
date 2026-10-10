"""Cold comparison admission: actual S2 evidence plus explicitly synthetic guards."""
from copy import deepcopy
import json
import sys
import unittest

from scripts import compare_shared_social_episode_evidence as comparison


class EpisodeComparisonTests(unittest.TestCase):
    def test_actual_registration_selects_same_budget_and_preserves_unknowns(self):
        registered = comparison.registration_document()
        self.assertEqual(registered["training_updates_each"], 2000)
        self.assertEqual(registered["physical_training_transitions_each"], 16384000)
        self.assertEqual(registered["fixed_real_AMASS_training_clips"], 2848)
        self.assertEqual(registered["baseline_post_receipt"]["sha256"], comparison.BASELINE_SHA)
        self.assertEqual(registered["candidate_owner_plan"]["sha256"], comparison.PLAN_SHA)
        self.assertFalse(registered["candidate_evaluation_completed_at_registration"])
        self.assertFalse(registered["single_variable_causal_experiment_claimed"])
        self.assertFalse(registered["author_protocol_equivalence_claimed"])
        self.assertFalse(registered["Test34_used"])
        self.assertNotIn("torch", sys.modules)

    def test_completed_baseline_is_not_a_candidate_owner_completion(self):
        actual = json.loads(comparison.BASELINE.read_text())
        with self.assertRaisesRegex(ValueError, "all actual fresh episode"):
            comparison.validate_owner(actual)

    def test_missing_candidate_is_rejected_before_comparison_or_output(self):
        self.assertFalse(comparison.COMPLETION.exists())
        self.assertFalse(comparison.OUTPUT.exists())
        with self.assertRaisesRegex(ValueError, "readonly recovery artifact"):
            comparison.admit_candidate(comparison.registration_document())
        self.assertFalse(comparison.OUTPUT.exists())

    def test_owner_contract_rejects_failed_missing_inferred_or_reordered_exits(self):
        # Synthetic contract fixture, never an actual completion document.
        fixture = {"schema": "shared_social.episode_after_recovery_pipeline_owner_state.v1",
                   "status": "completed_training_and_independent_evaluation_quality_unclassified",
                   "steps": [{"name": name, "status": "passed", "actual_exit_code": 0,
                              "exit_observation": "subprocess.Popen.wait"} for name in comparison.STEPS]}
        comparison.validate_owner(fixture)
        for mode in ("missing", "reordered", "bool", "float", "signal", "inferred", "running"):
            with self.subTest(mode=mode):
                wrong = deepcopy(fixture)
                if mode == "missing":
                    wrong["steps"].pop()
                elif mode == "reordered":
                    wrong["steps"].reverse()
                elif mode == "inferred":
                    wrong["steps"][-1]["exit_observation"] = "checkpoint exists"
                elif mode == "running":
                    wrong["status"] = "running"
                else:
                    wrong["steps"][-1]["actual_exit_code"] = {"bool": False, "float": 0., "signal": -11}[mode]
                with self.assertRaisesRegex(ValueError, "all actual fresh episode"):
                    comparison.validate_owner(wrong)

    def test_private_parent_view_preserves_full_provenance_and_original_children(self):
        base = {"receipt": {"sha256": "fixture_receipt"}, "checkpoint": {"sha256": "fixture_checkpoint"}}
        full = {**deepcopy(base), "episode_reference_experiment": {"fixture": True},
                "episode_after_recovery_execution": {"fixture": True}}
        pipeline = {"training_execution_admission": deepcopy(full),
                    "steps": [{"name": "fixture_child", "actual_exit_code": 0}], "metric": [1., 2.]}
        saved = deepcopy(pipeline)
        view = comparison.child_lineage_view(pipeline, base, full)
        self.assertEqual(view["training_execution_admission"], base)
        self.assertEqual(view["steps"], pipeline["steps"])
        self.assertEqual(view["metric"], pipeline["metric"])
        self.assertEqual(pipeline, saved)
        view["training_execution_admission"]["checkpoint"]["sha256"] = "different"
        self.assertEqual(base["checkpoint"]["sha256"], "fixture_checkpoint")

    def test_parent_child_core_or_undeclared_extension_mismatch_rejected(self):
        base = {"receipt": "fixture_receipt", "checkpoint": "fixture_checkpoint"}
        full = {**base, "episode_reference_experiment": {}, "episode_after_recovery_execution": {}}
        for mode in ("wrong_checkpoint", "missing_extension", "unknown_extension", "wrong_pipeline"):
            with self.subTest(mode=mode):
                wrong = deepcopy(full)
                if mode == "wrong_checkpoint":
                    wrong["checkpoint"] = "another_actor"
                elif mode == "missing_extension":
                    wrong.pop("episode_reference_experiment")
                elif mode == "unknown_extension":
                    wrong["undeclared"] = True
                pipeline = {"training_execution_admission": full if mode == "wrong_pipeline" else wrong}
                expected = {**full, "episode_after_recovery_execution": {"different": True}} if mode == "wrong_pipeline" else wrong
                with self.assertRaisesRegex(ValueError, "two exact verified candidate metadata extensions"):
                    comparison.child_lineage_view(pipeline, base, expected)


if __name__ == "__main__":
    unittest.main()
