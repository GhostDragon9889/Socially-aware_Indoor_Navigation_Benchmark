"""The recovery label bridge leaves real frozen metric populations unchanged."""
from copy import deepcopy
from pathlib import Path
import unittest

from scripts import compare_shared_social_evidence as original
from scripts import compare_shared_social_recovery_evidence as recovery


class RecoveryComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = original.admit_candidate(recovery.ROOT /
            "docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json")

    def test_actual_baseline_label_bridge_is_identity(self):
        bridged = recovery.single_run_label_bridge(self.bundle, "shared2000")
        self.assertEqual(bridged, self.bundle)
        self.assertIsNot(bridged, self.bundle)

    def test_recovered_label_keeps_every_metric_and_input(self):
        candidate = deepcopy(self.bundle)
        for name in ("style90", "pose45"):
            candidate[name]["runs"] = {"S3_R12500": candidate[name]["runs"]["shared2000"]}
        bridge = candidate["style90"]["shared_social_reader_bridge"]
        bridge["inputs"] = {"S3_R12500": bridge["inputs"]["shared2000"]}
        before = deepcopy(candidate)
        bridged = recovery.single_run_label_bridge(candidate, "S3_R12500")
        self.assertEqual(bridged, self.bundle)
        self.assertEqual(candidate, before)

    def test_multiple_or_wrong_style_population_rejected(self):
        for labels in ([], ["other"], ["shared2000", "S3_R12500"]):
            with self.subTest(labels=labels):
                candidate = deepcopy(self.bundle)
                candidate["style90"]["runs"] = {label: {} for label in labels}
                with self.assertRaisesRegex(ValueError, "one explicitly named style population"):
                    recovery.single_run_label_bridge(candidate, "shared2000")

    def test_real_baseline_self_comparison_remains_zero(self):
        result = original.compare_metrics(self.bundle, recovery.single_run_label_bridge(self.bundle, "shared2000"))
        for row in result["tracking"].values():
            self.assertEqual(row["candidate_minus_baseline"], 0)
        for name in ("fall_termination_actor_controls", "static_contact_actor_controls", "unordered_peer_pair_controls",
                     "route_vertices_total"):
            self.assertEqual(result["navigation"][name]["candidate_minus_baseline"], 0)
        self.assertEqual(result["style"]["pose45_MMD2"]["candidate_minus_baseline"], 0)

    def test_style_input_label_must_match_the_metric_label(self):
        candidate = deepcopy(self.bundle)
        candidate["style90"]["shared_social_reader_bridge"]["inputs"] = {"other": {}}
        with self.assertRaisesRegex(ValueError, "matching explicit style input label"):
            recovery.single_run_label_bridge(candidate, "shared2000")

    def test_running_evaluation_cannot_be_admitted(self):
        path = recovery.ROOT / "docs/style_improvement_20261008/shared_social_S3R_evaluation_v2/owner_state.json"
        if not path.exists():
            self.skipTest("The owner state is not present")
        # Completed or running state remains mutable; the readonly completion
        # must be admitted instead, never a progress snapshot.
        with self.assertRaisesRegex(ValueError, "readonly recovery artifact"):
            recovery.admit_recovery_evaluation(Path(path))


if __name__ == "__main__":
    unittest.main()
