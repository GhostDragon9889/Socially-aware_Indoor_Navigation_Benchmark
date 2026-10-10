"""Reject misleading support, estimator and receipt comparisons on CPU."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.compare_shared_social_evidence import admit_candidate, bound, change, common_speed_bins, record


def population(values):
    return {"minimum_metric_samples_per_bin": 30, "maximum_balanced_motion_samples_per_bin": 256,
            "speed_bins": [{"lower_m_s": index * .5, "upper_m_s": (index + 1) * .5,
                            "generated_count": 100 + index, "reference_count": 200,
                            "pose_velocity_gaussian_mmd2": None if value is None else {
                                "value": value, "sample_count_each": 100, "features": "pose_velocity",
                                "feature_width": 90, "velocity_time_scale_s": .1, "sigma": 1,
                                "estimator": "biased MMD squared including diagonal"}}
                           for index, value in enumerate(values)]}


class EvidenceComparisonTests(unittest.TestCase):
    def test_support_loss_cannot_enter_common_mean_as_zero(self):
        a, b = population([.1, .9, .3]), population([.2, None, .5])
        result = common_speed_bins(a, b)
        self.assertEqual(result["common_supported_bin_count"], 2)
        self.assertAlmostEqual(result["equal_common_bin_mean_MMD2"]["baseline"], .2)
        self.assertAlmostEqual(result["equal_common_bin_mean_MMD2"]["candidate"], .35)
        self.assertFalse(result["equal_common_bin_mean_MMD2"]["improved"])
        self.assertIsNone(result["bins"][1]["MMD2_comparison"])

    def test_no_common_support_has_no_numeric_score(self):
        result = common_speed_bins(population([None, .2]), population([.1, None]))
        self.assertEqual(result["common_supported_bin_count"], 0)
        self.assertIsNone(result["equal_common_bin_mean_MMD2"])

    def test_different_kernel_edges_or_sampling_are_rejected(self):
        original = population([.1])
        for kind in ("kernel", "edges", "sample_rule", "reference_population"):
            other = deepcopy(original)
            if kind == "kernel":
                other["speed_bins"][0]["pose_velocity_gaussian_mmd2"]["sigma"] = 2
            elif kind == "edges":
                other["speed_bins"][0]["upper_m_s"] = 1
            elif kind == "sample_rule":
                other["maximum_balanced_motion_samples_per_bin"] = 512
            else:
                other["speed_bins"][0]["reference_count"] = 201
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                common_speed_bins(original, other)

    def test_changed_artifact_or_missing_successful_exit_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text('{"status":"passed"}')
            identity = record(path)
            path.write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                bound(identity)
            path.write_text('{"schema":"shared_social.actual_post_training_owner_exit.v1","actual_exit_code":-11}')
            with self.assertRaises(ValueError):
                admit_candidate(path)

    def test_directions_and_nonfinite_values(self):
        self.assertTrue(change(22, 7)["improved"])
        self.assertFalse(change(.84, .64, False)["improved"])
        self.assertFalse(change(0, 0, False)["improved"])
        with self.assertRaises(ValueError):
            change(1, float("nan"))


if __name__ == "__main__":
    unittest.main()
