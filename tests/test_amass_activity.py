"""Check weighting and window boundaries; fixtures do not assess model quality."""
import unittest

import numpy as np

from CrowdSim.humanoid.analysis import amass_activity as activity


def clip(count, speed, joints=0.):
    root = np.zeros((count, 3), np.float32)
    root[:, 0] = speed
    return activity.clip_statistics(root, np.full((count, 57), joints, np.float32), np.zeros_like(root))


class AMASSActivityTests(unittest.TestCase):
    def test_unequal_lengths_distinguish_uniform_clip_from_uniform_frame(self):
        result = activity.aggregate([clip(12, 0.), clip(20, 1.)])
        current = result["legal_expert_centers"]
        self.assertEqual(current["frames_or_centers"], 12)
        self.assertEqual(current["clips_retained"], 2)
        self.assertEqual(current["measures"]["uniform_clip_then_frame"]["proxy_probabilities"]["low_planar_speed"], .5)
        self.assertAlmostEqual(current["measures"]["uniform_frame"]["proxy_probabilities"]["low_planar_speed"], 2 / 12)
        self.assertEqual(result["all_frames"]["frames_or_centers"], 32)
        self.assertAlmostEqual(result["all_frames"]["measures"]["uniform_frame"]["proxy_probabilities"]["low_planar_speed"], 12 / 32)

    def test_history_includes_past_but_excludes_future_action_frame(self):
        root = np.zeros((21, 3), np.float64)
        root[0, 0] = 1.
        root[-1, 0] = 1.
        row = activity.clip_statistics(root, np.zeros((21, 57)), np.zeros_like(root))
        self.assertEqual(row["views"][1]["count"], 11)
        self.assertEqual(row["views"][1]["proxy_counts"][0], 11)
        self.assertEqual(row["history_proxy_counts"][0], 10)

    def test_low_root_speed_does_not_imply_low_joint_motion(self):
        row = clip(12, 0., joints=2.)
        self.assertEqual(row["views"][1]["proxy_counts"].tolist(), [2, 0, 2, 0, 0])
        self.assertEqual(row["history_proxy_counts"].tolist(), [2, 0, 2, 0, 0])

    def test_rotation_invariance_and_histogram_conservation(self):
        rng = np.random.default_rng(13)
        root, joints, angular = rng.normal(size=(32, 3)), rng.normal(size=(32, 57)), rng.normal(size=(32, 3))
        matrix = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        left = activity.clip_statistics(root, joints, angular)
        right = activity.clip_statistics(root @ matrix, (joints.reshape(-1, 19, 3) @ matrix).reshape(-1, 57), angular @ matrix)
        for a, b in zip(left["views"], right["views"], strict=True):
            self.assertEqual(a["histogram"].sum(), a["count"])
            np.testing.assert_array_equal(a["histogram"], b["histogram"])
            np.testing.assert_allclose(a["metric_means"], b["metric_means"], rtol=0, atol=1e-15)

    def test_invalid_velocities_and_window_rejected(self):
        root = np.zeros((12, 3), np.float64)
        for joints in (np.zeros((12, 56)), np.full((12, 57), np.nan), np.zeros((12, 57), np.int64)):
            with self.subTest(shape=joints.shape, dtype=joints.dtype):
                with self.assertRaises(ValueError):
                    activity.clip_statistics(root, joints, root)
        with self.assertRaises(ValueError):
            clip(10, 0.)
        with self.assertRaises(ValueError):
            activity.clip_statistics(root, np.zeros((12, 57)), root, past=True)


if __name__ == "__main__":
    unittest.main()
