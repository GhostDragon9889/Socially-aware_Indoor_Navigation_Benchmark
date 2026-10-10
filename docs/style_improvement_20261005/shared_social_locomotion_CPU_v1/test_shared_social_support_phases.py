"""Small CPU phase fixtures, not AMASS or simulator quality evidence."""
import unittest

import numpy as np

from CrowdSim.humanoid.analysis.support_phases import support_phases


class SupportPhaseTests(unittest.TestCase):
    def test_reset_is_not_a_touchdown_and_does_not_join_standing_runs(self):
        contact = np.zeros((3, 4, 1, 2), dtype=bool)
        contact[0, :, 0] = True
        contact[1, 2:, 0] = True
        contact[2, :, 0] = True
        before = contact.copy()
        result = support_phases(contact, np.array([[0], [0], [1]], dtype=np.int64))
        self.assertEqual(result["contact_onsets_left_right"], [1, 1])
        self.assertEqual(result["cross_episode_actor_intervals_excluded_from_transitions"], 1)
        self.assertAlmostEqual(result["longest_double_support_in_one_episode_s_by_actor"][0], 4 / 120)
        self.assertEqual(sum(result["phase_actor_substeps"].values()), 12)
        np.testing.assert_array_equal(contact, before)

    def test_sides_and_chronology_do_not_mix_actors_or_substeps(self):
        contact = np.zeros((2, 4, 2, 2), dtype=bool)
        contact[:, :, 0, 0] = True
        contact[1, :, 1, 1] = True
        result = support_phases(contact, np.zeros((2, 2), dtype=np.int64))
        self.assertEqual(result["phase_actor_substeps"],
                         {"no_support": 4, "left_only": 8, "right_only": 4, "double_support": 0})
        self.assertEqual(result["contact_onsets_left_right"], [0, 1])
        self.assertEqual(result["same_episode_actor_intervals"], 14)
        self.assertEqual(result["actor_substeps"], 16)

    def test_reset_changes_are_neither_onsets_nor_releases(self):
        contact = np.zeros((2, 4, 1, 2), dtype=bool)
        contact[0, :, 0, 1] = True
        contact[1, :, 0, 0] = True
        result = support_phases(contact, np.array([[0], [1]], dtype=np.int64))
        self.assertEqual(result["contact_onsets_left_right"], [0, 0])
        self.assertEqual(result["contact_releases_left_right"], [0, 0])
        self.assertEqual(result["cross_episode_actor_intervals_excluded_from_transitions"], 1)

    def test_missing_or_invalid_episode_clock_fails(self):
        contact = np.zeros((2, 4, 1, 2), dtype=bool)
        with self.assertRaises(ValueError):
            support_phases(contact, np.array([[0], [2]], dtype=np.int64))
        with self.assertRaises(ValueError):
            support_phases(contact.astype(np.float32), np.zeros((2, 1), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
