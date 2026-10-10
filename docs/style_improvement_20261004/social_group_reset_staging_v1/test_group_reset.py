"""Bounded synthetic layout/mask cases; no simulator or model validation."""
import unittest

import numpy as np

from group_reset import expand_group_reset, make_mixed_layout


class GroupResetChecks(unittest.TestCase):
    def test_mixed_layout_keeps_flat_rows_and_two_sources(self):
        layout = make_mixed_layout(8)
        np.testing.assert_array_equal(layout["group_actor_rows"], [[0, 1, 4, 5], [2, 3, 6, 7]])
        np.testing.assert_array_equal(layout["scene_ids"], [0, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(layout["actor_ids"], [0, 1, 2, 3, 4, 5, 6, 7])
        full = make_mixed_layout(256)
        np.testing.assert_array_equal(full["group_actor_rows"][0], [0, 1, 128, 129])
        np.testing.assert_array_equal(full["group_actor_rows"][-1], [126, 127, 254, 255])
        self.assertTrue(np.all((full["group_actor_rows"] < 128).sum(axis=1) == 2))
        self.assertEqual(len(np.unique(full["actor_ids"])), 256)
        for value in full.values():
            self.assertEqual(value.dtype, np.int64)
        np.testing.assert_array_equal(make_mixed_layout(4)["group_actor_rows"], [[0, 1, 2, 3]])

    def test_one_fall_collateral_truncation_and_bootstrap_meaning(self):
        groups = make_mixed_layout(8)["group_actor_rows"]
        terminal, truncation = np.zeros(8, dtype=bool), np.zeros(8, dtype=bool)
        terminal[1] = True
        result = expand_group_reset(groups, terminal, truncation)
        np.testing.assert_array_equal(result["terminated"], terminal)
        np.testing.assert_array_equal(np.flatnonzero(result["truncated"]), [0, 4, 5])
        np.testing.assert_array_equal(np.flatnonzero(result["reset_mask"]), [0, 1, 4, 5])
        np.testing.assert_array_equal(np.flatnonzero(result["collateral_reset"]), [0, 4, 5])
        np.testing.assert_array_equal(result["triggered_groups"], [0])
        # Original GAE semantics: reward1, next_value2, gamma.9, future_carry100.
        # Fall => no bootstrap, collateral truncation => bootstrap but no carry.
        t, u = result["terminated"], result["truncated"]
        for row, expected in ((1, 1.), (4, 2.8), (2, 83.8)):
            actual = 1. + .9*2*(not t[row]) + .81*100*(not (t[row] or u[row]))
            self.assertAlmostEqual(actual, expected)

    def test_clip_end_no_reset_and_multiple_groups(self):
        groups = make_mixed_layout(8)["group_actor_rows"]
        terminal, truncation = np.zeros(8, dtype=bool), np.zeros(8, dtype=bool)
        clean = expand_group_reset(groups, terminal, truncation)
        self.assertFalse(clean["reset_mask"].any())
        self.assertEqual(clean["triggered_groups"].shape, (0,))
        truncation[2] = True  # AMASS clip-end, not a physical fall.
        clip = expand_group_reset(groups, terminal, truncation)
        self.assertFalse(clip["terminated"].any())
        np.testing.assert_array_equal(np.flatnonzero(clip["truncated"]), [2, 3, 6, 7])
        np.testing.assert_array_equal(np.flatnonzero(clip["collateral_reset"]), [3, 6, 7])
        np.testing.assert_array_equal(clip["triggered_groups"], [1])
        terminal[1] = True
        truncation[1] = True  # Preserve a genuine simultaneous terminal/cap flag.
        both = expand_group_reset(groups, terminal, truncation)
        np.testing.assert_array_equal(both["triggered_groups"], [0, 1])
        self.assertTrue(both["reset_mask"].all())
        self.assertTrue(both["truncated"].all())
        np.testing.assert_array_equal(both["terminated"], terminal)

    def test_explicit_irregular_group_rows_and_immutable_inputs(self):
        groups = np.array([[7, 0, 4, 3], [2, 6, 1, 5]], dtype=np.int64)
        terminal, truncation = np.zeros(8, dtype=bool), np.zeros(8, dtype=bool)
        terminal[3] = True
        before = [value.copy() for value in (groups, terminal, truncation)]
        result = expand_group_reset(groups, terminal, truncation)
        np.testing.assert_array_equal(np.flatnonzero(result["reset_mask"]), [0, 3, 4, 7])
        np.testing.assert_array_equal(np.flatnonzero(result["truncated"]), [0, 4, 7])
        np.testing.assert_array_equal(result["triggered_groups"], [0])
        for value, original in zip((groups, terminal, truncation), before):
            np.testing.assert_array_equal(value, original)
        # Returned masks own their values: subsequent caller mutation cannot leak back.
        result["terminated"][3] = False
        result["truncated"][0] = False
        self.assertTrue(terminal[3])
        self.assertFalse(truncation[0])

    def test_bad_layout_and_mask_contracts_are_rejected(self):
        for value in (0, 2, 6, -4, True, 8., np.int64(8)):
            with self.subTest(num_actors=value), self.assertRaises(ValueError):
                make_mixed_layout(value)
        groups = make_mixed_layout(8)["group_actor_rows"]
        masks = np.zeros(8, dtype=bool)
        bad_groups = [np.array([[0, 1, 4, 5], [0, 3, 6, 7]]), groups[:, :3],
                      groups.astype(float), groups.astype(bool), groups+1, groups-1]
        for bad in bad_groups:
            with self.subTest(groups=bad), self.assertRaises(ValueError):
                expand_group_reset(bad, masks, masks)
        for terminal, truncation in ((masks.astype(int), masks), (masks, masks.astype(int)),
                                     (masks[:6], masks[:6]), (masks, masks[:4]),
                                     (masks.reshape(2, 4), masks.reshape(2, 4))):
            with self.assertRaises(ValueError):
                expand_group_reset(groups, terminal, truncation)


if __name__ == "__main__":
    unittest.main()
