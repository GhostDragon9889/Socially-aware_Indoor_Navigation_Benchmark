"""CPU sampling boundaries and path reversal fixtures, not physical evidence."""
import unittest

import numpy as np

from reader import interval_velocity, prefix_speeds


class PlannerResponseTests(unittest.TestCase):
    def test_control_end_on_knot_uses_previous_interval(self):
        raw = np.zeros((1, 51, 2), np.float32)
        raw[0, 1, 0] = 1
        selected = np.zeros((2, 1), np.int64)
        phase = np.array([[3], [4]], np.int64)
        np.testing.assert_array_equal(interval_velocity(raw, selected, phase)[:, 0], [[10., 0.], [-10., 0.]])

    def test_closed_prefix_keeps_arc_motion_even_with_zero_net_displacement(self):
        raw = np.zeros((1, 51, 2), np.float32)
        raw[0, 1:5, 0] = [1, 0, 1, 0]
        metric = prefix_speeds(raw)
        self.assertEqual(metric["0.5s_net_speed_mps"][0], 0)
        self.assertEqual(metric["0.5s_arc_speed_mps"][0], 8)
        self.assertAlmostEqual(metric["5s_arc_speed_mps"][0], .8)

    def test_nonfinite_or_outside_executed_prefix_fails(self):
        raw = np.zeros((1, 51, 2), np.float32)
        with self.assertRaises(ValueError):
            interval_velocity(raw, np.zeros((1, 1), np.int64), np.array([[16]], np.int64))
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prefix_speeds(raw)


if __name__ == "__main__":
    unittest.main()
