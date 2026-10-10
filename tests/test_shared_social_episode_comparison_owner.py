"""Actual Linux process identity contracts; no simulator or network imports."""
import os
import unittest

from scripts import own_shared_social_episode_comparison as observer


class ComparisonObserverProcessTests(unittest.TestCase):
    def test_current_process_has_stable_start_ticks_and_actual_argv(self):
        before = observer.process_identity(os.getpid())
        after = observer.process_identity(os.getpid())
        self.assertEqual(before, after)
        self.assertEqual(before["pid"], os.getpid())
        self.assertTrue(before["start_ticks"].isdigit())
        self.assertGreater(int(before["start_ticks"]), 0)
        self.assertTrue(before["command"])
        self.assertTrue(all(isinstance(value, str) and value for value in before["command"]))

    def test_absent_process_is_not_reported_live(self):
        self.assertIsNone(observer.process_identity(2**31 - 1))


if __name__ == "__main__":
    unittest.main()
