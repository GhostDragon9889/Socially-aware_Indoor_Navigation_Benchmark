"""Synthetic metric fixtures and a cold read of the actual registered plan."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import shared_social_episode_status as status


class EpisodeStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "metrics.jsonl"
        self.runtime = {"actors": 256, "updates": 2000, "rollout_controls": 32}
        self.row = {"update_count": 1, "physical_transitions": 8192,
                    "completed_physical_controls": 32, "update_wall_s": 16.,
                    "nested": {"values": [1., 2.]}}

    def save(self, row):
        self.path.write_text(json.dumps(row) + "\n")

    def test_partial_last_write_is_not_a_completed_update(self):
        self.save(self.row)
        with self.path.open("a") as stream:
            stream.write('{"update_count":2')
        rows, incomplete = status.read_metrics(self.path, self.runtime)
        self.assertEqual(rows, [self.row])
        self.assertTrue(incomplete)

    def test_missing_metrics_are_zero_observed_updates(self):
        self.assertEqual(status.read_metrics(self.path, self.runtime), ([], False))

    def test_sequence_control_budget_and_exact_integer_guards(self):
        for key, value in (("update_count", 2), ("update_count", True),
                           ("physical_transitions", 8192.), ("completed_physical_controls", 31)):
            with self.subTest(key=key, value=value):
                wrong = deepcopy(self.row)
                wrong[key] = value
                self.save(wrong)
                with self.assertRaisesRegex(ValueError, "Noncontiguous"):
                    status.read_metrics(self.path, self.runtime)

    def test_nested_nonfinite_or_nonpositive_time_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                wrong = deepcopy(self.row)
                wrong["nested"]["values"][1] = value
                self.save(wrong)
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    status.read_metrics(self.path, self.runtime)
        self.row["update_wall_s"] = 0
        self.save(self.row)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            status.read_metrics(self.path, self.runtime)

    def test_actual_status_does_not_infer_completion_or_load_weights(self):
        report = status.read_status()
        self.assertEqual(report["planned_updates"], 2000)
        self.assertEqual(report["planned_physical_transitions"], 16384000)
        self.assertLessEqual(report["logged_updates"], 2000)
        self.assertFalse(report["execution_success_or_quality_inferred_from_progress"])
        self.assertFalse(report["training_started_or_restarted"])
        self.assertFalse(report["weights_loaded"])
        # Other tests may already have imported Torch in this pytest process.
        cold = subprocess.run([sys.executable, "-c",
            "import sys; from scripts.shared_social_episode_status import read_status; "
            "read_status(); assert 'torch' not in sys.modules"],
            cwd=status.ROOT, capture_output=True, text=True)
        self.assertEqual(cold.returncode, 0, cold.stderr)

    def test_reused_owner_pid_removes_live_claim_and_eta(self):
        observer_path = status.ROOT / "docs/style_improvement_20261008/shared_social_episode_comparison_preparation_v1/comparison_owner_state.json"
        retained = json.loads(observer_path.read_text())["observed_episode_owner"]
        state = json.loads((status.PLAN.parent / "owner_state.json").read_text())
        training = next(row for row in state["steps"] if row["name"] == "pilot2000_training")
        wrong_owner = {**retained, "start_ticks": "reused", "parent_pid": 1}
        child = {"pid": training["pid"], "parent_pid": retained["pid"], "start_ticks": "fixture",
                 "command": training["command"]}
        with patch.object(status, "process_identity", side_effect=[wrong_owner, child]):
            report = status.read_status()
        self.assertFalse(report["owner_command_and_start_observed"])
        self.assertFalse(report["training_command_and_parent_observed"])
        self.assertIsNone(report["estimated_training_finish_Asia_Shanghai"])


if __name__ == "__main__":
    unittest.main()
