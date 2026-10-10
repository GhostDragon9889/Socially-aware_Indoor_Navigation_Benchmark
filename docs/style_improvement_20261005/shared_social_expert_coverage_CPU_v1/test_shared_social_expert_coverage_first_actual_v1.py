"""Synthetic metadata probabilities and boundary guards; not native evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import diagnose_shared_social_expert_coverage as reader


def fixture():
    clips = [{"motion_id": index, "frames": frames, "parent_source_path": "parent" + str(index),
              "dataset": "dataset", "subject_namespace": "subject", "style_class": "turning_like"}
             for index, frames in enumerate((12, 14))]
    semantics = [{"original_v7_train_row": deepcopy(clip), "no_train_annotation_match": index == 1,
                  "raw_time_alignment": {"full_sample_time_range_union_covered": index == 0}}
                 for index, clip in enumerate(clips)]
    return clips, [np.array([0, 1], np.int64), np.array([1, 1, 1, 1], np.int64)], semantics


class ExpertCoverageTests(unittest.TestCase):
    def test_clip_uniform_probability_differs_from_window_uniform(self):
        clips, labels, semantics = fixture()
        report, selected = reader.prior_and_exposure(clips, labels, np.array([0, 1, 0], np.int64),
            np.array([9, 12, 10], np.int64), ("low_speed", "fast"), semantics)
        low = report["strata"]["low_speed"]
        self.assertEqual(low["actual_clip_uniform_legal_center_probability"], .25)
        self.assertEqual(low["window_uniform_fraction"], 1 / 6)
        self.assertEqual(low["terminal_sample_count"], 1)
        self.assertEqual(low["expected_terminal_count_under_original_prior"], .75)
        self.assertEqual(selected.tolist(), [0, 1, 1])
        self.assertEqual(report["terminal"]["counts"]["BABEL_train_coverage"]["no_train_annotation_match"], 1)
        self.assertFalse(report["terminal"]["entire_training_exposure_proven"])

    def test_unknown_partial_and_multiaction_coverage_are_not_action_labels(self):
        clips, labels, semantics = fixture()
        semantics[0]["raw_time_alignment"]["full_sample_time_range_union_covered"] = False
        report, _ = reader.prior_and_exposure(clips, labels, np.array([0, 1], np.int64),
            np.array([9, 9], np.int64), ("low_speed", "fast"), semantics)
        counts = report["terminal"]["counts"]["BABEL_train_coverage"]
        self.assertEqual(counts, {"matched_without_full_clip_timed_union": 1, "no_train_annotation_match": 1})
        self.assertEqual(report["terminal"]["counts"]["whole_clip_proxy"], {"turning_like": 2})

    def test_ids_history_future_bounds_registry_and_original_rows_are_strict(self):
        clips, labels, semantics = fixture()
        for ids, frames in (([2], [9]), ([0], [8]), ([0], [11])):
            with self.assertRaises(ValueError):
                reader.prior_and_exposure(clips, labels, np.array(ids, np.int64), np.array(frames, np.int64),
                                          ("low_speed", "fast"), semantics)
        with self.assertRaises(ValueError):
            reader.prior_and_exposure(clips, labels, np.array([0.], np.float64), np.array([9], np.int64),
                                      ("low_speed", "fast"), semantics)
        bad = deepcopy(labels)
        bad[0] = bad[0][:-1]
        with self.assertRaisesRegex(ValueError, "every legal center"):
            reader.prior_and_exposure(clips, bad, np.array([0], np.int64), np.array([9], np.int64),
                                      ("low_speed", "fast"), semantics)
        semantics[0]["original_v7_train_row"]["frames"] = 13
        with self.assertRaisesRegex(ValueError, "original train row"):
            reader.prior_and_exposure(clips, labels, np.array([0], np.int64), np.array([9], np.int64),
                                      ("low_speed", "fast"), semantics)

    def test_bound_artifact_changed_or_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.json"
            path.write_text(json.dumps({"synthetic": True}))
            expected = reader.identity(path)
            self.assertEqual(reader.bound(expected), {"synthetic": True})
            path.write_text(json.dumps({"synthetic": False}))
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.bound(expected)
            link = Path(temporary) / "link"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "ordinary"):
                reader.identity(link)


if __name__ == "__main__":
    unittest.main()
