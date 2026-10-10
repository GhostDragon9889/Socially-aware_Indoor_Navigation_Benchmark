"""Synthetic convention boundaries only; not action-time ground truth."""
from copy import deepcopy
import unittest

from scripts import extend_babel_dense_sequence_semantics as reader


def parent_fixture():
    label = {"raw_label": "walking", "proc_label": "walk", "act_cat": None}
    ref = {"member": "train.json", "record_key": "fixture", "family": "seq_ann", "label_index": 0}
    return {"parent_index": 0, "parent_source_path": "synthetic_motion", "exact_raw_source_key": "synthetic_key",
            "raw_duration_frames_over_fps_s": 4.02, "BABEL_minus_raw_duration_s": -.02,
            "matched_train_member_records": {"train.json": {"record_key": "fixture", "original_record": {
                "dur": 4.0, "seq_ann": {"mul_act": False, "labels": [label]}, "frame_ann": None},
                "label_evidence": [{"ref": ref, "original_label": label}]}}}


def clip_fixture(first=0, last=3.9, explicitly_covered=False):
    return {"train_index": 0, "original_v7_train_row": {"motion_id": 0}, "parent_index": 0,
            "raw_time_alignment": {"first_sample_time_s": first, "last_sample_time_s": last,
                                   "full_sample_time_range_union_covered": explicitly_covered},
            "no_train_annotation_match": False}


class DenseSequenceConventionTests(unittest.TestCase):
    def test_original_join_strips_only_packaging_prefix_and_retains_subject_path(self):
        feature_key, _ = reader.original_feature_key()
        self.assertEqual(feature_key("CMU/CMU/45/45_01_poses.npz"), "CMU/45/45_01_poses.npz")
        self.assertNotEqual(feature_key("CMU/CMU/45/45_01_poses.npz"), feature_key("CMU/CMU/46/45_01_poses.npz"))
        for invalid in ("/CMU/45/45_01_poses.npz", "CMU/../45/45_01_poses.npz"):
            with self.assertRaises(ValueError):
                feature_key(invalid)

    def test_dense_single_sequence_preserves_null_categories_and_original_data(self):
        parent = parent_fixture()
        original = deepcopy(parent)
        assumption = reader.dense_parent_assumption(parent)
        self.assertTrue(assumption["eligible"])
        self.assertEqual(assumption["assumed_interval_s"], [0, 4.0])
        self.assertTrue(assumption["null_category_retained"])
        self.assertFalse(assumption["measured_frame_interval"])
        self.assertEqual(parent, original)
        row = clip_fixture()
        projected = reader.project_to_clip(row, assumption)
        self.assertTrue(projected["assumption_covers_full_clip"])
        self.assertFalse(projected["original_explicit_timed_union_covers_full_clip"])
        self.assertNotIn("start_t", assumption["original_label"])

    def test_extra_annotation_is_never_used_as_dense_fallback(self):
        parent = parent_fixture()
        parent["matched_train_member_records"]["extra_train.json"] = parent["matched_train_member_records"].pop("train.json")
        result = reader.dense_parent_assumption(parent)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "no_dense_train_match")

    def test_existing_frames_and_multi_action_uncertainty_remain_excluded(self):
        for field, value in (("frame_ann", {}), ("mul_act", True), ("mul_act", 0), ("labels", [])):
            parent = parent_fixture()
            record = parent["matched_train_member_records"]["train.json"]["original_record"]
            if field == "frame_ann":
                record[field] = value
            else:
                record["seq_ann"][field] = value
            self.assertFalse(reader.dense_parent_assumption(parent)["eligible"])

    def test_assumed_span_is_not_stretched_to_raw_duration_or_clip_end(self):
        assumption = reader.dense_parent_assumption(parent_fixture())
        row = clip_fixture(3.0, 4.02, explicitly_covered=True)
        projected = reader.project_to_clip(row, assumption)
        self.assertFalse(projected["assumption_covers_full_clip"])
        self.assertEqual(projected["assumed_positive_duration_overlap_s"], [3.0, 4.0])
        self.assertTrue(projected["original_explicit_timed_union_covers_full_clip"])
        row = clip_fixture(4.1, 4.2)
        self.assertIsNone(reader.project_to_clip(row, assumption)["assumed_positive_duration_overlap_s"])

    def test_invalid_duration_time_conflicts_and_evidence_ownership_are_rejected(self):
        for duration in (False, -1, float("nan")):
            parent = parent_fixture()
            parent["matched_train_member_records"]["train.json"]["original_record"]["dur"] = duration
            self.assertFalse(reader.dense_parent_assumption(parent)["eligible"])
        parent = parent_fixture()
        parent["matched_train_member_records"]["train.json"]["original_record"]["seq_ann"]["labels"][0]["end_t"] = 5
        self.assertFalse(reader.dense_parent_assumption(parent)["eligible"])
        parent = parent_fixture()
        parent["matched_train_member_records"]["train.json"]["label_evidence"][0]["ref"]["member"] = "extra_train.json"
        with self.assertRaisesRegex(ValueError, "evidence ownership"):
            reader.dense_parent_assumption(parent)
        row = clip_fixture()
        row["parent_index"] = 1
        with self.assertRaisesRegex(ValueError, "parent ownership"):
            reader.project_to_clip(row, reader.dense_parent_assumption(parent_fixture()))


if __name__ == "__main__":
    unittest.main()
