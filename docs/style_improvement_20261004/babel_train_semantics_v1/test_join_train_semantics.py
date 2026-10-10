"""Synthetic metadata checks only; do not execute the real join."""
import unittest

from join_train_semantics import (
    clip_time_and_coverage,
    feature_key,
    label_evidence,
    merge_intervals,
    source_key,
)


class JoinMetadataTests(unittest.TestCase):
    def test_exact_source_path_and_no_basename_fuzz(self):
        self.assertEqual(feature_key("ACCAD/ACCAD/take/A4 - look_poses.npz"),
                         "ACCAD/take/A4 - look_poses.npz")
        self.assertNotEqual(feature_key("ACCAD/ACCAD/take/A4___look_poses.npz"),
                            "ACCAD/take/A4 - look_poses.npz")
        self.assertEqual(source_key("/assets/motion/raw/KIT/572/a.npz", "/assets/motion/raw"),
                         "KIT/572/a.npz")
        for bad in ("/KIT/KIT/a.npz", "KIT/../KIT/a.npz", "KIT//KIT/a.npz", "KIT/KIT/a.motion"):
            with self.assertRaises(ValueError):
                feature_key(bad)
        with self.assertRaises(ValueError):
            source_key("/assets/motion/other/KIT/572/a.npz", "/assets/motion/raw")

    def test_union_requires_no_gap_and_uses_raw_slice_origin(self):
        self.assertEqual(merge_intervals([(2, 3), (1, 2), (1.5, 2.5)]), [[1, 3]])
        row = {"parent_start_time_s": 1.0, "parent_end_time_s": 3.0}
        sidecar = {"source_fps": 60.0, "source_frame_slice": [30, 600]}
        def item(a, b):
            return {"timed_status": "valid_timed_metadata", "ref": {"interval": [a, b]},
                    "original_label": {"start_t": a, "end_t": b}}
        result = clip_time_and_coverage(row, sidecar, [item(1.5, 2.5), item(2.5, 3.5)])
        self.assertEqual((result["first_sample_time_s"], result["last_sample_time_s"]), (1.5, 3.5))
        self.assertTrue(result["full_sample_time_range_union_covered"])
        gap = clip_time_and_coverage(row, sidecar, [item(1.5, 2.0), item(3.0, 3.5)])
        self.assertFalse(gap["full_sample_time_range_union_covered"])
        self.assertEqual(gap["valid_label_interval_union_s"], [[1.5, 2.0], [3.0, 3.5]])
        self.assertEqual(row, {"parent_start_time_s": 1.0, "parent_end_time_s": 3.0})

    def test_multifamily_labels_nulls_and_reversed_interval_are_retained(self):
        walk = {"raw_label": "walk", "proc_label": "walk", "act_cat": None,
                "seg_id": "w", "start_t": 0.0, "end_t": 3.0}
        reverse = {"raw_label": "turn", "proc_label": "turn", "act_cat": ["turn"],
                   "seg_id": "r", "start_t": 3.0, "end_t": 1.0}
        record = {"babel_sid": 8, "dur": 4.0, "seq_ann": None,
                  "seq_anns": [{"anntr_id": "a", "labels": [{"raw_label": "look", "proc_label": "look"}]}],
                  "frame_anns": [{"anntr_id": "a", "labels": [walk]},
                                 {"anntr_id": "b", "labels": [reverse]}]}
        evidence, diagnostics = label_evidence("8", record, "extra_train.json")
        self.assertEqual(len(evidence), 3)
        self.assertEqual([x["ref"]["anntr_id"] for x in evidence], ["a", "a", "b"])
        self.assertIs(evidence[1]["original_label"], walk)
        self.assertEqual(evidence[1]["null_or_empty_fields"], ["act_cat"])
        self.assertEqual(evidence[2]["timed_status"], "reversed_interval")
        self.assertIn({"family": "seq_ann", "state": "null"}, diagnostics)
        aligned = clip_time_and_coverage({"parent_start_time_s": 0, "parent_end_time_s": 3.5},
                                        {"source_fps": 30, "source_frame_slice": [0, 120]}, evidence)
        self.assertEqual(aligned["valid_label_interval_union_s"], [[0.0, 3.0]])
        self.assertFalse(aligned["full_sample_time_range_union_covered"])
        self.assertEqual(reverse["start_t"], 3.0)


if __name__ == "__main__":
    unittest.main()
