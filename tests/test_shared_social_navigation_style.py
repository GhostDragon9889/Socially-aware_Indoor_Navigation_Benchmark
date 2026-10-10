"""CPU selection/admission regressions; fixtures are not physical evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import evaluate_shared_social_navigation_style as style


def features(count, speed=.2, offset=0.):
    return {"speed_m_s": np.full(count, speed),
            "local_pose": np.arange(count * 45).reshape(count, 45) * .0001 + offset,
            "local_pose_velocity": np.ones((count, 45)) * offset}


class NavigationStyleTests(unittest.TestCase):
    def test_public_sfm_provenance_uses_actual_generator_not_compatibility_argument(self):
        sfm = {"trajectory_source": "public_sfm_joint_future", "arguments": {"trajectory_source": "diffusion"},
               "high_level_planner": {"schema": "local.public_sfm.shared_AMP_fixture_bridge.v1",
                   "diffusion_weights_loaded": False, "diffusion_guidance_applied": False,
                   "actual_world_boundary_checked": True}}
        style.validate_planner("sfm", sfm)
        with self.assertRaisesRegex(ValueError, "wrong planner"):
            style.validate_planner("diffusion", sfm)
        sfm["high_level_planner"]["diffusion_weights_loaded"] = True
        with self.assertRaisesRegex(ValueError, "actual public planner"):
            style.validate_planner("sfm", sfm)

    def test_unequal_populations_use_identical_reference_indices_and_balanced_counts(self):
        runs = {"diffusion": features(35), "sfm": features(80, offset=.1)}
        reference = features(100)
        report = style.matched_motion_bins(runs, reference)
        row = next(value for value in report["bins"] if value["status"] == "supported")
        self.assertEqual(row["sample_count_each"], 35)
        ri = np.array(row["reference_sample_indices"])
        self.assertEqual(len(ri), 35)
        for name, score in row["scores"].items():
            gi = np.array(score["generated_sample_indices"])
            self.assertEqual(len(gi), 35)
            spec = style.StyleSpec()
            gf = np.concatenate((runs[name]["local_pose"][gi],
                                 spec.motion_velocity_time_scale_s * runs[name]["local_pose_velocity"][gi]), -1)
            rf = np.concatenate((reference["local_pose"][ri],
                                 spec.motion_velocity_time_scale_s * reference["local_pose_velocity"][ri]), -1)
            self.assertEqual(score["MMD2"], style.gaussian_motion_mmd2(gf, rf, sigma=spec.motion_kernel_sigma))

    def test_missing_arm_support_stays_unknown_and_has_no_zero_score(self):
        result = style.matched_motion_bins({"diffusion": features(29), "sfm": features(90)}, features(90))
        self.assertEqual(result["common_supported_bin_count"], 0)
        self.assertEqual(result["equal_common_bin_mean_MMD2"], {"diffusion": None, "sfm": None})
        self.assertTrue(all(row["scores"] is None for row in result["bins"]))

    def test_fixed_bin_boundary_and_cap(self):
        # Exactly0.5m/s belongs to [0.5,1), not [0.1,0.5).
        result = style.matched_motion_bins({"diffusion": features(400, .5), "sfm": features(500, .5)}, features(600, .5))
        row = next(value for value in result["bins"] if value["scores"] is not None)
        self.assertEqual(row["lower_m_s"], .5)
        self.assertEqual(row["sample_count_each"], 256)
        self.assertEqual(row["reference_sample_indices"][0], 0)
        self.assertEqual(row["reference_sample_indices"][-1], 599)

    def test_swapping_arms_does_not_change_scores(self):
        left, right, reference = features(50), features(40, offset=.3), features(90)
        original = style.matched_motion_bins({"A": left, "B": right}, reference)
        reverse = style.matched_motion_bins({"B": right, "A": left}, reference)
        self.assertEqual(original["equal_common_bin_mean_MMD2"], reverse["equal_common_bin_mean_MMD2"])

    def test_pending_or_boolean_exit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture_owner.json"
            for code in (None, False, 1):
                path.write_text(json.dumps({"schema": "shared_social.actual_high_level_CPU_audit_execution.v1",
                    "actual_exit_code": code, "terminal_tool_chunk": "fixture_only"}))
                with self.assertRaisesRegex(ValueError, "externally observed"):
                    style.read_completed_run(path)

    def test_smoke_audit_cannot_be_admitted_as_200s(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture_owner.json"
            audit = Path(temporary) / "fixture_audit.json"
            audit.write_text(json.dumps({"schema": "shared_social.all_registered_controls_native_CPU_audit.v1",
                "status": "passed", "actual_controls": 30}))
            path.write_text(json.dumps({"schema": "shared_social.actual_high_level_CPU_audit_execution.v1",
                "actual_exit_code": 0, "terminal_tool_chunk": "fixture_only", "audit": style.record(audit)}))
            with self.assertRaisesRegex(ValueError, "complete200s"):
                style.read_completed_run(path)

    def test_record_identity_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.json"
            path.write_text('{}')
            identity = style.record(path)
            path.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "input changed"):
                style.bound(identity)


if __name__ == "__main__":
    unittest.main()
