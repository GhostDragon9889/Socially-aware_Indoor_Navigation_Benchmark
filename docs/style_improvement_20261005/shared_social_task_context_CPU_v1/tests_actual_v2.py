"""CPU synthetic phase and target semantics; not physical quality evidence."""
from copy import deepcopy
import unittest

import numpy as np

from scripts import diagnose_shared_social_task_context as reader


def phase_fixture():
    shape = (3, 4)
    episode = np.ones(shape, dtype=np.int64)
    episode[2, 1] = 2
    reset = np.zeros(shape, dtype=np.bool_)
    reset[1, 1] = True
    pre = np.zeros(shape + (10, 195), np.float32)
    pre[..., 0, 6] = np.arange(12).reshape(shape)
    post = np.zeros_like(pre)
    post[:-1] = pre[1:]
    post[1, 1] = 42  # This terminal history is intentionally not the reset history.
    own = np.zeros(shape + (357,), np.float32)
    own[..., 213:216] = pre[..., 0, 6:9]
    age = np.array([[10, 10, 10, 10], [11, 11, 11, 11], [12, 0, 12, 12]], np.int64)
    rollout = {"episode_ids": episode, "history_episode_ids": np.repeat(episode[..., None], 10, axis=-1),
               "amp_history": pre, "self_obs": own, "task_obs": np.zeros(shape + (12308,), np.float32),
               "terminated": reset.copy(), "truncated": np.zeros_like(reset)}
    diagnostics = {"post_episode_ids": episode.copy(), "post_history_episode_ids": rollout["history_episode_ids"].copy(),
                   "post_amp_history": post, "reset_mask": reset, "pre_age": age, "post_age": age + 1,
                   **{key: np.zeros(shape, np.bool_) for key in
                      ("physical_fall", "nonfoot_normal_contact", "task_heading_defined", "singular_heading_fallback")}}
    metadata = {"controls": 3, "actors": 4, "control_step_start": 20, "control_step_end": 23,
                "physics_step_start": 80, "physics_step_end": 92}
    for kind, scale in (("control", 1), ("physics", 4)):
        clocks = np.broadcast_to((20 * scale + scale * np.arange(3))[:, None], shape).copy()
        diagnostics[kind + "_step_before"] = clocks
        diagnostics[kind + "_step_after"] = clocks + scale
    return rollout, diagnostics, metadata


class TaskContextTests(unittest.TestCase):
    def test_packet_tuple_json_list_conversion_keeps_values_and_types(self):
        self.assertTrue(reader.same_json_document({"layers": (2, 1)}, {"layers": [2, 1]}))
        self.assertFalse(reader.same_json_document({"layers": (2, 1)}, {"layers": [1, 2]}))
        self.assertFalse(reader.same_json_document({"layers": (2, 1)}, {"layers": [2, True]}))

    def test_future_segment_rates_exclude_existing_position_lag(self):
        prefix = np.zeros((1, 10, 2), np.float32)
        prefix[0, :, 0] = 4 + .5 * np.arange(1, 11)
        descriptors = reader.describe_targets(prefix)
        self.assertEqual(descriptors["root_to_first_future_target_rate"].item(), 9)
        np.testing.assert_array_equal(descriptors["future_segment_rate"], np.ones((1, 9)))
        self.assertEqual(descriptors["median_future_segment_rate"].item(), 1)
        rotated = prefix[..., ::-1].copy()
        np.testing.assert_array_equal(reader.describe_targets(rotated)["future_segment_rate"], descriptors["future_segment_rate"])

    def test_stationary_future_point_can_still_have_large_catchup_rate(self):
        prefix = np.zeros((1, 10, 2), np.float32)
        prefix[..., 0] = 2
        descriptors = reader.describe_targets(prefix)
        self.assertEqual(descriptors["root_to_first_future_target_rate"].item(), 4)
        self.assertEqual(descriptors["median_future_segment_rate"].item(), 0)

    def test_continuity_uses_original_episode_and_retains_reset_rows(self):
        rollout, diagnostics, metadata = phase_fixture()
        result = reader.verify_phase(rollout, diagnostics, metadata)
        self.assertEqual(result["same_episode_PRE_previous_POST_exact_rows"], 7)
        self.assertEqual(result["first_control_unpaired_rows"], 4)
        self.assertEqual(result["adjacent_reset_unpaired_rows"], 1)
        self.assertEqual(result["all_actor_controls"], 12)
        self.assertFalse(result["any_metrics_rows_removed"])

    def test_post_state_substitution_and_episode_crossing_are_rejected(self):
        for modification in ("post", "episode", "age", "root_slot"):
            rollout, diagnostics, metadata = phase_fixture()
            if modification == "post":
                diagnostics["post_amp_history"][0, 0, 0, 6] += 1
            elif modification == "episode":
                rollout["history_episode_ids"][0, 0, 1] += 1
            elif modification == "age":
                diagnostics["pre_age"][2, 1] = 12
            else:
                rollout["self_obs"][0, 0, 213] += 1
            with self.assertRaises(ValueError):
                reader.verify_phase(rollout, diagnostics, metadata)

    def test_dropped_physics_clock_and_unowned_reset_are_rejected(self):
        rollout, diagnostics, metadata = phase_fixture()
        bad = deepcopy(diagnostics)
        bad["physics_step_after"][1, 2] -= 1
        with self.assertRaisesRegex(ValueError, "native clock gap"):
            reader.verify_phase(rollout, bad, metadata)
        bad = deepcopy(diagnostics)
        bad["reset_mask"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "Reset/termination"):
            reader.verify_phase(rollout, bad, metadata)

    def test_no_normalized_dtype_or_nonfinite_target_fallback(self):
        for prefix in (np.zeros((1, 10, 2), np.float64), np.full((1, 10, 2), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                reader.describe_targets(prefix)
        rollout, diagnostics, metadata = phase_fixture()
        rollout["task_obs"] = rollout["task_obs"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            reader.verify_phase(rollout, diagnostics, metadata)


if __name__ == "__main__":
    unittest.main()
