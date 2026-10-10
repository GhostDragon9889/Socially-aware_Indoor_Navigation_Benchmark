"""Synthetic PRE/POST command contracts; not SDK or model quality evidence."""
from copy import deepcopy

import numpy as np
import pytest

from CrowdSim.humanoid.analysis.tracking_commands import audit_capture
from CrowdSim.humanoid.motion_tracking import body_tracking_errors


def fixture():
    reference = np.tile(np.arange(72, dtype=np.float32).reshape(24, 3) / 4, (4, 1, 1))
    reference[..., 0] += np.arange(4, dtype=np.float32)[:, None] / 8
    reference[..., 1] += np.arange(4, dtype=np.float32)[:, None] / 4
    actual = reference.copy()
    actual[..., 0] += np.array([9., .5, .625, .125], np.float32)[:, None]
    rotation = np.zeros((4, 24, 4), np.float32)
    rotation[..., 3] = 1
    metric = np.array([False, True, True, True])
    pre = np.array([0, 0, 1, 2])
    phase = np.array([9, 9, 10, 11], np.int64)
    requested = phase[:, None] + np.arange(15, 151, 15)
    command = reference[np.minimum(requested, 12) - 9, 0, :2][:, None]
    task = np.zeros((4, 1, 12308), np.float32)
    task[:, 0, :20] = (command[:, 0].astype(np.float64) - actual[pre, 0, :2].astype(np.float64)[:, None]).reshape(4, 20)
    arrays = {"metric_frame": metric, "native_reset_frame": ~metric, "action_valid": metric.copy(),
        "reference_frame": np.arange(9, 13, dtype=np.int64), "reference_time_s": np.arange(9, 13) / 30,
        "episode_id": np.ones((4, 1), np.int64), "fall_mask": np.array([False, False, True, False]),
        "root_pos": actual[:, None, 0], "root_rot": rotation[:, None, 0],
        "rigid_body_pos": actual[:, None], "rigid_body_rot": rotation[:, None],
        "reference_root_pos": reference[:, None, 0], "reference_root_rot": rotation[:, None, 0],
        "reference_rigid_body_pos": reference[:, None], "reference_rigid_body_rot": rotation[:, None],
        "trajectory_xy": command, "waypoint_query_phase_frame": phase, "waypoint_held_at_clip_end": requested > 12,
        "observation_actual_pelvis_quat_xyzw": rotation[pre, None, 0],
        "observation_episode_ids": np.ones((4, 1), np.int64), "observation_physics_step": 4 * pre,
        "observation_age": pre[:, None], "observation_used_task_heading": np.zeros((4, 1), np.float32),
        "observation_task_heading_defined": np.ones((4, 1), np.bool_),
        "observation_singular_heading_fallback": np.zeros((4, 1), np.bool_),
        "observation_physical_fall": np.zeros((4, 1), np.bool_), "observation_raw_task": task,
        **body_tracking_errors(actual, rotation, reference, rotation)}
    row = {"expected_control_steps": 3, "observed_metric_steps": 3, "complete_clip": True,
        "tracking_window": {"initial_frame": 9, "last_frame": 12, "available_steps": 3,
                            "requested_steps": 3, "eligible": True, "full_clip_requested": True},
        "fall_capture_frames": [2], "tracking_success": False, "first_tracking_failure_capture_frame": 2}
    return arrays, row


def test_PRE_commands_keep_all_POST_frames_and_endpoint_holds_without_claiming_reward_trace():
    arrays, row = fixture()
    result = audit_capture(arrays, row)
    assert result["metric_frames"] == 3 and result["metric_command_samples"] == 30
    assert result["held_endpoint_samples"] == 30 and result["all_ten_held_frames"] == 3
    assert result["fall_frames_retained"] == 1
    assert result["offline_reference_root_reward_sum"] == pytest.approx(np.exp(-2 * np.array([.5, .625, .125]) ** 2).sum())
    assert result["maximum_prefix_float64_difference_m"] == 0


@pytest.mark.parametrize("bad", ["POST_phase", "wrong_endpoint", "world_command", "PRE_quaternion",
    "observation_episode", "physical_clock", "age", "heading", "heading_definition",
    "lateral_sign", "normalized_prefix", "POST_root_prefix", "nan_prefix", "false_fallback"])
def test_concrete_channel_defects_fail_before_reporting_consistency(bad):
    arrays, row = fixture()
    arrays, row = deepcopy(arrays), deepcopy(row)
    if bad == "POST_phase":
        arrays["waypoint_query_phase_frame"] = arrays["reference_frame"].copy()
    elif bad == "wrong_endpoint":
        arrays["waypoint_held_at_clip_end"][2, 0] = False
    elif bad == "world_command":
        arrays["trajectory_xy"][2, 0, 0, 0] += .1
    elif bad == "PRE_quaternion":
        arrays["observation_actual_pelvis_quat_xyzw"][2, 0] = [0, 0, 1, 0]
    elif bad == "observation_episode":
        arrays["observation_episode_ids"][2, 0] += 1
    elif bad == "physical_clock":
        arrays["observation_physics_step"][2] += 4
    elif bad == "age":
        arrays["observation_age"][2] += 1
    elif bad == "heading":
        arrays["observation_used_task_heading"][2] = .01
    elif bad == "heading_definition":
        arrays["observation_task_heading_defined"][2] = False
    elif bad == "lateral_sign":
        arrays["observation_raw_task"][:, 0, 1:20:2] *= -1
    elif bad == "normalized_prefix":
        arrays["observation_raw_task"][:, 0, :20] *= 2
    elif bad == "POST_root_prefix":
        arrays["observation_raw_task"][:, 0, :20] = (arrays["trajectory_xy"][:, 0] - arrays["root_pos"][:, 0, None, :2]).reshape(4, 20)
    elif bad == "nan_prefix":
        arrays["observation_raw_task"][2, 0, 0] = np.nan
    else:
        arrays["observation_singular_heading_fallback"][2] = True
    with pytest.raises(ValueError):
        audit_capture(arrays, row)


def test_actual_singular_fallen_PRE_root_retains_same_episode_prior_heading():
    arrays, row = fixture()
    arrays["rigid_body_rot"] = arrays["rigid_body_rot"].copy()
    arrays["rigid_body_rot"][2, 0, 0] = [.5, .5, -.5, .5]
    arrays["root_rot"] = arrays["rigid_body_rot"][:, :, 0].copy()
    arrays["observation_actual_pelvis_quat_xyzw"] = arrays["root_rot"][[0, 0, 1, 2]].copy()
    arrays["observation_task_heading_defined"][3] = False
    arrays["observation_singular_heading_fallback"][3] = True
    arrays["observation_physical_fall"][3] = True
    arrays.update(body_tracking_errors(arrays["rigid_body_pos"][:, 0], arrays["rigid_body_rot"][:, 0],
        arrays["reference_rigid_body_pos"][:, 0], arrays["reference_rigid_body_rot"][:, 0]))
    assert audit_capture(arrays, row)["singular_heading_fallback_frames"] == 1
    arrays["observation_physical_fall"][3] = False
    with pytest.raises(ValueError, match="fallback"):
        audit_capture(arrays, row)
