"""Physical-context replay must mask resets and report disc collisions honestly."""

import numpy as np
import pytest

from scripts.benchmark_paper_clean_guidance import capture_context, scene_plan_metrics


def test_capture_context_does_not_infer_reset_teleport_velocity():
    frames = 130
    positions = np.zeros((frames, 2, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(frames)[:, None] / 30.
    positions[:, 1, 1] = 2.
    positions[60:, 0, 0] += 100  # A physical reset teleport, never a velocity.
    quaternion = np.zeros((frames, 2, 4), dtype=np.float32)
    quaternion[..., 3] = 1
    episodes = np.zeros((frames, 2), dtype=np.int64)
    episodes[60:, 0] = 1
    arrays = dict(root_pos=positions, root_rot=quaternion, episode_ids=episodes)
    context = capture_context(arrays, 120)
    assert context["states"][0, 2] == pytest.approx(1., abs=2e-4)
    assert not context["history_mask"][0, :11].any()
    assert context["history_mask"][0, 11:].all()
    assert (context["history"][0, :11] == 0).all()
    assert context["history_mask"][1].all()
    assert not context["neighbor_mask"].any()  # Current distance is >15m.
    reset_now = np.where(np.arange(frames)[:, None] >= 120, 2, episodes)
    with pytest.raises(ValueError, match="current root velocity"):
        capture_context({**arrays, "episode_ids": reset_now}, 120)


def test_static_target_adherence_does_not_hide_obstacle_or_joint_collisions():
    states = np.array([[1., 1., 1., 0., 0.], [1., 1.7, 1., 0., 0.]])
    xy = np.repeat(states[:, None, :2], 51, axis=1)
    xy[..., 0] += np.arange(51)[None] * .1
    targets, times = xy[:, -1:], np.array([5.])
    scene = {"bounds_xy": [0., 0., 20., 20.], "cuboids": [{"center": [4., 1., 1.], "size": [1., 1., 2.]}]}
    metrics = scene_plan_metrics(states, xy, targets, times, scene)
    assert metrics["endpoint_error_mean_m"] == pytest.approx(0)
    assert metrics["first_half_second_goal_progress_mean_m"] == pytest.approx(.5)
    assert metrics["obstacle_collision_point_fraction"] > 0
    assert metrics["joint_plan_neighbor_collision_point_fraction"] == 1.
    assert metrics["minimum_joint_neighbor_surface_gap_m"] == pytest.approx(-.1)
    assert metrics["max_segment_speed_m_s"] == pytest.approx(1.)
