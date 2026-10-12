"""Numerical observations, camera preprocessing and terminal reward contracts."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.control.drive import yaw_to_quaternion
from navigation_harness import attach_memory_environment, make_manager


@pytest.fixture
def manager(tmp_path):
    navigation = make_manager(
        tmp_path, rl_depth_enabled=True, rl_rgb_enabled=True, rl_rgb_size=16
    )
    attach_memory_environment(navigation)
    yield navigation
    navigation._close_trajectory_logs()


def camera(manager, **output):
    manager.env.crowdsim_robot_camera = NS(data=NS(output=output))


def test_camera_letterbox_keeps_metric_depth_invalid_pixels_and_uint8_rgb(manager):
    depth = torch.full((4, 8, 16, 1), 2.5)
    depth[0, 0, :4, 0] = torch.tensor([float("nan"), float("inf"), 0.01, 6.0])
    rgb = torch.ones((4, 8, 16, 3)) * 0.5
    camera(manager, distance_to_image_plane=depth, rgb=rgb)
    obs, neighbors, mask, result, patch = manager.get_robot_rl_observations()
    assert result.shape == (2, 224, 224) and torch.isfinite(result).all()
    assert result[:, 0].count_nonzero() == 0 and result[1, 112, 112] == 0.5
    assert result.min() == 0 and result.max() <= 1.0
    assert (
        manager.robot_rl_depth_obs_dim == 224**2 and manager.robot_rl_map_obs_dim == 64
    )
    image = manager._robot_last_rgb
    assert image.shape == (2, 3, 16, 16) and image.dtype == torch.uint8
    assert image[:, :, 0].count_nonzero() == 0 and image[0, 0, 8, 8] == 127
    rgb_uint = torch.full((4, 8, 16, 3), 213, dtype=torch.uint8)
    camera(manager, rgb=rgb_uint)
    assert manager._read_camera_rgb()[0, 0, 8, 8] == 213
    camera(manager, distance_to_image_plane=torch.full((4, 8, 16), 6.0))
    assert not manager._read_camera_depth().any()


def test_failed_camera_reads_use_zero_then_last_valid_frame_and_warn_once(
    manager, caplog
):
    manager.get_robot_rl_observations()
    assert not manager._robot_last_depth.any() and not manager._robot_last_rgb.any()
    camera(
        manager, distance_to_image_plane=torch.ones(4, 8, 8), rgb=torch.ones(4, 8, 8, 3)
    )
    manager.get_robot_rl_observations()
    saved_depth, saved_rgb = (
        manager._robot_last_depth.clone(),
        manager._robot_last_rgb.clone(),
    )
    camera(manager)
    manager.get_robot_rl_observations()
    manager.get_robot_rl_observations()
    torch.testing.assert_close(manager._robot_last_depth, saved_depth)
    torch.testing.assert_close(manager._robot_last_rgb, saved_rgb)
    assert (
        sum(
            "falling back to last valid depth" in item.message
            for item in caplog.records
        )
        == 1
    )


@pytest.mark.parametrize(
    "depth,rgb",
    [
        (torch.ones(4, 8), torch.ones(4, 8, 8)),
        (torch.ones(0, 8, 8), torch.ones(0, 8, 8, 3)),
        ("bad-depth", "bad-rgb"),
    ],
)
def test_malformed_camera_transport_is_rejected_with_bounded_warnings(
    manager, caplog, depth, rgb
):
    camera(manager, distance_to_image_plane=depth, rgb=rgb)
    for _ in range(3):
        assert (
            manager._read_camera_depth() is None and manager._read_camera_rgb() is None
        )
    assert len(caplog.records) <= 2


def test_invalid_depth_range_is_rejected_and_disabled_map_has_no_channel(manager):
    camera(manager, distance_to_image_plane=torch.ones(4, 8, 8))
    manager.config.rl_depth_max_range = 0
    assert manager._read_camera_depth() is None
    manager.config.rl_map_size = 0
    assert manager._read_map_patch() is None and manager.robot_rl_map_obs_dim == 0
    assert manager._local_obstacle_patch(np.array([3, 3]), 0).shape == (0, 0)


def test_ego_goal_motion_and_neighbor_features_are_independent_of_filler_rows(manager):
    positions = np.array(
        [[4.0, 3.0], [10.0, 10.0], [3.0, 3.0], [3.0, 5.0]], dtype=np.float32
    )
    velocities = np.array(
        [[0.5, 0.2], [0.0, 0.0], [0.2, 0.0], [0.0, -0.3]], dtype=np.float32
    )
    manager.robot.data.root_quat_w[:2] = yaw_to_quaternion(
        torch.tensor([0.0, np.pi / 2])
    )
    manager.drive._executed_angular_velocities[:] = [0.5, -0.5]
    manager.goals_xy[2:] = [[5.0, 3.0], [3.0, 7.0]]
    result = manager._build_robot_rl_observations(positions, velocities)
    np.testing.assert_allclose(
        result, [[0.2, 0, 1, 0.2, 0.5], [0.2, 0, 1, -0.3, -0.5]], atol=1e-6
    )
    neighbors, mask = manager._build_robot_neighbor_observations(positions, velocities)
    assert mask.tolist() == [[True, True, False, False], [True, True, False, False]]
    np.testing.assert_allclose(neighbors[0, 0], [0.25, 0, 1, 0.3, 0.2], atol=1e-6)
    np.testing.assert_allclose(neighbors[0, 1], [0.5, 1, 0, -0.2, -0.3], atol=1e-6)
    assert not neighbors[:, 2:].any()
    manager.config.rl_num_neighbors = 0
    feat, mask = manager._build_robot_neighbor_observations(positions, velocities)
    assert feat.shape == (2, 0, 5) and mask.shape == (2, 0)


@pytest.mark.parametrize(
    "terminal", ["reached", "collision", "blocked", "timeout", "stuck"]
)
def test_terminal_rewards_sum_exactly_and_terminal_states_are_exempt_from_stuck(
    manager, terminal
):
    positions = np.array(
        [[8.0, 8.0], [10.0, 10.0], [3.0, 3.0], [6.0, 6.0]], dtype=np.float32
    )
    velocities = np.zeros_like(positions)
    manager.config.rl_stuck_window = 3
    manager._stuck_pos_buf = np.tile(positions[2:, None, :], (1, 3, 1))
    manager._robot_episode_steps[:] = 2
    manager._update_robot_progress_targets(positions)
    pairs = set()
    if terminal == "reached":
        manager.reached[2] = True
    elif terminal == "collision":
        pairs = {(2, 0)}
    elif terminal == "blocked":
        manager.drive.wall_blocked[0] = True
    elif terminal == "timeout":
        manager.config.rl_max_episode_steps = 3
    manager._update_robot_rl_feedback(positions, velocities, pairs)
    info = manager._robot_last_info
    assert info["stuck"][0].item() == (terminal == "stuck")
    assert manager._robot_last_dones[0]
    components = [
        info[key]
        for key in (
            "reward_time",
            "reward_progress",
            "reward_velocity_direction",
            "reward_smoothness",
            "reward_proximity",
            "reward_static_proximity",
            "reward_goal",
            "reward_collision",
            "reward_timeout",
            "reward_stuck",
        )
    ]
    torch.testing.assert_close(sum(components), manager._robot_last_rewards)
    if terminal != "stuck":
        key = "collision" if terminal == "blocked" else terminal
        assert info[key][0] and not info["reward_stuck"][0]
