"""Exercise actual A*, SFM, robot feedback and reset flow on a CPU transport."""

import json

import numpy as np
import pytest
import torch

from navigation_harness import attach_memory_environment, make_manager


@pytest.fixture
def navigation(tmp_path):
    manager = make_manager(tmp_path)
    env, state = attach_memory_environment(manager)
    yield manager, env, state
    manager._close_trajectory_logs()


def test_constructor_routes_and_frozen_slots_are_traversable(navigation):
    manager, env, _ = navigation
    assert manager.num_agents == 4 and manager._update_interval_steps == 2
    assert manager.sfm_controller.dt == pytest.approx(0.08)
    assert all(len(path) >= 2 for path in manager.paths_xy)
    for path in manager.paths_xy:
        assert not manager.collision.hits_wall(path).any()
    spawn = manager.humanoid_spawn_xy_for_all_envs(env.num_envs).numpy()
    np.testing.assert_array_equal(spawn[:2], manager.starts_xy[:2])
    assert not manager.collision.hits_wall(spawn).any()
    poses = manager.robot_spawn_poses_for_all_envs(env.num_envs)
    assert poses.shape == (4, 3)
    assert torch.isfinite(poses).all()
    np.testing.assert_array_equal(poses[:2, :2], manager.starts_xy[2:])
    assert not manager.collision.hits_wall(poses[:, :2].numpy()).any()


def test_navigation_updates_observations_and_jsonl_follow_actual_transport(navigation):
    manager, env, state = navigation
    manager.set_robot_rl_actions(torch.tensor([[0.6, 0.2], [0.4, -0.2]]))
    initial = manager.drive.positions_xy().copy()
    env.step(None)
    obs, neighbors, mask, depth, patch = manager.get_robot_rl_observations()
    assert obs.shape == (2, 5) and neighbors.shape == (2, 4, 5)
    assert mask.dtype == torch.bool and depth is None and patch.shape == (2, 8, 8)
    assert torch.isfinite(obs).all() and torch.isfinite(neighbors).all()
    assert set(patch.unique().tolist()) <= {0.0, 1.0}
    moved = np.linalg.norm(manager.drive.positions_xy() - initial, axis=1)
    np.testing.assert_allclose(moved, [0.024, 0.016], atol=1e-6)
    assert env.extras["crowdsim_navigation"]["num_agents"] == 4
    assert manager._humanoid_velocity_initialized.all()
    assert np.all(
        np.linalg.norm(manager._humanoid_smoothed_velocities, axis=1) <= 0.12 + 1e-6
    )
    state.root_pos[2:, :2] = 999  # filler slots never enter SFM/robot observations
    positions, _ = manager._read_agent_state()
    assert len(positions) == 4 and (positions < 999).all()
    manager._close_trajectory_logs()
    rows = [
        json.loads(line)
        for line in manager.trajectory_log_path.read_text().splitlines()
    ]
    assert rows[0]["type"] == "metadata" and len(rows) == 3
    assert rows[-1]["step"] == 1 and len(rows[-1]["positions_xy"]) == 4
    assert np.isfinite(rows[-1]["sfm_desired_velocities_xy"]).all()


def test_terminal_feedback_survives_partial_repeat_reset(navigation):
    manager, env, _ = navigation
    positions, velocities = manager._read_agent_state()
    manager.reached[2] = True
    manager.collision_pairs = {(2, -1), (0, 1)}
    manager._update_robot_progress_targets(positions)
    manager._update_robot_rl_feedback(positions, velocities, {(2, -1)})
    saved = manager.get_robot_rl_feedback()
    assert saved[2].tolist() == [True, False]
    assert saved[3]["reached"].tolist() == [True, False]
    goal, start = manager.goals_xy[2].copy(), manager.starts_xy[2].copy()
    untouched = env.crowdsim_robot.data.root_pos_w[1:].clone()
    manager.reset_robot_rl_episodes(
        torch.tensor([True, False]), repeat_mask=np.array([True, False])
    )
    np.testing.assert_array_equal(manager.goals_xy[2], goal)
    np.testing.assert_array_equal(manager.starts_xy[2], start)
    torch.testing.assert_close(env.crowdsim_robot.data.root_pos_w[1:], untouched)
    assert manager.collision_pairs == {(0, 1)} and not manager.reached[2]
    assert manager._robot_episode_steps[0] == 0 and not manager._stuck_pos_buf[0].any()
    assert saved[2][0] and saved[3]["reached"][0] and saved[3]["collision"][0]
    assert not manager._robot_last_dones.any()


def test_resampling_reset_replans_and_records_route_before_spawn(navigation):
    manager, env, _ = navigation
    before = manager.starts_xy.copy()
    manager.reset_robot_rl_episodes(np.array([True, True]))
    assert not np.array_equal(manager.starts_xy[2:], before[2:])
    assert (
        np.linalg.norm(np.diff(manager.starts_xy[2:], axis=0))
        >= manager.config.min_spawn_spacing
    )
    np.testing.assert_array_equal(
        env.crowdsim_robot.data.root_pos_w[:2, :2], manager.starts_xy[2:]
    )
    rows = [
        json.loads(line)
        for line in manager.trajectory_log_path.read_text().splitlines()
    ]
    assert len(rows[-1]["path_updates"]) == 2
    paths = json.loads(manager.path_log_path.read_text())["agents"]
    assert paths[2]["start_xy"] == manager.starts_xy[2].tolist()
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(
        env._crowdsim_desired_root_xy[0], torch.from_numpy(manager.starts_xy[0])
    )
    assert manager._humanoid_velocity_initialized[0]


def test_skipped_navigation_step_clears_every_reward_component(navigation):
    manager, env, _ = navigation
    manager.config.rl_static_proximity_danger_penalty = -2.0
    positions, velocities = manager._read_agent_state()
    positions[2] = [0.15, 2.0]
    manager._update_robot_progress_targets(positions)
    manager._update_robot_rl_feedback(positions, velocities, set())
    assert manager._robot_last_info["reward_static_proximity"][0] < 0
    manager.env_step_count = 1  # scheduler skips this navigation update
    env.step(None)
    assert not manager._robot_last_rewards.any() and not manager._robot_last_dones.any()
    assert all(
        not value.any()
        for key, value in manager._robot_last_info.items()
        if key.startswith("reward_")
    )


def test_humanoid_only_flow_suppresses_frozen_slot_resets(tmp_path):
    manager = make_manager(tmp_path, num_robots=0)
    try:
        env, _ = attach_memory_environment(manager)
        env.reset_buf[:] = True
        env.step(None)
        assert not env.reset_buf[2:].any() and not env.progress_buf[2:].any()
        assert manager.robot_starts_xy_yaw.shape == (0, 3)
        assert manager._build_robot_rl_observations(
            manager.starts_xy, np.zeros((2, 2))
        ).shape == (0, 5)
        assert manager._read_map_patch() is None
    finally:
        manager._close_trajectory_logs()


@pytest.mark.parametrize(
    "field",
    [
        "humanoid_max_acceleration",
        "humanoid_max_yaw_rate",
        "humanoid_prediction_horizon",
        "humanoid_ttc_threshold",
        "rl_goal_observation_max_distance",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_nonfinite_positive_configuration_is_rejected_before_output(
    tmp_path, field, value
):
    with pytest.raises(ValueError, match="positive"):
        make_manager(tmp_path, **{field: value})
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((1, 2)),
        np.zeros((2, 3)),
        np.zeros(2),
        np.array([[float("inf"), 0], [0, 0]]),
        np.array([[float("nan"), 0], [0, 0]]),
    ],
)
@pytest.mark.parametrize("attached", [False, True])
def test_invalid_actions_do_not_mutate_manager_or_drive(tmp_path, invalid, attached):
    manager = make_manager(tmp_path)
    try:
        if attached:
            attach_memory_environment(manager)
        manager.set_robot_rl_actions(np.array([[0.3, 0.4], [0.5, -0.6]]))
        before, previous = (
            manager._robot_rl_actions.copy(),
            manager._robot_rl_prev_actions.copy(),
        )
        drive_before = manager.drive._actions.copy() if manager.drive else None
        with pytest.raises(ValueError):
            manager.set_robot_rl_actions(invalid)
        np.testing.assert_array_equal(manager._robot_rl_actions, before)
        np.testing.assert_array_equal(manager._robot_rl_prev_actions, previous)
        if manager.drive:
            np.testing.assert_array_equal(manager.drive._actions, drive_before)
    finally:
        manager._close_trajectory_logs()
