import torch

from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig


def test_environment_shapes_and_step():
    env = PointGoalBatchEnv(
        PointGoalEnvConfig(num_envs=3, num_neighbors=2), torch.device("cpu")
    )
    obs, neighbors, mask = env.observe()
    assert obs.shape == (3, 5)
    assert neighbors.shape == (3, 2, 5)
    assert mask.shape == (3, 2)
    next_state, reward, done, info = env.step(torch.zeros(3, 2))
    assert next_state[0].shape == obs.shape
    assert reward.shape == done.shape == (3,)
    assert set(info) == {
        "reached", "success", "collision", "timeout", "episode_return",
        "episode_length", "goal_distance",
        "terminal_observation",
    }


def test_terminal_observation_preserves_pre_reset_state_and_only_done_envs_reset():
    env = PointGoalBatchEnv(
        PointGoalEnvConfig(num_envs=4, num_neighbors=1, max_episode_steps=2,
                           neighbor_speed=0.0), torch.device("cpu")
    )
    env.position[:] = torch.tensor([2.0, 2.0])
    env.goal[:] = torch.tensor([6.0, 2.0])
    env.heading.zero_()
    env.neighbor_position[:] = torch.tensor([8.0, 8.0])
    env.neighbor_velocity.zero_()
    env.goal[2] = env.position[2]
    env.neighbor_position[3, 0] = env.position[3]
    env.episode_step[:] = torch.tensor([0, 1, 1, 1])
    before = env.observe()

    next_state, _, done, info = env.step(torch.zeros(4, 2))

    assert done.tolist() == [False, True, True, True]
    assert info["timeout"].tolist() == [False, True, False, False]
    assert env.episode_step.tolist() == [1, 0, 0, 0]
    for expected, terminal, following in zip(before, info["terminal_observation"], next_state):
        torch.testing.assert_close(terminal, expected)
        torch.testing.assert_close(following[0], expected[0])
    # Terminal observations must own their data, so subsequent resets cannot
    # overwrite the value-function inputs for the completed transition.
    env.reset()
    for expected, terminal in zip(before, info["terminal_observation"]):
        torch.testing.assert_close(terminal, expected)
