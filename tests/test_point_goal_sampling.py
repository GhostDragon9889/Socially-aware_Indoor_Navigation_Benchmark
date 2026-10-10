import math

import pytest
import torch

from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig


@pytest.mark.parametrize("distance", [(2., 4.), (11., 12.), (math.sqrt(2) * 8.5, math.sqrt(2) * 8.5)])
def test_reset_satisfies_distance_heading_and_map_bounds(distance):
    torch.manual_seed(42)
    env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=10_000, num_neighbors=0,
        min_start_goal_distance=distance[0], max_start_goal_distance=distance[1],
        rl_initial_heading_min_offset_degrees=5., rl_initial_heading_max_offset_degrees=15.), torch.device("cpu"))
    delta = env.goal - env.position
    lengths = delta.norm(dim=-1)
    offset = env._wrap_angle(torch.atan2(delta[:, 1], delta[:, 0]) - env.heading).abs()
    assert torch.all(lengths >= distance[0] - 1e-5)
    assert torch.all(lengths <= distance[1] + 1e-5)
    assert torch.all(offset >= math.radians(5.) - 1e-5)
    assert torch.all(offset <= math.radians(15.) + 1e-5)
    for value in (env.position, env.goal):
        assert torch.all(value >= .75 - 1e-5)
        assert torch.all(value <= 9.25 + 1e-5)


def test_subset_reset_preserves_other_slots_and_zero_neighbors_can_step():
    env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=3, num_neighbors=0), torch.device("cpu"))
    before = [value.clone() for value in (env.position, env.goal, env.heading)]
    env.reset(torch.tensor([False, True, False]))
    for value, original in zip((env.position, env.goal, env.heading), before):
        torch.testing.assert_close(value[[0, 2]], original[[0, 2]])
    _, reward, done, info = env.step(torch.zeros(3, 2))
    assert reward.isfinite().all()
    assert not info["collision"].any()
    assert done.shape == (3,)


@pytest.mark.parametrize("kwargs,match", [
    ({"world_size": 1.5}, "world_size"),
    ({"min_start_goal_distance": 13., "max_start_goal_distance": 14.}, "diagonal"),
    ({"min_start_goal_distance": 4., "max_start_goal_distance": 2.}, "Goal distances"),
    ({"num_neighbors": -1}, "nonnegative"),
])
def test_unsatisfiable_sampling_configuration_has_clear_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PointGoalBatchEnv(PointGoalEnvConfig(num_envs=1, **kwargs), torch.device("cpu"))


def test_collision_and_reaching_together_is_not_success():
    env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=1, num_neighbors=1), torch.device("cpu"))
    env.goal[:] = env.position
    env.neighbor_position[:] = env.position[:, None]
    env.neighbor_velocity[:] = 0
    _, _, done, info = env.step(torch.zeros(1, 2))
    assert done.item() and info["reached"].item() and info["collision"].item()
    assert not info["success"].item()
