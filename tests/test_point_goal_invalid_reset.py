"""Bad reset selections and headings must fail explicitly."""

import pytest
import torch

from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig


@pytest.mark.parametrize("shape", [(2,), (3, 1), (), (0,)])
def test_invalid_reset_mask_preserves_current_parallel_state(shape):
    env = PointGoalBatchEnv(
        PointGoalEnvConfig(num_envs=3, num_neighbors=0), torch.device("cpu")
    )
    fields = (
        "position",
        "goal",
        "heading",
        "neighbor_position",
        "neighbor_velocity",
        "last_action",
        "episode_step",
        "episode_return",
    )
    before = {name: getattr(env, name).clone() for name in fields}
    with pytest.raises(ValueError, match="reset mask must have shape"):
        env.reset(torch.zeros(shape, dtype=torch.bool))
    for name in fields:
        torch.testing.assert_close(getattr(env, name), before[name])


@pytest.mark.parametrize(
    "minimum,maximum",
    [
        (-1.0, 180.0),
        (10.0, 5.0),
        (0.0, 181.0),
        (float("nan"), 180.0),
        (0.0, float("inf")),
    ],
)
def test_invalid_initial_heading_configuration_is_rejected(minimum, maximum):
    with pytest.raises(ValueError, match="Initial heading offsets"):
        PointGoalBatchEnv(
            PointGoalEnvConfig(
                num_envs=1,
                num_neighbors=0,
                rl_initial_heading_min_offset_degrees=minimum,
                rl_initial_heading_max_offset_degrees=maximum,
            ),
            torch.device("cpu"),
        )
