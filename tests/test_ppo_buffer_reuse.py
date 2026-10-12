"""Reused rollout slots must never retain observations from an older rollout."""

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    "module_name", ["crowdsim_ppo.ppo_policy", "CrowdSim.ppo.ppo_policy"]
)
def test_missing_optional_channels_after_reset_are_zero_not_previous_values(
    module_name,
):
    module = importlib.import_module(module_name)
    buffer = module.RobotRolloutBuffer(
        1,
        1,
        5,
        2,
        torch.device("cpu"),
        depth_size=4,
        map_size=4,
        num_neighbors=1,
        neighbor_dim=5,
    )
    common = (
        torch.zeros(1, 5),
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1),
    )
    buffer.add(
        *common,
        depth=torch.ones(1, 4, 4),
        map_patch=torch.ones(1, 4, 4),
        neighbors=torch.ones(1, 1, 5),
        neighbor_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert buffer.depth.sum() == 16 and buffer.map.sum() == 16
    buffer.reset()
    buffer.add(*common)
    assert buffer.depth.count_nonzero() == 0
    assert buffer.map.count_nonzero() == 0
    assert buffer.neighbors.count_nonzero() == 0
    assert buffer.neighbor_mask.count_nonzero() == 0
