"""Real CPU PPO updates must consume singleton rollouts and tail batches."""

from copy import deepcopy
import importlib
import math

import pytest
import torch


@pytest.mark.parametrize("module_name", ["CrowdSim.ppo.ppo_policy", "crowdsim_ppo.ppo_policy"])
@pytest.mark.parametrize("update_method", ["update", "update_with_kl"])
@pytest.mark.parametrize("rollout_steps", [1, 3])
def test_ppo_singleton_batches_update_all_samples_with_finite_state(
    module_name, update_method, rollout_steps, monkeypatch
):
    module = importlib.import_module(module_name)
    torch.manual_seed(19)
    device = torch.device("cpu")
    config = module.RobotPPOConfig(
        obs_dim=3, hidden_dims=(8,), depth_enabled=False, map_enabled=False,
        num_neighbors=1, ppo_epochs=2, minibatch_size=2,
    )
    trainer = module.RobotPPOTrainer(config, device)
    reference = deepcopy(trainer.model).eval()
    buffer = module.RobotRolloutBuffer(rollout_steps, 1, 3, 2, device, num_neighbors=1)
    neighbors = torch.zeros(1, 1, 5)
    neighbor_mask = torch.zeros(1, 1, dtype=torch.bool)
    for step in range(rollout_steps):
        obs = torch.tensor([[0.1 * step, 0.2, -0.3]])
        _, raw_actions, log_probs, values = trainer.act(
            obs, neighbors=neighbors, neighbor_mask=neighbor_mask
        )
        buffer.add(obs, raw_actions, log_probs, torch.tensor([1.0 + step]),
                   torch.tensor([False]), values, neighbors=neighbors,
                   neighbor_mask=neighbor_mask)
    buffer.compute_returns_and_advantages(torch.zeros(1), gamma=0.0, gae_lambda=0.0)

    original_step = trainer.optimizer.step
    completed_steps = []

    def checked_step(*args, **kwargs):
        # A NaN guard must not silently discard the last sample. Verify both
        # gradient finiteness and the number of actual optimizer updates.
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()
                   if p.grad is not None)
        result = original_step(*args, **kwargs)
        completed_steps.append(1)
        return result

    monkeypatch.setattr(trainer.optimizer, "step", checked_step)
    if update_method == "update_with_kl":
        stats = trainer.update_with_kl(buffer, reference, kl_beta=0.3)
    else:
        stats = trainer.update(buffer)

    assert len(completed_steps) == config.ppo_epochs * math.ceil(rollout_steps / config.minibatch_size)
    assert all(math.isfinite(value) for value in stats.values())
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    assert all(torch.isfinite(value).all() for state in trainer.optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))
