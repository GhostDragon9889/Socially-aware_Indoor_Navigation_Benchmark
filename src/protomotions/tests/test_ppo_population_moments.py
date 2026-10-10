# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real core PPO updates with singleton rollout/minibatch moments."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import AdvantageNormalizationConfig
from protomotions.agents.utils.data import ExperienceBuffer


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.logstd = nn.Parameter(torch.tensor([-0.3]), requires_grad=False)

    def forward(self, td, log_internals=False):
        td["mean_action"] = td["obs"] * self.weight
        return td

    def compute_model_loss(self, td, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, td):
        td["value"] = td["obs"] * self.weight
        return td

    def compute_model_loss(self, td, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


def _agent(use_ema, shift_mean=True):
    agent = object.__new__(PPO)
    agent.device = torch.device("cpu")
    agent.current_epoch = 1
    agent.e_clip = 0.2
    agent.gamma = 0.99
    agent.tau = 0.95
    agent._skip_actor_for_epoch = False
    agent.actor = _Actor()
    agent.critic = _Critic()
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=0.01)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=0.01)
    agent.adv_mean_ema = torch.zeros(())
    agent.adv_std_ema = torch.ones(())
    agent.fabric = SimpleNamespace(
        global_rank=0, world_size=1,
        broadcast=lambda value, src: value,
        backward=lambda loss: loss.backward(),
    )
    agent.config = SimpleNamespace(
        normalize_rewards=False, task_reward_w=1.0,
        advantage_normalization=AdvantageNormalizationConfig(use_ema=use_ema, shift_mean=shift_mean),
        bounds_loss_coef=0.0, entropy_coef=0.0, clip_critic_loss=False,
        actor_clip_frac_threshold=None, gradient_clip_val=0.0,
        check_grad_mag=False, fail_on_bad_grads=True,
        adaptive_lr=SimpleNamespace(enabled=False),
        model=SimpleNamespace(actor=SimpleNamespace(learnable_std=False)),
    )
    agent.calculate_extra_actor_loss = lambda batch: (torch.zeros(()), {})
    return agent


def _rollout(agent, count):
    agent.num_envs = count
    buffer = ExperienceBuffer(num_envs=count, num_steps=1, device=agent.device)
    agent.experience_buffer = buffer
    for key, shape in (
        ("dones", ()), ("rewards", ()), ("returns", ()), ("advantages", ()),
        ("value", (1,)), ("next_value", (1,)),
    ):
        buffer.register_key(key, shape)
    buffer.dones.fill_(1)
    buffer.rewards.copy_(torch.arange(1, count + 1, dtype=torch.float).unsqueeze(0))
    PPO.pre_process_dataset(agent)
    obs = torch.arange(1, count + 1, dtype=torch.float).unsqueeze(-1)
    action = torch.full((count, 1), 0.5)
    old_distribution = torch.distributions.Normal(obs * agent.actor.weight.detach(), agent.actor.logstd.exp())
    return {
        "obs": obs, "action": action,
        "neglogp": -old_distribution.log_prob(action).sum(-1),
        "advantages": buffer.advantages.flatten().clone(),
        "returns": buffer.returns.flatten().clone(),
        "value": torch.zeros(count, 1),
    }


@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("shift_mean", [False, True])
def test_singleton_then_larger_rollout_updates_parameters_and_keeps_stats_finite(use_ema, shift_mean):
    agent = _agent(use_ema, shift_mean)
    initial_critic = agent.critic.weight.detach().clone()
    initial_actor = agent.actor.weight.detach().clone()
    for count in (1, 3):
        batch = _rollout(agent, count)
        logs = PPO.perform_optimization_step(agent, batch, batch_idx=0)
        assert torch.isfinite(agent.experience_buffer.advantages).all()
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in logs.values())
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in agent._adv_norm_log.values())
        assert torch.isfinite(agent.adv_std_ema)
        assert all(torch.isfinite(parameter).all() for model in (agent.actor, agent.critic) for parameter in model.parameters())
        for optimizer in (agent.actor_optimizer, agent.critic_optimizer):
            assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))
    assert agent.critic.weight.item() != initial_critic.item()
    assert agent.actor.weight.item() != initial_actor.item()


def test_distributed_batch_moments_include_between_rank_variance_for_singletons():
    agent = _agent(use_ema=False)
    calls = []

    def all_gather(value):
        if isinstance(value, int):
            return torch.tensor([1, 1])
        calls.append(value)
        if len(calls) == 1:
            return torch.tensor([1.0, 3.0])
        assert value.item() == 0.0  # Each local singleton has zero population variance.
        return torch.zeros(2)

    agent.fabric.world_size = 2
    agent.fabric.all_gather = all_gather
    batch = _rollout(agent, 1)
    # Combined raw advantages [1,3] have mean 2 and population variance 1.
    assert torch.allclose(batch["advantages"], torch.tensor([-1.0]))
