# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real core PPO updates with singleton rollout/minibatch moments."""

from datetime import timedelta
import faulthandler
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import AdvantageNormalizationConfig
from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.agents.utils.training import aggregate_scalar_metrics


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
    old_distribution = torch.distributions.Normal(obs * agent.actor_module.weight.detach(), agent.actor_module.logstd.exp())
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


def test_distributed_ema_uses_all_rank_samples_and_between_rank_variance():
    agent = _agent(use_ema=True)
    agent.config.advantage_normalization.clamp_range = 10.0
    agent.config.advantage_normalization.ema_alpha = 0.5
    agent.fabric.world_size = 2
    calls = []

    def all_gather(value):
        calls.append(value)
        if isinstance(value, int):
            return torch.tensor([1, 3])
        if len(calls) == 1:
            return torch.tensor([1.0, 5.0])
        return torch.tensor([0.0, 8.0 / 3.0])

    agent.fabric.all_gather = all_gather
    _rollout(agent, 1)
    global_values = torch.tensor([1.0, 3.0, 5.0, 7.0])
    torch.testing.assert_close(agent.adv_mean_ema, 0.5 * global_values.mean())
    torch.testing.assert_close(agent.adv_std_ema, 0.5 * global_values.std(unbiased=False) + 0.5)


class _GlooFabric:
    world_size = 2
    device = torch.device("cpu")

    def __init__(self, rank):
        self.global_rank = rank

    def all_gather(self, value):
        value = torch.as_tensor(value)
        gathered = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(gathered, value)
        return torch.stack(gathered)

    def broadcast(self, value, src):
        dist.broadcast(value, src)
        return value

    def backward(self, loss):
        loss.backward()


def _distributed_ppo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(60, repeat=True)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        agent = _agent(use_ema=True)
        agent.fabric = _GlooFabric(rank)
        agent.actor = nn.parallel.DistributedDataParallel(agent.actor)
        agent.critic = nn.parallel.DistributedDataParallel(agent.critic)
        agent.config.advantage_normalization.clamp_range = 10.0
        agent.config.advantage_normalization.ema_alpha = 0.5
        values = torch.tensor([1.0, 1.0, 2.0, 3.0])
        expected_mean, expected_std = torch.tensor(0.0), torch.tensor(1.0)
        initial_actor = agent.actor_module.weight.detach().clone()
        initial_critic = agent.critic_module.weight.detach().clone()
        for iteration in range(2):
            batch = _rollout(agent, 1 if rank == 0 else 3)
            expected_mean = 0.5 * values.mean() + 0.5 * expected_mean
            expected_std = 0.5 * values.std(unbiased=False) + 0.5 * expected_std
            torch.testing.assert_close(agent.adv_mean_ema, expected_mean)
            torch.testing.assert_close(agent.adv_std_ema, expected_std)
            logs = PPO.perform_optimization_step(agent, batch, batch_idx=iteration)
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in logs.values())
            for parameter in (agent.actor_module.weight, agent.critic_module.weight):
                gathered = agent.fabric.all_gather(parameter.detach())
                torch.testing.assert_close(gathered[0], gathered[1])
        assert agent.actor_module.weight.item() != initial_actor.item()
        assert agent.critic_module.weight.item() != initial_critic.item()
        metrics = aggregate_scalar_metrics(
            {"loss": torch.tensor([float(rank + 1)])}, agent.fabric,
            weight=1 if rank == 0 else 3,
        )
        assert metrics["loss"] == 1.75
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs CPU Gloo distributed backend")
def test_real_cpu_distributed_ppo_updates_with_unequal_environment_counts(tmp_path):
    workers = mp.spawn(_distributed_ppo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU distributed PPO workers did not exit within 90 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
