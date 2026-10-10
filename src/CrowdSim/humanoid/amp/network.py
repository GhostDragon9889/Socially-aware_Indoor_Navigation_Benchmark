"""Trainable PACER MLP structure and explicitly controlled running statistics.

Architecture reference: nv-tlabs/pacer, baeb6241f03fc9d78370a64e3c2dcef9d830d97f.
Inference matches AMPSeptBuilder and RunningMeanStd; new training owns its
task, history/action discriminator contract and initialization explicitly.
"""

from __future__ import annotations

import torch
from torch import nn


class RunningNormalizer(nn.Module):
    """Float64 statistics; PACER float32 normalization, epsilon and clipping.

    Forward never updates statistics implicitly. Call update on rollout data
    deliberately; population variance also handles a singleton batch safely.
    """

    def __init__(self, dimensions: int, epsilon: float = 1e-5):
        super().__init__()
        if dimensions <= 0 or epsilon <= 0:
            raise ValueError("Normalizer dimensions and epsilon must be positive")
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(dimensions, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(dimensions, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, values: torch.Tensor, *, unnorm: bool = False) -> torch.Tensor:
        if values.shape[-1] != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} normalizer dimensions")
        mean = self.running_mean.float()
        scale = torch.sqrt(self.running_var.float() + self.epsilon)
        if unnorm:
            return values.clamp(-5.0, 5.0) * scale + mean
        return ((values - mean) / scale).clamp(-5.0, 5.0)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.shape[-1] != self.dimensions or values.numel() == 0:
            raise ValueError("Invalid normalizer update shape")
        if not torch.isfinite(values).all():
            raise ValueError("Normalizer updates must be finite")
        samples = values.reshape(-1, self.dimensions).double()
        mean, variance = samples.mean(0), samples.var(0, correction=0)
        batch_count = samples.shape[0]
        total = self.count + batch_count
        delta = mean - self.running_mean
        combined = (
            self.running_var * self.count + variance * batch_count
            + delta.square() * self.count * batch_count / total
        )
        self.running_mean.add_(delta * batch_count / total)
        self.running_var.copy_(combined / total)
        self.count.copy_(total)


def _mlp(input_size: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_size, width), nn.ReLU()))
        input_size = width
    return nn.Sequential(*layers)


class PacerAMPNetwork(nn.Module):
    """Separate actor/critic, shared task encoder, and an independent D.

    eval_actor/critic accept already normalized concatenated self+task inputs.
    The default D is the public PACER reference's 10*195 state-only D. A new
    training run may explicitly supply another discriminator input dimension;
    this does not make the published PACER D action-conditioned.
    """

    def __init__(self, self_obs_size: int = 357, task_obs_size: int = 1044,
                 discriminator_obs_size: int = 1950, *, learn_sigma: bool = False):
        super().__init__()
        if min(self_obs_size, task_obs_size, discriminator_obs_size) <= 0:
            raise ValueError("Network input dimensions must be positive")
        self.self_obs_size = self_obs_size
        self.task_obs_size = task_obs_size
        self.discriminator_obs_size = discriminator_obs_size
        self._task_mlp = _mlp(task_obs_size, (512, 256))
        self.actor_mlp = _mlp(self_obs_size + 256, (2048, 1024))
        self.critic_mlp = _mlp(self_obs_size + 256, (2048, 1024))
        self.mu = nn.Linear(1024, 69)
        self.value = nn.Linear(1024, 1)
        self.sigma = nn.Parameter(torch.full((69,), -2.9), requires_grad=learn_sigma)
        self._disc_mlp = _mlp(discriminator_obs_size, (1024, 512))
        self._disc_logits = nn.Linear(512, 1)

    def _encoded_input(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        if normalized_observation.ndim != 2 or normalized_observation.shape[1] != self.self_obs_size + self.task_obs_size:
            raise ValueError("Actor/critic input violates the self+task observation contract")
        own = normalized_observation[:, :self.self_obs_size]
        task = normalized_observation[:, self.self_obs_size:]
        return torch.cat((own, self._task_mlp(task)), dim=-1)

    def eval_actor(self, normalized_observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(self.actor_mlp(self._encoded_input(normalized_observation)))
        return mu, self.sigma.expand_as(mu)

    def eval_critic(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_mlp(self._encoded_input(normalized_observation)))

    def eval_disc(self, normalized_history: torch.Tensor) -> torch.Tensor:
        if normalized_history.ndim != 2 or normalized_history.shape[1] != self.discriminator_obs_size:
            raise ValueError("Discriminator history dimensions do not match this network")
        return self._disc_logits(self._disc_mlp(normalized_history))
