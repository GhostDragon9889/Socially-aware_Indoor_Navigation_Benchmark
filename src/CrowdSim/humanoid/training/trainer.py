"""Actual Torch PPO with explicitly declared AMP discriminator inputs.

The architecture follows public PACER's separate actor/critic+shared task MLP.
The default discriminator consumes applied actions as printed in NavIsaacLab
Eq2. Its legacy initialization is random; an explicit state-prior strategy
transfers public history weights/moments with zero action columns. An explicit
state-only adaptation transfers the matching complete public discriminator.
Unspecified optimizer/feature choices are exposed in config and provenance.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import generalized_advantages, validate_rollout
from ..application.settings import TrainingConfig
from ..application.state_prior_initialization import (EXTENSION, STRATEGY,
    validate_action_state_prior_config, validate_action_state_prior_initialization)
from .replay import AMPPolicyReplay
from .terrain_statistics import PUBLIC_PACER_SHA256, TerrainStatisticsPolicy




class RunningNormalizer(nn.Module):
    """Population moments in float64, checkpointed and frozen during a rollout."""

    def __init__(self, dimension, clip=5.):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(dimension, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, len(self.mean)).double()
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Normalizer updates require nonempty finite observations")
        mean = values.mean(0)
        variance = values.var(0, unbiased=False)
        batch_count = len(values)
        delta = mean - self.mean
        total = self.count + batch_count
        combined = self.variance * self.count + variance * batch_count + delta.square() * self.count * batch_count / total
        self.mean.add_(delta * batch_count / total)
        self.variance.copy_(combined / total)
        self.count.copy_(total)

    def normalize(self, values, *, clamp=True):
        normalized = (values - self.mean.to(values.dtype)) / torch.sqrt(self.variance.to(values.dtype) + 1e-5)
        return normalized.clamp(-self.clip, self.clip) if clamp else normalized

    def denormalize(self, values):
        return values * torch.sqrt(self.variance.to(values.dtype) + 1e-5) + self.mean.to(values.dtype)


def _mlp(input_dim, layers, output_dim=None):
    modules = []
    for width in layers:
        modules.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    if output_dim is not None:
        modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class PaperAMPNetwork(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.task_encoder = _mlp(config.task_dim, config.task_layers)
        encoded_dim = config.self_dim + config.task_layers[-1]
        self.actor = _mlp(encoded_dim, config.policy_layers, 69)
        self.critic = _mlp(encoded_dim, config.policy_layers, 1)
        self.discriminator_input = config.discriminator_input
        action_dimensions = 69 if config.discriminator_input == "state_action" else 0
        self.discriminator = _mlp(config.history_frames * config.amp_frame_dim + action_dimensions, config.discriminator_layers, 1)
        self.register_buffer("log_std", torch.full((69,), config.log_std))

    def policy(self, self_obs, task_obs):
        features = torch.cat((self_obs, self.task_encoder(task_obs)), dim=-1)
        mean = self.actor(features)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return distribution, self.critic(features).squeeze(-1)

    def disc_logits(self, history, applied_actions):
        features = history.flatten(-2)
        if self.discriminator_input == "state_action":
            features = torch.cat((features, applied_actions), dim=-1)
        return self.discriminator(features).squeeze(-1)


class PaperAMPTrainer:
    """Physical runner supplies observations, action effects and episode masks.

    ``act`` returns Gaussian samples and clamped applied PD actions separately.
    Compute reward after the real physics step from actual joint torque/velocity;
    collect next_values before resets. ``update`` uses those actual rollouts.
    No simulator, fabricated episode or reference checkpoint is created here.
    """

    checkpoint_version = 1

    def __init__(self, config: TrainingConfig, *, device="cpu", provenance=None):
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.provenance = dict(provenance or {})
        if config.mode == "strict_author_experiment" and not self.provenance.get("strict_manifest_gate_passed"):
            raise ValueError("Strict author training requires the complete verified author manifest gate")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES
        frozen = {"L_Hand", "R_Hand"}
        if config.freeze_toes:
            frozen.update(("L_Toe", "R_Toe"))
        physical_mask = tuple(int(name.rsplit("_", 1)[0] not in frozen) for name in PACER_DOF_NAMES)
        if config.action_mask is None:
            config.action_mask = physical_mask
        elif any(config.action_mask[index] for index, value in enumerate(physical_mask) if not value):
            raise ValueError("Declared action mask cannot enable joints frozen by the physical PD decoder")
        self.action_mask = torch.tensor(config.action_mask, device=self.device, dtype=torch.float32)
        self.model = PaperAMPNetwork(config).to(self.device)
        self.self_normalizer = RunningNormalizer(config.self_dim, config.normalization_clip).to(self.device)
        self.task_normalizer = RunningNormalizer(config.task_dim, config.normalization_clip).to(self.device)
        amp_dimensions = config.amp_frame_dim * (config.history_frames if config.amp_normalization == "per_history_coordinate" else 1)
        self.amp_normalizer = RunningNormalizer(amp_dimensions, config.normalization_clip).to(self.device)
        self.value_normalizer = RunningNormalizer(1, config.normalization_clip).to(self.device)
        policy_parameters = list(self.model.task_encoder.parameters()) + list(self.model.actor.parameters()) + list(self.model.critic.parameters())
        self.optimizer = torch.optim.Adam(policy_parameters, lr=config.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=config.discriminator_learning_rate)
        self.update_count = 0
        self.physical_transition_count = 0
        self._pending_rng_state = None
        self.freeze_normalizers = False
        self.terrain_statistics_policy = TerrainStatisticsPolicy(config.task_terrain_statistics_mode)
        self.policy_replay = (AMPPolicyReplay(config.discriminator_replay_capacity,
            (config.history_frames, config.amp_frame_dim), device=self.device)
            if config.discriminator_replay_capacity else None)

    @classmethod
    def prepare(cls, config=None, **kwargs):
        return cls(config or TrainingConfig(), **kwargs)

    def _observations(self, self_obs, task_obs):
        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))
        self.terrain_statistics_policy.require_ready()
        if self_obs.shape[:-1] != task_obs.shape[:-1] or self_obs.shape[-1] != self.config.self_dim or task_obs.shape[-1] != self.config.task_dim:
            raise ValueError("Self/task observation shape does not match the declared feature contract")
        if not torch.isfinite(self_obs).all() or not torch.isfinite(task_obs).all():
            raise ValueError("Policy observations must be finite")
        if self.config.normalize_observations:
            return self.self_normalizer.normalize(self_obs), self.task_normalizer.normalize(task_obs)
        return self_obs, task_obs

    @torch.no_grad()
    def warm_start_pacer(self, checkpoint, dof_limits_lower, dof_limits_upper, **loader_kwargs):
        """Transfer the complete verified public actor/critic/task and obs stats.

        The default retains the legacy new action-conditioned discriminator.
        An explicit state-only baseline can transfer the complete public D
        and its per-history-coordinate moments without inventing expert actions.
        A separate explicit action strategy extends that state prior with zero
        action columns and copies the same AMP history moments before count cap.
        """
        from CrowdSim.humanoid.amp.controller import PacerAMPController

        validate_action_state_prior_config(self.config)
        if self.config.warm_start_action_discriminator_from_state_prior and (
                self.physical_transition_count or self.provenance.get("initialization") is not None
                or self.policy_replay is not None and self.policy_replay.size):
            raise ValueError("Action state prior requires a fresh uninitialized trainer before any physical transition/replay")
        self.terrain_statistics_policy.require_fresh_warm_start(self.update_count)
        if self.update_count or self.optimizer.state or self.discriminator_optimizer.state:
            raise ValueError("PACER warm start must happen before any training update")
        if not self.config.normalize_observations or not self.config.normalize_values or self.config.normalization_clip != 5.:
            raise ValueError("Public PACER warm start requires normalized observations/values with clip5")
        if (self.config.self_dim, self.config.task_dim, tuple(self.config.task_layers), tuple(self.config.policy_layers)) != (357, 1044, (512, 256), (2048, 1024)):
            raise ValueError("Warm start requires the complete public PACER actor/critic architecture")
        reference = PacerAMPController.from_checkpoint(checkpoint, dof_limits_lower,
            dof_limits_upper, device=self.device, **loader_kwargs)
        if ((self.config.task_terrain_statistics_mode == "preserve_declared_suffix"
                or self.config.warm_start_action_discriminator_from_state_prior)
                and reference.checkpoint_sha256 != PUBLIC_PACER_SHA256):
            raise ValueError("Protected terrain statistics require the verified public PACER checkpoint SHA")
        network = reference.network
        self.model.task_encoder.load_state_dict(network._task_mlp.state_dict(), strict=True)
        for destination, trunk, output in ((self.model.actor, network.actor_mlp, network.mu),
                                           (self.model.critic, network.critic_mlp, network.value)):
            for layer_index in (0, 2):
                destination[layer_index].load_state_dict(trunk[layer_index].state_dict(), strict=True)
            destination[4].load_state_dict(output.state_dict(), strict=True)
        self.model.log_std.copy_(network.sigma)
        obs = reference.observation_normalizer
        terrain_reference_count = (float(obs.count)
            if self.config.task_terrain_statistics_mode == "preserve_declared_suffix" else None)
        for destination, section in ((self.self_normalizer, slice(0, 357)),
                                      (self.task_normalizer, slice(357, 1401))):
            destination.mean.copy_(obs.running_mean[section])
            destination.variance.copy_(obs.running_var[section])
            destination.count.copy_(obs.count)
        value = reference.value_normalizer
        self.value_normalizer.mean.copy_(value.running_mean)
        self.value_normalizer.variance.copy_(value.running_var)
        self.value_normalizer.count.copy_(value.count)
        state_prior = self.config.warm_start_action_discriminator_from_state_prior
        transfer_disc = self.config.warm_start_discriminator
        if state_prior:
            self.model.discriminator[0].weight[:, :1950].copy_(network._disc_mlp[0].weight)
            self.model.discriminator[0].weight[:, 1950:].zero_()
            self.model.discriminator[0].bias.copy_(network._disc_mlp[0].bias)
            self.model.discriminator[2].load_state_dict(network._disc_mlp[2].state_dict(), strict=True)
            self.model.discriminator[4].load_state_dict(network._disc_logits.state_dict(), strict=True)
        if transfer_disc:
            for layer_index in (0, 2):
                self.model.discriminator[layer_index].load_state_dict(network._disc_mlp[layer_index].state_dict(), strict=True)
            self.model.discriminator[4].load_state_dict(network._disc_logits.state_dict(), strict=True)
        if transfer_disc or state_prior:
            amp = reference.discriminator_normalizer
            self.amp_normalizer.mean.copy_(amp.running_mean)
            self.amp_normalizer.variance.copy_(amp.running_var)
            self.amp_normalizer.count.copy_(amp.count)
        cap = self.config.warm_start_normalizer_count_cap
        original_counts = {}
        if cap is not None or state_prior:
            for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
                normalizer = getattr(self, name)
                original_counts[name] = float(normalizer.count)
                if cap is not None:
                    normalizer.count.clamp_max_(cap)
        self.provenance["initialization"] = {
            "mode": "transfer_warm_start_public_pacer_actor_critic", "path": str(Path(checkpoint).resolve()),
            "sha256": reference.checkpoint_sha256, "actor_critic_task_transferred": True,
            "observation_value_moments_transferred": True,
            "action_conditioned_discriminator_initialization": "independent_random" if self.config.discriminator_input == "state_action" else "not_applicable",
            "discriminator_input": self.config.discriminator_input,
            "discriminator_initialization": "public_pacer_state_only" if transfer_disc else "independent_random",
            "amp_normalization": self.config.amp_normalization,
            "published_state_only_discriminator_transferred": transfer_disc,
            "published_amp_history_moments_transferred": transfer_disc,
            "author_navisaaclab_weights": False,
            "normalizer_count_cap": cap,
            "normalizer_counts_before_cap": original_counts,
        }
        if state_prior:
            self.provenance["initialization"].update({
                "action_conditioned_discriminator_initialization": STRATEGY,
                "discriminator_initialization": STRATEGY,
                "published_state_only_discriminator_transferred": True,
                "published_amp_history_moments_transferred": True,
                "state_prior_action_extension": dict(EXTENSION)})
            validate_action_state_prior_initialization(self.config, self.provenance["initialization"])
        if self.config.task_terrain_statistics_mode == "preserve_declared_suffix":
            self.terrain_statistics_policy.bind_public_warm_start(
                self.task_normalizer, initialization=self.provenance["initialization"],
                reference_count=terrain_reference_count, warm_start_task_count=float(self.task_normalizer.count))
        return self.provenance["initialization"]

    @torch.no_grad()
    def act(self, self_obs, task_obs, *, deterministic=False):
        self_obs, task_obs = self._observations(self_obs.to(self.device), task_obs.to(self.device))
        distribution, value = self.model.policy(self_obs, task_obs)
        actions = distribution.mean if deterministic else distribution.sample()
        if self.config.normalize_values:
            value = self.value_normalizer.denormalize(value[:, None]).squeeze(-1)
        return {"actions": actions, "applied_actions": actions.clamp(-1, 1) * self.action_mask,
                "log_prob": self._policy_log_prob(distribution, actions), "values": value,
                "mean_actions": distribution.mean}

    def _policy_log_prob(self, distribution, actions):
        values = distribution.log_prob(actions)
        if self.config.effective_action_likelihood:
            values = values * self.action_mask
        return values.sum(-1)

    @torch.no_grad()
    def value(self, self_obs, task_obs):
        self_obs, task_obs = self._observations(self_obs.to(self.device), task_obs.to(self.device))
        _, value = self.model.policy(self_obs, task_obs)
        return self.value_normalizer.denormalize(value[:, None]).squeeze(-1) if self.config.normalize_values else value

    def _amp_inputs(self, history, applied_actions):
        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))
        if tuple(history.shape[-2:]) != (self.config.history_frames, self.config.amp_frame_dim) or applied_actions.shape != history.shape[:-2] + (69,):
            raise ValueError("AMP history/actions do not match the declared rollout input contract")
        if not torch.isfinite(history).all() or not torch.isfinite(applied_actions).all() or torch.any(applied_actions.abs() > 1 + 1e-6):
            raise ValueError("AMP inputs must be finite; applied actions must be normalized to [-1,1]")
        if torch.any(applied_actions[..., self.action_mask == 0].abs() > 1e-6):
            raise ValueError("AMP must observe effective physical actions with frozen coordinates zero")
        if self.config.normalize_observations:
            if self.config.amp_normalization == "per_history_coordinate":
                history = self.amp_normalizer.normalize(history.flatten(-2)).reshape(history.shape)
            else:
                history = self.amp_normalizer.normalize(history)
        return history, applied_actions

    @torch.no_grad()
    def amp_reward(self, history, applied_actions):
        history, applied_actions = self._amp_inputs(history.to(self.device), applied_actions.to(self.device))
        logits = self.model.disc_logits(history, applied_actions)
        # PACER negative log(1-sigmoid(D)) transform is an explicit assumption;
        # Eq2 does not specify how the discriminator output becomes a reward.
        reward = F.softplus(logits).clamp_max(-math.log(self.config.amp_probability_epsilon)) * self.config.amp_reward_scale
        return reward

    def _discriminator_update(self, agent_history, agent_actions, expert_history, expert_actions):
        count = min(self.config.discriminator_batch_size, len(agent_history), len(expert_history))
        if count < 1:
            raise ValueError("AMP update requires real policy and expert examples")
        replay_count = (int(count * self.config.discriminator_replay_fraction)
                        if self.policy_replay is not None and self.policy_replay.size else 0)
        ai = torch.randperm(len(agent_history), device=self.device)[:count - replay_count]
        ei = torch.randperm(len(expert_history), device=self.device)[:count]
        ah, aa = self._amp_inputs(agent_history[ai], agent_actions[ai])
        if replay_count:
            rh, ra = self.policy_replay.sample(replay_count)
            rh, ra = self._amp_inputs(rh, ra)
            ah, aa = torch.cat((ah, rh)), torch.cat((aa, ra))
        eh, ea = self._amp_inputs(expert_history[ei], expert_actions[ei])
        negative = self.model.disc_logits(ah.detach(), aa.detach())
        expert_features = eh.detach().flatten(-2)
        if self.config.discriminator_input == "state_action":
            expert_features = torch.cat((expert_features, ea.detach()), -1)
        expert_features = expert_features.requires_grad_(True)
        positive = self.model.discriminator(expert_features).squeeze(-1)
        classification = .5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
        gradient = torch.autograd.grad(positive.sum(), expert_features, create_graph=True)[0]
        penalty = gradient.square().sum(-1).mean()
        final_layer = self.model.discriminator[-1]
        logit_regularization = final_layer.weight.square().sum()
        l2 = sum(module.weight.square().sum() for module in self.model.discriminator if isinstance(module, nn.Linear))
        loss = self.config.discriminator_coefficient * (
            classification + self.config.discriminator_gradient_penalty * penalty
            + self.config.discriminator_logit_regularization * logit_regularization
            + self.config.discriminator_weight_decay * l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("AMP discriminator loss is not finite")
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        self.discriminator_optimizer.step()
        return {"disc_loss": float(loss.detach()), "disc_classification": float(classification.detach()),
                "disc_gradient_penalty": float(penalty.detach()), "disc_grad_norm": float(grad),
                "disc_expert_probability": float(positive.sigmoid().mean().detach()),
                "disc_policy_probability": float(negative.sigmoid().mean().detach()),
                "disc_live_policy_probability": float(negative[:count - replay_count].sigmoid().mean().detach()) if count > replay_count else 0.,
                "disc_replay_policy_probability": float(negative[count - replay_count:].sigmoid().mean().detach()) if replay_count else 0.,
                "disc_replay_samples": replay_count, "disc_live_samples": count - replay_count,
                "disc_expert_samples": count}

    def policy_regularization(self, distribution, self_obs, task_obs):
        """Optional deployment-preserving objective; default training is unchanged."""
        return distribution.mean.new_zeros(())

    def update(self, rollout, expert_history, expert_actions):
        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))
        self.terrain_statistics_policy.require_ready()
        self.terrain_statistics_policy.validate_checkpoint(self.task_normalizer,
            update_count=self.update_count, physical_transition_count=self.physical_transition_count)
        config = self.config
        steps, envs = validate_rollout(rollout, self_dim=config.self_dim, task_dim=config.task_dim,
            history_frames=config.history_frames, amp_frame_dim=config.amp_frame_dim,
            action_mask=self.action_mask)
        data = {key: value.detach().to(self.device) for key, value in rollout.items() if isinstance(value, torch.Tensor)}
        expert_history = expert_history.detach().to(self.device)
        expert_actions = expert_actions.detach().to(self.device)
        self._amp_inputs(expert_history, expert_actions)
        if not len(expert_history):
            raise ValueError("An empty expert reference batch cannot train AMP")
        advantages, returns = generalized_advantages(data["rewards"], data["values"], data["next_values"],
            data["terminated"], data["truncated"], gamma=config.gamma, gae_lambda=config.gae_lambda)
        flat = {key: value.flatten(0, 1) for key, value in data.items() if value.shape[:2] == (steps, envs)}
        advantage = advantages.flatten()
        # Population variance remains defined even for a singleton rollout.
        advantage = (advantage - advantage.mean()) / (advantage.var(unbiased=False) + 1e-8).sqrt()
        target = returns.flatten()
        if config.normalize_values:
            if not self.freeze_normalizers:
                self.value_normalizer.update(target[:, None])
            target = self.value_normalizer.normalize(target[:, None], clamp=False).squeeze(-1)
        batches = []
        discriminator_batches = []
        kl_stopped = False
        epochs_started = 0
        self.model.train()
        total = steps * envs
        for _ in range(config.update_epochs):
            epochs_started += 1
            indices = torch.randperm(total, device=self.device)
            for start in range(0, total, config.minibatch_size):
                selection = indices[start:start + config.minibatch_size]
                self_obs, task_obs = self._observations(flat["self_obs"][selection], flat["task_obs"][selection])
                distribution, value = self.model.policy(self_obs, task_obs)
                new_log_prob = self._policy_log_prob(distribution, flat["actions"][selection])
                log_ratio = new_log_prob - flat["log_prob"][selection]
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantage[selection], clipped * advantage[selection]).mean()
                value_loss = (value - target[selection]).square().mean()
                entropy_values = distribution.entropy()
                bounds_values = (distribution.mean.abs() - 1).clamp_min(0).square()
                if config.effective_action_likelihood:
                    entropy_values = entropy_values * self.action_mask
                    bounds_values = bounds_values * self.action_mask
                entropy = entropy_values.sum(-1).mean()
                bounds = bounds_values.sum(-1).mean()
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                # Test before the next step; earlier valid policy steps remain.
                if config.target_kl is not None and float(approximate_kl.detach()) > config.target_kl:
                    kl_stopped = True
                    break
                symmetry_loss = torch.zeros((), device=self.device)
                if config.motion_symmetry_coefficient:
                    from .symmetry import mirror_policy_observations, mirror_actions
                    mirrored_self, mirrored_task = mirror_policy_observations(
                        flat["self_obs"][selection], flat["task_obs"][selection])
                    mirrored_self, mirrored_task = self._observations(mirrored_self, mirrored_task)
                    mirrored_distribution, _ = self.model.policy(mirrored_self, mirrored_task)
                    # Official PACER optimizes both branches (not a detached target).
                    residual = mirrored_distribution.mean - mirror_actions(distribution.mean)
                    if config.effective_action_likelihood:
                        residual = residual * self.action_mask
                        symmetry_loss = residual.square().sum(-1).mean() / self.action_mask.sum()
                    else:
                        symmetry_loss = residual.square().mean()
                regularization = self.policy_regularization(distribution, self_obs, task_obs)
                loss = (policy_loss + config.critic_coefficient * value_loss
                        - config.entropy_coefficient * entropy + config.bounds_coefficient * bounds
                        + config.motion_symmetry_coefficient * symmetry_loss + regularization)
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is not finite: preserve the failed rollout")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                policy_parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
                grad = nn.utils.clip_grad_norm_(policy_parameters, config.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                disc = {}
                if config.discriminator_updates_per_update is None:
                    disc = self._discriminator_update(flat["amp_history"], flat["applied_actions"], expert_history, expert_actions)
                    discriminator_batches.append(disc)
                batches.append({"ppo_loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                    "bounds_loss": float(bounds.detach()), "policy_grad_norm": float(grad),
                    "motion_symmetry_loss": float(symmetry_loss.detach()),
                    "policy_regularization_loss": float(regularization.detach()),
                    "approximate_kl": float(approximate_kl.detach()),
                    "clip_fraction": float(((ratio - 1).abs() > config.clip_ratio).float().mean()), **disc})
            if kl_stopped:
                break
        if config.discriminator_updates_per_update is not None:
            for _ in range(config.discriminator_updates_per_update):
                discriminator_batches.append(self._discriminator_update(
                    flat["amp_history"], flat["applied_actions"], expert_history, expert_actions))
        if not batches:
            raise FloatingPointError("No policy step was accepted; behavior likelihood contract changed")
        if self.policy_replay is not None:
            self.policy_replay.add(flat["amp_history"], flat["applied_actions"],
                                   keep_probability=config.discriminator_replay_keep_probability)
        # Never update observation moments before the PPO log-probability
        # recomputation: behavior policy and update must share frozen moments.
        if config.normalize_observations and not self.freeze_normalizers:
            self.self_normalizer.update(flat["self_obs"])
            self.task_normalizer.update(flat["task_obs"])
            self.amp_normalizer.update(torch.cat((flat["amp_history"], expert_history), 0))
            self.terrain_statistics_policy.after_task_update(self.task_normalizer)
        self.update_count += 1
        self.physical_transition_count += total
        metrics = {key: sum(batch[key] for batch in batches) / len(batches) for key in batches[0]}
        metrics.update({key: sum(batch[key] for batch in discriminator_batches) / len(discriminator_batches)
                        for key in discriminator_batches[0]})
        metrics.update({"update_count": self.update_count, "physical_transitions": self.physical_transition_count,
                        "reward_mean": float(data["rewards"].mean()), "return_mean": float(returns.mean()),
                        "advantage_std": float(advantages.std(unbiased=False)), "optimizer_minibatches": len(batches),
                        "effective_action_coordinates": int(self.action_mask.sum()),
                        "discriminator_minibatches": len(discriminator_batches),
                        "policy_epochs_started": epochs_started,
                        "policy_target_kl_stopped": kl_stopped,
                        "replay_size": self.policy_replay.size if self.policy_replay else 0})
        for key in ("amp_rewards", "trajectory_rewards", "energy_rewards", "direction_rewards", "signed_power_w", "forward_velocity"):
            if key in data:
                metrics[key + "_mean"] = float(data[key].mean())
                metrics[key + "_min"] = float(data[key].min())
                metrics[key + "_max"] = float(data[key].max())
        if "forward_velocity" in data:
            metrics["backwards_fraction"] = float((data["forward_velocity"] < 0).float().mean())
        return metrics

    def save(self, path, *, runner_state=None):
        """Save full resume state; runner should add physical/history/plan state.

        Model weights alone cannot reproduce a mid-episode simulator. The
        checkpoint records whether the physical runner provided its state.
        """
        self.terrain_statistics_policy.require_ready()
        self.terrain_statistics_policy.validate_checkpoint(self.task_normalizer,
            update_count=self.update_count, physical_transition_count=self.physical_transition_count)
        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np_state = np.random.get_state()
        state = {"format": "navisaaclab_method_reconstruction_training", "version": self.checkpoint_version,
            "config": asdict(self.config), "provenance": self.provenance,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "normalizers": {key: getattr(self, key).state_dict() for key in
                ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")},
            "update_count": self.update_count, "physical_transition_count": self.physical_transition_count,
            "rng": {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []},
            "runner_state": runner_state, "physical_runner_state_included": runner_state is not None,
            "policy_replay": self.policy_replay.state_dict() if self.policy_replay is not None else None}
        if self.config.task_terrain_statistics_mode != "legacy":
            state["task_terrain_statistics_policy"] = self.terrain_statistics_policy.state_dict()
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="cpu", restore_rng=True):
        state = torch.load(path, map_location=device, weights_only=True)
        if state.get("format") != "navisaaclab_method_reconstruction_training" or state.get("version") != cls.checkpoint_version:
            raise ValueError("Checkpoint is not a complete paper reconstruction training bundle")
        trainer = cls(TrainingConfig(**state["config"]), device=device, provenance=state["provenance"])
        validate_action_state_prior_initialization(trainer.config, trainer.provenance.get("initialization"))
        trainer.model.load_state_dict(state["model"], strict=True)
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        normalizers = state.get("normalizers")
        expected_normalizers = {"self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"}
        if not isinstance(normalizers, dict) or set(normalizers) != expected_normalizers:
            raise ValueError("Complete resume requires exactly all four declared normalizers")
        for key, value in normalizers.items():
            expected = getattr(trainer, key).state_dict()
            if not isinstance(value, dict) or set(value) != set(expected):
                raise ValueError(f"Invalid resume normalizer schema: {key}")
            for name, tensor in value.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                        or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()):
                    raise ValueError(f"Invalid resume normalizer tensor: {key}.{name}")
            if bool((value["variance"] < 0).any()) or value["count"] < 0:
                raise ValueError(f"Invalid resume normalizer variance/count: {key}")
            getattr(trainer, key).load_state_dict(value, strict=True)
        trainer.update_count = state["update_count"]
        trainer.physical_transition_count = state["physical_transition_count"]
        trainer.terrain_statistics_policy = TerrainStatisticsPolicy.from_checkpoint(
            state.get("task_terrain_statistics_policy"), config=trainer.config,
            initialization=trainer.provenance.get("initialization"),
            saved_task_normalizer=normalizers["task_normalizer"],
            update_count=trainer.update_count, physical_transition_count=trainer.physical_transition_count)
        replay = state.get("policy_replay")
        if trainer.policy_replay is not None:
            if replay is None:
                raise ValueError("Replay-enabled resume requires the complete saved replay buffer")
            trainer.policy_replay.load_state_dict(replay)
        elif replay is not None:
            raise ValueError("Replay state supplied for a replay-disabled configuration")
        trainer._pending_rng_state = state["rng"]
        if restore_rng:
            trainer.restore_rng()
        return trainer, state.get("runner_state")

    def restore_rng(self):
        """Restore a pending saved RNG after all runner/model setup completes.

        Constructing a diffusion planner or restoring physical helpers can
        consume randomness. Runtime resumes therefore load with restore_rng
        False, complete those operations, then call this once before rollouts.
        """
        rng = self._pending_rng_state
        if rng is None:
            raise ValueError("No pending checkpoint RNG state remains to restore")
        if rng["torch_cuda"] and not torch.cuda.is_available():
            raise ValueError("CUDA RNG resume requested on a CPU-only host")
        random.setstate(rng["python"])
        ns = rng["numpy"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if rng["torch_cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
        self._pending_rng_state = None
