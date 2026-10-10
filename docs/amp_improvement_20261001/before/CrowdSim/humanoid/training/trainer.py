"""Actual Torch PPO and action-conditioned AMP updates on physical rollouts.

The architecture follows public PACER's separate actor/critic+shared task MLP.
The discriminator additionally consumes applied actions as printed in Eq2 of
NavIsaacLab. It must be trained anew; PACER's state-only D cannot initialize it.
Unspecified optimizer/feature choices are exposed in config and provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import generalized_advantages, validate_rollout


@dataclass
class TrainingConfig:
    self_dim: int = 357
    task_dim: int = 1044
    amp_frame_dim: int = 195
    history_frames: int = 10
    task_layers: tuple[int, ...] = (512, 256)
    policy_layers: tuple[int, ...] = (2048, 1024)
    discriminator_layers: tuple[int, ...] = (1024, 512)
    learning_rate: float = 2e-5
    discriminator_learning_rate: float = 2e-5
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    update_epochs: int = 6
    minibatch_size: int = 16384
    discriminator_batch_size: int = 4096
    critic_coefficient: float = 5.
    entropy_coefficient: float = 0.
    bounds_coefficient: float = 10.
    max_grad_norm: float = 50.
    log_std: float = -2.9
    discriminator_coefficient: float = 5.
    discriminator_gradient_penalty: float = 5.
    discriminator_logit_regularization: float = .01
    discriminator_weight_decay: float = .0001
    amp_reward_scale: float = 2.
    amp_probability_epsilon: float = 1e-4
    normalize_observations: bool = True
    normalize_values: bool = True
    normalization_clip: float = 5.
    direction_mode: str = "paper_equation"
    control_dt: float = 1 / 30
    target_horizon_s: float = 5.
    mode: str = "paper_method_reconstruction"
    expert_action_semantics: str = "inverse_pd_target_from_mocap_pose"
    freeze_toes: bool = True
    action_mask: tuple[int, ...] | None = None
    seed: int = 7

    def validate(self):
        for key in ("self_dim", "task_dim", "amp_frame_dim", "history_frames", "update_epochs", "minibatch_size", "discriminator_batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for layers in (self.task_layers, self.policy_layers, self.discriminator_layers):
            if not layers or any(type(width) is not int or width < 1 for width in layers):
                raise ValueError("Network layers must contain positive integer widths")
        for key in ("learning_rate", "discriminator_learning_rate", "max_grad_norm", "normalization_clip", "control_dt", "target_horizon_s"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("critic_coefficient", "entropy_coefficient", "bounds_coefficient", "discriminator_coefficient", "discriminator_gradient_penalty", "discriminator_logit_regularization", "discriminator_weight_decay", "amp_reward_scale"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_ratio < 1:
            raise ValueError("Discount/GAE/clip ratios are outside their valid ranges")
        if not math.isfinite(self.log_std) or not 0 < self.amp_probability_epsilon < 1:
            raise ValueError("Gaussian log_std/AMP epsilon are invalid")
        if self.history_frames not in (10, 11):
            raise ValueError("Declare PACER10 or inclusive paper11 history frames explicitly")
        if self.mode not in {"paper_method_reconstruction", "strict_author_experiment"}:
            raise ValueError("Unknown paper training provenance mode")
        if self.direction_mode not in {"paper_equation", "backwards_penalty"}:
            raise ValueError("Direction reward mode must be explicit")
        if self.expert_action_semantics not in {"inverse_pd_target_from_mocap_pose", "user_supplied_normalized_actions"}:
            raise ValueError("Expert actions require an explicit source; AMASS has no policy actions")
        if type(self.freeze_toes) is not bool:
            raise ValueError("freeze_toes must explicitly match the physical PD decoder")
        if self.action_mask is not None and (len(self.action_mask) != 69 or any(value not in (0, 1) for value in self.action_mask)):
            raise ValueError("action_mask must be a binary69 canonical vector")


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
        self.discriminator = _mlp(config.history_frames * config.amp_frame_dim + 69, config.discriminator_layers, 1)
        self.register_buffer("log_std", torch.full((69,), config.log_std))

    def policy(self, self_obs, task_obs):
        features = torch.cat((self_obs, self.task_encoder(task_obs)), dim=-1)
        mean = self.actor(features)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return distribution, self.critic(features).squeeze(-1)

    def disc_logits(self, history, applied_actions):
        return self.discriminator(torch.cat((history.flatten(-2), applied_actions), dim=-1)).squeeze(-1)


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
        self.amp_normalizer = RunningNormalizer(config.amp_frame_dim, config.normalization_clip).to(self.device)
        self.value_normalizer = RunningNormalizer(1, config.normalization_clip).to(self.device)
        policy_parameters = list(self.model.task_encoder.parameters()) + list(self.model.actor.parameters()) + list(self.model.critic.parameters())
        self.optimizer = torch.optim.Adam(policy_parameters, lr=config.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=config.discriminator_learning_rate)
        self.update_count = 0
        self.physical_transition_count = 0
        self._pending_rng_state = None

    @classmethod
    def prepare(cls, config=None, **kwargs):
        return cls(config or TrainingConfig(), **kwargs)

    def _observations(self, self_obs, task_obs):
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

        This is a declared initialization for new training. The state-only
        public discriminator and its history statistics are intentionally not
        transferred to the new action-conditioned discriminator.
        """
        from CrowdSim.humanoid.amp.controller import PacerAMPController

        if self.update_count or self.optimizer.state or self.discriminator_optimizer.state:
            raise ValueError("PACER warm start must happen before any training update")
        if (self.config.self_dim, self.config.task_dim, tuple(self.config.task_layers), tuple(self.config.policy_layers)) != (357, 1044, (512, 256), (2048, 1024)):
            raise ValueError("Warm start requires the complete public PACER actor/critic architecture")
        reference = PacerAMPController.from_checkpoint(checkpoint, dof_limits_lower,
            dof_limits_upper, device=self.device, **loader_kwargs)
        network = reference.network
        self.model.task_encoder.load_state_dict(network._task_mlp.state_dict(), strict=True)
        for destination, trunk, output in ((self.model.actor, network.actor_mlp, network.mu),
                                           (self.model.critic, network.critic_mlp, network.value)):
            for layer_index in (0, 2):
                destination[layer_index].load_state_dict(trunk[layer_index].state_dict(), strict=True)
            destination[4].load_state_dict(output.state_dict(), strict=True)
        self.model.log_std.copy_(network.sigma)
        obs = reference.observation_normalizer
        for destination, section in ((self.self_normalizer, slice(0, 357)),
                                      (self.task_normalizer, slice(357, 1401))):
            destination.mean.copy_(obs.running_mean[section])
            destination.variance.copy_(obs.running_var[section])
            destination.count.copy_(obs.count)
        value = reference.value_normalizer
        self.value_normalizer.mean.copy_(value.running_mean)
        self.value_normalizer.variance.copy_(value.running_var)
        self.value_normalizer.count.copy_(value.count)
        self.provenance["initialization"] = {
            "mode": "transfer_warm_start_public_pacer_actor_critic", "path": str(Path(checkpoint).resolve()),
            "sha256": reference.checkpoint_sha256, "actor_critic_task_transferred": True,
            "observation_value_moments_transferred": True,
            "action_conditioned_discriminator_initialization": "independent_random",
            "published_state_only_discriminator_transferred": False,
            "author_navisaaclab_weights": False,
        }
        return self.provenance["initialization"]

    @torch.no_grad()
    def act(self, self_obs, task_obs, *, deterministic=False):
        self_obs, task_obs = self._observations(self_obs.to(self.device), task_obs.to(self.device))
        distribution, value = self.model.policy(self_obs, task_obs)
        actions = distribution.mean if deterministic else distribution.sample()
        if self.config.normalize_values:
            value = self.value_normalizer.denormalize(value[:, None]).squeeze(-1)
        return {"actions": actions, "applied_actions": actions.clamp(-1, 1) * self.action_mask,
                "log_prob": distribution.log_prob(actions).sum(-1), "values": value,
                "mean_actions": distribution.mean}

    @torch.no_grad()
    def value(self, self_obs, task_obs):
        self_obs, task_obs = self._observations(self_obs.to(self.device), task_obs.to(self.device))
        _, value = self.model.policy(self_obs, task_obs)
        return self.value_normalizer.denormalize(value[:, None]).squeeze(-1) if self.config.normalize_values else value

    def _amp_inputs(self, history, applied_actions):
        if tuple(history.shape[-2:]) != (self.config.history_frames, self.config.amp_frame_dim) or applied_actions.shape != history.shape[:-2] + (69,):
            raise ValueError("AMP history/actions do not match the action-conditioned input contract")
        if not torch.isfinite(history).all() or not torch.isfinite(applied_actions).all() or torch.any(applied_actions.abs() > 1 + 1e-6):
            raise ValueError("AMP inputs must be finite; applied actions must be normalized to [-1,1]")
        if torch.any(applied_actions[..., self.action_mask == 0].abs() > 1e-6):
            raise ValueError("AMP must observe effective physical actions with frozen coordinates zero")
        history = self.amp_normalizer.normalize(history) if self.config.normalize_observations else history
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
        ai = torch.randperm(len(agent_history), device=self.device)[:count]
        ei = torch.randperm(len(expert_history), device=self.device)[:count]
        ah, aa = self._amp_inputs(agent_history[ai], agent_actions[ai])
        eh, ea = self._amp_inputs(expert_history[ei], expert_actions[ei])
        negative = self.model.disc_logits(ah.detach(), aa.detach())
        expert_features = torch.cat((eh.detach().flatten(-2), ea.detach()), -1).requires_grad_(True)
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
                "disc_policy_probability": float(negative.sigmoid().mean().detach())}

    def update(self, rollout, expert_history, expert_actions):
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
            self.value_normalizer.update(target[:, None])
            target = self.value_normalizer.normalize(target[:, None], clamp=False).squeeze(-1)
        batches = []
        self.model.train()
        total = steps * envs
        for _ in range(config.update_epochs):
            indices = torch.randperm(total, device=self.device)
            for start in range(0, total, config.minibatch_size):
                selection = indices[start:start + config.minibatch_size]
                self_obs, task_obs = self._observations(flat["self_obs"][selection], flat["task_obs"][selection])
                distribution, value = self.model.policy(self_obs, task_obs)
                new_log_prob = distribution.log_prob(flat["actions"][selection]).sum(-1)
                log_ratio = new_log_prob - flat["log_prob"][selection]
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantage[selection], clipped * advantage[selection]).mean()
                value_loss = (value - target[selection]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                bounds = (distribution.mean.abs() - 1).clamp_min(0).square().sum(-1).mean()
                loss = policy_loss + config.critic_coefficient * value_loss - config.entropy_coefficient * entropy + config.bounds_coefficient * bounds
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is not finite: preserve the failed rollout")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                policy_parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
                grad = nn.utils.clip_grad_norm_(policy_parameters, config.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                disc = self._discriminator_update(flat["amp_history"], flat["applied_actions"], expert_history, expert_actions)
                batches.append({"ppo_loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                    "bounds_loss": float(bounds.detach()), "policy_grad_norm": float(grad),
                    "approximate_kl": float(((ratio - 1) - log_ratio).mean().detach()),
                    "clip_fraction": float(((ratio - 1).abs() > config.clip_ratio).float().mean()), **disc})
        # Never update observation moments before the PPO log-probability
        # recomputation: behavior policy and update must share frozen moments.
        if config.normalize_observations:
            self.self_normalizer.update(flat["self_obs"])
            self.task_normalizer.update(flat["task_obs"])
            self.amp_normalizer.update(torch.cat((flat["amp_history"], expert_history), 0))
        self.update_count += 1
        self.physical_transition_count += total
        metrics = {key: sum(batch[key] for batch in batches) / len(batches) for key in batches[0]}
        metrics.update({"update_count": self.update_count, "physical_transitions": self.physical_transition_count,
                        "reward_mean": float(data["rewards"].mean()), "return_mean": float(returns.mean()),
                        "advantage_std": float(advantages.std(unbiased=False)), "optimizer_minibatches": len(batches),
                        "effective_action_coordinates": int(self.action_mask.sum())})
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
            "runner_state": runner_state, "physical_runner_state_included": runner_state is not None}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="cpu", restore_rng=True):
        state = torch.load(path, map_location=device, weights_only=True)
        if state.get("format") != "navisaaclab_method_reconstruction_training" or state.get("version") != cls.checkpoint_version:
            raise ValueError("Checkpoint is not a complete paper reconstruction training bundle")
        trainer = cls(TrainingConfig(**state["config"]), device=device, provenance=state["provenance"])
        trainer.model.load_state_dict(state["model"], strict=True)
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        for key, value in state["normalizers"].items():
            getattr(trainer, key).load_state_dict(value, strict=True)
        trainer.update_count = state["update_count"]
        trainer.physical_transition_count = state["physical_transition_count"]
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
