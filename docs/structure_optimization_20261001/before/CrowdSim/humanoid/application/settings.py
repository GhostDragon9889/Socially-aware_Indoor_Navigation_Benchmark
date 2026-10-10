"""Pure Python training settings shared by pre-Kit gates and Torch trainers."""
from __future__ import annotations

from dataclasses import dataclass
import math


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
    discriminator_input: str = "state_action"
    amp_normalization: str = "per_frame"
    warm_start_discriminator: bool = False
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
        if type(self.normalize_observations) is not bool or type(self.normalize_values) is not bool:
            raise ValueError("Observation/value normalization options must be explicit booleans")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Training seed must be an integer in the NumPy-compatible range [0,2**32)")
        if type(self.freeze_toes) is not bool:
            raise ValueError("freeze_toes must explicitly match the physical PD decoder")
        if self.action_mask is not None and (len(self.action_mask) != 69 or any(value not in (0, 1) for value in self.action_mask)):
            raise ValueError("action_mask must be a binary69 canonical vector")
        if self.discriminator_input not in {"state_action", "state"}:
            raise ValueError("Discriminator input must explicitly select state_action or state")
        if self.amp_normalization not in {"per_frame", "per_history_coordinate"}:
            raise ValueError("AMP normalization must explicitly select frame or history coordinates")
        if type(self.warm_start_discriminator) is not bool:
            raise ValueError("warm_start_discriminator must be an explicit boolean")
        if self.warm_start_discriminator and (self.discriminator_input != "state"
                or self.amp_normalization != "per_history_coordinate"
                or (self.history_frames, self.amp_frame_dim, tuple(self.discriminator_layers)) != (10, 195, (1024, 512))):
            raise ValueError("Public PACER discriminator transfer requires state-only 10x195 history-coordinate normalization and exact layers")
