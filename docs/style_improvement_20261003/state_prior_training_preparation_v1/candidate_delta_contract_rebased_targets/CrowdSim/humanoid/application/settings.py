"""Pure Python training settings shared by pre-Kit gates and Torch trainers."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .state_prior_initialization import validate_action_state_prior_config


def normalize_expert_sampling_config(value):
    """Validate an explicit expert-only prior without importing Torch or Kit."""
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("expert_sampling must be None or an explicit configuration mapping")
    defaults = {"dataset_weighting": "sqrt_parent_support", "dataset_cap": None,
                "minimum_parents_per_bucket": 5, "minimum_windows_per_bucket": 100,
                "thresholds": {"low_speed_m_s": .12, "fast_speed_m_s": 1.8,
                    "lateral_fraction": .55, "net_turn_angle_rad": .2,
                    "quaternion_norm_tolerance": 1e-3}}
    required = {"mode", "seed", "target_weights"}
    if set(value)-set(defaults)-required or not required <= set(value):
        raise ValueError("expert_sampling has unknown or missing configuration fields")
    result = {**defaults, **value}
    if result["mode"] != "stratified_windows":
        raise ValueError("Unknown expert_sampling mode")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("Expert sampling seed must be an integer in [0,2**32)")
    names = ("low_speed", "fast", "sideways", "turning", "moving")
    weights = result["target_weights"]
    if type(weights) is not dict or set(weights) != set(names):
        raise ValueError("Expert target_weights must explicitly declare all five known window strata")

    def number(number_value, name, *, positive=False):
        if (type(number_value) not in (int, float) or not math.isfinite(number_value)
                or number_value < 0 or positive and number_value == 0):
            raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
        return float(number_value)

    result["target_weights"] = {name: number(weights[name], f"expert target_weights.{name}") for name in names}
    if not math.isfinite(sum(result["target_weights"].values())) or sum(result["target_weights"].values()) <= 0:
        raise ValueError("Expert target weights need a finite positive total")
    if not isinstance(result["dataset_weighting"], str) or result["dataset_weighting"] not in {"sqrt_parent_support", "uniform"}:
        raise ValueError("Unknown expert dataset weighting")
    if result["dataset_cap"] is not None:
        result["dataset_cap"] = number(result["dataset_cap"], "Expert dataset cap", positive=True)
        if result["dataset_cap"] > 1:
            raise ValueError("Expert dataset cap must not exceed1")
    for name in ("minimum_parents_per_bucket", "minimum_windows_per_bucket"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"Expert {name} must be a positive integer")
    thresholds = result["thresholds"]
    if type(thresholds) is not dict or set(thresholds)-set(defaults["thresholds"]):
        raise ValueError("Unknown expert short-window threshold")
    result["thresholds"] = {key: number(thresholds.get(key, default), f"Expert threshold.{key}", positive=True)
                            for key, default in defaults["thresholds"].items()}
    t = result["thresholds"]
    if (t["low_speed_m_s"] >= t["fast_speed_m_s"] or t["lateral_fraction"] > 1
            or t["net_turn_angle_rad"] > math.pi or t["quaternion_norm_tolerance"] > .01):
        raise ValueError("Expert short-window thresholds violate their declared ranges")
    return result


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
    expert_dof_velocity_frame: str = "legacy_stored"
    freeze_toes: bool = True
    action_mask: tuple[int, ...] | None = None
    discriminator_input: str = "state_action"
    amp_normalization: str = "per_frame"
    warm_start_discriminator: bool = False
    warm_start_action_discriminator_from_state_prior: bool = False
    discriminator_replay_capacity: int = 0
    discriminator_replay_keep_probability: float = .01
    discriminator_replay_fraction: float = .5
    discriminator_updates_per_update: int | None = None
    warm_start_normalizer_count_cap: float | None = None
    amp_reward_weight: float = 1.
    task_reward_weight: float = .5
    effective_action_likelihood: bool = False
    target_kl: float | None = None
    motion_symmetry_coefficient: float = 0.
    expert_sampling: dict | None = None
    task_terrain_statistics_mode: str = "legacy"
    seed: int = 7

    def validate(self):
        if (type(self.task_terrain_statistics_mode) is not str
                or self.task_terrain_statistics_mode not in {"legacy", "preserve_declared_suffix"}):
            raise ValueError("Unknown task_terrain_statistics_mode")
        if self.task_terrain_statistics_mode == "preserve_declared_suffix" and (
                (self.self_dim, self.task_dim) != (357, 1044)
                or self.normalize_observations is not True or self.normalization_clip != 5.):
            raise ValueError("Protected terrain statistics require canonical357/1044 normalized observations with clip5")
        sampling = normalize_expert_sampling_config(self.expert_sampling)
        if sampling is not None and (self.history_frames, self.amp_frame_dim) != (10, 195):
            raise ValueError("Stratified experts require the canonical10x195 AMP history contract")
        numeric_scalars = (
            "learning_rate", "discriminator_learning_rate", "gamma", "gae_lambda",
            "clip_ratio", "critic_coefficient", "entropy_coefficient", "bounds_coefficient",
            "max_grad_norm", "log_std", "discriminator_coefficient",
            "discriminator_gradient_penalty", "discriminator_logit_regularization",
            "discriminator_weight_decay", "amp_reward_scale", "amp_probability_epsilon",
            "normalization_clip", "control_dt", "target_horizon_s",
            "discriminator_replay_keep_probability", "discriminator_replay_fraction",
            "amp_reward_weight", "task_reward_weight",
            "motion_symmetry_coefficient",
        )
        for key in numeric_scalars:
            if isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be numeric, not a boolean")
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
        for key in ("amp_reward_weight", "task_reward_weight", "motion_symmetry_coefficient"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.amp_reward_weight + self.task_reward_weight <= 0:
            raise ValueError("At least one reward component must have positive weight")
        if type(self.discriminator_replay_capacity) is not int or self.discriminator_replay_capacity < 0:
            raise ValueError("Discriminator replay capacity must be a nonnegative integer")
        for key in ("discriminator_replay_keep_probability", "discriminator_replay_fraction"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be a probability in [0,1]")
        if self.discriminator_updates_per_update is not None and (type(self.discriminator_updates_per_update) is not int or self.discriminator_updates_per_update < 1):
            raise ValueError("Independent discriminator update count must be a positive integer")
        for key in ("warm_start_normalizer_count_cap", "target_kl"):
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be None or a positive finite number")
        if type(self.effective_action_likelihood) is not bool:
            raise ValueError("Effective action likelihood must be an explicit boolean")
        if self.motion_symmetry_coefficient and (self.self_dim, self.task_dim) != (357, 1044):
            raise ValueError("PACER motion symmetry requires the canonical357/1044 observation contract")
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
        if self.expert_dof_velocity_frame not in {"legacy_stored", "child_joint"}:
            raise ValueError("Expert DOF velocity frame must explicitly select legacy_stored or child_joint")
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
        validate_action_state_prior_config(self)
