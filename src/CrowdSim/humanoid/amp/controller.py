"""Strict public PACER reference checkpoint loading and physical PD decoding."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .network import PacerAMPNetwork, RunningNormalizer
from .observations import (
    PACER_DOF_NAMES,
    PacerSMPLMapping,
    build_amp_observation,
    build_self_observation,
    build_task_observation,
)

NO_SHAPE_SHA256 = "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"


def _pd_parameters(lower: torch.Tensor, upper: torch.Tensor,
                   dof_names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if lower.shape != (69,) or upper.shape != (69,):
        raise ValueError("PACER PD decoding requires 69 actual joint limits")
    if lower.device != upper.device or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower > upper).any():
        raise ValueError("PD limits must be finite, ordered and on the same device")
    mapping = PacerSMPLMapping(dof_names=dof_names)
    canonical_lower = lower[list(mapping.dof_indices)].reshape(23, 3)
    canonical_upper = upper[list(mapping.dof_indices)].reshape(23, 3)
    scale = 1.2 * torch.maximum(canonical_lower.abs().amax(-1), canonical_upper.abs().amax(-1))
    scale = scale.clamp_max(math.pi).repeat_interleave(3)
    scale[PACER_DOF_NAMES.index("L_Knee_y")] = 5.0
    scale[PACER_DOF_NAMES.index("R_Knee_y")] = 5.0
    # bias_offset=False, has_smpl_pd_offset=False in the published no_shape cfg.
    return torch.zeros_like(scale), scale


def action_to_pd(raw_action: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, *,
                 dof_names: tuple[str, ...] = PACER_DOF_NAMES,
                 freeze_toes: bool = True, clip: bool = True) -> torch.Tensor:
    """Canonical policy action -> physical joint targets in dof_names order.

    The caller supplies real limits from the selected articulation; no old
    generated XML limits are guessed. All public no_shape joints are 3 DOF.
    Returns targets, not torques or another policy's scaled action vector.
    """
    if raw_action.ndim != 2 or raw_action.shape[1] != 69 or not torch.isfinite(raw_action).all():
        raise ValueError("PACER actions must be finite [B,69] canonical vectors")
    lower = lower.to(device=raw_action.device, dtype=raw_action.dtype)
    upper = upper.to(device=raw_action.device, dtype=raw_action.dtype)
    offset, scale = _pd_parameters(lower, upper, tuple(dof_names))
    target = offset + scale * (raw_action.clamp(-1.0, 1.0) if clip else raw_action)
    frozen = {"L_Hand", "R_Hand"}
    if freeze_toes:
        frozen.update(("L_Toe", "R_Toe"))
    indices = [index for index, name in enumerate(PACER_DOF_NAMES) if name.rsplit("_", 1)[0] in frozen]
    target[:, indices] = 0.0
    physical_order = [PACER_DOF_NAMES.index(name) for name in dof_names]
    return target[:, physical_order]


def _load_normalizer(normalizer: RunningNormalizer, state, label: str) -> None:
    if not isinstance(state, dict) or set(state) != {"running_mean", "running_var", "count"}:
        raise ValueError(f"{label} normalizer requires exactly mean/variance/count")
    expected = normalizer.state_dict()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {label}.{name} statistics")
    if (state["running_var"] < 0).any() or state["count"] <= 0:
        raise ValueError(f"{label} variance/count are invalid")
    normalizer.load_state_dict(state, strict=True)


class PacerAMPController(nn.Module):
    """Public no_shape AMP inference, with no MaskedMimic fallback.

    The trainable network is exposed for deliberate new training. This loader
    recognizes the public reference architecture and verifies its SHA-256
    before restricted torch.load; it does not claim a NavIsaacLab paper weight.
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, *,
                 mapping: PacerSMPLMapping | None = None):
        super().__init__()
        self.mapping = mapping or PacerSMPLMapping()
        lower, upper = torch.as_tensor(lower).float(), torch.as_tensor(upper).float()
        _pd_parameters(lower, upper, self.mapping.dof_names)
        self.register_buffer("dof_limits_lower", lower.clone())
        self.register_buffer("dof_limits_upper", upper.clone())
        self.network = PacerAMPNetwork()
        self.observation_normalizer = RunningNormalizer(1401)
        self.value_normalizer = RunningNormalizer(1)
        self.discriminator_normalizer = RunningNormalizer(1950)
        self.checkpoint_sha256 = None
        self.reference_only = True
        self.local_finetune = None

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, dof_limits_lower: torch.Tensor,
                        dof_limits_upper: torch.Tensor, *, body_names=None, dof_names=None,
                        device: str | torch.device = "cpu",
                        expected_sha256: str = NO_SHAPE_SHA256) -> "PacerAMPController":
        path = Path(checkpoint)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("An explicit SHA-256 is required before checkpoint loading")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint SHA-256 mismatch; refusing any fallback policy")
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_data, dict) or not {"model", "running_mean_std", "reward_mean_std", "amp_input_mean_std"}.issubset(checkpoint_data):
            raise ValueError("PACER checkpoint lacks actor/critic/discriminator/statistics")
        mapping_args = {}
        if body_names is not None:
            mapping_args["body_names"] = tuple(body_names)
        if dof_names is not None:
            mapping_args["dof_names"] = tuple(dof_names)
        result = cls(dof_limits_lower, dof_limits_upper, mapping=PacerSMPLMapping(**mapping_args))
        model = checkpoint_data["model"]
        expected = result.network.state_dict()
        if not isinstance(model, dict) or set(model) != {f"a2c_network.{key}" for key in expected}:
            raise ValueError("PACER checkpoint model keys violate the complete reference schema")
        weights = {}
        for key, example in expected.items():
            value = model[f"a2c_network.{key}"]
            if not isinstance(value, torch.Tensor) or value.shape != example.shape or value.dtype != example.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid PACER model tensor {key}")
            weights[key] = value
        result.network.load_state_dict(weights, strict=True)
        _load_normalizer(result.observation_normalizer, checkpoint_data["running_mean_std"], "observation")
        _load_normalizer(result.value_normalizer, checkpoint_data["reward_mean_std"], "value")
        _load_normalizer(result.discriminator_normalizer, checkpoint_data["amp_input_mean_std"], "discriminator")
        result.checkpoint_sha256 = digest.hexdigest()
        metadata = checkpoint_data.get("local_finetune")
        if metadata is not None:
            if (not isinstance(metadata, dict) or metadata.get("schema") != "crowdsim.pacer_amass_finetune.v1"
                    or metadata.get("parent_sha256") != NO_SHAPE_SHA256
                    or metadata.get("train_clip_count") != 2848
                    or metadata.get("normalizers_frozen") is not True
                    or metadata.get("all_train_clips_seen") is not True
                    or metadata.get("amp_sample_offsets_s") != [i*.5 for i in range(10)]
                    or type(metadata.get("updates")) is not int or metadata["updates"] < 1
                    or type(metadata.get("physical_transitions")) is not int or metadata["physical_transitions"] < 1):
                raise ValueError("Invalid local AMASS fine-tune provenance")
            ids = metadata.get("expert_unique_motion_ids")
            if not isinstance(ids, list) or len(ids) != 2848 or any(type(i) is not int for i in ids) or len(set(ids)) != 2848:
                raise ValueError("Fine-tune provenance requires2848 unique actual expert motion IDs")
            result.reference_only = False
            result.local_finetune = metadata
        result.to(device).eval()
        return result

    def raw_observation(self, state, trajectory, terrain, *, trajectory_frame="world"):
        own = build_self_observation(state, mapping=self.mapping)
        task = build_task_observation(state, trajectory, terrain, trajectory_frame=trajectory_frame, mapping=self.mapping)
        return torch.cat((own, task), dim=-1)

    def decode_action(self, action: torch.Tensor) -> torch.Tensor:
        return action_to_pd(action, self.dof_limits_lower, self.dof_limits_upper,
                            dof_names=self.mapping.dof_names)

    def _require_checkpoint(self) -> None:
        if self.checkpoint_sha256 is None:
            raise RuntimeError("Reference controller requires a strictly verified checkpoint; use PacerAMPNetwork for explicit new training")

    @torch.no_grad()
    def act(self, state, trajectory, terrain, *, trajectory_frame="world",
            return_raw: bool = False, deterministic: bool = True,
            generator: torch.Generator | None = None) -> torch.Tensor:
        self._require_checkpoint()
        observation = self.raw_observation(state, trajectory, terrain, trajectory_frame=trajectory_frame)
        normalized = self.observation_normalizer(observation.float())
        mean, log_std = self.network.eval_actor(normalized)
        if deterministic:
            action = mean
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            action = mean + log_std.exp() * noise
        action = action.clamp(-1.0, 1.0)
        return action if return_raw else self.decode_action(action)

    def evaluate_value(self, raw_observation: torch.Tensor, *, denormalize: bool = True) -> torch.Tensor:
        self._require_checkpoint()
        value = self.network.eval_critic(self.observation_normalizer(raw_observation))
        return self.value_normalizer(value, unnorm=True) if denormalize else value

    def eval_discriminator(self, history: torch.Tensor) -> torch.Tensor:
        self._require_checkpoint()
        if history.ndim != 3 or history.shape[1:] != (10, 195):
            raise ValueError("Published PACER discriminator expects [B,10,195], newest first")
        return self.network.eval_disc(self.discriminator_normalizer(history.flatten(1)))

    def discriminator_reward(self, history: torch.Tensor) -> torch.Tensor:
        probability = self.eval_discriminator(history).sigmoid()
        return -(1.0 - probability).clamp_min(1e-4).log() * 2.0

    def amp_observation(self, state) -> torch.Tensor:
        return build_amp_observation(state, mapping=self.mapping)
