"""Standalone neutral-SMPL social actor/critic boundary; not a trainer/loader.

Architecture adapted from fixed PACER group-CNN code baeb6241. Inputs are
already normalized by an external self357/task12308 RMS. No world observation,
PD action mask, AMP discriminator, normalizer or checkpoint migration is owned
here. Import/construction/forward require Torch and have not yet been executed.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

SELF_DIM = 357
TRAJECTORY_DIM = 20
GRID_SIZE = 64
RASTER_CHANNELS = 3
TASK_DIM = TRAJECTORY_DIM + GRID_SIZE * GRID_SIZE * RASTER_CHANNELS
ENCODED_TASK_DIM = 256 + TRAJECTORY_DIM
POLICY_INPUT_DIM = SELF_DIM + ENCODED_TASK_DIM
ACTION_DIM = 69
ARCHITECTURE_ID = "local.neutral_smpl.shared_social_CNN_actor_critic.v1"


def architecture_contract():
    """Metadata only; future saved config must additionally identify raster frames."""
    return {"architecture_id": ARCHITECTURE_ID, "self_dim": SELF_DIM, "task_dim": TASK_DIM,
            "raw_task_order": "trajectory20, HWC64x64x3 C-order raster",
            "raster_channels": ["height", "root_heading_relative_vx", "root_heading_relative_vy"],
            "encoded_task_order": "map256, normalized_trajectory20",
            "encoded_task_dim": ENCODED_TASK_DIM, "policy_input_dim": POLICY_INPUT_DIM,
            "task_encoder_sharing": "shared", "actor_critic_MLPs": "separate",
            "CNN": {"filters": [16, 16, 16], "kernel": 4, "stride": 2, "padding": 1,
                    "activation": "ReLU", "spatial_sizes": [64, 32, 16, 8], "flatten_dim": 1024},
            "map_projection": [1024, 512, 256], "policy_layers": [633, 2048, 1024],
            "action_dim": ACTION_DIM, "fixed_log_std": -2.9, "value_dim": 1,
            "input_normalization": "external raw self357 and entire task12308 RMS before encoder",
            "value_output": "normalized; external value RMS denormalizes",
            "initialization": "fresh torch Linear/Conv2d defaults; fixed log_std; no weight transfer",
            "old_1044_checkpoint_compatible": False}


def _validate_input(values, dimension, label):
    if (not isinstance(values, Tensor) or values.ndim != 2 or values.shape[0] < 1
            or values.shape[1] != dimension or values.dtype != torch.float32
            or values.device.type == "meta" or not bool(torch.isfinite(values).all())):
        raise ValueError(f"{label} requires finite materialized float32[B,{dimension}]")


def _mlp(input_dim, widths, output_dim=None):
    modules = []
    for width in widths:
        modules.extend((nn.Linear(input_dim, width, dtype=torch.float32), nn.ReLU()))
        input_dim = width
    if output_dim is not None:
        modules.append(nn.Linear(input_dim, output_dim, dtype=torch.float32))
    return nn.Sequential(*modules)


class _StrictSocialModule(nn.Module):
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Require complete new module keys/shapes, never a permissive migration.

        This alone is not a full checkpoint/RMS/dtype/provenance admission.
        A future social checkpoint reader must perform those separate checks.
        """
        if strict is not True:
            raise ValueError("Social architecture loading requires strict=True; no legacy migration")
        return super().load_state_dict(state_dict, strict=True, assign=assign)


class SocialTaskEncoder(_StrictSocialModule):
    """Externally normalized [traj20, HWC raster12288] -> [map256, traj20]."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1, dtype=torch.float32), nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2, padding=1, dtype=torch.float32), nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2, padding=1, dtype=torch.float32), nn.ReLU(),
        )
        self.projection = _mlp(1024, (512, 256))

    def forward(self, normalized_task_obs):
        _validate_input(normalized_task_obs, TASK_DIM, "normalized task observation")
        if normalized_task_obs.device != self.cnn[0].weight.device:
            raise ValueError("Task observations and social encoder must share a device")
        trajectory = normalized_task_obs[:, :TRAJECTORY_DIM]
        raster = normalized_task_obs[:, TRAJECTORY_DIM:].reshape(-1, GRID_SIZE, GRID_SIZE, RASTER_CHANNELS)
        image = raster.permute(0, 3, 1, 2)
        map_features = self.projection(self.cnn(image).flatten(1))
        return torch.cat((map_features, trajectory), dim=-1)


class SocialAMPActorCritic(_StrictSocialModule):
    """One shared task encoder, separate actor/critic MLPs, fixed Gaussian std.

    Both inputs are normalized externally. No action sampling/clamp/mask or
    value denormalization occurs here; existing PPO/PD semantics remain caller
    responsibilities. This class has no discriminator and is not a drop-in
    complete PaperAMPNetwork or current checkpoint format.
    """

    def __init__(self, *, task_encoder_sharing):
        super().__init__()
        if type(task_encoder_sharing) is not str or task_encoder_sharing != "shared":
            raise ValueError("This source-based stage implements only an explicitly shared task encoder")
        self.task_encoder = SocialTaskEncoder()
        self.actor = _mlp(POLICY_INPUT_DIM, (2048, 1024), ACTION_DIM)
        self.critic = _mlp(POLICY_INPUT_DIM, (2048, 1024), 1)
        self.register_buffer("log_std", torch.full((ACTION_DIM,), -2.9, dtype=torch.float32))

    def _features(self, normalized_self_obs, normalized_task_obs):
        _validate_input(normalized_self_obs, SELF_DIM, "normalized self observation")
        _validate_input(normalized_task_obs, TASK_DIM, "normalized task observation")
        if (normalized_self_obs.shape[0] != normalized_task_obs.shape[0]
                or normalized_self_obs.device != normalized_task_obs.device):
            raise ValueError("Self and task observations must share batch size/device")
        return torch.cat((normalized_self_obs, self.task_encoder(normalized_task_obs)), dim=-1)

    def policy(self, normalized_self_obs, normalized_task_obs):
        features = self._features(normalized_self_obs, normalized_task_obs)
        distribution = torch.distributions.Normal(self.actor(features), self.log_std.exp())
        return distribution, self.critic(features).squeeze(-1)

    def value(self, normalized_self_obs, normalized_task_obs):
        """Differentiable normalized scalar value, without actor/Normal work."""
        return self.critic(self._features(normalized_self_obs, normalized_task_obs)).squeeze(-1)

    def forward(self, normalized_self_obs, normalized_task_obs):
        return self.policy(normalized_self_obs, normalized_task_obs)
