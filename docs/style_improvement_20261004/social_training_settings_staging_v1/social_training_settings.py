"""SDK/Torch-free social feature/config validation; not a runnable trainer.

The original TrainingConfig continues to reject social symmetry observations.
Only its shared hyperparameter validation is reused here; social observation,
network and initialization guards are independently enforced beforehand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE_SOURCE_SHA = "5254c5b89d75d3eef52de8b5463db12801e8605cb02bff4d95551963171dc7dc"
_base_path = REPO / "CrowdSim/humanoid/application/settings.py"
if hashlib.sha256(_base_path.read_bytes()).hexdigest() != BASE_SOURCE_SHA:
    raise ValueError("Shared numeric configuration source identity changed")

from CrowdSim.humanoid.application import settings as _base_settings  # noqa: E402

if Path(_base_settings.__file__).resolve() != _base_path:
    raise ValueError("Imported numeric configuration is from another checkout")

TrainingConfig = _base_settings.TrainingConfig
SCHEMA = "local.neutral_smpl.social_training_settings.v1"
ARCHITECTURE_ID = "local.neutral_smpl.shared_social_CNN_actor_critic.v1"
RASTER_SHA = "234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe"
AMASS_MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
# Canonical PACER23 joint triples: L/R_Toe and L/R_Hand are physically frozen.
_frozen_coordinates = {9, 10, 11, 21, 22, 23, 51, 52, 53, 66, 67, 68}
PHYSICAL_ACTION_MASK = tuple(int(index not in _frozen_coordinates) for index in range(69))


@dataclass(kw_only=True)
class SocialTrainingConfig(TrainingConfig):
    """New feature contract with original shared PPO/reward numerical settings.

    Required grid/reference/self choices prevent inferring semantics from12308.
    Their SDK producer, whole-task RMS, social mirror and new network factory
    must be wired before this configuration can drive physical training.
    """

    grid_center: str
    grid_heading: str
    height_reference: str
    include_ego: bool
    actor_count: int
    self_dim: int = 357
    task_dim: int = 12308
    amp_frame_dim: int = 195
    history_frames: int = 10
    task_layers: tuple[int, ...] = (512, 256)
    policy_layers: tuple[int, ...] = (2048, 1024)
    discriminator_layers: tuple[int, ...] = (1024, 512)
    log_std: float = -2.9
    motion_symmetry_coefficient: float = 50.
    effective_action_likelihood: bool = True
    task_terrain_statistics_mode: str = "legacy"
    architecture_id: str = ARCHITECTURE_ID
    raster_source_sha256: str = RASTER_SHA
    amass_train_manifest_sha256: str = AMASS_MANIFEST_SHA

    @property
    def group_count(self):
        return self.actor_count // 4

    def validate(self):
        for name, allowed in (
            ("grid_center", {"root", "head"}),
            ("grid_heading", {"root", "head"}),
            ("height_reference", {"root_z", "static_center_ground_z"}),
        ):
            value = getattr(self, name)
            if type(value) is not str or value not in allowed:
                raise ValueError(f"Explicit social {name} must be one of {sorted(allowed)}")
        if type(self.include_ego) is not bool:
            raise ValueError("Social include_ego must be an explicit bool")
        if type(self.actor_count) is not int or self.actor_count < 4 or self.actor_count % 4:
            raise ValueError("Social actor_count must be divisible by4; it is not SDK num_envs")
        fixed = {
            "self_dim": 357, "task_dim": 12308, "amp_frame_dim": 195, "history_frames": 10,
            "log_std": -2.9, "architecture_id": ARCHITECTURE_ID,
            "raster_source_sha256": RASTER_SHA, "amass_train_manifest_sha256": AMASS_MANIFEST_SHA,
        }
        for name, expected in fixed.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"Social architecture feature identity differs: {name}")
        for name, expected in (("task_layers", (512, 256)), ("policy_layers", (2048, 1024)),
                               ("discriminator_layers", (1024, 512))):
            value = getattr(self, name)
            if type(value) is not tuple or value != expected:
                raise ValueError(f"Social fixed projection/head dimensions differ: {name}")
        if (self.normalize_observations is not True or self.normalize_values is not True
                or type(self.normalization_clip) is not float or self.normalization_clip != 5.):
            raise ValueError("Social training requires external input/value RMS with clip5")
        if self.task_terrain_statistics_mode != "legacy":
            raise ValueError("Old1044 protected terrain suffix is incompatible with the new full-task RMS")
        if (self.mode != "paper_method_reconstruction" or self.warm_start_discriminator is not False
                or self.warm_start_action_discriminator_from_state_prior is not False
                or self.warm_start_normalizer_count_cap is not None):
            raise ValueError("This stage declares fresh local initialization; legacy warm starts/author claims are unsupported")
        if self.freeze_toes is not True or self.effective_action_likelihood is not True:
            raise ValueError("Social fixed neutral PD contract retains toe/hand freeze and effective action likelihood")
        if self.action_mask is not None and (
                type(self.action_mask) is not tuple or any(type(value) is not int for value in self.action_mask)
                or self.action_mask != PHYSICAL_ACTION_MASK):
            raise ValueError("Explicit social action_mask must exactly preserve the canonical57 effective coordinates")
        if (type(self.motion_symmetry_coefficient) not in (int, float)
                or not math.isfinite(self.motion_symmetry_coefficient) or self.motion_symmetry_coefficient < 0):
            raise ValueError("Social raw-observation symmetry coefficient must be finite and nonnegative")
        if (type(self.control_dt) not in (int, float) or self.control_dt != 1 / 30
                or type(self.target_horizon_s) not in (int, float) or self.target_horizon_s != 5.):
            raise ValueError("Social task timing requires30Hz control and five-second future horizon")
        # Reuse only original shared numeric/enum validation on a detached
        # shadow. Original feature-specific checks remain intact. The actual
        # config retains12308 and its full symmetry coefficient throughout;
        # this shadow is never returned, saved or passed to a trainer.
        shared = {field.name: getattr(self, field.name) for field in fields(TrainingConfig)}
        shared["task_dim"] = 1044
        shared["motion_symmetry_coefficient"] = 0.
        TrainingConfig(**shared).validate()

    def to_document(self):
        self.validate()
        return {
            "schema": SCHEMA, "status": "declared_configuration_only",
            "configuration": asdict(self),
            "feature_layout": {"raw_task": "trajectory20,HWC64x64x3", "encoded_task": "map256,trajectory20",
                               "task_encoder_shared": True, "actor_critic_input": 633},
            "logical_layout": {"actors": self.actor_count, "SDK_envs": self.group_count, "actors_per_env": 4,
                               "group_rows": "[2g,2g+1,N/2+2g,N/2+2g+1]"},
            "normalization": {"self_dimensions": 357, "whole_task_dimensions": 12308, "value_dimensions": 1,
                              "moments": "local float64 population, count0, frozen rollout/PPO, post-update raw samples"},
            "action_projection": {"raw_dimensions": 69, "effective_dimensions": 57,
                                  "physical_mask": list(PHYSICAL_ACTION_MASK), "projection": "clamp[-1,1] then mask"},
            "initialization": "fresh; no checkpoint or moments loaded by this configuration",
            "runtime_validation_performed": False,
        }

    @classmethod
    def from_document(cls, document):
        if type(document) is not dict or document.get("schema") != SCHEMA:
            raise ValueError("New social configuration schema is required")
        values = document.get("configuration")
        if type(values) is not dict or set(values) != {field.name for field in fields(cls)}:
            raise ValueError("Social configuration fields must be complete and exact")
        values = dict(values)
        for name in ("task_layers", "policy_layers", "discriminator_layers", "action_mask"):
            if isinstance(values[name], list):
                values[name] = tuple(values[name])
        result = cls(**values)
        result.validate()
        # The derived layout/RMS/provenance fields are part of this declaration
        # too. JSON's list representation is compared without altering inputs.
        import json
        if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(result.to_document(), sort_keys=True, allow_nan=False):
            raise ValueError("Social declared layout/normalization/provenance fields changed")
        return result

    def detached_copy(self, **changes):
        """New dataclass instance; no network/optimizer or initialization occurs."""
        result = replace(self, **changes)
        result.validate()
        return result
