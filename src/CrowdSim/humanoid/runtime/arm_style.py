"""Experimental eval-only upper-limb blend of two reconstruction AMP actors.

The actors use their own frozen observation moments. Only 24 canonical upper
limb normalized mean coordinates can change; no PD decoding, optimizer,
discriminator substitution, pose injection, or physics fallback occurs here.
This is not the paper's single trained actor and has no physical safety claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math

import torch
from torch import nn

from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, PACER_DOF_NAMES

from .amp_inference import ReconstructionAMPInference
from .checkpoint_contract import (
    CANONICAL_ACTION_MASK,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    inspect_checkpoint_file,
    validate_bundle_identity,
    validate_reconstruction_config,
)


ARM_BODIES = ("L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
              "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist")
ARM_ACTION_NAMES = tuple(f"{body}_{axis}" for body in ARM_BODIES for axis in "xyz")
ARM_ACTION_INDICES = tuple(PACER_DOF_NAMES.index(name) for name in ARM_ACTION_NAMES)
BASE_ACTION_INDICES = tuple(index for index in range(69) if index not in ARM_ACTION_INDICES)
DOMAIN_FIELDS = ("terrain_extent_m", "isolated_spawn_margin_m")


def _validate_alpha(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Experimental arm blend alpha must be a finite real number in [0,1], not bool")
    return float(value)


def _checkpoint_physics(actor):
    """Read JSON physics only from the exact already-loaded restricted bundle."""
    description = actor.describe()
    selected = inspect_checkpoint_file(description["checkpoint"])
    digest = hashlib.sha256()
    with selected.path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if selected.bytes != description["checkpoint_bytes"] or digest.hexdigest() != description["checkpoint_sha256"]:
        raise ValueError("Arm blend physics artifact differs from its loaded inference checkpoint")
    artifact = torch.load(selected.path, weights_only=True, map_location="cpu", mmap=True)
    config = validate_bundle_identity(artifact)
    if asdict(config) != asdict(actor.config):
        raise ValueError("Arm blend physics artifact config differs from loaded actor")
    try:
        physics = json.loads(json.dumps(artifact["provenance"]["physics"], allow_nan=False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Arm blend requires complete finite JSON physics provenance") from error
    del artifact
    if not selected.unchanged():
        raise RuntimeError("Arm blend checkpoint changed during physics provenance load")
    required = {"engine", "physics_hz", "control_hz", "rig", "body_names", "dof_names",
                "dof_limits_lower", "dof_limits_upper", "joint_control", "terrain_material"}
    if (type(physics) is not dict or not required <= set(physics) or physics["engine"] != "PhysX"
            or physics["physics_hz"] != 120 or physics["control_hz"] != 30
            or type(physics["rig"]) is not str or not physics["rig"]
            or physics["body_names"] != list(PACER_BODY_NAMES) or physics["dof_names"] != list(PACER_DOF_NAMES)):
        raise ValueError("Arm blend requires the same explicit canonical SMPL PhysX120/30Hz rig contract")
    limits = [physics["dof_limits_lower"], physics["dof_limits_upper"]]
    if (any(type(values) is not list or len(values) != 69 for values in limits)
            or any(type(value) not in (float, int) or not math.isfinite(value) for values in limits for value in values)
            or any(low >= high for low, high in zip(*limits))):
        raise ValueError("Arm blend requires complete finite69-axis physical DOF limits")
    controls = physics["joint_control"]
    if type(controls) is not dict or set(controls) != set(PACER_DOF_NAMES):
        raise ValueError("Arm blend requires named69-axis joint control provenance")
    for name, values in controls.items():
        if type(values) is not dict or not {"stiffness", "damping", "armature", "effort_limit", "velocity_limit"} <= set(values):
            raise ValueError(f"Incomplete physical joint control: {name}")
        for key in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit"):
            value = values[key]
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or value < 0 or (key in {"stiffness", "effort_limit", "velocity_limit"} and value <= 0)):
                raise ValueError(f"Invalid physical joint control: {name}.{key}")
        friction = values.get("friction")
        if friction is not None and (type(friction) not in (float, int) or not math.isfinite(friction) or friction < 0):
            raise ValueError(f"Invalid physical joint control: {name}.friction")
    material = physics["terrain_material"]
    if type(material) is not dict or not {"static_friction", "dynamic_friction", "restitution", "height_offset", "combine_mode"} <= set(material):
        raise ValueError("Arm blend requires complete physical floor material provenance")
    for key in ("static_friction", "dynamic_friction", "restitution", "height_offset"):
        value = material[key]
        if type(value) not in (float, int) or not math.isfinite(value) or (key != "height_offset" and value < 0):
            raise ValueError(f"Invalid physical floor material: {key}")
    if type(material["combine_mode"]) is not str or not material["combine_mode"]:
        raise ValueError("Explicit physical material combine mode is required")
    for key in DOMAIN_FIELDS:
        if key in physics:
            value = physics[key]
            if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
                    or (key == "terrain_extent_m" and value == 0)):
                raise ValueError(f"Invalid declared training domain: {key}")
    return physics


class ExperimentalArmStyleInference(nn.Module):
    """Blend raw upper-limb means, then return once-projected normalized actions.

    Pass only ``applied_actions`` once to AMPWorld.step. No Gaussian, log_prob,
    critic value, or AMP reward is defined for this composite controller.
    alpha=0 returns the baseline result directly to preserve its exact bytes.
    """

    def __init__(self, baseline, style, *, alpha=0.):
        super().__init__()
        self._alpha = _validate_alpha(alpha)
        self.baseline = baseline
        self.style = style
        if (len(ARM_ACTION_INDICES) != 24 or len(set(ARM_ACTION_INDICES)) != 24
                or not all(CANONICAL_ACTION_MASK[index] == 1 for index in ARM_ACTION_INDICES)):
            raise ValueError("Canonical upper-limb coordinates must be 24 unique active axes")
        self._validate_actors()
        base_physics, style_physics = _checkpoint_physics(baseline), _checkpoint_physics(style)
        shared_base = {key: value for key, value in base_physics.items() if key not in DOMAIN_FIELDS}
        shared_style = {key: value for key, value in style_physics.items() if key not in DOMAIN_FIELDS}
        if shared_base != shared_style:
            raise ValueError("Arm blend actors differ in physical rig/body/DOF/PD/material contract")
        self._physics_contract = {"verified_shared_physics": shared_base,
            "allowed_training_domain_fields": list(DOMAIN_FIELDS),
            "training_domains": {"baseline": {key: base_physics.get(key) for key in DOMAIN_FIELDS},
                                 "style": {key: style_physics.get(key) for key in DOMAIN_FIELDS}},
            "artifact_identity": "Physics provenance read from the exact loaded checkpoint SHA/size; stat unchanged during load"}
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoints(cls, base_checkpoint, style_checkpoint, *, alpha=0., device="cpu",
                         base_sha256=None, style_sha256=None):
        """Use the same restricted v2 inference loader for both selected artifacts."""
        alpha = _validate_alpha(alpha)  # Reject bad blend configuration before either load.
        baseline = ReconstructionAMPInference.from_checkpoint(
            base_checkpoint, device=device, expected_sha256=base_sha256,
        )
        style = ReconstructionAMPInference.from_checkpoint(
            style_checkpoint, device=device, expected_sha256=style_sha256,
        )
        return cls(baseline, style, alpha=alpha)

    @property
    def alpha(self):
        return self._alpha

    @property
    def config(self):
        """An owned baseline config, not a composite training-policy declaration."""
        return self.baseline.config

    def _validate_actors(self):
        device = None
        for label, actor in (("baseline", self.baseline), ("style", self.style)):
            if not isinstance(actor, ReconstructionAMPInference):
                raise ValueError(f"{label} must be a fully loaded ReconstructionAMPInference v2")
            description = actor.describe()  # Reject incomplete constructor-only instances.
            if (description.get("format") != CHECKPOINT_FORMAT or description.get("version") != CHECKPOINT_VERSION
                    or description.get("action_contract", {}).get("space") != "normalized"
                    or description["action_contract"].get("PD_decoded") is not False):
                raise ValueError(f"{label} has an incompatible reconstruction inference schema")
            validate_reconstruction_config(asdict(actor.config))
            actor._validate_inference_storage()  # Frozen v2 FP32/model + FP64 moments contract.
            current_device = actor.action_mask.device
            if device is not None and device != current_device:
                raise ValueError("Both arm-style actors must use one identical explicit device")
            device = current_device
            expected = torch.tensor(CANONICAL_ACTION_MASK, dtype=torch.float32, device=device)
            if actor.action_mask.shape != (69,) or not torch.equal(actor.action_mask, expected):
                raise ValueError(f"{label} requires the exact canonical57-active action mask")

    @torch.inference_mode()
    def act(self, self_obs, task_obs, *, deterministic=True):
        if deterministic is not True:
            raise ValueError("Experimental arm blending requires deterministic=True")
        if self.training or self.baseline.training or self.style.training:
            raise RuntimeError("Experimental arm blending is evaluation-only; call eval()")
        self._validate_actors()  # Reject post-load mask/dtype/device/schema changes even at alpha0.
        baseline = self.baseline.act(self_obs, task_obs, deterministic=True)
        if self._alpha == 0.:
            return baseline  # No arithmetic/copy/style forward can change signed zero or FP32 bytes.
        style = self.style.act(self_obs, task_obs, deterministic=True)
        raw = baseline["mean_actions"].clone()
        indices = list(ARM_ACTION_INDICES)
        if self._alpha == 1.:
            raw[:, indices] = style["mean_actions"][:, indices]
        else:
            raw[:, indices] = ((1. - self._alpha) * baseline["mean_actions"][:, indices]
                               + self._alpha * style["mean_actions"][:, indices])
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Experimental arm blend produced nonfinite normalized action means")
        # Per-actor projected outputs are intentionally discarded. This final
        # projection is applied to the composite raw means, never to PD targets.
        applied = raw.clamp(-1, 1) * self.baseline.action_mask
        return {"actions": raw, "mean_actions": raw, "applied_actions": applied}

    def describe(self):
        self._validate_actors()
        return {"schema": "experimental_reconstruction_amp_arm_style.v1",
            "controller": "experimental baseline lower-body/torso plus style upper-limb normalized-mean blend",
            "status": "prepared_not_runtime_integrated", "alpha": self._alpha,
            "baseline": self.baseline.describe(), "style": self.style.describe(),
            "physical_compatibility": deepcopy(self._physics_contract),
            "arm_action_names": list(ARM_ACTION_NAMES), "arm_action_indices": list(ARM_ACTION_INDICES),
            "blended_active_coordinates": 24, "baseline_other_active_coordinates": 33,
            "baseline_frozen_coordinates": 12, "action_dimensions": 69,
            "projection": "blend24 raw means, then clamp[-1,1] times canonical57-active mask",
            "observation_normalization": "each selected actor uses its own frozen self/task RMS",
            "alpha0_contract": "direct baseline outputs; style forward not called; bytewise baseline",
            "PD_decoded": False, "deterministic_only": True, "optimizer_created": False,
            "composite_log_prob_available": False, "composite_critic_available": False,
            "composite_AMP_reward_available": False, "main_discriminator_changed": False,
            "paper_single_actor_reproduction": False, "physical_validation_performed": False,
            "physical_safety_guaranteed": False, "pose_injection_or_teleport": False,
            "limitations": ["Upper-limb changes can alter balance, ground contacts, and navigation; baseline legs do not guarantee stability.",
                "This is an engineering composite of two policies, not a newly trained or paper-exact AMP actor.",
                "CPU coordinate/RNG parity is not a real physical deployment or style quality result."]}
