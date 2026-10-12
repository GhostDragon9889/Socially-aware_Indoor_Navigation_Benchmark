"""Critic-only, differentiable evaluation snapshot of an existing social trainer.

Import is stdlib-only. Constructing/calling this object uses the already
imported trainer's Torch objects; neither operation has been executed here.
No policy, discriminator, optimizer, replay or new normalizer is constructed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
SCHEMA = "local.neutral_smpl.frozen_social_critic_snapshot.v1"
PINS = {
    "trainer": ("resources/runtime_sources/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py", 24289,
                '11d304d7002d31c96c2fcdf3766aa51616c84560e58bc11f7eeababb068c559b'),
    "policy": ("resources/runtime_sources/style_improvement_20261004/social_policy_staging_v1/social_policy.py", 6763,
               '2a4007f732f0854d05b784debcbd0798e0c587149ba8b63d623378ad1ab3ac15'),
    "raw_task_layout": ("resources/runtime_sources/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py", 8512,
                        '6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1'),
    "normalizer": ("CrowdSim/humanoid/training/trainer.py", 38177,
                   "707f355acbd8b5d33aa40c3299a2540610ea695ae4bb9705ede3498e27224f9d"),
}


def _record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_IMPORT_RECORD = _record(HERE / "value_adapter.py")


def _sources():
    records = {}
    for name, (relative, size, sha) in PINS.items():
        record = _record(ROOT / relative)
        if record["bytes"] != size or record["sha256"] != sha:
            raise ValueError("Frozen critic adapter dependency changed: " + name)
        records[name] = record
    if _record(HERE / "value_adapter.py") != _IMPORT_RECORD:
        raise ValueError("Critic adapter file changed after its initial import")
    records["execution_adapter"] = dict(_IMPORT_RECORD)
    return records


def _trainer_module(trainer):
    module = sys.modules.get(type(trainer).__module__)
    if (module is None or Path(getattr(module, "__file__", "")).resolve() != (ROOT / PINS["trainer"][0]).resolve()
            or type(trainer) is not getattr(module, "SocialAMPTrainer", None)):
        raise TypeError("An already constructed, source-pinned SocialAMPTrainer is required")
    return module


class SocialValueSnapshot:
    """Owned encoder/critic + three RMS copies, with frozen parameters.

    Caller must provide a quiescent trained object (no concurrent updates).
    Positive counts are an admission boundary, not evidence of physical
    training quality or safe critic preferences. No current trainer reference
    is retained after construction, and later calls read only owned copies.
    """

    def __init__(self, trainer):
        sources = _sources()
        module = _trainer_module(trainer)
        torch = module.torch
        if torch.is_inference_mode_enabled():
            raise ValueError("A differentiable snapshot cannot be created under inference_mode")
        trainer.config.validate()
        if trainer.config.normalize_observations is not True or trainer.config.normalize_values is not True:
            raise ValueError("This snapshot accepts only explicitly normalized observation/value configurations")
        if type(trainer.model) is not module.SocialAMPNetwork:
            raise TypeError("Existing model must be the source-pinned social CNN/D network")
        if type(trainer.model.task_encoder) is not module._policy.SocialTaskEncoder:
            raise TypeError("Existing encoder must have the declared social CNN architecture")
        if (type(trainer.update_count) is not int or type(trainer.physical_transition_count) is not int
                or trainer.update_count <= 0 or trainer.physical_transition_count <= 0
                or trainer.physical_transition_count % trainer.config.actor_count):
            raise ValueError("A zero-count/untrained critic cannot be exposed as a candidate value")
        device = torch.device(trainer.device)
        if device.type == "cpu":
            device = torch.device("cpu")
        if device.type not in ("cpu", "cuda") or device.type == "cuda" and device.index is None:
            raise ValueError("Snapshot requires CPU or an explicit CUDA index")
        self._torch, self._device = torch, device
        transitions = trainer.physical_transition_count
        children = (trainer.model.task_encoder, trainer.model.critic)
        self._value_modules(children)
        normalizers = (trainer.self_normalizer, trainer.task_normalizer, trainer.value_normalizer)
        for normalizer, dimension in zip(normalizers, (357, 12308, 1), strict=True):
            if type(normalizer) is not module._base.RunningNormalizer:
                raise TypeError("The original declared population normalizer is required")
            self._normalizer(normalizer, dimension, transitions)
        config_document = deepcopy(trainer.config.to_document())
        provenance = deepcopy(trainer.provenance)
        source_contract = deepcopy(trainer._checkpoint_contract())
        updates = trainer.update_count
        # Copy ONLY existing value-path modules and three RMS. A deepcopy of
        # trainer.model would wrongly include actor, D and log_std. No fresh
        # model/normalizer constructor, RNG seed, .to(), or parameter sampling.
        self._encoder, self._critic, self._self_rms, self._task_rms, self._value_rms = deepcopy((*children, *normalizers))
        for child in (self._encoder, self._critic, self._self_rms, self._task_rms, self._value_rms):
            child.eval()
            child.requires_grad_(False)
        self._value_modules((self._encoder, self._critic))
        for normalizer, dimension in zip((self._self_rms, self._task_rms, self._value_rms), (357, 12308, 1), strict=True):
            self._normalizer(normalizer, dimension, transitions)
        self._metadata = {
            "schema": SCHEMA, "sources": sources, "trainer_source_contract": source_contract,
            "trainer_provenance": provenance, "configuration_document": config_document,
            "update_count": updates, "physical_transition_count": transitions,
            "RMS_counts": {"self": float(self._self_rms.count), "task": float(self._task_rms.count), "value": float(self._value_rms.count)},
            "copied_components": ["task_encoder", "critic", "self_RMS", "task_RMS", "value_RMS"],
            "actor_D_AMP_RMS_optimizer_replay_copied": False,
            "weights_and_moments": "owned deepcopy of existing value path; no random initialization",
            "normalization": "original clip5/epsilon1e-5 and input dtype conversion; moments never updated",
            "quiescent_source_is_caller_responsibility": True,
            "training_quality_or_collision_safety_validated": False,
            "physical_state_or_clock_independently_observed": False,
            "world_to_local_candidate_conversion": "caller responsibility; this API only accepts root-local metres",
        }

    def _value_modules(self, children):
        torch = self._torch
        for child in children:
            for value in (*child.parameters(), *child.buffers()):
                if (value.dtype != torch.float32 or value.device != self._device or value.is_inference()
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("Existing/copied value-path tensors must be finite, non-inference float32 on the declared device")

    def _normalizer(self, normalizer, dimension, transitions):
        torch = self._torch
        if normalizer.clip != 5.:
            raise ValueError("Original declared normalization clip5 is required")
        for name, shape in (("mean", (dimension,)), ("variance", (dimension,)), ("count", ())):
            value = getattr(normalizer, name)
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float64
                    or value.device != self._device or value.is_inference() or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid existing/copied RMS tensor: " + name)
        if bool((normalizer.variance < 0).any()) or float(normalizer.count) != transitions:
            raise ValueError("Fresh-trained self/task/value moments must match positive physical transitions")

    def _input(self, value, shape, name):
        torch = self._torch
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != torch.float32
                or value.device != self._device or value.device.type == "meta" or not bool(torch.isfinite(value).all())):
            raise ValueError(name + " requires finite float32 of the exact shape on the snapshot device")
        if torch.is_inference_mode_enabled() or value.is_inference():
            raise ValueError("Inference-mode contexts/tensors cannot supply differentiable values")
        return value

    def _observations(self, raw_self, raw_task):
        if not isinstance(raw_self, self._torch.Tensor) or raw_self.ndim != 2 or len(raw_self) < 1:
            raise ValueError("Raw self requires a nonempty batch")
        batch = len(raw_self)
        self._input(raw_self, (batch, 357), "raw_self")
        self._input(raw_task, (batch, 12308), "raw_task")
        return batch

    def metadata(self):
        """Return isolated captured metadata; never reread later trainer state."""
        return deepcopy(self._metadata)

    def value_raw(self, raw_self, raw_task):
        """Denormalized value[B], preserving caller input gradients.

        Frozen parameters do not imply no_grad. Original RMS conversion,
        epsilon and clamp are retained; clipped inputs may have zero gradients.
        No Normal/actor construction, moment update, or trainer.value call.
        """
        batch = self._observations(raw_self, raw_task)
        torch = self._torch
        with torch.enable_grad():
            normalized_self = self._self_rms.normalize(raw_self)
            normalized_task = self._task_rms.normalize(raw_task)
            features = torch.cat((normalized_self, self._encoder(normalized_task)), dim=-1)
            normalized_value = self._critic(features).squeeze(-1)
            value = self._value_rms.denormalize(normalized_value[:, None]).squeeze(-1)
        self._input(value, (batch,), "denormalized value")
        return value

    def bind_current(self, raw_self, raw_task):
        """Own detached current self and raster suffix; candidates replace20 only."""
        self._observations(raw_self, raw_task)
        return _BoundCandidateValue(self, raw_self.detach().clone(), raw_task[:, 20:].detach().clone())


class _BoundCandidateValue:
    """Current body/raster are fixed; candidate root-local trajectory varies."""

    def __init__(self, snapshot, owned_self, owned_raster):
        self._snapshot = snapshot
        self._self, self._raster = owned_self, owned_raster

    def candidate_value(self, local_root_xy):
        """float32[B,S,10,2] -> denormalized[B,S]; S>0, +.5..5s in metres.

        Caller must convert world trajectories relative to the SAME current
        root XY/heading outside this API. No scale10, extrapolation or waypoint
        resampling occurs. This evaluates values; it does not choose a plan,
        step size, score, collision mask or optimizer.
        """
        snapshot, torch = self._snapshot, self._snapshot._torch
        batch = len(self._self)
        if (not isinstance(local_root_xy, torch.Tensor) or local_root_xy.ndim != 4
                or local_root_xy.shape[0] != batch or local_root_xy.shape[1] < 1
                or tuple(local_root_xy.shape[2:]) != (10, 2)):
            raise ValueError("Candidate trajectory requires nonempty Tensor[B,S,10,2] matching the bound batch")
        samples = local_root_xy.shape[1]
        snapshot._input(local_root_xy, (batch, samples, 10, 2), "candidate local root trajectory")
        with torch.enable_grad():
            prefix = local_root_xy.reshape(batch * samples, 20)
            suffix = self._raster[:, None].expand(batch, samples, 12288).reshape(batch * samples, 12288)
            raw_task = torch.cat((prefix, suffix), dim=-1)
            raw_self = self._self[:, None].expand(batch, samples, 357).reshape(batch * samples, 357)
            return snapshot.value_raw(raw_self, raw_task).reshape(batch, samples)
