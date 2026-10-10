"""Frozen AMP actor inference from a reconstruction training bundle.

No optimizer, replay, expert sampler, trainer constructor, RNG restore or SDK
is created. Outputs are normalized actions, not decoded physical PD targets.
This prepared adapter is not an integration with the old MaskedMimic entry.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch
from torch import nn

from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES
from CrowdSim.humanoid.training.trainer import PaperAMPNetwork, RunningNormalizer

from .checkpoint_contract import (
    CANONICAL_ACTION_MASK,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    inspect_checkpoint_file,
    validate_bundle_identity,
    validate_expected_sha256,
)


def _validate_tensor_dict(values, expected, label):
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError(f"{label} must contain exactly the declared tensor keys")
    for name, template in expected.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.shape != template.shape
                or value.dtype != template.dtype or value.device.type != "cpu"
                or value.layout != torch.strided or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid {label}.{name}: expected finite CPU {template.dtype}{tuple(template.shape)}")


def _validate_normalizers(values, config):
    dimensions = {"self_normalizer": config.self_dim, "task_normalizer": config.task_dim,
                  "amp_normalizer": config.amp_frame_dim * (
                      config.history_frames if config.amp_normalization == "per_history_coordinate" else 1),
                  "value_normalizer": 1}
    if not isinstance(values, dict) or set(values) != set(dimensions):
        raise ValueError("Inference bundle requires exactly all four saved normalizers")
    for label, dimension in dimensions.items():
        expected = {"mean": torch.empty(dimension, dtype=torch.float64),
                    "variance": torch.empty(dimension, dtype=torch.float64),
                    "count": torch.empty((), dtype=torch.float64)}
        _validate_tensor_dict(values[label], expected, label)
        if bool((values[label]["variance"] < 0).any()) or bool(values[label]["count"] < 0):
            raise ValueError(f"Invalid {label}: variance/count must be nonnegative")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_copy(value, label):
    """Retain small JSON metadata, never tensors or training state references."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON {label} provenance") from error


class ReconstructionAMPInference(nn.Module):
    """Deterministic69-action actor with frozen selected-checkpoint moments.

    Use ``from_checkpoint``. ``act`` accepts finite float32 observations with
    shapes[B,357]/[B,1044]. It returns ``actions``/``mean_actions`` (Gaussian
    means) and ``applied_actions`` (clamp times the exact57-active mask). Pass
    applied_actions once to AMPWorld.step; do not decode PD here or twice.
    """

    def __init__(self, config):
        super().__init__()
        self._config = deepcopy(config)
        # Linear layer initialization consumes CPU random numbers even when
        # immediately loading weights. Restore the caller's CPU state on all
        # constructor success/failure paths without touching CUDA RNG/devices.
        with torch.random.fork_rng(devices=[]):
            self.model = PaperAMPNetwork(self._config)
            self.self_normalizer = RunningNormalizer(config.self_dim, config.normalization_clip)
            self.task_normalizer = RunningNormalizer(config.task_dim, config.normalization_clip)
        frozen = {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
        named_mask = tuple(int(name.rsplit("_", 1)[0] not in frozen) for name in PACER_DOF_NAMES)
        if named_mask != CANONICAL_ACTION_MASK:
            raise ValueError("Current named SMPL order disagrees with the inference physical-action contract")
        self.register_buffer("action_mask", torch.tensor(named_mask, dtype=torch.float32))
        self._metadata = None

    @property
    def config(self):
        return deepcopy(self._config)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu", expected_sha256=None):
        """Restricted CPU/mmap load, strict inference tensors, then explicit device.

        Optimizer/replay/runner/RNG tensors are neither restored nor retained.
        AMP/value moments are checked for integrity and then discarded. Their
        presence does not claim optimizer or physical resume validation.
        """
        expected_sha256 = validate_expected_sha256(expected_sha256)
        selected = inspect_checkpoint_file(checkpoint)
        sha256 = _sha256(selected.path)
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ValueError("AMP inference checkpoint SHA256 differs from the selected immutable artifact")
        artifact = torch.load(selected.path, map_location="cpu", weights_only=True, mmap=True)
        config = validate_bundle_identity(artifact)
        if config.mode == "strict_author_experiment" and not artifact["provenance"].get("strict_manifest_gate_passed"):
            raise ValueError("Strict author inference lacks its declared verified author manifest gate")
        _validate_normalizers(artifact.get("normalizers"), config)
        inference = cls(config)
        _validate_tensor_dict(artifact.get("model"), inference.model.state_dict(), "model")
        log_std = artifact["model"]["log_std"]
        if not torch.isfinite(log_std.exp()).all() or bool((log_std.exp() <= 0).any()):
            raise ValueError("Model Gaussian scale is not finite and positive")
        inference.model.load_state_dict(artifact["model"], strict=True)
        for name in ("self_normalizer", "task_normalizer"):
            getattr(inference, name).load_state_dict(artifact["normalizers"][name], strict=True)
        if not selected.unchanged():
            raise RuntimeError("AMP inference checkpoint changed during restricted load")
        provenance = artifact["provenance"]
        initialization = provenance.get("initialization", {})
        inference._metadata = {"checkpoint": str(selected.path), "checkpoint_sha256": sha256,
            "checkpoint_bytes": selected.bytes, "format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION,
            "update_count": artifact["update_count"], "physical_transition_count": artifact["physical_transition_count"],
            "dataset_provenance": _metadata_copy(provenance.get("dataset"), "dataset"),
            "initialization": _metadata_copy({key: initialization.get(key) for key in ("mode", "path", "sha256")}, "initialization")
                if isinstance(initialization, dict) else None}
        del artifact
        inference.requires_grad_(False)
        inference.eval()
        inference.to(torch.device(device))
        return inference

    def _validate_inference_storage(self):
        # nn.Module.double()/half()/float() can mutate checkpoint dtypes after
        # restricted loading. Never let a converted mask redefine the input
        # contract, or silently cast a converted model/statistic back again.
        device = self.action_mask.device
        if self.action_mask.dtype != torch.float32:
            raise ValueError("AMP inference storage requires a float32 action mask; dtype conversion is unsupported")
        for name, value in (*self.model.named_parameters(), *self.model.named_buffers()):
            if value.dtype != torch.float32 or value.device != device:
                raise ValueError(f"AMP inference model.{name} requires float32 on action-mask device {device}")
        for label in ("self_normalizer", "task_normalizer"):
            for name, value in getattr(self, label).named_buffers():
                if value.dtype != torch.float64 or value.device != device:
                    raise ValueError(f"AMP inference {label}.{name} requires float64 on action-mask device {device}")

    @torch.inference_mode()
    def act(self, self_obs, task_obs, *, deterministic=True):
        if self._metadata is None:
            raise RuntimeError("AMP inference requires a completely validated loaded checkpoint")
        if deterministic is not True:
            raise ValueError("This frozen deployment adapter requires deterministic=True; no RNG is sampled")
        self._validate_inference_storage()
        for observations, dimension, label in ((self_obs, 357, "self_obs"), (task_obs, 1044, "task_obs")):
            if (not isinstance(observations, torch.Tensor) or observations.ndim != 2
                    or observations.shape[0] < 1 or observations.shape[1] != dimension
                    or observations.dtype != torch.float32 or not torch.isfinite(observations).all()):
                raise ValueError(f"{label} requires nonempty finite float32[B,{dimension}]")
        if self_obs.shape[0] != task_obs.shape[0]:
            raise ValueError("Self/task observation batch sizes differ")
        self_obs, task_obs = self_obs.to(self.action_mask.device), task_obs.to(self.action_mask.device)
        if self._config.normalize_observations:
            self_obs = self.self_normalizer.normalize(self_obs)
            task_obs = self.task_normalizer.normalize(task_obs)
        distribution, _ = self.model.policy(self_obs, task_obs)
        raw = distribution.mean
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Loaded AMP actor produced nonfinite normalized action means")
        return {"actions": raw, "mean_actions": raw, "applied_actions": raw.clamp(-1, 1) * self.action_mask}

    def describe(self):
        if self._metadata is None:
            raise RuntimeError("No validated inference checkpoint")
        return {**deepcopy(self._metadata), "config": asdict(self._config),
                "controller": "independent reconstruction AMP inference", "status": "prepared_not_runtime_integrated",
                "observation_dimensions": {"self": 357, "task": 1044},
                "action_contract": {"dimensions": 69, "active": 57, "space": "normalized",
                    "projection": "clamp[-1,1] times canonical hand/toe mask", "PD_decoded": False},
                "normalizers": {"retained": ["self_normalizer", "task_normalizer"],
                    "validated_only": ["amp_normalizer", "value_normalizer"], "clip": 5., "epsilon": 1e-5,
                    "updated_by_inference": False},
                "optimizer_created": False, "replay_created": False, "expert_sampler_created": False,
                "training_rng_restored": False, "physical_validation_performed": False}
