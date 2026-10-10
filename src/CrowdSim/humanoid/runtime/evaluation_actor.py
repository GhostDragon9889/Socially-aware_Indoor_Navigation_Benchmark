"""Evaluation-only actor facade with an explicit legacy initialization RNG seam.

Cold imports are stdlib/settings only. Loading/acting preserve caller RNG and
expose no trainer, optimizer, replay, discriminator reward or critic value API.
The separate legacy helper deliberately resets process RNG once before planner
construction; it never restores saved checkpoint RNG or advances physics.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import random

from CrowdSim.humanoid.application.settings import TrainingConfig

from .checkpoint_contract import inspect_checkpoint_file, validate_bundle_identity, validate_reconstruction_config


_MAX_PROVENANCE_BYTES = 2 * 1024 * 1024


def _json_provenance(value):
    if type(value) is not dict:
        raise ValueError("Evaluation actor requires a complete JSON provenance mapping")
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Evaluation actor provenance must contain finite JSON data, not training tensors") from error
    if len(encoded.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise ValueError("Evaluation actor provenance exceeds the declared 2MiB small-metadata limit")
    return json.loads(encoded)


class EvaluationActor:
    """A prepared deterministic actor interface, not a training/resume object.

    ``act`` returns only actions/mean_actions/applied_actions. Config/provenance
    accessors return copies; callers can add evaluation stages to their report
    copies without rewriting the selected checkpoint's training provenance.
    """

    def __init__(self, inference, provenance, update_count, physical_transition_count, *, baseline_description=None):
        self._inference = inference
        self._provenance = _json_provenance(provenance)
        self._update_count = update_count
        self._physical_transition_count = physical_transition_count
        self._baseline_description = deepcopy(baseline_description)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu", expected_sha256=None):
        from .amp_inference import ReconstructionAMPInference

        selected = inspect_checkpoint_file(checkpoint)
        # v2 owns full hash, restricted tensor/schema/normalizer validation,
        # unchanged-stat check, isolated CPU constructor and explicit device.
        inference = ReconstructionAMPInference.from_checkpoint(selected.path,
            device=device, expected_sha256=expected_sha256)
        return cls._from_loaded_inference(selected, inference, inference.describe())

    @classmethod
    def from_checkpoints(cls, base_checkpoint, style_checkpoint=None, *, alpha=0., device="cpu",
                         base_sha256=None, style_sha256=None):
        """Expose an explicit experimental blend without constructing a third NN.

        Baseline config/counters/provenance remain the baseline training record.
        The composite defines no policy likelihood, critic or AMP reward API.
        """
        if type(alpha) not in (float, int) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Experimental arm alpha must be a finite real number in [0,1], not bool")
        if style_checkpoint is None:
            if alpha != 0 or style_sha256 is not None:
                raise ValueError("A style checkpoint is required for an experimental arm alpha or style SHA")
            return cls.from_checkpoint(base_checkpoint, device=device, expected_sha256=base_sha256)
        from .arm_style import ExperimentalArmStyleInference

        selected = inspect_checkpoint_file(base_checkpoint)
        inference = ExperimentalArmStyleInference.from_checkpoints(base_checkpoint, style_checkpoint,
            alpha=alpha, device=device, base_sha256=base_sha256, style_sha256=style_sha256)
        return cls._from_loaded_inference(selected, inference, inference.baseline.describe())

    @classmethod
    def _from_loaded_inference(cls, selected, inference, baseline_description):
        import torch

        artifact = torch.load(selected.path, map_location="cpu", weights_only=True, mmap=True)
        config = validate_bundle_identity(artifact)
        provenance = _json_provenance(artifact["provenance"])
        updates, transitions = artifact["update_count"], artifact["physical_transition_count"]
        if asdict(config) != asdict(inference.config):
            raise ValueError("Evaluation metadata config differs from the validated inference tensors")
        description = baseline_description
        if (updates, transitions) != (description["update_count"], description["physical_transition_count"]):
            raise ValueError("Evaluation metadata counters differ from the validated inference checkpoint")
        if not selected.unchanged():
            raise RuntimeError("Evaluation actor checkpoint changed during metadata extraction")
        del artifact
        return cls(inference, provenance, updates, transitions, baseline_description=baseline_description)

    @property
    def config(self):
        return self._inference.config

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    @property
    def update_count(self):
        return self._update_count

    @property
    def physical_transition_count(self):
        return self._physical_transition_count

    @property
    def device(self):
        source = getattr(self._inference, "baseline", self._inference)
        return source.action_mask.device

    def act(self, self_obs, task_obs, *, deterministic=True):
        return self._inference.act(self_obs, task_obs, deterministic=deterministic)

    def describe(self):
        description = self._inference.describe()
        result = {**(self._baseline_description or description), "facade": "EvaluationActor",
            "full_training_provenance": self.provenance,
            "outputs": ["actions", "mean_actions", "applied_actions"],
            "log_probability_available": False, "critic_value_available": False,
            "AMP_reward_available": False, "saved_training_rng_restored": False,
            "legacy_rng_preparation_implicit": False,
            "runtime_deployment_changed": False}
        if "baseline" in description and "style" in description:
            result["experimental_arm_style"] = description
        return result


def prepare_legacy_evaluation_rng(config):
    """Explicitly reproduce Trainer.load(restore_rng=False) initialization RNG.

    Call once at the old load boundary, before constructing the diffusion
    planner. This resets Python/NumPy/Torch seeds and consumes only a temporary
    CPU PaperAMPNetwork initialization. torch.manual_seed retains its ordinary
    CUDA seeding behavior, including lazy seeding; CUDA RNG/physics equivalence
    require separate real runtime verification. No checkpoint RNG is restored.
    """
    mapping = asdict(config) if isinstance(config, TrainingConfig) else config
    owned = validate_reconstruction_config(mapping)
    import numpy as np
    import torch

    from CrowdSim.humanoid.training.trainer import PaperAMPNetwork

    # The legacy loader builds its network before .to(device), with the default
    # CPU float32 environment. Reject alternate global defaults rather than
    # promising the same random draw sequence under a different constructor.
    if torch.get_default_dtype() != torch.float32 or torch.get_default_device().type != "cpu":
        raise ValueError("Legacy evaluation RNG compatibility requires the default CPU float32 tensor environment")
    random.seed(owned.seed)
    np.random.seed(owned.seed)
    torch.manual_seed(owned.seed)
    temporary = PaperAMPNetwork(owned)
    parameter_count = sum(value.numel() for value in temporary.parameters())
    del temporary
    state_sha = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    return {"schema": "legacy_evaluation_initialization_rng.v1", "seed": owned.seed,
        "Python_numpy_torch_global_seeds_reset": True,
        "temporary_CPU_PaperAMPNetwork_parameters": parameter_count,
        "torch_CPU_rng_state_sha256": state_sha,
        "torch_manual_seed_CUDA_behavior": "normal seed_all/lazy seed behavior; no saved CUDA state restore",
        "saved_checkpoint_rng_restored": False, "trainer_created": False,
        "optimizer_created": False, "replay_created": False,
        "GPU_rng_equivalence_verified_by_this_helper": False,
        "physical_equivalence_verified_by_this_helper": False}
