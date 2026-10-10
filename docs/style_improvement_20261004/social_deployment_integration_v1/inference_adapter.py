"""Owned deterministic social actor snapshot; no checkpoint or SDK loader.

Import uses stdlib only. Future from_trainer accepts the existing trained
source-bound object and copies only its inference path, without initialization.
Neither construction nor inference has been executed for this source stage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HELPER = ROOT / "docs/style_improvement_20261004/social_value_adapter_staging_v1/value_adapter.py"
HELPER_BYTES = 12573
HELPER_SHA = "9875c58b90e23ab6613ee7298bb4a199295435ad7f540f5b760e524a8f3b4f76"
SCHEMA = "local.neutral_smpl.frozen_social_actor_snapshot.v1"


def _record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_IMPORT_RECORD = _record(HERE / "inference_adapter.py")


def _helper():
    """Reuse the frozen stdlib source/type and exact RMS predicates only."""
    source = HELPER.read_bytes()
    if len(source) != HELPER_BYTES or hashlib.sha256(source).hexdigest() != HELPER_SHA:
        raise ValueError("Frozen social snapshot helper source changed")
    if _record(HERE / "inference_adapter.py") != _IMPORT_RECORD:
        raise ValueError("Social inference source changed after import")
    # Module definitions and its byte identity record execute, not its critic
    # constructor. This source imports neither Torch nor its trainer dependency.
    name = "_frozen_social_inference_snapshot_helper"
    module = sys.modules.get(name)
    if module is not None:
        if Path(getattr(module, "__file__", "")).resolve() != HELPER:
            raise ValueError("Foreign social snapshot helper already imported")
        return module
    module = ModuleType(name)
    module.__file__ = str(HELPER)
    sys.modules[name] = module
    try:
        exec(compile(source, str(HELPER), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class FrozenSocialAMPInference:
    """Copy an existing trained actor; arbitrary positive inference batch B.

    Caller must quiesce training while copying. Positive observed counters
    admit deployment, but cannot prove physical training quality or completion.
    No trainer/model constructor, checkpoint load, seed, device move or RNG
    restoration occurs. Every retained module/tensor has independent storage.
    """

    @classmethod
    def from_trainer(cls, trainer):
        instance = cls.__new__(cls)
        instance._capture(trainer)
        return instance

    def _capture(self, trainer):
        helper = _helper()
        sources = helper._sources()
        module = helper._trainer_module(trainer)
        torch = module.torch
        if torch.is_inference_mode_enabled():
            raise ValueError("Capture an ordinary trained object outside inference_mode")
        if type(trainer.config) is not module.SocialTrainingConfig:
            raise TypeError("The source-bound social configuration is required")
        trainer.config.validate()
        if trainer.config.normalize_observations is not True:
            raise ValueError("Social inference requires saved whole-input normalization")
        if (type(trainer.model) is not module.SocialAMPNetwork
                or type(trainer.model.task_encoder) is not module._policy.SocialTaskEncoder):
            raise TypeError("The source-bound shared social CNN actor is required")
        updates, transitions = trainer.update_count, trainer.physical_transition_count
        if (type(updates) is not int or type(transitions) is not int or min(updates, transitions) <= 0
                or transitions % trainer.config.actor_count):
            raise ValueError("A zero-count/untrained actor cannot be exposed for deployment")
        device = torch.device(trainer.device)
        if device.type == "cpu":
            device = torch.device("cpu")
        if device.type not in {"cpu", "cuda"} or device.type == "cuda" and device.index is None:
            raise ValueError("CPU or an explicitly indexed CUDA source device is required")
        self._torch, self._device = torch, device
        self._transitions = transitions
        self._normalizer_predicate = helper.SocialValueSnapshot._normalizer
        self._expected_mask = tuple(module._settings.PHYSICAL_ACTION_MASK)
        if len(self._expected_mask) != 69 or sum(self._expected_mask) != 57:
            raise ValueError("The source-bound canonical physical action mask is required")
        encoder, actor = trainer.model.task_encoder, trainer.model.actor
        self._architecture(encoder, actor)
        self._modules((encoder, actor), frozen=False)
        normalizers = (trainer.self_normalizer, trainer.task_normalizer)
        for normalizer, dimension in zip(normalizers, (357, 12308), strict=True):
            if type(normalizer) is not module._base.RunningNormalizer:
                raise TypeError("The original population RMS object is required")
            self._normalizer_predicate(self, normalizer, dimension, transitions)
        self._mask(trainer.action_mask)
        self._config = trainer.config.detached_copy()
        self._provenance = deepcopy(trainer.provenance)
        contract = deepcopy(trainer._checkpoint_contract())
        # The one tuple copy includes only existing encoder/actor, two RMS and
        # mask. In particular, no critic, D, AMP/value RMS, Adam or replay copy.
        self._encoder, self._actor, self._self_rms, self._task_rms, self._action_mask = deepcopy(
            (encoder, actor, *normalizers, trainer.action_mask))
        for child in (self._encoder, self._actor, self._self_rms, self._task_rms):
            child.eval()
            child.requires_grad_(False)
        self._validate_storage()
        self._metadata = {
            "schema": SCHEMA, "sources": {**sources, "helper": _record(HELPER),
                "actor_execution_adapter": dict(_IMPORT_RECORD)},
            "trainer_source_contract": contract, "configuration_document": self._config.to_document(),
            "update_count": updates, "physical_transition_count": transitions,
            "copied_components": ["task_encoder", "actor", "self_RMS357", "task_RMS12308", "action_mask69"],
            "critic_D_AMP_value_RMS_optimizer_replay_copied": False,
            "weights_and_moments": "owned tuple deepcopy of existing trained objects; no initialization",
            "training_actor_count": self._config.actor_count, "inference_batch": "arbitrary B>0; not training actor_count",
            "grid_geometry_contract": {key: getattr(self._config, key) for key in
                ("grid_center", "grid_heading", "height_reference")},
            "include_ego": self._config.include_ego,
            "quiescent_source_is_caller_responsibility": True,
            "checkpoint_loaded": False, "checkpoint_loader_implemented": False,
            "RNG_seed_draw_or_restore_performed": False, "device_move_performed": False,
            "training_completion_quality_or_physical_validation_claimed": False,
        }

    def _architecture(self, encoder, actor):
        torch = self._torch

        def linear_sequence(sequence, widths, final_activation):
            expected = []
            for index, (input_dim, output_dim) in enumerate(zip(widths, widths[1:])):
                expected.append((torch.nn.Linear, input_dim, output_dim))
                if final_activation or index < len(widths) - 2:
                    expected.append((torch.nn.ReLU,))
            if type(sequence) is not torch.nn.Sequential or len(sequence) != len(expected):
                raise ValueError("Social actor/projection structure changed")
            for child, specification in zip(sequence, expected, strict=True):
                if type(child) is not specification[0]:
                    raise ValueError("Social actor/projection layer type changed")
                if len(specification) == 3:
                    _, inputs, outputs = specification
                    if (child.in_features, child.out_features) != (inputs, outputs) or (
                            tuple(child.weight.shape) != (outputs, inputs) or child.bias is None
                            or tuple(child.bias.shape) != (outputs,)):
                        raise ValueError("Social actor/projection layer dimensions changed")
                elif child.inplace is not False:
                    raise ValueError("Social activation contract changed")

        linear_sequence(actor, (633, 2048, 1024, 69), False)
        linear_sequence(encoder.projection, (1024, 512, 256), True)
        if type(encoder.cnn) is not torch.nn.Sequential or len(encoder.cnn) != 6:
            raise ValueError("Social CNN structure changed")
        for index, inputs in enumerate((3, 16, 16)):
            layer, activation = encoder.cnn[2 * index], encoder.cnn[2 * index + 1]
            if (type(layer) is not torch.nn.Conv2d or type(activation) is not torch.nn.ReLU
                    or activation.inplace is not False or layer.in_channels != inputs or layer.out_channels != 16
                    or layer.kernel_size != (4, 4) or layer.stride != (2, 2) or layer.padding != (1, 1)
                    or layer.dilation != (1, 1) or layer.groups != 1 or layer.padding_mode != "zeros"
                    or tuple(layer.weight.shape) != (16, inputs, 4, 4) or layer.bias is None
                    or tuple(layer.bias.shape) != (16,)):
                raise ValueError("Social CNN layer contract changed")

    def _modules(self, children, *, frozen):
        torch = self._torch
        for child in children:
            if frozen and any(value.training for value in child.modules()):
                raise ValueError("Copied inference modules must remain in eval mode")
            for value in (*child.parameters(), *child.buffers()):
                if (value.dtype != torch.float32 or value.device != self._device or value.is_inference()
                        or not bool(torch.isfinite(value).all()) or frozen and value.requires_grad):
                    raise ValueError("Inference-path tensors must remain finite float32 on the source device")

    def _mask(self, value):
        torch = self._torch
        expected = torch.tensor(self._expected_mask, dtype=torch.float32, device=self._device)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (69,) or value.dtype != torch.float32
                or value.device != self._device or value.requires_grad or value.is_inference()
                or not torch.equal(value, expected)):
            raise ValueError("The copied canonical57-active/69 action mask changed")

    def _validate_storage(self):
        self._architecture(self._encoder, self._actor)
        self._modules((self._encoder, self._actor), frozen=True)
        for normalizer, dimension in ((self._self_rms, 357), (self._task_rms, 12308)):
            if normalizer.training:
                raise ValueError("Copied input RMS must remain in eval mode")
            self._normalizer_predicate(self, normalizer, dimension, self._transitions)
        self._mask(self._action_mask)

    @property
    def config(self):
        return self._config.detached_copy()

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    @property
    def device(self):
        return self._device

    def act(self, raw_self, raw_task, *, deterministic=True):
        """Raw357/12308 -> normalized deterministic69 mean and applied57 mask.

        Inputs are same-device float32 and are never moved, normalized in
        place, or retained. Policy means are not physical PD position targets.
        """
        torch = self._torch
        if deterministic is not True:
            raise ValueError("Frozen social inference supports deterministic=True only")
        self._validate_storage()
        if not isinstance(raw_self, torch.Tensor) or raw_self.ndim != 2 or len(raw_self) < 1:
            raise ValueError("Raw self requires nonempty float32[B,357]")
        batch = len(raw_self)
        for value, width in ((raw_self, 357), (raw_task, 12308)):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, width)
                    or value.dtype != torch.float32 or value.device != self._device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Raw input shape/dtype/device/finiteness differs from the social snapshot")
        with torch.inference_mode():
            normalized_self = self._self_rms.normalize(raw_self)
            normalized_task = self._task_rms.normalize(raw_task)
            features = torch.cat((normalized_self, self._encoder(normalized_task)), dim=-1)
            mean = self._actor(features)
            if tuple(mean.shape) != (batch, 69) or not bool(torch.isfinite(mean).all()):
                raise FloatingPointError("Frozen social actor produced invalid means")
            applied = mean.clamp(-1, 1) * self._action_mask
        return {"actions": mean, "mean_actions": mean, "applied_actions": applied}

    def describe(self):
        return deepcopy({**self._metadata, "status": "source_prepared_runtime_unverified",
            "observations": {"raw_self": 357, "raw_task": 12308, "task_order": "trajectory20,HWC64x64x3"},
            "action_contract": {"normalized_dimensions": 69, "active": 57, "PD_decoded": False},
            "normalization": "saved population RMS, float64 buffers, clip5/epsilon1e-5; no updates",
            "Normal_critic_D_forward_performed": False})
