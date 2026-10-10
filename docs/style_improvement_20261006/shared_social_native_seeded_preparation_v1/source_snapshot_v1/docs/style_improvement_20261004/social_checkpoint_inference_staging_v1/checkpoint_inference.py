"""Strict social checkpoint -> actor-only inference; no SDK/resume restoration.

Import is stdlib plus the pinned stdlib actor snapshot definitions. Only an
explicit future from_checkpoint call imports Torch and constructs a temporary
full network to validate tensors. No load/construction has been executed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PINS = {
    "actor": ("docs/style_improvement_20261004/social_deployment_integration_v1/inference_adapter.py", 13766,
              "05261849db05c7c473c4a77ca9fd1e4c6930562a65bd46810849380484fbfc58"),
    "helper": ("docs/style_improvement_20261004/social_value_adapter_staging_v1/value_adapter.py", 12573,
               "9875c58b90e23ab6613ee7298bb4a199295435ad7f540f5b760e524a8f3b4f76"),
    "trainer": ("docs/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py", 24078,
                "81a6319f2520b3be547f30454c88b088638d559d492bba52fe850d1c1d5ba96e"),
}


def _source_record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_IMPORT_RECORD = _source_record(HERE / "checkpoint_inference.py")


def _sources():
    records = {}
    for name, (relative, length, sha) in PINS.items():
        path = ROOT / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
            raise ValueError("An independent readonly inference dependency is required: " + name)
        row = _source_record(path)
        if row["bytes"] != length or row["sha256"] != sha:
            raise ValueError("Social checkpoint reader dependency changed: " + name)
        records[name] = row
    if _source_record(HERE / "checkpoint_inference.py") != _IMPORT_RECORD:
        raise ValueError("Checkpoint inference source changed after import")
    records["checkpoint_execution_adapter"] = dict(_IMPORT_RECORD)
    return records


def _owned_module(name, key):
    _sources()  # Byte identity precedes executing any dependency definitions.
    path = (ROOT / PINS[key][0]).resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path:
            raise ValueError("Foreign social inference dependency already imported")
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


# This parent contains no Torch import or NN construction. The trainer remains
# lazy until the caller explicitly requests checkpoint inference admission.
_actor = _owned_module("_social_checkpoint_actor_snapshot", "actor")


def _file_stat(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_nlink)


def _checkpoint_record(path, expected_sha, descriptor):
    """Hash one readonly local inode and verify pathname/descriptor stability."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222:
        raise ValueError("Checkpoint must be an independent readonly regular file")
    # Keep the original descriptor open through the full load/copy. This
    # duplicate stream closes independently; always rewind the shared offset.
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _file_stat(opened) != _file_stat(before):
            raise RuntimeError("Checkpoint changed before hash admission")
        stream.seek(0)
        prefix = stream.read(80)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1") or not prefix.startswith(b"PK\x03\x04"):
            raise ValueError("A materialized Torch ZIP/mmap checkpoint is required")
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        if _file_stat(os.fstat(stream.fileno())) != _file_stat(before):
            raise RuntimeError("Checkpoint inode changed during hashing")
    if _file_stat(path.lstat()) != _file_stat(before):
        raise RuntimeError("Checkpoint pathname changed during hashing")
    actual = digest.hexdigest()
    if actual != expected_sha:
        raise ValueError("Checkpoint bytes differ from the required selected SHA256")
    return {"path": str(path), "bytes": before.st_size, "sha256": actual,
            "stat_identity": list(_file_stat(before))}


def _json_copy(value, label):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("JSON-only inference metadata is required: " + label) from error


def _cpu_tensor(module, value, expected, label):
    # Original whole-model shape/dtype/finiteness predicate plus this loader's
    # explicit materialized CPU/strided contract. No partial tensor migration.
    module.SocialAMPTrainer._tensor(value, expected, label)
    if value.device.type != "cpu" or value.layout != module.torch.strided or value.is_inference():
        raise ValueError("Checkpoint tensors require ordinary materialized CPU strided storage: " + label)


class FrozenSocialCheckpointInference(_actor.FrozenSocialAMPInference):
    """Reuse exact frozen actor normalization/forward/action/storage methods.

    Inference admission is narrower than a trainer resume: optimizer, replay,
    saved RNG and opaque runner state are never validated/restored/retained.
    Counter values are checkpoint declarations, not an independent observation
    that physical training or the intended terminal run actually completed.
    """

    @classmethod
    def from_checkpoint(cls, checkpoint, *, expected_sha256, device="cpu"):
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("A complete immutable selected checkpoint SHA256 is mandatory")
        expected_sha256 = expected_sha256.lower()
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip() or "://" in str(checkpoint):
            raise ValueError("An explicit local checkpoint path is required")
        requested = Path(checkpoint).expanduser().absolute()
        if requested.is_symlink():
            raise ValueError("A checkpoint symlink cannot stand in for the selected inode")
        path = requested.resolve(strict=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            return cls._from_open_checkpoint(path, expected_sha256, device, descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _from_open_checkpoint(cls, path, expected_sha256, device, descriptor):
        before = _checkpoint_record(path, expected_sha256, descriptor)
        sources = _sources()
        helper = _actor._helper()
        helper_sources = helper._sources()
        module = _owned_module("_social_checkpoint_source_trainer", "trainer")
        torch = module.torch
        if torch.is_inference_mode_enabled() or torch.get_default_dtype() != torch.float32:
            raise ValueError("Checkpoint admission requires ordinary mode and float32 default dtype")
        target = torch.device(device)
        if target.type == "cpu":
            target = torch.device("cpu")
        if target.type not in {"cpu", "cuda"} or target.type == "cuda" and target.index is None:
            raise ValueError("Inference target must be CPU or explicitly indexed CUDA")
        # Linux proc-fd pathname refers to the retained exact admitted inode,
        # even if a parent pathname is concurrently atomically replaced.
        artifact = torch.load(f"/proc/self/fd/{descriptor}", map_location="cpu", weights_only=True, mmap=True)
        config = module.SocialAMPTrainer._admit_metadata(artifact)  # Before the first NN constructor.
        updates, transitions = artifact["update_count"], artifact["physical_transition_count"]
        if (type(updates) is not int or type(transitions) is not int or min(updates, transitions) <= 0
                or transitions % config.actor_count):
            raise ValueError("Deployment requires positive declared social update/transition counts")
        provenance = _json_copy(artifact["provenance"], "training provenance")
        initialization = {"strategy": "fresh_local_social_CNN_and_state_action_D", "seed": config.seed,
                          "weights_or_moments_transferred": False, "author_navisaaclab_weights": False}
        if not module._equal_json(provenance.get("initialization"), initialization):
            raise ValueError("Checkpoint lacks the original fresh social initialization declaration")
        contract = _json_copy(artifact["social_training_contract"], "social source/config contract")
        amp_dim = 195 * (10 if config.amp_normalization == "per_history_coordinate" else 1)
        dimensions = {"self_normalizer": 357, "task_normalizer": 12308,
                      "amp_normalizer": amp_dim, "value_normalizer": 1}
        # The exact network has random initializers. CPU placement and fork_rng
        # preserve caller CPU RNG on construction/validation success or failure.
        # No trainer, Adam/replay or checkpoint RNG restoration is invoked.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            network = module.SocialAMPNetwork(config)
            normalizers = {name: module._base.RunningNormalizer(width, config.normalization_clip)
                           for name, width in dimensions.items()}
            expected_model = network.state_dict()
            saved_model = artifact["model"]
            if not isinstance(saved_model, dict) or set(saved_model) != set(expected_model):
                raise ValueError("Complete new social model keys are required; legacy/partial loading is forbidden")
            for name, expected in expected_model.items():
                _cpu_tensor(module, saved_model[name], expected, "model." + name)
            if not torch.equal(saved_model["log_std"], torch.full_like(expected_model["log_std"], -2.9)):
                raise ValueError("The fixed social log_std=-2.9 buffer changed")
            saved_rms = artifact["normalizers"]
            if type(saved_rms) is not dict or set(saved_rms) != set(dimensions):
                raise ValueError("Exactly all four original social RMS states are required")
            rms_counts = {}
            for name, normalizer in normalizers.items():
                expected = normalizer.state_dict()
                actual = saved_rms[name]
                if not isinstance(actual, dict) or set(actual) != set(expected) or normalizer.clip != 5.:
                    raise ValueError("Complete original clip5 RMS schema is required: " + name)
                for field, template in expected.items():
                    _cpu_tensor(module, actual[field], template, name + "." + field)
                count = float(actual["count"])
                if bool((actual["variance"] < 0).any()) or count < 0:
                    raise ValueError("RMS variance/count must be nonnegative: " + name)
                if name != "amp_normalizer" and count != transitions:
                    raise ValueError("Fresh social self/task/value RMS counts must equal declared physical transitions")
                rms_counts[name] = count
            network.load_state_dict(saved_model, strict=True)  # Copies every model tensor, not assign=True mmap aliases.
            for name, normalizer in normalizers.items():
                normalizer.load_state_dict(saved_rms[name], strict=True)
        instance = cls.__new__(cls)  # An inference owner, never a fabricated trainer.
        instance._torch, instance._device, instance._transitions = torch, torch.device("cpu"), transitions
        instance._normalizer_predicate = helper.SocialValueSnapshot._normalizer
        instance._expected_mask = tuple(module._settings.PHYSICAL_ACTION_MASK)
        instance._config, instance._provenance = config.detached_copy(), deepcopy(provenance)
        instance._encoder, instance._actor = network.task_encoder, network.actor
        instance._self_rms, instance._task_rms = normalizers["self_normalizer"], normalizers["task_normalizer"]
        instance._action_mask = torch.tensor(instance._expected_mask, dtype=torch.float32, device="cpu")
        for child in (instance._encoder, instance._actor, instance._self_rms, instance._task_rms):
            child.eval()
            child.requires_grad_(False)
        instance._validate_storage()  # Frozen parent checks exact architecture, F32/F64, counts and canonical57 mask.
        del expected_model, saved_model, saved_rms, network, normalizers, artifact
        if target != instance._device:
            for child in (instance._encoder, instance._actor, instance._self_rms, instance._task_rms):
                child.to(target)
            instance._action_mask = instance._action_mask.to(target)
            instance._device = target
        instance._validate_storage()
        after = _checkpoint_record(path, expected_sha256, descriptor)  # Full bytes/path/retained inode before return.
        if after != before or _sources() != sources or helper._sources() != helper_sources:
            raise RuntimeError("Selected checkpoint or reviewed inference sources changed during admission")
        module._verify_sources()
        instance._metadata = {
            "schema": "local.neutral_smpl.frozen_social_checkpoint_inference.v1",
            "sources": {**sources, "parent_snapshot_sources": helper_sources},
            "trainer_source_contract": contract, "configuration_document": instance._config.to_document(),
            "checkpoint": before, "checkpoint_identity_after_load": after,
            "update_count": updates, "physical_transition_count": transitions,
            "counter_source": "checkpoint declaration; no independent physical observation",
            "RMS_counts": rms_counts, "RMS_count_source": "validated saved float64 buffers",
            "retained_components": ["task_encoder", "actor", "self_RMS357", "task_RMS12308", "action_mask69"],
            "temporarily_constructed_components": ["full_social_actor_critic_encoder_D_log_std", "four_RMS"],
            "whole_model_and_four_RMS_validated": True,
            "critic_D_AMP_value_RMS_optimizer_replay_retained": False,
            "optimizer_replay_saved_RNG_runner_state_validated_or_restored": False,
            "temporary_full_NN_construction_performed": True,
            "initializer_CPU_RNG_draws_performed": True, "caller_CPU_RNG_preserved": True,
            "caller_CPU_RNG_preserved_by_fork_rng": True, "restore_training_rng": False,
            "checkpoint_training_RNG_restored": False,
            "target_device": str(target), "device_move_performed": target.type != "cpu",
            "retained_path_device_move_performed": target.type != "cpu",
            "owned_storage": "strict assign=False tensor copies; no mmap aliases/unused artifact retained",
            "load_inode_bound_to_retained_descriptor": True,
            "training_actor_count": config.actor_count, "inference_batch": "arbitrary B>0; not training actor_count",
            "grid_geometry_contract": {key: getattr(config, key) for key in
                ("grid_center", "grid_heading", "height_reference")}, "include_ego": config.include_ego,
            "checkpoint_loaded": True, "checkpoint_loader_implemented": True,
            "inference_forward_or_SDK_validation_performed_by_loader": False,
            "training_completion_quality_or_physical_validation_claimed": False,
        }
        return instance
