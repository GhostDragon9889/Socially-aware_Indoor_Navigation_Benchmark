"""Stdlib-only preflight for the current SMPL AMP reconstruction interface.

File headers establish a local ZIP payload, not checkpoint tensor compatibility.
The explicit Torch loader performs the latter check after this preflight.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import math
from pathlib import Path
import re

from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.application.state_prior_initialization import validate_action_state_prior_initialization


CHECKPOINT_FORMAT = "navisaaclab_method_reconstruction_training"
CHECKPOINT_VERSION = 1
FROZEN_ACTION_INDICES = (9, 10, 11, 21, 22, 23, 51, 52, 53, 66, 67, 68)
CANONICAL_ACTION_MASK = tuple(int(index not in FROZEN_ACTION_INDICES) for index in range(69))


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    bytes: int
    mtime_ns: int
    inode: int

    def unchanged(self):
        stat = self.path.stat()
        return (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (self.bytes, self.mtime_ns, self.inode)


def validate_expected_sha256(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)):
        raise ValueError("expected_sha256 must be None or a complete SHA256 hex digest")
    return None if value is None else value.lower()


def inspect_checkpoint_file(path_like):
    """Retain cwd path semantics and reject missing/LFS/non-ZIP local files."""
    if not isinstance(path_like, (str, Path)) or not str(path_like).strip() or "://" in str(path_like):
        raise ValueError("AMP inference requires an explicit local checkpoint path")
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AMP inference checkpoint is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(80)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("AMP inference checkpoint is a Git LFS pointer")
    if not header.startswith(b"PK\x03\x04"):
        raise ValueError("AMP inference requires a Torch ZIP checkpoint compatible with CPU mmap")
    stat = path.stat()
    return CheckpointFile(path, stat.st_size, stat.st_mtime_ns, stat.st_ino)


def validate_reconstruction_config(mapping):
    """Return an owned config for this 357/1044,69-action,30Hz interface.

    New optional TrainingConfig fields retain their original defaults in old
    checkpoints. Unknown fields and alternate physical/observation contracts
    cannot silently select another controller.
    """
    if type(mapping) is not dict:
        raise ValueError("AMP inference config must be a TrainingConfig mapping")
    if any(type(key) is not str for key in mapping):
        raise ValueError("AMP inference TrainingConfig field names must be strings")
    unknown = set(mapping) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown AMP inference TrainingConfig fields: {sorted(unknown)}")
    try:
        config = TrainingConfig(**deepcopy(mapping))
        config.validate()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid AMP inference TrainingConfig: {error}") from error
    if (config.self_dim, config.task_dim, config.amp_frame_dim, config.history_frames) != (357, 1044, 195, 10):
        raise ValueError("AMP inference requires the canonical357/1044/195/history10 SMPL contract")
    if (tuple(config.task_layers), tuple(config.policy_layers), tuple(config.discriminator_layers)) != (
            (512, 256), (2048, 1024), (1024, 512)):
        raise ValueError("AMP inference requires the current complete PACER-shaped reconstruction architecture")
    if (not math.isclose(config.control_dt, 1 / 30, rel_tol=0, abs_tol=1e-12)
            or config.target_horizon_s != 5. or config.normalization_clip != 5.):
        raise ValueError("AMP inference requires30Hz control,5s targets and the declared clip5 moments")
    if config.freeze_toes is not True:
        raise ValueError("AMP inference requires the current frozen hand/toe physical decoder")
    if config.action_mask is not None and (
            not isinstance(config.action_mask, (list, tuple))
            or any(type(value) is not int for value in config.action_mask)
            or tuple(config.action_mask) != CANONICAL_ACTION_MASK):
        raise ValueError("AMP inference requires the exact canonical57-active/69 action mask")
    config.action_mask = CANONICAL_ACTION_MASK
    return config


def validate_bundle_identity(bundle):
    if type(bundle) is not dict or bundle.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a reconstruction training bundle, not public PACER or MaskedMimic weights")
    if type(bundle.get("version")) is not int or bundle["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported reconstruction checkpoint version")
    for name in ("update_count", "physical_transition_count"):
        if type(bundle.get(name)) is not int or bundle[name] < 0:
            raise ValueError(f"Checkpoint {name} must be a nonnegative integer")
    if type(bundle.get("provenance")) is not dict:
        raise ValueError("Reconstruction checkpoint requires explicit training provenance")
    config = validate_reconstruction_config(bundle.get("config"))
    validate_action_state_prior_initialization(config, bundle["provenance"].get("initialization"))
    # Header/config admission stays stdlib-only. The tensor loader and complete
    # trainer resume independently use the shared saved-statistics validator.
    policy = bundle.get("task_terrain_statistics_policy")
    if config.task_terrain_statistics_mode == "legacy":
        if policy is not None:
            raise ValueError("Legacy checkpoint cannot contain a protected terrain-statistics policy")
    elif type(policy) is not dict:
        raise ValueError("Protected terrain statistics require complete saved policy state")
    return config
