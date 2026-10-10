# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic checkpoint publication and the backwards-compatible agent/env bundle.

New agent checkpoints retain their existing top-level model/optimizer keys and
embed all environment tasks in one serialized generation. Legacy files without
this metadata can still use their original environment sidecars.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping
import uuid


CHECKPOINT_BUNDLE_KEY = "_protomotions_checkpoint"
CHECKPOINT_BUNDLE_VERSION = 1


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(
    path: str | Path,
    writer: Callable,
    *,
    preserve_previous: bool = False,
) -> None:
    """Write through a same-directory temporary file, fsync, then atomically replace.

    ``writer`` receives a binary file handle. If it fails, the current checkpoint
    remains unchanged. With ``preserve_previous``, the prior complete file is
    published at ``<filename>.previous`` before the replacement. Atomic rename
    and directory fsync assume the output filesystem provides POSIX semantics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    previous_temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())

        if preserve_previous and path.exists():
            previous_temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.previous.tmp"
            try:
                # Old checkpoint bytes are immutable once published, so a hard
                # link avoids copying large model/optimizer archives.
                os.link(path.resolve(), previous_temporary)
            except OSError:
                shutil.copyfile(path, previous_temporary)
            with previous_temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(previous_temporary, path.with_name(path.name + ".previous"))
            previous_temporary = None
            _sync_directory(path.parent)

        os.replace(temporary, path)
        temporary = None
        _sync_directory(path.parent)
    finally:
        for leftover in (temporary, previous_temporary):
            if leftover is not None:
                leftover.unlink(missing_ok=True)


def atomic_torch_save(state: Any, path: str | Path, *, preserve_previous: bool = False) -> None:
    """Atomically publish a Torch checkpoint; imports Torch only when used."""
    import torch

    atomic_write(path, lambda handle: torch.save(state, handle), preserve_previous=preserve_previous)


def atomic_write_text(text: str, path: str | Path) -> None:
    atomic_write(path, lambda handle: handle.write(text.encode("utf-8")))


def make_checkpoint_bundle(environment_states: Mapping[str, dict]) -> dict:
    """Create generation metadata shared by training and inference artifacts."""
    return {
        "version": CHECKPOINT_BUNDLE_VERSION,
        "generation_id": uuid.uuid4().hex,
        "environment_states": dict(environment_states),
    }


def get_checkpoint_bundle(state: Mapping) -> dict | None:
    """Validate new metadata, returning None only for a genuine legacy file."""
    if CHECKPOINT_BUNDLE_KEY not in state:
        return None
    bundle = state[CHECKPOINT_BUNDLE_KEY]
    if not isinstance(bundle, dict) or bundle.get("version") != CHECKPOINT_BUNDLE_VERSION:
        raise ValueError("Unsupported or invalid ProtoMotions checkpoint bundle version")
    if not isinstance(bundle.get("generation_id"), str) or not bundle["generation_id"]:
        raise ValueError("Checkpoint bundle has no generation ID")
    environments = bundle.get("environment_states")
    if not isinstance(environments, dict) or any(
        not isinstance(task, str) or not isinstance(env_state, dict)
        for task, env_state in environments.items()
    ):
        raise ValueError("Invalid checkpoint bundle environment states")
    return bundle


def snapshot_environment_state(state: dict) -> dict:
    """Freeze environment state on CPU before distributed object collection.

    Object collectives pickle tensors with their device identifiers. Shipping
    CPU copies avoids restoring a remote rank's CUDA tensors on the wrong GPU.
    """
    import torch

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [snapshot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(snapshot(item) for item in value)
        return copy.deepcopy(value)

    if not isinstance(state, dict):
        raise TypeError("Environment checkpoint state must be a dictionary")
    return snapshot(state)
