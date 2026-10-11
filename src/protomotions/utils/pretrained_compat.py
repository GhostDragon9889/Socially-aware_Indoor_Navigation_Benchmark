"""Reject explicitly archived branch-specific configs before pickle loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protomotions.repository import repository_root


def load_upstream_reference_registry(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or repository_root()).resolve()
    path = root / "data/pretrained_models/upstream_reference_registry.json"
    if not path.is_file():
        return {"schema": "social_navigation.upstream_reference_registry.v1", "entries": []}
    value = json.loads(path.read_text())
    if value.get("schema") != "social_navigation.upstream_reference_registry.v1":
        raise RuntimeError("Unsupported upstream reference registry schema")
    for row in value.get("entries", []):
        directory = Path(row["directory"])
        if directory.is_absolute() or ".." in directory.parts:
            raise RuntimeError("Upstream reference directory must be project-relative")
        if row["status"] != "unsupported_in_current_runtime":
            raise RuntimeError("Unexpected upstream reference runtime status")
    return value


def archived_upstream_reference(path: str | Path, root: Path | None = None) -> dict[str, Any] | None:
    root = Path(root or repository_root()).resolve()
    candidate = Path(path).expanduser().resolve()
    for row in load_upstream_reference_registry(root)["entries"]:
        directory = root / row["directory"]
        if candidate.is_relative_to(directory.resolve()):
            return row
    return None


def require_supported_checkpoint_config(path: str | Path, root: Path | None = None) -> None:
    """Refuse archived assets; no aliases, tensor changes or sidecar rewriting."""
    row = archived_upstream_reference(path, root)
    if row is not None:
        raise RuntimeError(
            f"Archived upstream reference cannot run with the current project: {path}. "
            f"{row['reason']} Use its original project/SDK revision, or explicitly "
            "migrate and independently validate a new model release. The archived files are preserved."
        )
