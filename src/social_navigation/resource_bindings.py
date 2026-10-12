"""Explicit relocation of immutable resource identities, without rewriting receipts.

Only registered file names are translated. USD prim paths, motion IDs, hashes,
and numerical observations remain unchanged. No basename search or network I/O.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from protomotions.repository import repository_root


@lru_cache(maxsize=8)
def _bindings(root: str):
    path = Path(root) / "configs/reproduction/resource_bindings.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    if document.get("schema") != "social_navigation.resource_bindings.v1":
        raise ValueError("Unsupported resource binding schema")
    rows = document["bindings"]
    if len({row["source"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source identities in resource bindings")
    result = {}
    for row in rows:
        relative = Path(row["target"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Resource binding target must stay inside the checkout")
        result[row["source"]] = row
    return result


def resolve_path(value, *, root=None, relative_to=None):
    root = Path(root or repository_root()).resolve()
    from .sdk_paths import relocate_sdk_path
    rows = _bindings(str(root))
    row = rows.get(str(value))
    if row is None:
        normalized = relocate_sdk_path(value, root=root)
        path = Path(normalized)
        if path.is_absolute() and path.is_relative_to(root):
            row = rows.get('project:' + path.relative_to(root).as_posix())
    if row is not None:
        # External licensed datasets may be mounted anywhere. This explicit
        # registry changes location only; the caller still checks bytes/SHA.
        overrides = root / "data/external_resources.json"
        registry = json.loads(overrides.read_text()) if overrides.is_file() else {}
        selected = registry.get(row.get("resource_id"), row["target"])
        path = Path(selected).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    path = Path(relocate_sdk_path(value, root=root)).expanduser()
    return (path if path.is_absolute() else Path(relative_to or root) / path).resolve()


def relocate_document(value, *, root=None):
    """Return a location view of raw JSON; never modify its serialized bytes."""
    root = Path(root or repository_root()).resolve()
    rows = _bindings(str(root))
    if isinstance(value, str):
        if value in rows or value.startswith(("project:", "lab:", "sim:")):
            return str(resolve_path(value, root=root))
        if Path(value).is_absolute():
            from .sdk_paths import relocate_sdk_path
            normalized = Path(relocate_sdk_path(value, root=root))
            if normalized.is_absolute() and normalized.is_relative_to(root):
                key = 'project:' + normalized.relative_to(root).as_posix()
                if key in rows:
                    return str(resolve_path(value, root=root))
        return value
    if isinstance(value, list):
        return [relocate_document(item, root=root) for item in value]
    if isinstance(value, dict):
        return {relocate_document(key, root=root): relocate_document(item, root=root)
                for key, item in value.items()}
    return value


def read_json(path):
    path = resolve_path(path)
    return relocate_document(json.loads(path.read_text()))


def scene_locations(scene):
    """Top-level config paths are checkout relative, including frozen copies."""
    scene = dict(scene)
    for key in ("preflight", "native_map", "probe_script_snapshot"):
        if scene.get(key) is not None:
            scene[key] = dict(scene[key], path=str(resolve_path(scene[key]["path"])))
    return scene
