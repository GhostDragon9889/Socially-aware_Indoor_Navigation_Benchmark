"""Remove rebuildable checkout caches without following resource links."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat

from protomotions.repository import repository_root

CACHE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
}


def _root(path: Path) -> Path:
    root = path.expanduser().resolve()
    marker = root / ".social-navigation-root"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Cache cleanup requires a marked project checkout")
    return root


def _identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("Cache must be a regular directory: " + str(path))
    return info.st_dev, info.st_ino


def _size(path: Path) -> int:
    total = 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            item = Path(directory) / name
            if not item.is_symlink():
                total += item.stat().st_size
    return total


def _active_addons(root: Path) -> Path | None:
    binding = root / ".runtime/external_sdk.json"
    if not binding.is_file() or binding.is_symlink():
        return None
    value = json.loads(binding.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != "social_navigation.external_sdk.v1"
    ):
        return None
    addons = value.get("addons")
    if not isinstance(addons, str):
        return None
    path = Path(addons).resolve()
    expected = root / ".runtime/external_sdk_packages"
    return (
        path
        if path == expected and expected.is_dir() and not expected.is_symlink()
        else None
    )


def plan(root: Path | None = None) -> dict:
    """Select named generated caches; a directory named 'cache' is not enough."""
    root = _root(Path(root or repository_root()))
    active = _active_addons(root)
    entries = []
    for directory, dirs, _ in os.walk(root, followlinks=False):
        parent = Path(directory)
        retained = []
        for name in dirs:
            path = parent / name
            if name == ".git" or path.is_symlink():
                continue
            category = None
            if name in CACHE_NAMES:
                category = "python_or_test_cache"
            elif parent == root / ".cache" and name in {"uv", "pip", "pytest", "ruff"}:
                category = "package_or_test_download_cache"
            elif (
                name == "cache"
                and parent.name == "kit"
                and "site-packages/isaacsim/kit" in parent.as_posix()
                and path.is_relative_to(root / ".runtime")
            ):
                category = "kit_derived_cache"
            elif (
                active is not None
                and parent == root / ".runtime"
                and name.startswith(".external_sdk_packages_backup-")
            ):
                category = "unused_addons_backup"
            if category:
                device, inode = _identity(path)
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "category": category,
                        "logical_bytes": _size(path),
                        "device": device,
                        "inode": inode,
                    }
                )
            else:
                retained.append(name)
        dirs[:] = retained
    # Git LFS temporary download/index caches are separate from its object store.
    for name in ("cache", "tmp"):
        path = root / ".git/lfs" / name
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.resolve().is_relative_to(root)
        ):
            device, inode = _identity(path)
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "category": "lfs_temporary_cache",
                    "logical_bytes": _size(path),
                    "device": device,
                    "inode": inode,
                }
            )
    return {
        "schema": "social_navigation.cache_cleanup.v1",
        "root": str(root),
        "entries": entries,
        "logical_bytes": sum(row["logical_bytes"] for row in entries),
        "applied": False,
    }


def apply(inventory: dict) -> dict:
    """Recheck directory identities before deleting exactly the selected caches."""
    root = _root(Path(inventory["root"]))
    approved = {row["path"]: row for row in plan(root)["entries"]}
    for row in inventory["entries"]:
        relative = Path(row["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or row["path"] not in approved
        ):
            raise ValueError("Unregistered cache deletion: " + str(relative))
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Cache path escaped the checkout")
        if _identity(path) != (row["device"], row["inode"]):
            raise ValueError(
                "Cache directory changed after selection: " + str(relative)
            )
    removed = []
    for row in inventory["entries"]:
        path = root / row["path"]
        if _identity(path) != (row["device"], row["inode"]):
            raise ValueError("Cache directory changed before removal: " + row["path"])
        shutil.rmtree(path)
        removed.append(row["path"])
    return {**inventory, "applied": True, "removed": removed}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument(
        "--apply", action="store_true", help="Delete the listed rebuildable caches"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    inventory = plan(args.root)
    result = apply(inventory) if args.apply else inventory
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0
