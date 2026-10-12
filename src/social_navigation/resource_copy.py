"""Verify complete resource bytes before publishing a new destination."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _existing(target: Path, expected: str) -> bool:
    if target.is_symlink():
        raise ValueError(f"Existing resource link; retained: {target}")
    if not target.exists():
        return False
    if not target.is_file() or sha256(target) != expected:
        raise ValueError(f"Existing resource differs; retained: {target}")
    return True


def copy_exact(source: Path, target: Path, expected: str) -> bool:
    """Publish verified bytes without replacing another writer's destination."""
    if _existing(target, expected):
        return False
    if sha256(source) != expected:
        raise ValueError(f"Resource source differs: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="." + target.name + "-", suffix=".partial", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Copy checksum failed: {target}")
        shutil.copymode(source, temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not _existing(target, expected):
                raise RuntimeError(f"Resource disappeared during publication: {target}")
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)
