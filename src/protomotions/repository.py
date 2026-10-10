"""Locate a source checkout without assuming package depth or an old folder name."""
from pathlib import Path
import os


def repository_root(anchor=None):
    """Return the canonical checkout root, including when accessed through a link.

    This helper is for checkout tools that need configs, evidence and assets.
    Installed-package assets continue to use ``protomotions.assets``.
    """
    override = os.environ.get('SOCIAL_NAVIGATION_ROOT')
    start = Path(override) if override else Path(anchor or __file__)
    start = start.expanduser().resolve()
    candidates = [start, *start.parents]
    for path in candidates:
        if (path / 'pyproject.toml').is_file() and (path / 'configs').is_dir() and (
            (path / 'src/protomotions').is_dir() or (path / 'protomotions').is_dir()
        ):
            return path
    raise FileNotFoundError('Source checkout not found. Set SOCIAL_NAVIGATION_ROOT to its root.')
