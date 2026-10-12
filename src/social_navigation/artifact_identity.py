"""Compare recorded artifacts across the documented checkout rename.

Portable project references are resolved against the checkout. SDK paths, hashes, sizes and all
other provenance remain unchanged; callers must still verify files on disk.
"""
from pathlib import Path

from protomotions.repository import repository_root

LEGACY_PROJECT = Path('project:')
SOURCE_PACKAGES = frozenset(('CrowdSim', 'protomotions', 'crowdsim_ppo', 'social_navigation', 'deployment'))


def artifact_path(value, *, root=None):
    """Return the local spelling of an absolute, traversal-free artifact path."""
    if not isinstance(value, (str, Path)):
        raise ValueError('Artifact path must be an absolute path')
    local_root = Path(root or repository_root()).resolve()
    text = str(value)
    if text.startswith('project:'):
        relative = Path(text[len('project:'):].lstrip('/'))
        if '..' in relative.parts:
            raise ValueError('Artifact reference contains parent traversal')
        path = local_root / relative
    elif text.startswith(('lab:', 'sim:')):
        from .sdk_paths import relocate_sdk_path
        path = Path(relocate_sdk_path(text, root=local_root))
    else:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Artifact path must be absolute or use project: and contain no parent traversal')
    if path.is_relative_to(local_root):
        relative = path.relative_to(local_root)
        if relative.parts and relative.parts[0] in SOURCE_PACKAGES:
            path = local_root / 'src' / relative
    return path


def same_artifact(actual, expected, *, root=None):
    """Match the complete identity, allowing only the registered root rename.

    This is a comparison, not an admission or a rewrite of archived evidence.
    In particular, a matching stored SHA must still be checked against bytes.
    """
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    for item in (actual, expected):
        if (not {'path', 'bytes', 'sha256'} <= item.keys()
                or type(item['bytes']) is not int or item['bytes'] < 0
                or not isinstance(item['sha256'], str) or len(item['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in item['sha256'])):
            return False
    try:
        left = {**actual, 'path': str(artifact_path(actual['path'], root=root))}
        right = {**expected, 'path': str(artifact_path(expected['path'], root=root))}
    except ValueError:
        return False
    return left.keys() == right.keys() and all(
        same_recorded_value(left[key], right[key], root=root) for key in left)


def same_recorded_value(actual, expected, *, root=None):
    """Compare nested provenance, relocating only complete artifact records."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        if {'path', 'bytes', 'sha256'} <= actual.keys() or {'path', 'bytes', 'sha256'} <= expected.keys():
            return same_artifact(actual, expected, root=root)
        return actual.keys() == expected.keys() and all(
            same_recorded_value(actual[key], expected[key], root=root) for key in actual)
    if isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            same_recorded_value(a, b, root=root) for a, b in zip(actual, expected))
    return actual == expected
