"""Relocate sealed SDK guards while preserving hashes, methods and ownership."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

from social_navigation.sdk_paths import relocate_sdk_path
from protomotions.repository import repository_root


def derive_relocated_checkout(source, legacy_root, canonical_root):
    if Path(legacy_root).resolve() != repository_root(__file__):
        raise ValueError("Source path must resolve to this exact checkout")
    canonical_root = Path(canonical_root).resolve()
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = relocate_sdk_path(node.value, root=canonical_root)
            if value != node.value:
                before = ast.get_source_segment(source, node)
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                replacements.append((start, end, before, json.dumps(value)))
    derived = source
    for start, end, before, after in sorted(replacements, reverse=True):
        if derived[start:end] != before:
            raise ValueError('SDK path token identity differs')
        derived = derived[:start] + after + derived[end:]
    comparisons = (
        ('ROOT = Path(__file__).resolve().parents[3]', 'ROOT = __repository_root__'),
        ('manifest["extension_path"] != str(OVERLAY)',
         'Path(__relocate_sdk_path__(manifest["extension_path"])).resolve() != OVERLAY.resolve()'),
        ('{key: expected[key] for key in ("path", "bytes", "sha256")}',
         '{key: (__relocate_sdk_path__(expected[key]) if key == "path" else expected[key]) for key in ("path", "bytes", "sha256")}'),
        ('list(dev_paths or ()) != [str(OVERLAY)]',
         '[Path(path).resolve() for path in (dev_paths or ())] != [OVERLAY.resolve()]'),
        ('mapped_paths != {str(plugin)}',
         '{str(Path(path).resolve()) for path in mapped_paths} != {str(plugin.resolve())}'),
    )
    for before, after in comparisons:
        if derived.count(before) != 1:
            raise ValueError('Ambiguous sealed SDK path comparison')
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(comparisons):
        inverse = inverse.replace(after, before, 1)
    for start, end, before, after in sorted(replacements):
        if inverse[start:start + len(after)] != after:
            raise ValueError('SDK path inverse differs')
        inverse = inverse[:start] + before + inverse[start + len(after):]
    if inverse != source:
        raise RuntimeError('SDK location derivation is not reversible')
    ast.parse(derived)
    return derived, {'canonical_root': str(canonical_root),
        'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'derived_sha256': hashlib.sha256(derived.encode()).hexdigest(),
        'absolute_source_path_changes': len(replacements), 'canonical_path_comparisons': len(comparisons),
        'exact_inverse': True, 'SDK_source_changes': False,
        'sealed_source_changes': False, 'shutdown_operations_changed': False}


def overlay_kit_arguments(legacy_root, overlay):
    """Provide a whitespace-free owned alias for AppLauncher's str.split parser."""
    overlay = Path(overlay).resolve()
    cache = Path(tempfile.gettempdir()) / f'social-navigation-kit-{os.getuid()}'
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        raise ValueError('Kit alias directory must be private and owned by this user')
    slot = cache / hashlib.sha256(str(overlay).encode()).hexdigest()
    slot.mkdir(mode=0o700, exist_ok=True)
    if slot.is_symlink() or slot.stat().st_uid != os.getuid():
        raise ValueError('Foreign Kit alias owner')
    alias = slot / 'isaacsim.core.simulation_manager'
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(overlay, target_is_directory=True)
    if not alias.is_symlink() or alias.resolve() != overlay or any(c.isspace() for c in str(alias)):
        raise ValueError('Kit alias must resolve to this exact overlay without whitespace')
    return '--/app/exts/devPaths=' + json.dumps([str(alias)], separators=(',', ':'))
