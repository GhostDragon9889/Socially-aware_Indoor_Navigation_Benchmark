"""Reversible path derivation for the sealed local SDK verification helper.

Only repository path identities change. SDK source hashes, bound method origins,
ownership guards and shutdown operations are retained byte for byte.
"""
import ast
import hashlib
import json
from pathlib import Path


def derive_relocated_checkout(source, legacy_root, canonical_root):
    legacy_root, canonical_root = Path(legacy_root), Path(canonical_root).resolve()
    if legacy_root.resolve() != canonical_root:
        raise ValueError('The legacy root must resolve to this exact checkout')
    replacements = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(str(legacy_root) + '/'):
            before = ast.get_source_segment(source, node)
            after = json.dumps(str(canonical_root) + node.value[len(str(legacy_root)):])
            if before != after:
                replacements.append((before, after))
    if len(replacements) != 2:
        raise ValueError('The sealed helper must contain exactly two relocated SDK source paths')
    comparisons = (
        ('manifest["extension_path"] != str(OVERLAY)',
         'Path(manifest["extension_path"]).resolve() != OVERLAY.resolve()'),
        ('{key: expected[key] for key in ("path", "bytes", "sha256")}',
         '{key: (str(Path(expected[key]).resolve()) if key == "path" else expected[key]) for key in ("path", "bytes", "sha256")}'),
        ('list(dev_paths or ()) != [str(OVERLAY)]',
         '[Path(path).resolve() for path in (dev_paths or ())] != [OVERLAY.resolve()]'),
    )
    replacements.extend(comparisons)
    derived = source
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError('Ambiguous checkout path derivation')
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(replacements):
        if inverse.count(after) != 1:
            raise ValueError('Ambiguous checkout path inverse')
        inverse = inverse.replace(after, before, 1)
    if inverse != source:
        raise RuntimeError('Checkout path derivation is not reversible')
    ast.parse(derived)
    return derived, {
        'legacy_root': str(legacy_root), 'canonical_root': str(canonical_root),
        'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'derived_sha256': hashlib.sha256(derived.encode()).hexdigest(),
        'absolute_source_path_changes': 2, 'canonical_path_comparisons': 3,
        'exact_inverse': True, 'SDK_source_changes': False,
        'sealed_source_changes': False, 'shutdown_operations_changed': False,
    }


def overlay_kit_arguments(legacy_root, overlay):
    """Use the verified whitespace-free alias with the SDK's str.split parser."""
    legacy_root, overlay = Path(legacy_root), Path(overlay).resolve()
    alias = legacy_root / 'output/native_sdk_overlay/simulation_manager_shutdown_v1/isaacsim.core.simulation_manager'
    if alias.resolve() != overlay or any(char.isspace() for char in str(alias)):
        raise ValueError('A whitespace-free alias to the exact overlay is required')
    return '--/app/exts/devPaths=' + json.dumps([str(alias)], separators=(',', ':'))
