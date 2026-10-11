"""Relocate only file-valued USD opinions in caller-owned anonymous copies.

Original files and layer contents remain unchanged. A full inverse serialization
check proves all non-file opinions (geometry, transforms, physics) are retained.
"""
import hashlib
from pathlib import Path

from .resource_bindings import resolve_path


def relocate_layers(records, source_path):
    from pxr import Sdf, UsdUtils
    originals, copies = {}, {}
    for record in records:
        path = resolve_path(record['path'])
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Portable USD source identity differs: '+str(path))
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError('Cannot read portable USD source: '+str(path))
        originals[str(path)] = layer
        copy = Sdf.Layer.CreateAnonymous('portable_'+path.name)
        copy.TransferContent(layer)
        copies[str(path)] = copy
    receipts = []
    for path, original in originals.items():
        changes = {}
        def translate(value):
            # MDL names are provided by the installed SDK, not local files.
            if not value or (Path(value).suffix == '.mdl' and not Path(value).is_absolute()):
                return value
            selected = resolve_path(value, relative_to=Path(path).parent)
            if str(selected) in copies:
                result = copies[str(selected)].identifier
            elif selected.is_file():
                result = str(selected)
            else:
                raise ValueError('Unregistered or missing USD file dependency: '+value)
            if result != value:
                changes[value] = result
            return result
        UsdUtils.ModifyAssetPaths(copies[path], translate)
        reverse = {}
        for before, after in changes.items():
            if after in reverse and reverse[after] != before:
                raise ValueError('Ambiguous USD file-location inverse')
            reverse[after] = before
        inverse = Sdf.Layer.CreateAnonymous('portable_inverse.usda')
        inverse.TransferContent(copies[path])
        UsdUtils.ModifyAssetPaths(inverse, lambda value: reverse.get(value, value))
        original_text = original.ExportToString()
        if inverse.ExportToString() != original_text:
            raise ValueError('USD file-location inverse changed another opinion')
        receipts.append({'original':path,'original_ram_sha256':hashlib.sha256(original_text.encode()).hexdigest(),
            'owned_layer':copies[path].identifier,'owned_ram_sha256':hashlib.sha256(copies[path].ExportToString().encode()).hexdigest(),'file_location_changes':changes,
            'exact_inverse':True,'geometry_or_physics_opinions_changed':False})
    root = copies[str(resolve_path(source_path))]
    return root, tuple(copies.values()), receipts


def verify_owned_layers(layers, receipts):
    """Require the complete, unchanged in-memory source closure at live admission."""
    if len(layers) != len(receipts) or not layers:
        raise ValueError('Incomplete owned USD layer closure')
    expected = {row['owned_layer']:row for row in receipts}
    if len(expected) != len(receipts) or set(expected) != {layer.identifier for layer in layers}:
        raise ValueError('Owned USD layer identities differ')
    for layer in layers:
        row = expected[layer.identifier]
        if (row['exact_inverse'] is not True or row['geometry_or_physics_opinions_changed'] is not False
                or hashlib.sha256(layer.ExportToString().encode()).hexdigest() != row['owned_ram_sha256']):
            raise ValueError('Owned USD source changed after exact inverse verification')


def stable_layer_receipts(layers, receipts, *, source_records):
    """Compare independently created USD copies by content, not memory address.

    Live anonymous identifiers remain in the owner's diagnostic receipt. Only
    this pairing view replaces references to owned layers with original paths;
    all other file opinions and every geometry/physics opinion are retained.
    """
    from pxr import Sdf, UsdUtils
    verify_owned_layers(layers, receipts)
    aliases = {row['owned_layer']: row['original'] for row in receipts}
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Duplicate original USD layer')
    originals = {str(resolve_path(row['path'])) for row in source_records}
    if len(originals) != len(source_records) or set(aliases.values()) != originals:
        raise ValueError('Portable USD source closure is incomplete')
    def canonical_path(value):
        if value.startswith('anon:') and value not in aliases:
            raise ValueError('Unregistered anonymous USD dependency')
        return aliases.get(value, value)
    by_id = {layer.identifier: layer for layer in layers}
    expected_keys = {'original', 'original_ram_sha256', 'owned_layer', 'owned_ram_sha256',
                     'file_location_changes', 'exact_inverse', 'geometry_or_physics_opinions_changed'}
    result = []
    for row in sorted(receipts, key=lambda r: r['original']):
        if set(row) != expected_keys:
            raise ValueError('Unexpected portable USD receipt fields')
        changes = row['file_location_changes']
        if len(set(changes.values())) != len(changes):
            raise ValueError('Ambiguous portable USD receipt inverse')
        reverse = {value: key for key, value in changes.items()}
        copy = Sdf.Layer.CreateAnonymous('pairing_inverse.usda')
        copy.TransferContent(by_id[row['owned_layer']])
        UsdUtils.ModifyAssetPaths(copy, lambda v: reverse.get(v, v))
        if hashlib.sha256(copy.ExportToString().encode()).hexdigest() != row['original_ram_sha256']:
            raise ValueError('Portable USD inverse receipt changed')
        copy.TransferContent(by_id[row['owned_layer']])
        UsdUtils.ModifyAssetPaths(copy, canonical_path)
        result.append({'original': row['original'], 'original_ram_sha256': row['original_ram_sha256'],
                       'canonical_ram_sha256': hashlib.sha256(copy.ExportToString().encode()).hexdigest(),
                       'file_location_changes': {k: canonical_path(v) for k, v in changes.items()},
                       'exact_inverse': True, 'geometry_or_physics_opinions_changed': False})
    return {'schema': 'portable_usd.content_pairing.v1', 'layers': result}
