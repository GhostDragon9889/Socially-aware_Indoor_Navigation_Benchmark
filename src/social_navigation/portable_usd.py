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
