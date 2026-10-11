"""Separate processes' anonymous USD IDs cannot change a physical pairing."""
from copy import deepcopy
import hashlib

import pytest

from social_navigation.portable_usd import relocate_layers, stable_layer_receipts

Usd = pytest.importorskip('pxr.Usd')
Sdf = pytest.importorskip('pxr.Sdf')


@pytest.fixture
def layers(tmp_path):
    child = tmp_path / 'child.usda'
    stage = Usd.Stage.CreateNew(str(child))
    prim = stage.DefinePrim('/Robot', 'Xform')
    prim.CreateAttribute('test:mass', Sdf.ValueTypeNames.Float).Set(17.)
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()
    root = tmp_path / 'root.usda'
    stage = Usd.Stage.CreateNew(str(root))
    stage.DefinePrim('/World/Robot', 'Xform').GetReferences().AddReference('./child.usda')
    stage.GetRootLayer().Save()
    records = [{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in (root, child)]
    return root, records


def test_two_independent_usd_allocations_have_exact_stable_pairing(layers):
    root, records = layers
    _, left, a = relocate_layers(records, root)
    _, right, b = relocate_layers(records, root)
    assert {r['owned_layer'] for r in a}.isdisjoint(r['owned_layer'] for r in b)
    assert a != b
    assert stable_layer_receipts(left,a,source_records=records) == stable_layer_receipts(right,b,source_records=records)
    assert all('anon:' not in str(row) for row in stable_layer_receipts(left,a,source_records=records)['layers'])


@pytest.mark.parametrize('defect', ['geometry','inverse','closure','both_omit'])
def test_stable_pairing_rejects_mutation(layers, defect):
    root, records = layers
    _, owned, original = relocate_layers(records, root)
    receipts = deepcopy(original)
    if defect == 'geometry':
        stage = Usd.Stage.Open(owned[1])
        stage.GetPrimAtPath('/Robot').GetAttribute('test:mass').Set(18.)
    elif defect == 'inverse':
        receipts[0]['original_ram_sha256'] = '0' * 64
    else:
        owned = owned[:-1]
        if defect == 'both_omit':
            receipts = receipts[:-1]
    with pytest.raises(ValueError):
        stable_layer_receipts(owned,receipts,source_records=records)
