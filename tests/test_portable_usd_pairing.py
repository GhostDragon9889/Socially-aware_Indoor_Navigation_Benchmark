"""Separate processes' anonymous USD IDs cannot change a physical pairing."""

from copy import deepcopy
import hashlib

import pytest

from social_navigation.portable_usd import (
    relocate_layers,
    stable_layer_receipts,
    verify_owned_layers,
)

Usd = pytest.importorskip("pxr.Usd")
Sdf = pytest.importorskip("pxr.Sdf")


@pytest.fixture
def layers(tmp_path):
    child = tmp_path / "child.usda"
    stage = Usd.Stage.CreateNew(str(child))
    prim = stage.DefinePrim("/Robot", "Xform")
    prim.CreateAttribute("test:mass", Sdf.ValueTypeNames.Float).Set(17.0)
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()
    root = tmp_path / "root.usda"
    stage = Usd.Stage.CreateNew(str(root))
    stage.DefinePrim("/World/Robot", "Xform").GetReferences().AddReference(
        "./child.usda"
    )
    stage.GetRootLayer().Save()
    records = [
        {
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in (root, child)
    ]
    return root, records


def test_two_independent_usd_allocations_have_exact_stable_pairing(layers):
    root, records = layers
    _, left, a = relocate_layers(records, root)
    _, right, b = relocate_layers(records, root)
    assert {r["owned_layer"] for r in a}.isdisjoint(r["owned_layer"] for r in b)
    assert a != b
    assert stable_layer_receipts(
        left, a, source_records=records
    ) == stable_layer_receipts(right, b, source_records=records)
    assert all(
        "anon:" not in str(row)
        for row in stable_layer_receipts(left, a, source_records=records)["layers"]
    )


@pytest.mark.parametrize("defect", ["geometry", "inverse", "closure", "both_omit"])
def test_stable_pairing_rejects_mutation(layers, defect):
    root, records = layers
    _, owned, original = relocate_layers(records, root)
    receipts = deepcopy(original)
    if defect == "geometry":
        stage = Usd.Stage.Open(owned[1])
        stage.GetPrimAtPath("/Robot").GetAttribute("test:mass").Set(18.0)
    elif defect == "inverse":
        receipts[0]["original_ram_sha256"] = "0" * 64
    else:
        owned = owned[:-1]
        if defect == "both_omit":
            receipts = receipts[:-1]
    with pytest.raises(ValueError):
        stable_layer_receipts(owned, receipts, source_records=records)


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_source_identity_is_checked_before_copying(layers, field):
    root, records = layers
    corrupt = deepcopy(records)
    corrupt[0][field] = 0 if field == "bytes" else "0" * 64
    with pytest.raises(ValueError, match="source identity differs"):
        relocate_layers(corrupt, root)


def test_unreadable_sdk_layer_is_rejected_after_exact_file_check(layers, monkeypatch):
    root, records = layers
    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", lambda path: None)
    with pytest.raises(ValueError, match="Cannot read portable USD"):
        relocate_layers(records, root)


def test_missing_file_dependency_has_no_portable_copy(tmp_path):
    root = tmp_path / "missing.usda"
    root.write_text(
        '#usda 1.0\ndef Xform "World" {\n custom asset texture = @missing.png@\n}\n'
    )
    record = {
        "path": str(root),
        "bytes": root.stat().st_size,
        "sha256": hashlib.sha256(root.read_bytes()).hexdigest(),
    }
    before = root.read_bytes()
    with pytest.raises(ValueError, match="missing USD file dependency"):
        relocate_layers([record], root)
    assert root.read_bytes() == before


def test_existing_asset_and_sdk_mdl_keep_original_file_opinions(tmp_path):
    root = tmp_path / "external.usda"
    texture = tmp_path / "texture.bin"
    texture.write_bytes(b"fixture texture")
    root.write_text(
        '#usda 1.0\ndef Xform "World" {\n custom asset texture = @texture.bin@\n custom asset mdl = @OmniPBR.mdl@\n}\n'
    )
    record = {
        "path": str(root),
        "bytes": root.stat().st_size,
        "sha256": hashlib.sha256(root.read_bytes()).hexdigest(),
    }
    _, owned, receipts = relocate_layers([record], root)
    assert receipts[0]["file_location_changes"] == {"texture.bin": str(texture)}
    assert "OmniPBR.mdl" in owned[0].ExportToString()
    verify_owned_layers(owned, receipts)


def test_two_original_spellings_cannot_share_one_inverse(layers):
    root, records = layers
    stage = Usd.Stage.Open(str(root))
    stage.DefinePrim("/World/Second", "Xform").GetReferences().AddReference(
        "child.usda"
    )
    stage.GetRootLayer().Save()
    records[0].update(
        bytes=root.stat().st_size, sha256=hashlib.sha256(root.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Ambiguous USD file-location inverse"):
        relocate_layers(records, root)


def test_inverse_proof_detects_a_nonfile_sdk_mutation(layers, monkeypatch):
    from pxr import UsdUtils

    root, records = layers
    original = UsdUtils.ModifyAssetPaths

    def mutate_owned(layer, callback):
        original(layer, callback)
        if layer.identifier.endswith("portable_root.usda"):
            layer.customLayerData = {"unexpected": 1}

    monkeypatch.setattr(UsdUtils, "ModifyAssetPaths", mutate_owned)
    with pytest.raises(ValueError, match="inverse changed another opinion"):
        relocate_layers(records, root)
    assert "unexpected" not in Sdf.Layer.FindOrOpen(str(root)).customLayerData


@pytest.mark.parametrize(
    "defect", ["identities", "originals", "fields", "inverse_collision"]
)
def test_pairing_rejects_forged_receipt_structure(layers, defect):
    root, records = layers
    _, owned, original = relocate_layers(records, root)
    receipts = deepcopy(original)
    if defect == "identities":
        receipts[0]["owned_layer"] = receipts[1]["owned_layer"]
    elif defect == "originals":
        receipts[0]["original"] = receipts[1]["original"]
    elif defect == "fields":
        receipts[0]["extra"] = "unexpected"
    else:
        receipts[0]["file_location_changes"] = {"a": "one", "b": "one"}
    with pytest.raises(
        ValueError, match="identities|Duplicate original|Unexpected|Ambiguous"
    ):
        stable_layer_receipts(owned, receipts, source_records=records)


def test_pairing_rejects_unregistered_anonymous_layer(layers):
    root, records = layers
    _, owned, original = relocate_layers(records, root)
    receipts = deepcopy(original)
    unknown = Sdf.Layer.CreateAnonymous("unregistered.usda")
    stage = Usd.Stage.Open(owned[0])
    stage.GetPrimAtPath("/World/Robot").CreateAttribute(
        "test:unregistered", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(unknown.identifier))
    receipts[0]["owned_ram_sha256"] = hashlib.sha256(
        owned[0].ExportToString().encode()
    ).hexdigest()
    # Keep the original inverse proof valid for this deliberately forged input,
    # so admission reaches the separate anonymous-dependency registry check.
    inverse = Sdf.Layer.CreateAnonymous("inverse_fixture.usda")
    inverse.TransferContent(owned[0])
    from pxr import UsdUtils

    reverse = {
        value: key for key, value in receipts[0]["file_location_changes"].items()
    }
    UsdUtils.ModifyAssetPaths(inverse, lambda value: reverse.get(value, value))
    receipts[0]["original_ram_sha256"] = hashlib.sha256(
        inverse.ExportToString().encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="Unregistered anonymous USD"):
        stable_layer_receipts(owned, receipts, source_records=records)
