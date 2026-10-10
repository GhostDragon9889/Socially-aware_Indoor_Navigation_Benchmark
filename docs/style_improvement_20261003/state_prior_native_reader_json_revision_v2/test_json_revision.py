"""Regression for the actual native15 proof serialization failure."""
from copy import deepcopy
import json

import pytest

import audit_native15_json_revision as revision


def test_actual_frozen15_tuple_proof_explains_first_failure():
    core = revision.load_core()
    _, proof = core.readers(15)
    registered = revision.read(revision.CORE / "native15_registration.json")
    assert len(proof["origin_shape_constant_changes"]) == 4
    assert all(type(row) is tuple for row in proof["origin_shape_constant_changes"])
    assert all(type(row) is list for row in registered["adapter_proof"]["origin_shape_constant_changes"])
    assert proof != registered["adapter_proof"]
    assert revision.canonical_proof(proof) == registered["adapter_proof"]
    assert "Frozen mathematics adapter proof changed" in (revision.HERE / "native15_first_actual_failure.log").read_text()


def test_actual4_empty_shape_proof_is_already_JSON_equal():
    core = revision.load_core()
    _, proof = core.readers(4)
    registered = revision.read(revision.CORE / "native4_registration.json")
    assert proof == registered["adapter_proof"]
    assert revision.canonical_proof(proof) == proof


def test_normalization_preserves_every_original_numeric_shape_value():
    core = revision.load_core()
    _, proof = core.readers(15)
    original = deepcopy(proof)
    normalized = revision.canonical_proof(proof)
    assert proof == original
    assert [(row[1], row[2]) for row in normalized["origin_shape_constant_changes"]] == [(4, 15), (132, 143), (4, 15), (4, 15)]
    assert json.dumps(proof, allow_nan=False) == json.dumps(normalized, allow_nan=False)


def test_changed_math_or_shape_value_is_not_admitted(monkeypatch):
    core = revision.load_core()
    real_readers = core.readers

    def changed_readers(actors):
        module, proof = real_readers(actors)
        proof["origin_shape_constant_changes"][1] = (proof["origin_shape_constant_changes"][1][0], 132, 144)
        return module, proof

    core.readers = changed_readers
    monkeypatch.setattr(revision, "load_core", lambda: core)
    registered = revision.read(revision.HERE / "native15_revision_registration.json")
    with pytest.raises(ValueError, match="cannot change original proof values"):
        revision.audit(registered)


def test_rewritten_v2_driver_identity_rejected():
    registered = revision.read(revision.HERE / "native15_revision_registration.json")
    registered["representation_adapter"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Exact frozen v2"):
        revision.audit(registered)
