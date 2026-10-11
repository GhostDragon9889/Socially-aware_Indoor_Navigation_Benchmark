"""Fail-closed path migration of the pinned native verifier; no SDK starts."""
import hashlib

import pytest

from CrowdSim.humanoid.runtime.relocated_terminal import derive_relocated_checkout, overlay_kit_arguments
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import (
    ROOT, TERMINAL, TERMINAL_SHA, IsolatedTrackingLifecycle,
)


def test_actual_sealed_verifier_accepts_same_files_after_directory_rename(monkeypatch):
    monkeypatch.setattr("CrowdSim.humanoid.runtime.sdk61_lifecycle.uses_sim61", lambda: False)
    source = TERMINAL.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == TERMINAL_SHA
    derived, record = derive_relocated_checkout(source, ROOT, ROOT)
    assert record['exact_inverse'] and record['absolute_source_path_changes'] >= 2
    assert not record['SDK_source_changes'] and not record['shutdown_operations_changed']
    assert len(derived.splitlines()) == len(source.splitlines())
    report = {}
    owner = IsolatedTrackingLifecycle(report)
    assert report['native_overlay_inputs_verified']
    assert report['evaluation_lifecycle']['sealed_terminal']['sha256'] == TERMINAL_SHA
    assert owner.terminal.OVERLAY.resolve().is_dir()


def test_directory_alias_to_other_checkout_is_rejected(tmp_path):
    wrong = tmp_path / 'other'
    wrong.mkdir()
    with pytest.raises(ValueError, match='exact checkout'):
        derive_relocated_checkout(TERMINAL.read_text(), wrong, ROOT)


def test_expected_source_shape_cannot_silently_change():
    changed = TERMINAL.read_text().replace('manifest["extension_path"] != str(OVERLAY)', 'True')
    with pytest.raises(ValueError, match='Ambiguous'):
        derive_relocated_checkout(changed, ROOT, ROOT)


def test_sdk_split_arguments_preserve_exact_overlay_directory(monkeypatch):
    monkeypatch.setattr("CrowdSim.humanoid.runtime.sdk61_lifecycle.uses_sim61", lambda: False)
    import json
    report = {}
    owner = IsolatedTrackingLifecycle(report)
    argument = overlay_kit_arguments(ROOT.parent / "ProtoMotions", owner.terminal.OVERLAY)
    assert len(argument.split()) == 1
    path, = json.loads(argument.split("=", 1)[1])
    assert owner.terminal.Path(path).resolve() == owner.terminal.OVERLAY.resolve()


def test_file_alias_preserves_bytes_hash_and_exact_file_identity(tmp_path):
    from CrowdSim.humanoid.runtime.dynbench_scene import file_record, same_file_record, _check_record
    actual = tmp_path / "actual"
    actual.write_bytes(b"native map fixture")
    alias = tmp_path / "legacy"
    alias.symlink_to(actual)
    recorded = dict(file_record(actual), path=str(alias))
    _check_record(recorded)
    assert same_file_record(file_record(actual), recorded)
    duplicate = tmp_path / "other"
    duplicate.write_bytes(actual.read_bytes())
    assert not same_file_record(file_record(duplicate), recorded)
    actual.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="bytes changed"):
        _check_record(recorded)
