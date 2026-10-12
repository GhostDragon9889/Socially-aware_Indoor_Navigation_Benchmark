"""Fail-closed path migration of the pinned native verifier; no SDK starts."""
import hashlib

import pytest

from CrowdSim.humanoid.runtime.relocated_terminal import derive_relocated_checkout, overlay_kit_arguments
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import (
    ROOT, TERMINAL, TERMINAL_SHA, IsolatedTrackingLifecycle,
)




def test_directory_alias_to_other_checkout_is_rejected(tmp_path):
    wrong = tmp_path / 'other'
    wrong.mkdir()
    with pytest.raises(ValueError, match='exact checkout'):
        derive_relocated_checkout(TERMINAL.read_text(), wrong, ROOT)


def test_expected_source_shape_cannot_silently_change():
    changed = TERMINAL.read_text().replace('manifest["extension_path"] != str(OVERLAY)', 'True')
    with pytest.raises(ValueError, match='Ambiguous'):
        derive_relocated_checkout(changed, ROOT, ROOT)




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
