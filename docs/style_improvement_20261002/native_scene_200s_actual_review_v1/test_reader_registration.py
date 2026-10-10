"""Admission boundaries only; no terminal native200 result is fabricated."""
import importlib.util
import json
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native200_CPU_reader_gate", DOC / "audit_native_200s.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def test_exact_copied_existing_reader_functions_and_gates_load():
    actual = reader.readers()
    assert actual.BASE.__file__ == str(DOC / "frozen_v1.py")
    assert actual.ROUTE.__file__ == str(DOC / "frozen_routes.py")


def test_real_registration_is_pending_and_does_not_inspect_terminal_captures():
    result = reader.registration()
    assert result["status"] == "prepared_actual_terminal_artifacts_pending"
    assert result["native_execution_audited"] is False
    assert result["physical_capture_admission"] is result["measured_counts"] is None
    assert result["source_binding"]["source_count"] == 168
    assert result["source_binding"]["all_current_source168_bytes_exact"] is True
    assert result["expected_counts"]["shared_physics_substeps"] == 24000


def test_audit_cannot_promote_a_nonpending_registration_before_reading_terminal():
    registered = json.loads((DOC / "registration.json").read_text())
    registered["native_execution_audited"] = True
    with pytest.raises(ValueError, match="pending registration"):
        reader.audit(registered)


def test_preserve_existing_output_without_registration_or_terminal_read(tmp_path):
    destination = tmp_path / "existing.json"
    destination.write_text("retained historical evidence\n")
    with pytest.raises(SystemExit):
        reader.main(["--register-only", "--output", str(destination)])
    assert destination.read_text() == "retained historical evidence\n"
