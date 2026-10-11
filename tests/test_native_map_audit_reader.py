"""CPU evidence-reader failures/registration, without simulated SDK success."""
import json
from pathlib import Path

import numpy as np
import pytest

from CrowdSim.humanoid.runtime.dynbench_scene import file_record

ROOT = Path(__file__).resolve().parents[1]
from CrowdSim.humanoid.analysis import native_map_audit_reader as reader


def test_registration_does_not_infer_execution_from_missing_outputs(tmp_path):
    result = reader.registration(tmp_path / "absent", ROOT / "docs/style_improvement_20261002/native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py")
    assert result["status"] == "prepared_actual_results_pending"
    assert result["native_v7_execution_observed"] is False
    assert result["measured_map_admission"] is None and result["route_validation"] is None
    assert not (tmp_path / "absent").exists()


def test_actual_reader_saves_failure_for_absent_results(tmp_path):
    output = tmp_path / "review"
    assert reader.main(["--native-output", str(tmp_path / "absent"), "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "actual_evidence_audit_failed"
    assert report["native_v7_execution_observed"] is False
    assert report["scenes"] == [] and report["routes"] is None


def test_synthetic_or_old_native_results_cannot_be_actual_v7(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"scene": "Office", "CPU_contract_fixture_only": True,
                                                        "script": {"sha256": reader.PROBE_SHA256}}))
    with pytest.raises(ValueError, match="synthetic"):
        reader.audit_scene(tmp_path, ROOT / "scripts/probe_paper_scene_maps.py")
    (tmp_path / "result.json").write_text(json.dumps({"scene": "Office", "status": "scene_resource_gate_passed"}))
    with pytest.raises(ValueError, match="Older"):
        reader.audit_scene(tmp_path, ROOT / "scripts/probe_paper_scene_maps.py")


def test_inventory_retains_dtype_bytes_and_rejects_bad_buffers(tmp_path):
    path = tmp_path / "native.npz"
    values = np.array([1., 2.], np.float32)
    np.savez(path, values=values, paths=np.array(["/Ground"]))
    result = reader.array_inventory(file_record(path))
    assert result["arrays"]["values"]["dtype"] == "<f4"
    assert result["arrays"]["values"]["shape"] == [2]
    np.savez(path, values=np.array([float("nan")]))
    with pytest.raises(ValueError, match="nonfinite"):
        reader.array_inventory(file_record(path))


def test_reader_cannot_overwrite_prior_audit(tmp_path):
    with pytest.raises(SystemExit):
        reader.main(["--output", str(tmp_path), "--register-only"])


def test_relocated_reader_keeps_the_complete_historical_inverse():
    source = reader.PARENT.read_text()
    derived, proof = reader.derive_reader(source)
    assert proof['exact_inverse'] and not proof['SDK_executed']
    assert 'allow_pickle=False' in derived
    assert 'hashlib.sha256(content).hexdigest() != record["sha256"]' in derived
    with pytest.raises(ValueError, match='Frozen v7 audit implementation changed'):
        reader.derive_reader(source.replace('allow_pickle=False', 'allow_pickle=True'))
