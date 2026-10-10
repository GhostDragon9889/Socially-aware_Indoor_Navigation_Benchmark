"""Actual CPU subprocess import provenance, including uncatchable SDK-style exit."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_v1"
LAUNCHER = BUNDLE / "launch_isolated_v3.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("isolated_actual_ledger_v3", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module", ["CrowdSim.humanoid.application.settings", "CrowdSim.humanoid.application.configuration"])
def test_actual_owned_import_ledger_survives_uncatchable_exit_without_final_claim(tmp_path, module):
    report = tmp_path / "actual_imports.json"
    program = f"""
import importlib.util, importlib, os
spec=importlib.util.spec_from_file_location('isolated_v3', {str(LAUNCHER)!r})
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
ledger=launcher.ImportLedger({str(report)!r});launcher.activate(ledger)
ledger.phase='CPU_uncatchable_exit_probe';ledger.write('guard_installed')
importlib.import_module({module!r})
os._exit(0)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", program], cwd="/tmp", env=env, timeout=30,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    events = [json.loads(line) for line in Path(data["event_ledger"]).read_text().splitlines()]
    assert data["runtime_finalization_observed"] is False
    assert data["status"] == "observed_not_finalized"
    assert data["phase"] == "CPU_uncatchable_exit_probe"
    assert not any(row["event"] == "final" for row in events)
    assert [row["sequence"] for row in events] == list(range(1, len(events)+1))
    assert len(events) == data["event_count"]
    assert module in {row["module"] for row in data["owned_modules"]}
    for row in data["owned_modules"]:
        path = Path(row["path"])
        assert path.is_relative_to(BUNDLE / "code") and row["execution_observed"] is True
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert not report.with_suffix(".json.pending").exists()


def test_existing_import_report_is_never_replaced_by_new_run(tmp_path):
    launcher = load_launcher()
    path = tmp_path / "actual.json"
    launcher.ImportLedger(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Fresh import ledger"):
        launcher.ImportLedger(path)
    assert path.read_bytes() == original


def test_relative_report_is_bound_before_isolated_cwd_changes(tmp_path, monkeypatch):
    launcher = load_launcher()
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    monkeypatch.chdir(before)
    ledger = launcher.ImportLedger("actual.json")
    monkeypatch.chdir(after)
    ledger.write("cwd_changed")
    assert ledger.path == before / "actual.json"
    assert json.loads(ledger.path.read_text())["event_count"] == 2
    assert not (after / "actual.json").exists()


def test_current_checkout_loader_origin_rejected_before_execution(tmp_path):
    launcher = load_launcher()
    ledger = launcher.ImportLedger(tmp_path / "guard.json")
    finder = launcher.OwnedImportGuard(ledger)
    with pytest.raises(ImportError, match="outside isolated code"):
        finder.find_spec("CrowdSim.humanoid.application.settings", [str(ROOT / "CrowdSim/humanoid/application")])
    assert not json.loads(ledger.path.read_text())["owned_modules"]


def test_failed_owned_loader_records_failure_without_fabricated_success(tmp_path):
    launcher = load_launcher()
    ledger = launcher.ImportLedger(tmp_path / "failed.json")

    class Delegate:
        def exec_module(self, module):
            raise RuntimeError("explicit import failure")

    class Module:
        __name__ = "CrowdSim.failed_owned_loader"

    with pytest.raises(RuntimeError, match="explicit import failure"):
        launcher.ObservedOwnedLoader(Delegate(), ledger).exec_module(Module())
    data = json.loads(ledger.path.read_text())
    events = [json.loads(line) for line in ledger.events_path.read_text().splitlines()]
    assert not data["owned_modules"]
    assert events[-1]["event"] == "owned_import_failed"
    assert events[-1]["error_type"] == "RuntimeError"
    ledger.finish("failed")
    assert json.loads(ledger.path.read_text())["runtime_finalization_observed"] is True
