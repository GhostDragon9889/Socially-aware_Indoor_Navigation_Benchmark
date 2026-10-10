"""CPU metric source gates; never read a future checkpoint or capture."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("CPU_metric_preparation_tests", HERE / "run_CPU_metrics.py")
RUNNER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUNNER)
CATALOG = json.loads((HERE / "commands_CPU_metrics_v1.json").read_text())


def test_actual_frozen_sources_prepare_without_actual_capture_read(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    original = Path.read_text
    forbidden = {Path(row["input"]) / "result.json" for row in CATALOG["metrics"].values()}
    forbidden.update(Path(row["checkpoint_admission"]) for row in CATALOG["metrics"].values())

    def guarded(path, *args, **kwargs):
        assert path not in forbidden, "Future actual payload was consumed during preparation"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    assert RUNNER.main(["--prepare-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["future_effect"] is False and report["actual_checkpoint_read"] is False
    assert report["actual_metrics_computed"] is False and report["copied_files_hashed"] == 935
    assert "torch" not in sys.modules


@pytest.mark.parametrize("defect", ["source_SHA", "capture_alias", "terminal_alias", "metric_args", "output_alias"])
def test_unregistered_sources_arguments_or_aliases_are_rejected(defect):
    catalog = deepcopy(CATALOG)
    if defect == "source_SHA":
        catalog["sources"]["style90_math"]["sha256"] = "f" * 64
    elif defect == "capture_alias":
        catalog["metrics"]["tracking90"]["input"] = "/tmp/test34"
    elif defect == "terminal_alias":
        catalog["metrics"]["tracking90"]["checkpoint_admission"] = "/tmp/milestone500.json"
    elif defect == "metric_args":
        catalog["metrics"]["navigation90"]["metric_argv"][6] = "2"
    else:
        catalog["metrics"]["navigation45"]["output"] = "/tmp/metric"
    with pytest.raises(ValueError):
        RUNNER.validate_preparation(catalog)


def test_five_complete_metric_commands_keep_original_parameters():
    assert set(CATALOG["metrics"]) == set(RUNNER.METRICS)
    for key, row in CATALOG["metrics"].items():
        assert row["metric_argv"] == RUNNER.metric_argv(key, CATALOG)
        assert row["command"][-2:] == ["--metric", key]
    assert CATALOG["metric_rules"]["speed_edges_m_s"] == [0., .1, .5, 1., 1.5, 2., 3., None]
    assert CATALOG["metric_rules"]["all_failed_and_idle_samples_retained"] is True
    assert CATALOG["metric_rules"]["pose45_refit_allowed"] is False


def test_saved_45D_basis_reused_exact_without_fit_or_generated_samples():
    tool = SimpleNamespace(fit_reference=lambda *_args: pytest.fail("Original kmeans fit executed"))
    basis = Path(CATALOG["sources"]["pose45_basis"]["path"])
    original = json.loads(Path(CATALOG["sources"]["pose45_reference_evaluation"]["path"]).read_text())
    centers = RUNNER.reuse_saved_basis(tool, basis, original, RUNNER.record)
    count = original["method"]["fit"]["available_reference_poses"]
    values, fit = tool.fit_reference(np.zeros((count, 45), np.float64))
    with np.load(basis, allow_pickle=False) as saved:
        assert all(np.array_equal(values[key], saved[f"centers_{key}"]) for key in (50, 100))
    assert values is centers and fit["refit_performed_for_this_new_run"] is False
    assert fit["generated_samples_used_for_fit"] is False
    with pytest.raises(ValueError):
        tool.fit_reference(np.zeros((count - 1, 45), np.float64))


def test_failed_actual_source_gate_stops_before_any_metric_processing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(RUNNER, "validate_preparation", lambda _catalog: (None, None))
    original_read = Path.read_text
    result = Path(CATALOG["metrics"]["navigation45"]["input"]) / "result.json"

    def read(path, *args, **kwargs):
        return "{}" if path == result else original_read(path, *args, **kwargs)

    def reject(*_args, **_kwargs):
        raise ValueError("Actual source admission rejected by test")

    def only_bridge(name, _path):
        assert name == "actual_new935_metric_source_bridge", "Metric tool loaded before source admission"
        return SimpleNamespace(admit_actual_evaluation=reject)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(RUNNER, "load", only_bridge)
    monkeypatch.setattr(RUNNER, "reuse_saved_basis", lambda *_args: pytest.fail("Basis processing began before actual source gate"))
    with pytest.raises(ValueError, match="Actual source admission rejected"):
        RUNNER.main(["--metric", "navigation45"])
