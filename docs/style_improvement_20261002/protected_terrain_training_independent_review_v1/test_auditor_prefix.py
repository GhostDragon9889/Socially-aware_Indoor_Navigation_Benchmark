"""CPU contracts for committed evidence; these fixtures are not training runs."""
from pathlib import Path
import subprocess
import sys

import pytest

from protected_checks import read_committed_metric_prefix

HERE = Path(__file__).resolve().parent


def test_live_prefix_ignores_uncommitted_partial_future(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"update_count":1}\n{"update_count":2}\n{partial_future')
    assert read_committed_metric_prefix(metrics, 2, require_closed=False) == [
        '{"update_count":1}', '{"update_count":2}']
    with pytest.raises(ValueError, match="beyond"):
        read_committed_metric_prefix(metrics, 2, require_closed=True)


@pytest.mark.parametrize("text", ["", '{"update_count":1}', '{"update_count":1}\n'])
def test_missing_or_partial_committed_rows_are_rejected(tmp_path, text):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(text)
    with pytest.raises(ValueError, match="missing|incomplete"):
        read_committed_metric_prefix(metrics, 2, require_closed=False)


def test_closed_exact_budget_is_accepted(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"update_count":1}\n')
    assert read_committed_metric_prefix(metrics, 1, require_closed=True) == ['{"update_count":1}']


@pytest.mark.parametrize("phase,with_result", [
    ("milestone", True), ("pilot4", False), ("pilot8", False), ("terminal2000", False)])
def test_result_phase_mismatch_rejects_before_torch_or_output(tmp_path, phase, with_result):
    destination = tmp_path / "must_not_be_created"
    command = [sys.executable, str(HERE / "audit_protected_training.py"), "--phase", phase,
               "--checkpoint", str(tmp_path / "nonexistent.pt"), "--metrics", str(tmp_path / "nonexistent.jsonl"),
               "--output", str(destination)]
    if with_result:
        command.extend(["--result", str(tmp_path / "nonexistent_result.json")])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "Milestones have no terminal result" in result.stderr
    assert not destination.exists()
