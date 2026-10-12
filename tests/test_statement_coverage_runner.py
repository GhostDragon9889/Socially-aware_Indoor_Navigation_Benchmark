"""Trace real child interpreters; incomplete or stale execution must fail closed."""

import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


@pytest.fixture
def runner():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "statement_runner", root / "scripts/verification/run_statement_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path, runner, monkeypatch):
    root = tmp_path / "project's 目录"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".social-navigation-root").touch()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "program.py").write_text("answer = 6 * 7\nassert answer == 42\n")
    (root / "scripts/run_external_python.py").write_text(
        "import runpy, sys\nsys.argv = sys.argv[2:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    (root / "scripts/run_project_tests.py").write_text(
        "import argparse, subprocess, sys\nparser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-dir')\nparser.add_argument('--cpu-only', action='store_true')\n"
        "parser.add_argument('targets', nargs='*')\nargs = parser.parse_args()\n"
        "raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', *args.targets]))\n"
    )
    (root / "tests/test_program.py").write_text(
        "from pathlib import Path\nimport runpy\n"
        "def test_program():\n"
        "    path = Path(__file__).resolve().parents[1] / 'program.py'\n"
        "    assert runpy.run_path(str(path))['answer'] == 42\n"
    )
    monkeypatch.setattr(runner, "selected_python", lambda _: Path(sys.executable))
    return root


def test_real_nested_processes_verify_complete_source(checkout, tmp_path, runner):
    output = tmp_path / "output's 目录"
    result = runner.run(checkout, output, ["tests/test_program.py"], cpu_only=True)
    assert result["status"] == "passed" and result["gate_passed"]
    assert result["actual_test_returncode"] == 0
    measured = json.loads((output / "coverage.json").read_text())
    assert measured["source_files"] == 3 and measured["missing_statements"] == 0
    assert measured["branch_data_collected"] and measured["source_stability_verified"]
    assert len(list((output / "data").glob(".coverage.*"))) >= 2
    assert "--cpu-only" in result["command"]


def test_untouched_source_is_a_gap_even_when_tests_pass(checkout, tmp_path, runner):
    (checkout / "never_run.py").write_text("unverified = 7\n")
    output = tmp_path / "coverage"
    result = runner.run(checkout, output, ["tests/test_program.py"])
    assert result["status"] == "coverage_gap" and not result["gate_passed"]
    assert result["actual_test_returncode"] == 0
    assert result["missing_statements"] == 1
    assert json.loads((output / "coverage.json").read_text())["zero_hit_files"] == [
        "never_run.py"
    ]


def test_source_mutation_after_execution_never_passes(checkout, tmp_path, runner):
    test = checkout / "tests/test_program.py"
    test.write_text(
        test.read_text()
        + "    path.write_text(path.read_text() + '# changed after execution\\n')\n"
    )
    output = tmp_path / "coverage"
    result = runner.run(checkout, output, [])
    assert result["status"] == "coverage_gap" and not result["gate_passed"]
    assert (
        result["statement_percent"] == 100 and not result["source_stability_verified"]
    )
    assert json.loads((output / "coverage.json").read_text())["source_changes"] == [
        "program.py"
    ]


def test_real_test_failure_keeps_diagnostics(checkout, tmp_path, runner):
    test = checkout / "tests/test_program.py"
    test.write_text(test.read_text() + "    assert False, 'deliberate test failure'\n")
    output = tmp_path / "coverage"
    with pytest.raises(RuntimeError, match="Failed instrumented"):
        runner.run(checkout, output, [])
    result = json.loads((output / "result.json").read_text())
    assert (
        result["status"] == "failed_preserved" and result["actual_test_returncode"] == 1
    )
    assert "deliberate test failure" in (output / "test_owner.log").read_text()
    assert not (output / "coverage.json").exists()


def test_sdk_selection_failure_is_recorded(checkout, tmp_path, runner, monkeypatch):
    def unavailable(_):
        raise RuntimeError("SDK is not configured")

    monkeypatch.setattr(runner, "selected_python", unavailable)
    output = tmp_path / "coverage"
    with pytest.raises(RuntimeError, match="not configured"):
        runner.run(checkout, output, [])
    result = json.loads((output / "result.json").read_text())
    assert (
        result["status"] == "failed_preserved"
        and result["actual_test_returncode"] is None
    )
    assert "not configured" in result["error"]


def test_existing_output_is_never_overwritten(checkout, tmp_path, runner):
    output = tmp_path / "coverage"
    output.mkdir()
    receipt = output / "result.json"
    receipt.write_text("retained previous evidence")
    with pytest.raises(FileExistsError):
        runner.run(checkout, output, [])
    assert receipt.read_text() == "retained previous evidence"


def test_isaacgym_scope_and_hardware_waiver_reach_real_test_processes(
    checkout, tmp_path, runner
):
    backend = checkout / "src/protomotions/simulator/isaacgym"
    backend.mkdir(parents=True)
    (backend / "simulator.py").write_text(
        "raise RuntimeError('unused IsaacGym backend')\n"
    )
    hardware = checkout / "src/protomotions/tests"
    hardware.mkdir()
    (hardware / "test_isaacgym_friction_mode.py").write_text(
        "def test_retired_hardware():\n    assert False, 'must be explicitly waived'\n"
    )
    output = tmp_path / "coverage"
    result = runner.run(checkout, output, [], ignore_isaacgym=True)
    assert result["status"] == "passed" and result["gate_passed"]
    assert result["ignored_backends"] == ["isaacgym"]
    assert (
        "--ignore=src/protomotions/tests/test_isaacgym_friction_mode.py"
        in result["command"]
    )
    measured = json.loads((output / "coverage.json").read_text())
    assert len(measured["excluded_files"]) == 1 and measured["source_files"] == 3


@pytest.mark.parametrize(
    "gate,report_only,expected", [(True, False, 0), (False, False, 1), (False, True, 0)]
)
def test_cli_distinguishes_reporting_from_passing(
    runner, monkeypatch, tmp_path, gate, report_only, expected
):
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: {"gate_passed": gate})
    args = ["--output-dir", str(tmp_path / "output")]
    if report_only:
        args.append("--report-only")
    assert runner.main(args) == expected


@pytest.mark.parametrize(
    "script",
    [
        "verification/run_statement_coverage.py",
        "verification/audit_statement_coverage.py",
        "maintenance/clean_project_cache.py",
    ],
)
def test_public_entry_points_show_help(script, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(path), run_name="__main__")
    assert caught.value.code == 0 and "usage:" in capsys.readouterr().out
