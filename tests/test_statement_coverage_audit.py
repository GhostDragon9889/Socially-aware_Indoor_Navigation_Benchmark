"""The coverage gate must include untouched source and reject stale traces."""

import json
from pathlib import Path
import subprocess

import coverage
import pytest

from social_navigation import coverage_audit as audit


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source's 目录"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def source(root, name, contents):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def execute(path, data_file, *, virtual=False):
    cov = coverage.Coverage(
        data_file=str(data_file),
        branch=True,
        include=[str(path) + "*"],
        config_file=False,
    )
    cov.set_option("report:exclude_lines", [])
    cov.start()
    try:
        exec(
            compile(
                path.read_bytes(), str(path) + (":generated" if virtual else ""), "exec"
            ),
            {},
        )
    finally:
        cov.stop()
        cov.save()


def test_zero_hit_files_pragmas_type_checking_and_main_guards_stay_in_denominator(
    checkout, tmp_path
):
    active = source(checkout, "src/active.py", "value = 1\n")
    untouched = source(
        checkout,
        "src/从未\n执行.py",
        'if __name__ == "unused":  # pragma: no cover\n    value = 2\n'
        'if __name__ == "__main__":\n    raise ValueError("not executed")\n',
    )
    source(checkout, "tests/test_example.py", "raise ValueError('test code')\n")
    source(checkout, "tools/test_camera_probe.py", "value = 3\n")
    source(checkout, "src/alias.py", "temporary\n").unlink()
    (checkout / "src/alias.py").symlink_to(active)
    baseline = audit.inventory(checkout)
    assert baseline["source_files"] == 3
    assert baseline["aliases"] == [{"path": "src/alias.py", "target": "src/active.py"}]
    data = tmp_path / ".coverage"
    execute(active, data)
    result = audit.report(checkout, data, baseline=baseline)
    assert result["covered_statements"] == 1
    assert untouched.relative_to(checkout).as_posix() in result["zero_hit_files"]
    assert result["missing_statements"] > 0
    assert result["source_stability_verified"] and not result["gate_passed"]
    row = next(r for r in result["files"] if r["path"].endswith("执行.py"))
    assert 2 in row["missing_statements"] and 4 in row["missing_statements"]


def test_full_execution_requires_matching_baseline_not_only_a_percentage(
    checkout, tmp_path
):
    path = source(checkout, "program.py", "result = 1 + 2\nassert result == 3\n")
    baseline = audit.inventory(checkout)
    data = tmp_path / ".coverage"
    execute(path, data)
    assert audit.report(checkout, data, baseline=baseline)["gate_passed"]
    assert not audit.report(checkout, data)["gate_passed"]
    path.write_text(path.read_text().replace("1 + 2", "2 + 2"))
    result = audit.report(checkout, data, baseline=baseline)
    assert result["statement_percent"] == 100
    assert result["source_changes"] == ["program.py"] and not result["gate_passed"]


def test_virtual_generated_code_is_not_guessed_as_original_source_execution(
    checkout, tmp_path
):
    path = source(checkout, "program.py", "result = 3\nassert result == 3\n")
    baseline = audit.inventory(checkout)
    data = tmp_path / ".coverage"
    execute(path, data, virtual=True)
    result = audit.report(checkout, data, baseline=baseline)
    assert result["covered_statements"] == 0 and not result["gate_passed"]
    assert result["virtual_code"] == [
        {
            "filename": "program.py:generated",
            "executed_line_count": 2,
            "credited_to_source": False,
        }
    ]


def test_missing_source_and_external_alias_are_errors(checkout, tmp_path):
    path = source(checkout, "owned.py", "value = 1\n")
    subprocess.run(["git", "-C", str(checkout), "add", "owned.py"], check=True)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        audit.inventory(checkout)
    outside = source(tmp_path, "outside.py", "value = 2\n")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        audit.inventory(checkout)


def test_absent_data_empty_inventory_and_new_file_never_pass_the_gate(
    checkout, tmp_path
):
    data = tmp_path / ".coverage"
    with pytest.raises(FileNotFoundError, match="Combined"):
        audit.report(checkout, data)
    empty = coverage.CoverageData(basename=str(data))
    empty.add_lines({str(tmp_path / "outside.py"): []})
    empty.write()
    baseline = audit.inventory(checkout)
    assert not audit.report(checkout, data, baseline=baseline)["gate_passed"]
    source(checkout, "new.py", "value = 1\n")
    result = audit.report(checkout, data, baseline=baseline)
    assert result["source_changes"] == ["new.py"] and result["missing_statements"] == 1


def test_cli_inventory_and_gate_write_complete_receipts(checkout, tmp_path, capsys):
    path = source(checkout, "program.py", "value = 3\n")
    out = tmp_path / "reports/result.json"
    assert audit.main(["--root", str(checkout), "--output", str(out)]) == 0
    baseline = tmp_path / "baseline.json"
    baseline.write_text(out.read_text())
    data = tmp_path / ".coverage"
    execute(path, data)
    args = [
        "--root",
        str(checkout),
        "--output",
        str(out),
        "--data",
        str(data),
        "--baseline",
        str(baseline),
        "--require-complete",
    ]
    assert audit.main(args) == 0 and json.loads(out.read_text())["gate_passed"]
    source(checkout, "uncovered.py", "value = 4\n")
    assert audit.main(args) == 1 and not json.loads(out.read_text())["gate_passed"]
    with pytest.raises(SystemExit) as caught:
        audit.main(["--output", str(out), "--require-complete"])
    assert caught.value.code == 2
    assert "statement_count" in capsys.readouterr().out


def test_non_owned_measured_file_receives_no_source_credit(checkout, tmp_path):
    path = source(checkout, "program.py", "value = 3\n")
    source(checkout, ".gitignore", "ignored.py\n")
    ignored = source(checkout, "ignored.py", "value = 4\n")
    data = coverage.CoverageData(basename=str(tmp_path / ".coverage"))
    data.add_lines({str(path): [1], str(ignored): [1]})
    data.write()
    result = audit.report(
        checkout, Path(data.data_filename()), baseline=audit.inventory(checkout)
    )
    assert (
        result["source_files"] == 1
        and result["covered_statements"] == 1
        and result["gate_passed"]
    )


def test_multiline_statement_execution_agrees_with_coverage_analysis(
    checkout, tmp_path
):
    path = source(
        checkout,
        "program.py",
        "value = 1\nif (\n    value == 1\n):\n    result = 42\nassert result == 42\n",
    )
    baseline = audit.inventory(checkout)
    data = tmp_path / ".coverage"
    execute(path, data)
    official = coverage.Coverage(data_file=str(data), config_file=False)
    official.set_option("report:exclude_lines", [])
    official.load()
    expected_missing = official.analysis2(str(path))[3]
    assert expected_missing == []
    result = audit.report(checkout, data, baseline=baseline)
    assert result["files"][0]["missing_statements"] == expected_missing
    assert result["gate_passed"]


def test_isaacgym_waiver_is_explicit_and_cannot_change_after_measurement(
    checkout, tmp_path
):
    active = source(checkout, "core.py", "answer = 42\n")
    retired = source(
        checkout,
        "src/protomotions/simulator/isaacgym/simulator.py",
        "raise RuntimeError('retired backend is not executed')\n",
    )
    (checkout / "legacy_alias.py").symlink_to(retired)
    full = audit.inventory(checkout)
    scoped = audit.inventory(checkout, ignore_isaacgym=True)
    data = tmp_path / ".coverage"
    execute(active, data)
    assert full["source_files"] == 2 and scoped["source_files"] == 1
    assert scoped["ignored_backends"] == ["isaacgym"]
    assert [r["path"] for r in scoped["excluded_files"]] == [
        retired.relative_to(checkout).as_posix()
    ]
    assert not audit.report(checkout, data, baseline=full)["gate_passed"]
    assert audit.report(checkout, data, baseline=scoped, ignore_isaacgym=True)[
        "gate_passed"
    ]
    mismatch = audit.report(checkout, data, baseline=full, ignore_isaacgym=True)
    assert not mismatch["scope_stability_verified"] and not mismatch["gate_passed"]
    source(checkout, "src/isaacgym_related.py", "unverified = 3\n")
    actual = audit.report(
        checkout,
        data,
        baseline=audit.inventory(checkout, ignore_isaacgym=True),
        ignore_isaacgym=True,
    )
    assert not actual["gate_passed"] and actual["zero_hit_files"] == [
        "src/isaacgym_related.py"
    ]
