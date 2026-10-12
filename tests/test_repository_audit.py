"""Static audit must report malformed files, reference drift and lint failures."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_repository.py"
spec = importlib.util.spec_from_file_location("repository_audit_contract", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def checkout(tmp_path):
    root, reference = tmp_path / "checkout", tmp_path / "reference"
    (root / "src/protomotions").mkdir(parents=True)
    (reference / "protomotions").mkdir(parents=True)
    source = "import math\nfrom pathlib import Path\nclass Agent: pass\nasync def run(): return Path(str(math.pi))\n"
    (root / "src/protomotions/agent.py").write_text(source)
    (reference / "protomotions/agent.py").write_text(source)
    (reference / "protomotions/removed.py").write_text("VALUE = 1\n")
    (reference / "old.json").write_text("{}\n")
    return root, reference


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/protomotions/core.py", "protomotions/core.py"),
        ("src/CrowdSim/run.py", "CrowdSim/run.py"),
        ("src/deployment/run.py", "deployment/run.py"),
        ("configs/crowdsim/task.yaml", "CrowdSim/config/task.yaml"),
        ("tools/retargeting/motion.py", "pyroki/motion.py"),
        ("tools/asset_conversion/convert.py", "usd_convert/convert.py"),
        ("tests/test_contract.py", "tests/test_contract.py"),
    ],
)
def test_reference_paths_match_canonical_layout(path, expected):
    assert audit.reference_relative(path) == expected


@pytest.mark.parametrize(
    "extension,valid,invalid",
    [
        ("py", "def f(): return 1\n", "def broken(:\n"),
        ("sh", "#!/bin/bash\nprintf 'ok\\n'\n", "if true; then\n"),
        ("yaml", "items: [1, 2]\n", "items: [\n"),
        ("yml", "enabled: true\n", "a: {\n"),
        ("toml", "enabled = true\n", "enabled = [\n"),
        ("json", '{"enabled":true}\n', "{broken\n"),
    ],
)
def test_inventory_checks_each_supported_format_and_keeps_failures(
    checkout, extension, valid, invalid
):
    root, reference = checkout
    (root / f"valid.{extension}").write_text(valid)
    (root / f"invalid.{extension}").write_text(invalid)
    hidden = root / "src/.venv/ignored.py"
    hidden.parent.mkdir()
    hidden.write_text("invalid syntax\n")
    rows = {row["path"]: row for row in audit.inventory(root, reference)}
    assert len(rows) == 3 and "src/.venv/ignored.py" not in rows
    assert rows[f"valid.{extension}"]["syntax_status"] == "passed"
    assert rows[f"invalid.{extension}"]["syntax_status"] == "failed"
    assert rows[f"invalid.{extension}"]["syntax_error"]
    source = rows["src/protomotions/agent.py"]
    assert source["reference_status"] == "identical" and source["definitions"] == 2
    assert source["imports"] == ["math", "pathlib"]
    (reference / "protomotions/agent.py").write_text("CHANGED = True\n")
    rows = {row["path"]: row for row in audit.inventory(root, reference)}
    assert rows["src/protomotions/agent.py"]["reference_status"] == "different"


@pytest.mark.parametrize("fault", [None, "syntax", "lint"])
def test_real_cli_writes_json_csv_and_returns_nonzero_on_findings(checkout, fault):
    root, reference = checkout
    if fault:
        (root / "src/failure.py").write_text(
            "bad +\n" if fault == "syntax" else "missing_name()\n"
        )
    output = root / "output/audit"
    child = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--reference",
            str(reference),
            "--output-dir",
            "output/audit",
        ],
        text=True,
        capture_output=True,
    )
    assert child.returncode == (0 if fault is None else 1), child.stderr
    payload = json.loads((output / "source_inventory.json").read_text())
    assert payload["reference_only_files"] == ["old.json", "protomotions/removed.py"]
    assert payload["summary"]["ruff_status"] == "completed"
    rows = list(csv.DictReader((output / "source_inventory.csv").open()))
    assert len(rows) == payload["summary"]["files"]
    if fault == "lint":
        assert payload["summary"]["ruff_findings"] == 1
        assert (
            next(row for row in rows if row["path"] == "src/failure.py")[
                "ruff_findings"
            ]
            == "1"
        )


def test_cli_rejects_missing_reference_core(checkout):
    root, _ = checkout
    child = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--reference",
            str(root / "missing"),
        ],
        text=True,
        capture_output=True,
    )
    assert child.returncode == 2 and "Reference core is unavailable" in child.stderr
    assert not (root / "output").exists()


def test_lint_unavailable_still_records_syntax_results(checkout, monkeypatch, tmp_path):
    root, reference = checkout
    output = tmp_path / "absolute-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(root),
            "--reference",
            str(reference),
            "--output-dir",
            str(output),
        ],
    )
    monkeypatch.setattr(audit.shutil, "which", lambda _: None)
    assert audit.main() == 0
    assert (
        json.loads((output / "source_inventory.json").read_text())["summary"][
            "ruff_status"
        ]
        == "unavailable"
    )


def test_broken_linter_is_not_accepted_as_clean(checkout, monkeypatch):
    root, reference = checkout
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--root", str(root), "--reference", str(reference)]
    )
    monkeypatch.setattr(audit.shutil, "which", lambda _: "ruff")
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 2, "", "broken linter"
        ),
    )
    with pytest.raises(RuntimeError, match="Ruff did not complete: broken linter"):
        audit.main()
    assert not (root / "output").exists()
