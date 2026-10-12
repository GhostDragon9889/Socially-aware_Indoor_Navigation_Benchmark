"""Unified commands retain SDK argv, process status and preflight ownership."""

import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from social_navigation import cli, preflight


@pytest.fixture
def launch(tmp_path, monkeypatch):
    calls = []
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    monkeypatch.setattr(cli, "repository_root", lambda: root)

    def popen(command, **options):
        captured = {
            **options,
            "env": {
                key: options["env"][key]
                for key in ("PYTHONDONTWRITEBYTECODE", "PYTHONPATH")
            },
        }
        calls.append((command, captured))
        if "--output" in command:
            selected = Path(command[command.index("--output") + 1])
            if not selected.is_absolute():
                selected = options["cwd"] / selected
            selected.mkdir(parents=True)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(
        preflight,
        "verify_native_scene_config",
        lambda path: {"status": "passed", "scene": path.stem},
    )
    return root, calls


@pytest.mark.parametrize("action,filename", list(cli.TOOLS.items()))
def test_forwarding_preserves_argument_boundaries_and_interpreter(
    launch, action, filename
):
    root, calls = launch
    assert (
        cli.main([action, "--", "--config", "a path with spaces", "--seed", "19"]) == 0
    )
    command, options = calls[0]
    assert command == [
        cli.sys.executable,
        str(root / "scripts" / filename),
        "--config",
        "a path with spaces",
        "--seed",
        "19",
    ]
    assert options["cwd"] == root and options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert options["env"]["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(root / "src"),
        str(root),
    ]


@pytest.mark.parametrize(
    "action,module",
    [
        ("baseline-train", "crowdsim_ppo.train"),
        ("baseline-evaluate", "crowdsim_ppo.evaluate"),
    ],
)
def test_baseline_commands_select_the_owned_python_module(launch, action, module):
    _, calls = launch
    assert cli.main([action, "--output", "result directory"]) == 0
    assert calls[0][0] == [
        cli.sys.executable,
        "-m",
        module,
        "--output",
        "result directory",
    ]


@pytest.mark.parametrize("action", ["simulate", "preflight"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_scene_launch_and_preflight_receipt_belong_to_the_selected_output(
    launch, action, explicit_output, tmp_path
):
    root, calls = launch
    output = tmp_path / "specific output"
    argv = [action, "--scene", "ThirdFloor", "--steps", "120", "--seed", "42"]
    if explicit_output:
        argv += ["--output", str(output)]
    assert cli.main(argv) == 0
    command = calls[0][0]
    actual = Path(command[command.index("--output") + 1])
    assert (
        actual == output
        if explicit_output
        else actual.parent == root / "output/interactive"
    )
    assert command[command.index("--steps") + 1] == "120"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--scene") + 1].endswith("ThirdFloor_scene.yaml")
    if action == "preflight":
        assert "--preflight-only" in command
        assert (
            json.loads((actual / "scene_resources.json").read_text())["scene"]
            == "ThirdFloor_scene"
        )
    else:
        assert "--" in command and command[1].endswith("run_trace_pace_verified.py")
        assert not (actual / "scene_resources.json").exists()


def test_failed_child_exit_never_claims_preflight_pass(launch, monkeypatch, tmp_path):
    _, calls = launch
    output = tmp_path / "failed output"
    monkeypatch.setattr(
        cli.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(wait=lambda: 7)
    )
    assert cli.main(["preflight", "--output", str(output)]) == 7
    assert not output.exists() and not calls


@pytest.mark.parametrize(
    "argv",
    [["status", "--unknown"], ["preflight", "--unknown"], ["simulate", "--unknown"]],
)
def test_non_forwarding_commands_reject_unrecognized_arguments(launch, argv):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2 and not launch[1]


@pytest.mark.parametrize("available", [False, True])
def test_status_reports_actual_interpreter_and_validation_document(
    launch, capsys, available
):
    root, calls = launch
    if available:
        (root / "docs").mkdir()
        (root / "docs/validation.md").write_text("Current validation")
    assert cli.main(["status"]) == 0 and not calls
    status = json.loads(capsys.readouterr().out)
    assert status["python"] == cli.sys.executable
    assert status["root"] == str(root) and status["release_available"] is available


def test_module_entry_point_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["social-navigation", "--help"])
    with pytest.raises(SystemExit) as caught:
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            runpy.run_module("social_navigation.cli", run_name="__main__")
    assert caught.value.code == 0
    assert "preflight" in capsys.readouterr().out
