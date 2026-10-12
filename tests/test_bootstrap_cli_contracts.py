"""Keep setup orchestration bounded and preserve already installed SDKs."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    script = Path(__file__).resolve()
    root = next(p for p in script.parents if (p / ".social-navigation-root").is_file())
    spec = importlib.util.spec_from_file_location(
        "bootstrap_cli_contract", root / "scripts/bootstrap_reproduction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "checkout's 中文"
    profile = project / "env/isaaclab3_ea_sim61/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            {"isaaclab_commit": "pinned fixture commit", "python_recorded": "3.12.13"}
        )
    )
    monkeypatch.setattr(module, "ROOT", project)
    return module, project


def arguments(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["bootstrap_reproduction.py", *args])


@pytest.mark.parametrize(
    "action,name",
    [
        ("check", "check"),
        ("prepare-native", "prepare_native_runtime"),
        ("prepare-frozen", "prepare_frozen_inputs"),
    ],
)
def test_read_and_prepare_dispatch_to_selected_checkout(
    bootstrap, monkeypatch, capsys, action, name
):
    from social_navigation import external_runtime, reproduction

    module, project = bootstrap
    calls = []
    service = external_runtime if action == "check" else reproduction
    monkeypatch.setattr(
        service, name, lambda root: calls.append(root) or {"fixture": action}
    )
    arguments(monkeypatch, action)
    assert module.main() == 0 and calls == [project]
    assert json.loads(capsys.readouterr().out) == {"fixture": action}


@pytest.mark.parametrize("failure", ["missing_lab", "missing_uv", "installed"])
def test_invalid_setup_preserves_existing_files(bootstrap, monkeypatch, failure):
    module, project = bootstrap
    installed = project / ".runtime/IsaacLab61/.venv/bin/python"
    installed.parent.mkdir(parents=True)
    installed.write_text("existing SDK interpreter marker")
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda command: None if failure == "missing_uv" else "uv",
    )
    arguments(
        monkeypatch,
        "install" if failure == "installed" else "configure",
        *(
            []
            if failure == "missing_lab"
            else ["--lab-root", str(installed.parents[2])]
        ),
    )
    with pytest.raises(SystemExit) as exited:
        module.main()
    assert (
        exited.value.code == 2
        and installed.read_text() == "existing SDK interpreter marker"
    )


class Reexecuted(Exception):
    pass


@pytest.mark.parametrize("choice", ["explicit", "lab", "binding", "uv"])
def test_old_host_python_hands_off_only_to_selected_sdk_or_uv(
    bootstrap, monkeypatch, choice
):
    module, project = bootstrap
    lab = project / "SDK"
    sdk = lab / ".venv/bin/python"
    sdk.parent.mkdir(parents=True)
    sdk.touch()
    supplied = ["--lab-root", str(lab)]
    action = "configure"
    if choice == "explicit":
        supplied.extend(["--python", str(sdk)])
    if choice == "binding":
        action, supplied = "check", []
        binding = project / ".runtime/external_sdk.json"
        binding.parent.mkdir()
        binding.write_text(json.dumps({"python": str(sdk)}))
    if choice == "uv":
        supplied = []
    arguments(monkeypatch, action, *supplied)
    monkeypatch.setattr(
        module, "sys", SimpleNamespace(version_info=(3, 10), argv=list(sys.argv))
    )
    monkeypatch.setenv("SOCIAL_NAVIGATION_BOOTSTRAP_REEXEC", "0")
    monkeypatch.setattr(module.shutil, "which", lambda command: "uv")
    calls = []

    def reexecute(path, args):
        calls.append((path, args))
        raise Reexecuted

    monkeypatch.setattr(module.os, "execv", reexecute)
    with pytest.raises(Reexecuted):
        module.main()
    path, command = calls[0]
    if choice == "uv":
        assert path == "uv" and command[:6] == [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12.13",
            "python",
        ]
    else:
        assert path == str(sdk) and command[:2] == [str(sdk), module.__file__]
        assert module.os.environ["SOCIAL_NAVIGATION_BOOTSTRAP_REEXEC"] == "1"
    assert command[-len(supplied) :] == supplied if supplied else command[-1] == action


@pytest.mark.parametrize("failure", ["loop", "missing_uv"])
def test_old_interpreter_fails_closed_without_recursion(
    bootstrap, monkeypatch, failure
):
    module, _ = bootstrap
    arguments(monkeypatch, "install")
    monkeypatch.setattr(
        module, "sys", SimpleNamespace(version_info=(3, 10), argv=list(sys.argv))
    )
    monkeypatch.setenv(
        "SOCIAL_NAVIGATION_BOOTSTRAP_REEXEC", "1" if failure == "loop" else "0"
    )
    monkeypatch.setattr(module.shutil, "which", lambda command: None)
    with pytest.raises(SystemExit) as exited:
        module.main()
    assert exited.value.code == 2


@pytest.mark.parametrize("action", ["configure", "install"])
@pytest.mark.parametrize("native_exit", [0, 7])
@pytest.mark.parametrize("existing_checkout", [False, True])
def test_setup_command_transaction_and_native_status(
    bootstrap, monkeypatch, capsys, action, native_exit, existing_checkout
):
    from social_navigation import external_runtime, reproduction

    module, project = bootstrap
    lab = project / ".runtime/IsaacLab61"
    if existing_checkout:
        lab.mkdir(parents=True)
    monkeypatch.setenv("SOCIAL_NAVIGATION_SDK_PROFILE", "fixture_previous_profile")
    commands, events = [], []
    monkeypatch.setattr(module.shutil, "which", lambda command: "uv")

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs == {"check": True}
        if command[1] == "clone":
            lab.mkdir(parents=True)

    monkeypatch.setattr(module.subprocess, "run", execute)
    installer = SimpleNamespace(
        require_checkout=lambda value: events.append(("checkout", value)),
        apply_profile=lambda value, **kwargs: events.append(("profile", value, kwargs)),
    )
    original_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: (
            installer if name == "isaaclab3_environment" else original_import(name)
        ),
    )
    monkeypatch.setattr(
        external_runtime,
        "configure",
        lambda root, selected, **kwargs: (
            events.append(("configure", root, selected, kwargs))
            or {"status": "configured fixture"}
        ),
    )
    monkeypatch.setattr(
        reproduction, "prepare_frozen_inputs", lambda root: {"files": 17}
    )
    monkeypatch.setattr(
        external_runtime,
        "launch",
        lambda root, args: events.append(("native", root, args)) or native_exit,
    )
    arguments(monkeypatch, action, "--lab-root", str(lab), "--rebuild-addons")
    assert module.main() == native_exit
    assert events[-2][0] == "configure" and events[-2][-1] == {
        "python": None,
        "rebuild_addons": True,
    }
    assert events[-1] == ("native", project, [module.__file__, "prepare-native"])
    if action == "install":
        if not existing_checkout:
            assert commands[0][:4] == [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
            ]
            assert commands[1][-2:] == ["--detach", "pinned fixture commit"]
        assert commands[-1] == ["uv", "python", "install", "3.12.13"]
        assert len(commands) == (1 if existing_checkout else 3)
        assert events[1] == ("profile", lab, {"sync": True, "fresh": True})
    else:
        assert commands == []
    if native_exit == 0:
        report = json.loads(capsys.readouterr().out)
        assert (
            report["frozen_inputs"] == {"files": 17} and not report["datasets_restored"]
        )
    else:
        assert capsys.readouterr().out == ""


def test_public_bootstrap_module_help_needs_no_runtime_binding():
    root = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / ".social-navigation-root").is_file()
    )
    child = subprocess.run(
        [sys.executable, str(root / "scripts/bootstrap_reproduction.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "install,configure,check,prepare-frozen,prepare-native" in child.stdout
    assert "--lab-root" in child.stdout and "--python" in child.stdout
