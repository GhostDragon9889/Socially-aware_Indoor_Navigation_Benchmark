"""SDK admission checks must reject incompatible providers before binding."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from social_navigation import reproduction, runtime_environment as runtime
from social_navigation import resource_bindings, sdk_paths


def test_logged_process_uses_argument_boundaries_and_keeps_failure_log(tmp_path):
    log = tmp_path / "process's 中文.log"
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", "value's 空间"]
    assert runtime._run(command, cwd=tmp_path, log=log) == ""
    lines = log.read_text().splitlines()
    assert json.loads(lines[0]) == command and lines[1] == "value's 空间"
    with pytest.raises(subprocess.CalledProcessError):
        runtime._run([sys.executable, "-c", "raise SystemExit(3)"], log=log)
    assert "raise SystemExit(3)" in log.read_text()


@pytest.mark.parametrize("defect", [None, "python", "platform", "machine", "provider"])
def test_probe_validates_interpreter_and_passes_addons_without_shell(
    tmp_path, monkeypatch, defect
):
    profile = {
        "isaaclab_package_version": "17.0.2",
        "isaacsim": "6.1.0.0",
        "torch": "2.11.0+cu128",
        "numpy": "2.5.1",
        "warp": "1.16.0",
        "usd_exchange": "2.3.0",
        "python_recorded": "3.12.13",
        "platform": "linux",
        "machine": "x86_64",
    }
    expected = {
        "python_version": "3.12.13",
        "platform": "linux",
        "machine": "x86_64",
        "errors": [],
    }
    if defect in ("python", "platform", "machine"):
        expected["python_version" if defect == "python" else defect] = "incompatible"
    if defect == "provider":
        expected["errors"] = ["SDK provider bytes differ"]
    calls = []

    def execute(command):
        calls.append(command)
        return json.dumps(expected)

    monkeypatch.setattr(runtime, "_run", execute)
    addons = tmp_path / "addon's 中文"
    if defect:
        with pytest.raises(RuntimeError, match="Runtime verification failed"):
            runtime._probe_python(
                Path(sys.executable), profile, pins={"pytest": "9.0.3"}, addons=addons
            )
    else:
        assert (
            runtime._probe_python(
                Path(sys.executable), profile, pins={"pytest": "9.0.3"}, addons=addons
            )
            == expected
        )
    assert calls[0][:3] == [sys.executable, "-I", "-c"]
    assert calls[0][-1] == str(addons)
    arguments = json.loads(calls[0][-2])
    assert arguments["addons_site"] == str(addons)
    assert arguments["overlay_pins"] == {"pytest": "9.0.3"}
    assert arguments["versions"]["numpy"] == "2.5.1"


@pytest.fixture
def sdk_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "SDK's 中文"
    root.mkdir()
    (root / "source").mkdir()
    (root / "source/module.py").write_text("answer = 42\n")
    (root / "VERSION").write_text("3.0.0\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    profile = {
        "isaaclab_commit": runtime._run(
            ["git", "-C", str(root), "rev-parse", "HEAD"]
        ).strip(),
        "isaaclab_source_tree": runtime._run(
            ["git", "-C", str(root), "rev-parse", "HEAD:source"]
        ).strip(),
        "isaaclab_product_version": "3.0.0",
    }
    site = root / ".venv/lib/python3.12/site-packages"
    probe = {
        "site_packages": str(site),
        "providers": {"isaacsim": str(site / "isaacsim.dist-info")},
        "module_providers": {"pxr": str(site / "pxr")},
    }
    monkeypatch.setattr(runtime, "_probe_python", lambda *args: deepcopy(probe))
    return root, python, profile, probe


def test_real_git_sdk_snapshot_accepts_exact_source_in_another_commit(sdk_snapshot):
    root, python, profile, _ = sdk_snapshot
    exact = runtime._sdk_snapshot(root, profile)
    assert (
        exact["python"] == str(python)
        and exact["source_tree"] == profile["isaaclab_source_tree"]
    )
    (root / "README.md").write_text("unrelated SDK documentation\n")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "docs",
        ],
        check=True,
    )
    accepted = runtime._sdk_snapshot(root, profile)
    assert (
        accepted["head"] != exact["head"]
        and accepted["source_tree"] == exact["source_tree"]
    )


@pytest.mark.parametrize(
    "defect", ["missing_python", "source", "commit", "version", "provider", "module"]
)
def test_sdk_snapshot_rejects_missing_dirty_or_escaped_sdk(sdk_snapshot, defect):
    root, python, profile, probe = sdk_snapshot
    if defect == "missing_python":
        python.unlink()
    elif defect == "source":
        (root / "source/module.py").write_text("changed = True\n")
    elif defect == "commit":
        profile.update(isaaclab_commit="0" * 40, isaaclab_source_tree="0" * 40)
    elif defect == "version":
        (root / "VERSION").write_text("2.0.0\n")
    else:
        probe["providers" if defect == "provider" else "module_providers"][
            "escaped"
        ] = str(root / "foreign")
    with pytest.raises(RuntimeError, match="missing|differs|escaped"):
        runtime._sdk_snapshot(root, profile)


def test_fingerprint_records_all_visible_versions_not_metadata_locations(tmp_path):
    profile_dir = tmp_path / "env" / runtime.PROFILE_NAME
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.json").write_text("{}\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    profile = {"overlay_lock_sha256": "fixture lock"}
    sdk = {
        "root": "sdk",
        "python": "sdk/python",
        "source_tree": "tree",
        "manifest_sha256": "manifest",
        "lock_sha256": "lock",
        "probe": {
            "python_version": "3.12.13",
            "visible_packages": {
                "torch": {"version": "2.11.0"},
                "extra": {"version": "1.0"},
            },
        },
    }
    first = runtime._fingerprint(tmp_path, profile, sdk)
    assert first["sdk_visible_versions"] == {"torch": "2.11.0", "extra": "1.0"}
    sdk["probe"]["visible_packages"]["extra"]["version"] = "2.0"
    assert runtime._fingerprint(tmp_path, profile, sdk) != first


def test_bad_resource_and_frozen_input_schemas_are_rejected(tmp_path):
    config = tmp_path / "configs/reproduction"
    config.mkdir(parents=True)
    (config / "frozen_inputs.json").write_text('{"schema":"unknown"}')
    with pytest.raises(ValueError, match="schema"):
        reproduction.prepare_frozen_inputs(tmp_path)
    (config / "resource_bindings.json").write_text('{"schema":"unknown"}')
    resource_bindings._bindings.cache_clear()
    with pytest.raises(ValueError, match="schema"):
        resource_bindings.resolve_path("project:data/fixture", root=tmp_path)
    resource_bindings._bindings.cache_clear()


def test_native_runtime_admits_real_sim61_without_launching_kit(tmp_path, monkeypatch):
    report = reproduction.prepare_native_runtime(tmp_path)
    assert report["status"] == "stock_sim61_runtime"
    assert report["runtime_versions"]["isaacsim"] == "6.1.0.0"
    assert not report["physics_rerun"] and not report["installed_SDK_changed"]
    monkeypatch.setattr("importlib.metadata.version", lambda name: "6.0.0.0")
    with pytest.raises(RuntimeError, match="Only Isaac Sim 6.1"):
        reproduction.prepare_native_runtime(tmp_path)


def test_explicit_lab_override_must_be_a_source_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("ISAACLAB_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="pinned source checkout"):
        sdk_paths.lab_root()


def test_checkout_without_resource_registry_still_resolves_portable_paths(tmp_path):
    resource_bindings._bindings.cache_clear()
    assert (
        resource_bindings.resolve_path("project:data/fixture", root=tmp_path)
        == tmp_path / "data/fixture"
    )
    assert resource_bindings._bindings(str(tmp_path)) == {}
    resource_bindings._bindings.cache_clear()
