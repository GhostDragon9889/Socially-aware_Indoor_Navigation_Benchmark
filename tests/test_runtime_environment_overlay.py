"""SDK ownership, portable paths, locked input and failed-install guarantees."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from social_navigation import runtime_environment as runtime


@pytest.mark.parametrize("content", [
    "torch==2.11.0\n", "isaacsim-core==6.1.0.0\n", "numpy==2.5.1\n",
    "usd-core==25.5\n", "warp_lang==1.16.0\n", "nvidia-cuda-runtime-cu12==12.8.90\n",
    "easydict>=1.13\n", "easydict==1.13; python_version>='3.12'\n",
    "easydict @ https://example.invalid/wheel.whl\n", "-r other.txt\n",
    "easydict==1.13\nEASYDICT==1.13\n", "# empty\n",
    "chumpy==0.70\n",
])
def test_lock_rejects_provider_duplication_and_unlocked_inputs(tmp_path, content):
    lock = tmp_path / "requirements.overlay.lock"
    lock.write_text(content)
    with pytest.raises(RuntimeError):
        runtime.read_overlay_lock(lock)


def test_profile_tampering_is_rejected_before_any_install(tmp_path):
    profile_dir = tmp_path / "env" / runtime.PROFILE_NAME
    profile_dir.mkdir(parents=True)
    content = {"pyproject.toml": "[project]\n", "uv.lock": "version=1\n",
               "requirements.overlay.lock": "easydict==1.13\n"}
    keys = ["manifest_sha256", "lock_sha256", "overlay_lock_sha256"]
    profile = {}
    for (name, text), key in zip(content.items(), keys):
        path = profile_dir / name
        path.write_text(text)
        profile[key] = runtime._sha256(path)
    (profile_dir / "profile.json").write_text(json.dumps(profile))
    assert runtime.load_runtime_profile(tmp_path)[1] == {"easydict": "1.13"}
    (profile_dir / "requirements.overlay.lock").write_text("easydict==1.12\n")
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.load_runtime_profile(tmp_path)


def test_real_python_pth_handles_spaces_quotes_unicode_and_old_sdk_editable(tmp_path):
    # Real isolated Python startup loads both executable .pth lines. SDK .pth
    # processing must preserve SDK source paths while the new project wins over
    # an obsolete editable project already present in that SDK.
    sdk = tmp_path / "SDK's source 空 格"
    sdk.mkdir()
    sdk_source = tmp_path / "actual SDK source"
    sdk_source.mkdir()
    (sdk_source / "overlay_test_sdk.py").write_text("answer = 61\n")
    stale_project = tmp_path / "old project"
    stale_project.mkdir()
    (stale_project / "overlay_test_project.py").write_text("answer = 'old'\n")
    (sdk / "sdk_source.pth").write_text(str(sdk_source) + "\n" + str(stale_project) + "\n")
    project = tmp_path / "new project's 目录 with spaces"
    (project / "src").mkdir(parents=True)
    (project / "src/overlay_test_project.py").write_text("answer = 'new'\n")
    environment = tmp_path / "overlay Python"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site = Path(subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip())
    (site / "social_navigation_sdk_source.pth").write_text(runtime._pth_content(sdk, project))
    result = subprocess.check_output([str(python), "-I", "-c", "import overlay_test_sdk as a, overlay_test_project as b; print(a.answer, b.answer)"], text=True).strip()
    assert result == "61 new"
    assert not (sdk / "social_navigation_sdk_source.pth").exists()


def test_failed_install_preserves_existing_selected_python_and_cleans_candidate(tmp_path, monkeypatch):
    project = tmp_path / "project with spaces"
    project.mkdir()
    sdk_root = tmp_path / "SDK"
    sdk_root.mkdir()
    previous = tmp_path / "previous runtime"
    previous.mkdir()
    (project / ".venv").symlink_to(previous, target_is_directory=True)
    monkeypatch.setattr(runtime, "load_runtime_profile", lambda root: ({}, {"easydict": "1.13"}))
    monkeypatch.setattr(runtime, "_sdk_snapshot", lambda root, profile: {"python": str(sdk_root / ".venv/bin/python")})
    monkeypatch.setattr(runtime, "_fingerprint", lambda *args: {"project_root": str(project)})
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/a path with spaces/uv")
    commands = []
    def fail_command(command, **kwargs):
        commands.append(command)
        assert command[:3] == ["/a path with spaces/uv", "venv", "--relocatable"]
        kwargs['log'].write_text("deliberate build failure\n")
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(runtime, "_run", fail_command)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.prepare_runtime_environment(project, sdk_root)
    assert (project / ".venv").resolve() == previous
    assert len(commands) == 1
    assert not (project / ".runtime" / runtime.PROFILE_NAME).exists()
    assert not list((project / ".runtime").glob("*.build-*"))
    assert "deliberate build failure" in (project / ".runtime" / f".{runtime.PROFILE_NAME}.failed-install.log").read_text()


def test_existing_runtime_rejects_mismatched_sdk_without_changing_selection(tmp_path, monkeypatch):
    project = tmp_path / "project"
    directory = project / ".runtime" / runtime.PROFILE_NAME
    directory.mkdir(parents=True)
    (directory / runtime.RECEIPT_NAME).write_text(json.dumps({"fingerprint": {"sdk_root": "different SDK"}}))
    monkeypatch.setattr(runtime, "load_runtime_profile", lambda root: ({}, {}))
    monkeypatch.setattr(runtime, "_sdk_snapshot", lambda root, profile: {})
    monkeypatch.setattr(runtime, "_fingerprint", lambda *args: {"sdk_root": "current SDK"})
    with pytest.raises(RuntimeError, match="exactly match"):
        runtime.check_runtime_environment(project, tmp_path / "SDK")
    assert json.loads((directory / runtime.RECEIPT_NAME).read_text())["fingerprint"]["sdk_root"] == "different SDK"


def test_real_metadata_probe_rejects_unsatisfied_core_requirement():
    # This is a real target-runtime check, not a simulated package resolver.
    # Generic CPU CI does not install the Isaac SDK and reports it as unavailable.
    from importlib import metadata
    import platform
    try:
        if metadata.version("isaacsim") != "6.1.0.0":
            pytest.skip("requires installed Isaac Sim 6.1 SDK, without launching Kit")
    except metadata.PackageNotFoundError:
        pytest.skip("Isaac SDK metadata unavailable")
    project = Path(__file__).resolve().parents[1]
    profile, pins = runtime.load_runtime_profile(project)
    with pytest.raises(RuntimeError, match="project-core requires numpy>=999"):
        runtime._probe_python(Path(sys.executable), profile, pins=pins, requirements=["numpy>=999"])


def test_runtime_directory_symlink_cannot_write_into_sdk(tmp_path, monkeypatch):
    project = tmp_path / "project"
    sdk = tmp_path / "SDK"
    project.mkdir()
    sdk.mkdir()
    (project / ".runtime").symlink_to(sdk, target_is_directory=True)
    monkeypatch.setattr(runtime, "load_runtime_profile", lambda root: ({}, {}))
    monkeypatch.setattr(runtime, "_sdk_snapshot", lambda root, profile: {})
    monkeypatch.setattr(runtime, "_fingerprint", lambda *args: {})
    with pytest.raises(RuntimeError, match="outside the SDK"):
        runtime.prepare_runtime_environment(project, sdk)
    assert list(sdk.iterdir()) == []
