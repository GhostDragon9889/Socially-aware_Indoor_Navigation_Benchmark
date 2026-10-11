"""Installation boundaries exercised through actual standalone CLIs and metadata."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("mujoco", "newton", "isaacgym", "isaaclab", "isaaclab6", "genesis")


def test_all_simulator_extras_are_pairwise_exclusive_and_lab_sources_are_pinned():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    conflicts = metadata["tool"]["uv"]["conflicts"]
    actual = {frozenset(item["extra"] for item in pair) for pair in conflicts}
    assert actual == {frozenset(pair) for pair in itertools.combinations(BACKENDS, 2)}
    assert len(conflicts) == len(actual) == 15
    sources = {item["extra"]: item for item in metadata["tool"]["uv"]["sources"]["isaaclab"]}
    for extra, profile in (("isaaclab", "isaaclab3_ea_sim61"), ("isaaclab6", "isaaclab3_sim6")):
        source = sources[extra]
        manifest = json.loads((ROOT / "env" / profile / "profile.json").read_text())
        assert source["git"] == "https://github.com/isaac-sim/IsaacLab.git"
        assert source["rev"] == manifest["isaaclab_commit"]
        assert source["subdirectory"] == "source/isaaclab"
    assert metadata["tool"]["uv"]["environments"] == ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
    target = metadata["project"]["optional-dependencies"]["isaaclab"]
    assert any(Requirement(text).name == "tinyobjloader" and "2.0.0rc13" in Requirement(text).specifier for text in target)
    python = SpecifierSet(metadata["project"]["requires-python"])
    assert "3.12.13" in python and "3.13" not in python


def test_sdk_newton_provider_is_not_overwritten_by_historical_extra():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [Requirement(text) for text in metadata["project"]["optional-dependencies"]["newton"]]
    newton = next(item for item in requirements if item.name == "newton")
    assert newton.extras == {"sim"}
    assert "1.0.0" in newton.specifier
    locked = (ROOT / "env/isaaclab3_ea_sim61/requirements.overlay.lock").read_text()
    assert not any(Requirement(line).name.lower() in {"newton", "usd-core", "warp-lang", "numpy", "torch"}
                   for line in locked.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def test_moviepy_metadata_allows_both_supported_writer_apis_without_changing_sdk_lock():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    moviepy = next(Requirement(text) for text in metadata["project"]["dependencies"]
                   if Requirement(text).name == "moviepy")
    assert "1.0.3" in moviepy.specifier and "2.2.1" in moviepy.specifier
    assert "3.0" not in moviepy.specifier
    overlay = (ROOT / "env/isaaclab3_ea_sim61/requirements.overlay.lock").read_text()
    assert "moviepy==1.0.3" in overlay.splitlines()


def test_legacy_bootstrap_install_rejected_before_sdk_or_network_mutation(tmp_path):
    sdk = tmp_path / "SDK's 旧 source with spaces"
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/bootstrap_reproduction.py"),
                             "install", "--profile", "isaaclab3_sim6", "--lab-root", str(sdk)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "frozen Sim6.0 profile" in result.stderr
    assert not sdk.exists()


def test_existing_sdk_install_rejected_and_all_bytes_preserved(tmp_path):
    sdk = tmp_path / "SDK's existing 空 格"
    python = sdk / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"SDK executable sentinel; must not execute or replace")
    manifest = sdk / "pyproject.toml"
    manifest.write_bytes(b"SDK manifest sentinel")
    before = {path.relative_to(sdk): path.read_bytes() for path in sdk.rglob("*") if path.is_file()}
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/bootstrap_reproduction.py"),
                             "install", "--profile", "isaaclab3_ea_sim61", "--lab-root", str(sdk)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "installed SDK already exists" in result.stderr
    assert before == {path.relative_to(sdk): path.read_bytes() for path in sdk.rglob("*") if path.is_file()}


def test_legacy_apply_rejected_without_creating_destination(tmp_path):
    sdk = tmp_path / "old SDK must remain absent"
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/isaaclab3_environment.py"),
                             "--profile", "isaaclab3_sim6", "apply", "--lab-root", str(sdk)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "frozen Sim6.0 manifest" in result.stderr
    assert not sdk.exists()


@pytest.mark.parametrize("name,key", [("pyproject.toml", "manifest_sha256"), ("uv.lock", "lock_sha256")])
def test_frozen_sdk_manifests_still_match_original_receipts(name, key):
    for profile in ("isaaclab3_sim6", "isaaclab3_ea_sim61"):
        directory = ROOT / "env" / profile
        receipt = json.loads((directory / "profile.json").read_text())
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == receipt[key]
