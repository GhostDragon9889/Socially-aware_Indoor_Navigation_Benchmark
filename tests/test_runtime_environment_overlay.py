"""SDK ownership, portable paths, locked input and failed-install guarantees."""
from __future__ import annotations

import json
from pathlib import Path
import sys

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
