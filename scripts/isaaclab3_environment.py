#!/usr/bin/env python3
"""Apply or verify the repository's existing IsaacLab 3 / Sim 6 environment.

The apply command requires the pinned IsaacLab checkout and an already installed
Isaac Sim environment. It never launches Kit or installs a different Sim version.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "env" / "isaaclab3_sim6"
PROFILE = json.loads((PROFILE_DIR / "profile.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_checkout(lab_root: Path) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=lab_root, text=True
    ).strip()
    source_tree = subprocess.check_output(["git", "rev-parse", "HEAD:source"], cwd=lab_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "source"], cwd=lab_root, text=True).strip()
    if dirty or (head != PROFILE["isaaclab_commit"] and source_tree != PROFILE.get("isaaclab_source_tree")):
        raise RuntimeError(f"IsaacLab commit is {head}; expected {PROFILE['isaaclab_commit']}")
    if (lab_root / "VERSION").read_text().strip() != PROFILE["isaaclab_product_version"]:
        raise RuntimeError("IsaacLab product VERSION differs from the profile")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This environment profile supports Linux x86_64 only")


def installed_sim(python: Path) -> str:
    return subprocess.check_output(
        [str(python), "-I", "-c", "from importlib.metadata import version; print(version('isaacsim'))"],
        text=True,
    ).strip()


def check_current() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run check with the target .venv's Python 3.12")
    expected = {
        "isaaclab": PROFILE["isaaclab_package_version"],
        "isaacsim": PROFILE["isaacsim"],
        "torch": PROFILE["torch"],
        "usd-exchange": PROFILE["usd_exchange"],
    }
    versions = {name: metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Package versions differ: {versions}; expected {expected}")
    try:
        metadata.version("usd-core")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("usd-core must not be installed beside usd-exchange")

    provider = metadata.distribution("usd-exchange")
    checked = 0
    mismatches = []
    for item in provider.files or []:
        if not item.hash or item.hash.mode != "sha256":
            continue
        if not str(item).startswith(("pxr/", "usdex/", "usd_exchange.libs/")):
            continue
        checked += 1
        path = Path(provider.locate_file(item))
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
            if path.is_file() else None
        )
        if actual != item.hash.value:
            mismatches.append(str(path))
    if not checked or mismatches:
        raise RuntimeError(f"USD provider RECORD check failed: {mismatches}")

    from pxr import Sdf, Usd, UsdGeom
    import usdex.core
    import mujoco_usd_converter
    import urdf_usd_converter

    runtime = list(Usd.GetVersion())
    if runtime != PROFILE["standalone_usd_version"]:
        raise RuntimeError(f"Standalone USD runtime is {runtime}")
    with tempfile.TemporaryDirectory(prefix="crowdsim-usd-check-") as folder:
        path = str(Path(folder) / "smoke.usda")
        stage = Usd.Stage.CreateNew(path)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Cube.Define(stage, "/World/Cube")
        stage.GetRootLayer().Save()
        restored = Usd.Stage.Open(Sdf.Layer.FindOrOpen(path))
        if restored.GetPrimAtPath("/World/Cube").GetTypeName() != "Cube":
            raise RuntimeError("USD/Sdf file round trip failed")
    return {
        "versions": versions,
        "standalone_usd_version": runtime,
        "usd_binary": str(provider.locate_file("pxr/Usd/_usd.so")),
        "record_files_checked": checked,
        "usdex_version": usdex.core.version(),
        "mujoco_converter": mujoco_usd_converter.__version__,
        "urdf_converter": urdf_usd_converter.__version__,
        "usd_round_trip": "PASS",
        "kit_launched": False,
    }


def apply_profile(lab_root: Path, sync: bool, *, fresh: bool = False) -> None:
    require_checkout(lab_root)
    python = lab_root / ".venv" / "bin" / "python"
    if not fresh and (not python.is_file() or installed_sim(python) != PROFILE["isaacsim"]):
        raise RuntimeError("This command requires an existing .venv with Isaac Sim 6.0.0.1 installed")
    for name, key in [("pyproject.toml", "manifest_sha256"), ("uv.lock", "lock_sha256")]:
        if sha256(PROFILE_DIR / name) != PROFILE[key]:
            raise RuntimeError(f"Stored profile {name} changed without updating its hash")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    backup = lab_root / ".crowdsim-environment-backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for name in ["pyproject.toml", "uv.lock"]:
        if (lab_root / name).exists():
            shutil.copy2(lab_root / name, backup / name)
        content = (PROFILE_DIR / name).read_text()
        # The source snapshot is relative to the IsaacLab checkout. Only the
        # downstream repository location changes when the checkout is elsewhere.
        proto_relative = os.path.relpath(REPO_ROOT, lab_root)
        content = content.replace(json.dumps(PROFILE.get("project_editable_path", "../ProtoMotions")), json.dumps(proto_relative))
        tomllib.loads(content)
        (lab_root / name).write_text(content)
    print(f"Profile applied; prior manifest/lock saved to {backup}")
    if sync:
        subprocess.run(
            ["uv", "sync", "--locked", "--python", PROFILE["python_recorded"], "--extra", "isaacsim", "--reinstall-package", "usd-exchange"],
            cwd=lab_root,
            check=True,
        )
        subprocess.run([str(python), "-I", str(Path(__file__).resolve()), "check"], check=True)
    else:
        print("Environment packages have not changed. Use apply --sync to synchronize.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verify installed packages, USD hashes and file read/write")
    apply = commands.add_parser("apply", help="Apply the recorded workspace manifest and lock")
    apply.add_argument("--lab-root", type=Path, default=REPO_ROOT.parent / "IsaacLab-Protomotions")
    apply.add_argument("--sync", action="store_true", help="Synchronize existing environment and restore USD files")
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(check_current(), ensure_ascii=False, indent=2))
    else:
        apply_profile(args.lab_root.resolve(), args.sync)


if __name__ == "__main__":
    main()
