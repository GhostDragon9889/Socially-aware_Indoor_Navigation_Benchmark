"""Build and verify the project dependency overlay without modifying Isaac's SDK.

The SDK owns Torch, NumPy, Warp, OpenUSD and Isaac libraries. Only the exact
project overlay lock is installed, with dependency resolution disabled. Both
SDK and project default dependency requirements are recorded; this check does
not claim that simulation, training or GPU tests have run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

PROFILE_NAME = "isaaclab3_ea_sim61"
RECEIPT_NAME = "runtime_receipt.json"
_PROTECTED_EXACT = {
    "torch", "torchaudio", "torchvision", "numpy", "warp-lang", "usd-core",
    "usd-exchange", "newton", "mujoco", "mujoco-warp", "omni-usd",
    "isaaclab", "isaacsim",
}
_LOCK_LINE = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: str | Path, root: str | Path) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def read_overlay_lock(path: Path) -> dict[str, str]:
    """Accept only unique exact package pins and reject SDK provider installs."""
    pins: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = _LOCK_LINE.fullmatch(text)
        if not match:
            raise RuntimeError(f"{path}:{number}: require an exact name==version pin")
        name, version = match.groups()
        name = _canonical(name)
        if name in pins:
            raise RuntimeError(f"Duplicate overlay pin: {name}")
        if (name in _PROTECTED_EXACT or name.startswith(("isaacsim-", "isaaclab-", "nvidia-", "usd-"))):
            raise RuntimeError(f"The SDK must provide {name}; it cannot be installed in the overlay")
        pins[name] = version
    if not pins:
        raise RuntimeError("The overlay lock is empty")
    if "chumpy" in pins and not {"pip", "wheel"}.issubset(pins):
        raise RuntimeError("chumpy's build requires locked pip and wheel packages")
    return pins


def load_runtime_profile(project_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the checked-in profile and all install inputs before mutation."""
    directory = project_root / "env" / PROFILE_NAME
    profile = json.loads((directory / "profile.json").read_text())
    for filename, key in (("pyproject.toml", "manifest_sha256"), ("uv.lock", "lock_sha256"),
                          ("requirements.overlay.lock", "overlay_lock_sha256")):
        expected = profile.get(key)
        if not isinstance(expected, str) or _sha256(directory / filename) != expected:
            raise RuntimeError(f"Profile {filename} does not match its recorded SHA256 ({key})")
    pins = read_overlay_lock(directory / "requirements.overlay.lock")
    return profile, pins


def _run(command: list[str], *, cwd: Path | None = None, log: Path | None = None) -> str:
    # Lists are essential here: checkout and runtime paths can contain spaces,
    # quotes and non-ASCII characters. Never interpolate them into a shell.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    if log is None:
        result = subprocess.run(command, cwd=cwd, env=environment, check=True,
                                text=True, capture_output=True)
        return result.stdout
    with log.open("a") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=cwd, env=environment, check=True,
                       stdout=stream, stderr=subprocess.STDOUT, text=True)
    return ""


# Executed by the selected Python interpreter, never by an Isaac/Kit launcher.
# Imports metadata, packaging and top-level module specifications only. Looking
# up pxr's namespace does not load any pxr binary or initialize Kit/GPU.
_PROBE = r'''
import base64, hashlib, importlib.util, json, platform, sys, sysconfig
from importlib import metadata as md
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
args=json.loads(sys.argv[1]); local=Path(sysconfig.get_paths()['purelib']).resolve()
expected=args['versions']; versions={}; providers={}; errors=[]
for name, wanted in expected.items():
 try:
  dist=md.distribution(name); versions[name]=dist.version
  providers[name]=str(Path(dist._path).resolve())
  if dist.version!=wanted:errors.append(f'{name}=={dist.version}; expected {wanted}')
 except md.PackageNotFoundError:errors.append(f'missing SDK package {name}')
for name in args['excluded']:
 try:md.version(name)
 except md.PackageNotFoundError:pass
 else:errors.append(f'conflicting USD provider {name} is installed')
module_providers={}
for name in ('torch','numpy','warp','pxr','usdex'):
 spec=importlib.util.find_spec(name)
 if spec is None:errors.append(f'module {name} unavailable')
 else:
  origin=spec.origin or next(iter(spec.submodule_search_locations or []), '')
  module_providers[name]=str(Path(origin).resolve())
# Check actual SDK USD ownership bytes; do not import a second binary provider.
usd_checked=0; usd_mismatches=[]
try:
 dist=md.distribution('usd-exchange')
 for item in dist.files or []:
  if not item.hash or item.hash.mode!='sha256':continue
  if not str(item).startswith(('pxr/','usdex/','usd_exchange.libs/')):continue
  usd_checked+=1;path=Path(dist.locate_file(item))
  digest=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=') if path.is_file() else None
  if digest!=item.hash.value:usd_mismatches.append(str(path))
except md.PackageNotFoundError:pass
if not usd_checked or usd_mismatches:errors.append(f'USD RECORD failed ({usd_checked} files; {usd_mismatches[:5]})')
visible={}
for dist in md.distributions():
 name=canonicalize_name(dist.metadata['Name'])
 visible.setdefault(name,dict(version=dist.version,metadata=str(Path(dist._path).resolve())))
local_dists={canonicalize_name(d.metadata['Name']):d for d in md.distributions(path=[str(local)])}
overlay_pins=args.get('overlay_pins'); checks=[]
def check_requirement(parent, raw):
 req=Requirement(raw)
 if req.marker and not req.marker.evaluate({'extra':''}):return
 provider=visible.get(canonicalize_name(req.name)); satisfied=provider is not None and (not req.specifier or req.specifier.contains(provider['version'],prereleases=True))
 checks.append(dict(parent=parent,requires=str(req),provider=provider,satisfied=bool(satisfied)))
 if not satisfied:errors.append(f'{parent} requires {req}, found {provider}')
if overlay_pins is not None:
 if set(local_dists)!=set(overlay_pins):errors.append(f'overlay package set differs: extra {sorted(set(local_dists)-set(overlay_pins))}, missing {sorted(set(overlay_pins)-set(local_dists))}')
 for name,wanted in overlay_pins.items():
  dist=local_dists.get(name)
  if dist is None:continue
  if dist.version!=wanted:errors.append(f'overlay {name}=={dist.version}; expected {wanted}')
  for raw in dist.requires or []:check_requirement(name,raw)
 for raw in args.get('project_requirements',[]):check_requirement('project-core',raw)
result=dict(python=sys.executable,python_version=platform.python_version(),platform=sys.platform,machine=platform.machine(),site_packages=str(local),versions=versions,providers=providers,module_providers=module_providers,visible_packages=visible,overlay_packages={n:d.version for n,d in local_dists.items()} if overlay_pins is not None else {},default_requirements=checks,usd_record_files_checked=usd_checked,usd_record_mismatches=usd_mismatches,errors=errors,kit_launched=False,gpu_initialized=False)
print(json.dumps(result))
'''


def _expected_versions(profile: dict[str, Any]) -> dict[str, str]:
    return {"isaaclab": profile["isaaclab_package_version"], "isaacsim": profile["isaacsim"],
            "torch": profile["torch"], "numpy": profile["numpy"],
            "warp-lang": profile["warp"], "usd-exchange": profile["usd_exchange"]}


def _probe_python(python: Path, profile: dict[str, Any], *, pins: dict[str, str] | None = None,
                  requirements: list[str] | None = None) -> dict[str, Any]:
    args = {"versions": _expected_versions(profile), "excluded": profile.get("excluded_packages", ["usd-core"]),
            "overlay_pins": pins, "project_requirements": requirements or []}
    output = _run([str(python), "-I", "-c", _PROBE, json.dumps(args)])
    report = json.loads(output)
    if report["python_version"] != profile["python_recorded"]:
        report["errors"].append(f"Python is {report['python_version']}; expected {profile['python_recorded']}")
    if report["platform"] != profile["platform"] or report["machine"] != profile["machine"]:
        report["errors"].append("SDK platform differs from the profile")
    if report["errors"]:
        raise RuntimeError("Runtime verification failed: " + "; ".join(report["errors"]))
    return report


def _sdk_snapshot(lab_root: Path, profile: dict[str, Any]) -> dict[str, Any]:
    python = lab_root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Install the pinned SDK first; interpreter is missing: {python}")
    head = _run(["git", "--no-optional-locks", "-C", str(lab_root), "rev-parse", "HEAD"]).strip()
    source = _run(["git", "--no-optional-locks", "-C", str(lab_root), "rev-parse", "HEAD:source"]).strip()
    dirty = _run(["git", "--no-optional-locks", "-C", str(lab_root), "status", "--porcelain", "--", "source"]).strip()
    if dirty or (head != profile["isaaclab_commit"] and source != profile.get("isaaclab_source_tree")):
        raise RuntimeError("Isaac Lab source checkout differs from the pinned profile")
    if (lab_root / "VERSION").read_text().strip() != profile["isaaclab_product_version"]:
        raise RuntimeError("Isaac Lab product version differs from the profile")
    report = _probe_python(python, profile)
    sdk_site = Path(report["site_packages"])
    for name, path in {**report["providers"], **report["module_providers"]}.items():
        if not _inside(path, sdk_site):
            raise RuntimeError(f"SDK {name} provider escaped its SDK environment: {path}")
    return {"root": str(lab_root), "python": str(python), "head": head, "source_tree": source,
            "manifest_sha256": _sha256(lab_root / "pyproject.toml"),
            "lock_sha256": _sha256(lab_root / "uv.lock"), "probe": report}


def _fingerprint(project_root: Path, profile: dict[str, Any], sdk: dict[str, Any]) -> dict[str, Any]:
    # Store all visible package versions: upgrading any SDK provider invalidates
    # an existing runtime, rather than silently keeping an untested environment.
    return {"schema": 1, "project_root": str(project_root), "profile": PROFILE_NAME,
            "profile_sha256": _sha256(project_root / "env" / PROFILE_NAME / "profile.json"),
            "overlay_lock_sha256": profile["overlay_lock_sha256"],
            "project_manifest_sha256": _sha256(project_root / "pyproject.toml"),
            "sdk_root": sdk["root"], "sdk_python": sdk["python"],
            "sdk_source_tree": sdk["source_tree"], "sdk_manifest_sha256": sdk["manifest_sha256"],
            "sdk_lock_sha256": sdk["lock_sha256"], "sdk_python_version": sdk["probe"]["python_version"],
            "sdk_visible_versions": {n: p["version"] for n, p in sdk["probe"]["visible_packages"].items()}}


def _pth_content(sdk_site: Path, project_root: Path) -> str:
    # Executable .pth statements use repr(), preserving spaces, quotes and Unicode.
    # The project wins over any older editable project entry in the SDK's .pth.
    return (f"import site, sys; site.addsitedir({str(sdk_site)!r}); "
            f"sys.path.insert(0, {str(project_root / 'src')!r}); "
            f"sys.path.insert(0, {str(project_root)!r})\n")


def _verify_overlay(python: Path, project_root: Path, profile: dict[str, Any], pins: dict[str, str],
                    sdk: dict[str, Any]) -> dict[str, Any]:
    requirements = tomllib.loads((project_root / "pyproject.toml").read_text())["project"]["dependencies"]
    report = _probe_python(python, profile, pins=pins, requirements=requirements)
    sdk_site = Path(sdk["probe"]["site_packages"])
    for name, path in {**report["providers"], **report["module_providers"]}.items():
        if not _inside(path, sdk_site):
            raise RuntimeError(f"Overlay duplicated/replaced SDK provider {name}: {path}")
    site = Path(report["site_packages"])
    binding = site / "social_navigation_sdk_source.pth"
    if not binding.is_file() or binding.read_text() != _pth_content(sdk_site, project_root):
        raise RuntimeError("Runtime SDK/project .pth bindings differ from the recorded roots")
    return report


def check_runtime_environment(project_root: str | Path, lab_root: str | Path) -> dict[str, Any]:
    """Read-only verification of an existing overlay and its actual providers."""
    project_root = Path(project_root).expanduser().resolve()
    lab_root = Path(lab_root).expanduser().resolve()
    profile, pins = load_runtime_profile(project_root)
    sdk = _sdk_snapshot(lab_root, profile)
    expected = _fingerprint(project_root, profile, sdk)
    directory = project_root / ".runtime" / PROFILE_NAME
    receipt_path = directory / RECEIPT_NAME
    if not receipt_path.is_file():
        raise RuntimeError("Runtime receipt is missing; run prepare-runtime first")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("fingerprint") != expected:
        raise RuntimeError("Existing runtime does not exactly match SDK, project, profile or lock; preserved")
    python = directory / ".venv/bin/python"
    actual = _verify_overlay(python, project_root, profile, pins, sdk)
    return {"status": "verified", "python": str(python), "receipt_path": str(receipt_path),
            "receipt": {**receipt, "actual_probe": actual}}


def prepare_runtime_environment(project_root: str | Path, lab_root: str | Path) -> dict[str, Any]:
    """Install atomically, or verify an exact matching existing project runtime.

    Does not modify the SDK or change the project's .venv selection. Bootstrap
    selects the returned interpreter only after this function has succeeded.
    """
    project_root = Path(project_root).expanduser().resolve()
    lab_root = Path(lab_root).expanduser().resolve()
    if _inside(project_root, lab_root):
        raise RuntimeError("Project must be outside the SDK checkout to keep the SDK unmodified")
    profile, pins = load_runtime_profile(project_root)
    sdk = _sdk_snapshot(lab_root, profile)
    fingerprint = _fingerprint(project_root, profile, sdk)
    runtime_parent = project_root / ".runtime"
    if not _inside(runtime_parent, project_root) or _inside(runtime_parent, lab_root):
        raise RuntimeError("Project runtime path must stay inside the project and outside the SDK")
    runtime_parent.mkdir(parents=True, exist_ok=True)
    directory = runtime_parent / PROFILE_NAME
    with (runtime_parent / f".{PROFILE_NAME}.install.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is preparing this runtime") from exc
        if directory.exists() or directory.is_symlink():
            return check_runtime_environment(project_root, lab_root)
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("Install uv before preparing the project runtime")
        candidate = Path(tempfile.mkdtemp(prefix=f".{PROFILE_NAME}.build-", dir=runtime_parent))
        published = False
        try:
            venv = candidate / ".venv"
            log = candidate / "install.log"
            _run([uv, "venv", "--relocatable", "--python", sdk["python"], str(venv)], log=log)
            python = venv / "bin/python"
            site = Path(_run([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).strip())
            (site / "social_navigation_sdk_source.pth").write_text(_pth_content(Path(sdk["probe"]["site_packages"]), project_root))
            partial = candidate / "overlay-build-requirements.txt"
            partial.write_text("".join(f"{name}=={version}\n" for name, version in pins.items() if name != "chumpy"))
            _run([uv, "pip", "install", "--python", str(python), "--no-deps", "-r", str(partial)], log=log)
            if "chumpy" in pins:
                _run([uv, "pip", "install", "--python", str(python), "--no-deps", "--no-build-isolation",
                      f"chumpy=={pins['chumpy']}"], log=log)
            actual = _verify_overlay(python, project_root, profile, pins, sdk)
            receipt = {"schema": "social_navigation.runtime_overlay.v1", "fingerprint": fingerprint,
                       "sdk_probe": sdk["probe"], "actual_probe": actual,
                       "overlay_packages": pins, "sdk_modified": False,
                       "dependency_resolution": "disabled; exact overlay lock", "physical_tests_run": False}
            (candidate / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2))
            # No existing runtime is replaced. A failed build cannot change the
            # .venv selected by the caller or publish a success receipt.
            candidate.rename(directory)
            published = True
            verified = check_runtime_environment(project_root, lab_root)
            receipt["actual_probe"] = verified["receipt"]["actual_probe"]
            (directory / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2))
            verified["receipt"] = receipt
            verified["status"] = "prepared"
            return verified
        except BaseException:
            # This attempt only owns the new candidate (and its renamed final
            # directory). Keep the failure log; preserve all prior environments.
            failed_directory = directory if published else candidate
            if failed_directory.exists():
                failure_log = runtime_parent / f".{PROFILE_NAME}.failed-install.log"
                if (failed_directory / "install.log").is_file():
                    shutil.copyfile(failed_directory / "install.log", failure_log)
                shutil.rmtree(failed_directory)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lab-root", type=Path, required=True)
    args = parser.parse_args()
    function = prepare_runtime_environment if args.action == "prepare" else check_runtime_environment
    print(json.dumps(function(args.project_root, args.lab_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
