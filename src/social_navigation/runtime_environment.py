"""Verify the external SDK and locked project dependencies without modifying Isaac's SDK.

The SDK owns Torch, NumPy, Warp, OpenUSD and Isaac libraries. Only the exact
project overlay lock is installed, with dependency resolution disabled. Both
SDK and project default dependency requirements are recorded; this check does
not claim that simulation, training or GPU tests have run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

PROFILE_NAME = "isaaclab3_ea_sim61"
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
local_dists={canonicalize_name(d.metadata['Name']):d for d in md.distributions(path=[args.get('addons_site',str(local))])}
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
                  requirements: list[str] | None = None, addons: Path | None = None) -> dict[str, Any]:
    args = {"versions": _expected_versions(profile), "excluded": profile.get("excluded_packages", ["usd-core"]),
            "overlay_pins": pins, "project_requirements": requirements or []}
    code = _PROBE
    if addons is not None:
        args['addons_site'] = str(addons)
        code = ('import site, sys\nsite.addsitedir(sys.argv[2])\n'
                'sys.path.remove(sys.argv[2]); sys.path.insert(0, sys.argv[2])\n' + code)
    output = _run([str(python), "-I", "-c", code, json.dumps(args), *([str(addons)] if addons is not None else [])])
    report = json.loads(output)
    if report["python_version"] != profile["python_recorded"]:
        report["errors"].append(f"Python is {report['python_version']}; expected {profile['python_recorded']}")
    if report["platform"] != profile["platform"] or report["machine"] != profile["machine"]:
        report["errors"].append("SDK platform differs from the profile")
    if report["errors"]:
        raise RuntimeError("Runtime verification failed: " + "; ".join(report["errors"]))
    return report


def _sdk_snapshot(lab_root: Path, profile: dict[str, Any], *, python: Path | None = None) -> dict[str, Any]:
    python = python or lab_root / ".venv" / "bin" / "python"
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
