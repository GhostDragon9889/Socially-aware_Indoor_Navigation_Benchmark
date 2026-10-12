#!/usr/bin/env python3
"""Download only assets resolved by the parent's live public storage API.

CPU-only preparation: bounded curl transfers, exact USD/MDL dependency
discovery, hashes, and bounded offline USD composition checks. Nothing outside
the workspace is written. The original scene/USD resources are not patched.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urljoin, urlparse


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"Forklift": "Isaac/Props/Forklift/forklift.usd",
            "KLT_Bin": "Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd"}
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
SDK_MDL_ROOT = ROOT.parent / "IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/kit/mdl/core"
SDK_MDL = SDK_MDL_ROOT / "Base"


def mdl_dependency_records(source):
    """Read authored MDL module imports; unsupported syntax blocks completion."""
    source = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                    lambda match: match.group() if match.group().startswith('"') else "", source)
    # Descriptions and anno::thumbnail strings are editor metadata, not material
    # resource constructors. Static material resources are collected explicitly.
    records = [{"path": value, "kind": "mdl_file_literal"} for value in
               re.findall(r'\b(?:texture_[23]d|texture_cube|texture_ptex|light_profile|bsdf_measurement)\s*\(\s*"([^"\n]+)"', source)
               if value]
    declarations = re.findall(r"(?:\busing\s+[^;]+\s+)?\bimport\s+[^;]+;", source)
    for declaration in declarations:
        module_pattern = r"(?:(?:\.|\.\.)::)?[A-Za-z]\w*(?:::[A-Za-z]\w*)*"
        using = re.fullmatch(r"using\s+(::)?(" + module_pattern + r")\s+import\s+[^;]+;", declaration)
        ordinary = re.fullmatch(r"import\s+(::)?(" + module_pattern + r")(::\*)?\s*;", declaration)
        if using:
            absolute, module = using.group(1), using.group(2)
        elif ordinary:
            absolute, symbol = ordinary.group(1), ordinary.group(2)
            module = symbol if ordinary.group(3) else symbol.rpartition("::")[0]
            if not module:
                raise ValueError(f"Cannot resolve an MDL symbol import: {declaration}")
        else:
            raise ValueError(f"Unsupported MDL import syntax: {declaration}")
        records.append({"path": module.replace("::", "/") + ".mdl", "kind": "mdl_module_import",
                        "absolute_module": bool(absolute)})
    return records


def sdk_module_path(dependency, importing_path=None):
    """Return an actual installed module; never infer remote module objects."""
    raw = dependency["path"]
    candidates = []
    if importing_path is not None and not dependency.get("absolute_module"):
        candidates.append(Path(importing_path).parent / raw)
    candidates.extend([SDK_MDL_ROOT / "mdl" / raw, SDK_MDL / raw])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(SDK_MDL_ROOT.resolve()) and candidate.is_file():
            return candidate
    return None


def verify_installed_mdl(path):
    """Verify installed material source closure, without a renderer/compiler."""
    pending, seen, files = [Path(path)], set(), []
    while pending:
        current = pending.pop().resolve()
        if current in seen:
            continue
        seen.add(current)
        if len(seen) > 256 or not current.is_relative_to(SDK_MDL_ROOT.resolve()) or not current.is_file():
            raise ValueError(f"Installed MDL dependency missing, outside SDK, or beyond file bound: {current}")
        files.append({"installed_path": str(current), "bytes": current.stat().st_size, "sha256": digest(current)})
        for dependency in dependency_records(current):
            if dependency["kind"] == "mdl_module_import":
                module = sdk_module_path(dependency, current)
                if module is None:
                    raise ValueError(f"Installed MDL module unresolved: {dependency['path']}")
                pending.append(module)
            else:
                pending.append(current.parent / dependency["path"])
    return {"source_closure_verified": True, "mdl_compiler_verified": False, "files": files}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dependency_records(path):
    """Collect authored paths, not guessed sibling filenames or directory listings."""
    from pxr import Sdf

    path = Path(path)
    records = []
    if path.suffix.lower() in USD_SUFFIXES:
        with path.open("rb") as stream:
            header = stream.read(128)
        if not (header.startswith(b"PXR-USDC") or header.lstrip().startswith(b"#usda")):
            raise ValueError(f"Invalid USD magic or unmaterialized pointer: {path.name}")
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError(f"Unreadable USD: {path.name}")
        records.extend({"path": value, "kind": "composition"}
                       for value in layer.GetCompositionAssetDependencies())

        def collect(spec_path):
            attribute = layer.GetAttributeAtPath(spec_path)
            if attribute is None or str(attribute.typeName) not in {"asset", "asset[]"}:
                return
            values = [attribute.default]
            values.extend(layer.QueryTimeSample(spec_path, time)
                          for time in layer.ListTimeSamplesForPath(spec_path))
            for value in values:
                assets = [value] if isinstance(value, Sdf.AssetPath) else value or []
                for asset in assets:
                    if isinstance(asset, Sdf.AssetPath) and asset.path:
                        records.append({"path": asset.path, "kind": "attribute_asset", "spec": str(spec_path)})

        layer.Traverse("/", collect)
    elif path.suffix.lower() == ".mdl":
        records.extend(mdl_dependency_records(path.read_text(errors="strict")))
    return list({(record["path"], record["kind"]): record for record in records}.values())


def relative_destination(uri, asset_root, output):
    parsed = urlparse(uri)
    base = urlparse(asset_root)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credential/query-bearing dependency URL rejected")
    if parsed.scheme not in {"http", "https"} or parsed.netloc != base.netloc:
        raise ValueError("Dependency is not an HTTP file under the API-resolved asset root")
    prefix = base.path.rstrip("/") + "/"
    if not parsed.path.startswith(prefix):
        raise ValueError("Dependency escapes the versioned API-resolved root; requires a separate source/relocation plan")
    relative = unquote(parsed.path[len(prefix):])
    destination = (Path(output) / relative).resolve()
    if not destination.is_relative_to(Path(output).resolve()):
        raise ValueError("Asset destination escapes workspace output")
    return destination


def fetch(uri, destination, *, proxy, max_file_bytes, timeout):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--proto", "=http,https",
               "--connect-timeout", "10", "--max-time", str(timeout), "--retry", "1",
               "--retry-delay", "1", "--max-filesize", str(max_file_bytes),
               "--output", str(temporary), "--write-out", "%{http_code}\n%{size_download}\n%{url_effective}", uri]
    if proxy:
        command += ["--proxy", proxy, "--noproxy", ""]
    else:
        # Explicit direct mode must also override ambient *_PROXY variables.
        command += ["--proxy", "", "--noproxy", "*"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2 * timeout + 15)
        if completed.returncode:
            raise RuntimeError(f"curl failed: {completed.stderr.strip()[:500]}; result {completed.stdout.strip()}")
        if not temporary.is_file() or temporary.stat().st_size < 1 or temporary.stat().st_size > max_file_bytes:
            raise ValueError("Downloaded file is empty or exceeds the declared bound")
        status, transferred, effective_uri = completed.stdout.strip().splitlines()
        effective = urlparse(effective_uri)
        if effective.username or effective.password or effective.query or effective.fragment:
            raise ValueError("Authenticated/query-bearing redirect rejected")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"uri": uri, "path": str(destination), "bytes": destination.stat().st_size,
            "sha256": digest(destination), "http_status": int(status), "transferred_bytes": float(transferred),
            "effective_uri": effective_uri}


def validate_local_model(path, timeout=60):
    """Composed model bounds in a disposable CPU process, no Kit bootstrap."""
    code = r'''
from pathlib import Path
import json,sys
from pxr import Usd,UsdGeom
root=Path(sys.argv[1])
stage=Usd.Stage.Open(str(root),load=Usd.Stage.LoadAll)
if stage is None or stage.GetCompositionErrors():
    raise ValueError("Offline USD composition has errors")
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","render","proxy"],False,True)
bounds=cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
count=sum(prim.IsA(UsdGeom.Gprim) for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
if bounds.IsEmpty() or count==0:
    raise ValueError("No composed model geometry")
print(json.dumps({"gprim_count":count,"default_prim":str(stage.GetDefaultPrim().GetPath()),"meters_per_unit":float(UsdGeom.GetStageMetersPerUnit(stage)),"up_axis":str(UsdGeom.GetStageUpAxis(stage)),"bbox_stage_units":[list(bounds.GetMin()),list(bounds.GetMax())],"usd_version":list(Usd.GetVersion())}))
'''
    completed = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                               text=True, timeout=timeout,
                               env=dict(os.environ, PXR_WORK_THREAD_LIMIT="2", OMP_NUM_THREADS="1"))
    if completed.returncode:
        raise ValueError(f"Offline composition validation failed: {completed.stderr[-1000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def download_asset(asset, request, args, already_downloaded):
    root_uri = request["live_configured_root"].rstrip("/")
    pending = [(asset["resolved_uri"], "root")]
    seen, records, missing, absolute_composition, builtin = set(), [], [], [], []
    total_bytes = 0
    while pending:
        uri, kind = pending.pop()
        if uri in seen:
            continue
        seen.add(uri)
        if len(seen) > args.max_files:
            missing.append({"uri": uri, "reason": "declared dependency-count bound exceeded"})
            break
        try:
            destination = relative_destination(uri, root_uri, args.output)
            existing = already_downloaded.get(uri)
            if existing and destination.is_file() and digest(destination) == existing["sha256"]:
                record = {**existing, "reused_verified_file": True}
            else:
                remaining = args.max_total_bytes - total_bytes
                if remaining < 1:
                    raise ValueError("Declared total dependency-byte bound exhausted")
                record = fetch(uri, destination, proxy=args.proxy,
                               max_file_bytes=min(args.max_file_bytes, remaining), timeout=args.timeout)
                already_downloaded[uri] = record
            total_bytes += record["bytes"]
            if total_bytes > args.max_total_bytes:
                raise ValueError("Declared total dependency-byte bound exceeded")
            records.append(record)
            for dependency in dependency_records(destination):
                raw = dependency["path"]
                module = SDK_MDL / raw
                if "/" not in raw and raw in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"} and module.is_file():
                    builtin.append({"asset": raw, "kind": dependency["kind"], "installed_path": str(module),
                                    **verify_installed_mdl(module)})
                    continue
                if dependency["kind"] == "mdl_module_import":
                    module = sdk_module_path(dependency)
                    if module is not None:
                        builtin.append({"asset": raw, "kind": dependency["kind"],
                                        **verify_installed_mdl(module)})
                        continue
                    if dependency.get("absolute_module"):
                        missing.append({"asset": raw, "reason": "Absolute MDL module unavailable in verified SDK roots"})
                        continue
                if "://" in raw or raw.startswith("/"):
                    if dependency["kind"] == "composition":
                        absolute_composition.append({"layer": str(destination), "asset": raw})
                dependent_uri = urljoin(uri, raw)
                pending.append((dependent_uri, dependency["kind"]))
        except Exception as error:
            missing.append({"uri": uri, "kind": kind, "reason": f"{type(error).__name__}: {error}"})
    result = {"name": asset["name"], "relative_path": asset["relative_path"],
              "primary_source": asset["primary_source"], "files": records,
              "unresolved": missing, "absolute_composition_requires_relocation": absolute_composition,
              "verified_installed_mdl_modules": builtin,
              "author_asset_release_or_hash_known": False, "exact_author_bytes_claim": False,
              "external_symlinks_written": False, "physx_verified": False}
    if not missing and not absolute_composition:
        try:
            model = relative_destination(asset["resolved_uri"], root_uri, args.output)
            result["offline_model_validation"] = validate_local_model(model)
            result["status"] = "downloaded_local_dependencies_verified"
        except Exception as error:
            result.update(status="incomplete", validation_error=f"{type(error).__name__}: {error}")
    else:
        result["status"] = "incomplete"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default=str(ROOT / "data/scene_resources/dynbench_missing/official_api_request.json"))
    parser.add_argument("--output", default=str(ROOT / "data/scene_resources/dynbench_missing"))
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-files", type=int, default=256)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT) and not output.is_relative_to(Path(tempfile.gettempdir())):
        parser.error("Only workspace or the system temporary directory is permitted")
    if min(args.timeout, args.max_file_bytes, args.max_total_bytes, args.max_files) < 1:
        parser.error("Transport and dependency bounds must be positive")
    if not Path(args.request).is_file():
        parser.error("A request from the parent's live public storage API is required; no endpoint fallback")
    request = json.loads(Path(args.request).read_text())
    if request.get("schema") != "navisaaclab.official_asset_request.v1" or not request.get("resolution_api", "").startswith("isaacsim.storage.native.get_assets_root_path"):
        parser.error("The request is not the documented live public API schema")
    for asset in request["assets"]:
        if EXPECTED.get(asset["name"]) != asset["relative_path"]:
            parser.error("Only the two exact inspected missing assets are permitted")
        destination = relative_destination(asset["resolved_uri"], request["live_configured_root"], output)
        if destination.relative_to(output).as_posix() != asset["relative_path"]:
            parser.error("Resolved URI does not match the inspected exact relative asset path")
    if sorted(asset["name"] for asset in request["assets"]) != sorted(EXPECTED):
        parser.error("The request must contain each of the two exact assets once")
    old_manifest = output / "download_manifest.json"
    reused = {}
    if old_manifest.is_file():
        for asset in json.loads(old_manifest.read_text()).get("assets", []):
            reused.update({record["uri"]: record for record in asset.get("files", [])})
    results = []
    for asset in request["assets"]:
        result = download_asset(asset, request, args, reused)
        results.append(result)
        print(json.dumps({"name": result["name"], "status": result["status"],
                          "files": len(result["files"]), "unresolved": len(result["unresolved"])}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "navisaaclab.official_asset_download.v1",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "request_sha256": digest(args.request), "request": request, "assets": results,
                "license_source": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/licenses.html",
                "external_assets_or_sdk_modified": False}
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return int(any(result["status"] != "downloaded_local_dependencies_verified" for result in results))


if __name__ == "__main__":
    sys.exit(main())
