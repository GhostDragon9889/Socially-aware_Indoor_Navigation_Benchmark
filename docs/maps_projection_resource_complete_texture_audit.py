"""Bounded-parent CPU audit of authored scene texture/MDL dependencies.

No USD stage, SDK bootstrap, HTTP transfer, or original resource write occurs.
Existing texture bytes are not all decoded; this checks authored path closure.
"""
from pathlib import Path
import argparse
import json
import runpy
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = runpy.run_path(str(ROOT / "scripts/download_dynbench_missing.py"))


def audit(scene_directory):
    composition = json.loads((scene_directory / "composition.json").read_text())
    layers = [Path(layer["path"]) for layer in composition["layers"] if layer["exists"]]
    pending = list(layers)
    visited, assets, mdl, missing, remote, invalid = set(), {}, {}, [], [], []
    references = 0
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        if len(visited) > 20_000:
            raise ValueError("Authored dependency file bound exceeded")
        for dependency in DOWNLOAD["dependency_records"](path):
            if dependency["kind"] == "composition":
                continue  # Already inventoried recursively before stage generation.
            references += 1
            raw = dependency["path"]
            authored = {"layer": str(path), "asset": raw, "kind": dependency["kind"],
                        "spec": dependency.get("spec")}
            if urlparse(raw).scheme:
                remote.append(authored)
                continue
            local = Path(raw) if Path(raw).is_absolute() else path.parent / raw
            local = local.resolve()
            if not local.is_file() and local.suffix.lower() == ".mdl":
                module = DOWNLOAD["sdk_module_path"](dependency)
                if module is not None:
                    if str(module) not in mdl:
                        try:
                            mdl[str(module)] = DOWNLOAD["verify_installed_mdl"](module)
                        except ValueError as error:
                            invalid.append({**authored, "reason": str(error)})
                    continue
            if not local.is_file():
                missing.append({**authored, "resolved_local_path": str(local)})
                continue
            with local.open("rb") as stream:
                header = stream.read(128)
            if not header or header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                invalid.append({**authored, "resolved_local_path": str(local),
                                "reason": "empty_or_unmaterialized_LFS_resource"})
                continue
            assets[str(local)] = {"path": str(local), "bytes": local.stat().st_size,
                                  "mtime_ns": local.stat().st_mtime_ns}
            if local.suffix.lower() in DOWNLOAD["USD_SUFFIXES"] | {".mdl"}:
                pending.append(local)
    return {"schema": "navisaaclab.scene_authored_visual_dependency_audit.v1",
            "scene": scene_directory.name, "composition_layers_scanned": len(layers),
            "unique_layer_or_mdl_files_scanned": len(visited), "authored_asset_records_scanned": references,
            "existing_local_assets": list(assets.values()), "installed_MDL_source_closures": mdl,
            "missing_local_assets": missing, "remote_authored_asset_URIs": remote,
            "invalid_assets": invalid, "authored_local_visual_dependency_closure_complete": not (missing or remote or invalid),
            "existing_texture_bytes_all_decoded": False, "MDL_compiler_or_runtime_resolver_verified": False,
            "physx_or_Kit_verified": False, "original_resources_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_directory", type=Path)
    args = parser.parse_args()
    directory = args.scene_directory.resolve()
    if not directory.is_relative_to(ROOT):
        parser.error("Report output must be under the workspace")
    result = audit(directory)
    (directory / "visual_dependency_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["scene", "authored_local_visual_dependency_closure_complete",
                    "composition_layers_scanned", "authored_asset_records_scanned"]}
                     | {"missing_local": len(result["missing_local_assets"]),
                        "remote": len(result["remote_authored_asset_URIs"]), "invalid": len(result["invalid_assets"])}))


if __name__ == "__main__":
    main()
