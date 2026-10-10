"""Copy verified old926 files and apply only the frozen RMS candidate delta.

Creates a fresh exclusive bundle and command templates. Does not launch any
training, SDK, GPU, or write the old bundle/data/weights. Linux patch is used
with exact contexts (fuzz0) so unrelated later file changes are not copied.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD_BUNDLE = REPO / "output/isolated_training/mixed_state_action_v1"
MODIFIED = ("CrowdSim/humanoid/application/settings.py", "CrowdSim/humanoid/application/resources.py",
            "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/runtime/checkpoint_contract.py",
            "CrowdSim/humanoid/runtime/amp_inference.py")
ADDED = ("CrowdSim/humanoid/training/terrain_statistics.py",
         "configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml",
         "tests/test_task_terrain_statistics.py", "tests/test_terrain_statistics_integration.py")
SDK_PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_inputs(path):
    """Call after all candidate owners have frozen their files and CPU gates."""
    files = []
    for group, names in (("before", MODIFIED), ("proposed", (*MODIFIED, *ADDED))):
        for name in names:
            row = record(HERE/group/name)
            row["candidate_path"] = f"{group}/{name}"
            files.append(row)
    write_json(path, {"schema": "terrain_RMS.isolation_candidate_inputs.v1", "status": "frozen_inputs_not_launched",
                      "files": files, "old_source_manifest": record(OLD_BUNDLE/"source_manifest_v3.json"),
                      "old_launcher": record(OLD_BUNDLE/"launch_isolated_v3.py"),
                      "builder": record(__file__), "CPU_only": True})


def build(destination, inputs_path):
    destination = Path(destination).absolute()
    require(destination.is_relative_to(REPO/"output/isolated_training"), "Fresh bundle must be inside workspace output/isolated_training")
    require(not destination.exists(), "Bundle destination must be entirely fresh; no existing code/weights overwrite")
    inputs = json.loads(Path(inputs_path).read_text())
    require(inputs["schema"] == "terrain_RMS.isolation_candidate_inputs.v1", "Wrong candidate input schema")
    require(inputs["builder"] == record(__file__), "Builder changed after freezing inputs")
    for row in inputs["files"]:
        require(record(HERE/row["candidate_path"]) == {key: row[key] for key in ("path", "bytes", "sha256")},
                "Candidate source changed: "+row["candidate_path"])
    require(inputs["old_source_manifest"] == record(OLD_BUNDLE/"source_manifest_v3.json"), "Old source manifest changed")
    require(inputs["old_launcher"] == record(OLD_BUNDLE/"launch_isolated_v3.py"), "Old launcher changed")
    original = json.loads((OLD_BUNDLE/"source_manifest_v3.json").read_text())
    require(len(original["files"]) == 926, "Expected actual old926 regular-file epoch")
    destination.mkdir()
    code = destination/"code"
    original_rows = []
    for row in original["files"]:
        relative = Path(row["path"])
        require(relative.parts[0] == "code" and ".." not in relative.parts, "Invalid old source relative path")
        source = OLD_BUNDLE/relative
        require(not source.is_symlink() and source.stat().st_size == row["bytes"] and digest(source) == row["sha256"],
                "Old frozen source changed: "+str(relative))
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(target) == row["sha256"] and target.stat().st_ino != source.stat().st_ino,
                "Copy must preserve source bytes without hardlink")
        original_rows.append({"path": str(relative), "old_sha256": row["sha256"], "copy_not_hardlink": True})
    patch_proof = []
    for name in MODIFIED:
        delta = "".join(difflib.unified_diff((HERE/"before"/name).read_text().splitlines(True),
                                            (HERE/"proposed"/name).read_text().splitlines(True),
                                            fromfile="a/"+name, tofile="b/"+name))
        require(bool(delta), "No declared RMS delta for "+name)
        result = subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-d", str(code)],
                                input=delta, text=True, capture_output=True, timeout=30, check=False)
        require(result.returncode == 0, "Exact RMS delta failed for "+name+": "+result.stdout+result.stderr)
        patch_proof.append({"path": name, "delta_sha256": hashlib.sha256(delta.encode()).hexdigest(),
                            "stdout": result.stdout, "only_RMS_before_to_proposed_delta": True})
    for name in ADDED:
        target = code/name
        require(not target.exists(), "New candidate path unexpectedly exists in old snapshot: "+name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/"proposed"/name, target)
    old_profile = yaml.safe_load((code/"configs/diffusion_amp/isolated_state_action_absolute.yaml").read_text())
    profile = yaml.safe_load((code/ADDED[1]).read_text())
    profile["data_manifest"] = old_profile["data_manifest"]
    profile["initialization"]["checkpoint"] = old_profile["initialization"]["checkpoint"]
    profile_path = code/"configs/diffusion_amp/isolated_terrain_protected_absolute.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False, width=104))
    new_field = "task_terrain_statistics_mode"
    require({key: value for key, value in profile["training"].items() if key != new_field} == old_profile["training"],
            "More than the single terrain-statistics mode training variable changed")
    (destination/"runtime_scratch").mkdir()
    for name in ("data", "third_party"):
        old_link = OLD_BUNDLE/"code"/name
        require(old_link.is_symlink(), "Expected explicitly shared old resource link")
        (code/name).symlink_to(old_link.resolve(), target_is_directory=True)
    (code/"output").symlink_to(destination/"runtime_scratch", target_is_directory=True)
    files = []
    for path in sorted(code.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            files.append({"path": str(path.relative_to(destination)), "bytes": stat.st_size, "sha256": digest(path),
                          "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino, "copy_not_hardlink": True})
    original_by_path = {row["path"]: row["sha256"] for row in original["files"]}
    actual_changes = {row["path"] for row in files if row["path"] in original_by_path and row["sha256"] != original_by_path[row["path"]]}
    require(actual_changes == {"code/"+name for name in MODIFIED}, "Unexpected old file byte changes")
    additions = {row["path"] for row in files} - set(original_by_path)
    require(additions == {"code/"+name for name in ADDED} | {"code/configs/diffusion_amp/isolated_terrain_protected_absolute.yaml"},
            "Unexpected new code/asset file set")
    launcher_path = destination/"launch_isolated.py"
    launcher = (OLD_BUNDLE/"launch_isolated_v3.py").read_text().replace("source_manifest_v3.json", "source_manifest.json")
    launcher_path.write_text(launcher)
    manifest = deepcopy(original)
    manifest.update({"schema": "terrain_RMS.isolated_source.v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "code_root": str(code), "files": files, "GPU_started": False, "previous_source_manifest": record(OLD_BUNDLE/"source_manifest_v3.json"),
                     "source_epoch": {"base": "actual old146 plus individually frozen old926", "old926_verified_before_delta": True,
                                      "unchanged_old_files_after_delta": 921, "modified_old_paths": sorted(actual_changes),
                                      "new_paths": sorted(additions), "patch_proof": patch_proof},
                     "candidate_inputs": record(inputs_path), "launcher": record(launcher_path), "profile": record(profile_path),
                     "business_source_bytes_unchanged_from_v1": False, "business_source_bytes_unchanged_from_v2": False,
                     "CPU_only_preparation": True, "normalization_policy_integrated_in_this_copy": True})
    manifest["shared_resource_links"] = [{"path": "code/"+name, "target": str((code/name).resolve())} for name in ("data", "third_party", "output")]
    write_json(destination/"source_manifest.json", manifest)
    old_commands = json.loads((OLD_BUNDLE/"commands_v3.json").read_text())
    commands = deepcopy(old_commands)
    commands.update({"schema": "terrain_RMS.isolated_commands.v1", "status": "prepared_not_executed",
                     "cwd": str(code), "python": SDK_PYTHON, "launcher": str(launcher_path), "source_manifest": str(destination/"source_manifest.json")})
    commands["env"]["PYTHONPATH"] = str(code)
    commands["env"]["PROTOMOTIONS_ASSET_ROOT"] = str(code/"protomotions/data/assets")
    new_pilot = REPO/"output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1"
    new_full = REPO/"output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_v1"
    for label, argv in commands["commands"].items():
        argv[argv.index("--config")+1] = str(profile_path)
        argv[argv.index("--output")+1] = str(new_full if "full" in label else new_pilot)
        if "--resume" in argv:
            argv[argv.index("--resume")+1] = str(new_pilot/"step_4.pt")
    write_json(destination/"commands.json", commands)
    for path in code.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    code.chmod(0o555)
    write_json(destination/"build_report.json", {"status": "copy_and_delta_verified_not_CPU_preflight_or_GPU_training",
               "base926_regular_files": 926, "base921_preserved_bytes": 921, "old5_RMS_delta_only": sorted(actual_changes),
               "new5_paths": sorted(additions), "no_old_bundle_writes": True, "resource_full_SHA_recomputed_here": False,
               "resources_bound_to_old_manifest_SHA_and_stat_pending_launcher_verify": True,
               "source_manifest": record(destination/"source_manifest.json"), "commands": record(destination/"commands.json"),
               "original_source_rows_before_delta": original_rows, "builder": record(__file__), "inputs": record(inputs_path)})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-inputs", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.freeze_inputs is not None:
        require(args.inputs is None and args.destination is None, "Freeze and build are separate explicit operations")
        freeze_inputs(args.freeze_inputs)
    else:
        require(args.inputs is not None and args.destination is not None, "Build requires frozen inputs and fresh destination")
        print(json.dumps({"prepared_bundle": str(build(args.destination, args.inputs)), "GPU_started": False}))


if __name__ == "__main__":
    main()
