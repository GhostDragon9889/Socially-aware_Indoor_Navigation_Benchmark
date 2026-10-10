"""Prepare a fresh old931 copy with an independently approved state-prior delta.

Preparation is CPU only. Freeze/build are separate explicit operations; this
script never starts training, Kit, or CUDA and never changes old source files.
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
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = REPO / "output/isolated_training/mixed_state_action_terrain_protected_v1"
NEW = REPO / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
PILOT = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_pilot_v1"
FULL = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_v1"
SNAPSHOTS = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1"
SDK_PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
OLD_PROFILE = "configs/diffusion_amp/isolated_terrain_protected_absolute.yaml"
NEW_PROFILE = "configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml"
FLAG = "warm_start_action_discriminator_from_state_prior"
EXPECTED_TARGETS = {
    "CrowdSim/humanoid/application/settings.py": "modified",
    "CrowdSim/humanoid/application/resources.py": "modified",
    "CrowdSim/humanoid/training/trainer.py": "modified",
    "CrowdSim/humanoid/runtime/checkpoint_contract.py": "modified",
    "CrowdSim/humanoid/application/state_prior_initialization.py": "added",
    "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml": "added",
}
CANDIDATE = REPO / "docs/style_improvement_20261003/staged_action_state_prior_v1"
BASE_SHA = {
    "source_manifest.json": "4227b3d8eef584ded2348519b4ac8d7809b21c16c9a8c2624a2d8d2b82ccdd07",
    "launch_isolated.py": "6d56a3efb73534ca2cd54cc5ab82063fbb102b4a1a58f413c2421c26be199914",
    "commands_v2.json": "33a566d113b633316d29759630c736a7c951694972ec4e4fe9c39b96b1f3d4ce",
    "watch_snapshots.py": "be59ac0bb3224cbc5b2473219e96b742d182c97bb021eab2dfd14492c5691389",
    "code/" + OLD_PROFILE: "285ab2d912205087ab3b94175f3dd669a3f22a72fa9312a7f016a86906f5e113",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Expected regular file: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def baseline():
    for name, expected in BASE_SHA.items():
        require(digest(OLD / name) == expected, "Old baseline identity changed: " + name)
    manifest = json.loads((OLD / "source_manifest.json").read_text())
    require(len(manifest["files"]) == 931, "Expected immutable old931 source epoch")
    return manifest


def fresh_paths():
    for path in (NEW, PILOT, FULL, SNAPSHOTS):
        require(not path.exists() and not path.is_symlink(), "Exclusive new path already exists: " + str(path))


def profile():
    old = yaml.safe_load((OLD / "code" / OLD_PROFILE).read_text())
    new = deepcopy(old)
    require(FLAG not in old["training"], "New experiment flag unexpectedly exists in old profile")
    new["training"][FLAG] = True
    new["initialization"]["discriminator"] = "public_pacer_state_prior_zero_action_extension"
    new["profile"] = "fixed_slots_amass_procedural_xy_state_action_terrain_public_state_prior"
    replacements = {
        "All PPO, PD": "All PPO, PD, reward, replay, expert sampling, data, state_action2019 input, pre-step timing, protected public terrain moments and mixed command settings retain the old protected-terrain fresh2000 baseline. The single new experimental mechanism is coupled public state-prior D and AMP history-RMS cold initialization. The actor receives no branch ID or future reference3D poses.",
        "Fresh training transfers": "Fresh training transfers the same public PACER actor/critic/task, Gaussian sigma and observation/value moments. The2019-input discriminator copies the public1950 state-history columns, biases, middle layer and head, with69 new action columns initialized to zero. Its1950 history-coordinate mean/variance/count are transferred together with D and use the existing count cap200000. All discriminator parameters remain trainable after initialization.",
        "AMP normalization remains": "AMP normalization remains per_history_coordinate with1950 entries. This fresh initialization copies public AMP history mean/variance and records public count15482880001 before the existing count cap200000; action coordinates retain their existing normalized action convention and receive no learned action RMS.",
    }
    for prefix, replacement in replacements.items():
        matches = [i for i, text in enumerate(new["assumptions"]) if text.startswith(prefix)]
        require(len(matches) == 1, "Old initialization documentation changed: " + prefix)
        new["assumptions"][matches[0]] = replacement
    require({k: v for k, v in new["training"].items() if k != FLAG} == old["training"], "Unexpected training recipe change")
    return new


def commands():
    old = json.loads((OLD / "commands_v2.json").read_text())
    new = {key: deepcopy(old[key]) for key in ("env", "commands")}
    new.update(schema="public_state_prior.isolated_commands.v1", status="prepared_not_executed",
               cwd=str(NEW / "code"), python=SDK_PYTHON, launcher=str(NEW / "launch_isolated.py"),
               source_manifest=str(NEW / "source_manifest.json"), future_effect=False)
    new["env"]["PYTHONPATH"] = str(NEW / "code")
    new["env"]["PROTOMOTIONS_ASSET_ROOT"] = str(NEW / "code/protomotions/data/assets")
    for name, argv in new["commands"].items():
        argv[argv.index("--config") + 1] = str(NEW / "code" / NEW_PROFILE)
        argv[argv.index("--output") + 1] = str(FULL if name == "fullfresh2000" else PILOT)
        if "--resume" in argv:
            argv[argv.index("--resume") + 1] = str(PILOT / "step_4.pt")
    new["GPU_import_reports"] = {name: str(NEW / (name + "_GPU_imports.json")) for name in new["commands"]}
    new["watch_command"] = [SDK_PYTHON, str(NEW / "watch_snapshots.py"), "--training-output", str(FULL),
                            "--snapshot-dir", str(SNAPSHOTS), "--num-envs", "256", "--rollout-steps", "32",
                            "--final-update", "2000", "--targets", "500", "1000", "2000", "--max-seconds", "6000"]
    new["CPU_env"] = {**new["env"], "CUDA_VISIBLE_DEVICES": ""}
    new["pilot_seal_requirement"] = "Root must observe pilot4 process exit, seal exclusive step_4.pt plus result/metrics, and admit complete state before resume4to8. Seal step8 after process exit. Fresh2000 has no --resume."
    new["single_Kit_GPU_at_a_time"] = True
    return new


def validate_contract(path):
    value = json.loads(Path(path).read_text())
    require(value["schema"] == "public_state_prior.candidate_delta_contract.v1", "Wrong candidate contract schema")
    require(record(value["candidate_merge_manifest"]["path"]) == value["candidate_merge_manifest"], "Candidate owner manifest changed")
    require(type(value["targets"]) is list and value["targets"], "Explicit candidate target list required")
    names = set()
    for row in value["targets"]:
        name = Path(row["path"])
        require(not name.is_absolute() and ".." not in name.parts and row["path"] not in names, "Unsafe or duplicate candidate path")
        names.add(row["path"])
        require(row["kind"] in ("modified", "added"), "Unknown candidate file operation")
        for label in (("before", "proposed") if row["kind"] == "modified" else ("proposed",)):
            expected_path = CANDIDATE / ("before" if label == "before" else "proposal") / name
            require(row[label]["path"] == str(expected_path), "Candidate source must use declared staged path")
            require(record(row[label]["path"]) == row[label], "Candidate file identity changed: " + row["path"])
        require((OLD / "code" / name).is_file() == (row["kind"] == "modified"), "Candidate operation disagrees with old931 path")
        if row["kind"] == "modified":
            require(record(OLD / "code" / name) == row["isolated_before"], "Old931 candidate target identity changed")
            require(Path(row["isolated_after"]["path"]).is_relative_to(HERE), "Rebased review source must remain in preparation docs")
            require(record(row["isolated_after"]["path"]) == row["isolated_after"], "Rebased review source identity changed")
    require({row["path"]: row["kind"] for row in value["targets"]} == EXPECTED_TARGETS, "Unknown or missing candidate targets")
    return value


def apply_delta(row, code):
    delta = "".join(difflib.unified_diff(Path(row["before"]["path"]).read_text().splitlines(True),
        Path(row["proposed"]["path"]).read_text().splitlines(True), fromfile="a/" + row["path"], tofile="b/" + row["path"]))
    require(bool(delta), "Declared modified file has no delta")
    result = subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-d", str(code)],
                            input=delta, text=True, capture_output=True, timeout=30, check=False)
    require(result.returncode == 0, "Exact candidate delta failed: " + result.stdout + result.stderr)
    return {"path": row["path"], "delta_sha256": hashlib.sha256(delta.encode()).hexdigest(), "stdout": result.stdout, "fuzz": 0}


def record_contract(output):
    merge_path = CANDIDATE / "merge_manifest.json"
    merge = json.loads(merge_path.read_text())
    require(merge["schema"] == "bounded_candidate_patch.v1", "Wrong candidate merge schema")
    rows = []
    for entry in merge["targets"]:
        row = {"path": entry["path"], "kind": "added" if entry["before_sha256"] is None else "modified"}
        for label, folder, prefix in (("before", "before", "before"), ("proposed", "proposal", "after")):
            if row["kind"] == "added" and label == "before":
                continue
            row[label] = record(CANDIDATE / folder / entry["path"])
            require(row[label]["bytes"] == entry[prefix + "_bytes"] and row[label]["sha256"] == entry[prefix + "_sha256"],
                    "Candidate bytes disagree with candidate owner's manifest")
        rows.append(row)
    require({row["path"]: row["kind"] for row in rows} == EXPECTED_TARGETS, "Unknown or missing candidate targets")
    rebased = Path(output).absolute().parent / (Path(output).stem + "_rebased_targets")
    require(rebased.is_relative_to(HERE) and not rebased.exists(), "Rebased staged-review path must be fresh inside preparation docs")
    with tempfile.TemporaryDirectory(prefix="state_prior_old931_delta_", dir="/tmp") as temporary:
        for row in rows:
            if row["kind"] == "modified":
                row["isolated_before"] = record(OLD / "code" / row["path"])
                target = Path(temporary) / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OLD / "code" / row["path"], target)
                row["isolated_delta_proof"] = apply_delta(row, Path(temporary))
                staged = rebased / row["path"]
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, staged)
                row["isolated_after"] = record(staged)
    value = {"schema": "public_state_prior.candidate_delta_contract.v1", "status": "recorded_pending_independent_review",
             "candidate_merge_manifest": record(merge_path), "targets": rows, "future_effect": False}
    write_json(output, value)
    validate_contract(output)


def freeze_inputs(contract_path, review_path, output):
    baseline()
    fresh_paths()
    contract = validate_contract(contract_path)
    review = json.loads(Path(review_path).read_text())
    require(review.get("status") == "passed" and review.get("candidate_contract") == record(contract_path),
            "Peer review must explicitly pass this exact candidate contract")
    require(review.get("source_bound_CPU_proof_passed") is True, "Concept-only CPU proof cannot authorize candidate source")
    write_json(output, {"schema": "public_state_prior.isolation_inputs.v1", "status": "frozen_inputs_not_built_or_launched",
        "candidate_contract": record(contract_path), "peer_review": record(review_path), "targets": contract["targets"],
        "builder": record(__file__), "baseline": {name: record(OLD / name) for name in BASE_SHA},
        "CPU_only": True, "GPU_started": False, "Kit_started": False, "future_effect": False})


def build(inputs_path):
    old = baseline()
    fresh_paths()
    inputs = json.loads(Path(inputs_path).read_text())
    require(inputs["schema"] == "public_state_prior.isolation_inputs.v1", "Wrong frozen input schema")
    require(inputs["builder"] == record(__file__), "Builder changed after input freeze")
    for row in [inputs["candidate_contract"], inputs["peer_review"], *inputs["baseline"].values()]:
        require(record(row["path"]) == row, "Frozen gate input changed: " + row["path"])
    contract = validate_contract(inputs["candidate_contract"]["path"])
    require(inputs["targets"] == contract["targets"], "Frozen targets disagree with approved contract")
    declared = {row["path"] for row in old["files"]}
    actual = {str(path.relative_to(OLD)) for path in (OLD / "code").rglob("*") if path.is_file() and not path.is_symlink()}
    require(actual == declared, "Old931 regular file set changed")
    NEW.mkdir()
    code = NEW / "code"
    for row in old["files"]:
        relative = Path(row["path"])
        require(relative.parts[0] == "code" and ".." not in relative.parts, "Unsafe source manifest path")
        source, target = OLD / relative, NEW / relative
        require(not source.is_symlink() and source.stat().st_size == row["bytes"] and digest(source) == row["sha256"], "Old source changed: " + str(relative))
        require(not source.stat().st_mode & 0o222, "Old copied source is no longer immutable")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(target) == row["sha256"] and (target.stat().st_dev, target.stat().st_ino) != (source.stat().st_dev, source.stat().st_ino), "Source copy is not independent")
    proof = []
    for row in contract["targets"]:
        target = code / row["path"]
        if row["kind"] == "modified":
            result = apply_delta(row, code)
            require(result == row["isolated_delta_proof"], "Actual patch execution differs from approved old931 delta proof")
            require({key: record(target)[key] for key in ("bytes", "sha256")} == {key: row["isolated_after"][key] for key in ("bytes", "sha256")}, "Actual rebased target bytes differ from approved source")
            proof.append(result)
        else:
            require(not target.exists(), "Candidate addition already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["proposed"]["path"], target)
            require({key: record(target)[key] for key in ("bytes", "sha256")} == {key: row["proposed"][key] for key in ("bytes", "sha256")}, "Added candidate target identity differs")
    new_profile = code / NEW_PROFILE
    require(not new_profile.exists(), "Absolute profile path unexpectedly exists")
    new_profile.write_text(yaml.safe_dump(profile(), sort_keys=False, width=104))
    (NEW / "runtime_scratch").mkdir()
    for name in ("data", "third_party"):
        require((OLD / "code" / name).is_symlink(), "Expected declared shared resource symlink")
        (code / name).symlink_to((OLD / "code" / name).resolve(), target_is_directory=True)
    (code / "output").symlink_to(NEW / "runtime_scratch", target_is_directory=True)
    files = []
    for path in sorted(code.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            files.append({"path": str(path.relative_to(NEW)), "bytes": stat.st_size, "sha256": digest(path),
                          "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino, "copy_not_hardlink": True})
    prior = {row["path"]: row["sha256"] for row in old["files"]}
    changed = {row["path"] for row in files if row["path"] in prior and row["sha256"] != prior[row["path"]]}
    added = {row["path"] for row in files} - prior.keys()
    require(changed == {"code/" + row["path"] for row in contract["targets"] if row["kind"] == "modified"}, "Unexpected changed old931 files")
    require(added == {"code/" + row["path"] for row in contract["targets"] if row["kind"] == "added"} | {"code/" + NEW_PROFILE}, "Unexpected new files")
    for name in ("launch_isolated.py", "watch_snapshots.py"):
        shutil.copyfile(OLD / name, NEW / name)
    manifest = deepcopy(old)
    manifest.update(schema="public_state_prior.isolated_source.v1", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_root=str(code), files=files, profile=record(new_profile), launcher=record(NEW / "launch_isolated.py"),
        GPU_started=False, CPU_only_preparation=True, future_effect=False, previous_source_manifest=record(OLD / "source_manifest.json"),
        candidate_inputs=record(inputs_path), source_epoch={"base": "immutable protected-terrain old931", "old931_verified_before_delta": True,
            "unchanged_old_files_after_delta": 931 - len(changed), "modified_old_paths": sorted(changed), "new_paths": sorted(added), "patch_proof": proof})
    manifest["shared_resource_links"] = [{"path": "code/" + name, "target": str((code / name).resolve()),
        "access_policy": "readonly_by_all_preparation_and_training_operations" if name != "output" else "new_bundle_runtime_scratch"} for name in ("data", "third_party", "output")]
    manifest["resources_shared_not_OS_readonly"] = True
    write_json(NEW / "source_manifest.json", manifest)
    write_json(NEW / "commands.json", commands())
    write_json(NEW / "build_report.json", {"status": "copied_and_delta_verified_pending_CPU_preflight_and_training", "base_regular_files": 931,
        "changed_old_paths": sorted(changed), "added_paths": sorted(added), "base_bytes_preserved": 931 - len(changed),
        "all_source_copies_independent": True, "launcher_and_watcher_byte_identical": True, "old_sources_mutated": False,
        "builder": record(__file__), "inputs": record(inputs_path), "future_effect": False,
        "resource_full_SHA_recomputed_here": False, "strict_launcher_resource_verify_required_before_execution": True})
    for path in code.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    code.chmod(0o555)
    for name in ("launch_isolated.py", "watch_snapshots.py", "source_manifest.json", "commands.json", "build_report.json"):
        (NEW / name).chmod(0o444)
    return NEW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--record-candidate", type=Path)
    action.add_argument("--freeze-inputs", type=Path)
    action.add_argument("--build", type=Path)
    parser.add_argument("--candidate-contract", type=Path)
    parser.add_argument("--peer-review", type=Path)
    args = parser.parse_args()
    if args.prepare:
        baseline()
        fresh_paths()
        write_json(HERE / "commands_prepared.json", commands())
        with (HERE / "profile_prepared.yaml").open("x") as stream:
            yaml.safe_dump(profile(), stream, sort_keys=False, width=104)
    elif args.record_candidate:
        baseline()
        fresh_paths()
        record_contract(args.record_candidate)
    elif args.freeze_inputs:
        require(args.candidate_contract is not None and args.peer_review is not None, "Freeze requires exact candidate contract and passed source-bound peer proof")
        freeze_inputs(args.candidate_contract, args.peer_review, args.freeze_inputs)
    else:
        print(json.dumps({"prepared_bundle": str(build(args.build)), "GPU_started": False, "future_effect": False}))


if __name__ == "__main__":
    main()
