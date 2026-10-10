"""Prepare only the AMP outer-weight ablation from immutable state-prior934.

No training/Kit/CUDA is started. The source delta is only the independently
reviewed training-resume reward admission guard; fresh training math is copied.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
NEW = REPO / "output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2"
PILOT = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_pilot_v1"
FULL = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_v1"
SNAPSHOTS = REPO / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_snapshots_v1"
PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
BASE_PROFILE = "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml"
NEW_PROFILE = "code/configs/diffusion_amp/isolated_terrain_state_prior_pacer_reward_absolute.yaml"
GUARD_TARGETS = {"code/CrowdSim/humanoid/application/configuration.py", "code/scripts/train_paper_amp.py"}
BASE_MANIFEST_SHA = "3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a"
LAUNCHER_SHA = "6d56a3efb73534ca2cd54cc5ab82063fbb102b4a1a58f413c2421c26be199914"
WATCHER_SHA = "be59ac0bb3224cbc5b2473219e96b742d182c97bb021eab2dfd14492c5691389"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink(), "Expected regular file: " + str(p))
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def profile_text():
    text = (BASE / BASE_PROFILE).read_text()
    for old, new in (("  amp_reward_weight: 2.0\n", "  amp_reward_weight: 0.5\n"),
                     ("  amp_weight: 2.0\n", "  amp_weight: 0.5\n")):
        require(text.count(old) == 1, "Expected unique old reward setting")
        text = text.replace(old, new)
    old, new = yaml.safe_load((BASE / BASE_PROFILE).read_text()), yaml.safe_load(text)
    expected = deepcopy(old)
    expected["training"]["amp_reward_weight"] = .5
    expected["reward"]["amp_weight"] = .5
    require(new == expected, "Profile numeric change exceeds two declared settings")
    for i, note in enumerate(expected["assumptions"]):
        if note.startswith("All PPO, PD"):
            expected["assumptions"][i] = ("Independent fresh reward-factor ablation of the immutable public state-prior934 baseline: "
                "AMP outer weight changes from2.0 to0.5; internal AMP scale2.0 is unchanged, giving effective softplus coefficient1.0. "
                "Public coupled state-prior D/history-RMS initialization is retained from the baseline. "
                "All other PPO, PD, replay, expert sampling, data, input timing, protected terrain moments and mixed commands remain identical. "
                "The resume-only reward guard changes admission, not fresh training math or RNG draws.")
        elif note.startswith("NavIsaacLab Eq2"):
            expected["assumptions"][i] = ("NavIsaacLab Eq2 prints a history-and-action discriminator. This experiment retains the "
                "engineering10-frame newest-first state_action2019 convention and negative-direction reward. "
                "AMP outer weight0.5 with internal scale2.0 is a PACER-code coefficient comparison, not an author NavIsaacLab paper-exact recipe.")
    expected["profile"] = "fixed_slots_amass_procedural_xy_state_action_terrain_public_state_prior_pacer_reward_factor"
    return yaml.safe_dump(expected, sort_keys=False, width=104)


def commands():
    old = json.loads((REPO / "docs/style_improvement_20261003/state_prior_training_preparation_v1/root_GPU_launch_commands.json").read_text())
    value = deepcopy(old)
    value.update(schema="public_state_prior.pacer_reward_root_launch_commands.v1", status="prepared_not_executed",
                 cwd=str(NEW / "code"), future_effect=False)
    value["env"]["PYTHONPATH"] = str(NEW / "code")
    value["env"]["PROTOMOTIONS_ASSET_ROOT"] = str(NEW / "code/protomotions/data/assets")
    value["commands"].pop("resume4to8")
    for name, argv in value["commands"].items():
        argv[1] = str(NEW / "launch_isolated.py")
        argv[argv.index("--import-report") + 1] = str(NEW / (name + "_GPU_imports.json"))
        argv[argv.index("--config") + 1] = str(NEW / NEW_PROFILE)
        argv[argv.index("--output") + 1] = str(FULL if name == "fullfresh2000" else PILOT)
        if "--resume" in argv:
            argv[argv.index("--resume") + 1] = str(PILOT / "step_4.pt")
        require("--resume" not in argv or name == "resume4to8", "Fresh commands must not resume")
    value["watch_command"][1] = str(NEW / "watch_snapshots.py")
    watch = value["watch_command"]
    watch[watch.index("--training-output") + 1] = str(FULL)
    watch[watch.index("--snapshot-dir") + 1] = str(SNAPSHOTS)
    value["CPU_watch_env"] = {**value["env"], "CUDA_VISIBLE_DEVICES": ""}
    value["reward_contract"] = {"amp_reward_scale": 2., "amp_reward_weight": .5,
                                "reward.amp_weight": .5, "softplus_effective_coefficient_before_clip": 1.,
                                "expert_sampling": None, "resume_old_weight_2_allowed": False}
    value["watcher_scope"] = "Original watcher only checks serialization/payload/budget/recipe/dimensions; independent reward-config audit and readonly snapshot seal are required."
    value["pilot_seal_requirement"] = "Root must observe pilot4 process exit, seal exclusive step_4.pt/result/metrics and audit complete payload plus saved amp_reward_weight=.5. Fresh2000 has no --resume and does not continue the pilot. No additional pilot GPU resume is required by this preparation."
    return value


def baseline():
    require(digest(BASE / "source_manifest.json") == BASE_MANIFEST_SHA, "Immutable934 manifest changed")
    value = json.loads((BASE / "source_manifest.json").read_text())
    require(len(value["files"]) == 934, "Expected source934")
    declared = {r["path"] for r in value["files"]}
    actual = {str(p.relative_to(BASE)) for p in (BASE / "code").rglob("*") if p.is_file() and not p.is_symlink()}
    require(declared == actual, "Base regular set differs")
    for row in value["files"]:
        p = BASE / row["path"]
        require(not p.is_symlink() and not p.stat().st_mode & 0o222 and p.stat().st_size == row["bytes"] and digest(p) == row["sha256"], "Base source changed: " + row["path"])
    require(digest(BASE / "launch_isolated.py") == LAUNCHER_SHA, "Launcher changed")
    require(digest(BASE / "watch_snapshots.py") == WATCHER_SHA, "Watcher changed")
    for row in value["resources"]:
        p = Path(row["path"])
        require(p.stat().st_size == row["bytes"] and p.stat().st_mtime_ns == row["mtime_ns"] and digest(p) == row["sha256"], "Shared input changed: " + str(p))
    return value


def fresh():
    for p in (NEW, PILOT, FULL, SNAPSHOTS):
        require(not p.exists() and not p.is_symlink(), "Exclusive experiment path already exists: " + str(p))


def build(contract_path):
    base = baseline()
    fresh()
    contract = json.loads(Path(contract_path).read_text())
    require(contract["status"] == "passed_exact_delta_review" and contract["future_effect"] is False, "Explicit approved delta contract required")
    require({r["path"] for r in contract["targets"]} == GUARD_TARGETS, "Expected only two resume admission targets")
    for row in [contract["owner_proof"], contract["peer_review"], *[r[k] for r in contract["targets"] for k in ("isolated_before", "isolated_after", "patch")]]:
        require(record(row["path"]) == row, "Approved guard evidence changed")
    for row in contract["targets"]:
        require(record(BASE / row["path"]) == row["isolated_before"], "Base guard target mismatch")
    with tempfile.TemporaryDirectory(prefix="pacer_reward_guard_", dir="/tmp") as temp:
        rebased = Path(temp)
        for row in contract["targets"]:
            target = rebased / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / row["path"], target)
            run = subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-d", str(rebased / "code")],
                                 input=Path(row["patch"]["path"]).read_text(), text=True, capture_output=True, check=False)
            require(run.returncode == 0 and digest(target) == row["isolated_after"]["sha256"] and target.stat().st_size == row["isolated_after"]["bytes"], "Exact reviewed guard delta failed")
        NEW.mkdir()
        for row in base["files"]:
            source, target = BASE / row["path"], NEW / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rebased / row["path"] if row["path"] in GUARD_TARGETS else source, target)
            require(target.stat().st_nlink == 1 and (target.stat().st_dev, target.stat().st_ino) != (source.stat().st_dev, source.stat().st_ino), "Copies must be independent")
    (NEW / NEW_PROFILE).write_text(profile_text())
    (NEW / "runtime_scratch").mkdir()
    links = []
    for row in base["shared_resource_links"]:
        target = NEW / "runtime_scratch" if row["path"] == "code/output" else Path(row["target"])
        (NEW / row["path"]).symlink_to(target, target_is_directory=True)
        links.append({**row, "target": str(target)})
    files = []
    for p in sorted((NEW / "code").rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            files.append({"path": str(p.relative_to(NEW)), "bytes": st.st_size, "sha256": digest(p),
                          "mtime_ns": st.st_mtime_ns, "device": st.st_dev, "inode": st.st_ino, "copy_not_hardlink": True})
    prior = {r["path"]: r for r in base["files"]}
    changed = {r["path"] for r in files if r["path"] in prior and r["sha256"] != prior[r["path"]]["sha256"]}
    added = {r["path"] for r in files} - set(prior)
    require(len(files) == 935 and changed == GUARD_TARGETS and added == {NEW_PROFILE}, "Unexpected935 delta")
    for name in ("launch_isolated.py", "watch_snapshots.py"):
        shutil.copyfile(BASE / name, NEW / name)
    manifest = deepcopy(base)
    manifest.update(schema="public_state_prior.pacer_reward_isolated_source.v1", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    code_root=str(NEW / "code"), files=files, profile=record(NEW / NEW_PROFILE), launcher=record(NEW / "launch_isolated.py"),
                    shared_resource_links=links, previous_source_manifest=record(BASE / "source_manifest.json"), candidate_inputs=record(contract_path),
                    source_epoch={"base_regular_files": 934, "unchanged_base_files": 932, "modified_old_paths": sorted(changed), "new_paths": sorted(added),
                                  "guard_only_affects_training_resume_admission": True, "fresh_training_single_mechanism": "AMP external weight2 to0.5"},
                    CPU_only_preparation=True, GPU_started=False, future_effect=False)
    for key in ("business_source_bytes_unchanged_from_v1", "business_source_bytes_unchanged_from_v2"):
        manifest.pop(key, None)
    write_json(NEW / "source_manifest.json", manifest)
    write_json(NEW / "commands.json", commands())
    write_json(NEW / "build_report.json", {"status": "built_935_pending_actual_CPU_gate_no_training", "regular_source_count": 935,
               "unchanged_base_source_count": 932, "delta_contract": record(contract_path), "builder": record(__file__),
               "profile_delta_keys": ["training.amp_reward_weight", "reward.amp_weight"], "metadata_delta_keys": ["assumptions[1]", "assumptions[7]", "profile"], "source_copies_independent": True,
               "six_shared_inputs_verified": True, "launcher_watcher_byte_identical": True, "future_effect": False})
    for p in (NEW / "code").rglob("*"):
        if not p.is_symlink():
            p.chmod(0o555 if p.is_dir() else 0o444)
    (NEW / "code").chmod(0o555)
    for name in ("launch_isolated.py", "watch_snapshots.py", "source_manifest.json", "commands.json", "build_report.json"):
        (NEW / name).chmod(0o444)
    print(json.dumps({"built_bundle": str(NEW), "future_effect": False, "GPU_started": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--build", type=Path)
    args = parser.parse_args()
    if args.prepare:
        base = baseline()
        fresh()
        with (HERE / "profile_prepared_v2.yaml").open("x") as stream:
            stream.write(profile_text())
        write_json(HERE / "root_GPU_launch_commands_v2.json", commands())
        write_json(HERE / "prepared_metadata_v2.json", {"status": "prepared_pending_guard_source_and_peer_no_bundle", "base": record(BASE / "source_manifest.json"),
                   "new_profile": record(HERE / "profile_prepared_v2.yaml"), "guard_targets": sorted(GUARD_TARGETS), "six_inputs": base["resources"],
                   "profile_delta_only": ["training.amp_reward_weight", "reward.amp_weight"], "future_effect": False})
    else:
        build(args.build)


if __name__ == "__main__":
    main()
