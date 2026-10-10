"""Prepare only: old934 exact source plus one explicit expert-prior profile."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
NEW = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"
PROFILE = "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"
OLD_PROFILE = "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml"
OUTPUT = ROOT / "output/diffusion_amp/amp_style_state_prior_stratified_isolated_v1"
PILOT = ROOT / "output/diffusion_amp/amp_style_state_prior_stratified_isolated_pilot_v1"
SNAPSHOTS = ROOT / "output/diffusion_amp/amp_style_state_prior_stratified_isolated_snapshots_v1"


def record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert record(OLD / "source_manifest.json")["sha256"] == "3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a"
    baseline = json.loads((OLD / "source_manifest.json").read_text())
    assert len(baseline["files"]) == 934
    assert not any(path.exists() or path.is_symlink() for path in (NEW, OUTPUT, PILOT, SNAPSHOTS))
    old_profile = yaml.safe_load((OLD / OLD_PROFILE).read_text())
    source_profile = yaml.safe_load((ROOT / "configs/diffusion_amp/amp_style_amass_stratified_experts.yaml").read_text())
    new_profile = deepcopy(old_profile)
    new_profile["training"]["expert_sampling"] = source_profile["training"]["expert_sampling"]
    assert old_profile["training"]["expert_sampling"] is None
    assert {k: v for k, v in new_profile["training"].items() if k != "expert_sampling"} == {k: v for k, v in old_profile["training"].items() if k != "expert_sampling"}
    assert new_profile["training"]["amp_reward_weight"] == new_profile["reward"]["amp_weight"] == 2
    new_profile["profile"] = "fixed_slots_state_prior_weight2_stratified_expert_single_variable"
    new_profile["assumptions"][1] = (
        "Relative to completed19f state-prior weight2 fresh2000 control, only training.expert_sampling changes. "
        "Public actor/critic/state-prior discriminator/history RMS, PPO, PD, task/reward weights, replay, "
        "fixed-half AMASS/procedural commands and protected terrain statistics remain identical. "
        "The2848 registry and train/validation/test parent separation are unchanged.")
    new_profile["assumptions"].append(
        "Local engineering expert prior: low_speed.10/fast.05/sideways.10/turning.25/moving.50; "
        "sqrt(dataset parent support), then uniform parent and uniform legal history10/future1 center. "
        "Kinematic window proxies are not author style labels. No p/q importance correction is applied. "
        "A private CPU generator uses seed7. Each expert call still makes one discarded actual legacy "
        "corpus call, preserving shared reset RNG only for matched call schedules. Full resume must retain "
        "both private sampler and shared corpus RNG states; changed falls/reset schedules can diverge.")
    NEW.mkdir()
    copied = []
    for row in baseline["files"]:
        source, destination = OLD / row["path"], NEW / row["path"]
        identity = record(source)
        assert (identity["bytes"], identity["sha256"]) == (row["bytes"], row["sha256"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert record(destination)["sha256"] == row["sha256"]
        assert destination.stat().st_ino != source.stat().st_ino
        copied.append(row["path"])
    profile_path = NEW / PROFILE
    profile_path.write_text(yaml.safe_dump(new_profile, sort_keys=False, width=105))
    (HERE / "profile_prepared.yaml").write_bytes(profile_path.read_bytes())
    (NEW / "runtime_scratch").mkdir()
    links = deepcopy(baseline["shared_resource_links"])
    for row in links:
        if row["path"] == "code/output":
            row["target"] = str(NEW / "runtime_scratch")
        destination = NEW / row["path"]
        destination.symlink_to(row["target"], target_is_directory=True)
    for name in ("launch_isolated.py", "watch_snapshots.py"):
        shutil.copy2(OLD / name, NEW / name)
        assert record(NEW / name)["sha256"] == record(OLD / name)["sha256"]
    plan = json.loads((ROOT / "docs/style_improvement_20261003/state_prior_training_preparation_v1/root_GPU_launch_commands.json").read_text())
    plan.update(schema="stratified_prior.root_GPU_commands.v1", status="prepared_not_executed", future_effect=False, cwd=str(NEW / "code"))
    for key in ("PYTHONPATH", "PROTOMOTIONS_ASSET_ROOT"):
        plan["env"][key] = plan["env"][key].replace(str(OLD), str(NEW))
        plan["CPU_watch_env"][key] = plan["CPU_watch_env"][key].replace(str(OLD), str(NEW))
    for name, command in plan["commands"].items():
        command[:] = [value.replace(str(OLD), str(NEW)).replace(OLD_PROFILE.removeprefix("code/"), PROFILE.removeprefix("code/")) for value in command]
        command[command.index("--output") + 1] = str(OUTPUT if name == "fullfresh2000" else PILOT)
        if "--resume" in command:
            command[command.index("--resume") + 1] = str(PILOT / "step_4.pt")
    plan["watch_command"] = [value.replace(str(OLD), str(NEW)) for value in plan["watch_command"]]
    watch = plan["watch_command"]
    watch[watch.index("--training-output") + 1], watch[watch.index("--snapshot-dir") + 1] = str(OUTPUT), str(SNAPSHOTS)
    assert "--resume" not in plan["commands"]["fullfresh2000"]
    plan["source_manifest"] = str(NEW / "source_manifest.json")
    write(HERE / "root_GPU_launch_commands.json", plan)
    write(NEW / "commands.json", plan)
    manifest = deepcopy(baseline)
    manifest.update(schema="stratified_prior.isolated_training_source.v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    code_root=str(NEW / "code"), source_epoch="completed19f934_exact_plus_one_stratified_expert_profile",
                    previous_source_manifest=record(OLD / "source_manifest.json"), shared_resource_links=links,
                    profile=record(profile_path), launcher=record(NEW / "launch_isolated.py"),
                    GPU_started=False, future_effect=False, CPU_only_preparation=True,
                    candidate_inputs={"baseline934_files_exact": len(copied), "approved_math_delta": "training.expert_sampling only"},
                    actual_import_provenance={"status": "pending_actual_new_bundle_CPU_and_training_imports", "complete_runtime_finalization_claimed": False})
    manifest["files"] = []
    for relative in sorted(copied + [PROFILE]):
        path = NEW / relative
        item = record(path)
        manifest["files"].append({"path": relative, "bytes": item["bytes"], "sha256": item["sha256"],
                                  "mtime_ns": path.stat().st_mtime_ns, "device": path.stat().st_dev,
                                  "inode": path.stat().st_ino, "copy_not_hardlink": True})
    assert len(manifest["files"]) == 935
    write(NEW / "source_manifest.json", manifest)
    for row in manifest["files"]:
        (NEW / row["path"]).chmod(0o444)
    for name in ("source_manifest.json", "launch_isolated.py", "watch_snapshots.py", "commands.json"):
        (NEW / name).chmod(0o444)
    write(HERE / "registration_prepared.json", {
        "schema": "stratified_prior.prepared_fresh2000_experiment.v1", "status": "prepared_not_executed",
        "source_manifest": record(NEW / "source_manifest.json"), "profile": record(profile_path),
        "baseline_source": record(OLD / "source_manifest.json"), "baseline_checkpoint_sha256": "19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f",
        "baseline_actual_tracking": record(ROOT / "output/diffusion_amp/state_prior_tracking2000_v1/result.json"),
        "commands": record(HERE / "root_GPU_launch_commands.json"), "single_mathematical_variable": "expert sampling prior",
        "training_clips": 2848, "validation_clips": 22, "sealed_test_clips": 34, "fullfresh_updates": 2000, "actors": 256, "rollout_steps": 32, "transitions": 16384000,
        "AMP_internal_scale": 2, "AMP_outer_weight": 2, "task_weight": .5, "public_state_prior": True,
        "future_checkpoint": {"path": str(SNAPSHOTS / "step_2000.pt"), "sha256": None, "bytes": None},
        "future_evaluation": {"tracking": "all22 validation clips/full11982 frames, including every failed clip; same-frame reference/phase/initial pairing to19f",
                              "fixture": "all4 actors x6000 controls/200s, original guide20 eta0 routes and source-bound raw contacts/falls/ground measurements",
                              "native_progression": "only after actual terminal and complete fixture zero falls/static/peer; then native4 actual zero-physical before15",
                              "metrics": "unchanged inclusive90 MMD and parent-relative/world joint diagnostics; toe fixed-reference stance explanation, not sole-slip gate",
                              "sealed34_outputs_used": False, "score_filtered_launches": False},
        "future_model_or_quality_claimed": False, "training_started": False, "GPU_used": False,
        "required_before_GPU": "Independent source/profile/real-data CPU peer; root owns serial pilot4/resume4to8/fullfresh2000 launch and complete payload admission."})
    print(json.dumps({"status": "prepared_not_executed", "934_exact_plus_one_profile": True,
                      "source_manifest": record(NEW / "source_manifest.json"), "profile": record(profile_path)}))


if __name__ == "__main__":
    main()
