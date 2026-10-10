"""Copy q935, apply its exact staged pre-step runner delta and add one recipe."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / "state_only_matched_q_preparation_v1"
OLD = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"
NEW = ROOT / "output/isolated_training/mixed_state_terrain_state_prior_stratified_matched_v1"
PROFILE = "code/configs/diffusion_amp/isolated_state_only_matched_q_absolute.yaml"
OLD_PROFILE = "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"
RUNNER = "code/scripts/train_paper_amp.py"
OUTPUT = ROOT / "output/diffusion_amp/amp_style_state_only_matched_q_isolated_v1"
PILOT = ROOT / "output/diffusion_amp/amp_style_state_only_matched_q_isolated_pilot_v1"
SNAPSHOTS = ROOT / "output/diffusion_amp/amp_style_state_only_matched_q_isolated_snapshots_v1"


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
    assert record(OLD / "source_manifest.json")["sha256"] == "2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359"
    baseline = json.loads((OLD / "source_manifest.json").read_text())
    delta = json.loads((STAGED / "source_delta.json").read_text())
    assert len(baseline["files"]) == 935
    assert record(OLD / RUNNER) == delta["parent_runner"]
    assert record(STAGED / "train_paper_amp.py") == delta["staged_runner"]
    assert record(STAGED / "state_only_matched_q_absolute.yaml") == delta["staged_profile"]
    assert not any(path.exists() or path.is_symlink() for path in (NEW, OUTPUT, PILOT, SNAPSHOTS))
    NEW.mkdir()
    for row in baseline["files"]:
        source, destination = OLD / row["path"], NEW / row["path"]
        identity = record(source)
        assert (identity["bytes"], identity["sha256"]) == (row["bytes"], row["sha256"])
        assert not source.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert record(destination)["sha256"] == row["sha256"]
        assert source.stat().st_ino != destination.stat().st_ino and destination.stat().st_nlink == 1
    runner = NEW / RUNNER
    runner.chmod(0o644)
    before = runner.read_text()
    for operation in delta["runner_operations"]:
        assert before.count(operation["before"]) == 1
        before = before.replace(operation["before"], operation["after"])
    assert before.encode() == (STAGED / "train_paper_amp.py").read_bytes()
    runner.write_text(before)
    assert record(runner)["sha256"] == delta["staged_runner"]["sha256"]
    profile = NEW / PROFILE
    profile.write_bytes((STAGED / "state_only_matched_q_absolute.yaml").read_bytes())
    (NEW / "runtime_scratch").mkdir()
    links = deepcopy(baseline["shared_resource_links"])
    for row in links:
        if row["path"] == "code/output":
            row["target"] = str(NEW / "runtime_scratch")
        (NEW / row["path"]).symlink_to(row["target"], target_is_directory=True)
    for name in ("launch_isolated.py", "watch_snapshots.py"):
        shutil.copy2(OLD / name, NEW / name)
        assert record(NEW / name)["sha256"] == record(OLD / name)["sha256"]
    plan = json.loads((HERE.parent / "state_prior_stratified_training_preparation_v1/root_GPU_launch_commands.json").read_text())
    plan.update(schema="state_only_matched_q.root_GPU_commands.v1", status="prepared_not_executed", future_effect=False,
                cwd=str(NEW / "code"), source_manifest=str(NEW / "source_manifest.json"))
    plan["commands"].pop("resume4to8")
    for settings in (plan["env"], plan["CPU_watch_env"]):
        for key in ("PYTHONPATH", "PROTOMOTIONS_ASSET_ROOT"):
            settings[key] = settings[key].replace(str(OLD), str(NEW))
    for name, command in plan["commands"].items():
        command[:] = [value.replace(str(OLD), str(NEW)).replace(OLD_PROFILE.removeprefix("code/"), PROFILE.removeprefix("code/")) for value in command]
        command[command.index("--output") + 1] = str(OUTPUT if name == "fullfresh2000" else PILOT)
        assert "--resume" not in command
    watch = [value.replace(str(OLD), str(NEW)) for value in plan["watch_command"]]
    watch[watch.index("--training-output") + 1] = str(OUTPUT)
    watch[watch.index("--snapshot-dir") + 1] = str(SNAPSHOTS)
    plan["watch_command"] = watch
    plan["pilot_seal_requirement"] = "Root observes closed pilot4, seals exclusive step4/result/metrics/owners, then requires complete state1950 and q payload audit. Separate fullfresh2000 has no resume."
    write(HERE / "root_GPU_launch_commands.json", plan)
    write(NEW / "commands.json", plan)
    manifest = deepcopy(baseline)
    manifest.update(schema="state_only_matched_q.isolated_training_source.v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    code_root=str(NEW / "code"), source_epoch="q935_parent_934_exact_one_prestep_runner_delta_one_added_state_recipe",
                    previous_source_manifest=record(OLD / "source_manifest.json"), shared_resource_links=links,
                    profile=record(profile), launcher=record(NEW / "launch_isolated.py"), GPU_started=False,
                    future_effect=False, CPU_only_preparation=True,
                    business_source_bytes_unchanged_from_v1=False, business_source_bytes_unchanged_from_v2=False,
                    candidate_inputs={"parent_code_count":935,"unchanged_parent_files":934,"modified_parent_paths":[RUNNER],
                                      "added_paths":[PROFILE],"approved_source_delta":record(STAGED / "source_delta.json"),
                                      "mechanism":"state-only D with matched q pre-step history; three flags plus three runner fragments"},
                    actual_import_provenance={"status":"pending_actual_new_bundle_CPU_and_training_imports","complete_runtime_finalization_claimed":False})
    paths = sorted({row["path"] for row in baseline["files"]} | {PROFILE})
    actual = {str(path.relative_to(NEW)) for path in (NEW / "code").rglob("*") if path.is_file() and not path.is_symlink()}
    assert set(paths) == actual
    manifest["files"] = []
    for relative in paths:
        path = NEW / relative
        item = record(path)
        manifest["files"].append({"path":relative,"bytes":item["bytes"],"sha256":item["sha256"],
                                  "mtime_ns":path.stat().st_mtime_ns,"device":path.stat().st_dev,
                                  "inode":path.stat().st_ino,"copy_not_hardlink":True})
    write(NEW / "source_manifest.json", manifest)
    for row in manifest["files"]:
        (NEW / row["path"]).chmod(0o444)
    for name in ("source_manifest.json", "launch_isolated.py", "watch_snapshots.py", "commands.json"):
        (NEW / name).chmod(0o444)
    archive = HERE / "actual_source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in paths + ["source_manifest.json","launch_isolated.py","commands.json","watch_snapshots.py"]:
            tar.add(NEW / name, arcname=name, recursive=False)
    archive.chmod(0o444)
    source_count = len(paths)
    assert source_count == len(baseline["files"]) + 1
    write(HERE / "registration_prepared.json", {
        "schema":"state_only_matched_q.prepared_fresh2000_experiment.v1","status":"prepared_not_executed",
        "source_manifest":record(NEW / "source_manifest.json"),"source_count":source_count,"source_archive":record(archive),
        "archive_regular_members":source_count + 4,"profile":record(profile),"parent_source":record(OLD / "source_manifest.json"),
        "approved_source_delta":record(STAGED / "source_delta.json"),"commands":record(HERE / "root_GPU_launch_commands.json"),
        "fresh_pilot_updates":4,"fullfresh_updates":2000,"actors":256,"rollout_steps":32,"transitions":16384000,
        "D_input":1950,"discriminator_initialization":"public_pacer_state_only","AMP_internal_scale":2.0,"AMP_outer_weight":2.0,
        "task_weight":.5,"q_expert_sampling_preserved":True,"AMP_reward_timestamp":"pre-step h[t..]; state-only discriminator ignores applied a[t]",
        "future_checkpoint":{"path":str(SNAPSHOTS / "step_2000.pt"),"sha256":None,"bytes":None},
        "training_started":False,"GPU_used":False,"SDK_started":False,"future_effect":False,"sealed34_outputs_read":False})
    print(json.dumps({"status":"prepared_not_executed","source_count":source_count,"source_manifest":record(NEW / "source_manifest.json"),"archive":record(archive)}))


if __name__ == "__main__":
    main()
