"""Bounded CPU-only snapshots of exact atomic training-checkpoint milestones."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import types


ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path":str(path), "bytes":path.stat().st_size, "sha256":digest(path)}


def frozen_helpers(source_manifest):
    source = json.loads(source_manifest.read_text())
    archive = source["archive"]
    archive_path = Path(archive["path"])
    if not archive_path.is_absolute():
        archive_path = ROOT/archive_path
    assert digest(archive_path) == archive["sha256"]
    frozen = Path("/tmp")/f"protomotions-stratified-checkpoint-source-{archive['sha256'][:16]}"
    frozen.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as package:
        package.extractall(frozen, filter="data")
    for row in source["files"]:
        path = frozen/row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    # Empty namespace shells avoid executing package initializers not needed by
    # this CPU verifier. Every imported implementation below is archived code.
    for name in ("CrowdSim", "CrowdSim.humanoid", "CrowdSim.humanoid.application", "CrowdSim.humanoid.training"):
        module = types.ModuleType(name)
        module.__path__ = [str(frozen/name.replace(".", "/"))]
        sys.modules[name] = module
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler
    from CrowdSim.humanoid.application.settings import TrainingConfig
    return source, frozen, AMASSCorpus, ExpertReferenceSampler, TrainingConfig


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid {name}")
    return value


def validate_bundle(bundle, bridge, training_config, torch):
    if bundle.get("format") != "navisaaclab_method_reconstruction_training" or bundle.get("version") != 1:
        raise ValueError("Unknown full checkpoint format/version")
    required = {"config", "provenance", "model", "optimizer", "discriminator_optimizer", "normalizers", "rng", "policy_replay", "runner_state"}
    if not required <= bundle.keys() or not bundle.get("physical_runner_state_included"):
        raise ValueError("Checkpoint lacks complete training/physical runner state")
    config = training_config(**bundle["config"])
    config.validate()
    if config.expert_sampling != bridge.caller_config:
        raise ValueError("Training checkpoint expert configuration changed")
    update = integer(bundle["update_count"], "update_count")
    transitions = integer(bundle["physical_transition_count"], "transitions")
    if transitions != update*8192:
        raise ValueError("Update/physical transition budget does not match256x32")
    runner = bundle["runner_state"]
    runner_required = {"physical", "amp_history", "amp_history_episode_ids", "planar_history", "history_available", "age", "episode_ids", "reference_ids", "plans", "plan_age", "goals", "corpus_rng", "curriculum_state", "forced_replanning", "runner_contract", "control_step", "runtime_execution", "physics_resume", "expert_sampling"}
    if not runner_required <= runner.keys() or runner["control_step"] != update*32:
        raise ValueError("Checkpoint lacks complete matching runner/control state")
    if runner["physical"].keys() != {"root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"}:
        raise ValueError("Physical state incomplete")
    for key, value in runner["physical"].items():
        expected = (256,69 if key.startswith("dof_") else 4 if key == "root_rot" else 3)
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError(f"Physical state invalid:{key}")
    if runner["amp_history"].shape != (256,10,195) or not torch.isfinite(runner["amp_history"]).all():
        raise ValueError("AMP history invalid")
    if not torch.equal(runner["amp_history_episode_ids"], runner["episode_ids"][:,None].expand(-1,10)):
        raise ValueError("AMP history crossed episode boundaries")
    bridge.load_state_dict(runner["expert_sampling"])
    sampler = runner["expert_sampling"]["sampler"]
    if (sampler["sampled_batches"], sampler["sampled_windows"]) != (update,update*4096):
        raise ValueError("Expert sample counters do not match actual update calls")
    for label, rng in (("corpus",runner["corpus_rng"]), ("globalCPU",bundle["rng"]["torch_cpu"])):
        torch.Generator(device="cpu").set_state(rng.cpu())
    if not {"python", "numpy", "torch_cpu", "torch_cuda"} <= bundle["rng"].keys():
        raise ValueError("Full global RNG incomplete")
    if not {"self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"} <= bundle["normalizers"].keys():
        raise ValueError("Normalizer state incomplete")
    if not bundle["model"] or not bundle["optimizer"].get("state") or not bundle["discriminator_optimizer"].get("state"):
        raise ValueError("Actual model/optimizer state empty")
    if not isinstance(bundle["policy_replay"], dict) or not bundle["policy_replay"]:
        raise ValueError("Policy replay incomplete")
    return {"update_count":update, "physical_transition_count":transitions, "control_step":runner["control_step"],
            "expert_sampled_batches":sampler["sampled_batches"], "expert_sampled_windows":sampler["sampled_windows"],
            "sampler_config_sha256":sampler["config_sha256"], "sampler_registry_sha256":sampler["registry_sha256"],
            "sampler_probabilities_sha256":sampler["probabilities_sha256"], "sampler_manifest_sha256":sampler["manifest_sha256"],
            "sampler_rng_sha256":hashlib.sha256(sampler["generator_state"].cpu().numpy().tobytes()).hexdigest(),
            "complete_resume_payload_checked":True, "physical_bitwise_equivalence_claim":False}


def copy_fd_snapshot(source, destination, expected_stat):
    temporary = destination.with_name(destination.name+f".watcher-{os.getpid()}.tmp")
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        actual = os.fstat(stream.fileno())
        if (actual.st_ino,actual.st_size,actual.st_mtime_ns) != expected_stat:
            return None
        with temporary.open("xb") as output:
            for block in iter(lambda:stream.read(8*1024*1024),b""):
                output.write(block)
                hasher.update(block)
        end = os.fstat(stream.fileno())
        if (end.st_ino,end.st_size,end.st_mtime_ns) != expected_stat:
            temporary.unlink()
            raise ValueError("Opened checkpoint was mutated in place while copying")
    return temporary, hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--max-seconds", type=float, default=2700.)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    source_manifest = Path(args.source_manifest).resolve()
    source, frozen, corpus_type, bridge_type, config_type = frozen_helpers(source_manifest)
    output = Path(args.output).resolve()
    source_file = output/"last.pt"
    profile = json.loads((ROOT/"docs/style_improvement_20261002/expert_sampling_integration_preflight.json").read_text())
    config = profile["TrainingConfig_difference_from_negative_backward_baseline"]["expert_sampling"]
    corpus = corpus_type(ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json", "train", device="cpu", dof_velocity_frame="child_joint")
    bridge = bridge_type(corpus, config)
    bridge.caller_config = config
    targets = {200,500,1000}
    ledger_path = output/"snapshots.json"
    ledger = {"schema":"navisaaclab.exact_checkpoint_snapshots.v1", "status":"watching", "started_at_utc":now(),
              "CPU_only":True, "Kit_started":False, "max_watch_seconds":args.max_seconds,
              "training_source_manifest":record(source_manifest), "training_source_archive":source["archive"],
              "frozen_verification_source_root":str(frozen), "script":record(Path(__file__)),
              "observed_checkpoints":[], "snapshots":[], "missing_milestones":[], "limitations":["Full logical state checked; PhysX contact/solver caches not serialized", "No GPU/physical run or motion-quality acceptance performed"]}

    def save_ledger():
        temporary = ledger_path.with_name(ledger_path.name+".watcher.tmp")
        temporary.write_text(json.dumps(ledger,indent=2,allow_nan=False)+"\n")
        temporary.replace(ledger_path)

    save_ledger()
    last_stat = None
    started = time.monotonic()
    try:
        while time.monotonic()-started < args.max_seconds:
            if source_file.exists():
                stat = source_file.stat()
                current = (stat.st_ino,stat.st_size,stat.st_mtime_ns)
                if current != last_stat:
                    bundle = torch.load(source_file,weights_only=True,map_location="cpu",mmap=True)
                    after = source_file.stat()
                    if current != (after.st_ino,after.st_size,after.st_mtime_ns):
                        continue
                    validation = validate_bundle(bundle,bridge,config_type,torch)
                    update = validation["update_count"]
                    for target in sorted(targets):
                        if (target < update and not any(row["update_count"] == target for row in ledger["snapshots"])
                                and not any(row["update_count"] == target for row in ledger["missing_milestones"])):
                            ledger["missing_milestones"].append({"update_count":target,
                                "first_observed_later_update":update,"reason":"Original atomic checkpoint already replaced; no reconstruction or relabeling"})
                            print(f"MISSING exact milestone update={target}; observed later update={update}",flush=True)
                    if update in targets and not any(row["update_count"] == update for row in ledger["snapshots"]):
                        destination = output/f"step_{update}.pt"
                        if destination.exists():
                            raise ValueError(f"Refusing to overwrite existing snapshot:{destination}")
                        copied = copy_fd_snapshot(source_file,destination,current)
                        if copied is None:
                            continue
                        temporary, sha = copied
                        copied_bundle = torch.load(temporary,weights_only=True,map_location="cpu",mmap=True)
                        copied_validation = validate_bundle(copied_bundle,bridge,config_type,torch)
                        if copied_validation != validation:
                            raise ValueError("Copied snapshot metadata differs from the opened source version")
                        temporary.replace(destination)
                        ledger["snapshots"].append({**validation,"path":str(destination),"bytes":destination.stat().st_size,
                                                    "sha256":sha,"saved_at_utc":now(),"source_inode":stat.st_ino})
                        print(f"SNAPSHOT update={update} transitions={validation['physical_transition_count']} bytes={destination.stat().st_size} sha256={sha}",flush=True)
                    ledger["observed_checkpoints"].append({**validation,"observed_at_utc":now(),"source_inode":stat.st_ino})
                    last_stat = current
                    save_ledger()
                    print(f"VERIFIED checkpoint update={update}",flush=True)
                    del bundle
            result_path = output/"result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("status") == "failed":
                    ledger.update(status="training_failed",result=record(result_path),training_error=result.get("error"))
                    break
                if result.get("status") == "passed" and result.get("update_count") == 1000:
                    preserved = {row["update_count"] for row in ledger["snapshots"]}
                    missing = {row["update_count"] for row in ledger["missing_milestones"]}
                    if preserved | missing != targets:
                        raise ValueError("Successful1000 result has an unaccounted snapshot milestone")
                    ledger.update(status="partial_missing_milestones" if missing else "passed",result=record(result_path))
                    break
            time.sleep(5.)
        else:
            ledger["status"] = "timeout"
        ledger.update(finished_at_utc=now(),watch_elapsed_s=time.monotonic()-started)
        save_ledger()
        print(f"WATCH FINISHED status={ledger['status']}",flush=True)
        if ledger["status"] != "passed":
            raise SystemExit(1)
    except Exception as error:
        ledger.update(status="verification_failed",error=f"{type(error).__name__}:{error}",finished_at_utc=now())
        save_ledger()
        print(ledger["error"],flush=True)
        raise


if __name__ == "__main__":
    main()
