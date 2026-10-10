"""Bounded CPU-only snapshots validated against an actual frozen AMASS-root pilot.

The training directory is read-only. A fresh, exclusive snapshot directory holds
only this watcher's copied milestones and ledger. No train/kill/resume API is used.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import types

from watch_stratified_expert_checkpoints import copy_fd_snapshot, integer, now, record


ROOT = Path(__file__).resolve().parents[2]


def frozen_helpers(source_manifest):
    source = json.loads(source_manifest.read_text())
    archive = source["source_archive"]
    archive_path = Path(archive["path"])
    archive_path = archive_path if archive_path.is_absolute() else ROOT / archive_path
    if record(archive_path) != {**archive, "path": str(archive_path.resolve())}:
        raise ValueError("Frozen source archive hash/bytes mismatch")
    frozen = Path("/tmp") / f"protomotions-amass-root-checkpoint-source-{archive['sha256'][:16]}"
    frozen.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as package:
        package.extractall(frozen, filter="data")
    for row in source["files"]:
        if record(frozen / row["path"]) != {**row, "path": str((frozen / row["path"]).resolve())}:
            raise ValueError(f"Frozen source entry mismatch: {row['path']}")
    for name in ("CrowdSim", "CrowdSim.humanoid", "CrowdSim.humanoid.application", "CrowdSim.humanoid.training"):
        namespace = types.ModuleType(name)
        namespace.__path__ = [str(frozen / name.replace(".", "/"))]
        sys.modules[name] = namespace
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler
    from CrowdSim.humanoid.application.settings import TrainingConfig
    return source, frozen, AMASSCorpus, ExpertReferenceSampler, TrainingConfig


def frozen_origins(frozen, pilot, num_envs, torch):
    """Execute only the three verified deterministic origin assignments on CPU."""
    module = ast.parse((frozen / "CrowdSim/humanoid/amp_world.py").read_text())
    world = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    constructor = next(node for node in world.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    selected = []
    for node in constructor.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"side", "ids"}:
            selected.append(node)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr == "origins":
            selected.append(node)
            break
    if len(selected) != 3:
        raise ValueError("Frozen deterministic world-origin construction not recognized")
    context = {"self": types.SimpleNamespace(device="cpu"), "num_envs": num_envs,
               "spawn_margin_m": pilot["provenance"]["physics"]["isolated_spawn_margin_m"]}
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, "<frozen CPU origins>", "exec"), {"math": math, "torch": torch}, context)
    return context["self"].origins


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def prepare_contract(pilot, corpus_type, curriculum_type, bridge_type, config_type, torch, frozen):
    config = config_type(**pilot["config"])
    config.validate()
    runner = pilot["runner_state"]
    saved = runner["curriculum_state"]
    declaration = saved["config"]
    if runner["runner_contract"]["trajectory_source"] != "amass_root" or declaration["split"] != "train":
        raise ValueError("The actual pilot must be an AMASS-root training checkpoint")
    num_envs = int(runner["physical"]["root_pos"].shape[0])
    update = integer(pilot["update_count"], "pilot update")
    if not update or runner["control_step"] % update:
        raise ValueError("Actual pilot does not declare an integer fixed rollout length")
    rollout_steps = runner["control_step"] // update
    if pilot["physical_transition_count"] != num_envs * rollout_steps * update:
        raise ValueError("Actual pilot transition/control/environment counts disagree")
    manifest = pilot["provenance"]["dataset"]["manifest"]
    corpus = corpus_type(manifest, "train", device="cpu", dof_velocity_frame=declaration["dof_velocity_frame"])
    curriculum = curriculum_type(corpus, num_envs, device="cpu", bounds_xy=declaration["bounds_xy"],
                                 root_margin_m=declaration["root_margin_m"], tail_policy=declaration["tail_policy"])
    bridge = bridge_type(corpus, config.expert_sampling) if config.expert_sampling is not None else None
    return {"config_type": config_type, "config": pilot["config"], "provenance": pilot["provenance"],
            "runner_contract": runner["runner_contract"], "runtime_execution": runner["runtime_execution"],
            "num_envs": num_envs, "rollout_steps": rollout_steps, "history_frames": config.history_frames,
            "amp_frame_dim": config.amp_frame_dim, "corpus": corpus, "curriculum": curriculum,
            "bridge": bridge, "torch": torch, "origins_xy": frozen_origins(frozen, pilot, num_envs, torch)}


def validate_bundle(bundle, contract):
    torch = contract["torch"]
    if bundle.get("format") != "navisaaclab_method_reconstruction_training" or bundle.get("version") != 1:
        raise ValueError("Unknown full training checkpoint format/version")
    required = {"config", "provenance", "model", "optimizer", "discriminator_optimizer", "normalizers", "rng", "policy_replay", "runner_state"}
    if not required <= bundle.keys() or not bundle.get("physical_runner_state_included"):
        raise ValueError("Full trainer/physical resume payload missing")
    config = contract["config_type"](**bundle["config"])
    config.validate()
    if bundle["config"] != contract["config"] or bundle["provenance"] != contract["provenance"]:
        raise ValueError("Actual pilot training configuration/provenance changed")
    update = integer(bundle["update_count"], "update_count")
    transitions = integer(bundle["physical_transition_count"], "physical_transition_count")
    actors, rollout = contract["num_envs"], contract["rollout_steps"]
    if transitions != update * actors * rollout:
        raise ValueError("Checkpoint transition count disagrees with the actual pilot budget")
    runner = bundle["runner_state"]
    fields = {"physical", "amp_history", "amp_history_episode_ids", "planar_history", "history_available", "age", "episode_ids", "reference_ids", "plans", "plan_age", "goals", "corpus_rng", "curriculum_state", "forced_replanning", "runner_contract", "control_step", "runtime_execution", "physics_resume"}
    if not fields <= runner.keys() or runner["control_step"] != update * rollout:
        raise ValueError("Matching complete runner/control state missing")
    if runner["runner_contract"] != contract["runner_contract"] or runner["runtime_execution"] != contract["runtime_execution"]:
        raise ValueError("Actual pilot runner/execution contract changed")
    if set(runner["physical"]) != {"root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"}:
        raise ValueError("Physical state incomplete")
    for key, value in runner["physical"].items():
        width = 69 if key.startswith("dof_") else 4 if key == "root_rot" else 3
        if value.shape != (actors, width) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid physical field {key}")
    if (runner["amp_history"].shape != (actors, contract["history_frames"], contract["amp_frame_dim"])
            or not torch.isfinite(runner["amp_history"]).all()):
        raise ValueError("AMP history shape/finite contract violated")
    if not torch.equal(runner["amp_history_episode_ids"], runner["episode_ids"][:, None].expand(-1, contract["history_frames"])):
        raise ValueError("AMP history crosses physical episodes")
    curriculum = contract["curriculum"]
    saved = dict(runner["curriculum_state"])
    saved["generator_state"] = saved["generator_state"].cpu()
    if not torch.equal(saved["generator_state"], runner["corpus_rng"].cpu()):
        raise ValueError("AMASS curriculum RNG differs from shared corpus RNG")
    curriculum.load_state_dict(saved)
    curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"], runner["reference_ids"], runner["episode_ids"],
                                     replan_steps=runner["runner_contract"]["replan_steps"], origins_xy=contract["origins_xy"])
    if saved["statistics"]["control_actor_steps"] != transitions:
        raise ValueError("Curriculum physical actor-step counter differs from trainer transitions")
    # The frozen public validator also verifies every active151-knot plan against
    # the actual bank/fixed offset/declared endpoint extension without build_plan.
    phase = saved["actors"]
    bridge = contract["bridge"]
    expert = runner.get("expert_sampling")
    if bridge is None and expert is not None:
        raise ValueError("Disabled expert sampling unexpectedly has enabled saved state")
    if bridge is not None:
        if expert is None:
            raise ValueError("Enabled expert sampler lacks its complete checkpoint state")
        bridge.load_state_dict(expert)
    for label, state in (("corpus", runner["corpus_rng"]), ("globalCPU", bundle["rng"]["torch_cpu"])):
        try:
            torch.Generator(device="cpu").set_state(state.cpu())
        except Exception as error:
            raise ValueError(f"Invalid {label} RNG state") from error
    if not {"python", "numpy", "torch_cpu", "torch_cuda"} <= bundle["rng"].keys():
        raise ValueError("Full global RNG payload incomplete")
    if not {"self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"} <= bundle["normalizers"].keys():
        raise ValueError("Normalizer state incomplete")
    if not bundle["model"] or not bundle["optimizer"].get("state") or not bundle["discriminator_optimizer"].get("state"):
        raise ValueError("Actual model/optimizer state empty")
    if not isinstance(bundle["policy_replay"], dict) or not bundle["policy_replay"]:
        raise ValueError("Complete policy replay payload missing")
    return {"update_count": update, "physical_transition_count": transitions, "control_step": runner["control_step"],
            "num_envs": actors, "rollout_steps": rollout, "curriculum_schema": saved["schema"],
            "curriculum_config": saved["config"], "curriculum_statistics": saved["statistics"],
            "curriculum_actor_state_hashes": {key: tensor_hash(value) for key, value in phase.items()},
            "corpus_rng_sha256": tensor_hash(runner["corpus_rng"]), "global_CPU_rng_sha256": tensor_hash(bundle["rng"]["torch_cpu"]),
            "expert_sampler_enabled": bridge is not None, "expert_sampler_state_present": expert is not None,
            "complete_resume_payload_checked": True, "active_plans_match_actual_bank": True,
            "physical_bitwise_equivalence_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--contract-checkpoint", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=2700.)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 2700:
        parser.error("Watcher deadline must be positive and at most45 minutes")
    import torch
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU watcher must not initialize CUDA")
    source, frozen, corpus_type, bridge_type, config_type = frozen_helpers(args.source_manifest.resolve())
    from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
    pilot = torch.load(args.contract_checkpoint, weights_only=True, map_location="cpu", mmap=True)
    contract = prepare_contract(pilot, corpus_type, AMASSRootTrajectoryCurriculum, bridge_type, config_type, torch, frozen)
    pilot_validation = validate_bundle(pilot, contract)
    output, snapshots = args.training_output.resolve(), args.snapshot_dir.resolve()
    if snapshots == output or output in snapshots.parents:
        raise ValueError("Snapshot directory must be separate from the training directory")
    snapshots.mkdir(parents=True, exist_ok=False)
    ledger_path = snapshots / "snapshots.json"
    ledger = {"schema": "navisaaclab.amass_root_checkpoint_snapshots.v1", "status": "watching", "started_at_utc": now(),
              "CPU_only": True, "Kit_started": False, "training_output_read_only": str(output),
              "source_manifest": record(args.source_manifest), "source_archive": source["source_archive"],
              "frozen_source_root": str(frozen), "contract_checkpoint": record(args.contract_checkpoint),
              "actual_pilot_contract_validation": pilot_validation, "script": record(__file__),
              "helper_script": record(Path(__file__).with_name("watch_stratified_expert_checkpoints.py")),
              "observed": [], "snapshots": [], "missing": [],
              "limitations": ["No PhysX solver/contact cache serialization or bitwise physical equivalence claim", "No full replay float scan or training-effect acceptance performed"]}

    def save_ledger():
        temporary = ledger_path.with_name("snapshots.watcher.tmp")
        temporary.write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
        temporary.replace(ledger_path)

    save_ledger()
    source_file = output / "last.pt"
    targets, last_stat, started = {200, 500, 1000}, None, time.monotonic()
    try:
        while time.monotonic() - started < args.max_seconds:
            if source_file.exists():
                stat = source_file.stat()
                current = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if current != last_stat:
                    bundle = torch.load(source_file, weights_only=True, map_location="cpu", mmap=True)
                    after = source_file.stat()
                    if current != (after.st_ino, after.st_size, after.st_mtime_ns):
                        continue
                    validation = validate_bundle(bundle, contract)
                    update = validation["update_count"]
                    for target in sorted(targets):
                        seen = {row["update_count"] for row in ledger["snapshots"] + ledger["missing"]}
                        if target < update and target not in seen:
                            ledger["missing"].append({"update_count": target, "first_observed_later_update": update,
                                                      "reason": "Original atomic checkpoint replaced; no reconstruction/relabeling"})
                    if update in targets and update not in {row["update_count"] for row in ledger["snapshots"]}:
                        destination = snapshots / f"step_{update}.pt"
                        copied = copy_fd_snapshot(source_file, destination, current)
                        if copied is None:
                            continue
                        temporary, sha = copied
                        copied_bundle = torch.load(temporary, weights_only=True, map_location="cpu", mmap=True)
                        if validate_bundle(copied_bundle, contract) != validation:
                            raise ValueError("Copied checkpoint state differs from the pinned source version")
                        # link publishes exclusively; it cannot replace an existing snapshot.
                        os.link(temporary, destination)
                        temporary.unlink()
                        ledger["snapshots"].append({**validation, "path": str(destination), "bytes": destination.stat().st_size,
                                                    "sha256": sha, "saved_at_utc": now(), "source_inode": stat.st_ino})
                        print(f"SNAPSHOT update={update} bytes={destination.stat().st_size} sha256={sha}", flush=True)
                    ledger["observed"].append({**validation, "observed_at_utc": now()})
                    last_stat = current
                    save_ledger()
                    print(f"VERIFIED checkpoint update={update}", flush=True)
            result_path = output / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("status") == "failed":
                    ledger.update(status="training_failed", result=record(result_path), error=result.get("error"))
                    break
                if result.get("status") == "passed" and result.get("update_count") == 1000:
                    accounted = {row["update_count"] for row in ledger["snapshots"] + ledger["missing"]}
                    if accounted == targets:
                        ledger.update(status="partial_missing_milestones" if ledger["missing"] else "passed", result=record(result_path))
                        break
            time.sleep(1.)
        else:
            ledger["status"] = "timeout"
        ledger.update(finished_at_utc=now(), watch_elapsed_s=time.monotonic() - started)
        save_ledger()
        print(f"WATCH FINISHED status={ledger['status']}", flush=True)
        return 0 if ledger["status"] == "passed" else 1
    except Exception as error:
        ledger.update(status="verification_failed", error=f"{type(error).__name__}:{error}", finished_at_utc=now())
        save_ledger()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
