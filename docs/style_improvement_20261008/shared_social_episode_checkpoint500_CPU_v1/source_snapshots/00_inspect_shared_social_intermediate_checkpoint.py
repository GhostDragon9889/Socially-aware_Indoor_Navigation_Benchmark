#!/usr/bin/env python3
"""Validate CPU model/Adam/RMS/replay loading of a saved episode checkpoint.

No optimization, inference, SDK launch, training restart or terminal admission.
The still-running training file stays untouched; only a separate copy is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_social_episode_status as status  # noqa: E402
from scripts import own_shared_social_episode_pipeline as owner  # noqa: E402
from CrowdSim.humanoid.runtime import social_popart_training as popart  # noqa: E402


def equal_trees(saved, loaded, torch):
    if isinstance(saved, torch.Tensor):
        return (isinstance(loaded, torch.Tensor) and saved.shape == loaded.shape and saved.dtype == loaded.dtype
                and saved.device.type == loaded.device.type == "cpu" and torch.equal(saved, loaded))
    if isinstance(saved, dict):
        return isinstance(loaded, dict) and saved.keys() == loaded.keys() and all(equal_trees(value, loaded[key], torch) for key, value in saved.items())
    if isinstance(saved, (tuple, list)):
        return type(saved) is type(loaded) and len(saved) == len(loaded) and all(equal_trees(a, b, torch) for a, b in zip(saved, loaded, strict=True))
    return type(saved) is type(loaded) and saved == loaded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--update", type=int, choices=(500, 1000, 1500), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require, record = owner.release.recovery.require, owner.release.record
    require(not output.exists(), "Preserve prior checkpoint inspections")
    plan = owner.admit_plan(status.PLAN)
    selected = plan["stages"][-1]
    run = Path(selected["output_directory"])
    original = run / f"step_{args.update:06d}.pt"
    require(not original.is_symlink() and original.is_file() and not original.with_name(original.name + ".tmp").exists(),
            "Require the original atomically published checkpoint")
    rows, _ = status.read_metrics(run / "metrics.jsonl", selected["runtime"])
    require(len(rows) > args.update, "Wait for the next completed update after this checkpoint save")
    protected = {Path(row["path"]).resolve() for row in plan["sources"] + plan["inputs"]}
    require(output != ROOT and not output.is_relative_to(run)
        and not any(output == path or output in path.parents for path in protected), "Use a separate inspection output")
    before, configuration = record(original), record(run / "resolved_configuration.json")
    sources = [record(Path(__file__)), record(Path(popart.__file__)), record(popart.TRAINER)]
    output.mkdir(parents=True)
    copied = output / original.name
    shutil.copyfile(original, copied)
    copied.chmod(0o444)
    identity = record(copied)
    require(record(original) == before and (identity["bytes"], identity["sha256"]) == (before["bytes"], before["sha256"]),
            "Original checkpoint changed while making its independent copy")
    import torch

    torch.set_num_threads(2)
    state = torch.load(copied, map_location="cpu", weights_only=True, mmap=True)
    require(state["update_count"] == args.update
        and state["physical_transition_count"] == args.update * selected["runtime"]["actors"] * selected["runtime"]["rollout_controls"],
        "Intermediate checkpoint budget differs from the actual logged update")
    saved_configuration = json.loads(Path(configuration["path"]).read_text())
    require(state["provenance"] == saved_configuration["provenance"], "Checkpoint and actual run provenance differ")
    module = popart.load_trainer("_independent_intermediate_CPU_popart_loader")
    trainer, runner = module.SocialAMPTrainer.load(copied, device="cpu", restore_rng=False)
    observed = {"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
        "discriminator_optimizer": trainer.discriminator_optimizer.state_dict(),
        "normalizers": {name: getattr(trainer, name).state_dict() for name in state["normalizers"]},
        "policy_replay": trainer.policy_replay.state_dict()}
    require(all(equal_trees(state[name], value, torch) for name, value in observed.items()),
            "Original strict loader did not preserve full model/Adam/RMS/replay values")
    require(trainer.update_count == state["update_count"] and trainer.physical_transition_count == state["physical_transition_count"]
        and runner is None and state["physical_runner_state_included"] is False
        and torch.cuda.is_initialized() is False
        and not any(name.startswith(("isaacsim", "isaaclab", "omni.")) for name in sys.modules),
        "Inspection must stay CPU-only; do not claim a physical World resume")
    owner.admit_plan(status.PLAN)
    require(record(original) == before and record(copied) == identity
        and record(configuration["path"]) == configuration
        and all(record(row["path"]) == row for row in sources), "Inspection input/source changed")
    report = {"schema": "shared_social.intermediate_CPU_resume_integrity.v1", "status": "passed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "original_checkpoint": before,
        "readonly_checkpoint_copy": identity, "configuration": configuration, "sources": sources,
        "updates": trainer.update_count, "physical_transitions": trainer.physical_transition_count,
        "full_model_two_Adam_four_RMS_and_replay_loaded_exactly": True,
        "normalizer_counts": {name: float(value["count"]) for name, value in observed["normalizers"].items()},
        "replay_size": observed["policy_replay"]["size"], "saved_RNG_schema_validated": True,
        "global_RNG_restored": False, "CUDA_RNG_bytes_structurally_validated_only": True,
        "physical_World_or_curriculum_resumed": False, "full_physical_resume_supported": False,
        "optimization_or_inference_performed": False, "training_restarted": False,
        "training_terminal_exit_code_observed": False, "full_2000_training_complete": False,
        "heldout_quality_evaluation": False, "CUDA_initialized": False, "SDK_started": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    owner.io.write_json(output / "result.json", report, readonly=True)
    print(json.dumps({"status": "passed", "result": record(output / "result.json"),
        "updates": args.update, "CPU_loaded_model_Adam_RMS_replay_exactly": True, "training_restarted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
