"""Future closed-terminal CPU validation; no forward, optimizer step or CUDA restore."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "commands_and_audit_plan.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def sealed_json(path, sha):
    row = record(path)
    require(row["sha256"] == sha and not Path(path).stat().st_mode & 0o222,
            "Actual sealed document SHA required")
    return json.loads(Path(path).read_text()), row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-receipt", type=Path, required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU auditor must hide CUDA")
    plan = json.loads(PLAN.read_text())
    require(record(__file__) == plan["terminal_auditor"], "Registered terminal reader changed")
    parent_row = plan["source_registration"]
    require(record(parent_row["path"]) == parent_row, "Frozen source registration changed")
    parent = json.loads(Path(parent_row["path"]).read_text())
    sources = parent["source_binding"]
    for key in ("reviewed_formal_sources", "reused_staged_conditioning_and_input_sources",
                "selected_production_helpers_verified_against_epoch", "additional_reused_helpers_outside_185_epoch",
                "all_temporal_view_sources"):
        for row in sources[key]:
            require(record(row["path"]) == row, "Registered source bytes changed")
    require(record(sources["production_epoch"]["path"]) == sources["production_epoch"], "Current185 epoch changed")
    for row in plan["production_deployment_sources"]:
        require(record(row["path"]) == row, "Production deployment ABI source changed")
    owner, owner_row = sealed_json(args.owner_receipt, args.owner_sha256)
    require(type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
            and owner.get("child_exit_observed") is True and owner.get("command") == plan["formal_GPU_argv"],
            "Actual root closure and exact fresh40000 command required")
    output = Path(plan["formal_output"])
    result, result_row = sealed_json(output / "training_result.json", args.result_sha256)
    config = json.loads((output / "run_config_initial.json").read_text())
    require(config["args"] == plan["expected_fresh_arguments"]
            and config["BN_microbatch_differs_from_author_batch400"] is True
            and config["default_deployment_model_key_is_EMA"] is True, "Actual registered recipe or BN identity differs")
    require(result["status"] == "completed" and result["schema"] == "local_orca.official_five_state_formal_training_result.v2"
            and result["start_optimizer_update"] == 0 and result["completed_optimizer_updates"] == 40000
            and result["additional_optimizer_updates"] == 40000 and result["microsteps"] == 80000
            and result["training_windows"] == 16000000 and result["sampling_omitted"] is False
            and result["default_inference_weight_source"] == "EMA" and result["raw_training_state_preserved"] is True,
            "Full fresh200x2 budget or terminal state differs")
    require(len(result["updates"]) == 40000
            and all(row["optimizer_updates"] == index + 1
                    and row["microsteps"] == (index + 1) * 2
                    and row["training_windows"] == (index + 1) * 400
                    and sum(row["batch_domain_counts"].values()) == 400
                    for index, row in enumerate(result["updates"])), "Whole update/window record differs")
    snapshot = output / "step_40000.ckpt"
    snapshot_row = record(snapshot)
    require(snapshot_row["sha256"] == args.snapshot_sha256 == result["snapshot_sha256"]
            and result["snapshot"] == str(snapshot) and snapshot.stat().st_nlink == 1
            and not snapshot.stat().st_mode & 0o222
            and snapshot.stat().st_ino != (output / "last.ckpt").stat().st_ino,
            "Independent readonly terminal snapshot required")
    for domain, row in parent["data"]["manifest_records_from_actual_all_temporal_view_QC"].items():
        require(record(row["path"]) == row, "Actual domain manifest changed: " + domain)
    stage = ROOT / "docs/style_improvement_20261003/orca_official_mixed_training_staging_v2"
    for path in (ROOT, stage, stage.parent / "orca_interact_usable_staging_v1"):
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from paper_interact import PaperDiffusionConfig, PaperDiffusionPlanner
    from training_state import recipe_identity, validate_resume
    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner as ProductionPlanner
    require(not torch.cuda.is_initialized(), "CPU reader cannot initialize CUDA")
    payload = torch.load(snapshot, weights_only=True, map_location="cpu", mmap=True)
    recipe = recipe_identity(microbatch=200, accumulation=2, learning_rate=.0002,
                            max_grad_norm=None, missing_map_encoding="trace_unknown_raster")
    model_config = PaperDiffusionConfig(**config["model"])
    validate_resume(payload, config=model_config, identity=recipe, corpus_identity=plan["expected_corpus_identity"])
    require(result["training_recipe"] == recipe == config["training_recipe"]
            and payload["optimizer_updates"] == 40000 and payload["inference_weight_source"] == "EMA"
            and payload["dataset_metadata"]["initialization"] == config["initialization"]
            and config["initialization"] == plan["expected_initialization"], "Raw/EMA/initialization identity differs")
    # Original strict loading ABI, without forward or optimizer execution.
    with torch.random.fork_rng(devices=[]):
        production = ProductionPlanner(checkpoint_path=snapshot, device="cpu")
        deployment = production.model.state_dict()
        require(production.trained_steps == 40000 and deployment.keys() == payload["model"].keys()
                and all(torch.equal(value, payload["model"][key]) for key, value in deployment.items()),
                "Production Planner model key is not the exact terminal EMA")
        del deployment, production
        planner = PaperDiffusionPlanner(config=model_config, device="cpu")
        planner.model.load_state_dict(payload["model"], strict=True)
        planner.model.load_state_dict(payload["training_raw_model"], strict=True)
        optimizer = torch.optim.Adam(planner.model.parameters(), lr=.0002)
        optimizer.load_state_dict(payload["optimizer"])
        require(optimizer.state, "Complete nonempty Adam state expected after40000")
        for parameter, state in optimizer.state.items():
            require(state["exp_avg"].shape == parameter.shape == state["exp_avg_sq"].shape
                    and 0 < float(state["step"]) <= 40000, "Adam moments/step shape differs")
    sampler_next = {}
    for split, state in payload["dataset_metadata"]["sampler_rng"].items():
        generator = np.random.default_rng()
        generator.bit_generator.state = state
        count = sum(row["conditional_windows"] for row in plan["expected_corpus_identity"][split]["domains"].values())
        sampler_next[split] = generator.integers(count, size=8).tolist()
    require(set(sampler_next) == {"train", "validation", "test"}, "All3 saved union RNG streams required")
    rng = payload["rng"]
    torch.Generator(device="cpu").set_state(rng["torch"])
    random.Random().setstate(rng["python"])
    state = rng["numpy"]
    np.random.RandomState().set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    require(isinstance(rng["cuda"], list) and len(rng["cuda"]) == 1
            and all(isinstance(row, torch.Tensor) and row.device.type == "cpu" and row.dtype == torch.uint8
                    and row.ndim == 1 and row.numel() > 0 for row in rng["cuda"]), "Saved single-visible-GPU RNG byte state required")
    require(not torch.cuda.is_initialized() and not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
            "CPU reader imported SDK or initialized CUDA")
    destination = HERE / "actual_terminal40000_audit.json"
    with destination.open("x") as stream:
        json.dump({"schema": "official_orca.all_temporal_actual_formal40000_CPU_checkpoint_audit.v1", "status": "passed",
                   "CPU_only": True, "owner": owner_row, "training_result": result_row, "checkpoint": snapshot_row,
                   "reader": record(__file__), "plan": record(PLAN), "training_recipe": recipe,
                   "full_raw_EMA_Adam_strict_load_passed": True, "original_validate_resume_passed": True,
                   "production_deployment_model_key_strict_load_equal_to_EMA": True,
                   "production_deployment_sources": plan["production_deployment_sources"],
                   "production_deployment_forward_executed": False,
                   "saved3_union_CPU_next_logical_indices": sampler_next,
                   "global_CPU_RNG_states_decodable_in_isolated_objects": True,
                   "saved_CUDA_RNG_preserved_and_structure_only_validated": True,
                   "restore_rng_called": False, "full_GPU_restore_or_bitwise_resume_claimed": False,
                   "NN_forward_optimizer_step_or_sealed34_read": False, "paper_or_physical_effect_proven": False},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "output": str(destination), "optimizer_updates": 40000}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
