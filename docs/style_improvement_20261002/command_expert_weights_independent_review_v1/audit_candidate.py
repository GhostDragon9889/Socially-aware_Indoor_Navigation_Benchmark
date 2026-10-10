"""Read-only CPU evidence review of an explicitly unmerged loss prototype."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/style_improvement_20261002/staged_command_expert_weights_v1"
OUT = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CPU-only execution required")
    import numpy as np
    import torch

    torch.set_num_threads(4)
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    source = json.loads((SOURCE / "source_freeze.json").read_text())
    for name, expected in source["files"].items():
        actual = record(SOURCE / name)
        check("frozen_candidate_bytes:" + name,
              actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"])
    check("exact_reviewed_prototype", record(SOURCE / "prototype.py")["sha256"]
          == "3889cfcb3841abbf7b5a7ca8a0860d200d9f9610a24c67d5e3bd18f0eea32a09")
    spec = importlib.util.spec_from_file_location("unmerged_command_prototype_review", SOURCE / "prototype.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    before = torch.get_rng_state().clone()
    proof = json.loads((SOURCE / "actual_CPU_proof_final.json").read_text())
    check("actual_proof_binds_frozen_reader_and_prototype",
          proof["source"]["script"] == record(SOURCE / "actual_CPU_proof.py")
          and proof["source"]["prototype"] == record(SOURCE / "prototype.py"))
    check("actual_proof_checkpoint_current_bytes", proof["source"]["checkpoint"]
          == record(proof["source"]["checkpoint"]["path"]))
    check("actual_proof_manifest_current_bytes", proof["source"]["manifest"]
          == record(proof["source"]["manifest"]["path"]))
    check("actual_prior_selection_current_bytes", proof["source"]["expert_indices"]
          == record(proof["source"]["expert_indices"]["path"]))
    ledger_path = Path(proof["commands"]["source_run"]) / "pre_action_observations/samples.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    check("exact30_preaction_4actor_records", len(ledger) == 30
          and [row["file"] for row in ledger] == proof["source"]["actual_preaction_files"])
    commands = []
    for index, row in enumerate(ledger):
        check(f"preaction_source_phase_and_file:{index}", row["file"] == record(row["file"]["path"])
              and row["raw_observation_capture_timing"]
              == "immediately before original actor.act and before driver world.step")
        with np.load(row["file"]["path"], allow_pickle=False) as saved:
            task = saved["task_obs"].copy()
        check(f"raw_task_actual_array_SHA:{index}", task.shape == (4, 1044)
              and task.dtype == np.float32 and hashlib.sha256(task.tobytes()).hexdigest()
              == row["arrays"]["task_obs"]["sha256_C_order"])
        # Independent formula for the request's first0.5s velocity component.
        expected_velocity = torch.from_numpy(task[:, :2].copy()) / .5
        actual = module.command_context(torch.from_numpy(task), capture_phase="pre_action",
                                        observation_space="raw_task_1044")
        check(f"request_forward_lateral_formula:{index}", torch.equal(actual[:, :2], expected_velocity))
        commands.append(actual)
    commands = torch.cat(commands)
    with np.load(SOURCE / "actual_descriptors_weights_final.npz", allow_pickle=False) as saved:
        stored = {name: saved[name].copy() for name in saved.files}
    check("actual_saved120_commands_exact", np.array_equal(commands.numpy(), stored["commands"]))
    experts = torch.from_numpy(stored["expert_descriptors"])
    weights, description = module.expert_weights(commands, experts)
    check("actual512_recomputed_weights_bitwise", len(experts) == 512
          and np.array_equal(weights.numpy(), stored["weights"]))
    check("actual_descriptor_identity_and_contiguous_windows",
          np.array_equal(stored["local_frames"][:, 1:] - stored["local_frames"][:, :-1],
                         np.ones((512, 10), dtype=np.int64))
          and len(stored["motion_ids"]) == 512)
    # The actual fixture has no saturated bound. Dense kernel computation and
    # its closed-form mean-one solution independently check chunked bisection.
    scale = torch.tensor([.5, .35, 1.], dtype=torch.float64)
    delta = (experts.double()[None] - commands.double()[:, None]) / scale
    raw = .5 + 1.5 * (-.5 * delta.square().sum(-1)).exp().mean(0)
    analytic = raw / raw.mean()
    check("actual_unclipped_dense_kernel_solution_eligible", analytic.min() > .5 and analytic.max() < 2.)
    check("actual_dense_closed_form_weight_solution_bitwise", torch.equal(analytic.to(weights.dtype), weights))
    check("actual_bounds_and_mean_joint", weights.min() >= .5 and weights.max() <= 2.
          and abs(weights.double().mean().item() - 1) <= 5e-7)
    try:
        module.renormalize_bounded(torch.tensor([1e-308, 1.], dtype=torch.float64))
    except ValueError:
        check("previous_subnormal_overflow_now_rejected", True)
    else:
        check("previous_subnormal_overflow_now_rejected", False)
    check("read_only_prototype_no_CPU_RNG_draw", torch.equal(before, torch.get_rng_state()))
    check("no_CUDA_or_SDK_started", not torch.cuda.is_initialized()
          and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
    report = {"schema": "command_expert_weights.independent_readonly_review.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False,
              "production_or_frozen_candidate_modified": False, "checks": checks, "check_count": len(checks),
              "reader": record(__file__), "candidate_freeze": record(SOURCE / "source_freeze.json"),
              "actual_prior_D_checkpoint_rehashed": record(proof["source"]["checkpoint"]["path"]),
              "request_records": 30, "request_rows": 120, "sampled_expert_rows": 512,
              "weight_min": weights.min().item(), "weight_max": weights.max().item(),
              "weight_mean": weights.double().mean().item(),
              "batch_relative_ESS": description["batch_relative_ESS"],
              "full_shared_AMASS_bank_SHA_recomputed": False,
              "expert_geometry_reconstructed_by_independent_reader": False,
              "new_training_or_quality_gain_proven": False,
              "findings": [],
              "scope": "frozen unmerged CPU prototype; raw preaction provenance and actual kernel/loss-weight evidence",
              "limitations": ["No GPU implementation or integrated trainer/resume contract was executed.",
                              "Expert geometry descriptors use the frozen candidate's actual512 proof; SO3/window semantics reviewed in source and37CPU tests.",
                              "Conditional actor-task style cannot be inferred from global marginal expert weighting.",
                              "Request catch-up velocity spans future0.5s; expert motion velocity spans preceding0.333s.",
                              "Uniform sampling, negative/GP/replay/reg preservation remains an integration contract, not an implemented production hook."]}
    with (OUT / "actual_review.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "rows": [120, 512]}))


if __name__ == "__main__":
    main()
