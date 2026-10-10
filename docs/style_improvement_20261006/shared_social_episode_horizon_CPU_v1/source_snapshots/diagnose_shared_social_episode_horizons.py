#!/usr/bin/env python3
"""CPU audit of completed training episodes and the fixed AMASS reset prior.

Reads actual successful S1/S2 receipts, counters and terminal POST ages. The
future81 prior is analytic feasibility only, not a trained policy or effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import episode_horizons as kernel  # noqa: E402
from scripts import diagnose_shared_social_task_context as phase  # noqa: E402
from scripts import verify_shared_social_evidence_admission as gate  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
PINNED = {**phase.PINNED,
    "scripts/diagnose_shared_social_task_context.py": "54973c30e7af6be264248268357e75a80bc908d255a43258267ffe8c0ea954e4",
    "CrowdSim/humanoid/amass_corpus.py": "2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7",
    "CrowdSim/humanoid/training/amass_trajectories.py": "ddca3403a59422045f051b5d9f653d6f411db1523840ed8bc021fc96ef82566e"}


def analyze(receipt_path, torch, manifest_record, entry, sources):
    bundle, measurements = gate.admit_post(receipt_path)
    inputs = [identity(receipt_path), *measurements.values(), *gate.bundle_sources(bundle, measurements)]
    admission = bundle["tracking"]["training_execution_admission"]
    inputs.extend(admission[key] for key in ("receipt", "result", "audit", "checkpoint"))
    result = gate.original.bound(admission["result"])
    kernel.require(result["completed_updates"] == 2000 and result["actual_training_transitions"] == 16384000
        and result["training_loop_completed"] is True and result["cleanup_succeeded"] is True
        and result["SDK_corpus_curriculum_history_checkpoint_state_included"] is False
        and result["expert_contract"]["manifest"] == manifest_record,
        "Require actual completed fixed2848 S1/S2 pilot2000")
    for scope, key, relative in (("entry_sources", "collector", "docs/style_improvement_20261004/social_world_lifetime_dependency_migration_v1/social_rollout_collector.py"),
        ("setup_sources", "World", "docs/style_improvement_20261004/shared_group_world_owned_lifetime_revision_v2/shared_group_world.py"),
        ("setup_sources", "corpus", "CrowdSim/humanoid/amass_corpus.py")):
        kernel.require(result[scope][key] == sources[relative], "Actual source differs: " + relative)
    rollouts = [row for row in result["artifacts"] if Path(row["path"]).name == "rollout_update_002000.pt"]
    kernel.require(len(rollouts) == 1 and identity(rollouts[0]["path"]) == rollouts[0], "Require uniquely sealed actual terminal rollout")
    inputs.append(rollouts[0])
    # Official PyTorch2.11: CPU remapping and lazy storage loading.
    # https://docs.pytorch.org/docs/2.11/generated/torch.load.html
    packet = torch.load(rollouts[0]["path"], weights_only=True, map_location="cpu", mmap=True)
    collection = packet["collection"]
    rollout, diagnostics = ({key: value.numpy() for key, value in collection[name].items()}
                            for name in ("rollout", "diagnostics"))
    metadata = collection["metadata"]
    checked_phase = phase.verify_phase(rollout, diagnostics, metadata)
    world = result["final_world_description"]
    curriculum = world["curriculum"]
    config = curriculum["config"]
    ac = config["amass_config"]
    kernel.require(checked_phase["controls"] == 32 and checked_phase["actors"] == 256
        and phase.same_json_document(metadata["configuration"], result["actual_configuration"])
        and metadata["training_update_count"] + 1 == result["completed_updates"]
        and metadata["training_physical_transition_count"] + checked_phase["all_actor_controls"] == result["actual_training_transitions"]
        and metadata["control_step_end"] == world["control_step"] == 64000,
        "Saved last batch differs from actual completed training")
    kernel.require(curriculum["source"] == "engineering_fixed_actor_amass_procedural_mixture"
        and config["amass_actor_ids"] == list(range(128)) and config["procedural_actor_ids"] == list(range(128, 256))
        and config["real_reset_future_frames"] == ac["future_frames"] == 31
        and config["procedural_reset_future_frames"] == 1 and ac["history_frames"] == 10
        and ac["motion_ids"] == [row["motion_id"] for row in entry["clips"]]
        and ac["manifest_sha256"] == manifest_record["sha256"] and ac["source_bank_sha256"] == entry["sha256"]
        and ac["control_dt_s"] == 1 / 30 and ac["knot_dt_s"] == .1 and ac["horizon_s"] == 5
        and ac["tail_policy"] == "linear_velocity", "Actual mixed reset/source/target protocol differs")
    driver = world["driver_reason_counts"]
    kernel.require(driver["actor_controls"] == result["actual_training_transitions"] and driver["cap_actor_controls"] == 0,
        "Complete driver/cap counts differ")
    accounting = kernel.group_accounting(driver, np.asarray(world["group_actor_rows"], np.int64),
                                        diagnostics["post_age"], diagnostics["reset_mask"])
    stats = curriculum["amass"]["statistics"]
    kernel.require(stats["future_knots"] == stats["plans_generated"] * 50
        and stats["held_future_knots"] == 0
        and stats["beyond_source_future_knots"] == stats["extrapolated_future_knots"]
        and stats["control_actor_steps"] * 2 == driver["actor_controls"], "Actual canonical50-knot counters differ")
    tail_fraction = stats["beyond_source_future_knots"] / stats["future_knots"]
    kernel.require(tail_fraction == stats["sampled_plan_beyond_source_fraction"]
        == stats["sampled_plan_extrapolated_fraction"], "Reported plan fraction differs from actual counts")
    tracking = bundle["tracking"]
    evaluation_steps = sum(row["observed_metric_steps"] for row in tracking["sequences"])
    kernel.require(len(tracking["sequences"]) == 22 and evaluation_steps == 11982, "Require original complete validation22")
    return tracking, inputs, {"post_receipt": identity(receipt_path), "training_admission": admission,
        "terminal_rollout": rollouts[0], "phase_verification": checked_phase,
        "actual_cumulative_driver_counts": driver, "actual_AMASS_plan_counters": stats,
        "canonical50_future_knots_beyond_source_fraction": tail_fraction,
        "completed_episode_accounting": accounting,
        "validation_post_controls": evaluation_steps,
        "validation_complete_sequence_mean_s": evaluation_steps / 22 / 30,
        "episode_duration_histogram_available": False, "reset_motion_ID_frame_ledger_available": False,
        "SDK_curriculum_restore_state_in_checkpoint": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-post-receipt", type=Path, required=True)
    parser.add_argument("--candidate-post-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(), "Fresh CPU-hidden CUDA output required")
    sources = {path: identity(ROOT / path) for path in PINNED}
    kernel.require(all(row["sha256"] == PINNED[path] for path, row in sources.items()), "Pinned sampling/phase source changed")
    extra_sources = [identity(path) for path in (Path(__file__), Path(kernel.__file__), Path(gate.__file__),
        Path(gate.original.__file__), ROOT / "CrowdSim/humanoid/analysis/tracking_error_decomposition.py",
        ROOT / "scripts/diagnose_shared_social_training.py")]
    manifest_record = identity(MANIFEST)
    kernel.require(manifest_record["sha256"] == MANIFEST_SHA, "Fixed2848 manifest changed")
    entry = json.loads(MANIFEST.read_text())["splits"]["train"]
    kernel.require(entry["clip_count"] == len(entry["clips"]) == 2848
        and [row["motion_id"] for row in entry["clips"]] == list(range(2848)), "Incomplete fixed2848 source registry")
    bank_path = Path(entry["path"])
    bank_record = identity(bank_path if bank_path.is_absolute() else MANIFEST.parent / bank_path)
    kernel.require(bank_record["sha256"] == entry["sha256"] and bank_record["bytes"] == entry["bytes"], "Actual train bank differs")
    import torch
    kernel.require(not torch.cuda.is_initialized(), "CPU audit must not initialize CUDA")
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    data = torch.load(bank_record["path"], weights_only=True, map_location="cpu", mmap=True)
    for row in entry["clips"]:
        mid = row["motion_id"]
        kernel.require(int(data["motion_num_frames"][mid]) == row["frames"]
            and data["motion_files"][mid] == row["source_path"]
            and abs(float(data["motion_dt"][mid]) - 1 / 30) < 1e-7, "Actual bank counts/source/30Hz timing differ")
    counts = np.asarray([row["frames"] for row in entry["clips"]], np.int64)
    priors = {f"future{future}": kernel.reset_prior(counts, future) for future in (31, 81)}
    reports, inputs, runs = [], [manifest_record, bank_record], {}
    for name, receipt in (("baseline", args.baseline_post_receipt), ("candidate", args.candidate_post_receipt)):
        tracking, bound, result = analyze(receipt, torch, manifest_record, entry, sources)
        reports.append(tracking)
        inputs.extend(bound)
        runs[name] = result
    gate.tracking_protocol(*reports)
    inputs = gate.unique_artifacts(inputs)
    kernel.require(all(identity(row["path"]) == row for row in list(sources.values()) + extra_sources + inputs),
                   "Actual input/source changed during horizon audit")
    after = np.random.get_state()
    kernel.require(random.getstate() == rng[0] and after[0] == rng[1][0] and np.array_equal(after[1], rng[1][1])
        and after[2:] == rng[1][2:] and torch.equal(torch.random.get_rng_state(), rng[2])
        and not torch.cuda.is_initialized(), "CPU audit consumed RNG or initialized CUDA")
    kernel.require(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "CPU audit imported SDK")
    report = {"schema": "shared_social.actual_episode_horizon_CPU_diagnostic.v1", "status": "passed_CPU_accounting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": list(sources.values()) + extra_sources,
        "inputs": inputs, "runs": runs, "reset_priors": priors,
        "actual_bank_frame_count_verification": {"clips": len(counts), "frames": int(counts.sum()),
            "minimum_frames": int(counts.min()), "maximum_frames": int(counts.max()), "mean_frames": float(counts.mean()),
            "all_clips_support_future81": bool((counts >= 91).all()), "bank_counts_sources_timing_match_manifest": True},
        "accounting_formula": "(cumulative actor_controls/4 - sum(final right-censored group ages)) / reset_groups /30",
        "denominator_note": "Cumulative plan counters contain50 canonical future knots per generated plan, not the10 raw task-prefix samples",
        "inference_limits": "No full reset source/frame ledger or episode histogram; source-end flags overlap physical terminals. Analytic independent two-AMASS minimum ignores contacts and falls. Neither exposure nor cause is inferred from this prior.",
        "future81_training_executed": False, "causal_training_failure_proven": False,
        "NN_constructed": False, "SDK_started": False, "CUDA_initialized": False, "RNG_consumed": False,
        "failed_frames_filtered": False, "paper_effect_achieved": False, "default_model_promoted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": identity(args.output), "runs": {name: {"completed_group_episode_mean_s":
        row["completed_episode_accounting"]["completed_group_episode_mean_s"],
        "canonical50_future_knots_beyond_source_fraction": row["canonical50_future_knots_beyond_source_fraction"]}
        for name, row in runs.items()}, "actual_bank_frame_count_verification": report["actual_bank_frame_count_verification"],
        "SDK_started": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
