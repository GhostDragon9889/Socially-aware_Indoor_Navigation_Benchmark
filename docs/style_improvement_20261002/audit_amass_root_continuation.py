"""CPU admission and incremental metadata audit for sealed AMASS-root resumes.

Preparation reads only the immutable starting snapshot and frozen preflight.
Final mode requires a sealed ending snapshot plus terminal result and metrics.
It deliberately does not use the fresh-step1000 terminal audit's assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import tarfile

import numpy as np
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
from audit_amass_root_actual_checkpoint import fingerprint, tensor_stats

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/style_improvement_20261002"
SOURCE = DOCS / "amass_root_linear_training_source_v1.json"


def rng_admission(state):
    """Validate serialized RNG on private CPU objects; no global restoration."""
    random.Random().setstate(state["python"])
    value = state["numpy"]
    np.random.RandomState().set_state((value[0], np.asarray(value[1], dtype=np.uint32),
                                      value[2], value[3], value[4]))
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    assert all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
               and value.ndim == 1 and value.numel() > 0 for value in state["torch_cuda"])
    return {"python_numpy_private_validators_accepted": True,
            "torch_CPU_private_generator_accepted": True,
            "CUDA_serialized_uint8_bytes": [value.numel() for value in state["torch_cuda"]],
            "CUDA_RNG_restored": False}


def moment_summary(state):
    result = {}
    for name, value in state["normalizers"].items():
        assert bool(torch.isfinite(value["mean"]).all())
        assert bool(torch.isfinite(value["variance"]).all()) and bool((value["variance"] >= 0).all())
        std = value["variance"].sqrt()
        result[name] = {"count": float(value["count"]), "dimensions": value["mean"].numel(),
                        "std_min": float(std.min()), "std_median": float(std.median()),
                        "std_max": float(std.max())}
        if name == "task_normalizer":
            height = std[-1024:]
            result[name]["height1024_std"] = {"min": float(height.min()),
                                              "median": float(height.median()), "max": float(height.max())}
    return result


def adam_summary(state):
    result = {}
    for name in ("optimizer", "discriminator_optimizer"):
        value = state[name]
        ids = [item for group in value["param_groups"] for item in group["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == set(value["state"])
        steps = {float(item["step"]) for item in value["state"].values()}
        assert len(steps) == 1
        result[name] = {"parameter_states": len(ids), "common_Adam_step": steps.pop(),
                        "learning_rates": [group["lr"] for group in value["param_groups"]]}
    return result


def payload_admission(path, state, corpus, arguments, check, label):
    n = arguments["num_envs"]
    runner = state["runner_state"]
    check(label+":format_complete", state["format"] == "navisaaclab_method_reconstruction_training"
          and state["version"] == PaperAMPTrainer.checkpoint_version
          and state["physical_runner_state_included"] is True and isinstance(runner, dict) and bool(runner))
    check(label+":committed_clock_budget", state["physical_transition_count"]
          == state["update_count"]*n*arguments["rollout_steps"] == runner["control_step"]*n)
    finite = tensor_stats(state)
    check(label+":entire_bundle_finite_including_replay", finite["finite"], finite)
    trainer, loaded_runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
    check(label+":strict_current_CPU_loader_accepts", trainer.update_count == state["update_count"]
          and trainer.physical_transition_count == state["physical_transition_count"]
          and loaded_runner.keys() == runner.keys()
          and all(torch.equal(value, state["model"][key]) for key, value in trainer.model.state_dict().items()))
    del trainer, loaded_runner
    check(label+":bound_native_child_dataset", state["provenance"]["dataset"]["manifest_sha256"]
          == file_sha256(corpus.manifest_path)
          and state["provenance"]["dataset"]["source_sha256"] == corpus.entry["sha256"]
          and state["config"]["expert_dof_velocity_frame"] == "child_joint")
    curriculum = AMASSRootTrajectoryCurriculum(corpus, n, tail_policy="linear_velocity",
        bounds_xy=[0., 0., arguments["terrain_extent_m"], arguments["terrain_extent_m"]])
    curriculum.load_state_dict(runner["curriculum_state"])
    side = math.isqrt(n)
    check(label+":exact_square_spawn_grid", side*side == n)
    ids = torch.arange(n)
    origins = torch.stack((arguments["spawn_margin_m"]+(ids//side)*4,
                           arguments["spawn_margin_m"]+(ids % side)*4), -1).float()
    curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"],
        runner["reference_ids"], runner["episode_ids"], replan_steps=arguments["replan_steps"], origins_xy=origins)
    check(label+":native151_phase_fixed_offset_episode_stale_reset_plan_gate", True)
    check(label+":corpus_curriculum_shared_RNG_exact", torch.equal(runner["corpus_rng"],
                                                                   runner["curriculum_state"]["generator_state"]))
    stats = curriculum.statistics
    check(label+":cumulative_curriculum_steps_and_linear_tail", stats["control_actor_steps"]
          == state["physical_transition_count"] and stats["held_future_knots"] == 0
          and stats["extrapolated_future_knots"] == stats["beyond_source_future_knots"]
          and stats["future_knots"] == 50*stats["plans_generated"])
    physical = runner["physical"]
    shapes = {"root_pos": (n, 3), "root_rot": (n, 4), "root_vel": (n, 3),
              "root_ang_vel": (n, 3), "dof_pos": (n, 69), "dof_vel": (n, 69)}
    check(label+":COMMON_physical_fields_and_unit_XYZW", set(physical) == set(shapes)
          and all(physical[key].shape == shape for key, shape in shapes.items())
          and torch.allclose(physical["root_rot"].norm(dim=-1), torch.ones(n), atol=1e-4, rtol=0))
    check(label+":history_same_episode", runner["amp_history"].shape == (n, 10, 195)
          and runner["amp_history_episode_ids"].shape == (n, 10)
          and bool((runner["amp_history_episode_ids"] == runner["episode_ids"][:, None]).all())
          and runner["age"].dtype == runner["episode_ids"].dtype == torch.long)
    replay = state["policy_replay"]
    inactive = torch.tensor(state["config"]["action_mask"]) == 0
    check(label+":replay_counters_shape_and_frozen_actions", replay["capacity"] >= n*arguments["rollout_steps"]
          and replay["size"] == min(replay["accepted"], replay["capacity"])
          and replay["cursor"] == replay["accepted"] % replay["capacity"]
          and replay["history"].shape == (replay["size"], 10, 195)
          and replay["actions"].shape == (replay["size"], 69)
          and not bool((replay["actions"][:, inactive] != 0).any()))
    return {"updates": state["update_count"], "physical_transitions": state["physical_transition_count"],
            "control_steps": runner["control_step"], "whole_bundle_tensor_statistics": finite,
            "curriculum_statistics": stats, "normalizers": moment_summary(state),
            "optimizers": adam_summary(state), "RNG": rng_admission(state["rng"]),
            "policy_replay": {key: replay[key] for key in ("capacity", "size", "cursor", "accepted")},
            "newly_reset_actors": int((runner["age"] == 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--fresh-result", type=Path, required=True)
    parser.add_argument("--fresh-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal = args.checkpoint is not None
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        parser.error("Empty CUDA visibility and a fresh output are required")
    if any(value is not None for value in (args.checkpoint, args.result, args.metrics)) and not all(
            value is not None for value in (args.checkpoint, args.result, args.metrics)):
        parser.error("Final audit requires checkpoint, terminal result, and metrics together")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    input_paths = {"source_checkpoint": args.source_checkpoint, "preflight": args.preflight,
                   "fresh_result": args.fresh_result, "fresh_metrics": args.fresh_metrics}
    if terminal:
        input_paths.update(checkpoint=args.checkpoint, result=args.result, metrics=args.metrics)
    before = {key: fingerprint(value) for key, value in input_paths.items()}
    source_manifest = json.loads(SOURCE.read_text())
    archive = ROOT/source_manifest["source_archive"]["path"]
    check("source124_archive_SHA_size", fingerprint(archive)["sha256"] == source_manifest["source_archive"]["sha256"]
          and archive.stat().st_size == source_manifest["source_archive"]["bytes"])
    allowed = {row["path"]: row for row in source_manifest["files"]}
    check("unique124_frozen_sources", len(allowed) == len(source_manifest["files"]) == 124)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        check("source_archive_safe_exact_paths", len(members) == len(allowed)
              and {item.name for item in members} == set(allowed)
              and all(item.isfile() and not PurePosixPath(item.name).is_absolute()
                      and ".." not in PurePosixPath(item.name).parts for item in members))
        for item in members:
            raw = stream.extractfile(item).read()
            check("archive:"+item.name, len(raw) == allowed[item.name]["bytes"]
                  and hashlib.sha256(raw).hexdigest() == allowed[item.name]["sha256"])
    runtime_rows = [row for path, row in allowed.items() if path.startswith(("CrowdSim/humanoid/", "protomotions/"))
                    or path == "scripts/train_paper_amp.py"]
    check("actual_CPU_loader_and_training_business_sources_match_source124", all(
        file_sha256(ROOT/row["path"]) == row["sha256"] for row in runtime_rows),
        {"checked_business_files": len(runtime_rows), "not_using_current_docs_as_historical_source": True})
    preflight = json.loads(args.preflight.read_text())
    arguments = preflight["arguments"]
    fresh = json.loads(args.fresh_result.read_text())
    check("preflight_is_explicit_resumed_AMASS_training", preflight["resource_preflight"]["status"] == "local_file_preflight_passed"
          and arguments["evaluate_steps"] == 0 and arguments["trajectory_source"] == "amass_root"
          and Path(arguments["resume"]).resolve() == args.source_checkpoint.resolve())
    allowed_changes = {"output", "updates", "resume", "checkpoint_every"}
    changed = {key: {"fresh": fresh["arguments"].get(key), "continuation": arguments.get(key)}
               for key in set(arguments)|set(fresh["arguments"]) if arguments.get(key) != fresh["arguments"].get(key)}
    check("only_declared_resume_budget_output_checkpoint_frequency_arguments_changed", set(changed) <= allowed_changes, changed)
    starting = torch.load(args.source_checkpoint, weights_only=True, map_location="cpu", mmap=True)
    check("starting_snapshot_name_matches_committed_update", args.source_checkpoint.name == f"step_{starting['update_count']}.pt")
    corpus = AMASSCorpus(ROOT/arguments["manifest"], split="train", device="cpu", seed=arguments["seed"],
                         dof_velocity_frame="child_joint")
    beginning = payload_admission(args.source_checkpoint, starting, corpus, arguments, check, "source")
    config = starting["config"]
    check("starting_training_config_matches_original_fresh", json.loads(json.dumps(config)) == fresh["training_config"])
    old_rows = [json.loads(line) for line in args.fresh_metrics.read_text().splitlines()]
    old_rows = [row for row in old_rows if row["update_count"] <= starting["update_count"]]
    batch = arguments["num_envs"]*arguments["rollout_steps"]
    check("original_prefix_update_and_budget_sequence", [row["update_count"] for row in old_rows]
          == list(range(1, starting["update_count"]+1))
          and [row["physical_transitions"] for row in old_rows] == [batch*i for i in range(1, starting["update_count"]+1)])
    cap = config["warm_start_normalizer_count_cap"]
    initial_counts = {key: min(value, cap) if cap is not None else value
                      for key, value in fresh["initialization"]["normalizer_counts_before_cap"].items()}
    history_factor = config["history_frames"] if config["amp_normalization"] == "per_frame" else 1
    increments = {"self_normalizer": batch, "task_normalizer": batch, "value_normalizer": batch,
                  "amp_normalizer": (batch+min(batch, 4096))*history_factor}
    for key, row in beginning["normalizers"].items():
        enabled = config["normalize_values"] if key == "value_normalizer" else config["normalize_observations"]
        check("source_normalizer_logged_prefix:"+key, row["count"]
              == initial_counts[key]+(increments[key]*starting["update_count"] if enabled else 0))
    for key, metric in (("optimizer", "optimizer_minibatches"), ("discriminator_optimizer", "discriminator_minibatches")):
        check("source_Adam_logged_prefix:"+key, beginning["optimizers"][key]["common_Adam_step"]
              == sum(row[metric] for row in old_rows))
    planned_updates = starting["update_count"]+arguments["updates"]
    output = {"schema": "amass_root_sealed_continuation_independent_audit.v1", "status": "passed",
        "preparation_only": not terminal, "terminal_checkpoint_audited": terminal,
        "CPU_only": True, "SDK_started": False, "GPU_used": False, "heldout34_test_outputs_read": False,
        "source_manifest": fingerprint(SOURCE), "source_archive": fingerprint(archive), "starting_payload": beginning,
        "additional_updates": arguments["updates"], "additional_transitions": arguments["updates"]*batch,
        "cumulative_updates_expected": planned_updates, "cumulative_transitions_expected": planned_updates*batch,
        "argument_changes": changed, "claim": "continuation from sealed starting snapshot, not the original uninterrupted fresh run at this update",
        "limits": ["No PhysX contact/solver caches are serialized; bitwise physics replay equivalence is not proven.",
                   "Private CPU RNG admission does not restore CUDA RNG or launch the SDK.",
                   "No heldout test34 output, pose-style score, or tracking-decomposition result is read."]}
    if terminal:
        result = json.loads(args.result.read_text())
        rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
        ending = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
        check("terminal_arguments_exact_frozen_preflight", result["arguments"] == arguments)
        check("terminal_checkpoint_name_update_budget", args.checkpoint.name == f"step_{planned_updates}.pt"
              and result["status"] == "passed" and ending["update_count"] == result["update_count"] == planned_updates
              and ending["physical_transition_count"] == result["physical_transition_count"] == planned_updates*batch
              and result["actual_control_steps"] == arguments["updates"]*arguments["rollout_steps"])
        check("continuation_logged_update_sequence", [row["update_count"] for row in rows]
              == list(range(starting["update_count"]+1, planned_updates+1))
              and [row["physical_transitions"] for row in rows]
              == [batch*i for i in range(starting["update_count"]+1, planned_updates+1)])
        check("logged_scalars_finite_no_Diff_rollouts", all(
            all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
            and row["diffusion_plans"] == row["planner_forward_calls"] == 0 for row in rows))
        check("all_loss_hyper_dataset_physics_and_runner_contracts_unchanged", ending["config"] == config
              and ending["provenance"] == starting["provenance"]
              and ending["runner_state"]["runner_contract"] == starting["runner_state"]["runner_contract"]
              and ending["runner_state"]["curriculum_state"]["config"] == starting["runner_state"]["curriculum_state"]["config"])
        check("terminal_config_and_curriculum_actual_result_equal", json.loads(json.dumps(config)) == result["training_config"]
              and ending["runner_state"]["curriculum_state"]["statistics"] == {key: result["curriculum"]["statistics"][key]
                  for key in AMASSRootTrajectoryCurriculum.STAT_FIELDS})
        final = payload_admission(args.checkpoint, ending, corpus, arguments, check, "terminal")
        for key, row in final["normalizers"].items():
            enabled = config["normalize_values"] if key == "value_normalizer" else config["normalize_observations"]
            expected = beginning["normalizers"][key]["count"]+(increments[key]*arguments["updates"] if enabled else 0)
            check("incremental_normalizer_count:"+key, row["count"] == expected, {"expected": expected})
        for key, metric in (("optimizer", "optimizer_minibatches"), ("discriminator_optimizer", "discriminator_minibatches")):
            check("incremental_Adam_steps:"+key, final["optimizers"][key]["common_Adam_step"]
                  == beginning["optimizers"][key]["common_Adam_step"]+sum(row[metric] for row in rows)
                  and ending[key]["param_groups"] == starting[key]["param_groups"])
        delta = {key: final["curriculum_statistics"][key]-beginning["curriculum_statistics"][key]
                 for key in AMASSRootTrajectoryCurriculum.STAT_FIELDS}
        check("segment_counters_are_deltas_without_initial_resets", all(value >= 0 for value in delta.values())
              and delta["control_actor_steps"] == arguments["updates"]*batch
              and delta["resets"] == rows[-1]["resets"] == result["resets"]
              and delta["fall_terminations"] == rows[-1]["fall_resets"] == result["fall_resets"]
              and delta["clip_end_truncations"]+delta["cap_only_truncations"]
              == rows[-1]["time_limit_resets"] == result["time_limit_resets"], delta)
        check("replay_kept_full_and_accepted_count_monotonic", final["policy_replay"]["capacity"] == beginning["policy_replay"]["capacity"]
              and final["policy_replay"]["accepted"] >= beginning["policy_replay"]["accepted"]
              and final["policy_replay"]["size"] == rows[-1]["replay_size"])
        output.update(terminal_payload=final, segment_curriculum_statistics=delta,
            segment_fall_resets=result["fall_resets"], segment_time_limit_resets=result["time_limit_resets"],
            elapsed_s=result["elapsed_s"], fresh_initialization_performed=False,
            model_tensor_count=len(ending["model"]),
            model_tensors_changed_from_start=sum(not torch.equal(value, starting["model"][key]) for key, value in ending["model"].items()))
    after = {key: fingerprint(value) for key, value in input_paths.items()}
    check("all_input_artifacts_immutable_during_CPU_audit", before == after)
    output.update(inputs=after, script=fingerprint(__file__), checks=checks, check_count=len(checks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "preparation_only": not terminal,
                      "checks": len(checks), "planned_cumulative_updates": planned_updates}))


if __name__ == "__main__":
    main()
