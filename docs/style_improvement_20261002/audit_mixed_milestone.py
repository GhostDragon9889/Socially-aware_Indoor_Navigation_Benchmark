"""CPU payload admission for a sealed mixed snapshot before terminal result.

An immutable preflight supplies intended CLI, not an invented successful GPU
result. Final execution/counter reconciliation still requires the real result.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import torch
import yaml

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
from audit_amass_root_actual_checkpoint import fingerprint, tensor_stats
from audit_amass_root_continuation import adam_summary, moment_summary, rng_admission
from audit_mixed_actual_checkpoint import ROOT, archive_admission, physical_admission


def archived_arguments(argv):
    """Read frozen CLI registrations/defaults without today's overwrite guard.

    Existing output is expected for a historical preflight. The full prelaunch
    resource validation is an immutable input, not rerun or invented here.
    """
    tree = ast.parse((ROOT/"CrowdSim/humanoid/application/configuration.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_amp_args")
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets))
    function.body = function.body[:index+1]+[ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    description = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "AMP_CLI_DESCRIPTION" for target in node.targets))
    namespace = {"argparse": argparse, "AMP_CLI_DESCRIPTION": description}
    exec(compile(module, "frozen_cli_registration_only", "exec"), namespace)
    return vars(namespace["parse_amp_args"](argv))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--metrics-prefix", type=Path,
                        help="Optional root-sealed immutable prefix; never point at growing metrics.jsonl")
    parser.add_argument("--metrics-sha256")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Fresh output and explicitly empty CUDA visibility required")
    if bool(args.metrics_prefix) != bool(args.metrics_sha256):
        parser.error("Optional immutable metrics prefix requires its expected SHA")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    report = {"schema": "mixed_sealed_milestone_payload_admission.v1", "status": "failed", "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "heldout_outputs_read": False,
              "real_terminal_result_read": False, "terminal_execution_reconciliation_performed": False,
              "preflight_is_not_a_GPU_success_result": True, "checks": checks, "script": fingerprint(__file__)}
    try:
        source, archive = archive_admission(args.source_manifest, args.source_manifest_sha256, check)
        report.update(source_manifest=source, source_archive=archive)
        before, preflight_id = fingerprint(args.checkpoint), fingerprint(args.preflight)
        check("root_sealed_checkpoint_and_preflight_exact_SHA", before["sha256"] == args.checkpoint_sha256
              and preflight_id["sha256"] == args.preflight_sha256)
        preflight = json.loads(args.preflight.read_text())
        cli = archived_arguments(preflight["argv"])
        check("intended_explicit_fresh_mixed256_recipe", preflight["status"] == "passed"
              and preflight["resources"]["status"] == "local_file_preflight_passed"
              and cli["num_envs"] == 256 and cli["rollout_steps"] == 32 and not cli["resume"]
              and cli["evaluate_steps"] == 0 and cli["trajectory_source"] == "mixed_amass_procedural"
              and cli["curriculum_mode"] == "smooth" and cli["curriculum_speed_max_mps"] == 2.
              and cli["curriculum_accel_max_mps2"] == 1.5 and cli["episode_steps"] == 300
              and cli["replan_steps"] == 15 and cli["terrain_extent_m"] == 192. and cli["spawn_margin_m"] == 48.
              and cli["synchronized_replanning"] is True and cli["reference_adapter"] == "none")
        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
        runner, update = state["runner_state"], state["update_count"]
        transitions, n, batch = state["physical_transition_count"], 256, 8192
        check("actual_committed_counter_without_terminal_result", type(update) is int and 1 <= update <= cli["updates"]
              and state["format"] == "navisaaclab_method_reconstruction_training" and state["version"] == 1
              and state["physical_runner_state_included"] is True
              and args.checkpoint.name == f"step_{update}.pt"
              and transitions == batch*update == runner["control_step"]*n)
        expected = {"replan_steps": 15, "episode_steps": 300, "planner_batch_size": cli["planner_batch_size"],
                    "trajectory_source": "mixed_amass_procedural", "planning_clock": "synchronized"}
        check("exact_saved_runner_contract_no_unknown_keys", runner["runner_contract"] == expected)
        config = state["config"]
        profile = yaml.safe_load((ROOT/cli["config"]).read_text())
        projection = json.loads(json.dumps(config))
        check("profile_all_training_fields_except_resolved_action_mask_exact", all(
            projection[key] == value for key, value in profile["training"].items() if key != "action_mask"))
        check("saved_same_public_initialization_uniform_experts_and_no_diffusion_rollout",
              state["provenance"]["trajectory_source"] == "mixed_amass_procedural"
              and state["provenance"]["diffusion_used_for_training"] is False
              and state["provenance"]["initialization"]["sha256"] == profile["initialization"]["checkpoint_sha256"]
              and config["expert_sampling"] is None and "expert_sampling" not in runner
              and config["expert_dof_velocity_frame"] == "child_joint")
        finite = tensor_stats(state)
        check("complete_serialized_model_Adam_RMS_replay_physical_and_RNG_Tensors_finite", finite["finite"], finite)
        trainer, restored = PaperAMPTrainer.load(args.checkpoint, device="cpu", restore_rng=False)
        check("strict_CPU_training_loader_preserves_exact_model_and_counters",
              trainer.update_count == update and trainer.physical_transition_count == transitions
              and restored.keys() == runner.keys()
              and all(torch.equal(value, state["model"][key]) for key, value in trainer.model.state_dict().items()))
        del trainer, restored
        corpus = AMASSCorpus(ROOT/cli["manifest"], "train", device="cpu", seed=cli["seed"], dof_velocity_frame="child_joint")
        check("actual2848_train_dataset_and_native_child_frame_bound",
              len(corpus.ids) == 2848 and state["provenance"]["dataset"]["manifest_sha256"] == file_sha256(corpus.manifest_path)
              and state["provenance"]["dataset"]["source_sha256"] == corpus.entry["sha256"]
              and corpus.stored_dof_velocity_frame == "child_joint")
        corpus.generator.set_state(runner["corpus_rng"])
        original_rng = corpus.generator.get_state().clone()
        curriculum = MixedTrajectoryCurriculum(corpus, n, device="cpu", bounds_xy=(0., 0., 192., 192.),
            procedural_config=PacerTrajectoryConfig(mode="smooth", speed_max_mps=2., scalar_accel_max_mps2=1.5,
                                                    vector_accel_max_mps2=1.5), tail_policy="linear_velocity", seed=cli["seed"])
        curriculum.load_state_dict(runner["curriculum_state"])
        check("three_CPU_RNG_streams_admitted_with_original_not_mutated", torch.equal(original_rng, corpus.generator.get_state())
              and torch.equal(curriculum.private_corpus.generator.get_state(), runner["curriculum_state"]["amass_state"]["generator_state"])
              and torch.equal(curriculum.procedural.generator.get_state(), runner["curriculum_state"]["generator_state"]))
        actor = torch.arange(n)
        origins = torch.stack((48+(actor//16)*4, 48+(actor % 16)*4), -1).float()
        curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"], runner["reference_ids"],
                                         runner["episode_ids"], replan_steps=15, origins_xy=origins)
        check("active_real151_proc51_full_source_phase_episode_and_reset_plan_gate", True)
        stats, real = curriculum.statistics, curriculum.amass.statistics
        check("actual_fixed128_plus128_transition_exposure", stats["control_steps"] == runner["control_step"]
              and stats["amass_actor_transitions"] == stats["procedural_actor_transitions"] == transitions//2
              and real["control_actor_steps"] == transitions//2)
        replay = physical_admission(state, n, check, "milestone")
        moments, adam = moment_summary(state), adam_summary(state)
        rng = rng_admission(state["rng"])
        initial = state["provenance"]["initialization"]
        cap = config["warm_start_normalizer_count_cap"]
        increments = {"self_normalizer": batch, "task_normalizer": batch, "value_normalizer": batch,
                      "amp_normalizer": 12288*(10 if config["amp_normalization"] == "per_frame" else 1)}
        for key, value in moments.items():
            origin = initial["normalizer_counts_before_cap"][key]
            origin = min(origin, cap) if cap is not None else origin
            check("exact_committed_normalizer_count:"+key, value["count"] == origin+increments[key]*update)
        metrics_check = {"performed": False, "reason": "No immutable metrics prefix supplied; Adam/update step counts not reconciled to log"}
        if args.metrics_prefix:
            metrics_id = fingerprint(args.metrics_prefix)
            check("explicit_immutable_metrics_identity", metrics_id["sha256"] == args.metrics_sha256)
            metrics = [json.loads(line) for line in args.metrics_prefix.read_text().splitlines() if line.strip()]
            check("metrics_prefix_complete_actual_updates", [row["update_count"] for row in metrics] == list(range(1, update+1))
                  and all(row["physical_transitions"] == row["update_count"]*batch for row in metrics))
            check("Adam_steps_equal_actual_accepted_minibatches", adam["optimizer"]["common_Adam_step"]
                  == sum(row["optimizer_minibatches"] for row in metrics)
                  and adam["discriminator_optimizer"]["common_Adam_step"] == sum(row["discriminator_minibatches"] for row in metrics))
            check("metrics_prefix_immutable_after_audit", fingerprint(args.metrics_prefix) == metrics_id)
            metrics_check = {"performed": True, "artifact": metrics_id}
        check("checkpoint_and_preflight_immutable_after_admission", fingerprint(args.checkpoint) == before
              and fingerprint(args.preflight) == preflight_id)
        report.update(status="passed", checkpoint=before, preflight=preflight_id, update_count=update,
                      archived_argv=preflight["argv"], intended_arguments=cli,
                      physical_transitions=transitions, group_statistics=dict(stats), AMASS_statistics=dict(real),
                      whole_bundle_tensor_statistics=finite, normalizers=moments, Adam=adam, replay=replay, RNG=rng,
                      metrics_prefix_reconciliation=metrics_check)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["check_count"] = len(checks)
        report["limitations"] = ["Full serialized payload admission, not terminal runtime success or locomotion convergence.",
                                "Final actual result is still required to reconcile runtime fall/reset/cap counts.",
                                "CUDA RNG and PhysX solver/contact caches not replayed.",
                                "No mutable last.pt or growing metrics file is read."]
        with args.output.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
