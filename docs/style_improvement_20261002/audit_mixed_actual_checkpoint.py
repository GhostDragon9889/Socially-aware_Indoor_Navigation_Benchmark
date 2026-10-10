"""CPU audit of sealed fixed-slot mixed-curriculum training checkpoints.

The same auditor admits pilot4/resume8 or fresh500/1000/2000 milestones.
It never restores global/CUDA RNG, starts physics, or reads heldout captures.
Milestones preceding the terminal result do not inherit terminal fall counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.application.resources import resolve_local_path
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
from audit_amass_root_actual_checkpoint import fingerprint, tensor_stats
from audit_amass_root_continuation import adam_summary, moment_summary, rng_admission

ROOT = Path(__file__).resolve().parents[2]


def archive_admission(path, expected_sha, check):
    check("prelaunch_manifest_explicit_identity", file_sha256(path) == expected_sha)
    source = json.loads(path.read_text())
    archive_row = source["archive"]
    archive = ROOT/archive_row["path"]
    check("prelaunch_archive_identity", file_sha256(archive) == archive_row["sha256"]
          and archive.stat().st_size == archive_row["bytes"])
    allowed = {row["path"]: row for row in source["files"]}
    check("source_manifest_unique_nonempty_paths", bool(allowed) and len(allowed) == len(source["files"]))
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        check("archive_exact_regular_relative_nontraversal_paths", len(members) == len(allowed)
              and {member.name for member in members} == set(allowed)
              and all(member.isfile() and not PurePosixPath(member.name).is_absolute()
                      and ".." not in PurePosixPath(member.name).parts for member in members))
        for member in members:
            raw = stream.extractfile(member).read()
            check("source_archive:"+member.name, len(raw) == allowed[member.name]["bytes"]
                  and hashlib.sha256(raw).hexdigest() == allowed[member.name]["sha256"])
    runtime = [row for name, row in allowed.items() if name.startswith(("CrowdSim/humanoid/", "protomotions/"))
               or name == "scripts/train_paper_amp.py"]
    check("current_CPU_admission_uses_exact_frozen_training_business", bool(runtime)
          and all(file_sha256(ROOT/row["path"]) == row["sha256"] for row in runtime),
          {"files": len(runtime), "CPU_document_changes_are_not_runtime_source_changes": True})
    return fingerprint(path), fingerprint(archive)


def physical_admission(state, n, check, prefix):
    runner = state["runner_state"]
    shape = {"root_pos": (n, 3), "root_rot": (n, 4), "root_vel": (n, 3),
             "root_ang_vel": (n, 3), "dof_pos": (n, 69), "dof_vel": (n, 69)}
    physical = runner["physical"]
    check(prefix+":COMMON_fields_shapes_unit_XYZW", set(physical) == set(shape)
          and all(physical[key].shape == value for key, value in shape.items())
          and torch.allclose(physical["root_rot"].norm(dim=-1), torch.ones(n), atol=1e-4, rtol=0))
    check(prefix+":same_episode_AMP_history", runner["amp_history"].shape == (n, 10, 195)
          and runner["amp_history_episode_ids"].shape == (n, 10)
          and torch.equal(runner["amp_history_episode_ids"], runner["episode_ids"][:, None].expand(n, 10)))
    check(prefix+":integer_clocks_planar_history", runner["age"].dtype == runner["episode_ids"].dtype == torch.long
          and bool((runner["age"] >= 0).all()) and bool((runner["episode_ids"] >= 1).all())
          and runner["planar_history"].shape == (n, 91, 5)
          and runner["history_available"].shape == (n, 91)
          and runner["history_available"].dtype == torch.bool)
    replay = state["policy_replay"]
    frozen = torch.tensor(state["config"]["action_mask"]) == 0
    check(prefix+":complete_replay_counters_and_frozen_coordinates", replay["capacity"] >= n
          and replay["size"] == min(replay["accepted"], replay["capacity"])
          and replay["cursor"] == replay["accepted"] % replay["capacity"]
          and replay["history"].shape == (replay["size"], 10, 195)
          and replay["actions"].shape == (replay["size"], 69)
          and not bool((replay["actions"][:, frozen] != 0).any()))
    return {key: replay[key] for key in ("capacity", "size", "accepted", "cursor")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--milestones", action="store_true",
                        help="Allow earlier sealed milestones paired with a later terminal result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or not len(args.checkpoint) == len(args.result) == len(args.metrics)):
        parser.error("Fresh output, empty CUDA visibility and equal input-list lengths required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checks, rows = [], []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    report = {"schema": "navisaaclab.mixed_actual_checkpoint_audit.v1", "status": "failed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False, "heldout_outputs_read": False,
              "checks": checks, "checkpoints": rows, "script": fingerprint(__file__)}
    try:
        source, archive = archive_admission(args.source_manifest, args.source_manifest_sha256, check)
        report.update(source_manifest=source, source_archive=archive)
        for path, result_path, metrics_path in zip(args.checkpoint, args.result, args.metrics, strict=True):
            prefix = path.name
            identities = {"checkpoint": fingerprint(path), "result": fingerprint(result_path),
                          "metrics": fingerprint(metrics_path)}
            result = json.loads(result_path.read_text())
            arguments = result["arguments"]
            n, rollout = arguments["num_envs"], arguments["rollout_steps"]
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            update, transitions = state["update_count"], state["physical_transition_count"]
            runner = state["runner_state"]
            check(prefix+":weights_only_complete_committed_payload", state["format"] == "navisaaclab_method_reconstruction_training"
                  and state["version"] == PaperAMPTrainer.checkpoint_version
                  and state["physical_runner_state_included"] is True and isinstance(runner, dict) and bool(runner))
            check(prefix+":fixed_even_assignment_and_budget", type(n) is int and n >= 2 and n % 2 == 0
                  and type(update) is int and update > 0 and type(transitions) is int
                  and transitions == update*n*rollout == runner["control_step"]*n
                  and path.name == f"step_{update}.pt")
            terminal = result["update_count"] == update
            check(prefix+":actual_result_admits_selected_milestone", result["status"] == "passed"
                  and (terminal or args.milestones and result["update_count"] > update)
                  and result["physical_transition_count"] == result["update_count"]*n*rollout)
            check(prefix+":mixed_training_only_recipe", arguments["evaluate_steps"] == 0
                  and arguments["trajectory_source"] == runner["runner_contract"]["trajectory_source"] == "mixed_amass_procedural"
                  and result["runtime_execution"]["diffusion_used_for_rollouts"] is False
                  and arguments["curriculum_mode"] == "smooth" and arguments["curriculum_speed_max_mps"] == 2.
                  and arguments["curriculum_accel_max_mps2"] == 1.5
                  and arguments["episode_steps"] == 300 and arguments["replan_steps"] == 15
                  and arguments["terrain_extent_m"] == 192. and arguments["spawn_margin_m"] == 48.
                  and arguments["synchronized_replanning"] is True and arguments["reference_adapter"] == "none")
            check(prefix+":training_configuration_JSON_exact", json.loads(json.dumps(state["config"])) == result["training_config"])
            finite = tensor_stats(state)
            check(prefix+":entire_model_Adam_RMS_replay_physical_RNG_bundle_finite", finite["finite"], finite)
            loaded, admitted_runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
            check(prefix+":strict_current_CPU_trainer_load_model_exact", loaded.update_count == update
                  and loaded.physical_transition_count == transitions and admitted_runner.keys() == runner.keys()
                  and all(torch.equal(value, state["model"][key]) for key, value in loaded.model.state_dict().items()))
            del loaded, admitted_runner
            corpus = AMASSCorpus(ROOT/arguments["manifest"], "train", device="cpu", seed=arguments["seed"],
                                 dof_velocity_frame=state["config"]["expert_dof_velocity_frame"])
            check(prefix+":native_child_dataset_and_bank_identity", state["config"]["expert_dof_velocity_frame"] == "child_joint"
                  and state["provenance"]["dataset"]["manifest_sha256"] == file_sha256(corpus.manifest_path)
                  and state["provenance"]["dataset"]["source_sha256"] == corpus.entry["sha256"]
                  and file_sha256(resolve_local_path(corpus.entry["path"], relative_to=corpus.manifest_path.parent)) == corpus.entry["sha256"])
            corpus.generator.set_state(runner["corpus_rng"])
            procedural = PacerTrajectoryConfig(mode="smooth", speed_max_mps=2., scalar_accel_max_mps2=1.5,
                                               vector_accel_max_mps2=1.5)
            curriculum = MixedTrajectoryCurriculum(corpus, n, device="cpu", bounds_xy=(0., 0., 192., 192.),
                tail_policy="linear_velocity", procedural_config=procedural, seed=arguments["seed"])
            original_rng = corpus.generator.get_state().clone()
            curriculum.load_state_dict(runner["curriculum_state"])
            check(prefix+":three_RNG_states_valid_distinct_ownership_and_top_proc_alias",
                  torch.equal(corpus.generator.get_state(), original_rng)
                  and torch.equal(curriculum.private_corpus.generator.get_state(), runner["curriculum_state"]["amass_state"]["generator_state"])
                  and torch.equal(curriculum.procedural.generator.get_state(), runner["curriculum_state"]["generator_state"])
                  and torch.equal(runner["curriculum_state"]["generator_state"], runner["curriculum_state"]["procedural_state"]["generator_state"]))
            side = math.isqrt(n)
            check(prefix+":square_spawn_grid", side*side == n)
            actor = torch.arange(n)
            origins = torch.stack((48+(actor//side)*4, 48+(actor % side)*4), -1).float()
            curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"],
                runner["reference_ids"], runner["episode_ids"], replan_steps=15, origins_xy=origins)
            check(prefix+":real151_proc51_episode_phase_reset_and_active_resume_gate", True)
            stats, real = curriculum.statistics, curriculum.amass.statistics
            check(prefix+":both_groups_exact_actual_half_transition_exposure",
                  stats["control_steps"] == runner["control_step"]
                  and stats["amass_actor_transitions"] == stats["procedural_actor_transitions"] == transitions//2
                  and real["control_actor_steps"] == transitions//2)
            check(prefix+":real_tail_linear_only_and_all_plans_counted", real["held_future_knots"] == 0
                  and real["extrapolated_future_knots"] == real["beyond_source_future_knots"]
                  and real["future_knots"] == 50*real["plans_generated"]
                  and curriculum.plans_generated == stats["amass_plans"]+stats["procedural_plans"])
            replay = physical_admission(state, n, check, prefix)
            check(prefix+":uniform_expert_sampler_unchanged", state["config"]["expert_sampling"] is None
                  and "expert_sampling" not in runner)
            moments, adam, rng = moment_summary(state), adam_summary(state), rng_admission(state["rng"])
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            selected = [row for row in metrics if row["update_count"] <= update]
            check(prefix+":metrics_contiguous_actual_committed_budget", bool(selected)
                  and [row["update_count"] for row in selected] == list(range(selected[0]["update_count"], update+1))
                  and all(row["physical_transitions"] == row["update_count"]*n*rollout for row in selected))
            # Public transferred counts are capped once. These counts are cumulative,
            # even when metrics/result describe only a restarted segment.
            initialization = state["provenance"]["initialization"]
            cap = state["config"]["warm_start_normalizer_count_cap"]
            initial_counts = {key: min(value, cap) if cap is not None else value
                              for key, value in initialization["normalizer_counts_before_cap"].items()}
            factor = 10 if state["config"]["amp_normalization"] == "per_frame" else 1
            increments = {"self_normalizer": n*rollout, "task_normalizer": n*rollout,
                          "value_normalizer": n*rollout, "amp_normalizer": (n*rollout+min(n*rollout, 4096))*factor}
            for key, row in moments.items():
                check(prefix+":normalizer_committed_count:"+key,
                      row["count"] == initial_counts[key]+increments[key]*update)
            if selected[0]["update_count"] == 1:
                check(prefix+":Adam_steps_equal_all_actual_minibatches",
                      adam["optimizer"]["common_Adam_step"] == sum(row["optimizer_minibatches"] for row in selected)
                      and adam["discriminator_optimizer"]["common_Adam_step"] == sum(row["discriminator_minibatches"] for row in selected))
            if terminal:
                check(prefix+":terminal_parent_child_statistics_exact_result",
                      stats == result["group_transition_exposure"] == result["curriculum"]["statistics"]
                      and real == {key: result["curriculum"]["amass"]["statistics"][key] for key in real})
                resumed = bool(arguments["resume"])
                previous = next((row for row in rows if Path(row["checkpoint"]["path"]) == Path(arguments["resume"]).resolve()), None) if resumed else None
                check(prefix+":resume_source_explicitly_audited_if_resumed", not resumed or previous is not None)
                base_stats = previous["group_statistics"] if previous else dict.fromkeys(MixedTrajectoryCurriculum.STAT_FIELDS, 0)
                base_real = previous["AMASS_statistics"] if previous else dict.fromkeys(curriculum.amass.STAT_FIELDS, 0)
                base_transitions = previous["physical_transitions"] if previous else 0
                initial_reset_count = 0 if resumed else n
                falls = real["fall_terminations"]-base_real["fall_terminations"]+stats["procedural_fall_terminations"]-base_stats["procedural_fall_terminations"]
                limits = (real["clip_end_truncations"]-base_real["clip_end_truncations"]
                          +real["cap_only_truncations"]-base_real["cap_only_truncations"]
                          +stats["procedural_cap_truncations"]-base_stats["procedural_cap_truncations"])
                resets = stats["amass_resets"]+stats["procedural_resets"]-base_stats["amass_resets"]-base_stats["procedural_resets"]-initial_reset_count
                check(prefix+":actual_segment_falls_caps_clipends_resets_and_controls",
                      falls == result["fall_resets"] and limits == result["time_limit_resets"]
                      and resets == result["resets"] and result["actual_control_steps"]*n == transitions-base_transitions,
                      {"segment_falls": falls, "segment_time_limit_resets": limits, "segment_resets": resets,
                       "initial_physical_resets_excluded": initial_reset_count})
            check(prefix+":all_audited_inputs_remain_immutable", all(fingerprint(value) == identities[key]
                  for key, value in (("checkpoint", path), ("result", result_path), ("metrics", metrics_path))))
            row = {**identities, "update_count": update, "physical_transitions": transitions,
                   "control_step": runner["control_step"], "num_envs": n,
                   "terminal_result_at_this_checkpoint": terminal,
                   "result_fall_counts_apply_to_this_checkpoint": terminal,
                   "group_statistics": dict(stats), "AMASS_statistics": dict(real),
                   "normalizers": moments, "Adam": adam, "RNG": rng, "replay": replay,
                   "whole_bundle_tensors": finite, "newly_reset_actors": int((runner["age"] == 0).sum()),
                   "configuration": state["config"], "runner_contract": runner["runner_contract"],
                   "curriculum_config": runner["curriculum_state"]["config"]}
            rows.append(row)
            del state, runner, curriculum, corpus
        check("selected_updates_strictly_increase", [row["update_count"] for row in rows]
              == sorted({row["update_count"] for row in rows}))
        check("selected_exact_config_runner_and_assignment_unchanged", all(
            row["configuration"] == rows[0]["configuration"] and row["runner_contract"] == rows[0]["runner_contract"]
            and row["curriculum_config"] == rows[0]["curriculum_config"] for row in rows))
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["check_count"] = len(checks)
        report["limitations"] = [
            "Serialized logical state and strict CPU admission only; no bitwise PhysX contact/solver cache replay claim.",
            "CUDA RNG bytes are present/finite but not restored on a CUDA device by this auditor.",
            "Transport watcher SHA/counters do not replace this full payload audit.",
            "Execution/checkpoint integrity is not tracking/style/navigation quality acceptance or author reproduction.",
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
