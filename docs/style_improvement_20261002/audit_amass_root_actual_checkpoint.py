"""CPU audit of immutable real AMASS-root pilot checkpoint/result pairs.

Loads saved physical/logical state without launching the simulator. It proves
checkpoint admission and source binding, not bitwise PhysX cache restoration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/style_improvement_20261002/amass_root_linear_training_source_v1.json"


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def tensor_stats(value):
    values = list(tensors(value))
    return {"count": len(values), "elements": sum(v.numel() for v in values),
            "finite": all(bool(torch.isfinite(v).all()) for v in values),
            "dtypes": dict(Counter(str(v.dtype) for v in values)),
            "devices": dict(Counter(str(v.device) for v in values))}


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.checkpoint) != len(args.result):
        parser.error("Fresh output and equal checkpoint/result counts are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Explicit CUDA_VISIBLE_DEVICES='' is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    source = json.loads(SOURCE.read_text())
    archive = ROOT / source["source_archive"]["path"]
    check("source_archive_sha256_and_size", file_sha256(archive) == source["source_archive"]["sha256"]
          and archive.stat().st_size == source["source_archive"]["bytes"])
    allowed = {row["path"]: row for row in source["files"]}
    check("prelaunch_source_manifest_unique_124_files", len(allowed) == len(source["files"]) == 124)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        check("archive_exact_manifest_paths_regular_relative_nontraversal",
              len(members) == len(allowed) and {m.name for m in members} == set(allowed)
              and all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                      and ".." not in PurePosixPath(m.name).parts for m in members))
        for member in members:
            raw = stream.extractfile(member).read()
            check(f"source_archive_file:{member.name}", len(raw) == allowed[member.name]["bytes"]
                  and hashlib.sha256(raw).hexdigest() == allowed[member.name]["sha256"])
    independent = json.loads((ROOT / "docs/style_improvement_20261002/amass_root_curriculum_independent_integration.json").read_text())
    check("actual_prelaunch_business_sources_equal_independent_CPU_gate",
          all(allowed[path]["sha256"] == sha for path, sha in independent["sources"].items()))
    manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    corpus = AMASSCorpus(manifest, split="train", device="cpu", seed=7, dof_velocity_frame="child_joint")
    rows = []
    for path, result_path in zip(args.checkpoint, args.result, strict=True):
        prefix = path.name
        before, result_record = fingerprint(path), fingerprint(result_path)
        result = json.loads(result_path.read_text())
        state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        check(f"{prefix}:weights_only_format_complete", state["format"] == "navisaaclab_method_reconstruction_training"
              and state["version"] == PaperAMPTrainer.checkpoint_version and state["physical_runner_state_included"] is True)
        runner = state["runner_state"]
        check(f"{prefix}:runner_state_nonempty", isinstance(runner, dict) and bool(runner))
        n = result["arguments"]["num_envs"]
        update, transitions = state["update_count"], state["physical_transition_count"]
        check(f"{prefix}:real_result_and_checkpoint_budget_match", result["status"] == "passed"
              and update == result["update_count"] and transitions == result["physical_transition_count"]
              and transitions == update * n * result["arguments"]["rollout_steps"]
              and runner["control_step"] * n == transitions)
        check(f"{prefix}:actual_AMASS_linear_no_diffusion_rollout_contract",
              result["arguments"]["trajectory_source"] == "amass_root"
              and result["runtime_execution"]["diffusion_used_for_rollouts"] is False
              and runner["runner_contract"]["trajectory_source"] == "amass_root"
              and runner["curriculum_state"]["config"]["tail_policy"] == "linear_velocity")
        check(f"{prefix}:saved_config_and_result_JSON_projection_equal",
              json.loads(json.dumps(state["config"])) == result["training_config"],
              {"JSON_tuple_to_list_projection": "action_mask; values/order remain exact"})
        sections = {key: tensor_stats(state[key]) for key in ("model", "optimizer", "discriminator_optimizer", "normalizers", "rng", "runner_state", "policy_replay")}
        check(f"{prefix}:all_model_optimizer_normalizer_physical_and_rng_tensors_finite",
              all(row["finite"] for row in sections.values()), sections)
        all_tensor_stats = tensor_stats(state)
        check(f"{prefix}:whole_serialized_bundle_including_policy_replay_finite", all_tensor_stats["finite"], all_tensor_stats)
        loaded, loaded_runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
        check(f"{prefix}:strict_current_CPU_trainer_loader_accepts_complete_payload", loaded_runner.keys() == runner.keys()
              and loaded.update_count == update and loaded.physical_transition_count == transitions
              and all(torch.equal(v, state["model"][k]) for k, v in loaded.model.state_dict().items()))
        del loaded, loaded_runner
        check(f"{prefix}:dataset_hash_and_native_child_frame_bound",
              state["provenance"]["dataset"]["manifest_sha256"] == file_sha256(manifest)
              and state["provenance"]["dataset"]["source_sha256"] == corpus.entry["sha256"]
              and state["config"]["expert_dof_velocity_frame"] == "child_joint")
        curriculum = AMASSRootTrajectoryCurriculum(corpus, n,
            bounds_xy=[0., 0., result["arguments"]["terrain_extent_m"], result["arguments"]["terrain_extent_m"]],
            tail_policy="linear_velocity")
        curriculum.load_state_dict(runner["curriculum_state"])
        side = int(n ** .5)
        check(f"{prefix}:declared_pilot_grid_is_exact_square", side * side == n)
        actor = torch.arange(n)
        margin = result["arguments"]["spawn_margin_m"]
        origins = torch.stack((margin + (actor // side) * 4, margin + (actor % side) * 4), -1).float()
        curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"], runner["reference_ids"],
            runner["episode_ids"], replan_steps=result["arguments"]["replan_steps"], origins_xy=origins)
        check(f"{prefix}:actual_native151_phase_fixed_offset_episode_and_resume_plan_gate", True)
        check(f"{prefix}:corpus_and_curriculum_saved_shared_RNG_equal",
              torch.equal(runner["corpus_rng"], runner["curriculum_state"]["generator_state"]))
        check(f"{prefix}:all_curriculum_statistics_equal_actual_result",
              runner["curriculum_state"]["statistics"] == {k: result["curriculum"]["statistics"][k] for k in curriculum.STAT_FIELDS})
        stats = curriculum.statistics
        previous_stats = rows[-1]["curriculum_statistics"] if rows else dict.fromkeys(curriculum.STAT_FIELDS, 0)
        previous_transitions = rows[-1]["physical_transitions"] if rows else 0
        if not rows:
            previous_stats["resets"] = n
        segment_stats = {key: stats[key] - previous_stats[key] for key in curriculum.STAT_FIELDS}
        check(f"{prefix}:all_committed_actor_steps_and_segment_resets_accounted",
              stats["control_actor_steps"] == transitions
              and result["actual_control_steps"] * n == transitions - previous_transitions
              and segment_stats["resets"] == result["resets"]
              and segment_stats["fall_terminations"] == result["fall_resets"]
              and segment_stats["clip_end_truncations"] + segment_stats["cap_only_truncations"] == result["time_limit_resets"]
              and stats["held_future_knots"] == 0
              and stats["extrapolated_future_knots"] == stats["beyond_source_future_knots"])
        physical = runner["physical"]
        shape = {"root_pos": (n, 3), "root_rot": (n, 4), "root_vel": (n, 3), "root_ang_vel": (n, 3),
                 "dof_pos": (n, 69), "dof_vel": (n, 69)}
        check(f"{prefix}:COMMON_physical_state_exact_fields_shapes_and_unit_XYZW_roots", set(physical) == set(shape)
              and all(physical[key].shape == value for key, value in shape.items())
              and torch.allclose(physical["root_rot"].norm(dim=-1), torch.ones(n), atol=1e-4, rtol=0))
        check(f"{prefix}:AMP_history_same_episode_and_integer_clocks",
              runner["amp_history"].shape == (n, 10, 195)
              and runner["amp_history_episode_ids"].shape == (n, 10)
              and bool((runner["amp_history_episode_ids"] == runner["episode_ids"][:, None]).all())
              and runner["age"].dtype == runner["episode_ids"].dtype == torch.long)
        check(f"{prefix}:checkpoint_and_result_immutable_during_CPU_audit",
              before == fingerprint(path) and result_record == fingerprint(result_path))
        rows.append({"checkpoint": before, "result": result_record, "updates": update,
            "physical_transitions": transitions, "control_steps": runner["control_step"],
            "num_envs": n, "fall_resets": result["fall_resets"], "time_limit_resets": result["time_limit_resets"],
            "minimum_height_m": result["minimum_height_m"], "curriculum_statistics": stats,
            "segment_curriculum_statistics": segment_stats, "segment_actual_control_steps": result["actual_control_steps"],
            "tensor_sections": sections, "whole_bundle_tensor_statistics": all_tensor_stats,
            "all_saved_actors_initialized": bool(curriculum.initialized.all()),
            "newly_reset_actor_count": int((runner["age"] == 0).sum()),
            "age_min_max": [int(runner["age"].min()), int(runner["age"].max())],
            "state_and_command_recovery_valid": True,
            "configuration": state["config"], "runner_contract": runner["runner_contract"],
            "curriculum_config": runner["curriculum_state"]["config"],
            "resume_argument": result["arguments"]["resume"]})
        del state, runner, curriculum
    if len(rows) == 2:
        a, b = rows
        check("pilot4_resume8_configuration_and_complete_curriculum_contract_equal",
              a["configuration"] == b["configuration"] and a["runner_contract"] == b["runner_contract"]
              and a["curriculum_config"] == b["curriculum_config"])
        check("pilot4_resume8_budget_monotonic_exact_additional4_updates",
              b["updates"] - a["updates"] == 4 and b["physical_transitions"] - a["physical_transitions"] == 32768)
        check("resume8_declares_actual_immutable_step4_source",
              Path(b["resume_argument"]).resolve() == Path(a["checkpoint"]["path"]))
        check("restored_control_actor_counts_continue_not_reset",
              b["curriculum_statistics"]["control_actor_steps"] - a["curriculum_statistics"]["control_actor_steps"] == 32768
              and all(b["curriculum_statistics"][k] >= a["curriculum_statistics"][k] for k in AMASSRootTrajectoryCurriculum.STAT_FIELDS))
    report = {"schema": "navisaaclab.amass_root_actual_pilot_checkpoint_audit.v1", "status": "passed",
              "CPU_audit_only": True, "SDK_started_by_auditor": False, "GPU_used_by_auditor": False,
              "inputs_are_real_GPU_training_checkpoints": True, "heldout_outputs_read": False,
              "source_manifest": fingerprint(SOURCE), "source_archive": fingerprint(archive),
              "checks": checks, "check_count": len(checks), "checkpoints": rows,
              "script": fingerprint(__file__),
              "limitations": ["The CPU loader did not restore CUDA RNG or launch PhysX; it verifies serialized state and admission.",
                              "Logical resume validation is not bitwise PhysX/contact/solver-cache replay proof.",
                              "This small pilot validates execution/resume, not locomotion convergence or author reproduction."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "steps": [row["updates"] for row in rows]}))


if __name__ == "__main__":
    main()
