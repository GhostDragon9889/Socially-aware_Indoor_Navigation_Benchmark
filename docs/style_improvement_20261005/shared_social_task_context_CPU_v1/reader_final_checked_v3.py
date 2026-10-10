#!/usr/bin/env python3
"""Observe saved pre-action targets and root speeds without changing training.

Future positions are displacement-rate proxies, not velocity commands. All
terminal actor-controls, including failures/resets, stay in the denominator.
Only the completed training rollout is mmap-loaded, with no model or SDK.
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
from scripts.diagnose_shared_social_expert_coverage import admit_diagnostic, bound, identity, require  # noqa: E402

PINNED = {
    "scripts/diagnose_shared_social_expert_coverage.py": "53d6e82b41cb0f75c7683b83e42253e2fdd6d8da5f80e3d22b381179d2bdd514",
    "docs/style_improvement_20261004/social_world_lifetime_dependency_migration_v1/social_rollout_collector.py": "4497ed99f0424a33148a31c137d2c7f705acd9a72f32b8eae6e66c833b321898",
    "docs/style_improvement_20261004/shared_group_world_owned_lifetime_revision_v2/shared_group_world.py": "e76182d2be5a3fa4dad19a04d71275d91929ce3ea2d63a1c23f081aa1a59880b",
    "docs/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py": "6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1",
    "CrowdSim/humanoid/amp/observations.py": "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba",
    "CrowdSim/humanoid/training/mixed_trajectories.py": "c79732a675ac737d4ce0aff0d2b76cfa30cebed2d9c115e1a35da39299694d67",
    "CrowdSim/humanoid/training/expert_sampling.py": "b16984a920f21ca92c3fe01976d9c0453a75bf54d3f46df369984ff27453f9cd",
}


def array(mapping, key, shape, dtype):
    value = mapping[key]
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype,
            "Wrong saved shape/dtype: " + key)
    require(np.isfinite(value).all(), "Nonfinite saved field: " + key)
    return value


def same_json_document(left, right):
    """Original tensor packet tuples become lists in the result JSON only."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(len(values) > 0 and np.isfinite(values).all() and (values >= 0).all(), "Require nonempty finite speeds")
    return {"rows": len(values), "mean": float(values.mean()),
            "quantiles_0_25_50_75_90_95_100": np.quantile(values, [0, .25, .5, .75, .9, .95, 1]).tolist(),
            "less_than_0_12_count": int((values < .12).sum()), "at_least_1_8_count": int((values >= 1.8).sum())}


def describe_targets(prefix):
    require(prefix.ndim == 3 and prefix.shape[1:] == (10, 2) and prefix.dtype == np.float32
            and np.isfinite(prefix).all(), "Require raw ten local metre targets")
    points = prefix.astype(np.float64)
    # The root is the first segment's origin. Its rate includes accumulated lag.
    first = np.linalg.norm(points[:, 0], axis=-1) / .5
    future = np.linalg.norm(np.diff(points, axis=1), axis=-1) / .5
    return {"root_to_first_future_target_rate": first, "future_segment_rate": future,
            "median_future_segment_rate": np.median(future, axis=1)}


def verify_phase(rollout, diagnostics, metadata):
    """Check clocks, same-episode continuity and the two raw root-velocity slots.

    The diagnostic quaternion/task heading belongs to POST action. Never use
    it to rotate a PRE target. First rows and reset boundaries are unpaired.
    """
    controls, actors = metadata["controls"], metadata["actors"]
    require(type(controls) is type(actors) is int and controls >= 2 and actors > 0, "Invalid collection dimensions")
    shape = (controls, actors)
    episode = array(rollout, "episode_ids", shape, np.int64)
    history_episode = array(rollout, "history_episode_ids", shape + (10,), np.int64)
    post_episode = array(diagnostics, "post_episode_ids", shape, np.int64)
    post_history_episode = array(diagnostics, "post_history_episode_ids", shape + (10,), np.int64)
    pre_history = array(rollout, "amp_history", shape + (10, 195), np.float32)
    post_history = array(diagnostics, "post_amp_history", shape + (10, 195), np.float32)
    self_state = array(rollout, "self_obs", shape + (357,), np.float32)
    array(rollout, "task_obs", shape + (12308,), np.float32)
    require(np.array_equal(history_episode, np.broadcast_to(episode[..., None], history_episode.shape))
            and np.array_equal(post_history_episode, np.broadcast_to(post_episode[..., None], post_history_episode.shape))
            and np.array_equal(episode, post_episode), "History or terminal archive crossed episodes")
    require(np.array_equal(self_state[..., 213:216], pre_history[..., 0, 6:9]), "PRE self/AMP root velocity disagrees")
    body_velocity = array(diagnostics, "common_rigid_body_vel", shape + (24, 3), np.float32)
    native_velocity = array(diagnostics, "native_root_link_root_vel", shape + (3,), np.float32)
    require(np.array_equal(body_velocity[..., 0, :], native_velocity), "POST COMMON/native root velocity differs")
    # A heading rotation preserves XY speed; do not apply POST heading to PRE.
    post_speed = np.linalg.norm(post_history[..., 0, 6:8].astype(np.float64), axis=-1)
    native_speed = np.linalg.norm(native_velocity[..., :2].astype(np.float64), axis=-1)
    speed_error = float(np.max(np.abs(post_speed - native_speed)))
    require(speed_error <= 1e-5, "POST AMP/native root XY speed differs")
    for kind, scale in (("control", 1), ("physics", 4)):
        start, end = metadata[kind + "_step_start"], metadata[kind + "_step_end"]
        require(type(start) is type(end) is int and end == start + scale * controls, "Clock extent differs")
        before = array(diagnostics, kind + "_step_before", shape, np.int64)
        after = array(diagnostics, kind + "_step_after", shape, np.int64)
        expected = np.broadcast_to((start + scale * np.arange(controls))[:, None], shape)
        require(np.array_equal(before, expected) and np.array_equal(after, expected + scale), "Saved native clock gap")
    reset = array(diagnostics, "reset_mask", shape, np.bool_)
    for key in ("terminated", "truncated"):
        array(rollout, key, shape, np.bool_)
    for key in ("physical_fall", "nonfoot_normal_contact", "task_heading_defined", "singular_heading_fallback"):
        array(diagnostics, key, shape, np.bool_)
    require(np.array_equal(reset, rollout["terminated"] | rollout["truncated"]), "Reset/termination ownership differs")
    same = episode[1:] == post_episode[:-1]
    require(np.array_equal(~same, reset[:-1]) and np.array_equal(episode[1:] - post_episode[:-1], reset[:-1].astype(np.int64)),
            "Adjacent episode/reset boundary differs")
    require(np.array_equal(pre_history[1:][same], post_history[:-1][same]), "PRE history is not previous POST history")
    pre_age = array(diagnostics, "pre_age", shape, np.int64)
    post_age = array(diagnostics, "post_age", shape, np.int64)
    require((pre_age >= 0).all() and np.array_equal(post_age, pre_age + 1)
            and np.array_equal(pre_age[1:], np.where(reset[:-1], 0, post_age[:-1])), "Saved age/reset phase differs")
    return {"all_actor_controls": controls * actors, "same_episode_PRE_previous_POST_exact_rows": int(same.sum()),
            "first_control_unpaired_rows": actors, "adjacent_reset_unpaired_rows": int((~same).sum()),
            "PRE_self_AMP_root_velocity_exact_rows": controls * actors, "controls": controls, "actors": actors,
            "POST_COMMON_native_root_velocity_exact_rows": controls * actors,
            "POST_AMP_native_XY_speed_max_abs_error_m_s": speed_error,
            "POST_AMP_native_XY_speed_tolerance_m_s": 1e-5,
            "reset_actor_controls_retained": int(reset.sum()), "any_metrics_rows_removed": False}


def branch_measurement(prefix, history, rollout, diagnostics, rows):
    targets = describe_targets(prefix[:, rows].reshape(-1, 10, 2))
    speed = np.linalg.norm(history[:, rows, 0, 6:8].astype(np.float64), axis=-1).reshape(-1)
    fast_target = targets["median_future_segment_rate"] >= 1.8
    low_current = speed < .12
    return {"actor_ids": rows.tolist(), "actor_controls": len(speed), "current_root_XY_speed_m_s": summary(speed),
            "target_rate_proxies_m_s": {name: summary(value) for name, value in targets.items()},
            "fast_median_future_segments_and_low_current_speed": {"numerator": int((fast_target & low_current).sum()),
                "fast_target_denominator": int(fast_target.sum()), "all_rows_denominator": len(speed)},
            "retained_flags": {key: int(value[:, rows].sum()) for key, value in {
                "terminated": rollout["terminated"], "truncated": rollout["truncated"],
                **{key: diagnostics[key] for key in ("reset_mask", "physical_fall", "nonfoot_normal_contact",
                                                    "task_heading_defined", "singular_heading_fallback")}}.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--diagnostic-exit-receipt", required=True, type=Path)
    parser.add_argument("--coverage-exit-receipt", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output_directory.exists(), "Require CPU-hidden CUDA and new output")
    sources = [identity(Path(__file__))]
    for path, expected in PINNED.items():
        record = identity(ROOT / path)
        require(record["sha256"] == expected, "Pinned equation/source changed: " + path)
        sources.append(record)
    diagnostic, rollout_record, inputs, exposure = admit_diagnostic(args.diagnostic_exit_receipt)
    require(identity(rollout_record["path"]) == rollout_record, "Sealed rollout changed")
    coverage_receipt_record = identity(args.coverage_exit_receipt)
    coverage_receipt = bound(coverage_receipt_record)
    require(coverage_receipt["schema"] == "shared_social.actual_expert_prior_coverage_CPU_execution.v1", "Unknown coverage receipt")
    executions = [item for item in coverage_receipt["checks"] if item["name"] == "actual_S1_final_v2"]
    require(len(executions) == 1 and type(executions[0]["actual_exit_code"]) is int
            and executions[0]["actual_exit_code"] == 0 and bool(executions[0]["actual_tool_chunk"]), "Require actual coverage exit0")
    coverage_record, centers_record = (coverage_receipt[key] for key in ("actual_prior_and_terminal_evidence", "actual_terminal_centers"))
    coverage = bound(coverage_record)
    require(coverage["actual_completed_inputs"] == inputs and coverage["actual_terminal_centers"] == centers_record
            and coverage["training_exposure_counters"] == exposure, "Coverage and task packet ownership differs")
    require(identity(centers_record["path"]) == centers_record, "Actual center archive changed")
    result = bound(diagnostic["training_admission"]["result"])
    require(result["entry_sources"]["collector"] == sources[2] and result["setup_sources"]["World"] == sources[3],
            "Actual collection used different phase sources")
    inputs += [coverage_receipt_record, coverage_record, centers_record, executions[0]["log"]]
    require(identity(executions[0]["log"]["path"]) == executions[0]["log"], "Actual coverage log changed")
    import torch
    require(not torch.cuda.is_initialized(), "CUDA must not initialize")
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    # Official torch.load2.11: map_location=cpu, weights_only=True, mmap=True.
    # https://docs.pytorch.org/docs/2.11/generated/torch.load.html
    packet = torch.load(rollout_record["path"], weights_only=True, map_location="cpu", mmap=True)
    collection, expert = packet["collection"], packet["expert"]
    rollout, diagnostics = ({key: value.numpy() for key, value in collection[name].items()}
                            for name in ("rollout", "diagnostics"))
    phase = verify_phase(rollout, diagnostics, collection["metadata"])
    require(phase["controls"] == 32 and phase["actors"] == 256
            and same_json_document(collection["metadata"]["configuration"], result["actual_configuration"])
            and exposure["completed_updates"] == collection["metadata"]["training_update_count"] + 1
            and exposure["physical_transitions"] == collection["metadata"]["training_physical_transition_count"] + phase["all_actor_controls"],
            "Require actual completed terminal32x256 update")
    curriculum = result["final_world_description"]["curriculum"]
    amass, procedural = (np.asarray(curriculum["config"][key], np.int64) for key in ("amass_actor_ids", "procedural_actor_ids"))
    require(curriculum["source"] == "engineering_fixed_actor_amass_procedural_mixture"
            and np.array_equal(amass, np.arange(128)) and np.array_equal(procedural, np.arange(128, 256))
            and result["final_world_description"]["group_actor_rows"] == [[2*g, 2*g+1, 128+2*g, 129+2*g] for g in range(64)],
            "Actual mixed branch/group ownership differs")
    prefix = rollout["task_obs"][..., :20].reshape(32, 256, 10, 2)
    branches = {name: branch_measurement(prefix, rollout["amp_history"], rollout, diagnostics, rows)
                for name, rows in (("all", np.arange(256)), ("AMASS_targets", amass), ("procedural_targets", procedural))}
    expert_array = {key: value.numpy() for key, value in expert.items()}
    history = array(expert_array, "history", (4096, 10, 195), np.float32)
    with np.load(centers_record["path"], allow_pickle=False) as centers:
        require(np.array_equal(expert_array["motion_ids"], centers["motion_ids"])
                and np.array_equal(expert_array["frame_indices"], centers["frame_indices"]), "Actual expert IDs differ")
        labels = centers["window_strata"].copy()
    # Source classifier uses the mean of sorted middle values for even ten frames.
    expert_speed = np.linalg.norm(history[..., 6:8].astype(np.float64), axis=-1)
    median_speed = np.median(expert_speed, axis=-1)
    require(np.array_equal(median_speed < .12, labels == 0) and np.array_equal(median_speed >= 1.8, labels == 1),
            "Expert velocity descriptor differs from original low/fast classification")
    after = np.random.get_state()
    require(random.getstate() == rng[0] and after[0] == rng[1][0] and np.array_equal(after[1], rng[1][1])
            and after[2:] == rng[1][2:] and torch.equal(torch.random.get_rng_state(), rng[2])
            and not torch.cuda.is_initialized(), "CPU observation changed RNG/CUDA")
    require(not any(name.split('.')[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "SDK imported")
    require(all(identity(item["path"]) == item for item in sources + inputs), "Source/input changed during analysis")
    report = {"schema": "shared_social.actual_PRE_task_context_CPU.v1", "status": "completed_CPU_observation",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "actual_completed_inputs": inputs,
              "training_exposure_counters": exposure, "phase_checks": phase, "measurement": branches,
              "actual_expert": {"rows": 4096, "current_root_XY_speed_m_s": summary(expert_speed[:, 0]),
                                "median_ten_frame_root_XY_speed_m_s": summary(median_speed),
                                "low_fast_membership_matches_saved_original_classifier": True},
              "definitions": {"targets": "raw metre positions at +.5,+1,...,+5s relative to PRE root in PRE heading frame",
                  "root_to_first_future_target_rate": "norm(target+.5)/.5; includes accumulated position lag",
                  "future_segment_rate": "norm(target[k+1]-target[k])/.5, nine future intervals",
                  "current_root_speed": "raw AMP newest index0, velocity indices6:8; self357 indices213:215 equal",
                  "phase": "POST diagnostics are not PRE task heading; no POST rotation used for PRE task",
                  "retained_flags": "termination/reset and diagnostic flags are transition outcomes/POST states, not PRE heading validity",
                  "expert_history": "actual4096 newest-first reference windows; not paired to individual policy task rows"},
              "CPU_rng_unchanged": True, "CUDA_initialized": False, "SDK_imported": False,
              "model_checkpoint_loaded": False, "expert_prior_changed": False, "causal_training_failure_proven": False,
              "paper_effect_achieved": False, "limits": [
                  "One completed terminal rollout, not whole-training task distribution or independent quality evaluation.",
                  "Future displacement rates are not instantaneous desired velocities or expert/task pairing.",
                  "Policy reset history can contain reference initialization; current newest velocity is separately measured.",
                  "Unknown first-control/reset-adjacent physical pairing retained; no failure rows removed.",
                  "No style reweighting, training intervention, learned context relevance or default model promotion."]}
    args.output_directory.mkdir(parents=True, exist_ok=False)
    row_archive = args.output_directory / "all_PRE_actor_controls.npz"
    target_rates = describe_targets(prefix.reshape(-1, 10, 2))
    with row_archive.open("xb") as stream:
        np.savez(stream, actor_id=np.tile(np.arange(256, dtype=np.int64), 32),
                 control_step=diagnostics["control_step_before"].reshape(-1),
                 episode_id=rollout["episode_ids"].reshape(-1),
                 root_speed=np.linalg.norm(rollout["amp_history"][..., 0, 6:8].astype(np.float64), axis=-1).reshape(-1),
                 **target_rates, **{key: diagnostics[key].reshape(-1) for key in
                                   ("reset_mask", "physical_fall", "task_heading_defined", "singular_heading_fallback")})
    report["all_PRE_actor_controls"] = identity(row_archive)
    with (args.output_directory / "diagnostic.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"phase": phase, "measurement": branches, "expert": report["actual_expert"]}))


if __name__ == "__main__":
    main()
