#!/usr/bin/env python3
"""CPU-only actual command generator probes; no policy, SDK or gait claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.procedural_anchoring import reference_clocks, reward_summary  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402

MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
OWNER = ROOT / "resources/runtime_sources/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json"
TRAINING = ROOT / "resources/runtime_sources/style_improvement_20261005/shared_social_full_epochs_v1/pilot2000_execution.json"
METHODS = ("actual_periodic_generator", "same_first_plan_periodic_restart", "same_first_plan_episode_clock")
PINNED = {
    "CrowdSim/humanoid/training/trajectories.py": "81701988535cf847b8b86979a836ba37089e93553aa4e3fe5ecfc9529d62d2c9",
    "CrowdSim/humanoid/training/mixed_trajectories.py": "c79732a675ac737d4ce0aff0d2b76cfa30cebed2d9c115e1a35da39299694d67",
    "resources/runtime_sources/style_improvement_20261004/shared_group_world_owned_lifetime_revision_v2/shared_group_world.py": "709e9fa0347f0860244e8edc78098ff83376b7e799093d37b4722e1362fdd96f"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe(initial, torch, config):
    from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
    from CrowdSim.humanoid.training.rewards import aligned_position_error_squared, paper_task_reward
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryCurriculum
    origins, local_times, episode_times = reference_clocks()
    generator = PacerTrajectoryCurriculum(config, seed=7)
    batch = len(initial.root_pos)
    first = generator.build_plan(initial)
    active = first.xy
    first_rng = generator.generator.get_state().clone()
    # Exactly the original canonical51 interpolation used by the procedural branch.
    lerp = MixedTrajectoryCurriculum._lerp51
    targets = torch.empty(3, 150, batch, 2)
    roots = torch.empty(150, batch, 2)
    generated = [first.xy.clone()]
    zero = torch.zeros(batch, 69)
    delta, rewards = torch.empty(3, 150, batch), torch.empty(3, 150, batch)
    for pre in range(150):
        state = SimpleNamespace(root_pos=initial.root_pos.clone(), root_vel=initial.root_vel,
                                root_rot=initial.root_rot)
        state.root_pos[:, :2] += initial.root_vel[:, :2] * (pre / 30)
        if pre and pre % 15 == 0:
            active = generator.build_plan(state).xy
            generated.append(active.clone())
        local = torch.full((batch, 1), float(local_times[pre]))
        episode = torch.full((batch, 1), float(episode_times[pre]))
        targets[0, pre] = lerp(local, active)[:, 0]
        targets[1, pre] = lerp(local, first.xy)[:, 0] + initial.root_vel[:, :2] * (int(origins[pre]) / 30)
        targets[2, pre] = lerp(episode, first.xy)[:, 0]
        roots[pre] = initial.root_pos[:, :2] + initial.root_vel[:, :2] * ((pre + 1) / 30)
        for method in range(3):
            delta[method, pre] = aligned_position_error_squared(roots[pre], targets[method, pre])
            rewards[method, pre] = paper_task_reward(delta[method, pre], zero, zero,
                                                   torch.zeros(batch))["trajectory"]
    require(torch.equal(targets[0, :15], targets[1, :15])
            and torch.equal(targets[0, :15], targets[2, :15]), "First interval must use exactly the same plan and clock")
    summaries = reward_summary(delta.numpy(), rewards.numpy())
    arrays = {"post_root_xy": roots.numpy(), "target_xy": targets.numpy(),
        "delta_squared_m2": delta.numpy(), "trajectory_reward": rewards.numpy(),
        "canonical_generated_plans": torch.stack(generated).numpy(),
        "first_plan_velocities_xy": first.velocities_xy.numpy(),
        "first_plan_speed_goals_mps": first.speed_goals_mps.numpy(),
        "first_plan_rng_state": first_rng.numpy(), "final_rng_state": generator.generator.get_state().numpy(),
        "replan_origin_controls": origins, "local_post_times_s": local_times,
        "episode_post_times_s": episode_times}
    return arrays, {"methods": dict(zip(METHODS, summaries, strict=True)),
        "actual_generator_plans": generator.plans_generated,
        "episode_reference_plans": batch,
        "fixed_path_uses_exact_actual_first_plan_and_recorded_private_rng_state": True,
        "actual_generator_consumes_nine_additional_plans_per_clip": True,
        "same_first_plan_restart_is_an_isolating_counterfactual_not_the_actual_generator": True,
        "physical_dynamics_simulated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output_directory.exists(),
            "Require CUDA hidden and a fresh output directory")
    direct = [identity(ROOT / path) for path in PINNED]
    require(all(record["sha256"] == PINNED[str(Path(record["path"]).relative_to(ROOT))] for record in direct),
            "Original command/World sources changed")
    direct.extend(identity(path) for path in (Path(__file__), ROOT / "CrowdSim/humanoid/analysis/procedural_anchoring.py",
        ROOT / "CrowdSim/humanoid/training/rewards.py", ROOT / "scripts/diagnose_shared_social_training.py"))
    guards = json.loads(OWNER.read_text())["bindings"]["guarded_sources"]
    require(len(guards) == 23 and all(identity(row["path"]) == row for row in guards), "Active S3 source guard changed")
    manifest_record = identity(MANIFEST)
    require(manifest_record["sha256"] == "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca", "Fixed2848 changed")
    entry = json.loads(MANIFEST.read_text())["splits"]["train"]
    require(entry["clip_count"] == len(entry["clips"]) == 2848
        and [row["motion_id"] for row in entry["clips"]] == list(range(2848)), "Require every registered train clip")
    bank_record = identity(Path(entry["path"]) if Path(entry["path"]).is_absolute() else MANIFEST.parent / entry["path"])
    require(bank_record["sha256"] == entry["sha256"] and bank_record["bytes"] == entry["bytes"], "Train bank changed")
    import torch
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    torch.set_num_threads(4)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    # Official 2.11 CPU remapping, restricted unpickling and memory mapping:
    # https://docs.pytorch.org/docs/2.11/generated/torch.load.html
    bank = torch.load(bank_record["path"], weights_only=True, map_location="cpu", mmap=True)
    for row in entry["clips"]:
        mid = row["motion_id"]
        require(int(bank["motion_num_frames"][mid]) == row["frames"] and row["frames"] >= 41
            and bank["motion_files"][mid] == row["source_path"]
            and abs(float(bank["motion_dt"][mid]) - 1 / 30) < 1e-7, "Actual source/count/rate changed")
    index = bank["length_starts"] + 9
    require(len(index) == 2848, "Bank has extra clips")
    source = SimpleNamespace(root_pos=bank["gts"][index, 0].clone(),
        root_rot=bank["grs"][index, 0].clone(), root_vel=bank["gvs"][index, 0].clone())
    _, _, admission = admit_training(TRAINING)
    training = json.loads(TRAINING.read_text())
    observed = json.loads(Path(training["external_exit_metadata"]["path"]).read_text())
    require(identity(training["external_exit_metadata"]["path"]) == training["external_exit_metadata"]
        and all(type(training[key]) is int and training[key] == 0 for key in ("terminal_exit_code", "CPU_audit_exit_code"))
        and observed["training"]["chunk_id"] == training["terminal_tool_chunk"]
        and observed["CPU_audit"]["chunk_id"] == training["CPU_audit_terminal_tool_chunk"]
        and all(type(observed[key]["exit_code"]) is int and observed[key]["exit_code"] == 0 for key in ("training", "CPU_audit")),
        "Require observed actual S2 training and CPU audit exits")
    actual = json.loads(Path(admission["result"]["path"]).read_text())
    config = PacerTrajectoryConfig()
    require(actual["completed_updates"] == 2000 and actual["actual_training_transitions"] == 16384000
        and actual["final_world_description"]["curriculum"]["config"]["procedural_config"] == asdict(config)
        and actual["expert_contract"]["manifest"] == manifest_record
        and actual["setup_sources"]["World"] == direct[2], "Actual completed S2 command configuration differs")
    inputs = [manifest_record, bank_record, admission["receipt"], training["external_exit_metadata"],
        admission["result"], admission["audit"], admission["checkpoint"]]
    direct.extend(identity(ROOT / path) for path in ("scripts/evaluate_shared_social_tracking.py",
        "CrowdSim/humanoid/runtime/social_popart_training.py", "CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py"))
    args.output_directory.mkdir(parents=True)
    cohorts, artifacts = {}, []
    for name in ("stationary", "constant_AMASS_initial_XY_velocity"):
        velocity = source.root_vel.clone()
        velocity[:, 2] = 0
        if name == "stationary":
            velocity.zero_()
        initial = SimpleNamespace(root_pos=source.root_pos, root_rot=source.root_rot, root_vel=velocity)
        arrays, result = probe(initial, torch, config)
        arrays.update(motion_ids=np.arange(2848, dtype=np.int64), source_frames=np.full(2848, 9, dtype=np.int64),
            initial_root_pos=initial.root_pos.numpy(), initial_root_rot_xyzw=initial.root_rot.numpy(),
            declared_root_vel=velocity.numpy())
        path = args.output_directory / (name + ".npz")
        np.savez_compressed(path, **arrays)
        artifacts.append(identity(path))
        cohorts[name] = result
    require(all(identity(row["path"]) == row for row in inputs + direct + guards), "Input/source changed during execution")
    after = np.random.get_state()
    require(random.getstate() == rng[0] and after[0] == rng[1][0] and np.array_equal(after[1], rng[1][1])
        and after[2:] == rng[1][2:] and torch.equal(torch.random.get_rng_state(), rng[2])
        and not torch.cuda.is_initialized(), "Global RNG changed or CUDA initialized")
    require(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "SDK imported")
    report = {"schema": "shared_social.procedural_anchoring_CPU.v1", "status": "actual_command_and_reward_probe_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "methods": METHODS,
        "population": "all fixed2848 train clips once, earliest legal frame9; not random reset exposure or held-out evaluation",
        "post_controls_per_clip": 150, "total_reward_rows": 2 * 3 * 150 * 2848,
        "procedural_config": asdict(config), "actual_S2_training_admission": admission,
        "cohorts": cohorts, "inputs": inputs, "sources": direct,
        "unchanged_active_S3_sources": len(guards), "artifacts": artifacts,
        "policy_executed": False, "physical_dynamics_simulated": False, "paper_quality_improvement_proven": False,
        "scope": "original local smooth generator/interpolation/reward; episode clock mechanism comparison, not exact PACER distribution or Nav author training"}
    (args.output_directory / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output_directory), "cohorts": cohorts, "rows": report["total_reward_rows"]}, indent=2))


if __name__ == "__main__":
    main()
