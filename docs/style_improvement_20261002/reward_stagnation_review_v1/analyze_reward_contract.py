"""CPU-only reward arithmetic and fixed-capture diagnostics; no training edits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "output/isolated_training/mixed_state_action_terrain_protected_v1/code"
OUT = Path(__file__).resolve().parent
DT = 1 / 30


def file_record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def velocity_cost(actual, command, coefficient=1., speed_floor=.1):
    """Unmerged proposal; world/local-heading norms are rotation invariant."""
    squared = np.sum((actual - command) ** 2, axis=-1)
    return -coefficient * np.minimum(squared / (np.sum(command ** 2, axis=-1)
                                               + speed_floor ** 2), 1.)


def actual_training_metrics(directory):
    path = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2000
    identity_error = []
    for index, row in enumerate(rows, 1):
        assert row["update_count"] == index
        reconstructed = 2 * row["amp_rewards_mean"] + .5 * sum(
            row[key] for key in ("trajectory_rewards_mean", "energy_rewards_mean",
                                 "direction_rewards_mean"))
        identity_error.append(abs(row["reward_mean"] - reconstructed))
    assert max(identity_error) < 1e-6
    windows = {}
    fields = ("reward_mean", "amp_rewards_mean", "trajectory_rewards_mean",
              "energy_rewards_mean", "direction_rewards_mean", "forward_velocity_mean",
              "backwards_fraction", "signed_power_w_mean", "disc_expert_probability",
              "disc_policy_probability")
    for lo, hi in ((1, 100), (901, 1000), (1901, 2000)):
        selected = rows[lo - 1:hi]
        values = {field: sum(row[field] for row in selected) / len(selected)
                  for field in fields}
        values["weighted_style"] = 2 * values["amp_rewards_mean"]
        values["weighted_task"] = .5 * sum(values[field] for field in
            ("trajectory_rewards_mean", "energy_rewards_mean", "direction_rewards_mean"))
        windows[f"{lo}:{hi}"] = values
    return {"file": file_record(path), "updates": len(rows),
            "reward_identity_max_abs_error": max(identity_error), "windows": windows,
            "scope": "Full-batch means; no command-speed/idle conditioning saved."}


def fixed_old_navigation():
    directory = ROOT / "output/diffusion_amp/shared_mixed_state_action_isolated_ground_support_200s_v1"
    with np.load(directory / "physical_states.npz", allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    with np.load(directory / "reference_plans.npz", allow_pickle=False) as archive:
        plans = {key: archive[key] for key in archive.files}
    root = fields["root_pos"][..., :2]
    assert root.shape == (6000, 4, 2)
    finite = all(np.isfinite(fields[key]).all() for key in
                 ("root_pos", "target_xy", "signed_power_w", "dof_vel"))
    assert finite
    same_episode = (fields["episode_ids"][1:] == fields["episode_ids"][:-1])
    same_episode &= ~fields["reset_mask"][1:]
    actual = (root[1:].astype(np.float64) - root[:-1].astype(np.float64)) / DT
    speed = np.linalg.norm(actual, axis=-1)
    command = np.zeros_like(actual)
    reconstructed_target = np.zeros_like(root[1:])
    for actor in range(4):
        indices = np.flatnonzero(plans["actor_ids"] == actor)
        steps = plans["control_step"][indices]
        assert len(steps) and steps[0] == 0 and np.all(np.diff(steps) > 0)
        controls = np.arange(1, 6000)
        selection = np.searchsorted(steps, controls, side="right") - 1
        ages = controls - steps[selection]
        assert np.all((ages >= 0) & (ages <= 15))
        rows = plans["reference_xy"][indices[selection]]
        future = rows[np.arange(len(rows)), ages + 15]
        command[:, actor] = (future.astype(np.float64) - root[:-1, actor]) / .5
        reconstructed_target[:, actor] = rows[np.arange(len(rows)), ages + 1]
    phase_error = np.abs(reconstructed_target - fields["target_xy"][1:])
    assert float(phase_error.max()) < 1e-6
    trajectory = np.exp(-2 * np.sum((root[1:] - fields["target_xy"][1:]) ** 2, -1))
    energy = -.0005 * fields["signed_power_w"][1:]
    joint_rms = np.sqrt(np.mean(fields["dof_vel"][1:] ** 2, -1))
    command_speed = np.linalg.norm(command, axis=-1)
    cost = velocity_cost(actual, command)
    groups = {}
    for name, mask in {
            "all_same_episode": same_episode,
            "idle": same_episode & (speed < .1),
            "idle_weak_request": same_episode & (speed < .1) & (command_speed < .1),
            "idle_moving_request": same_episode & (speed < .1) & (command_speed >= .1),
            "moving": same_episode & (speed >= .1)}.items():
        groups[name] = {"actor_intervals": int(mask.sum()),
                       "duration_actor_s": float(mask.sum() * DT),
                       "actual_control_average_speed_m_s": summary(speed[mask]),
                       "reconstructed_half_second_request_speed_m_s": summary(command_speed[mask]),
                       "exact_saved_position_trajectory_reward": summary(trajectory[mask]),
                       "exact_saved_signed_power_energy_reward": summary(energy[mask]),
                       "native_dof_velocity_rms_rad_s": summary(joint_rms[mask]),
                       "unmerged_extra_velocity_cost_outer_weight_0_5": summary(.5 * cost[mask])}
    return {"files": [file_record(directory / name) for name in
                       ("result.json", "physical_states.npz", "reference_plans.npz")],
            "groups": groups, "post_target_phase_max_abs_error_m": float(phase_error.max()),
            "initial_or_cross_reset_intervals_excluded": int(same_episode.size - same_episode.sum()),
            "pre_action_request_is_reconstructed_not_saved_observation": True,
            "request_formula": "[p_reference(control_age+15)-root_pre]/0.5; world norm equals heading-local raw_task[:2]/0.5",
            "capture_contains_actual_AMP_reward_or_policy_action": False,
            "actual_AMP_reward_reconstructed": False,
            "new_cost_scope": "Fixed-record arithmetic only; no retraining, policy rollout or quality prediction."}


def mathematical_cases():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("isolated_literal_reward", CODE / "CrowdSim/humanoid/training/rewards.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for error in (0., .083, .25, .5, 1.):
        for probability in (.1, .15, .5):
            components = module.paper_task_reward(torch.tensor([error ** 2], dtype=torch.float64),
                torch.zeros(1, 69, dtype=torch.float64), torch.zeros(1, 69, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64), direction_mode="backwards_penalty")
            amp = torch.tensor([-2 * math.log(1 - probability)], dtype=torch.float64)
            actual = module.paper_total_reward(amp, components, amp_weight=2., task_weight=.5).item()
            expected = -4 * math.log(1 - probability) + .5 * math.exp(-2 * error ** 2)
            assert abs(actual - expected) < 1e-12
            cases.append({"position_error_m": error, "synthetic_D_probability": probability,
                          "stationary_all_joint_velocities_zero_power_w": 0.,
                          "weighted_style": -4 * math.log(1 - probability),
                          "weighted_position": .5 * math.exp(-2 * error ** 2),
                          "total": actual})
    proposed = []
    for command_speed in (0., .05, .106, .25, .5, 1., 2.):
        command = np.array([[command_speed, 0.]])
        stationary = float(velocity_cost(np.zeros((1, 2)), command)[0])
        perfect = float(velocity_cost(command, command)[0])
        assert perfect == 0. and -1 <= stationary <= 0
        assert (stationary == 0) == (command_speed == 0)
        proposed.append({"command_speed_m_s": command_speed,
                         "stationary_added_weighted_cost": .5 * stationary,
                         "perfect_added_cost": perfect})
    angle = .731
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    command, actual = np.array([[.2, -.3], [1., .5]]), np.array([[.1, .1], [-.5, .3]])
    assert np.allclose(velocity_cost(actual, command), velocity_cost(actual @ rotation, command @ rotation), atol=1e-15)
    assert not torch.cuda.is_initialized()
    return {"synthetic_stationarity_cases": cases, "proposed_synthetic_cases": proposed,
            "rotation_invariance_checked": True, "GPU_used": False,
            "synthetic_cases_are_not_actual_discriminator_scores": True}


def main():
    tracking = ROOT / "docs/style_improvement_20261002/terrain_protected_style_v1/full22_step2000_comparison_v1/actual_summary.json"
    tables = json.loads(tracking.read_text())["tables"]
    result = {"schema": "reward_stagnation.independent_CPU_review.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False,
              "sealed34_outputs_read": False, "business_or_training_source_modified": False,
              "source_records": [file_record(CODE / name) for name in
                  ("CrowdSim/humanoid/training/rewards.py", "CrowdSim/humanoid/training/trainer.py",
                   "CrowdSim/humanoid/application/settings.py", "CrowdSim/humanoid/amp/observations.py",
                   "scripts/train_paper_amp.py", "configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml")],
              "mathematics": mathematical_cases(),
              "training": {name: actual_training_metrics(ROOT / "output/diffusion_amp" / name) for name in
                  ("amp_style_mixed_state_action_isolated_v1", "amp_style_mixed_state_action_terrain_protected_isolated_v1")},
              "completed_old_state_action_navigation": fixed_old_navigation(),
              "completed_full22_source": file_record(tracking),
              "completed_full22": {name: {field: table[field] for field in
                  ("aggregate", "inclusive90D_MMD2", "speed_Wasserstein_mps", "decomposition_means")}
                  for name, table in tables.items()},
              "current_navigation_terminal_read": False,
              "limitations": ["No new training or on-policy navigation is executed by this reader.",
                  "Marginal training means do not authenticate command-conditioned inactivity or stylistic naturalness.",
                  "Native joint RMS is a movement diagnostic, not gait ground truth or measured contact-point slip.",
                  "Neither existing task XY nor the proposed cost specifies a desired body yaw."]}
    with (OUT / "actual_CPU_review.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "path": str(OUT / "actual_CPU_review.json"),
                      "old_idle_actor_s": result["completed_old_state_action_navigation"]["groups"]["idle"]["duration_actor_s"]}))


if __name__ == "__main__":
    main()
