"""Declared kinematic probes of periodic versus episode reference clocks.

These are command/reward diagnostics, not simulated gait or quality scores.
"""
from __future__ import annotations

import numpy as np


def reference_clocks(controls=150, replan_controls=15):
    if (type(controls) is not int or type(replan_controls) is not int
            or not 1 <= replan_controls <= controls <= 150):
        raise ValueError("Require positive integer clocks within the five-second probe")
    pre = np.arange(controls, dtype=np.int64)
    origins = pre // replan_controls * replan_controls
    return origins, (pre + 1 - origins).astype(np.float32) / 30, (pre + 1).astype(np.float32) / 30


def reward_summary(delta_squared, rewards):
    """Complete [method, control, clip] raw trajectory component populations."""
    if (not isinstance(delta_squared, np.ndarray) or delta_squared.dtype != np.float32
            or delta_squared.ndim != 3 or min(delta_squared.shape) < 1
            or not isinstance(rewards, np.ndarray) or rewards.shape != delta_squared.shape
            or rewards.dtype != np.float32 or not np.isfinite(delta_squared).all()
            or not np.isfinite(rewards).all() or (delta_squared < 0).any()
            or (rewards < 0).any() or (rewards > 1).any()):
        raise ValueError("Require finite nonnegative float32 errors and bounded matching rewards")
    distance = np.sqrt(delta_squared.astype(np.float64))
    return [{"actor_controls": int(row.size),
        "mean_trajectory_reward": float(row.astype(np.float64).mean()),
        "mean_root_target_distance_m": float(distance[index].mean()),
        "last_control_mean_trajectory_reward": float(row[-1].astype(np.float64).mean()),
        "per_clip_mean_reward_quantiles_0_25_50_75_100": np.quantile(row.mean(0, dtype=np.float64), [0, .25, .5, .75, 1]).tolist()}
        for index, row in enumerate(rewards)]
