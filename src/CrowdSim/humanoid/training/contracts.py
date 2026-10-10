"""Explicit physical rollout boundaries for PPO+action-conditioned AMP."""

from __future__ import annotations

import torch


def validate_rollout(rollout: dict, *, self_dim: int, task_dim: int,
                     history_frames: int, amp_frame_dim: int,
                     action_mask=None) -> tuple[int, int]:
    required = {"self_obs", "task_obs", "amp_history", "actions", "applied_actions",
                "log_prob", "values", "next_values", "rewards", "terminated", "truncated"}
    missing = required - rollout.keys()
    if missing:
        raise ValueError(f"Rollout is missing actual physical fields: {sorted(missing)}")
    rewards = rollout["rewards"]
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError("Rewards must have nonempty (T,B) shape")
    steps, envs = rewards.shape
    shapes = {
        "self_obs": (steps, envs, self_dim), "task_obs": (steps, envs, task_dim),
        "amp_history": (steps, envs, history_frames, amp_frame_dim),
        "actions": (steps, envs, 69), "applied_actions": (steps, envs, 69),
    }
    for key in required:
        tensor = rollout[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shapes.get(key, (steps, envs)):
            raise ValueError(f"Rollout {key} has an incompatible shape")
        if key in {"terminated", "truncated"}:
            if tensor.dtype != torch.bool:
                raise ValueError(f"Rollout {key} must distinguish episode termination using bool")
        elif not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Rollout {key} must contain finite floating point data")
    if torch.any(rollout["applied_actions"].abs() > 1 + 1e-6):
        raise ValueError("Applied normalized PD actions must be in [-1,1]")
    # Store pre-clamp Gaussian samples for valid PPO likelihoods, and actual
    # post-clamp actions separately for the action-conditioned discriminator.
    mask = torch.ones(69, device=rollout["actions"].device) if action_mask is None else torch.as_tensor(
        action_mask, device=rollout["actions"].device, dtype=rollout["actions"].dtype
    )
    if mask.shape != (69,) or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Declared normalized action projection must be a binary69 mask")
    if not torch.allclose(rollout["applied_actions"], rollout["actions"].clamp(-1, 1) * mask, atol=1e-6, rtol=0):
        raise ValueError("Applied actions must equal the declared policy-to-physics clamp and mask projection")
    if "episode_ids" in rollout or "history_episode_ids" in rollout:
        if "episode_ids" not in rollout or "history_episode_ids" not in rollout:
            raise ValueError("Both current and history episode IDs are required to audit history resets")
        episode = rollout["episode_ids"]
        history = rollout["history_episode_ids"]
        if tuple(episode.shape) != (steps, envs) or tuple(history.shape) != (steps, envs, history_frames):
            raise ValueError("Episode ID history shape does not match the AMP window")
        if torch.any(history != episode[..., None]):
            raise ValueError("AMP history crosses an environment reset: initialize reference history explicitly")
    return steps, envs


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           *, gamma=.99, gae_lambda=.95):
    """Physical terminal states do not bootstrap; time limits do, but end GAE.

    next_values at a time limit must be evaluated on the terminal observation,
    before an environment reset. A reset observation is not a valid substitute.
    """
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("Rewards/current/next values must have identical (T,B) shapes")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("Episode boundary masks must have the reward shape")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_values[index] * (~terminated[index]) - values[index]
        carry = delta + gamma * gae_lambda * (~(terminated[index] | truncated[index])) * carry
        advantages[index] = carry
    return advantages, advantages + values
