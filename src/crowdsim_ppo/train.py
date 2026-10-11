"""End-to-end PPO training entry point for the public PointGoal example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from .goal_curriculum import PPOGoalCurriculum
from .pipeline import build_pipeline, load_config, resolve_device
from .ppo_policy import RobotRolloutBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Configuration JSON (default: bundled PointGoal settings)")
    parser.add_argument("--output", default="output/point_goal_ppo")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = load_config(args.config)
    device = resolve_device(args.device)
    training = values.get("training", {})
    rollout_steps = int(training.get("rollout_steps", 128))
    total_steps = int(args.total_steps if args.total_steps is not None else training.get("total_steps", 1_000_000))
    save_interval = int(training.get("save_interval", 100_000))
    if rollout_steps <= 0 or total_steps <= 0:
        raise ValueError("rollout_steps and total_steps must be positive")
    if save_interval < 0:
        raise ValueError("save_interval must be nonnegative; zero disables periodic saves")
    env, trainer = build_pipeline(values, device)
    curriculum = PPOGoalCurriculum(env.config, values.get("curriculum"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(values, indent=2), encoding="utf-8"
    )

    step = 0
    if args.resume:
        step = trainer.load(args.resume)
        curriculum.load_state_dict(trainer.extra_state.get("curriculum"))
    obs, neighbors, neighbor_mask = env.observe()
    next_save = ((step // save_interval) + 1) * save_interval if save_interval else None
    published_step = None

    while step < total_steps:
        buffer = RobotRolloutBuffer(
            rollout_steps=rollout_steps,
            num_envs=env.num_envs,
            obs_dim=env.obs_dim,
            action_dim=2,
            device=device,
            num_neighbors=env.config.num_neighbors,
            neighbor_dim=env.neighbor_dim,
        )
        completed = 0
        reached_count = 0
        return_sum = 0.0
        length_sum = 0.0
        for _ in range(rollout_steps):
            with torch.no_grad():
                action, raw_action, log_prob, value = trainer.act(
                    obs, neighbors=neighbors, neighbor_mask=neighbor_mask
                )
            next_state, reward, done, info = env.step(action)
            timeouts = info["timeout"]
            if timeouts.any():
                terminal_obs, terminal_neighbors, terminal_mask = info["terminal_observation"]
                with torch.no_grad():
                    terminal_value = trainer.value(
                        terminal_obs[timeouts], neighbors=terminal_neighbors[timeouts],
                        neighbor_mask=terminal_mask[timeouts],
                    )
                # Keep done=True to stop GAE at the reset boundary, but include
                # the final state's value when the episode only hit its limit.
                reward = reward.clone()
                reward[timeouts] += trainer.config.gamma * terminal_value
            buffer.add(
                obs, raw_action, log_prob, reward, done, value,
                neighbors=neighbors, neighbor_mask=neighbor_mask,
            )
            if done.any():
                finished = done.nonzero(as_tuple=False).squeeze(-1)
                success = info["success"][finished]
                curriculum.observe(success.detach().cpu().numpy())
                completed += int(finished.numel())
                reached_count += int(success.sum().item())
                return_sum += float(info["episode_return"][finished].sum().item())
                length_sum += float(info["episode_length"][finished].sum().item())
            obs, neighbors, neighbor_mask = next_state

        with torch.no_grad():
            last_value = trainer.value(
                obs, neighbors=neighbors, neighbor_mask=neighbor_mask
            )
        buffer.compute_returns_and_advantages(
            last_value, trainer.config.gamma, trainer.config.gae_lambda
        )
        metrics = trainer.update(buffer)
        step += rollout_steps * env.num_envs
        summary = {
            "step": step,
            "episodes": completed,
            "success_rate": reached_count / max(completed, 1),
            "episode_return": return_sum / max(completed, 1),
            "episode_length": length_sum / max(completed, 1),
            "curriculum_stage": curriculum.stage.name,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "explained_variance": metrics["explained_variance"],
        }
        print(json.dumps(summary))

        if (next_save is not None and step >= next_save) or step >= total_steps:
            extra_state = {"curriculum": curriculum.state_dict(), "config": values}
            trainer.save(output / f"checkpoint_{step}.pt", step, extra_state)
            trainer.save(output / "latest.pt", step, extra_state)
            published_step = step
            if save_interval:
                next_save = ((step // save_interval) + 1) * save_interval
    # Resume may already satisfy the requested step count. Publish the loaded
    # complete checkpoint in the requested output directory in that case too.
    if published_step != step:
        trainer.save(output / "latest.pt", step,
                     {"curriculum": curriculum.state_dict(), "config": values})
    return output / "latest.pt"


def main() -> None:
    checkpoint = train(parse_args())
    print(f"saved: {checkpoint}")


if __name__ == "__main__":
    main()
