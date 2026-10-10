"""Compare real CPU implementations against the read-only NavIsaaclab2.0 tree.

Run with PYTHONPATH=.:src and the dedicated Python environment. No Kit is started.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("/project/EXP/NavIsaaclab2.0")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def point_goal(module):
    torch.manual_seed(42)
    config = module.PointGoalEnvConfig(
        num_envs=10_000, num_neighbors=4, min_start_goal_distance=2., max_start_goal_distance=4.,
        rl_initial_heading_min_offset_degrees=0., rl_initial_heading_max_offset_degrees=15.,
    )
    env = module.PointGoalBatchEnv(config, torch.device("cpu"))
    delta = env.goal - env.position
    distance = delta.norm(dim=-1)
    heading = env._wrap_angle(torch.atan2(delta[:, 1], delta[:, 0]) - env.heading).abs().rad2deg()
    return {"short_distance_count": int((distance < 2. - 1e-5).sum()),
            "heading_over_15_count": int((heading > 15. + 1e-5).sum()),
            "minimum_distance": float(distance.min()), "maximum_heading_degrees": float(heading.max())}


def episode(length):
    return {"steps": [{"sys_state": np.array([float(i), .2], np.float32),
                       "action": np.array([.5, -.1], np.float32),
                       "depth": np.full((8, 8), i / 10, np.float32)} for i in range(length)]}


def padding(module):
    torch.manual_seed(42)
    model = module.CrowdSimRewardNet(system_dim=4, vec_dim=2, depth_size=8,
        depth_embed_dim=8, state_embed_dim=16, action_embed_dim=16, max_len=8).eval()
    with torch.no_grad():
        alone = float(model([episode(2)])[0])
        mixed = float(model([episode(2), episode(4)])[0])
    return {"short_alone": alone, "short_in_mixed_batch": mixed, "absolute_difference": abs(alone - mixed)}


def infinite_gradient(module, method):
    torch.manual_seed(11)
    trainer = module.RobotPPOTrainer(module.RobotPPOConfig(obs_dim=5, num_neighbors=1,
        map_enabled=False, depth_enabled=False, hidden_dims=(8,), ppo_epochs=1, minibatch_size=2),
        torch.device("cpu"))
    reference = deepcopy(trainer.model).eval()
    buffer = module.RobotRolloutBuffer(2, 1, 5, 2, torch.device("cpu"), num_neighbors=1)
    neighbors, mask = torch.zeros(1, 1, 5), torch.zeros(1, 1, dtype=torch.bool)
    for i in range(2):
        obs = torch.full((1, 5), .1 * (i + 1))
        with torch.no_grad():
            _, raw, logp, value = trainer.act(obs, neighbors=neighbors, neighbor_mask=mask)
        buffer.add(obs, raw, logp, torch.tensor([1. + i]), torch.ones(1, dtype=torch.bool),
                   value, neighbors=neighbors, neighbor_mask=mask)
    buffer.compute_returns_and_advantages(torch.zeros(1), 0., 0.)
    initial = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    hook = next(trainer.model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("inf")))
    error = None
    try:
        if method == "update": trainer.update(buffer)
        else: trainer.update_with_kl(buffer, reference, .1)
    except (RuntimeError, FloatingPointError) as exc:
        error = type(exc).__name__
    finally:
        hook.remove()
    return {"error": error, "all_parameters_finite": all(value.isfinite().all().item()
                for value in trainer.model.state_dict().values()),
            "weights_unchanged": all(torch.equal(value, initial[key])
                for key, value in trainer.model.state_dict().items()),
            "optimizer_state_entries": len(trainer.optimizer.state)}


def main():
    from crowdsim_ppo import point_goal_env, ppo_policy as public_ppo
    from CrowdSim.pref import pref_train_reward
    from CrowdSim.ppo import ppo_policy, train_ppo
    from CrowdSim.ppo.episode_metrics import terminal_counts

    original_point_goal = load_file("reference_point_goal", REFERENCE / "src/crowdsim_ppo/point_goal_env.py")
    original_reward = load_file("reference_reward_wrapper", REFERENCE / "CrowdSim/pref/pref_train_reward.py")
    original_backbone = load_file("reference_reward_backbone", REFERENCE / "CrowdSim/pref/reward_model_utils.py")
    original_reward.RewardNet_Traj_Sys_SelfAttn = original_backbone.RewardNet_Traj_Sys_SelfAttn
    original_training = load_file("reference_training", REFERENCE / "CrowdSim/ppo/train_ppo.py")
    original_ppo = load_file("reference_ppo", REFERENCE / "CrowdSim/ppo/ppo_policy.py")
    original_public_ppo = load_file("reference_public_ppo", REFERENCE / "src/crowdsim_ppo/ppo_policy.py")
    counts = terminal_counts(torch.tensor([True]), {"reached": torch.tensor([True]),
                              "timeout": torch.tensor([True])})
    result = {"reference_root": str(REFERENCE), "current_root": str(ROOT),
        "torch_version": torch.__version__, "device": "cpu", "simulator_started": False,
        "point_goal": {"reference": point_goal(original_point_goal), "current": point_goal(point_goal_env)},
        "preference_padding": {"reference": padding(original_reward), "current": padding(pref_train_reward)},
        "overlapping_events": {
            "reference": original_training.RobotTrainingLogger._outcome_stats({"reached": 1, "timeout": 1}),
            "current": train_ppo.RobotTrainingLogger._outcome_stats(counts)},
        "infinite_backward_gradient": {}}
    for name, original, current in (("CrowdSim", original_ppo, ppo_policy),
                                    ("public", original_public_ppo, public_ppo)):
        result["infinite_backward_gradient"][name] = {
            method: {"reference": infinite_gradient(original, method),
                     "current": infinite_gradient(current, method)} for method in ("update", "update_with_kl")}
    output = Path(__file__).with_name("crowdsim_reproductions.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
