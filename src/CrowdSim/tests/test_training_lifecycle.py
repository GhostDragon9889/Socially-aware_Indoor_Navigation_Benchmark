"""Real Torch PPO updates with CPU lifecycle fixtures; no simulator physics."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.ppo import train_ppo, train_ppo_finetune
from CrowdSim.ppo.ppo_policy import RobotPPOConfig, RobotPPOTrainer, RobotRolloutBuffer
from CrowdSim.ppo.reward_model_ppo import RewardCompositionConfig


@pytest.mark.parametrize("finetune", [False, True], ids=["PPO", "reward-model-PPO"])
def test_real_training_counts_done_once_and_refreshes_camera_after_robot_reset(monkeypatch, tmp_path, finetune):
    device = torch.device("cpu")
    events, input_frames, logs, curriculum_outcomes = [], [], [], []
    class Env:
        def __init__(self):
            self.steps, self.world, self.camera = 0, 0., -1.
            self.simulator = NS(headless=True, _sim=NS(render=self.render))
        def render(self):
            self.camera = self.world
            events.append("render")
        def reset(self, ids):
            if ids is None:
                self.world = 10.
                events.append("full_reset")
            return {}, {}
        def step(self, action):
            self.steps += 1
            self.world += 1.
            events.append("step")
            return {}, None, torch.zeros(2, dtype=torch.bool), None, {}
    env = Env()
    def observations():
        input_frames.append(env.camera)
        return torch.tensor([[.5, 0., 1., 0., 0.], [.8, 0., 1., 0., 0.]]), None, None, torch.full((2, 32, 32), env.camera / 100), None
    info = {"reached": torch.tensor([True, True]), "collision": torch.tensor([False, True]),
            "timeout": torch.tensor([True, True]), "stuck": torch.tensor([False, False])}
    def reset_robots(done):
        env.world = 100.
        events.append("robot_reset")
    curriculum = NS(spec=NS(enabled=False), observe=lambda values: curriculum_outcomes.append(values.copy()) or False,
                    metrics=lambda: {}, state_dict=lambda: {})
    nav = NS(config=NS(num_robots=2, update_hz=25., rl_max_episode_steps=10), goal_curriculum=curriculum,
             robot_rl_obs_dim=5, get_robot_rl_observations=observations, set_robot_rl_actions=lambda action: None,
             get_robot_rl_feedback=lambda: (None, torch.tensor([1., 2.]), torch.tensor([True, True]), info,
                                            None, None, None, None), reset_robot_rl_episodes=reset_robots)
    agent = NS(add_agent_info_to_obs=lambda obs: obs, obs_dict_to_tensordict=lambda obs: obs,
               model=lambda obs: {"mean_action": torch.zeros(2, 1)})
    cfg = RobotPPOConfig(obs_dim=5, vector_obs_dim=5, num_neighbors=0,
                        vector_hidden_dims=(8,), actor_hidden_dims=(8,), critic_hidden_dims=(8,),
                        depth_enabled=True, depth_size=32, depth_feature_dim=8,
                        ppo_epochs=1, minibatch_size=4)
    ppo = RobotPPOTrainer(cfg, device)
    before = {key: value.clone() for key, value in ppo.model.state_dict().items()}
    buffer = RobotRolloutBuffer(rollout_steps=2, num_envs=2, obs_dim=5, action_dim=2, device=device, depth_size=32)
    class Logger:
        def __init__(self, *args, **kwargs): pass
        def log(self, step, **metrics): logs.append(metrics)
        def log_trajectory_image(self, *args, **kwargs): pass
        def close(self): pass
    module = train_ppo_finetune if finetune else train_ppo
    monkeypatch.setattr(module, "RobotTrainingLogger", Logger)
    omni, kit, app = ModuleType("omni"), ModuleType("omni.kit"), ModuleType("omni.kit.app")
    app.get_app = lambda: NS(post_quit=lambda: None)
    omni.kit, kit.app = kit, app
    for name, value in (("omni", omni), ("omni.kit", kit), ("omni.kit.app", app)):
        monkeypatch.setitem(sys.modules, name, value)
    runtime = NS(env=env, agent=agent)
    if finetune:
        history = NS(append_and_score=lambda *args: torch.zeros(2), reset=lambda done: None)
        monkeypatch.setattr(module, "_save_training_gif", lambda *args: None)
        module.train_loop(runtime=runtime, nav_manager=nav, ppo=ppo, reference_model=deepcopy(ppo.model).eval(),
            buffer=buffer, reward_history=history, reward_config=RewardCompositionConfig(), reward_stride=1,
            kl_beta=.01, total_steps=4, start_step=0, save_interval=0, output_dir=tmp_path,
            base_ppo=tmp_path / "base.pt", reward_checkpoint=tmp_path / "reward.pt", use_wandb=False, wandb_config={})
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["episodes"] == 4
        assert summary["success_rate"] == summary["collision_rate"] == .5
        assert summary["timeout_rate"] == 1
    else:
        module.train_loop(runtime, nav, ppo, buffer, total_steps=4, start_step=0, save_interval=0,
                          output_dir=tmp_path, use_wandb=False)
    assert logs[0]["episodes"] == 4 and logs[0]["success"] == 2
    assert logs[0]["reached"] == logs[0]["timeout"] == 4
    assert input_frames[:2] == [10., 100.]
    assert events[:7] == ["full_reset", "render", "step", "render", "robot_reset", "render", "step"]
    assert all(np.array_equal(values, [True, False]) for values in curriculum_outcomes)
    assert (tmp_path / "robot_ppo_latest.pt").exists()
    assert all(value.isfinite().all() for value in ppo.model.state_dict().values())
    assert any(not torch.equal(value, before[key]) for key, value in ppo.model.state_dict().items())
