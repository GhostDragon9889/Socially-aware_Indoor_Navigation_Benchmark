"""CPU camera/reset sequencing fixtures, not rendering/physics validation."""

from types import SimpleNamespace as NS

import pytest
import torch

from CrowdSim import crowd_sim
from CrowdSim.ppo import ppo_policy


def test_robot_inference_renders_after_reset_before_camera_reads_and_honors_quit(monkeypatch):
    events, depth_inputs = [], []
    class Env:
        def __init__(self):
            self.steps, self.world, self.camera, self.running = 0, 0., -1., True
            self.simulator = NS(headless=True, _simulation_app=NS(is_running=lambda: True),
                                is_simulation_running=lambda: self.running, _sim=NS(render=self.render))
        def render(self):
            events.append("render")
            self.camera = self.world
        def reset(self, ids):
            if ids is None:
                self.world = 10.
                events.append("full_reset")
            elif ids.numel():
                self.world = 200.
                events.append("human_reset")
            return {}, {}
        def step(self, action):
            self.steps += 1
            assert self.steps <= 2, "Q must stop while the app remains open"
            events.append("step")
            self.world += 1
            if self.steps == 2:
                self.running = False
            return {}, None, torch.tensor([self.steps == 1]), None, {}
    env = Env()
    def observations():
        events.append("camera_read")
        return torch.zeros(1, 5), None, None, torch.full((1, 4, 4), env.camera), None
    def reset_robot(done):
        if done.any():
            env.world = 100.
            events.append("robot_reset")
    nav = NS(config=NS(device=torch.device("cpu"), num_robots=1, rl_depth_enabled=True,
                       rl_depth_size=224, rl_rgb_enabled=False, rl_map_size=0, rl_num_neighbors=0),
             robot_rl_obs_dim=5, robot_rl_vector_obs_dim=5,
             get_robot_rl_observations=observations, set_robot_rl_actions=lambda action: None,
             get_robot_rl_feedback=lambda: (None, None, torch.tensor([env.steps == 1]), {}, None, None, None, None),
             reset_robot_rl_episodes=reset_robot)
    agent = NS(eval=lambda: None, add_agent_info_to_obs=lambda obs: obs,
               obs_dict_to_tensordict=lambda obs: obs, model=lambda obs: {"mean_action": torch.zeros(1, 1)})
    class Model(torch.nn.Module):
        def forward(self, obs, depth, *args):
            assert not self.training
            depth_inputs.append(float(depth[0, 0, 0]))
            return torch.zeros(1, 2), None, None
    class PPO:
        def __init__(self, *args): self.model = Model()
        def load(self, path): return 0
    monkeypatch.setattr(ppo_policy, "RobotPPOTrainer", PPO)
    monkeypatch.setattr(crowd_sim, "record_gif_frame", lambda *args: None)
    crowd_sim.run_masked_mimic_with_robot_ppo(NS(env=env, agent=agent), nav,
        {"car": {"policy_checkpoint": "unused.pt"}})
    assert depth_inputs == [10., 200.]
    assert events == ["full_reset", "render", "camera_read", "step", "robot_reset",
                      "human_reset", "render", "camera_read", "step"]


def test_humanoid_only_inference_respects_simulation_quit_flag():
    steps = []
    simulator = NS(_simulation_app=NS(is_running=lambda: True), is_simulation_running=lambda: not steps)
    def step(action):
        assert not steps
        steps.append(1)
        return {}, None, torch.tensor([False]), None, {}
    env = NS(simulator=simulator, reset=lambda ids: ({}, {}), step=step)
    agent = NS(eval=lambda: None, add_agent_info_to_obs=lambda obs: obs,
               obs_dict_to_tensordict=lambda obs: obs, model=lambda obs: {"mean_action": torch.zeros(1, 1)})
    crowd_sim.run_masked_mimic_policy_loop(NS(env=env, agent=agent))
    assert steps == [1]


def test_main_prints_failure_before_shutdown_and_sets_nonzero_exit(monkeypatch, capsys):
    from CrowdSim.world import builder
    from protomotions.utils import simulator_imports
    closed = []
    simulator = NS(_shutdown_exit_code=None)
    def close():
        closed.append(capsys.readouterr().err)
    env = NS(simulator=simulator, close=close)
    nav = NS(config=NS(car_rl_policy=False), _close_trajectory_logs=lambda: None)
    monkeypatch.setattr(crowd_sim, "parse_args", lambda: NS(env_config="unused.yaml", num_envs=1,
        headless=True, scene_physics=False, empty=False, full_eval=False))
    monkeypatch.setattr(crowd_sim, "load_config", lambda path: {})
    monkeypatch.setattr(simulator_imports, "import_simulator_before_torch", lambda name: None)
    monkeypatch.setattr(builder, "build_env", lambda *args, **kwargs: (env, None, nav, None))
    def fail(runtime): raise RuntimeError("original inference failure")
    monkeypatch.setattr(crowd_sim, "run_masked_mimic_policy_loop", fail)
    with pytest.raises(RuntimeError, match="original inference failure"):
        crowd_sim.main()
    assert simulator._shutdown_exit_code == 1
    assert "RuntimeError: original inference failure" in closed[0]
