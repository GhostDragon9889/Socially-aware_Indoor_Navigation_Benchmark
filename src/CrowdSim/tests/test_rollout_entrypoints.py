"""CPU lifecycle/CLI checks; these fixtures do not run simulation physics."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.flow import collect_ppo_buffer as collector
from CrowdSim.tools.eval_policy import _evaluate_episode, _reset_evaluation_route
from CrowdSim.tools import eval_policy


@pytest.mark.parametrize("terminal_key,outcome", [
    ("reached", "success"), ("collision", "collision"),
    ("stuck", "stuck"), ("timeout", "timeout"), (None, "timeout"),
])
def test_evaluation_refreshes_observations_and_consumes_humanoid_done(terminal_key, outcome):
    class Env:
        device = torch.device("cpu")
        def __init__(self):
            self.value, self.steps, self.reset_calls, self.renders = 0, 0, [], 0
            self.simulator = NS(_sim=NS(render=self.render))
        def render(self): self.renders += 1
        def reset(self, ids):
            self.reset_calls.append(ids.tolist())
            if ids.numel(): self.value += 100
            return {"value": torch.tensor([self.value]).clone()}, {}
        def step(self, action):
            self.steps += 1
            self.value += 1
            return {}, None, torch.tensor([False, self.steps == 1]), None, {}
    env = Env()
    seen = []
    def humanoid_model(obs):
        seen.append(int(obs["value"][0]))
        return {"mean_action": torch.zeros(2, 1)}
    agent = NS(add_agent_info_to_obs=lambda obs: obs,
               obs_dict_to_tensordict=lambda obs: obs, model=humanoid_model)
    actions, records = [], []
    def feedback():
        done = terminal_key is not None and env.steps == 3
        info = {} if terminal_key is None else {terminal_key: torch.tensor([done])}
        return None, None, torch.tensor([done]), info, None, None, None, None
    nav = NS(config=NS(rl_depth_enabled=True, rl_max_episode_steps=3),
             drive=NS(positions_xy=lambda: np.array([[env.steps, 0.]], np.float32),
                      velocities_xy=lambda: np.zeros((1, 2), np.float32)),
             get_robot_rl_observations=lambda: (torch.zeros(1, 5), None, None, None, None),
             set_robot_rl_actions=actions.append, get_robot_rl_feedback=feedback)
    ppo = NS(model=lambda *args: (torch.zeros(1, 2), None, None))
    traj, result = _evaluate_episode(env, agent, nav, ppo, headless=True,
                                    record_step=lambda xy, velocity: records.append((xy.copy(), velocity.copy())))
    assert seen == [0, 101, 102]
    assert env.reset_calls == [[], [1], []]
    assert env.renders == 3
    assert result == outcome
    assert len(actions) == len(records) == len(traj) == 3
    # A blocked robot's actual speed is zero even though PPO requests motion.
    assert float(actions[0][0, 0]) > 0
    assert [point[2] for point in traj] == [0., 0., 0.]


def test_evaluation_route_uses_snapped_start_and_unified_repeat_reset():
    starts, goals = np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32)
    paths = [np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32)]
    task = NS(pixel_to_world=lambda pixel: np.asarray(pixel)[::-1] + .5,
              refresh_visualization_markers=lambda: None)
    events = []
    nav = NS(config=NS(num_humanoids=1), task=task,
             starts_px=np.zeros((2, 2), np.int64), goals_px=np.zeros((2, 2), np.int64),
             starts_xy=starts, goals_xy=goals, paths_xy=paths,
             _resample_robot_spawn_yaw=lambda index: events.append(("yaw", index)),
             reset_robot_rl_episodes=lambda done, repeat_mask: events.append(
                 ("reset", done.copy(), repeat_mask.copy(), starts[1].copy())))
    route = np.array([[7.5, 5.5], [8.5, 6.5]], np.float32)
    _reset_evaluation_route(nav, np.array([5, 7]), np.array([6, 8]), route)
    np.testing.assert_allclose(starts[1], route[0])
    np.testing.assert_allclose(goals[1], route[-1])
    np.testing.assert_array_equal(paths[1], route)
    assert events[0] == ("yaw", 0)
    assert events[1][0] == "reset"
    np.testing.assert_array_equal(events[1][1], [True])
    np.testing.assert_array_equal(events[1][2], [True])
    np.testing.assert_allclose(events[1][3], route[0])


def test_full_evaluation_freezes_policy_and_resets_before_sampled_route(monkeypatch, tmp_path):
    from CrowdSim import crowd_sim
    from CrowdSim.ppo import ppo_policy
    from CrowdSim.world import builder

    config = {"navigation": {"num_humanoids": 2, "rl": {"goal_curriculum": {"enabled": True}},
                              "path": {"min_start_goal_distance": 7., "max_start_goal_distance": 10.}}}
    events, models = [], []
    class Env:
        num_envs = 2
        device = torch.device("cpu")
        def reset(self, ids):
            events.append("full_reset" if ids is None else "current_obs")
            return {}, {}
        def step(self, action): return {}, None, torch.zeros(2, dtype=torch.bool), None, {}
    env = Env()
    def plan(xy):
        assert events[-1] == "full_reset"
        events.append("sample_route")
        return np.array([2, 3]), np.array([2, 11]), np.array([[3.5, 17.5], [11.5, 17.5]], np.float32)
    task = NS(config=NS(map_origin_xy=(0., 0.), map_resolution=1.),
              height=20, width=20, planner_free_mask=np.ones((20, 20), bool),
              pixel_to_world=lambda pixel: np.array([pixel[1] + .5, 19.5 - pixel[0]]),
              sample_goal_and_plan_path=plan, refresh_visualization_markers=lambda: None)
    nav = NS(task=task, config=NS(num_humanoids=2, num_robots=1, rl_depth_enabled=False, rl_depth_size=224,
                                 rl_map_size=0, rl_num_neighbors=0, rl_max_episode_steps=1),
             robot_rl_obs_dim=5, robot_rl_vector_obs_dim=5,
             starts_px=np.zeros((3, 2), np.int64), goals_px=np.zeros((3, 2), np.int64),
             starts_xy=np.zeros((3, 2), np.float32), goals_xy=np.zeros((3, 2), np.float32),
             paths_xy=[np.zeros((2, 2), np.float32) for _ in range(3)],
             _resample_robot_spawn_yaw=lambda index: None,
             reset_robot_rl_episodes=lambda done, repeat_mask: events.append("robot_route_reset"),
             get_robot_rl_observations=lambda: (torch.zeros(1, 5), None, None, None, None),
             set_robot_rl_actions=lambda action: None,
             get_robot_rl_feedback=lambda: (None, None, torch.tensor([True]), {"reached": torch.tensor([True])},
                                            None, None, None, None))
    nav.drive = NS(positions_xy=lambda: nav.starts_xy[2:3], velocities_xy=lambda: np.zeros((1, 2), np.float32))
    def humanoid_model(obs):
        assert events[-2:] == ["robot_route_reset", "current_obs"]
        return {"mean_action": torch.zeros(2, 1)}
    agent = NS(add_agent_info_to_obs=lambda obs: obs, obs_dict_to_tensordict=lambda obs: obs, model=humanoid_model)
    def build(config, **kwargs):
        assert config["navigation"]["rl"]["goal_curriculum"]["enabled"] is False
        assert config["navigation"]["path"]["min_start_goal_distance"] == 7.
        assert kwargs["num_envs"] == 2
        return env, agent, nav, NS(fabric=NS(device=torch.device("cpu")))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, *args):
            assert not self.training and not self.weight.requires_grad
            return torch.zeros(1, 2), None, None
    class PPO:
        def __init__(self, *args): self.model = Model(); models.append(self.model)
        def load(self, path): pass
    monkeypatch.setattr(crowd_sim, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(builder, "build_env", build)
    monkeypatch.setattr(ppo_policy, "RobotPPOTrainer", PPO)
    monkeypatch.setattr(eval_policy, "save_plots", lambda *args: None)
    monkeypatch.setattr(eval_policy, "_save_evaluation_gif", lambda *args: None)
    omni, kit, app = ModuleType("omni"), ModuleType("omni.kit"), ModuleType("omni.kit.app")
    app.get_app = lambda: NS(post_quit=lambda: events.append("quit"))
    omni.kit, kit.app = kit, app
    for name, module in (("omni", omni), ("omni.kit", kit), ("omni.kit.app", app)):
        monkeypatch.setitem(sys.modules, name, module)
    eval_policy.run_evaluation(NS(ckpt=str(tmp_path / "policy.pt"), env_config="unused.yaml",
                                 num_envs=1, headless=True, search_step=1., min_dist=0.,
                                 num_episodes=1, seed=42, grid_cell=1.))
    metrics = json.loads((tmp_path / "eval_policy" / "metrics.json").read_text())
    assert metrics["success"] == metrics["evaluated_episodes"] == 1
    assert events == ["full_reset", "sample_route", "robot_route_reset", "current_obs", "quit"]
    assert len(models) == 1


@pytest.mark.parametrize("options,collection_values,expected", [
    ([], {}, (False, 0, 0)),
    ([], {"repeat_robot_routes": True, "target_episodes": 8, "route_repeats": 2}, (True, 8, 2)),
    (["--fixed-route-episodes", "5", "--route-repeats", "3"], {}, (True, 5, 3)),
    (["--route-count", "4", "--route-repeats", "2"], {}, (False, 8, 2)),
])
def test_collection_cli_runs_main_and_passes_merged_route_config(monkeypatch, tmp_path, options, collection_values, expected):
    config = {"navigation": {"num_robots": 10, "rl": {"max_linear_velocity": 1., "rgb_enabled": True}},
              "collection": {"output_dir": str(tmp_path), **collection_values},
              "wandb": {"enabled": False}, "gif": {"out": "preview/run.gif"}}
    captured = []
    monkeypatch.setattr("sys.argv", ["collect_ppo_buffer", "--ppo-ckpt", "policy.pt",
                                    "--num-robots", "2", "--stochastic", "--disable-rgb", "--no-gif", *options])
    monkeypatch.setattr(collector, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(collector, "collect", lambda **kwargs: captured.append(kwargs) or {"episodes": [], "meta": {}})
    monkeypatch.setattr(collector, "_atomic_torch_save", lambda payload, path: path.write_bytes(b"saved"))
    collector.main()
    kwargs = captured[0]
    assert (kwargs["repeat_robot_routes"], kwargs["target_episodes"], kwargs["route_repeats"]) == expected
    assert kwargs["config"]["navigation"]["num_robots"] == 2
    assert kwargs["config"]["navigation"]["rl"]["deterministic"] is False
    assert kwargs["config"]["navigation"]["rl"]["rgb_enabled"] is False
    assert kwargs["gif_out"] is None
    assert len(list(tmp_path.glob("raw_*.pt"))) == 1


@pytest.mark.parametrize("option", ["--num-robots", "--fixed-route-episodes", "--route-repeats", "--route-count"])
def test_collection_rejects_nonpositive_cli_route_values_before_collect(monkeypatch, tmp_path, option):
    monkeypatch.setattr("sys.argv", ["collect_ppo_buffer", "--ppo-ckpt", "policy.pt", option, "0"])
    monkeypatch.setattr(collector, "load_config", lambda path: {"navigation": {"rl": {}}, "collection": {"output_dir": str(tmp_path)}})
    monkeypatch.setattr(collector, "collect", lambda **kwargs: pytest.fail("Invalid CLI must not build a simulation"))
    with pytest.raises(ValueError, match="must be positive"):
        collector.main()
