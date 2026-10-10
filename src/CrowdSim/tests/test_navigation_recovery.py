"""Navigation recovery contracts independent of Isaac Sim/Kit."""

import logging
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from CrowdSim.nav_manager import CrowdNavigationManager


def test_failed_replanning_keeps_previous_polyline_and_spawn():
    manager = object.__new__(CrowdNavigationManager)
    manager.config = NS(num_humanoids=1,device=torch.device("cpu"))
    manager.starts_px = np.array([[0,0]])
    manager.goals_px = np.array([[2,0]])
    manager.starts_xy = np.array([[0.,0.]],np.float32)
    manager.goals_xy = np.array([[2.,0.]],np.float32)
    old_route = np.array([[0.,0.],[0.,2.],[2.,2.],[2.,0.]],np.float32)
    manager.paths_xy = [old_route.copy()]
    manager.waypoint_ids = np.array([2])
    manager.reached = np.array([True])
    manager.collision_pairs = {(0,1),(1,2)}
    manager._trajectory_log_file = manager._trajectory_timestamp_file = None
    manager.env = NS(_crowdsim_desired_root_xy=torch.tensor([[9.,9.]]),respawn_root_offset=torch.tensor([[9.,9.,1.]]))
    def cannot_plan(start): raise RuntimeError("No collision-free replacement route")
    manager.task = NS(sample_goal_and_plan_path=cannot_plan,pixel_to_world=lambda point: point.astype(np.float32),
        starts_px=manager.starts_px,goals_px=manager.goals_px,starts_xy=manager.starts_xy,
        goals_xy=manager.goals_xy,paths_xy=manager.paths_xy,refresh_visualization_markers=lambda: None)
    manager._replan_agent_from_xy(0,np.array([9.,9.],np.float32))
    np.testing.assert_array_equal(manager.paths_xy[0],old_route)
    torch.testing.assert_close(manager.env._crowdsim_desired_root_xy[0],torch.tensor([0.,0.]))
    assert manager.collision_pairs == {(1,2)}
    assert not manager.reached[0]


def test_failed_mesh_update_is_logged_once_and_overlay_disabled(caplog):
    manager = object.__new__(CrowdNavigationManager)
    manager.config = NS(num_humanoids=3,num_robots=0,update_hz=25.,device=torch.device("cpu"),
        articulation_quaternion_order="xyzw",humanoid_target_enabled=False,
        visual_markers_enabled=False,car_rl_policy=False)
    manager.sfm_controller = NS(dt=.04)
    manager.task = NS(create_visualization_markers=lambda **kwargs: None)
    manager.path_log_path = Path("unused")
    manager._trajectory_log_file = manager._trajectory_timestamp_file = None
    manager.drive = None
    manager.env_step_count = 0
    manager._should_update_navigation = lambda: False
    manager._clear_robot_rl_env_step_feedback = lambda: None
    class FailedOverlay:
        calls = 0
        def update(self):
            self.calls += 1
            raise RuntimeError("overlay geometry failed")
    overlay = FailedOverlay()
    manager._human_mesh_adapter = overlay
    env = NS(num_envs=3,dt=.04,simulator=NS(headless=True,native_quaternion_order="xyzw"),
        reset_buf=torch.zeros(3,dtype=torch.bool),progress_buf=torch.zeros(3,dtype=torch.long),
        reset=lambda *args,**kwargs: ({},{}),step=lambda action: ({},None,None,None,{}))
    manager.attach(env)
    with caplog.at_level(logging.ERROR,logger="CrowdSim"):
        env.step(None)
        env.step(None)
    assert overlay.calls == 1
    assert manager._human_mesh_adapter is None
    assert manager.env_step_count == 2
    assert sum("SMPL mesh update failed" in record.message for record in caplog.records) == 1
