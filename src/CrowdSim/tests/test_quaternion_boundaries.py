"""Native articulation and USD math checks; these fixtures do not run physics."""

import math
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.control.drive import (
    DriveController, articulation_torch_view, convert_quaternion_order, quaternion_yaw,
    resolve_articulation_quaternion_order, yaw_to_quaternion,
)
from CrowdSim.scene_setup import apply_fixed_crowd_robot_spawns
from CrowdSim.nav_manager import CrowdNavigationManager


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_yaw_read_write_and_motion_use_declared_native_order(order):
    angles = torch.tensor([0., math.pi / 2, -math.pi / 2])
    quats = yaw_to_quaternion(angles, order)
    torch.testing.assert_close(quaternion_yaw(quats, order), angles)
    config = NS(num_robots=1, articulation_quaternion_order=order,
                rl_max_linear_velocity=1., rl_max_angular_velocity=1., map_resolution=.05)
    written = []
    robot = NS(data=NS(root_pos_w=torch.zeros(1,3), root_quat_w=quats[1:2]),
        write_root_pose_to_sim=lambda poses, env_ids=None: written.append(poses.clone()),
        write_root_velocity_to_sim=lambda velocities, env_ids=None: None)
    controller = DriveController(robot, config, torch.device("cpu"))
    controller._env_dt = .1
    controller.set_action(torch.tensor([[1., 0.]]))
    controller._apply_kinematic()
    torch.testing.assert_close(written[0][0,:2], torch.tensor([0., .1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(written[0][:,3:7], quats[1:2])


def test_native_order_is_explicit_and_invalid_value_is_rejected():
    config = NS(articulation_quaternion_order="wxyz")
    env = NS(simulator=NS(native_quaternion_order="xyzw"))
    assert resolve_articulation_quaternion_order(env=env, config=config) == "xyzw"
    env._crowdsim_articulation_quaternion_order = "wxyz"
    assert resolve_articulation_quaternion_order(env=env, config=config) == "wxyz"
    with pytest.raises(ValueError, match="Quaternion order"):
        resolve_articulation_quaternion_order(config=NS(articulation_quaternion_order="xyz"))
    with pytest.raises(ValueError, match="quaternions"):
        convert_quaternion_order(torch.zeros(2,3), "xyzw", "wxyz")


def test_numpy_conversion_preserves_rotation_for_usd():
    xyzw = np.array([[0.,0.,math.sin(.3),math.cos(.3)]], dtype=np.float32)
    wxyz = convert_quaternion_order(xyzw, "xyzw", "wxyz")
    np.testing.assert_allclose(wxyz, xyzw[:,[3,0,1,2]])
    np.testing.assert_allclose(convert_quaternion_order(wxyz,"wxyz","xyzw"), xyzw)


def test_native_proxy_array_components_use_real_torch_view(monkeypatch):
    """Use pinned SDK's real Warp-backed ProxyArray on CPU, without Kit."""
    wp = pytest.importorskip("warp")
    proxy_module = pytest.importorskip("isaaclab.utils.warp.proxy_array")
    proxy = proxy_module.ProxyArray
    angle = math.pi / 2
    quats = proxy(wp.array([[0., 0., math.sin(angle / 2), math.cos(angle / 2)]],
                          dtype=wp.quatf, device="cpu"))
    positions = proxy(wp.array([[1., 2., 3.]], dtype=wp.vec3f, device="cpu"))
    assert quats.shape == (1,)
    assert positions.shape == (1,)
    assert articulation_torch_view(quats).shape == (1, 4)
    torch.testing.assert_close(convert_quaternion_order(quats, "xyzw", "wxyz"),
                               quats.torch[:, [3, 0, 1, 2]])
    config = NS(num_robots=1, articulation_quaternion_order="xyzw",
                rl_max_linear_velocity=1., rl_max_angular_velocity=1., map_resolution=.05)
    written = []
    robot = NS(data=NS(root_pos_w=positions, root_quat_w=quats,
                      root_ang_vel_w=proxy(wp.array([[0., 0., .7]], dtype=wp.vec3f, device="cpu"))),
               write_root_pose_to_sim=lambda value, env_ids=None: written.append(value.clone()),
               write_root_velocity_to_sim=lambda value, env_ids=None: None)
    drive = DriveController(robot, config, torch.device("cpu"))
    drive._env_dt = .1
    np.testing.assert_allclose(drive.positions_xy(), [[1., 2.]])
    np.testing.assert_allclose(drive.yaws(), [angle], atol=1e-6)
    monkeypatch.setattr(drive, "_sync_prim_transforms", lambda: None)
    drive.sync_teleported_poses(torch.cat((positions.torch, quats.torch), dim=1))
    drive.set_action(torch.tensor([[1., 0.]]))
    drive._apply_kinematic()
    torch.testing.assert_close(written[-1][0, :3], torch.tensor([1., 2.1, 3.]))
    nav = CrowdNavigationManager.__new__(CrowdNavigationManager)
    nav.config, nav.robot, nav.drive = config, robot, None
    nav._trajectory_log_file = nav._trajectory_timestamp_file = None
    np.testing.assert_allclose(nav._robot_yaws(), [angle], atol=1e-6)
    np.testing.assert_allclose(nav._robot_angular_velocities(), [.7], atol=1e-6)


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
@pytest.mark.parametrize("cached", [True, False])
def test_prim_sync_converts_native_quaternion_to_usd_wxyz(monkeypatch, order, cached):
    values = {}
    class Op:
        def __init__(self, name): self.name = name
        def GetName(self): return self.name
        def Set(self, value): values[self.name] = value
    prim = NS(IsValid=lambda: True)
    stage = NS(GetPrimAtPath=lambda path: prim)
    pxr = ModuleType("pxr")
    pxr.UsdGeom = NS(Xformable=lambda p: NS(GetOrderedXformOps=lambda: [Op("xformOp:translate"),Op("xformOp:orient")]))
    pxr.Gf = NS(Vec3d=lambda *v: v, Quatd=lambda *v: v)
    omni = ModuleType("omni")
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: NS(get_stage=lambda: stage)
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    quat = yaw_to_quaternion(torch.tensor([.6]), order)
    robot = NS(data=NS(root_pos_w=torch.tensor([[1.,2.,3.]]), root_quat_w=quat))
    controller = DriveController(robot, NS(num_robots=1,articulation_quaternion_order=order), torch.device("cpu"))
    if cached:
        controller._last_written_poses = torch.cat((robot.data.root_pos_w, quat), dim=1)
    controller._sync_prim_transforms()
    np.testing.assert_allclose(values["xformOp:orient"], [math.cos(.3),0.,0.,math.sin(.3)], atol=1e-7)
    assert values["xformOp:translate"] == (1.,2.,3.)


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_spawn_writes_native_order_and_stores_same_reset_pose(order):
    written = []
    default = torch.zeros(2,13)
    default[:,3:7] = yaw_to_quaternion(torch.zeros(2), order)
    robot = NS(data=NS(default_root_state=default, default_joint_pos=torch.zeros(2,2), default_joint_vel=torch.zeros(2,2)),
        write_root_pose_to_sim=lambda value: written.append(value.clone()),
        write_root_velocity_to_sim=lambda value: None,
        write_joint_state_to_sim=lambda pos,vel: None)
    class Scene(dict):
        def reset(self): pass
    env = NS(num_envs=2, device=torch.device("cpu"),
        _crowdsim_articulation_quaternion_order=order,
        simulator=NS(_scene=Scene(crowdsim_robot=robot)))
    poses = torch.tensor([[1.,2.,0.],[3.,4.,math.pi/2]])
    apply_fixed_crowd_robot_spawns(env, poses)
    torch.testing.assert_close(written[-1][:,:2], poses[:,:2])
    torch.testing.assert_close(quaternion_yaw(written[-1][:,3:7],order), poses[:,2])
    torch.testing.assert_close(env._crowdsim_robot_spawn_poses, written[-1])
