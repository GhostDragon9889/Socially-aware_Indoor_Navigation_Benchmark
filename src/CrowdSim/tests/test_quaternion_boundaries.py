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


def test_keyboard_kinematics_uses_xyzw_heading_for_native_velocity_write():
    from CrowdSim.tools import test_robot_kinematics as keyboard

    wp = pytest.importorskip("warp")
    proxy = pytest.importorskip("isaaclab.utils.warp.proxy_array").ProxyArray
    quaternion = proxy(wp.array([[0., 0., math.sqrt(.5), math.sqrt(.5)]],
                               dtype=wp.quatf, device="cpu"))
    written = []
    robot = NS(data=NS(root_quat_w=quaternion),
               set_joint_velocity_target=lambda *args, **kwargs: None,
               write_root_velocity_to_sim=lambda values: written.append(values.clone()))
    keyboard._apply_command(robot, None, 1., .2, "kinematic", "cpu", 1, [0, 1])
    torch.testing.assert_close(written[0], torch.tensor([[0., 1., 0., 0., 0., .2]]), atol=1e-6, rtol=0)


def test_keyboard_wheel_command_uses_differential_drive_geometry_and_limits():
    from CrowdSim.tools import test_robot_kinematics as keyboard

    config = keyboard.DifferentialDriveConfig(wheel_radius=.2, wheel_base=.4,
        max_linear_speed=1., max_angular_speed=1., max_wheel_speed=5.)
    written = []
    robot = NS(set_joint_velocity_target=lambda value, **kwargs: written.append((value, kwargs)))
    keyboard._apply_command(robot, config, 2., 2., "wheel", "cpu", 2, [4, 8])
    # Limited (v, omega) = (1, 1), left/right = (4, 6), then wheel limit = 5.
    torch.testing.assert_close(written[0][0], torch.tensor([[4., 5.], [4., 5.]]))
    assert written[0][1] == {"joint_ids": [4, 8]}


@pytest.mark.parametrize("key,linear,angular,quit_requested", [
    ("\x1b[A", .2, 0., False), ("\x1b[B", -.2, 0., False),
    ("\x1b[C", 0., -.3, False), ("\x1b[D", 0., .3, False),
    ("\x1b", 0., 0., True),
])
def test_keyboard_arrows_change_commands_while_escape_quits(key, linear, angular, quit_requested):
    from CrowdSim.tools import test_robot_kinematics as keyboard

    driver = keyboard.KeyboardDriver.__new__(keyboard.KeyboardDriver)
    driver.v = driver.w = 0.
    driver.quit = driver.reset_pose = False
    chars = iter(key)
    driver._read_key = lambda: next(chars, None)
    driver.poll()
    assert (driver.v, driver.w, driver.quit) == (linear, angular, quit_requested)


def test_keyboard_reset_keeps_spawn_height_and_writes_valid_identity_pose():
    from CrowdSim.tools import test_robot_kinematics as keyboard

    wp = pytest.importorskip("warp")
    proxy = pytest.importorskip("isaaclab.utils.warp.proxy_array").ProxyArray
    default_state = torch.tensor([[4., 5., .4, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
    pose, joints = [], []
    robot = NS(data=NS(default_root_state=proxy(wp.from_torch(default_state)),
                       default_joint_pos=proxy(wp.array([[.1, -.1]], dtype=wp.float32, device="cpu")),
                       default_joint_vel=proxy(wp.zeros((1, 2), dtype=wp.float32, device="cpu"))),
               write_root_pose_to_sim=lambda value: pose.append(value.clone()),
               write_root_velocity_to_sim=lambda value: None,
               write_joint_state_to_sim=lambda position, velocity: joints.append((position, velocity)))
    keyboard._reset_robot(robot, "cpu", 1)
    torch.testing.assert_close(pose[0], torch.tensor([[0., 0., .4, 0., 0., 0., 1.]]))
    torch.testing.assert_close(joints[0][0], torch.tensor([[.1, -.1]]))
    torch.testing.assert_close(joints[0][1], torch.zeros(1, 2))
    torch.testing.assert_close(default_state[:, :3], torch.tensor([[4., 5., .4]]))


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_drive_validation_measures_declared_native_turn(monkeypatch, order):
    from CrowdSim.tools import validate_drive_integration as validation

    wp = pytest.importorskip("warp")
    proxy = pytest.importorskip("isaaclab.utils.warp.proxy_array").ProxyArray
    def vectors(values, dtype):
        return proxy(wp.array(values, dtype=dtype, device="cpu"))
    data = NS(root_pos_w=vectors([[0., 0., .4]], wp.vec3f),
              root_quat_w=vectors(yaw_to_quaternion(torch.tensor([.2]), order).tolist(), wp.quatf),
              root_lin_vel_w=vectors([[0., 0., 0.]], wp.vec3f),
              root_ang_vel_w=vectors([[0., 0., 0.]], wp.vec3f))
    def step(action):
        data.root_quat_w = vectors(yaw_to_quaternion(torch.tensor([.3]), order).tolist(), wp.quatf)
        return {}, None, None, None, None
    env = NS(reset=lambda ids: ({}, {}), step=step,
             simulator=NS(native_quaternion_order=order))
    nav = NS(robot=NS(data=data), config=NS(device="cpu"), set_robot_rl_actions=lambda command: None)
    monkeypatch.setattr(validation, "_humanoid_action", lambda *args: torch.zeros(1, 1))
    distance, yaw_delta, velocity = validation._advance(NS(env=env), nav, np.array([0., .2]), 1)
    assert yaw_delta == pytest.approx(.1, abs=1e-6)
    assert distance == 0. and velocity == 0.
