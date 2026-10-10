"""World-frame link-origin velocity contracts; no Isaac Sim Kit startup."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState, RootOnlyState, ResetState, ObjectState, StateConversion,
)
from protomotions.simulator.mujoco.simulator import MujocoSimulator


def _mujoco_robot():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root"><freejoint/><geom pos=".3 0 0" size=".1"/>
        <body name="child" pos="1 0 0"><joint axis="0 0 1"/>
          <geom pos=".4 0 0" size=".1"/>
        </body>
      </body></worldbody></mujoco>''')
    simulator = object.__new__(MujocoSimulator)
    simulator.model = model
    simulator.data = mujoco.MjData(model)
    simulator._has_free_joint = True
    simulator._num_robot_bodies = 2
    simulator._num_actuated_dofs = 1
    # Non-identity root orientation distinguishes local and world angular rates.
    simulator.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    simulator.data.qvel[:] = [.1, -.2, .3, .4, -.5, 2., .6]
    mujoco.mj_forward(model, simulator.data)
    return simulator


def test_mujoco_body_origin_velocities_match_real_fk_derivative():
    sim = _mujoco_robot()
    original_pos = sim.data.xpos[1:3].copy()
    state = sim._get_simulator_bodies_state()
    mujoco.mj_integratePos(sim.model, sim.data.qpos, sim.data.qvel, 1e-7)
    mujoco.mj_forward(sim.model, sim.data)
    derivative = (sim.data.xpos[1:3] - original_pos) / 1e-7
    np.testing.assert_allclose(state.rigid_body_vel[0].numpy(), derivative, atol=1e-6)


def test_mujoco_root_velocity_is_world_frame_and_agrees_with_body_state():
    sim = _mujoco_robot()
    root = sim._get_simulator_root_state()
    bodies = sim._get_simulator_bodies_state()
    torch.testing.assert_close(root.root_vel, bodies.root_vel)
    torch.testing.assert_close(root.root_ang_vel, bodies.root_ang_vel)
    np.testing.assert_allclose(root.root_ang_vel[0].numpy(), [.5, .4, 2.], atol=1e-6)


def test_mujoco_reset_and_impulse_convert_world_angular_velocity():
    sim = _mujoco_robot()
    root = sim._get_simulator_root_state()
    new_state = ResetState(
        state_conversion=StateConversion.SIMULATOR,
        root_pos=root.root_pos, root_rot=root.root_rot,
        root_vel=root.root_vel, root_ang_vel=root.root_ang_vel,
        dof_pos=torch.zeros(1, 1), dof_vel=torch.zeros(1, 1),
    )
    angular_qvel = sim.data.qvel[3:6].copy()
    sim._set_simulator_env_state(new_state)
    np.testing.assert_allclose(sim.data.qvel[3:6], angular_qvel, atol=1e-6)
    sim._apply_root_velocity_impulse(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.tensor([0]))
    np.testing.assert_allclose(sim._get_simulator_root_state().root_ang_vel[0].numpy(), [1.5, .4, 2.], atol=1e-6)
    torch.testing.assert_close(sim._get_simulator_bodies_state().root_ang_vel, sim._get_simulator_root_state().root_ang_vel)


def _isaaclab_methods():
    # Load the actual method bodies, avoiding import of Kit-dependent scene APIs.
    path = Path(__file__).parents[1] / 'simulator/isaaclab/simulator.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IsaacLabSimulator')
    names = {'_get_simulator_bodies_state', '_get_simulator_root_state',
             '_set_simulator_env_state', '_apply_root_velocity_impulse'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, wp=wp, Optional=Optional, RobotState=RobotState,
                     RootOnlyState=RootOnlyState, ResetState=ResetState, ObjectState=ObjectState)
    from protomotions.simulator.base_simulator.simulator_state import StateConversion
    namespace['StateConversion'] = StateConversion
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def _proxy(values, dtype):
    from isaaclab.utils.warp.proxy_array import ProxyArray
    return ProxyArray(wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device='cpu'))


def test_isaaclab3_actual_methods_use_link_velocity_and_writes():
    calls = []
    data = SimpleNamespace(
        body_pos_w=_proxy([[[0, 0, 0], [1, 0, 0]]], wp.vec3f),
        body_quat_w=_proxy([[[0, 0, 0, 1], [0, 0, 0, 1]]], wp.quatf),
        body_link_lin_vel_w=_proxy([[[.1, 0, 0], [.1, 2, 0]]], wp.vec3f),
        body_link_ang_vel_w=_proxy([[[0, 0, 2], [0, 0, 2]]], wp.vec3f),
        body_lin_vel_w=_proxy([[[99, 99, 99], [99, 99, 99]]], wp.vec3f),
        root_pos_w=_proxy([[0, 0, 0]], wp.vec3f),
        root_quat_w=_proxy([[0, 0, 0, 1]], wp.quatf),
        root_link_lin_vel_w=_proxy([[.1, 0, 0]], wp.vec3f),
        root_link_ang_vel_w=_proxy([[0, 0, 2]], wp.vec3f),
        root_link_vel_w=_proxy([[.1, 0, 0, 0, 0, 2]], wp.spatial_vectorf),
    )
    robot = SimpleNamespace(
        data=data,
        write_root_link_state_to_sim=lambda value, ids: calls.append(('state', value.clone(), ids)),
        write_root_link_velocity_to_sim=lambda value, **kw: calls.append(('velocity', value.clone(), kw)),
        set_joint_position_target=lambda *a, **k: None,
        write_joint_state_to_sim=lambda *a, **k: None,
    )
    sim = SimpleNamespace(_robot=robot, num_envs=1, _num_bodies=2)
    methods = _isaaclab_methods()
    state = methods['_get_simulator_bodies_state'](sim)
    root = methods['_get_simulator_root_state'](sim)
    torch.testing.assert_close(state.root_vel, root.root_vel)
    assert state.rigid_body_vel[0, 1, 1] == 2
    reset = ResetState(state_conversion=StateConversion.SIMULATOR, root_pos=root.root_pos, root_rot=root.root_rot,
                       root_vel=root.root_vel, root_ang_vel=root.root_ang_vel,
                       dof_pos=torch.zeros(1, 1), dof_vel=torch.zeros(1, 1))
    methods['_set_simulator_env_state'](sim, reset, env_ids=torch.tensor([0]))
    methods['_apply_root_velocity_impulse'](sim, torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]), torch.tensor([0]))
    assert [call[0] for call in calls] == ['state', 'velocity']
    torch.testing.assert_close(calls[1][1], torch.tensor([[.1, 0., 0., 0., 0., 3.]]))


def test_newton_real_gpu_link_velocities_match_rigid_rotation():
    if not torch.cuda.is_available():
        pytest.skip('Actual Newton integration requires CUDA')
    pytest.importorskip('newton')
    from protomotions.tests.test_newton_simulator_fk import NewtonSimulatorFKHarness, DEFAULT_G1_MOTION_FILE
    harness = NewtonSimulatorFKHarness(motion_file=str(DEFAULT_G1_MOTION_FILE), num_envs=1, device='cuda')
    state = harness.motion_lib.get_motion_state_exact_frame(torch.tensor([0], device='cuda'), torch.tensor([0], device='cuda'))
    state.dof_vel.zero_()
    state.rigid_body_vel.zero_()
    state.rigid_body_ang_vel.zero_()
    state.rigid_body_ang_vel[:, 0] = torch.tensor([.4, -.5, 2.], device='cuda')
    harness.simulator.reset_envs(state)
    actual = harness.simulator.get_robot_state()
    root = harness.simulator.get_root_state()
    offsets = actual.rigid_body_pos - actual.root_pos[:, None]
    expected = torch.cross(state.root_ang_vel[:, None].expand_as(offsets), offsets, dim=-1)
    torch.testing.assert_close(actual.rigid_body_vel, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.root_vel, root.root_vel, atol=2e-5, rtol=0)
    torch.testing.assert_close(actual.root_ang_vel, root.root_ang_vel, atol=2e-5, rtol=0)
    # An after-step angular push must not expose stale body-origin velocities.
    linear = torch.tensor([[.3, .2, -.1]], device='cuda')
    angular = torch.tensor([[.1, .2, .3]], device='cuda')
    harness.simulator._apply_root_velocity_impulse(linear, angular, torch.tensor([0], device='cuda'))
    pushed = harness.simulator.get_robot_state()
    expected_pushed = expected + linear[:, None] + torch.cross(angular[:, None].expand_as(offsets), offsets, dim=-1)
    torch.testing.assert_close(pushed.rigid_body_vel, expected_pushed, atol=2e-5, rtol=2e-5)


