"""Euler-hinge backends must not silently accept exp-map humanoid policies."""
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.base_simulator.utils import validate_three_dof_representation
from protomotions.simulator.newton.config import NewtonSimulatorConfig
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
from protomotions.utils.rotations import exp_map_to_quat, quaternion_to_matrix


@pytest.mark.parametrize('backend', ['Newton','MuJoCo'])
def test_bundled_smpl_exp_map_state_is_rejected_before_simulation_initialization(backend):
    cfg = robot_config('smpl')
    if backend == 'Newton':
        from protomotions.simulator.newton.simulator import NewtonSimulator
        simulator = NewtonSimulator
        config = NewtonSimulatorConfig(num_envs=1, headless=True, experiment_name="representation_contract")
    else:
        from protomotions.simulator.mujoco.simulator import MujocoSimulator
        simulator = MujocoSimulator
        config = MujocoSimulatorConfig(num_envs=1, headless=True, experiment_name="representation_contract")
    with pytest.raises(NotImplementedError, match='independent hinge/Euler.*exp_map'):
        simulator(config, cfg, terrain=None, device=torch.device('cpu'), scene_lib=None)


def test_hinge_override_is_explicit_and_single_axis_g1_stays_compatible():
    smpl = robot_config('smpl')
    validate_three_dof_representation(smpl.kinematic_info,'hinge','Newton')
    g1 = robot_config('g1')
    validate_three_dof_representation(g1.kinematic_info,'exp_map','Newton')
    with pytest.raises(ValueError,match='must be'):
        validate_three_dof_representation(g1.kinematic_info,'quaternion','Newton')


def test_old_config_missing_new_field_defaults_to_exp_map_guard():
    smpl = robot_config('smpl')
    from protomotions.simulator.newton.simulator import NewtonSimulator
    with pytest.raises(NotImplementedError, match='exp_map'):
        NewtonSimulator(SimpleNamespace(), smpl, terrain=None,device=torch.device('cpu'),scene_lib=None)


def test_real_mujoco_three_hinges_do_not_match_exp_map_for_non_small_angles():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="root"><freejoint/><geom size=".1"/>
        <body name="joint"><joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom size=".1"/>
          <body name="tip" pos="1 0 0"><geom size=".1"/></body>
        </body></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    data.qpos[7:]=[.8,-.6,.5]
    mujoco.mj_forward(model,data)
    rot=quaternion_to_matrix(exp_map_to_quat(torch.tensor([[.8,-.6,.5]]),w_last=True),w_last=True)[0].numpy()
    exp_map_tip=rot @ np.array([1.,0.,0.])
    assert np.linalg.norm(data.xpos[3]-exp_map_tip) > .1
