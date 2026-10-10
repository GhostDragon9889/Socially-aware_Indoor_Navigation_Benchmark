"""PoseLib's MJCF frame/anchor contract against independent real MuJoCo FK."""
import numpy as np
import mujoco
import pytest
import torch

from protomotions.components.pose_lib import (
    extract_kinematic_info, extract_transforms_from_qpos,
    compute_forward_kinematics_from_transforms, extract_qpos_from_transforms,
)


def _asset(tmp_path, orientation='', joints='<joint name="hinge" axis="0 0 2" pos=".2 .1 0" ref=".13"/>', compiler='angle="radian"'):
    path = tmp_path / 'anchored.xml'
    path.write_text(f'''<mujoco><compiler {compiler}/><worldbody>
       <body name="root"><freejoint/><geom size=".1"/>
         <body name="child" pos="1 .2 .3" {orientation}>{joints}<geom size=".1"/>
           <body name="tip" pos=".4 .1 0"><geom size=".05"/></body>
         </body></body></worldbody></mujoco>''')
    return path


@pytest.mark.parametrize('orientation,compiler', [
    ('', 'angle="radian"'),
    ('quat="1 .2 -.1 .3"', 'angle="radian"'),
    ('euler=".2 -.3 .4"', 'angle="radian" eulerseq="XYZ"'),
    ('euler="20 -30 40"', 'angle="degree"'),
    ('axisangle="1 2 3 .4"', 'angle="radian"'),
    ('xyaxes="1 .2 .3 -.2 1 .1"', 'angle="radian"'),
    ('zaxis=".2 .3 1"', 'angle="radian"'),
])
def test_nonzero_joint_anchor_and_body_orientation_match_mujoco(tmp_path, orientation, compiler):
    path = _asset(tmp_path, orientation, compiler=compiler)
    info = extract_kinematic_info(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    qpos = torch.tensor([[.4, -.5, .7, .9, .1, -.2, .3, angle] for angle in (-.8, .13, .9)])
    root_pos, transforms = extract_transforms_from_qpos(info, qpos)
    positions, rotations = compute_forward_kinematics_from_transforms(info, root_pos, transforms)
    for i in range(len(qpos)):
        data.qpos[:] = qpos[i].double().numpy()
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(positions[i].numpy(), data.xpos[1:], atol=3e-7)
        np.testing.assert_allclose(rotations[i].numpy(), data.xmat[1:].reshape(-1, 3, 3), atol=4e-7)
    recovered = extract_qpos_from_transforms(info, root_pos, transforms)
    torch.testing.assert_close(recovered[:, 7:], qpos[:, 7:], atol=2e-7, rtol=1e-6)


def test_three_hinges_common_anchor_matches_mujoco(tmp_path):
    joints = ''.join(f'<joint name="j{i}" axis="{axis}" pos=".2 .1 0"/>' for i, axis in enumerate(['1 0 0', '0 1 0', '0 0 1']))
    path = _asset(tmp_path, joints=joints)
    info = extract_kinematic_info(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    qpos = torch.tensor([[0., 0., 0., 1., 0., 0., 0., .2, -.3, .4]])
    root, transforms = extract_transforms_from_qpos(info, qpos)
    positions, rotations = compute_forward_kinematics_from_transforms(info, root, transforms)
    data.qpos[:] = qpos[0].numpy()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(positions[0].numpy(), data.xpos[1:], atol=2e-7)
    np.testing.assert_allclose(rotations[0].numpy(), data.xmat[1:].reshape(-1, 3, 3), atol=2e-7)


def test_noncoincident_three_hinge_pivots_fail_explicitly(tmp_path):
    joints = ''.join(f'<joint name="j{i}" axis="{axis}" pos="{i*.1} 0 0"/>' for i, axis in enumerate(['1 0 0', '0 1 0', '0 0 1']))
    with pytest.raises(ValueError, match='noncoincident'):
        extract_kinematic_info(str(_asset(tmp_path, joints=joints)))


def test_slide_joint_is_not_misinterpreted_as_hinge(tmp_path):
    with pytest.raises(ValueError, match='Unsupported joint type'):
        extract_kinematic_info(str(_asset(tmp_path, joints='<joint name="slide" type="slide" axis="1 0 0"/>')))


@pytest.mark.parametrize('value', [0., float('nan'), float('inf')])
def test_invalid_root_quaternion_does_not_silently_produce_nan_fk(tmp_path, value):
    info = extract_kinematic_info(str(_asset(tmp_path)))
    qpos = torch.zeros(1, info.nq)
    qpos[:, 3] = value
    with pytest.raises(ValueError, match='finite.*nonzero'):
        extract_transforms_from_qpos(info, qpos)
