"""Exercise dictionary control overrides against the real MJCF/dataclass schema."""

import pytest

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import ControlConfig
from protomotions.robot_configs.h1_2 import H1_2RobotConfig


def test_dictionary_and_typed_overrides_reach_real_robot_joint_controls():
    original = {"stiffness": 123., "damping": 4., "armature": .02, "friction": .1,
                "effort_limit": 50., "velocity_limit": 6.}
    typed = ControlInfo(stiffness=45., damping=2.)
    config = ControlConfig(override_control_info={"left_knee_joint": original, "right_knee_joint": typed})
    assert config.override_control_info["left_knee_joint"] == ControlInfo(**original)
    assert config.override_control_info["right_knee_joint"] is typed
    robot = H1_2RobotConfig(control=config)
    assert robot.control.control_info["left_knee_joint"] == ControlInfo(**original)
    assert robot.control.control_info["right_knee_joint"].stiffness == 45.
    assert original == {"stiffness": 123., "damping": 4., "armature": .02, "friction": .1,
                        "effort_limit": 50., "velocity_limit": 6.}


@pytest.mark.parametrize("unknown", ["dampng", "nested", "control_type"])
def test_dictionary_overrides_reject_keys_outside_control_info_schema(unknown):
    with pytest.raises(TypeError, match=unknown):
        ControlConfig(override_control_info={".*": {unknown: 1.}})


def test_control_overrides_allow_none_and_already_constructed_control_info():
    assert ControlConfig().override_control_info is None
    typed = ControlInfo(damping=.5)
    assert ControlConfig(override_control_info={".*": typed}).override_control_info[".*"] is typed
