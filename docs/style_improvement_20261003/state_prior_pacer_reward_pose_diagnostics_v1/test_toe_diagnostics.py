"""Necessary analytic checks for the new geometric toe-only diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("toe_diagnostic", HERE / "diagnose_tracking.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
STYLE = TOOL.style_module()


def toe_motion(height, vx):
    pos = np.zeros((4, 24, 3), dtype=np.float64)
    for name in ("L_Toe", "R_Toe"):
        pos[:, STYLE.BODY_NAMES.index(name), 2] = height
        pos[:, STYLE.BODY_NAMES.index(name), 0] = np.arange(4) * vx / 30
    return pos


def test_ground_origin_known_world_speed():
    arrays = TOOL.toe_intervals(toe_motion(.02, .3), STYLE)
    np.testing.assert_allclose(arrays["speed"], .3, atol=1e-14)
    assert arrays["stance"].all()


def test_airborne_horizontal_motion_never_becomes_zero_slip_measurement():
    arrays = TOOL.toe_intervals(toe_motion(.5, .3), STYLE)
    assert not arrays["stance"].any()
    assert STYLE.summary(arrays["speed"][arrays["stance"]])["mean"] is None


def test_both_endpoints_and_vertical_speed_are_required_negative_height_retained():
    pos = toe_motion(-.01, .3)
    left = STYLE.BODY_NAMES.index("L_Toe")
    pos[1, left, 2] = .2
    arrays = TOOL.toe_intervals(pos, STYLE)
    assert arrays["stance"][:, 0].tolist() == [False, False, True]
    assert arrays["stance"][:, 1].all()
    assert (arrays["end_height"][:, 1] < 0).all()
