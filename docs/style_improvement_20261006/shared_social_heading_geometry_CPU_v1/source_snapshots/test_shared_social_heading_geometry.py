"""Analytic rigid rotations catch root/hip-axis sign and projection mistakes."""
import numpy as np
import pytest

from CrowdSim.humanoid.analysis.heading_geometry import body_axes, summarize, SMPL_BODY_NAMES


def pose(yaw=0., *, swap=False):
    q = np.array([0., 0., np.sin(yaw/2), np.cos(yaw/2)])
    rotation = np.broadcast_to(q, (3, 24, 4)).copy()
    positions = np.zeros((3, 24, 3))
    lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.]) * (-.2 if swap else .2)
    positions[:, SMPL_BODY_NAMES.index("L_Hip")] = lateral/2
    positions[:, SMPL_BODY_NAMES.index("R_Hip")] = -lateral/2
    positions += np.array([12., -7., 1.])
    return positions, rotation


@pytest.mark.parametrize("yaw", [0., .7, -1.9, np.pi])
def test_anatomical_forward_follows_measured_root_X_under_yaw(yaw):
    result = body_axes(*pose(yaw))
    assert np.allclose(result["hip_local_unit"], [0., 1., 0.], atol=1e-12)
    assert np.allclose(result["anatomical_minus_root_heading_rad"], 0., atol=1e-12)
    assert result["heading_comparison_defined"].all()
    assert summarize(result)["frames"] == 3


def test_swapped_left_right_is_measured_as_reversed_not_silently_corrected():
    result = body_axes(*pose(swap=True))
    assert np.allclose(result["hip_to_local_Y_deg"], 180.)
    assert np.allclose(np.abs(result["anatomical_minus_root_heading_rad"]), np.pi)


def test_zero_projected_forward_has_explicit_mask_and_retains_frames():
    p, q = pose()
    q[:] = [0., np.sin(np.pi/4), 0., np.cos(np.pi/4)]
    result = body_axes(p, q)
    assert not result["heading_comparison_defined"].any()
    report = summarize(result)
    assert report["frames"] == report["undefined_heading_comparison_frames"] == 3
    assert report["signed_anatomical_minus_root_heading_rad"]["mean"] is None


@pytest.mark.parametrize("problem", ["shape", "nonfinite", "integer", "nonunit", "degenerate", "empty"])
def test_invalid_geometry_rejected(problem):
    p, q = pose()
    if problem == "shape":
        p = p[:, :23]
    if problem == "nonfinite":
        p[0, 0, 0] = np.nan
    if problem == "integer":
        p = p.astype(np.int64)
    if problem == "nonunit":
        q *= 2
    if problem == "degenerate":
        p[:, SMPL_BODY_NAMES.index("L_Hip")] = p[:, SMPL_BODY_NAMES.index("R_Hip")]
    if problem == "empty":
        p, q = p[:0], q[:0]
    with pytest.raises(ValueError):
        body_axes(p, q)
