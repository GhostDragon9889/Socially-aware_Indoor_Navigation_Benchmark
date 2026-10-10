"""CPU geometric contracts; synthetic poses are not physical effect evidence."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.analysis import tracking_error_decomposition as diagnostic
from CrowdSim.humanoid.motion_tracking import body_tracking_errors


def pose(count=2):
    reference = np.tile(np.arange(72, dtype=np.float64).reshape(24, 3) / 4, (count, 1, 1))
    rotation = np.zeros((count, 24, 4), np.float64)
    rotation[..., 3] = 1
    return reference, rotation


def test_pure_translation_still_fails_world_threshold_and_has_zero_relative_error():
    reference, rotation = pose()
    actual = reference + np.array([.75, 0., .25])
    result = diagnostic.decompose_positions(actual, rotation, reference, rotation)
    np.testing.assert_allclose(result["mean_body_position_error_m"], np.sqrt(.625))
    assert (result["mean_body_position_error_m"] > .5).all()
    np.testing.assert_array_equal(result["root_XY_error_m"], [.75, .75])
    np.testing.assert_array_equal(result["root_height_error_m"], [.25, .25])
    np.testing.assert_array_equal(result["root_relative_world_axes_mean_body_error_m"], [0., 0.])
    np.testing.assert_array_equal(result["own_root_frame_mean_body_error_m"], [0., 0.])
    np.testing.assert_array_equal(result["world_body_MSE_m2"], result["root_translation_MSE_m2"])
    np.testing.assert_array_equal(result["signed_cross_MSE_m2"], [0., 0.])


def test_measured_root_rotation_is_separate_from_local_articulation_and_keeps_world_score():
    reference, rotation = pose()
    relative = reference - reference[:, :1]
    actual_relative = np.stack((-relative[..., 1], relative[..., 0], relative[..., 2]), axis=-1)
    actual = reference[:, :1] + actual_relative
    actual_rotation = rotation.copy()
    actual_rotation[:] = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    result = diagnostic.decompose_positions(actual, actual_rotation, reference, rotation)
    np.testing.assert_array_equal(result["root_XY_error_m"], [0., 0.])
    np.testing.assert_allclose(result["root_rotation_error_rad"], np.pi / 2)
    assert (result["root_relative_world_axes_mean_body_error_m"] > .5).all()
    assert (result["mean_body_position_error_m"] > .5).all()
    np.testing.assert_allclose(result["own_root_frame_mean_body_error_m"], 0., atol=1e-14)


def test_local_hand_error_is_preserved_without_root_drift():
    reference, rotation = pose()
    actual = reference.copy()
    actual[:, 18, 2] += 1
    result = diagnostic.decompose_positions(actual, rotation, reference, rotation)
    np.testing.assert_array_equal(result["root_XYZ_error_m"], [0., 0.])
    for name in ("mean_body_position_error_m", "root_relative_world_axes_mean_body_error_m", "own_root_frame_mean_body_error_m"):
        np.testing.assert_allclose(result[name], 1 / 24)


def test_signed_cross_term_can_cancel_root_error_and_is_not_a_positive_contribution():
    reference, rotation = pose()
    actual = reference + [1., 0., 0.]
    actual[:, 1:, 0] -= .8
    result = diagnostic.decompose_positions(actual, rotation, reference, rotation)
    np.testing.assert_allclose(result["root_translation_MSE_m2"], 1.)
    np.testing.assert_allclose(result["root_relative_world_axes_MSE_m2"], 23 * .8 ** 2 / 24)
    np.testing.assert_allclose(result["signed_cross_MSE_m2"], -2 * 23 * .8 / 24)
    np.testing.assert_allclose(result["world_body_MSE_m2"], (1 + 23 * .2 ** 2) / 24)
    np.testing.assert_allclose(result["MSE_closure_absolute_error_m2"], 0., atol=1e-12)


def test_antipodal_root_quaternions_and_common_world_translation_keep_all_errors():
    reference, rotation = pose()
    actual = reference + [.75, .25, -.5]
    original = diagnostic.decompose_positions(actual, rotation, reference, rotation)
    shifted = diagnostic.decompose_positions(actual + [16., -8., 4.], -rotation,
                                             reference + [16., -8., 4.], rotation)
    for name in original:
        np.testing.assert_array_equal(original[name], shifted[name])


@pytest.mark.parametrize("bad", ["position_shape", "nonfinite_position", "nonunit_rotation"])
def test_invalid_geometry_cannot_produce_aligned_diagnostic(bad):
    reference, rotation = pose()
    actual = reference.copy()
    if bad == "position_shape":
        actual = actual[:, :23]
    elif bad == "nonfinite_position":
        actual[0, 18, 2] = np.nan
    else:
        rotation *= 2
    with pytest.raises(ValueError):
        diagnostic.decompose_positions(actual, rotation, reference, rotation)


def capture():
    reference, rotation = pose(4)
    reference, rotation = reference.astype(np.float32), rotation.astype(np.float32)
    actual = reference.copy()
    actual[..., 0] += np.array([9., .5, .625, .125], np.float32)[:, None]
    metric = np.array([False, True, True, True])
    arrays = {"metric_frame": metric, "native_reset_frame": ~metric, "action_valid": metric.copy(),
        "reference_frame": np.arange(9, 13, dtype=np.int64), "reference_time_s": np.arange(9, 13) / 30,
        "episode_id": np.ones((4, 1), np.int64), "fall_mask": np.array([False, False, True, False]),
        "root_pos": actual[:, None, 0], "root_rot": rotation[:, None, 0],
        "rigid_body_pos": actual[:, None], "rigid_body_rot": rotation[:, None],
        "reference_root_pos": reference[:, None, 0], "reference_root_rot": rotation[:, None, 0],
        "reference_rigid_body_pos": reference[:, None], "reference_rigid_body_rot": rotation[:, None],
        **body_tracking_errors(actual, rotation, reference, rotation)}
    row = {"expected_control_steps": 3, "observed_metric_steps": 3, "complete_clip": True,
        "tracking_window": {"initial_frame": 9, "last_frame": 12, "available_steps": 3,
                            "requested_steps": 3, "eligible": True, "full_clip_requested": True},
        "fall_capture_frames": [2], "tracking_success": False, "first_tracking_failure_capture_frame": 2}
    return arrays, row


def test_full_capture_keeps_fall_and_post_failure_frames_and_strict_threshold():
    arrays, row = capture()
    values = diagnostic.verify_capture(arrays, row)
    assert len(values["mean_body_position_error_m"]) == 4
    np.testing.assert_array_equal(values["mean_body_position_error_m"], [9., .5, .625, .125])
    np.testing.assert_array_equal(values["own_root_frame_mean_body_error_m"], np.zeros(4))
    assert row["tracking_success"] is False


@pytest.mark.parametrize("bad", ["drop_failed_frame", "integer_reset_mask", "integer_action_mask",
    "reference_phase", "time_dtype", "episode_reset", "root_not_pelvis", "stored_metric",
    "remove_fall", "first_failure", "aligned_success", "partial_window"])
def test_capture_defects_are_rejected_before_alignment(bad):
    arrays, row = capture()
    arrays, row = deepcopy(arrays), deepcopy(row)
    if bad == "drop_failed_frame":
        arrays["metric_frame"][2] = False
    elif bad == "integer_reset_mask":
        arrays["native_reset_frame"] = arrays["native_reset_frame"].astype(np.int64)
    elif bad == "integer_action_mask":
        arrays["action_valid"] = arrays["action_valid"].astype(np.int64)
    elif bad == "reference_phase":
        arrays["reference_frame"] += 1
    elif bad == "time_dtype":
        arrays["reference_time_s"] = arrays["reference_time_s"].astype(np.float32)
    elif bad == "episode_reset":
        arrays["episode_id"][2:] = 2
    elif bad == "root_not_pelvis":
        arrays["root_pos"] = arrays["root_pos"].copy()
        arrays["root_pos"][2, 0, 0] += 1
    elif bad == "stored_metric":
        arrays["mean_body_position_error_m"][2] = 0
    elif bad == "remove_fall":
        row["fall_capture_frames"] = []
    elif bad == "first_failure":
        row["first_tracking_failure_capture_frame"] = 1
    elif bad == "aligned_success":
        row["tracking_success"] = True
    else:
        row["tracking_window"]["requested_steps"] = 2
    with pytest.raises(ValueError):
        diagnostic.verify_capture(arrays, row)


def test_cli_help_imports_no_torch_or_simulator():
    script = Path(__file__).resolve().parents[1] / "scripts/diagnose_shared_social_tracking_errors.py"
    code = "import runpy,sys; sys.argv=[sys.argv[1],'--help']; " \
        "\ntry: runpy.run_path(sys.argv[0],run_name='__main__')" \
        "\nexcept SystemExit as error: assert error.code==0" \
        "\nassert not any(n.split('.')[0] in {'torch','isaacsim','isaaclab','omni','pxr','warp'} for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert result.returncode == 0, result.stderr
    assert "--baseline-post-receipt" in result.stdout
