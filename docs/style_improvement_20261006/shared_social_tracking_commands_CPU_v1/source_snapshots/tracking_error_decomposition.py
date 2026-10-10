"""Diagnostic root/body error decomposition; the world-space score stays primary."""
from __future__ import annotations

import numpy as np

from CrowdSim.humanoid.motion_tracking import body_tracking_errors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inverse_rotate(q, vectors):
    """Express vectors in the measured unit-XYZW root frame; no best-fit rotation."""
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.isfinite(q).all() and np.allclose(norms, 1., atol=1e-3, rtol=0), "Require finite unit root quaternions")
    unit = q / norms
    xyz, w = -unit[:, None, :3], unit[:, None, 3:4]
    twice = 2 * np.cross(xyz, vectors)
    return vectors + w * twice + np.cross(xyz, twice)


def decompose_positions(actual, rotation, reference, reference_rotation):
    """Keep every frame/body and report signed MSE cross terms explicitly.

    Root-relative values are diagnostics only. They cannot replace the original
    world-space threshold or establish a model's tracking success.
    """
    errors = body_tracking_errors(actual, rotation, reference, reference_rotation)
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    rotation, reference_rotation = np.asarray(rotation, np.float64), np.asarray(reference_rotation, np.float64)
    root_difference = actual[:, 0] - reference[:, 0]
    actual_relative = actual - actual[:, :1]
    reference_relative = reference - reference[:, :1]
    relative_difference = actual_relative - reference_relative
    world_difference = actual - reference
    world_mse = np.square(world_difference).sum(-1).mean(-1)
    root_mse = np.square(root_difference).sum(-1)
    relative_mse = np.square(relative_difference).sum(-1).mean(-1)
    cross_mse = 2 * (root_difference[:, None] * relative_difference).sum(-1).mean(-1)
    reconstructed = root_mse + relative_mse + cross_mse
    require(np.allclose(world_mse, reconstructed, atol=1e-12, rtol=0), "Root/body signed MSE identity does not close")
    own_root_difference = inverse_rotate(rotation[:, 0], actual_relative) - inverse_rotate(reference_rotation[:, 0], reference_relative)
    return {**errors, "root_XY_error_m": np.linalg.norm(root_difference[:, :2], axis=-1),
        "root_XYZ_error_m": np.linalg.norm(root_difference, axis=-1),
        "root_height_error_m": np.abs(root_difference[:, 2]),
        "root_rotation_error_rad": errors["body_rotation_error_rad"][:, 0],
        "root_relative_world_axes_mean_body_error_m": np.linalg.norm(relative_difference, axis=-1).mean(-1),
        "own_root_frame_mean_body_error_m": np.linalg.norm(own_root_difference, axis=-1).mean(-1),
        "world_body_MSE_m2": world_mse, "root_translation_MSE_m2": root_mse,
        "root_relative_world_axes_MSE_m2": relative_mse, "signed_cross_MSE_m2": cross_mse,
        "MSE_closure_absolute_error_m2": np.abs(world_mse - reconstructed)}


def verify_capture(arrays, row):
    """Verify full fixed-phase, single-actor captures before diagnostic alignment."""
    steps = row["expected_control_steps"]
    require(type(steps) is int and steps > 0 and row["complete_clip"] is True
        and row["observed_metric_steps"] == steps, "Require the completed full reference sequence")
    count = steps + 1
    metric = arrays["metric_frame"]
    require(metric.shape == (count,) and metric.dtype == np.bool_
        and np.array_equal(metric, np.arange(count) > 0), "Exclude exactly the one native reset frame")
    require(arrays["native_reset_frame"].dtype == arrays["action_valid"].dtype == np.bool_
        and np.array_equal(arrays["native_reset_frame"], ~metric)
        and np.array_equal(arrays["action_valid"], metric), "Reset/action/metric frame masks disagree")
    frames = arrays["reference_frame"]
    window = row["tracking_window"]
    require(window["eligible"] is True and window["full_clip_requested"] is True
        and window["available_steps"] == window["requested_steps"] == steps,
        "Tracking window must retain the full eligible clip")
    require(frames.dtype == np.int64 and np.array_equal(frames, np.arange(window["initial_frame"], window["last_frame"] + 1))
        and frames.shape == (count,) and arrays["reference_time_s"].dtype == np.float64
        and np.array_equal(arrays["reference_time_s"], frames / 30),
        "Reference phase or 30Hz timing changed")
    episode = arrays["episode_id"]
    require(episode.shape == (count, 1) and episode.dtype == np.int64 and np.all(episode == episode[0]),
        "Tracking capture crosses an episode reset")
    falls = arrays["fall_mask"]
    require(falls.shape == (count,) and falls.dtype == np.bool_
        and np.flatnonzero(falls & metric).tolist() == row["fall_capture_frames"], "Fall frames were removed or changed")
    for prefix in ("", "reference_"):
        for name, size in (("pos", 3), ("rot", 4)):
            root, body = arrays[prefix + "root_" + name], arrays[prefix + "rigid_body_" + name]
            require(root.shape == (count, 1, size) and body.shape == (count, 1, 24, size)
                and root.dtype == body.dtype == np.float32 and np.array_equal(root, body[:, :, 0]),
                "Named Pelvis root/body shape, dtype or values disagree")
    values = decompose_positions(*(arrays[key][:, 0] for key in
        ("rigid_body_pos", "rigid_body_rot", "reference_rigid_body_pos", "reference_rigid_body_rot")))
    for key in ("body_position_error_m", "body_rotation_error_rad", "mean_body_position_error_m", "mean_body_rotation_error_rad"):
        require(np.array_equal(values[key], arrays[key]), "Frozen original world metric changed: " + key)
    failures = np.flatnonzero(metric & (values["mean_body_position_error_m"] > .5))
    require(row["tracking_success"] is (not bool(len(failures)))
        and row["first_tracking_failure_capture_frame"] == (int(failures[0]) if len(failures) else None),
        "Original world-space failure latch changed")
    return values
