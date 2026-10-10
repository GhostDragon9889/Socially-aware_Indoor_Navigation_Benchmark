"""CPU checks for actual PRE commands versus POST tracking captures."""
from __future__ import annotations

import numpy as np

from .tracking_error_decomposition import require, verify_capture


def check_array(arrays, name, shape, dtype):
    value = arrays[name]
    require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
            "Unexpected command channel shape, dtype or finite values: " + name)
    return value


def audit_capture(arrays, sequence):
    """Retain every frame; never rotate a PRE command using a POST root.

    Prefix math follows the source's float64 subtraction/trigonometry and final
    float32 cast. Its numerical bound allows cast rounding plus 32 float64 eps
    times operand magnitude, rather than a fixed metric tolerance.
    """
    verify_capture(arrays, sequence)
    count = sequence["expected_control_steps"] + 1
    pre = np.maximum(np.arange(count) - 1, 0)
    reference = arrays["reference_frame"]
    phase = check_array(arrays, "waypoint_query_phase_frame", (count,), np.int64)
    require(np.array_equal(phase, reference[pre]), "Command query must use the PRE reference phase")
    requested = phase[:, None] + np.arange(15, 151, 15)
    held = check_array(arrays, "waypoint_held_at_clip_end", (count, 10), np.bool_)
    require(np.array_equal(held, requested > reference[-1]), "Endpoint hold mask changed")
    indices = np.minimum(requested, reference[-1]) - reference[0]
    expected_world = arrays["reference_root_pos"][indices, 0, :2][:, None]
    command = check_array(arrays, "trajectory_xy", (count, 1, 10, 2), np.float32)
    require(np.array_equal(command, expected_world), "World commands differ from actual fixed-phase reference")
    for name, expected in (("observation_actual_pelvis_quat_xyzw", arrays["root_rot"][pre]),
                           ("observation_episode_ids", arrays["episode_id"][pre])):
        require(arrays[name].dtype == expected.dtype and np.array_equal(arrays[name], expected),
                "Observation is not the PRE root/episode: " + name)
    clocks = check_array(arrays, "observation_physics_step", (count,), np.int64)
    ages = check_array(arrays, "observation_age", (count, 1), np.int64)
    require(np.array_equal(clocks, clocks[0] + 4 * pre)
        and np.array_equal(ages, ages[0] + pre[:, None]), "PRE clock must increment four physical substeps per control")
    heading = check_array(arrays, "observation_used_task_heading", (count, 1), np.float32)[:, 0]
    q = arrays["observation_actual_pelvis_quat_xyzw"][:, 0]
    x, y, z, w = q.T
    projection_x, projection_y = 2 * w ** 2 - 1 + 2 * x ** 2, 2 * w * z + 2 * x * y
    defined = check_array(arrays, "observation_task_heading_defined", (count, 1), np.bool_)[:, 0]
    fallback = check_array(arrays, "observation_singular_heading_fallback", (count, 1), np.bool_)[:, 0]
    require(np.array_equal(defined, np.hypot(projection_x, projection_y) > 1e-12)
        and np.array_equal(fallback, ~defined), "Projected-X heading definition changed")
    angle = np.arctan2(projection_y, projection_x)
    difference = heading.astype(np.float64) - angle.astype(np.float64)
    wrapped = np.arctan2(np.sin(difference), np.cos(difference))
    require(np.all(np.abs(wrapped[defined]) <= 8 * np.finfo(np.float32).eps),
            "Used heading differs from measured PRE projected-X heading")
    physical_fall = check_array(arrays, "observation_physical_fall", (count, 1), np.bool_)[:, 0]
    for frame in np.flatnonzero(fallback):
        require(frame > 0 and physical_fall[frame] and defined[:frame].any()
            and heading[frame] == heading[frame - 1], "Singular fallback must retain a same-episode prior heading after a fall")
    delta = command[:, 0].astype(np.float64) - arrays["root_pos"][pre, 0, :2].astype(np.float64)[:, None]
    cosine, sine = np.cos(heading.astype(np.float64))[:, None], np.sin(heading.astype(np.float64))[:, None]
    local = np.stack((cosine * delta[..., 0] + sine * delta[..., 1],
                      -sine * delta[..., 0] + cosine * delta[..., 1]), axis=-1)
    task = check_array(arrays, "observation_raw_task", (count, 1, 12308), np.float32)
    actual = task[:, 0, :20].reshape(count, 10, 2).astype(np.float64)
    bound = .5 * np.abs(np.spacing(local.astype(np.float32)).astype(np.float64))
    bound += (32 * np.finfo(np.float64).eps * np.abs(delta).sum(-1))[..., None]
    difference = np.abs(actual - local)
    require(np.all(difference <= bound), "Trajectory prefix exceeds float32 cast/float64 computation error bound")
    metric = arrays["metric_frame"]
    root_error = arrays["root_pos"][metric, 0, :2].astype(np.float64) - arrays["reference_root_pos"][metric, 0, :2].astype(np.float64)
    # An offline application of paper Eq.(4), not the captured training reward.
    formula_reward = np.exp(-2 * np.square(root_error).sum(-1))
    return {"metric_frames": int(metric.sum()), "metric_command_samples": int(metric.sum()) * 10,
        "held_endpoint_samples": int(held[metric].sum()), "all_ten_held_frames": int(held[metric].all(-1).sum()),
        "maximum_prefix_float64_difference_m": float(difference.max()),
        "maximum_defined_heading_wrapped_difference_rad": float(np.abs(wrapped[defined]).max(initial=0.)),
        "singular_heading_fallback_frames": int(fallback[metric].sum()),
        "fall_frames_retained": int(arrays["fall_mask"][metric].sum()),
        "offline_reference_root_reward_sum": float(formula_reward.sum()),
        "clock_phase_world_command_prefix_checks_passed": True}
