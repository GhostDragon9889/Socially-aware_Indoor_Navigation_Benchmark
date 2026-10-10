"""CPU contracts for full-body AMASS tracking, separate from XY navigation.

World-space body-origin errors use one initial translation and a fixed mocap
phase. A trajectory-conditioned AMP actor does not receive reference 3D poses.
These measurements alone do not reproduce the unpublished author evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

SMPL_BODY_NAMES = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee",
    "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)


@dataclass(frozen=True)
class TrackingWindow:
    initial_frame: int
    last_frame: int
    available_steps: int
    requested_steps: int
    minimum_steps: int
    eligible: bool
    full_clip_requested: bool

    def as_dict(self):
        return asdict(self)


def tracking_window(frames, dt_s, *, history_frames=10, minimum_seconds=1., max_steps=None):
    """Reserve a real preceding history; no padded history earns success credit."""
    if type(frames) is not int or frames < 1 or type(history_frames) is not int or history_frames < 1:
        raise ValueError("Frame/history counts must be positive integers")
    if type(dt_s) not in (int, float) or not math.isfinite(dt_s) or not math.isclose(dt_s, 1/30, abs_tol=1e-7):
        raise ValueError("Tracking requires an explicitly resampled 30Hz reference")
    if type(minimum_seconds) not in (int, float) or not math.isfinite(minimum_seconds) or minimum_seconds <= 0:
        raise ValueError("Minimum evaluation duration must be positive and finite")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Optional step cap must be a positive integer")
    initial = history_frames-1
    available = max(0, frames-1-initial)
    minimum = math.ceil(minimum_seconds*30-1e-9)
    requested = available if max_steps is None else min(available, max_steps)
    return TrackingWindow(initial, frames-1, available, requested, minimum,
                          available >= minimum, requested == available)


def future_reference_frames(frame, frames, *, waypoint_seconds=tuple(.5*i for i in range(1, 11))):
    """Ten actual reference XY samples at +.5..5s; hold clip end for commands."""
    if type(frame) is not int or type(frames) is not int or not 0 <= frame < frames:
        raise ValueError("Reference query phase must be inside its clip")
    seconds = np.asarray(waypoint_seconds, dtype=np.float64)
    if seconds.shape != (10,) or not np.isfinite(seconds).all() or (seconds <= 0).any() or (np.diff(seconds) <= 0).any():
        raise ValueError("Ten increasing positive waypoint times are required")
    raw = seconds*30
    if not np.allclose(raw, np.rint(raw), atol=1e-8, rtol=0):
        raise ValueError("Waypoint times must lie on the reference 30Hz grid")
    raw_frames = frame+np.rint(raw).astype(np.int64)
    return np.minimum(raw_frames, frames-1), raw_frames > frames-1


def _finite_float(value, name):
    result = np.asarray(value)
    if result.dtype.kind != "f" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return result.astype(np.float64)


def align_reference_positions(positions, reference_initial_root, physical_initial_root):
    """Apply ONE constant XYZ translation, preserving motion and orientation."""
    positions = _finite_float(positions, "reference positions")
    source = _finite_float(reference_initial_root, "reference initial root")
    target = _finite_float(physical_initial_root, "physical initial root")
    if positions.shape[-1:] != (3,) or source.shape != (3,) or target.shape != (3,):
        raise ValueError("Alignment requires XYZ positions and two [3] roots")
    return positions+(target-source)


def body_tracking_errors(actual_pos, actual_quat_xyzw, reference_pos, reference_quat_xyzw, *,
                         actual_body_names=SMPL_BODY_NAMES, reference_body_names=SMPL_BODY_NAMES):
    """Per-frame mean of 24 Euclidean errors and SO(3) geodesic radians.

    Quaternions have antipodal invariance; frames are never pelvis/yaw aligned.
    All input shapes are [T,24,3 or 4] and output body axes are canonical SMPL.
    """
    values = [_finite_float(value, name) for value, name in (
        (actual_pos, "actual positions"), (actual_quat_xyzw, "actual rotations"),
        (reference_pos, "reference positions"), (reference_quat_xyzw, "reference rotations"))]
    pos, rot, target, target_rot = values
    if pos.ndim != 3 or pos.shape[0] < 1 or pos.shape[1:] != (24, 3) or target.shape != pos.shape:
        raise ValueError("Position captures require matching nonempty [T,24,3] shapes")
    if rot.shape != (*pos.shape[:2], 4) or target_rot.shape != rot.shape:
        raise ValueError("Rotation captures require matching [T,24,4] shapes")
    indices = []
    for names in (actual_body_names, reference_body_names):
        names = tuple(names)
        if len(names) != 24 or len(set(names)) != 24 or set(names) != set(SMPL_BODY_NAMES):
            raise ValueError("Body names require an exact unique 24-body SMPL bijection")
        indices.append([names.index(name) for name in SMPL_BODY_NAMES])
    pos, rot = pos[:, indices[0]], rot[:, indices[0]]
    target, target_rot = target[:, indices[1]], target_rot[:, indices[1]]
    norms = [np.linalg.norm(q, axis=-1, keepdims=True) for q in (rot, target_rot)]
    if not all(np.allclose(norm, 1., atol=1e-3, rtol=0) for norm in norms):
        raise ValueError("Body rotations must be unit XYZW quaternions")
    position = np.linalg.norm(pos-target, axis=-1)
    dot = np.abs(np.sum((rot/norms[0])*(target_rot/norms[1]), axis=-1))
    rotation = 2*np.arccos(np.clip(dot, 0., 1.))
    return {"body_position_error_m": position, "body_rotation_error_rad": rotation,
            "mean_body_position_error_m": position.mean(-1),
            "mean_body_rotation_error_rad": rotation.mean(-1)}


def summarize_tracking_sequence(errors, *, metric_mask, fall_mask, expected_steps,
                                eligible=True, full_clip_requested=True, threshold_m=.5):
    """Latch strict per-step mean-position failure; partial clips get no success."""
    position = _finite_float(errors["mean_body_position_error_m"], "position errors")
    rotation = _finite_float(errors["mean_body_rotation_error_rad"], "rotation errors")
    metric, falls = np.asarray(metric_mask), np.asarray(fall_mask)
    if position.ndim != 1 or rotation.shape != position.shape or not len(position):
        raise ValueError("Mean tracking errors must be matching nonempty frame vectors")
    if metric.shape != position.shape or falls.shape != position.shape or metric.dtype != np.bool_ or falls.dtype != np.bool_:
        raise ValueError("Metric/fall masks must be explicit matching boolean vectors")
    if type(expected_steps) is not int or expected_steps < 1 or metric.sum() > expected_steps:
        raise ValueError("Expected legal metric steps must be positive and not exceeded")
    if type(eligible) is not bool or type(full_clip_requested) is not bool:
        raise ValueError("Eligibility and complete-clip request must be boolean")
    if type(threshold_m) not in (int, float) or not math.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("Position failure threshold must be positive and finite")
    failures = np.flatnonzero(metric & (position > threshold_m))
    complete = bool(metric.sum() == expected_steps and full_clip_requested)
    in_denominator = eligible and complete
    measured = position[metric]
    measured_rotation = rotation[metric]
    fall_frames = falls & metric
    onsets = fall_frames & ~np.concatenate(([False], fall_frames[:-1]))
    return {"expected_control_steps": expected_steps, "observed_metric_steps": int(metric.sum()),
            "complete_clip": complete, "eligible": eligible,
            "in_full_sequence_success_denominator": in_denominator,
            "tracking_failed": bool(len(failures)),
            "first_tracking_failure_capture_frame": int(failures[0]) if len(failures) else None,
            "tracking_success": bool(not len(failures)) if in_denominator else None,
            "sequence_had_fall": bool(fall_frames.any()), "fall_onset_events": int(onsets.sum()),
            "fall_capture_frames": np.flatnonzero(fall_frames).tolist(),
            "mean_body_position_error_m": float(measured.mean()) if len(measured) else None,
            "maximum_step_mean_body_position_error_m": float(measured.max()) if len(measured) else None,
            "mean_body_rotation_error_rad": float(measured_rotation.mean()) if len(measured) else None,
            "failure_rule": f"any measured step: mean of 24 world body-origin Euclidean errors > {threshold_m} m",
            "fall_rule_separate_from_tracking_failure": True}
