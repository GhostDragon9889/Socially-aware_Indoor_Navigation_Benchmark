"""Differentiable literal reward equations from NavIsaacLab IV-B, equations3–6.

The published direction equation rewards negative forward velocity, contrary
to the prose. Its sign is preserved in ``paper_equation``. An explicitly chosen
``backwards_penalty`` is a reconstruction experiment, not a silent correction.
Signed torque*velocity is likewise preserved, including positive braking reward.
"""

from __future__ import annotations

import torch
from torch import Tensor


def aligned_position_error_squared(position_xy: Tensor, target_xy: Tensor) -> Tensor:
    """Sum squared XY differences for targets explicitly aligned to this time."""
    if position_xy.shape != target_xy.shape or position_xy.shape[-1] != 2:
        raise ValueError("Aligned position and target must have the same (...,2) shape")
    return (position_xy - target_xy).square().sum(dim=-1)


def paper_task_reward(delta: Tensor, torques: Tensor, joint_velocities: Tensor,
                      forward_velocity: Tensor, *, direction_mode="paper_equation",
                      signed_power_w: Tensor | None = None) -> dict[str, Tensor]:
    """Return raw task components, without AMP or the outer0.5 task weight.

    Torque and angular speed must share the same 69DOF order and timestamp.
    ``signed_power_w`` can supply a time average of synchronized physics-substep
    samples. The caller records whether effort is measured or a PD proxy;
    normalized policy actions and PD target angles are not physical effort.
    """
    if torques.shape != joint_velocities.shape or torques.shape[-1] != 69:
        raise ValueError("SMPL torque and angular velocity must share (...,69) shape")
    batch_shape = torques.shape[:-1]
    if delta.shape != batch_shape or forward_velocity.shape != batch_shape:
        raise ValueError("Delta/forward velocity shape must match torque batch dimensions")
    if direction_mode not in {"paper_equation", "backwards_penalty"}:
        raise ValueError("direction_mode must be paper_equation or backwards_penalty")
    if any(not torch.isfinite(value).all() for value in (delta, torques, joint_velocities, forward_velocity)):
        raise ValueError("Reward inputs must be finite")
    if torch.any(delta < 0):
        raise ValueError("Squared trajectory error delta must be nonnegative")
    trajectory = torch.exp(-2.0 * delta)
    if signed_power_w is not None:
        if signed_power_w.shape != batch_shape or not torch.isfinite(signed_power_w).all():
            raise ValueError("Measured signed power must be finite and match torque batch dimensions")
        signed_power = signed_power_w
    else:
        signed_power = (torques * joint_velocities).sum(dim=-1)
    energy = -.0005 * signed_power
    direction = -torch.expm1(torch.minimum(forward_velocity, torch.zeros_like(forward_velocity)) / 1.8)
    if direction_mode == "backwards_penalty":
        direction = -direction
    return {"trajectory": trajectory, "energy": energy, "direction": direction,
            "signed_power_w": signed_power, "task": trajectory + energy + direction}


def paper_total_reward(amp_reward: Tensor, components: dict[str, Tensor], *,
                       amp_weight: float = 1., task_weight: float = .5) -> Tensor:
    """Published weights by default; adaptations must pass explicit weights."""
    if amp_reward.shape != components["task"].shape or not torch.isfinite(amp_reward).all():
        raise ValueError("AMP reward must be finite and match the task reward shape")
    import math
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (amp_weight, task_weight)) or amp_weight + task_weight <= 0:
        raise ValueError("Reward weights must be finite, nonnegative and nonzero in total")
    return amp_weight * amp_reward + task_weight * components["task"]


def align_trajectory_xy(trajectory: Tensor, sample_times: Tensor, query_times: Tensor) -> Tensor:
    """Interpolate each environment's time-stamped plan without silent clamping.

    Times are seconds in one declared frame (absolute or relative); this helper
    does not guess whether a plan originated at the latest control/reset step.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2 or sample_times.shape != trajectory.shape[:2]:
        raise ValueError("Trajectory/sample_times must have shapes (B,F,2)/(B,F)")
    if trajectory.shape[1] < 2 or query_times.shape != trajectory.shape[:1]:
        raise ValueError("At least two samples and one query time per environment are required")
    if any(not torch.isfinite(value).all() for value in (trajectory, sample_times, query_times)):
        raise ValueError("Trajectory positions and times must be finite")
    if torch.any(sample_times[:, 1:] <= sample_times[:, :-1]):
        raise ValueError("Trajectory sample times must be strictly increasing")
    if torch.any(query_times < sample_times[:, 0]) or torch.any(query_times > sample_times[:, -1]):
        raise ValueError("Query outside the current trajectory horizon: replan explicitly")
    upper = torch.searchsorted(sample_times.contiguous(), query_times[:, None].contiguous(), right=True).squeeze(-1)
    upper = upper.clamp(1, trajectory.shape[1] - 1)
    lower = upper - 1
    rows = torch.arange(len(trajectory), device=trajectory.device)
    ratio = (query_times - sample_times[rows, lower]) / (sample_times[rows, upper] - sample_times[rows, lower])
    return torch.lerp(trajectory[rows, lower], trajectory[rows, upper], ratio[:, None])
