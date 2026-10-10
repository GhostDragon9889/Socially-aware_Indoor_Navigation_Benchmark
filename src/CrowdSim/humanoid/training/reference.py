"""Optional causal execution reference for learned Diffusion XY plans.

This is a deployment adaptation, not a NavIsaacLab paper definition. It limits
the reference velocity/acceleration, not actual humanoid motion. There is no
boundary, obstacle, peer-contact, route-arrival or controller-stability guarantee.
The caller owns reference state, resets and checkpointing; no hidden state/RNG
or simulation operation is held here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral

import torch


@dataclass(frozen=True)
class ReferenceAdapterConfig:
    max_speed_mps: float = 3.
    max_acceleration_mps2: float = 2.
    control_dt_s: float = 1 / 30
    raw_knot_dt_s: float = .1
    horizon_s: float = 5.
    tracking_time_s: float = .25
    initial_speed_policy: str = "reject"
    speed_tolerance_mps: float = 1e-6

    def validate(self):
        for key in ("max_speed_mps", "max_acceleration_mps2", "control_dt_s",
                    "raw_knot_dt_s", "horizon_s", "tracking_time_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        tolerance = self.speed_tolerance_mps
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("speed_tolerance_mps must be nonnegative and finite")
        if self.initial_speed_policy not in {"reject", "brake"}:
            raise ValueError("initial_speed_policy must explicitly select reject or brake")
        for interval in (self.control_dt_s, self.raw_knot_dt_s):
            count = round(self.horizon_s / interval)
            if count < 1 or not math.isclose(count * interval, self.horizon_s, abs_tol=1e-8, rel_tol=0):
                raise ValueError("The horizon must contain an integer number of control/raw intervals")


@dataclass(frozen=True)
class ReferenceState:
    xy: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True)
class DenseReferencePlan:
    raw_xy: torch.Tensor
    raw_times_s: torch.Tensor
    xy: torch.Tensor
    velocity: torch.Tensor
    times_s: torch.Tensor
    raw_sampled_xy: torch.Tensor
    initial_overspeed: torch.Tensor
    speed_violation: torch.Tensor
    deviation_m: torch.Tensor

    def state_at_step(self, steps):
        """Return caller-owned copies of the already executed reference state.

        ``steps`` is one integer for all actors or one integer per actor. Never
        re-anchor a normal replan to physical root position: use this state.
        A real episode reset must explicitly initialize from its physical state.
        """
        if isinstance(steps, Integral) and not isinstance(steps, bool):
            if not 0 <= steps < self.xy.shape[1]:
                raise ValueError("Reference step is outside the saved five-second plan")
            return ReferenceState(self.xy[:, steps].clone(), self.velocity[:, steps].clone())
        if not isinstance(steps, torch.Tensor) or steps.shape != (len(self.xy),) or steps.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise TypeError("Reference steps must be an integer or a per-actor integer Tensor")
        if torch.any(steps < 0) or torch.any(steps >= self.xy.shape[1]):
            raise ValueError("Reference step is outside the saved five-second plan")
        steps = steps.to(device=self.xy.device, dtype=torch.long)
        actors = torch.arange(len(self.xy), device=self.xy.device)
        return ReferenceState(self.xy[actors, steps].clone(), self.velocity[actors, steps].clone())


def _norm_limit(vector, maximum):
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = (maximum / norm.clamp_min(torch.finfo(vector.dtype).tiny)).clamp(max=1)
    return vector * scale


class CausalReferenceAdapter:
    """Build a dense reference while preserving each original learned plan.

    For each actual control interval, the desired velocity is the model's
    interval velocity plus current model/reference position error divided by
    ``tracking_time_s``. Limit its vector norm, then limit the velocity change
    by ``max_acceleration_mps2 * control_dt_s`` and integrate reference position.
    Position error uses the current time, avoiding a feed-forward one-step lead.

    If v0 is within the speed ball, the update is a convex combination of two
    velocities in that ball. If v0 is outside, instantaneous speed limiting
    conflicts with acceleration limiting: reject by default, or explicitly
    brake with the original v0 preserved and every speed violation reported.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else ReferenceAdapterConfig()
        if not isinstance(self.config, ReferenceAdapterConfig):
            raise TypeError("Reference adapter requires ReferenceAdapterConfig")
        self.config.validate()

    def contract(self):
        return {"method": "causal velocity-slew-limited learned-XY reference",
                "configuration": asdict(self.config), "deployment_adaptation": True,
                "paper_definition": False, "raw_model_plan_preserved": True,
                "hidden_state": False, "initialization": "actual root XY/velocity only at episode initialization/reset; previous executed reference state at ordinary replan",
                "integration": "v_next = v + norm_clip(v_desired-v, a_max*dt); xy_next = xy + v_next*dt",
                "limitations": ["limits reference, not physical speed or acceleration",
                                "may lag the learned plan or cut through obstacles",
                                "no boundary, contact, route-arrival or stability guarantee",
                                "do not resample to sparse knots and claim dense acceleration bounds"],
                "numerics": "floating-point arithmetic; initial-speed classification uses the explicit m/s tolerance"}

    def describe(self):
        """JSON-safe metadata for the caller's report/checkpoint provenance."""
        return self.contract()

    @staticmethod
    def _check_tensor(value, shape, name, *, like=None):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must be a Tensor with shape {tuple(shape)}")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use float32 or float64")
        if like is not None and (value.dtype != like.dtype or value.device != like.device):
            raise ValueError(f"{name} must share the plan's dtype and device")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains nonfinite values")

    def _times(self, times, *, count, interval, like, name):
        expected = torch.arange(count, device=like.device, dtype=like.dtype) * interval
        if times is None:
            return expected
        self._check_tensor(times, (count,), name, like=like)
        if not torch.allclose(times, expected, atol=1e-6, rtol=0):
            raise ValueError(f"{name} must declare the configured uniform actual control/raw grid")
        return times.clone()

    def _check_initial_velocity(self, velocity):
        overspeed = torch.linalg.vector_norm(velocity, dim=-1) > self.config.max_speed_mps + self.config.speed_tolerance_mps
        if self.config.initial_speed_policy == "reject" and torch.any(overspeed):
            actors = overspeed.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Initial reference velocity exceeds speed limit for actors {actors}; explicit brake preserves v0 and reports overspeed")
        return overspeed

    def _step(self, xy, velocity, raw_position, raw_velocity):
        desired = raw_velocity + (raw_position - xy) / self.config.tracking_time_s
        desired = _norm_limit(desired, self.config.max_speed_mps)
        delta = _norm_limit(desired - velocity, self.config.max_acceleration_mps2 * self.config.control_dt_s)
        velocity = velocity + delta
        return ReferenceState(xy + velocity * self.config.control_dt_s, velocity)

    def step(self, reference_xy, reference_velocity, raw_position, raw_velocity):
        """Advance one interval; input raw position is its current-time sample.

        The future raw sample belongs to the already generated model plan, never
        to a future physical state. Caller state and all raw inputs are unchanged.
        """
        if not isinstance(reference_xy, torch.Tensor) or reference_xy.ndim != 2 or reference_xy.shape[1:] != (2,):
            raise ValueError("reference_xy must have shape (B,2)")
        shape = reference_xy.shape
        for value, name in ((reference_xy, "reference_xy"), (reference_velocity, "reference_velocity"),
                            (raw_position, "raw_position"), (raw_velocity, "raw_velocity")):
            self._check_tensor(value, shape, name, like=reference_xy)
        self._check_initial_velocity(reference_velocity)
        return self._step(reference_xy, reference_velocity, raw_position, raw_velocity)

    def build(self, raw_xy, reference_xy, reference_velocity, *, times_s=None, raw_times_s=None):
        """Return 151 samples at30Hz for a default51-knot five-second plan.

        ``reference_xy/velocity`` are measured only at the first/reset frame.
        Ordinary replans must pass the last executed reference state. ``raw_xy``
        stays exactly available alongside its dense interpolation and adaptation
        error. Defaults are meters, seconds, world XY and vector-norm limits.
        """
        if not isinstance(raw_xy, torch.Tensor) or raw_xy.ndim != 3 or raw_xy.shape[-1] != 2 or not len(raw_xy):
            raise ValueError("raw_xy must be a nonempty Tensor with shape (B,raw_knots,2)")
        raw_count = round(self.config.horizon_s / self.config.raw_knot_dt_s) + 1
        count = round(self.config.horizon_s / self.config.control_dt_s) + 1
        self._check_tensor(raw_xy, (len(raw_xy), raw_count, 2), "raw_xy")
        self._check_tensor(reference_xy, (len(raw_xy), 2), "reference_xy", like=raw_xy)
        self._check_tensor(reference_velocity, (len(raw_xy), 2), "reference_velocity", like=raw_xy)
        initial_overspeed = self._check_initial_velocity(reference_velocity)
        times = self._times(times_s, count=count, interval=self.config.control_dt_s, like=raw_xy, name="times_s")
        raw_times = self._times(raw_times_s, count=raw_count, interval=self.config.raw_knot_dt_s, like=raw_xy, name="raw_times_s")
        coordinate = times / self.config.raw_knot_dt_s
        lower = coordinate.floor().long().clamp(0, raw_count - 1)
        upper = (lower + 1).clamp(max=raw_count - 1)
        alpha = (coordinate - lower).clamp(0, 1)
        sampled = raw_xy[:, lower] * (1 - alpha[None, :, None]) + raw_xy[:, upper] * alpha[None, :, None]
        positions, velocities = [reference_xy.clone()], [reference_velocity.clone()]
        for index in range(count - 1):
            raw_velocity = (sampled[:, index + 1] - sampled[:, index]) / self.config.control_dt_s
            state = self._step(positions[-1], velocities[-1], sampled[:, index], raw_velocity)
            positions.append(state.xy)
            velocities.append(state.velocity)
        xy = torch.stack(positions, dim=1)
        velocity = torch.stack(velocities, dim=1)
        speed_violation = torch.linalg.vector_norm(velocity, dim=-1) > self.config.max_speed_mps + self.config.speed_tolerance_mps
        return DenseReferencePlan(raw_xy.clone(), raw_times, xy, velocity, times, sampled,
                                  initial_overspeed, speed_violation,
                                  torch.linalg.vector_norm(xy - sampled, dim=-1))


__all__ = ["ReferenceAdapterConfig", "ReferenceState", "DenseReferencePlan", "CausalReferenceAdapter"]
