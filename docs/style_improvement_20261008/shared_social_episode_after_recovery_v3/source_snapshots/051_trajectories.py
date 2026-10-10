"""Independent, reproducible PACER-style low-level training commands.

TRACE/PACER were trained independently (Trace and Pace, section3.3). The
public PACER procedural generator changes scalar speed and heading at knots.
Its rare sharp turns do not bound vector acceleration. ``smooth`` explicitly
slews heading and limits vector acceleration; it is a local curriculum choice,
not an exact recreation of the public generator or an author checkpoint claim.
This module computes commands on CPU and never changes a physical robot state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class PacerTrajectoryConfig:
    horizon_s: float = 5.
    knot_dt_s: float = .1
    speed_min_mps: float = 0.
    speed_max_mps: float = 3.
    scalar_accel_max_mps2: float = 2.
    heading_rate_max_rad_s: float = 2.
    sharp_turn_probability: float = .02
    mode: str = "smooth"
    vector_accel_max_mps2: float = 2.
    bounds_xy: tuple[float, float, float, float] | None = None
    boundary_mode: str = "none"

    def validate(self):
        for name in ("horizon_s", "knot_dt_s", "speed_min_mps", "speed_max_mps", "scalar_accel_max_mps2",
                     "heading_rate_max_rad_s", "sharp_turn_probability", "vector_accel_max_mps2"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, not a boolean")
        if self.horizon_s <= 0 or self.knot_dt_s <= 0 or not math.isclose(self.horizon_s / self.knot_dt_s, round(self.horizon_s / self.knot_dt_s), abs_tol=1e-9):
            raise ValueError("Positive horizon must contain an integer number of knot intervals")
        if not 0 <= self.speed_min_mps <= self.speed_max_mps or min(self.scalar_accel_max_mps2, self.heading_rate_max_rad_s, self.vector_accel_max_mps2) <= 0:
            raise ValueError("Speed range and positive acceleration/heading limits are invalid")
        if not 0 <= self.sharp_turn_probability <= 1 or self.mode not in {"smooth", "pacer_knots"}:
            raise ValueError("Unknown curriculum mode or invalid sharp-turn probability")
        if self.boundary_mode not in {"none", "reflection"}:
            raise ValueError("Boundary mode must explicitly select none or reflection")
        if self.bounds_xy is not None:
            if len(self.bounds_xy) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.bounds_xy):
                raise ValueError("bounds_xy requires four finite numbers")
            if self.bounds_xy[0] >= self.bounds_xy[2] or self.bounds_xy[1] >= self.bounds_xy[3]:
                raise ValueError("bounds_xy requires xmin,ymin,xmax,ymax")
        if self.boundary_mode == "reflection" and self.bounds_xy is None:
            raise ValueError("Explicit reflection requires bounds_xy")


@dataclass
class CurriculumPlan:
    xy: Tensor
    velocities_xy: Tensor
    times_s: Tensor
    sharp_turn_mask: Tensor
    boundary_reflection_mask: Tensor
    initial_velocity_clipped: Tensor
    speed_goals_mps: Tensor
    metadata: dict


def _pacer_random_knots(batch, segments, config, generator):
    """Public PACER reset's draw order and scalar/heading recurrence, on CPU.

    Formula parity uses the fixed official source's normal procedural branch;
    no network, legacy SDK, real_path server or fixed_path flags are imported.
    """
    dt = config.knot_dt_s
    angles = (2 * torch.rand(batch, segments, generator=generator) - 1) * config.heading_rate_max_rad_s * dt
    sharp_angles = math.pi * (2 * torch.rand(batch, segments, generator=generator) - 1)
    sharp = torch.bernoulli(torch.full_like(angles, config.sharp_turn_probability), generator=generator) == 1
    angles[sharp] = sharp_angles[sharp]
    angles[:, 0] = math.pi * (2 * torch.rand(batch, generator=generator) - 1)
    increments = (2 * torch.rand(batch, segments, generator=generator) - 1) * config.scalar_accel_max_mps2 * dt
    increments[:, 0] = (config.speed_max_mps - config.speed_min_mps) * torch.rand(batch, generator=generator) + config.speed_min_mps
    speed = torch.zeros_like(increments)
    speed[:, 0] = increments[:, 0]
    for index in range(1, segments):
        speed[:, index] = (speed[:, index - 1] + increments[:, index]).clamp(config.speed_min_mps, config.speed_max_mps)
    heading = angles.cumsum(-1)
    velocity = speed[..., None] * torch.stack((heading.cos(), -heading.sin()), -1)
    return velocity, sharp


class PacerTrajectoryCurriculum:
    """Private CPU RNG; save ``state_dict`` alongside plans/physical resume state.

    ``build_plan(state, actor_ids=None)`` accepts a common-order RobotState with
    root_pos[B,3], root_vel[B,3], root_rot[B,4] in world meters / XYZW. It returns
    absolute world-XY commands at t=0..5s (51 knots by default), beginning exactly
    at the selected actual root. Returning commands never teleports that root.
    """

    def __init__(self, config: PacerTrajectoryConfig | None = None, *, seed=7):
        self.config = config or PacerTrajectoryConfig()
        self.config.validate()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Curriculum seed must be an integer in [0,2**63)")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.plans_generated = 0

    def describe(self):
        return {"source": "independent_procedural_pacer_style_curriculum", "config": asdict(self.config),
                "reference_revision": "baeb6241f03fc9d78370a64e3c2dcef9d830d97f",
                "reference_source": "pacer/env/util/traj_generator.py:reset; humanoid_traj.py:_build_traj_generator",
                "primary_paper": "https://arxiv.org/html/2304.01893v1#S3.SS3",
                "command_coordinates": "absolute world XY meters; explicit seconds from current physical root",
                "rng": "private CPU torch.Generator; persisted state required on resume",
                "training_deployment_separation": "optional low-level training source; diffusion remains a separate deployment planner",
                "author_experiment_equivalence": False,
                "scalar_acceleration_semantics": "public accelMax limits speed changes, not vector acceleration",
                "smooth_speed_target": "independent uniform [speed_min,speed_max] per plan, held for the horizon; actual speed initializes a bounded ramp only",
                "vector_acceleration_bound_mps2": self.config.vector_accel_max_mps2 if self.config.mode == "smooth" and self.config.boundary_mode == "none" else None,
                "boundary_semantics": "none: no floor guarantee; caller needs horizon*speed_max floor margin" if self.config.boundary_mode == "none" else "explicit mirrored command extension; reflections may violate acceleration limits; no physical teleport"}

    @torch.no_grad()
    def build_plan(self, robot_state, actor_ids=None):
        fields = [getattr(robot_state, name) for name in ("root_pos", "root_vel", "root_rot")]
        batch = fields[0].shape[0] if isinstance(fields[0], Tensor) and fields[0].ndim == 2 else -1
        for value, width in zip(fields, (3, 3, 4), strict=True):
            if not isinstance(value, Tensor) or value.shape != (batch, width) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Current physical state requires finite floating root_pos/root_vel/root_rot [B,3/3/4]")
        if batch < 1 or any(value.device != fields[0].device for value in fields):
            raise ValueError("Physical state requires a nonempty consistent device batch")
        if actor_ids is not None:
            ids = torch.as_tensor(actor_ids, device="cpu")
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not len(ids) or ((ids < 0) | (ids >= batch)).any() or len(ids.unique()) != len(ids):
                raise ValueError("actor_ids requires unique in-range nonempty integer indices")
            fields = [value[ids.to(value.device)] for value in fields]
        position, velocity, quaternion = [value.detach().cpu().float().clone() for value in fields]
        if not torch.allclose(quaternion.norm(dim=-1), torch.ones(len(quaternion)), atol=1e-4, rtol=0):
            raise ValueError("Physical root quaternion must be unit XYZW")
        config = self.config
        if config.boundary_mode == "reflection":
            low = torch.tensor(config.bounds_xy[:2])
            high = torch.tensor(config.bounds_xy[2:])
            if ((position[:, :2] < low) | (position[:, :2] > high)).any():
                raise ValueError("An actual root outside reflection bounds cannot be recovered by a command/teleport")
        segments = round(config.horizon_s / config.knot_dt_s)
        batch = len(position)
        initial_velocity = velocity[:, :2]
        initial_speed = initial_velocity.norm(dim=-1)
        initial_clipped = initial_speed > config.speed_max_mps
        initial_velocity *= (config.speed_max_mps / initial_speed.clamp_min(1e-8)).clamp_max(1)[:, None]
        if config.mode == "pacer_knots":
            future_velocity, sharp = _pacer_random_knots(batch, segments, config, self.generator)
            speed_goal = future_velocity[:, 0].norm(dim=-1)
        else:
            dt = config.knot_dt_s
            yaw = torch.atan2(2 * (quaternion[:, 3] * quaternion[:, 2] + quaternion[:, 0] * quaternion[:, 1]),
                              1 - 2 * (quaternion[:, 1].square() + quaternion[:, 2].square()))
            heading = torch.where(initial_speed > 1e-4, torch.atan2(initial_velocity[:, 1], initial_velocity[:, 0]), yaw)
            heading_goal = heading.clone()
            speed = initial_velocity.norm(dim=-1).clamp(config.speed_min_mps, config.speed_max_mps)
            speed_goal = config.speed_min_mps + (config.speed_max_mps - config.speed_min_mps) * torch.rand(batch, generator=self.generator)
            previous = initial_velocity.clone()
            future_velocity = torch.empty(batch, segments, 2)
            sharp = torch.zeros(batch, segments, dtype=torch.bool)
            for index in range(segments):
                speed += (speed_goal - speed).clamp(-config.scalar_accel_max_mps2 * dt, config.scalar_accel_max_mps2 * dt)
                turn = (2 * torch.rand(batch, generator=self.generator) - 1) * config.heading_rate_max_rad_s * dt
                sharp[:, index] = torch.rand(batch, generator=self.generator) < config.sharp_turn_probability
                sharp_angle = (2 * torch.rand(batch, generator=self.generator) - 1) * math.pi
                heading_goal += torch.where(sharp[:, index], sharp_angle, turn)
                difference = torch.atan2((heading_goal - heading).sin(), (heading_goal - heading).cos())
                heading += difference.clamp(-config.heading_rate_max_rad_s * dt, config.heading_rate_max_rad_s * dt)
                requested = speed[:, None] * torch.stack((heading.cos(), heading.sin()), -1)
                change = requested - previous
                change *= (config.vector_accel_max_mps2 * dt / change.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)[:, None]
                previous += change
                future_velocity[:, index] = previous
        xy = torch.empty(batch, segments + 1, 2)
        xy[:, 0] = position[:, :2]
        reflections = torch.zeros(batch, segments, dtype=torch.bool)
        if config.boundary_mode == "none":
            xy[:, 1:] = xy[:, :1] + (future_velocity * config.knot_dt_s).cumsum(1)
        else:
            width = high - low
            raw_position = position[:, :2].clone()
            cell = torch.floor((raw_position - low) / width).long()
            for index in range(segments):
                raw_position += future_velocity[:, index] * config.knot_dt_s
                next_cell = torch.floor((raw_position - low) / width).long()
                reflections[:, index] = (next_cell != cell).any(-1)
                cell = next_cell
                phase = torch.remainder(raw_position - low, 2 * width)
                xy[:, index + 1] = low + torch.minimum(phase, 2 * width - phase)
                future_velocity[:, index] *= torch.where(torch.remainder(cell, 2) == 0, 1., -1.)
        self.plans_generated += batch
        destination = fields[0].device
        return CurriculumPlan(xy=xy.to(destination), velocities_xy=torch.cat((initial_velocity[:, None], future_velocity), 1).to(destination),
                              times_s=(torch.arange(segments + 1) * config.knot_dt_s).to(destination),
                              sharp_turn_mask=sharp.to(destination), boundary_reflection_mask=reflections.to(destination),
                              initial_velocity_clipped=initial_clipped.to(destination), speed_goals_mps=speed_goal.to(destination), metadata=self.describe())

    def state_dict(self):
        return {"schema": "pacer_trajectory_curriculum_v1", "config": asdict(self.config),
                "generator_state": self.generator.get_state().clone(), "plans_generated": self.plans_generated}

    def load_state_dict(self, state):
        expected = {"schema", "config", "generator_state", "plans_generated"}
        if not isinstance(state, dict) or set(state) != expected or state["schema"] != "pacer_trajectory_curriculum_v1" or state["config"] != asdict(self.config):
            raise ValueError("Curriculum resume requires the exact schema/config/keys")
        rng, count = state["generator_state"], state["plans_generated"]
        if not isinstance(rng, Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8 or rng.shape != self.generator.get_state().shape or type(count) is not int or count < 0:
            raise ValueError("Curriculum resume requires valid CPU RNG state and nonnegative count")
        validation_generator = torch.Generator(device="cpu")
        validation_generator.set_state(rng)
        self.generator.set_state(rng)
        self.plans_generated = count
