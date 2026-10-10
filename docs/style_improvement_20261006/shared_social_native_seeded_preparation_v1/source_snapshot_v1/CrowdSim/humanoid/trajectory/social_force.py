"""Pure CPU public-method SFM reconstruction for a future shared AMP comparison.

Force equations follow public CrowdSim/world/sfm.py, SHA below. Joint 5-second
prediction is an explicit local extension of that one-step controller. There is
no SDK, learned controller, physical rollout, native map admission or RNG here.
The caller applies the existing causal AMP reference adapter to BOTH planners.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np


PUBLIC_SFM_SHA256 = "88f3274e1bfa2361dd92bb827b7b924d0d7c90d88f4a868ba76913bdfeccfd9c"
PUBLIC_SFM_URL = (
    "https://github.com/yh83305/NavIsaaclab2.0/blob/"
    "d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/world/sfm.py"
)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _array_sha(value):
    array = np.asarray(value)
    return _sha(str((array.dtype.str, array.shape)).encode() + array.tobytes(order="C"))


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _xy(value, name, count=None):
    if not isinstance(value, np.ndarray) or value.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name} must be a CPU NumPy float32/float64 array")
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or (count is not None and len(value) != count):
        raise ValueError(f"{name} must have matched nonempty shape [N,2]")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} contains nonfinite values")
    return value.astype(np.float64, copy=True)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceBinding:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def bind_sfm(cls, path):
        source = Path(path).resolve()
        data = source.read_bytes()
        if _sha(data) != PUBLIC_SFM_SHA256:
            raise ValueError("SFM source does not match the exact verified public implementation")
        return cls(str(source), len(data), PUBLIC_SFM_SHA256)

    def verify(self):
        try:
            data = Path(self.path).read_bytes()
        except OSError as error:
            raise RuntimeError("Bound SFM source changed or became unavailable") from error
        if (len(data) != self.bytes or _sha(data) != self.sha256
                or self.sha256 != PUBLIC_SFM_SHA256):
            raise RuntimeError("Bound SFM source changed")


@dataclass(frozen=True)
class SFMParameters:
    """Explicit local parameters; defaults combine public class/env settings.

    FOV is disabled, matching Social_Force.fov=False. Every other actor within
    the strict neighbor radius participates; there is no top-k approximation.
    integration_dt_s is an actual integration interval, not an output interval.
    """
    integration_dt_s: float = 1 / 30
    max_speed_mps: float = 1.
    safe_distance_m: float = 1.2
    neighbor_radius_m: float = 4.
    reach_distance_m: float = .7
    attraction_gain: float = 5.
    static_gain: float = 15.
    static_decay_m: float = .25
    agent_gain: float = 25.
    agent_decay_m: float = .5
    ttc_gain: float = 28.
    prediction_horizon_s: float = 3.5
    ttc_threshold_s: float = 3.

    def validate(self):
        for name, value in asdict(self).items():
            _positive(value, name, zero=name in {"attraction_gain", "static_gain", "agent_gain", "ttc_gain"})
        steps = round(.1 / self.integration_dt_s)
        if steps < 1 or not math.isclose(steps * self.integration_dt_s, .1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("integration_dt_s must divide the exact .1s output interval")


class StaticDistanceGrid:
    """Source-bound metric distance field in ROS image/world coordinates.

    The caller supplies the real EDT in meters and labels 0/205/254 for
    occupied/unknown/free. No map, unknown cells or exterior is synthesized.
    Source name/SHA are caller provenance; numerical input hashes independently
    bind the exact arrays/frame. Private copies are immutable and the original
    arrays remain bound so modifying them faults the next query/generation.
    """
    def __init__(self, distance_m, labels, *, origin_xy, resolution_m, source_id, source_sha256):
        if not isinstance(distance_m, np.ndarray) or distance_m.dtype not in (np.float32, np.float64):
            raise TypeError("distance_m must be a CPU NumPy float32/float64 array")
        if (distance_m.ndim != 2 or min(distance_m.shape) < 2 or not isinstance(labels, np.ndarray)
                or labels.shape != distance_m.shape or labels.dtype.kind not in "iu"):
            raise ValueError("Distance/label grids require matched [H>=2,W>=2] arrays")
        if not np.isfinite(distance_m).all():
            raise FloatingPointError("Distance field contains nonfinite values")
        if (distance_m < 0).any() or not np.isin(labels, (0, 205, 254)).all():
            raise ValueError("Distance must be nonnegative; labels must be occupied0/unknown205/free254")
        if ((distance_m[labels != 254] != 0).any() or (distance_m[labels == 254] <= 0).any()):
            raise ValueError("Distance field must include occupied/unknown cells as zero-distance obstacles and positive known-free cells")
        _positive(resolution_m, "resolution_m")
        origin = np.asarray(origin_xy, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin_xy must be two finite lower-left world coordinates")
        if (not isinstance(source_id, str) or not source_id or not isinstance(source_sha256, str)
                or len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256)):
            raise ValueError("Grid requires explicit source identity and lowercase SHA256")
        self.distance_m, self.labels = distance_m, labels
        self.origin_xy, self.resolution_m = tuple(origin), float(resolution_m)
        self.source_id, self.source_sha256 = source_id, source_sha256
        self._distance, self._labels = _readonly(distance_m.astype(np.float64)), _readonly(labels)
        self._input_hashes = (_array_sha(distance_m), _array_sha(labels))
        self._frame_hash = _json_sha(self._frame())
        # Public compute_gradient lines39..54: image Y is opposite world Y.
        grad_y, grad_x = np.gradient(self._distance, self.resolution_m, self.resolution_m)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude[magnitude == 0] = 1e-6
        self._gradient = _readonly(np.stack((grad_x / magnitude, -grad_y / magnitude), axis=-1))

    def _frame(self):
        return {"origin_xy": self.origin_xy, "resolution_m": self.resolution_m,
                "source_id": self.source_id, "source_sha256": self.source_sha256,
                "layout": "row0=max_world_y,column0=min_world_x; cell centers; origin=lower-left edge"}

    def verify(self):
        if ((_array_sha(self.distance_m), _array_sha(self.labels)) != self._input_hashes
                or _json_sha(self._frame()) != self._frame_hash):
            raise RuntimeError("Bound grid input/frame changed")

    def describe(self):
        self.verify()
        return {**self._frame(), "array_sha256": list(self._input_hashes),
                "shape_hw": list(self._labels.shape), "physical_admission_established": False}

    def pixel_indices(self, xy):
        self.verify()
        return self._pixel_indices(xy)

    def _pixel_indices(self, xy):
        points = _xy(xy, "grid XY")
        height, width = self._labels.shape
        grid = (points - np.asarray(self.origin_xy)) / self.resolution_m
        if ((grid < 0).any() or (grid[:, 0] >= width).any() or (grid[:, 1] >= height).any()):
            raise ValueError("XY lies in map exterior; no clipping/fallback is allowed")
        columns, bottom_rows = np.floor(grid).astype(np.int64).T
        pixels = np.column_stack((height - 1 - bottom_rows, columns))
        labels = self._labels[pixels[:, 0], pixels[:, 1]]
        if (labels == 205).any():
            raise ValueError("XY lies in unknown map cells")
        if (labels != 254).any():
            raise ValueError("XY lies in occupied map cells")
        return pixels

    def sample(self, xy):
        pixels = self.pixel_indices(xy)
        return self._distance[pixels[:, 0], pixels[:, 1]], self._gradient[pixels[:, 0], pixels[:, 1]]

    def _sample(self, xy):
        pixels = self._pixel_indices(xy)
        return self._distance[pixels[:, 0], pixels[:, 1]], self._gradient[pixels[:, 0], pixels[:, 1]]

    def check_segments(self, before, after):
        self.verify()
        self._check_segments(before, after)

    def _check_segments(self, before, after):
        # Split at every crossed grid edge, then check every open interval and
        # boundary. Uniform distance samples can miss a short diagonal cell.
        self._pixel_indices(before)
        self._pixel_indices(after)
        for start, end in zip(before, after):
            grid_start = (start - np.asarray(self.origin_xy)) / self.resolution_m
            grid_end = (end - np.asarray(self.origin_xy)) / self.resolution_m
            fractions = [0., 1.]
            for axis in range(2):
                difference = grid_end[axis] - grid_start[axis]
                if difference == 0:
                    continue
                lower, upper = sorted((grid_start[axis], grid_end[axis]))
                edges = np.arange(math.floor(lower) + 1, math.ceil(upper), dtype=np.float64)
                crossings = (edges - grid_start[axis]) / difference
                fractions.extend(crossings[(crossings > 0) & (crossings < 1)].tolist())
            fractions = np.unique(fractions)
            samples = np.concatenate((fractions, (fractions[1:] + fractions[:-1]) / 2))
            self._pixel_indices(start[None] + samples[:, None] * (end - start))


@dataclass(frozen=True)
class ForceComponents:
    attraction: np.ndarray
    static: np.ndarray
    agent: np.ndarray
    ttc: np.ndarray


@dataclass(frozen=True)
class SFMState:
    xy: np.ndarray
    velocity: np.ndarray
    forces: ForceComponents


@dataclass(frozen=True)
class SFMTrajectory:
    raw_xy: np.ndarray
    raw_velocity: np.ndarray
    times_s: np.ndarray
    future_xy: np.ndarray
    future_times_s: np.ndarray
    internal_xy: np.ndarray
    metadata: dict

    def reference_adapter_kwargs(self):
        """Arguments for existing ReferenceAdapterConfig; no adapter execution.

        Pass raw_xy/times_s to the caller's existing CausalReferenceAdapter.
        Use its prior executed reference state on replan and physical state only
        at actual reset. Sparse ten targets alone do not establish acceleration
        bounds. The public SFM speed clamp is the only constraint applied here.
        """
        return {"max_speed_mps": 1.5, "max_acceleration_mps2": 2., "control_dt_s": 1 / 30,
                "raw_knot_dt_s": .1, "horizon_s": 5., "tracking_time_s": .25,
                "initial_speed_policy": "brake", "speed_tolerance_mps": 1e-6}


class SocialForceTrajectoryGenerator:
    """Joint semi-implicit rollout using immutable previous-step peer states.

    This extends a one-step public SFM to constant caller-provided goals for
    five seconds. Goals may be current registered A* waypoints; route updates
    and later actual replans belong to the caller, never to this predictor.
    Seed is explicitly recorded, with no random perturbations or RNG resets.
    """
    def __init__(self, grid, *, parameters, seed, sources):
        if not isinstance(grid, StaticDistanceGrid) or not isinstance(parameters, SFMParameters):
            raise TypeError("Require StaticDistanceGrid and SFMParameters")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an explicit integer in [0,2**32)")
        self.grid, self.parameters, self.seed = grid, parameters, int(seed)
        self.sources = tuple(sources)
        if not self.sources or not all(isinstance(source, SourceBinding) for source in self.sources):
            raise ValueError("Require at least one exact public SFM source binding")
        parameters.validate()
        self._parameter_hash = _json_sha(asdict(parameters))
        self._identity_hash = _json_sha({"seed": self.seed, "sources": [asdict(s) for s in self.sources]})
        self._grid = grid
        self.verify()

    def verify(self):
        if _json_sha(asdict(self.parameters)) != self._parameter_hash:
            raise RuntimeError("Bound SFM parameters changed")
        if _json_sha({"seed": self.seed, "sources": [asdict(s) for s in self.sources]}) != self._identity_hash:
            raise RuntimeError("Bound seed/source identities changed")
        if self.grid is not self._grid:
            raise RuntimeError("Bound grid object changed")
        self.grid.verify()
        for source in self.sources:
            source.verify()

    def describe(self):
        self.verify()
        return {"schema": "local.public_sfm.joint_future.v1", "mode": "public_method_reconstruction",
                "author_private_configuration": False, "source_url": PUBLIC_SFM_URL,
                "sources": [asdict(s) for s in self.sources], "parameters": asdict(self.parameters),
                "seed": self.seed, "rng_consumed": False, "grid": self.grid.describe(),
                "internal_dt_s": self.parameters.integration_dt_s, "output_hz": 10,
                "output_frames": 51, "horizon_s": 5., "future_offsets_s": (np.arange(1, 11) / 2).tolist(),
                "neighbor_update": "all rows read one previous-step snapshot; strict radius; no self/FOV/top-k",
                "integration": "v_next=public_sfm_speed_clip(v+dt*sum_forces); xy_next=xy+dt*v_next",
                "numerics": "float64 state/EDT; public float32 neighbor positions, closest-time and pass-right intermediates retained; no bitwise claim",
                "goal_protocol": "constant explicit per-actor goal over each predicted horizon",
                "reference_constraints_applied": False, "SDK_executed": False,
                "physical_or_eight_scene_validation_established": False}

    def _forces(self, positions, velocities, goals):
        p = self.parameters
        delta = goals - positions
        distance = np.linalg.norm(delta, axis=-1)
        desired = np.zeros_like(positions)
        active = distance > p.reach_distance_m
        desired[active] = p.max_speed_mps * delta[active] / distance[active, None]
        attraction = p.attraction_gain * (desired - velocities)
        static_distance, gradient = self.grid._sample(positions)
        static = np.zeros_like(positions)
        active = static_distance < p.neighbor_radius_m  # public avoid_dis == neighbor_radius
        static[active] = (p.static_gain * np.exp((.5 * p.safe_distance_m - static_distance[active])
                                               / p.static_decay_m)[:, None] * gradient[active])
        # [ego, neighbor, XY], with neighbor-minus-ego signs from nav_manager.
        relative = positions[None, :, :] - positions[:, None, :]
        relative_velocity = velocities[None, :, :] - velocities[:, None, :]
        distances = np.linalg.norm(relative, axis=-1)
        neighbors = (distances < p.neighbor_radius_m) & ~np.eye(len(positions), dtype=bool)
        # Preserve the public implementation's explicit intermediate casts.
        relative_positions = relative.astype(np.float32)
        safe = np.maximum(distances, 1e-6)
        agent_pair = (p.agent_gain * np.exp((p.safe_distance_m - safe) / p.agent_decay_m)[..., None]
                      * (-relative_positions / safe[..., None]))
        agent = np.where(neighbors[..., None], agent_pair, 0.).sum(axis=1)
        speed_squared = np.sum(relative_velocity * relative_velocity, axis=-1)
        product = np.sum(relative_positions * relative_velocity, axis=-1)
        approaching = product < 0.
        closest_time = np.zeros(distances.shape, dtype=np.float32)
        moving = speed_squared > 1e-6
        closest_time[moving] = np.clip(-product[moving] / speed_squared[moving], 0., p.prediction_horizon_s)
        closest = relative_positions + relative_velocity * closest_time[..., None]
        closest_distance = np.linalg.norm(closest, axis=-1)
        distance_risk = np.clip((p.safe_distance_m - closest_distance) / max(p.safe_distance_m, 1e-6), 0., 1.)
        time_risk = np.clip((p.ttc_threshold_s - closest_time) / max(p.ttc_threshold_s, 1e-6), 0., 1.)
        away = -closest / np.maximum(closest_distance, 1e-6)[..., None]
        preferred = delta / np.maximum(distance, 1e-6)[:, None]
        pass_right = np.stack((preferred[:, 1], -preferred[:, 0]), axis=-1).astype(np.float32)
        away = np.where((closest_distance < 1e-4)[..., None], pass_right[:, None, :], away)
        ttc = (p.ttc_gain * np.where(neighbors, distance_risk * time_risk * approaching, 0.)[..., None]
               * away).sum(axis=1)
        return ForceComponents(attraction, static, agent, ttc)

    def _step(self, positions, velocities, goals):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            forces = self._forces(positions, velocities, goals)
            next_velocity = velocities + self.parameters.integration_dt_s * (
                forces.attraction + forces.static + forces.agent + forces.ttc)
            speed = np.linalg.norm(next_velocity, axis=-1)
            next_velocity *= np.minimum(1., self.parameters.max_speed_mps / np.maximum(speed, 1e-300))[:, None]
            next_position = positions + next_velocity * self.parameters.integration_dt_s
        if not np.isfinite(next_position).all() or not np.isfinite(next_velocity).all():
            raise FloatingPointError("SFM prediction contains nonfinite values")
        self.grid._check_segments(positions, next_position)
        return SFMState(next_position, next_velocity, forces)

    @staticmethod
    def _inputs(positions, velocities, goals):
        # Bind the exact original dtype/bytes before conversion to float64.
        inputs = (positions, velocities, goals)
        for name, value in zip(("positions", "velocities", "goals"), inputs):
            _xy(value, name)
        snapshots = tuple(np.array(value, copy=True) for value in inputs)
        hashes = tuple(_array_sha(value) for value in snapshots)
        if tuple(_array_sha(value) for value in inputs) != hashes:
            raise RuntimeError("Bound positions/velocities/goals input changed during snapshot")
        current = _xy(snapshots[0], "positions")
        return (current, _xy(snapshots[1], "velocities", len(current)),
                _xy(snapshots[2], "goals", len(current)), hashes)

    def step(self, positions, velocities, goals):
        self.verify()
        current, velocity, goal, hashes = self._inputs(positions, velocities, goals)
        self.grid.pixel_indices(goal)
        result = self._step(current, velocity, goal)
        self.verify()
        if tuple(_array_sha(value) for value in (positions, velocities, goals)) != hashes:
            raise RuntimeError("Bound positions/velocities/goals input changed during generation")
        return result

    def generate(self, positions, velocities, goals):
        self.verify()
        inputs = (positions, velocities, goals)
        current, velocity, goal, input_hashes = self._inputs(positions, velocities, goals)
        self.grid.pixel_indices(current)
        self.grid.pixel_indices(goal)
        stride = round(.1 / self.parameters.integration_dt_s)
        steps = stride * 50
        xy_trace, velocity_trace = [current.copy()], [velocity.copy()]
        for _ in range(steps):
            next_state = self._step(current, velocity, goal)
            current, velocity = next_state.xy, next_state.velocity
            xy_trace.append(current.copy())
            velocity_trace.append(velocity.copy())
        self.verify()
        if tuple(_array_sha(value) for value in inputs) != input_hashes:
            raise RuntimeError("Bound positions/velocities/goals input changed during generation")
        internal_xy = np.stack(xy_trace, axis=1)
        raw_xy, raw_velocity = internal_xy[:, ::stride], np.stack(velocity_trace, axis=1)[:, ::stride]
        metadata = {**self.describe(), "input_sha256": list(input_hashes), "actor_count": len(current),
                    "internal_steps": steps, "causal_adapter_owner": "existing AMP backend; not applied here"}
        return SFMTrajectory(_readonly(raw_xy), _readonly(raw_velocity), _readonly(np.arange(51) / 10),
                             _readonly(raw_xy[:, 5::5]), _readonly(np.arange(1, 11) / 2),
                             _readonly(internal_xy), metadata)
