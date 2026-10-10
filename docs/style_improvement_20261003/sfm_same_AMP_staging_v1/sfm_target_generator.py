"""External, CPU-only SFM targets; no controller, runner or simulator changes.

Each call receives one actual full-scene body snapshot. Fifty synchronous
0.1 s prediction steps return 51 world-XY knots, including the true t=0.
The caller owns the nominal 0.5 s/forced replan clock and physical feedback.
Predicted states and waypoint arrivals never update the real world or routes.
This future-target adaptation is local; NavIsaacLab did not publish its rule.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np


SFM_SOURCE = Path(__file__).resolve().parents[1] / "sfm_FOV_staged_fix_v1/after/sfm.py"
SFM_SHA256 = "4b1fe12f980dde21b4105e0757d00f4d02bd7fa86bb437030a6c46b46ad67be3"
PREDICTION_DT_S = 0.1
PREDICTION_STEPS = 50


def _social_force_class():
    """Use the exact frozen class without its unrelated cv2/plot/demo imports."""
    source = SFM_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SFM_SHA256:
        raise ValueError("The isolated SFM equation source changed")
    tree = ast.parse(source, filename=str(SFM_SOURCE))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Social_Force"]
    if len(classes) != 1:
        raise ValueError("Exactly one original Social_Force class is required")
    namespace = {"np": np, "__name__": "isolated_original_sfm_equations"}
    # No AST node in the class is changed: all force/TTC/clamp math is original.
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SFM_SOURCE), "exec"), namespace)
    return namespace["Social_Force"]


@dataclass(frozen=True)
class SFMConfig:
    """Existing local nav_manager defaults, not disclosed author parameters."""
    radius_m: float = 0.35
    max_speed_mps: float = 1.5
    safe_distance_m: float = 0.9
    neighbor_radius_m: float = 4.0
    reach_distance_m: float = 0.45
    ttc_prediction_horizon_s: float = 2.0
    ttc_threshold_s: float = 1.5
    ttc_gain: float = 12.0

    def validate(self):
        for name, value in asdict(self).items():
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (name != "ttc_gain" and value == 0)):
                raise ValueError(f"{name} must be finite and positive (ttc_gain may be zero)")


class WorldEDTMap:
    """Raw known-free Boolean cells, ROS origin, top-left image rows, metres.

    False covers obstacle/unknown; exterior is padded blocked when constructing
    the centre-to-centre EDT. No radius inflation/half-diagonal subtraction is
    applied. This sampled EDT is not a full-body/swept-clearance guarantee.
    """

    def __init__(self, known_free, *, origin_xy_m, resolution_m, origin_yaw_rad=0.0):
        from scipy.ndimage import distance_transform_edt

        free = np.asarray(known_free)
        origin = np.asarray(origin_xy_m, dtype=np.float64)
        if free.ndim != 2 or min(free.shape) < 2 or free.dtype != np.bool_:
            raise ValueError("Raw known-free map must be Boolean[H>=2,W>=2]; no label guessing")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Map origin requires finite world XY metres")
        if (type(resolution_m) not in (int, float) or not math.isfinite(resolution_m)
                or resolution_m <= 0):
            raise ValueError("Map resolution requires positive finite metres/pixel")
        if type(origin_yaw_rad) not in (int, float) or origin_yaw_rad != 0.0:
            raise ValueError("Original SFM gradient supports only exact zero map yaw")
        self.known_free = free.copy()
        self.origin_xy_m = origin.copy()
        self.resolution_m = float(resolution_m)
        self.edt_m = (distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
                      * self.resolution_m)
        for array in (self.known_free, self.origin_xy_m, self.edt_m):
            array.setflags(write=False)

    def pixels(self, xy):
        """No clipping: outside remains invalid; row0=maxY, col0=minX."""
        xy = np.asarray(xy)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("Map query requires finite world[...,2]")
        cell = np.floor((xy - self.origin_xy_m) / self.resolution_m).astype(np.int64)
        return np.stack((self.known_free.shape[0] - 1 - cell[..., 1], cell[..., 0]), -1)

    def inside(self, xy):
        pixels = self.pixels(xy)
        return ((pixels >= 0) & (pixels < np.asarray(self.known_free.shape))).all(-1)


@dataclass(frozen=True)
class SFMTargetPlan:
    xy: np.ndarray
    velocity_xy: np.ndarray
    times_s: np.ndarray
    private_waypoint_indices: np.ndarray
    diagnostics: dict


class SFMTargetGenerator:
    """Stateless physical feedback; synchronous, caller-discardable predictions.

    Scene IDs are explicit. Different groups may share a map, but never interact.
    There is no RNG, physical reset, real route-progress mutation or App import.
    """

    def __init__(self, scene_maps, config=None):
        self.config = SFMConfig() if config is None else config
        if not isinstance(self.config, SFMConfig):
            raise ValueError("Explicit SFMConfig required")
        self.config.validate()
        if (not isinstance(scene_maps, dict) or not scene_maps
                or any(type(key) is not int or not isinstance(value, WorldEDTMap)
                       for key, value in scene_maps.items())):
            raise ValueError("Scene maps require {integer scene_id: WorldEDTMap}")
        self.scene_maps = dict(scene_maps)
        social_force = _social_force_class()
        self._controllers = {}
        for scene_id, scene_map in self.scene_maps.items():
            controller = social_force(scene_map.edt_m, {
                "map": {"resolution": scene_map.resolution_m},
                "env": {"dt": PREDICTION_DT_S, "safe_distance": self.config.safe_distance_m,
                        "neighbor_radius": self.config.neighbor_radius_m,
                        "reach_distance": self.config.reach_distance_m,
                        "prediction_horizon": self.config.ttc_prediction_horizon_s,
                        "ttc_threshold": self.config.ttc_threshold_s, "ttc_gain": self.config.ttc_gain},
                "agent": {"radius": self.config.radius_m, "max_vel": self.config.max_speed_mps},
            })
            if controller.fov is not False:
                raise ValueError("This isolated comparison requires original FOV=False")
            self._controllers[scene_id] = controller

    def generate(self, states, route_waypoints, *, scene_ids, actor_ids=None):
        """Consume actual[N,5] and ordered world waypoints[N,K>=1,2].

        All inputs are copied. At each prediction substep all forces read the
        same previous positions/velocities; new velocities/XY commit together.
        Waypoints advance only on this call's private proximity index, not by
        altering true route arrivals or claiming their declared timed speeds.
        Domain exits stop at the last in-domain prediction and retain attempted
        XY. A true t=0 outside root is preserved and held, never edge-clamped.
        SFM may still enter obstacle/unknown cells; retain that negative evidence.
        """
        states = np.asarray(states)
        if (states.ndim != 2 or states.shape[1] != 5 or not len(states)
                or states.dtype not in (np.float32, np.float64) or not np.isfinite(states).all()):
            raise ValueError("Actual full-scene states require finite float32/64[N,5]")
        waypoints = np.asarray(route_waypoints, dtype=states.dtype)
        if (waypoints.ndim != 3 or waypoints.shape[0] != len(states)
                or waypoints.shape[1] < 1 or waypoints.shape[2] != 2
                or not np.isfinite(waypoints).all()):
            raise ValueError("Ordered route waypoints require finite world[N,K>=1,2]")
        groups = np.asarray(scene_ids)
        ids = np.arange(len(states)) if actor_ids is None else np.asarray(actor_ids)
        if (groups.shape != (len(states),) or groups.dtype.kind not in "iu"
                or ids.shape != (len(states),) or ids.dtype.kind not in "iu"
                or len(np.unique(ids)) != len(ids)
                or any(int(group) not in self.scene_maps for group in groups)):
            raise ValueError("One known integer scene and unique integer actor ID per actual body required")
        positions, velocities = states[:, :2].copy(), states[:, 2:4].copy()
        waypoints, groups, ids = waypoints.copy(), groups.copy(), ids.copy()
        shape = (len(states), PREDICTION_STEPS + 1)
        xy, velocity_xy = np.empty((*shape, 2), dtype=states.dtype), np.empty((*shape, 2), dtype=states.dtype)
        xy[:, 0], velocity_xy[:, 0] = positions, velocities
        private_indices = np.zeros(shape, dtype=np.int64)
        stopped = np.zeros(len(states), dtype=bool)
        exits, blocked_samples = [], np.zeros(len(states), dtype=np.int64)
        minimum_edt = np.full(len(states), np.inf)

        for step in range(PREDICTION_STEPS):
            next_velocity = np.zeros_like(velocities)
            for row in range(len(states)):
                if stopped[row]:
                    continue
                scene_map = self.scene_maps[int(groups[row])]
                if not scene_map.inside(positions[row]):
                    stopped[row] = True
                    exits.append({"actor_id": int(ids[row]), "scene_id": int(groups[row]),
                                  "step": step, "time_s": step * PREDICTION_DT_S,
                                  "kind": "actual_initial_outside" if step == 0 else "predicted_outside",
                                  "attempted_xy_m": positions[row].tolist()})
                    continue
                pixel = scene_map.pixels(positions[row])
                index = private_indices[row, step]
                while (index < waypoints.shape[1] - 1
                       and np.linalg.norm(waypoints[row, index] - positions[row]) <= self.config.reach_distance_m):
                    index += 1
                private_indices[row, step + 1] = index
                relative = positions - positions[row]
                distances = np.linalg.norm(relative, axis=1)
                neighbors = np.flatnonzero((groups == groups[row]) & (np.arange(len(states)) != row)
                                          & (distances < self.config.neighbor_radius_m))
                next_velocity[row], _ = self._controllers[int(groups[row])].get_action(
                    (positions[row], pixel, velocities[row], waypoints[row, index]),
                    (ids[neighbors], distances[neighbors], relative[neighbors], velocities[neighbors] - velocities[row]))
            if not np.isfinite(next_velocity).all():
                raise FloatingPointError("Original SFM returned nonfinite predicted velocity; do not silently replace")
            candidates = positions + next_velocity * PREDICTION_DT_S
            for row in range(len(states)):
                if stopped[row]:
                    private_indices[row, step + 1] = private_indices[row, step]
                    continue
                if not self.scene_maps[int(groups[row])].inside(candidates[row]):
                    stopped[row] = True
                    exits.append({"actor_id": int(ids[row]), "scene_id": int(groups[row]),
                                  "step": step + 1, "time_s": (step + 1) * PREDICTION_DT_S,
                                  "kind": "predicted_domain_exit", "attempted_xy_m": candidates[row].tolist()})
                    candidates[row], next_velocity[row] = positions[row], 0
            positions, velocities = candidates, next_velocity
            xy[:, step + 1], velocity_xy[:, step + 1] = positions, velocities

        # Count all returned knots, including a stopped actor's repeated positions.
        for row in range(len(states)):
            scene_map = self.scene_maps[int(groups[row])]
            valid = scene_map.inside(xy[row])
            pixels = scene_map.pixels(xy[row, valid])
            if len(pixels):
                blocked_samples[row] = int((~scene_map.known_free[pixels[:, 0], pixels[:, 1]]).sum())
                minimum_edt[row] = float(scene_map.edt_m[pixels[:, 0], pixels[:, 1]].min())
        diagnostics = {
            "sfm_equation_source": str(SFM_SOURCE), "sfm_equation_sha256": SFM_SHA256,
            "local_future_target_adaptation": True, "author_exact_configuration": False,
            "configuration": asdict(self.config), "prediction_dt_s": PREDICTION_DT_S,
            "prediction_steps": PREDICTION_STEPS, "nominal_external_replan_dt_s": 0.5,
            "FOV": False, "waypoint_progress": "private predicted proximity only; input route untouched",
            "body_feedback": "one actual full-scene snapshot per call; no predicted state written to physical/history",
            "domain_exit_events": exits, "stopped_actor_ids": ids[stopped].tolist(),
            "blocked_or_unknown_knot_counts": blocked_samples.tolist(),
            "minimum_raw_EDT_m": [None if not np.isfinite(value) else float(value) for value in minimum_edt],
            "initial_overspeed_actor_ids": ids[np.linalg.norm(states[:, 2:4], axis=1) > self.config.max_speed_mps].tolist(),
            "geometry_guarantee": False, "candidate_selection": "single deterministic target; no diffusion samples",
        }
        return SFMTargetPlan(xy, velocity_xy, np.arange(PREDICTION_STEPS + 1, dtype=np.float64) / 10,
                             private_indices, diagnostics)
