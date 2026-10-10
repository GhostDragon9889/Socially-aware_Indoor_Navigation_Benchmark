"""Declared static geometry and bounded routes for reconstruction validation.

These fixtures are newly authored test scenes, not the eight unpublished
NavIsaacLab benchmark scenes. The same cuboids define PhysX geometry, height
observations and the diffusion raster, avoiding a visual-only obstacle map.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
import numpy as np


class ReconstructionScene:
    def __init__(self, description, *, device="cpu"):
        if description.get("schema") != "navisaaclab.reconstruction_scene.v1":
            raise ValueError("Unsupported reconstruction scene schema")
        self.description = description
        self.device = torch.device(device)
        self.bounds = self.tensor(description["bounds_xy"])
        self.origins = self.tensor(description["origins_xy"])
        self.routes = self.tensor(description["routes_xy"])
        boxes = description.get("cuboids", [])
        self.centers = self.tensor([box["center"] for box in boxes]).reshape(-1, 3)
        self.sizes = self.tensor([box["size"] for box in boxes]).reshape(-1, 3)
        self.arrival_radius = float(description.get("arrival_radius_m", .75))
        self.lookahead = float(description.get("goal_lookahead_m", 5.))
        self.planning_clearance = float(description.get("planning_clearance_m", .35))
        if not math.isfinite(self.planning_clearance) or self.planning_clearance < .35:
            raise ValueError("Planning clearance must be finite and at least .35m")
        if (self.bounds.shape != (4,) or bool((self.bounds[2:] <= self.bounds[:2]).any())
                or self.origins.ndim != 2 or self.origins.shape[1] != 2
                or self.routes.ndim != 3 or self.routes.shape[0] != len(self.origins)
                or self.routes.shape[1] < 2 or self.routes.shape[2] != 2
                or bool((self.sizes <= 0).any()) or not math.isfinite(self.arrival_radius)
                or not math.isfinite(self.lookahead) or min(self.arrival_radius, self.lookahead) <= 0):
            raise ValueError("Invalid scene geometry or route dimensions")
        for points in (self.origins, self.routes):
            if not bool(self.in_domain(points).all()) or bool(self.occupied(points).any()):
                raise ValueError("Spawns and route vertices must be inside the free scene domain")
        self.right_hand_traffic = description.get("right_hand_traffic", False)
        if type(self.right_hand_traffic) is not bool:
            raise ValueError("Right-hand traffic must be explicitly boolean")
        self.right_hand_padding_distance = description.get("right_hand_padding_distance_m", .5)
        self.right_hand_cost_weight = description.get("right_hand_cost_weight", 2.)
        self.right_hand_corridor_width = description.get("right_hand_corridor_width_m", 4.)
        for value in (self.right_hand_padding_distance, self.right_hand_cost_weight, self.right_hand_corridor_width):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Lane distance, cost weight and corridor width must be positive finite numbers")
        self._global_planner = None
        self.global_path_failures = 0
        recovery_radius = description.get("start_recovery_radius_m", 0.)
        if isinstance(recovery_radius, bool):
            raise ValueError("Invalid start recovery radius")
        self.start_recovery_radius = float(recovery_radius)
        recovery_limit = max(.75, self.planning_clearance + .4)
        if not math.isfinite(self.start_recovery_radius) or not 0 <= self.start_recovery_radius <= recovery_limit:
            raise ValueError("Start recovery radius must fit planning clearance plus grid quantisation")
        self.successful_start_recoveries = 0
        self.last_start_recovery_segments = []
        self.last_path_query_failures = []

    def _get_global_planner(self):
        from CrowdSim.world.planning import Path_Planner

        resolution = .25
        bounds = self.bounds.detach().cpu().numpy()
        if self._global_planner is None:
            cols = torch.arange(bounds[0], bounds[2]+resolution/2, resolution, device=self.device)
            rows = torch.arange(bounds[1], bounds[3]+resolution/2, resolution, device=self.device)
            y, x = torch.meshgrid(rows, cols, indexing="ij")
            # Mark cells whose square footprint intersects a cuboid. Point
            # samples can entirely miss a20cm wall on a25cm grid, especially
            # when its floating-point endpoint lies just outside the sample.
            xy = torch.stack((x, y), -1)
            blocked = torch.zeros(xy.shape[:-1], device=self.device, dtype=torch.bool)
            for center, size in zip(self.centers, self.sizes):
                blocked |= ((xy-center[:2]).abs() <= size[:2]/2 + resolution/2 + 1e-6).all(-1)
            occupied = blocked.cpu().numpy().astype(np.uint8)
            self._global_planner = Path_Planner(occupied, resolution, step_size_m=.25,
                                               clearance_m=self.planning_clearance, smooth=False, verbose=False)
            if self.right_hand_traffic:
                from CrowdSim.humanoid.runtime.keep_right_planning import KeepRightPathPlanner
                self._global_planner = KeepRightPathPlanner(
                    self._global_planner,
                    padding_distance_m=self.right_hand_padding_distance,
                    weight=self.right_hand_cost_weight,
                    corridor_width_m=self.right_hand_corridor_width,
                )
        return self._global_planner

    def _connector_avoids_cuboids(self, start, end):
        """Check a .35m body envelope; wider planning clearance is not certified."""
        delta = end-start
        centers, sizes = self.centers.cpu().numpy(), self.sizes.cpu().numpy()
        for center, size in zip(centers, sizes):
            lower, upper = center[:2]-size[:2]/2-.35, center[:2]+size[:2]/2+.35
            enter, leave = 0., 1.
            for axis in range(2):
                if delta[axis] == 0:
                    if not lower[axis] <= start[axis] <= upper[axis]:
                        break
                else:
                    a, b = (lower[axis]-start[axis])/delta[axis], (upper[axis]-start[axis])/delta[axis]
                    enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
                if enter > leave:
                    break
            else:
                return False
        return True

    def _recover_path_start(self, position, end, bounds, resolution):
        """Find the nearest reachable free node within the declared radius."""
        planner = self._global_planner
        radius = self.start_recovery_radius
        lower = np.maximum(np.ceil((position-bounds[:2]-radius)/resolution).astype(int)[::-1], 0)
        upper = np.minimum(np.floor((position-bounds[:2]+radius)/resolution).astype(int)[::-1],
                           np.asarray(planner.map_size)-1)
        if np.any(lower > upper):
            return None, None
        nodes = np.argwhere(planner.map_dialate[lower[0]:upper[0]+1, lower[1]:upper[1]+1] == 0)+lower
        points = nodes[:, ::-1]*resolution+bounds[:2]
        distances = np.linalg.norm(points-position, axis=-1)
        order = np.lexsort((nodes[:, 1], nodes[:, 0], distances))
        for candidate in order:
            if distances[candidate] > radius or not self._connector_avoids_cuboids(position, points[candidate]):
                continue
            path = planner.get_astar_path(nodes[candidate], end)
            if path is not None:
                return path, points[candidate]
        return None, None

    def _planning_paths(self, positions, destinations, *, timed=False):
        """One A* path per query; optional recovery is exclusive to timed mode."""
        resolution = .25
        bounds = self.bounds.detach().cpu().numpy()
        planner = self._get_global_planner()
        positions_np, destinations_np = positions.detach().cpu().numpy(), destinations.detach().cpu().numpy()
        paths = []
        self.last_path_query_failures = []
        limits = np.asarray(planner.map_size)-1
        valid_roots = (self.in_domain(positions) & ~self.occupied(positions)).cpu().numpy() if timed else None
        for row, (position, destination) in enumerate(zip(positions_np, destinations_np)):
            start = np.rint((position-bounds[:2])/resolution).astype(int)[::-1]
            end = np.rint((destination-bounds[:2])/resolution).astype(int)[::-1]
            recovery_point = None
            reason = "astar_unreachable_or_destination_occupied"
            if np.any(start < 0) or np.any(start > limits) or (timed and not valid_roots[row]):
                path = None
                reason = "actual_root_outside_domain_or_inside_physical_cuboid"
            elif timed and self.start_recovery_radius > 0 and planner.map_dialate[tuple(start)] != 0:
                path, recovery_point = self._recover_path_start(position, end, bounds, resolution)
                reason = "no_reachable_body_safe_connector_within_recovery_radius"
            else:
                path = planner.get_astar_path(start, end)
            if path is None:
                self.last_path_query_failures.append({
                    "query_row": row,
                    "actual_position_xy": position.tolist(),
                    "destination_xy": destination.tolist(),
                    "reason": reason,
                    "recovery_radius_m": self.start_recovery_radius if timed else 0.,
                    "planning_clearance_m": self.planning_clearance,
                })
                self.global_path_failures += 1
                paths.append(None)
                continue
            path_xy = np.concatenate((position[None], path[:, ::-1]*resolution+bounds[:2]), 0)
            paths.append(path_xy)
            if recovery_point is not None:
                self.successful_start_recoveries += 1
                self.last_start_recovery_segments.append({
                    "query_row": row,
                    "actual_position_xy": position.tolist(),
                    "planner_start_xy": recovery_point.tolist(),
                    "connector_length_m": float(np.linalg.norm(recovery_point-position)),
                    "initial_segment_clearance_certified": False,
                })
        return paths

    def _path_goal(self, positions, destinations):
        """Keep the original endpoint contract, including failure counters."""
        paths = self._planning_paths(positions, destinations)
        goals = []
        for position, path_xy in zip(positions.detach().cpu().numpy(), paths):
            if path_xy is None:
                goals.append(position)
                continue
            cumulative = np.concatenate(([0.], np.linalg.norm(np.diff(path_xy, axis=0), axis=-1).cumsum()))
            distance = min(self.lookahead, cumulative[-1])
            goals.append([np.interp(distance, cumulative, path_xy[:, axis]) for axis in range(2)])
        return self.tensor(np.asarray(goals))

    @classmethod
    def load(cls, path, **kwargs):
        return cls(yaml.safe_load(Path(path).read_text()), **kwargs)

    def tensor(self, value):
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Scene coordinates must be finite")
        return value

    def in_domain(self, xy):
        return ((xy >= self.bounds[:2]) & (xy <= self.bounds[2:])).all(-1)

    def occupied(self, xy):
        result = torch.zeros(xy.shape[:-1], dtype=torch.bool, device=xy.device)
        for center, size in zip(self.centers, self.sizes):
            result |= ((xy-center[:2]).abs() <= size[:2]/2).all(-1)
        return result

    def obstacle_height(self, xy):
        result = torch.zeros(xy.shape[:-1], device=xy.device)
        for center, size in zip(self.centers, self.sizes):
            inside = ((xy-center[:2]).abs() <= size[:2]/2).all(-1)
            result = torch.where(inside, torch.maximum(result, center[2]+size[2]/2), result)
        return result

    def raster(self, states, *, outside_value=.5):
        """ORCA crop: local x -> col=12*x+56; local y -> row=12*y+112."""
        if states.ndim != 2 or states.shape[1] != 5 or not bool(torch.isfinite(states).all()):
            raise ValueError("Expected finite world states[B,5]")
        if not math.isfinite(outside_value) or not 0 <= outside_value <= 1:
            raise ValueError("Invalid outside raster value")
        row, col = torch.meshgrid(torch.arange(224, device=self.device),
                                  torch.arange(224, device=self.device), indexing="ij")
        x, y = (col-56)/12, (row-112)/12
        c, s = states[:, 4, None, None].cos(), states[:, 4, None, None].sin()
        world = torch.stack((states[:, 0, None, None]+c*x-s*y,
                             states[:, 1, None, None]+s*x+c*y), -1)
        valid, occupied = self.in_domain(world), self.occupied(world)
        unknown = torch.full(valid.shape, outside_value, device=self.device)
        free = torch.where(valid, (~occupied).float(), unknown)
        obstacle = torch.where(valid, occupied.float(), unknown)
        return torch.stack((free, obstacle), 1)

    def _route_inputs(self, positions, route_indices, actor_ids):
        positions = self.tensor(positions)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Expected finite positions[B,2]")
        batch = len(positions)

        def indices(value, name, limit):
            value = torch.as_tensor(value, device=self.device)
            if (value.ndim != 1 or len(value) != batch or value.dtype == torch.bool
                    or value.is_floating_point() or value.is_complex()
                    or bool(((value < 0) | (value >= limit)).any())):
                raise ValueError(f"Expected integer {name}[B] within range")
            return value.long()

        route_indices = indices(route_indices, "route_indices", self.routes.shape[1])
        ids = indices(torch.arange(batch, device=self.device) if actor_ids is None else actor_ids,
                      "actor_ids", len(self.origins))
        return positions, route_indices, ids

    def _route_progress(self, positions, route_indices, ids, advance):
        if not isinstance(advance, bool):
            raise ValueError("Route advance must be a boolean")
        reached = (torch.linalg.vector_norm(self.routes[ids, route_indices]-positions, dim=-1)
                   <= self.arrival_radius) if advance else torch.zeros(
                       len(positions), dtype=torch.bool, device=self.device)
        return (route_indices+reached.long()) % self.routes.shape[1], reached

    def advance_route(self, positions, route_indices, *, actor_ids=None):
        """Check one vertex against an actual control-frame position.

        The runner calls this after a nonterminal physics step, before reset.
        This pure query returns new indices and a reached mask; it does not
        change physical state, perform A*, or accumulate counters. A reset
        position must not be passed as evidence of traversing a route.
        """
        positions, route_indices, ids = self._route_inputs(positions, route_indices, actor_ids)
        return self._route_progress(positions, route_indices, ids, True)

    def route_goal(self, positions, route_indices, *, actor_ids=None, advance=True):
        """Advance reached vertices; expose a reachable five-second local goal."""
        positions, route_indices, ids = self._route_inputs(positions, route_indices, actor_ids)
        updated, reached = self._route_progress(positions, route_indices, ids, advance)
        destination = self.routes[ids, updated]
        goal = self._path_goal(positions, destination)
        return goal, updated, reached

    def route_waypoints(self, positions, route_indices, *, actor_ids=None, advance=True,
                        times=(.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.)):
        """Sample one global path at ``lookahead * time / 5`` metres.

        Times are seconds in (0, 5], strictly increasing. The point at 5s
        equals the legacy endpoint for valid unrecovered starts. Short paths
        hold their final grid point. Failed queries hold the actual root.
        Recovery never changes physical state; its initial connector is free
        of cuboids inflated by a .35m body envelope, without certifying the
        wider planner clearance.
        """
        positions, route_indices, ids = self._route_inputs(positions, route_indices, actor_ids)
        batch = len(positions)
        times = torch.as_tensor(times, device=self.device)
        if (times.ndim != 1 or not len(times) or times.dtype == torch.bool or times.is_complex()
                or not bool(torch.isfinite(times).all()) or bool(((times <= 0) | (times > 5)).any())
                or bool((times[1:] <= times[:-1]).any())):
            raise ValueError("Expected finite increasing times[K] in (0,5] seconds")
        times_np = times.double().cpu().numpy()
        updated, reached = self._route_progress(positions, route_indices, ids, advance)
        self.last_start_recovery_segments = []
        if batch == 0:
            return positions.new_empty((0, len(times), 2)), updated, reached
        paths = self._planning_paths(positions, self.routes[ids, updated], timed=True)
        for failure in self.last_path_query_failures:
            failure["actor_id"] = int(ids[failure["query_row"]])
        for recovery in self.last_start_recovery_segments:
            recovery["actor_id"] = int(ids[recovery["query_row"]])
        points = []
        for position, path_xy in zip(positions.detach().cpu().numpy(), paths):
            if path_xy is None:
                points.append(np.repeat(position[None], len(times), axis=0))
                continue
            cumulative = np.concatenate(([0.], np.linalg.norm(np.diff(path_xy, axis=0), axis=-1).cumsum()))
            distances = np.minimum(self.lookahead*(times_np/5.), cumulative[-1])
            points.append(np.stack([np.interp(distances, cumulative, path_xy[:, axis]) for axis in range(2)], -1))
        return self.tensor(np.asarray(points)), updated, reached
