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
        self._global_planner = None
        self.global_path_failures = 0

    def _path_goal(self, positions, destinations):
        """Use the repository's clearance-aware A* for sparse global waypoints.

        No actor state is moved when planning fails: its local goal stays at
        the actual position and the failure is recorded.
        """
        from CrowdSim.world.planning import Path_Planner

        resolution = .25
        bounds = self.bounds.detach().cpu().numpy()
        if self._global_planner is None:
            cols = torch.arange(bounds[0], bounds[2]+resolution/2, resolution, device=self.device)
            rows = torch.arange(bounds[1], bounds[3]+resolution/2, resolution, device=self.device)
            y, x = torch.meshgrid(rows, cols, indexing="ij")
            occupied = self.occupied(torch.stack((x, y), -1)).cpu().numpy().astype(np.uint8)
            self._global_planner = Path_Planner(occupied, resolution, step_size_m=.25,
                                               clearance_m=.35, smooth=False, verbose=False)
        positions_np, destinations_np = positions.detach().cpu().numpy(), destinations.detach().cpu().numpy()
        goals = []
        limits = np.asarray(self._global_planner.map_size)-1
        for position, destination in zip(positions_np, destinations_np):
            start = np.rint((position-bounds[:2])/resolution).astype(int)[::-1]
            end = np.rint((destination-bounds[:2])/resolution).astype(int)[::-1]
            if np.any(start < 0) or np.any(start > limits):
                path = None
            else:
                path = self._global_planner.get_astar_path(start, end)
            if path is None:
                self.global_path_failures += 1
                goals.append(position)
                continue
            path_xy = np.concatenate((position[None], path[:, ::-1]*resolution+bounds[:2]), 0)
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

    def route_goal(self, positions, route_indices, *, actor_ids=None):
        """Advance reached vertices; expose a reachable five-second local goal."""
        ids = torch.arange(len(positions), device=self.device) if actor_ids is None else actor_ids
        destination = self.routes[ids, route_indices]
        reached = torch.linalg.vector_norm(destination-positions, dim=-1) <= self.arrival_radius
        updated = (route_indices+reached.long()) % self.routes.shape[1]
        destination = self.routes[ids, updated]
        goal = self._path_goal(positions, destination)
        return goal, updated, reached
