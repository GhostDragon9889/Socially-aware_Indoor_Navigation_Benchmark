"""Prepared navigation companion for source-bound native DynBench maps.

No App, SDK, world, or tensor runtime is imported at module import. Occupancy
is a measured grid, never reconstructed as fictitious PhysX cuboids. The SDF
is a differentiable grid proxy, not proof of body or contact-point clearance.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
import json
import math
from pathlib import Path
import sys

from .dynbench_scene import DynBenchSceneProvider, HEIGHT_BAND, _check_record


def _number(value, label, *, minimum=0., positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value) or value < minimum
            or (positive and value == minimum)):
        raise ValueError(f"{label} must be finite and {'above' if positive else 'at least'} {minimum}")
    return float(value)


def _array(value):
    import numpy as np
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _points(value, label):
    import numpy as np
    points = np.asarray(_array(value), dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all() or np.any(np.abs(points) > 1e6):
        raise ValueError(f"{label} requires finite world[...,2] within the safe indexing range; no clamping")
    return points


def _archive(record):
    import numpy as np
    _check_record(record)
    with np.load(record["path"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _native_centers(points, minimum, dimensions):
    """Return ROS pixels and fixed SDK indices; never fit layout from labels."""
    import numpy as np
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Native world cell centers require finite [N,3]")
    grid = np.rint((points[:, :2] - minimum[:2]) / .1 - .5).astype(np.int64)
    if (np.any(grid < 0) or np.any(grid >= dimensions[:2])
            or not np.allclose(points[:, :2], minimum[:2] + (grid + .5) * .1, atol=2e-4, rtol=0)
            or len(np.unique(grid, axis=0)) != len(grid)):
        raise ValueError("Native positions are not unique source-grid cell centers")
    pixels = np.column_stack((dimensions[1] - 1 - grid[:, 1], grid[:, 0]))
    indices = grid[:, 1] * dimensions[0] + dimensions[0] - 1 - grid[:, 0]
    return pixels, indices


def _full_height_admission(provider):
    """Cross-check raw fixed-layout labels, complete height hits, and floor rays."""
    import numpy as np
    from PIL import Image

    provider.require_navigation_map()
    provider.verify_sources()
    native = provider._native
    _check_record(native["record"])
    data = deepcopy(native["data"])
    if json.loads(Path(native["record"]["path"]).read_text()) != data:
        raise ValueError("Admitted native evidence differs from its source-bound provider snapshot")
    try:
        refinement, band, coordinates = data["height_band_refinement"], data["full_height_band_queries"], data["coordinate_validation"]
        for record in data["exports"].values():
            _check_record(record)
        from CrowdSim.utils.map_metadata import load_occupancy_map_metadata
        metadata = load_occupancy_map_metadata(data["exports"]["occupancy.yaml"]["path"])
        if (metadata.resolution != .1 or metadata.origin_yaw != 0. or metadata.negate != 0
                or not np.allclose(metadata.origin_xy, provider.bounds_xy[0], atol=2e-4, rtol=0)
                or str(metadata.image_path) != data["exports"]["occupancy.png"]["path"]):
            raise ValueError("Native metadata requires exact finite .1m source-bound world coordinates")
        if (refinement["all_native_known5_cells_tested"] is not True
                or refinement["native_unknown_never_promoted"] is not True
                or refinement["untested_known5_exported_unknown"] != 0
                or refinement["unsupported_tested_cells"] != 0 or refinement["unclassified_cells"] != 0
                or refinement["cell_xy_size_m"] != .1
                or refinement["band_above_selected_floor_m"] != list(HEIGHT_BAND)
                or refinement["engineering_refinement"] is not True
                or refinement["pure_omap2D_full_height_claim"] is not False
                or not np.allclose(data["requested_height_band_world_m"], np.asarray(HEIGHT_BAND) + provider.ground_height_m, atol=1e-8, rtol=0)
                or coordinates["fixed_native_layout"] != "row-major x-fastest; row0=minY, column0=maxX"
                or coordinates["layout_fitted_dynamically"] is not False
                or coordinates["unknown_promoted_to_free"] is not False
                or coordinates["positions_roundtrip_verified"] is not True):
            raise ValueError("Navigation requires complete source-bound native full-height/fixed-layout admission")
        raw = _archive(data["native_omap"])
        dims, lower, upper, buffer = raw["dimensions"], raw["min_bound"], raw["max_bound"], raw["buffer"]
        if (dims.shape != (3,) or dims.dtype.kind not in "iu" or dims[2] not in (0, 1)
                or np.any(dims[:2] <= 0) or lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or buffer.shape != (int(dims[0] * dims[1]),) or not np.isin(buffer, [4, 5, 6]).all()
                or not np.allclose(np.stack((lower[:2], upper[:2])), provider.bounds_xy, atol=2e-4, rtol=0)
                or not np.allclose(upper[:2] - lower[:2], dims[:2] * .1, atol=2e-4, rtol=0)):
            raise ValueError("Native raw dimensions/bounds/4,5,6 labels disagree with source grid")
        image = np.where(buffer.reshape(int(dims[1]), int(dims[0]))[::-1, ::-1] == 4, 0,
                         np.where(buffer.reshape(int(dims[1]), int(dims[0]))[::-1, ::-1] == 5, 254, 205)).astype(np.uint8)
        _, occupied_indices = _native_centers(raw["occupied_positions"], lower, dims)
        _, api_indices = _native_centers(raw["free_positions"], lower, dims)
        if (not np.array_equal(np.sort(occupied_indices), np.flatnonzero(buffer == 4))
                or np.intersect1d(api_indices, occupied_indices).size
                or not np.isin(np.flatnonzero(buffer == 5), api_indices).all()
                or np.any(buffer[api_indices] == 4)):
            raise ValueError("Native world positions contradict fixed SDK buffer labels")
        canonical = raw["free_positions"][buffer[api_indices] == 5]
        canonical_pixels, _ = _native_centers(canonical, lower, dims)
        count = len(canonical)
        world = _archive(coordinates["canonical_world_positions_artifact"])
        if count < 1 or not np.array_equal(world["canonical_free_world_positions"], canonical):
            raise ValueError("Canonical known5 positions differ from original native API data")
        for name in ("native_reconstructed.png", "height_band_refined.png", "occupancy.png"):
            _check_record(data["exports"][name])
        if not np.array_equal(np.asarray(Image.open(data["exports"]["native_reconstructed.png"]["path"])), image):
            raise ValueError("Reconstructed image differs from the official fixed native layout")
        rays = _archive(data["floor_support"]["artifact"])
        hits = _archive(band["artifact"])
        selected = np.arange(count, dtype=np.int64)
        for arrays in (rays, hits):
            if (arrays["native_free_index"].dtype.kind not in "iu"
                    or not np.array_equal(arrays["native_free_index"], selected)
                    or not np.array_equal(arrays["free_world_positions"], canonical)):
                raise ValueError("Full-height and floor queries must cover every canonical native known5 row")
        if (rays["supported"].shape != (count,) or rays["supported"].dtype != np.bool_
                or not rays["supported"].all() or count != data["floor_support"]["tested"]
                or count != refinement["known5_source"] or count != refinement["known5_tested"]
                or band["queried_cells"] != count):
            raise ValueError("Incomplete native floor/full-height coverage")
        offsets, paths, classes = hits["hit_offsets"], hits["hit_collision_paths"], hits["hit_classification"]
        if (offsets.shape != (count + 1,) or offsets.dtype.kind != "i" or offsets[0] != 0
                or np.any(np.diff(offsets) < 0) or offsets[-1] != len(paths)
                or np.any(offsets < 0) or np.any(offsets > len(paths))
                or paths.ndim != 1 or paths.dtype.kind not in "US" or classes.shape != paths.shape
                or classes.dtype.kind not in "US" or hits["hit_rigid_body_paths"].shape != paths.shape):
            raise ValueError("Full-height hit offsets and path/class buffers are inconsistent")
        owner = np.repeat(np.arange(count), np.diff(offsets))
        floor = np.isin(paths, provider.floor_paths)
        static = np.isin(paths, provider.static_paths)
        if (not (floor | static).all() or not np.array_equal(classes, np.where(floor, "selected_floor_excluded", "authored_obstacle"))):
            raise ValueError("Height hits must be exact mutually exclusive floor/static paths; no foreign hit")
        obstacle = np.zeros(count, dtype=bool)
        obstacle[owner[static]] = True
        floor_count = np.bincount(owner[floor], minlength=count)
        for name, expected in (("authored_obstacle_hit", obstacle), ("all_hit_paths_classified", np.ones(count, bool)), ("full_height_clear", ~obstacle)):
            if hits[name].dtype != np.bool_ or not np.array_equal(hits[name], expected):
                raise ValueError(f"Raw full-height hit paths contradict {name}")
        if (hits["selected_floor_hit_count"].dtype.kind not in "iu"
                or not np.array_equal(hits["selected_floor_hit_count"], floor_count)
                or refinement["height_blocked_known5_cells"] != int(obstacle.sum())
                or refinement["height_clear_known5_cells"] != int((~obstacle).sum())):
            raise ValueError("Recorded full-height classification counts disagree with raw hits")
        _check_record(band["clock_artifact"])
        clock = json.loads(Path(band["clock_artifact"]["path"]).read_text())
        before, after = band["clock_before"], band["clock_after"]
        if (before != after or band["query_time_unchanged"] is not True or band["foreign_hit_paths"] != []
                or clock != {key: value for key, value in band.items() if key not in {"artifact", "clock_artifact", "queried_cells"}}
                or type(before["submitted_physics_step"]) is not int or before["submitted_physics_step"] < 0
                or type(before["timeline_s"]) not in (float, int) or not math.isfinite(before["timeline_s"]) or before["timeline_s"] < 0):
            raise ValueError("Native full-height queries lack one unchanged observed counter/timeline")
        refined = image.copy()
        refined[canonical_pixels[obstacle, 0], canonical_pixels[obstacle, 1]] = 0
        if not np.array_equal(np.asarray(Image.open(data["exports"]["height_band_refined.png"]["path"])), refined):
            raise ValueError("Height-refined grid differs from actual complete native hits")
        projection = np.asarray(Image.open(provider._data["input_projection"]["occupancy.png"]["path"]))
        refined[(projection == 205) & (refined != 0)] = 205
        if not np.array_equal(refined, provider._map):
            raise ValueError("Admitted grid differs from full-height native grid with retained unknown support")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Native full-height navigation evidence is incomplete") from error
    return {"native_evidence": deepcopy(native["record"]), "canonical_known5_cells": count,
            "height_blocked_known5_cells": int(obstacle.sum()), "source_bound_artifacts_checked": True,
            "fixed_native_layout": coordinates["fixed_native_layout"], "query_clock": deepcopy(before),
            "full_height_band_above_floor_m": list(HEIGHT_BAND), "new_physical_execution_verified": False}


class NativeGridScene:
    """Measured-grid A* and route companion, without creating physical geometry.

    Route methods match ReconstructionScene. NumPy-only construction/grid
    queries do not import Torch; tensor/route/raster/SDF use it lazily. The
    future runtime owns one existing provider/App/floor and all physical steps.
    """

    @classmethod
    def from_provider(cls, provider, *, origins_xy, routes_xy, clearance_m=.35,
                      arrival_radius_m=.75, goal_lookahead_m=5., device="cpu"):
        return cls(provider, origins_xy=origins_xy, routes_xy=routes_xy, clearance_m=clearance_m,
                   arrival_radius_m=arrival_radius_m, goal_lookahead_m=goal_lookahead_m, device=device)

    def __init__(self, provider, *, origins_xy, routes_xy, clearance_m=.35,
                 arrival_radius_m=.75, goal_lookahead_m=5., device="cpu"):
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        if not isinstance(provider, DynBenchSceneProvider):
            raise ValueError("An admitted concrete DynBenchSceneProvider is required; no projection adapter")
        self._admission = _full_height_admission(provider)
        self.provider = provider
        self.clearance_m = _number(clearance_m, "clearance radius")
        self.arrival_radius = _number(arrival_radius_m, "arrival radius", positive=True)
        self.lookahead = _number(goal_lookahead_m, "goal lookahead", positive=True)
        torch_module = sys.modules.get("torch")
        if (type(device) is not str and not (torch_module is not None and isinstance(device, torch_module.device))) or not str(device):
            raise ValueError("An explicit tensor device string is required")
        self.device = str(device)
        self.resolution = .1
        self._labels = provider._map.copy()
        self._origin = np.asarray(provider._map_origin_xy, dtype=np.float64)
        self._bounds = np.asarray(provider.bounds_xy, dtype=np.float64).reshape(4)
        blocked = self._labels != 254
        # A padded blocked ring treats all map-exterior space as forbidden.
        free_distance = distance_transform_edt(np.pad(~blocked, 1, constant_values=False))[1:-1, 1:-1] * .1
        half_diagonal = math.sqrt(2) * .1 / 2
        # Subtract both the blocked-cell and candidate-cell half diagonal:
        # every point of an admitted cell has at least clearance_m separation.
        self._walkable = ~blocked & (free_distance - 2 * half_diagonal >= self.clearance_m)
        if not self._walkable.any():
            raise ValueError("No native free cells remain at the declared conservative clearance")
        occupied_distance = distance_transform_edt(np.pad(blocked, 1, constant_values=True))[1:-1, 1:-1] * .1
        self._signed_grid = np.where(blocked, -occupied_distance, free_distance - half_diagonal).astype(np.float32)
        self._origins = _points(origins_xy, "spawn origins").copy()
        self._routes = _points(routes_xy, "route vertices").copy()
        if (self._origins.ndim != 2 or not len(self._origins) or self._routes.ndim != 3
                or self._routes.shape[0] != len(self._origins) or self._routes.shape[1] < 2
                or not self.is_walkable(self._origins).all() or not self.is_walkable(self._routes).all()):
            raise ValueError("Actor spawns/routes require matched [B,2]/[B,V>=2,2] conservative known-free cells")
        for value in (self._labels, self._origin, self._bounds, self._walkable, self._signed_grid, self._origins, self._routes):
            value.setflags(write=False)
        self._sdf_tensors = {}
        self.global_path_failures = 0
        self.successful_start_recoveries = 0
        self.last_start_recovery_segments = []

    def tensor(self, value):
        import torch
        result = torch.as_tensor(_array(value).copy(), device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Native navigation tensors require finite coordinates")
        return result

    @property
    def origins(self):
        return self.tensor(self._origins)

    @property
    def routes(self):
        return self.tensor(self._routes)

    @property
    def bounds(self):
        return self.tensor(self._bounds)

    @property
    def description(self):
        return {**self.describe(), "origins_xy": self._origins.tolist(), "routes_xy": self._routes.tolist()}

    def world_to_pixel(self, xy):
        import numpy as np
        xy = _points(xy, "world coordinates")
        cell = np.floor((xy - self._origin) / self.resolution).astype(np.int64)
        return np.stack((self._labels.shape[0] - 1 - cell[..., 1], cell[..., 0]), -1)

    def pixel_to_world(self, pixels):
        import numpy as np
        pixels = _array(pixels)
        if pixels.ndim < 1 or pixels.shape[-1] != 2 or pixels.dtype.kind not in "iu":
            raise ValueError("Pixels require integer [...,2] ROS row/column")
        if np.any(pixels < 0) or np.any(pixels >= self._labels.shape):
            raise ValueError("Pixel outside native map; no clamping")
        grid = np.stack((pixels[..., 1], self._labels.shape[0] - 1 - pixels[..., 0]), -1)
        return self._origin + (grid + .5) * self.resolution

    def in_domain(self, xy):
        points = _points(xy, "domain coordinates")
        return ((points >= self._bounds[:2]) & (points < self._bounds[2:])).all(-1)

    def is_walkable(self, xy):
        import numpy as np
        pixels = self.world_to_pixel(xy)
        valid = ((pixels >= 0) & (pixels < self._labels.shape)).all(-1)
        result = np.zeros(valid.shape, dtype=bool)
        result[valid] = self._walkable[pixels[valid, 0], pixels[valid, 1]]
        return result

    def occupied(self, xy):
        """Conservative forbidden cells including unknown/exterior/clearance."""
        return ~self.is_walkable(xy)

    def nearest_spawn(self, xy, *, max_distance_m=5.):
        """Explicit nearest valid cell query, not physical relocation/recovery."""
        import numpy as np
        hint = _points(xy, "spawn hint")
        if hint.shape != (2,):
            raise ValueError("Nearest spawn requires one world[2] hint")
        radius = _number(max_distance_m, "nearest spawn search radius", positive=True)
        cells = np.argwhere(self._walkable)
        world = self.pixel_to_world(cells)
        distance = np.linalg.norm(world - hint, axis=-1)
        order = np.lexsort((cells[:, 1], cells[:, 0], distance))
        if distance[order[0]] > radius:
            raise ValueError("No conservative known-free spawn within explicit search radius")
        return world[order[0]].copy()

    def astar_path(self, start_xy, goal_xy):
        """8-connected cell A*, rejecting diagonal corner cutting and fallback."""
        import numpy as np
        start_xy, goal_xy = _points(start_xy, "path start"), _points(goal_xy, "path goal")
        if start_xy.shape != (2,) or goal_xy.shape != (2,):
            raise ValueError("A* endpoints require world[2]")
        if not self.is_walkable(start_xy) or not self.is_walkable(goal_xy):
            return None
        start, goal = tuple(self.world_to_pixel(start_xy)), tuple(self.world_to_pixel(goal_xy))
        costs, parent = {start: 0.}, {}
        queue = [(math.dist(start, goal), 0., start)]
        visited = set()
        height, width = self._walkable.shape
        while queue:
            _, cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                nodes = [goal]
                while nodes[-1] != start:
                    nodes.append(parent[nodes[-1]])
                centers = self.pixel_to_world(np.asarray(nodes[::-1]))
                path = np.concatenate((start_xy[None], centers, goal_xy[None]), axis=0)
                keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=-1) > 1e-12]
                result = path[keep]
                result[-1] = goal_xy  # Preserve the requested endpoint rather than its rounded cell center.
                return result
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nxt = (node[0] + dy, node[1] + dx)
                if not (0 <= nxt[0] < height and 0 <= nxt[1] < width) or not self._walkable[nxt]:
                    continue
                if dy and dx and not (self._walkable[node[0] + dy, node[1]] and self._walkable[node[0], node[1] + dx]):
                    continue
                next_cost = cost + math.hypot(dy, dx)
                if next_cost < costs.get(nxt, math.inf):
                    costs[nxt], parent[nxt] = next_cost, node
                    heapq.heappush(queue, (next_cost + math.dist(nxt, goal), next_cost, nxt))
        return None

    def _route_inputs(self, positions, route_indices, actor_ids):
        import numpy as np
        positions = _points(positions, "actual route positions")
        if positions.ndim != 2:
            raise ValueError("Actual route positions require [B,2]")
        batch = len(positions)
        indices = _array(route_indices)
        ids = np.arange(batch) if actor_ids is None else _array(actor_ids)
        for value, name, limit in ((indices, "route indices", self._routes.shape[1]), (ids, "actor IDs", len(self._origins))):
            if value.shape != (batch,) or value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= limit):
                raise ValueError(f"Integer {name}[B] must stay within the declared route/actor range")
        return positions, indices.astype(np.int64), ids.astype(np.int64)

    def _segment_walkable(self, start, end):
        """Exact closed-cell slab test; blocked corner touches are rejected."""
        import numpy as np
        if not self.is_walkable(start) or not self.is_walkable(end):
            return False
        pixels = self.world_to_pixel(np.stack((start, end)))
        low, high = pixels.min(0), pixels.max(0)
        blocked = np.argwhere(~self._walkable[low[0]:high[0] + 1, low[1]:high[1] + 1]) + low
        if not len(blocked):
            return True
        centers = self.pixel_to_world(blocked)
        lower, upper = centers - .05, centers + .05
        enter, leave = np.zeros(len(blocked)), np.ones(len(blocked))
        delta = end - start
        for axis in (0, 1):
            if delta[axis] == 0:
                outside = (start[axis] < lower[:, axis]) | (start[axis] > upper[:, axis])
                enter[outside] = 2.
            else:
                a, b = (lower[:, axis] - start[axis]) / delta[axis], (upper[:, axis] - start[axis]) / delta[axis]
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        return not np.any(enter <= leave + 1e-12)

    def _progress(self, positions, indices, ids, advance):
        import numpy as np
        if type(advance) is not bool:
            raise ValueError("Route advance must be an explicit boolean")
        reached = np.linalg.norm(self._routes[ids, indices] - positions, axis=-1) <= self.arrival_radius
        reached &= advance
        # Nearby targets across thin walls/unknown cells are not actual arrivals.
        for row in np.flatnonzero(reached):
            reached[row] = self._segment_walkable(positions[row], self._routes[ids[row], indices[row]])
        return (indices + reached) % self._routes.shape[1], reached

    def advance_route(self, positions, route_indices, *, actor_ids=None):
        """Call only on actual nonterminal post-control positions, before reset."""
        import torch
        positions, indices, ids = self._route_inputs(positions, route_indices, actor_ids)
        updated, reached = self._progress(positions, indices, ids, True)
        return torch.as_tensor(updated, device=self.device), torch.as_tensor(reached, device=self.device)

    def route_waypoints(self, positions, route_indices, *, actor_ids=None, advance=True,
                        times=(.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.)):
        import numpy as np
        import torch
        positions, indices, ids = self._route_inputs(positions, route_indices, actor_ids)
        raw_times = _array(times)
        if raw_times.dtype.kind not in "fiu":
            raise ValueError("Waypoint times require real seconds, not booleans/complex values")
        times = np.asarray(raw_times, dtype=np.float64)
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(times <= 0) or np.any(times > 5) or np.any(np.diff(times) <= 0)):
            raise ValueError("Waypoint times require finite strictly increasing seconds in (0,5]")
        updated, reached = self._progress(positions, indices, ids, advance)
        points = []
        self.last_start_recovery_segments = []
        for position, destination in zip(positions, self._routes[ids, updated]):
            path = self.astar_path(position, destination)
            if path is None:
                self.global_path_failures += 1
                points.append(np.repeat(position[None], len(times), axis=0))
            else:
                cumulative = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=-1).cumsum()]
                distances = np.minimum(self.lookahead * times / 5, cumulative[-1])
                points.append(np.stack([np.interp(distances, cumulative, path[:, axis]) for axis in range(2)], -1))
        output = np.asarray(points).reshape(len(positions), len(times), 2)
        return self.tensor(output), torch.as_tensor(updated, device=self.device), torch.as_tensor(reached, device=self.device)

    def route_goal(self, positions, route_indices, *, actor_ids=None, advance=True):
        points, updated, reached = self.route_waypoints(positions, route_indices, actor_ids=actor_ids, advance=advance, times=(5.,))
        return points[:, 0], updated, reached

    def raster(self, states, **kwargs):
        if kwargs:
            raise ValueError("Native raster keeps unknown/outside=.5; no override")
        return self.provider.raster(states, device=self.device)

    def ground_height_at(self, xy):
        return self.provider.ground_height_at(xy)

    def height_at(self, xy):
        return self.provider.height_at(xy)

    def signed_distance(self, world_xy):
        """Differentiable bilinear grid SDF with explicit forbidden exterior.

        Positive inside known-free cells, negative in occupied/unknown cells.
        Bilinear interpolation is a guidance proxy; A* uses the separate
        conservative whole-cell mask. Coordinates themselves are not clipped.
        """
        import torch
        from torch.nn import functional as functional
        if (not isinstance(world_xy, torch.Tensor) or world_xy.ndim < 1 or world_xy.shape[-1] != 2 or not world_xy.numel()
                or world_xy.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(world_xy).all())
                or bool((world_xy.abs() > 1e6).any())):
            raise ValueError("Grid SDF requires finite FP32/64 world[...,2] within safe range; no trajectory clipping")
        key = (str(world_xy.device), world_xy.dtype)
        if key not in self._sdf_tensors:
            self._sdf_tensors[key] = torch.as_tensor(self._signed_grid.copy(), dtype=world_xy.dtype, device=world_xy.device)[None, None]
        height, width = self._labels.shape
        origin = world_xy.new_tensor(self._origin)
        cell = (world_xy - origin) / .1 - .5
        # align_corners=False maps cell-center0 to -1+1/size, including size1.
        grid = torch.stack((2 * (cell[..., 0] + .5) / width - 1,
                            1 - 2 * (cell[..., 1] + .5) / height), -1)
        sampled = functional.grid_sample(self._sdf_tensors[key], grid.reshape(1, 1, -1, 2),
            mode="bilinear", padding_mode="border", align_corners=False).reshape(world_xy.shape[:-1])
        bounds = world_xy.new_tensor(self._bounds)
        q = (world_xy - (bounds[:2] + bounds[2:]) / 2).abs() - (bounds[2:] - bounds[:2]) / 2
        domain = -(torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0))
        return torch.minimum(sampled, domain)

    def collision_cost(self, world_xy, radius=.5):
        """Callback [B,S,T,2] -> per-candidate [B,S] soft clearance deficit.

        A future clean-guidance hook must call this within its autograd graph
        on world-frame predictions and apply its explicit weight. Existing
        The separately owned planner callback hook can consume this function;
        this component does not install a world/driver hook. No AABB conversion
        occurs, and the caller records actual hook use separately.
        """
        radius = _number(radius, "grid SDF guidance radius")
        if getattr(world_xy, "ndim", None) != 4 or any(size < 1 for size in world_xy.shape[:3]):
            raise ValueError("Grid collision guidance requires world[B,S,T>=1,2]")
        return (radius - self.signed_distance(world_xy)).clamp_min(0).mean(-1)

    def describe(self):
        return {"schema": "dynbench_native_grid_navigation.v1", "status": "prepared_not_runtime_integrated",
            "status_scope": "component CPU readiness; caller records actual hook use/physics validation separately",
            "scene": self.provider.scene_name, "admission": deepcopy(self._admission),
            "provider": self.provider.describe(), "resolution_m": .1,
            "world_bounds_xy_m": self._bounds.tolist(), "origin_xy_m": self._origin.tolist(),
            "ROS_export_layout": "row0=maxY,column0=minX; cell-center world coordinates; no dynamic layout fitting",
            "negative_world_XY_preserved": True, "unknown_and_outside_blocked": True,
            "clearance_radius_m": self.clearance_m,
            "clearance_method": "EDT center distance minus two cell half-diagonals; padded blocked exterior; full candidate cell certified",
            "astar": "8-connected, no diagonal corner cutting, no smoothing, no blocked-start recovery",
            "route_goal_lookahead_m": self.lookahead, "route_arrival_radius_m": self.arrival_radius,
            "global_path_failures": self.global_path_failures, "successful_start_recoveries": 0,
            "failure_behavior": "hold actual finite root as explicit failed query; never count navigation success",
            "SDF": "lazy Torch bilinear signed-grid proxy plus analytic forbidden exterior; soft guidance only",
            "SDF_clean_guidance_hook_installed_by_this_component": False,
            "SDF_clean_guidance_callback_available": True, "grid_as_PhysX_cuboids": False,
            "physical_geometry_created": False, "App_created": False, "SimulationContext_created": False,
            "human_navigation_quality_validated": False,
            "limitations": ["Native resource admission is a static snapshot, not new provider cooking/navigation validation.",
                "A conservative .1m map is not exact full-body swept-volume or contact clearance.",
                "A* supplies global waypoints only; it does not replace diffusion output or move physical actors.",
                "Grid SDF needs the separately owned explicit planner callback hook; AABB geometry cannot substitute for measured occupancy."]}
