"""Optional directed A* waypoint guidance for reciprocal corridor traffic.

This authored traffic convention only changes global waypoints. TRACE still
samples and admits every executable trajectory; AMP produces physical actions.
Costs use the SAME inflated occupancy grid as the original static planner.
"""

import heapq
import math

import numpy as np


class KeepRightPathPlanner:
    def __init__(self, base, *, padding_distance_m=.5, weight=2., corridor_width_m=4.):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (padding_distance_m, weight, corridor_width_m)):
            raise ValueError("Positive finite lane distance, weight and corridor width required")
        self.map_dialate = base.map_dialate
        self.map_size = base.map_size
        self.resolution = base.map_resolution
        self.distance = padding_distance_m
        self.weight = weight
        self.corridor_width = corridor_width_m
        self.directions = tuple(base._build_directions(1))
        self._cache = {}
        rows, cols = np.indices(self.map_size)
        blocked = self.map_dialate != 0
        self._right_distance = []
        self._opposite = [self.directions.index((-dy, -dx)) for dy, dx in self.directions]
        # A finite ray is sufficient for a bounded preference; it does not
        # certify visibility or physical clearance beyond the occupancy grid.
        max_distance = max(6., padding_distance_m * 2)
        for dy, dx in self.directions:
            right_y, right_x = -dx, dy
            step_length = math.hypot(right_y, right_x) * self.resolution
            maximum = math.ceil(max_distance / step_length)
            distance = np.full(self.map_size, maximum * step_length)
            unresolved = ~blocked
            for step in range(1, maximum + 1):
                y, x = rows + right_y * step, cols + right_x * step
                inside = (y >= 0) & (y < self.map_size[0]) & (x >= 0) & (x < self.map_size[1])
                hit = ~inside | blocked[np.clip(y, 0, self.map_size[0]-1), np.clip(x, 0, self.map_size[1]-1)]
                first = unresolved & hit
                distance[first] = step * step_length
                unresolved[first] = False
            self._right_distance.append(distance)

    def get_astar_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        for node in (start, goal):
            if len(node) != 2 or any(type(v) not in (int, np.int32, np.int64) for v in node):
                raise ValueError("Integer grid coordinates required")
            if any(v < 0 or v >= limit for v, limit in zip(node, self.map_size)):
                return None
            if self.map_dialate[node] != 0:
                return None
        key = (start, goal)
        if key in self._cache:
            path = self._cache[key]
            return None if path is None else path.copy()
        costs = np.full(self.map_size, np.inf)
        costs[start] = 0.
        parents = {}
        queue = [(math.dist(start, goal) * self.resolution, 0., start)]
        while queue:
            _, accumulated, current = heapq.heappop(queue)
            if accumulated > costs[current]:
                continue
            if current == goal:
                break
            for index, (direction, right_distance) in enumerate(zip(self.directions, self._right_distance)):
                dy, dx = direction
                neighbor = current[0] + dy, current[1] + dx
                if any(v < 0 or v >= limit for v, limit in zip(neighbor, self.map_size)):
                    continue
                if self.map_dialate[neighbor] != 0:
                    continue
                # Unlike point-only diagonal moves, both corner cells must be
                # free. No smoothing or lane offset can cut occupied corners.
                if dy and dx and (self.map_dialate[current[0]+dy, current[1]] != 0
                                  or self.map_dialate[current[0], current[1]+dx] != 0):
                    continue
                right = .5 * (right_distance[current] + right_distance[neighbor])
                left_distance = self._right_distance[self._opposite[index]]
                left = .5 * (left_distance[current] + left_distance[neighbor])
                # Lane convention applies only to bounded narrow cross-sections.
                # Penalising distance from a wall in open space can turn a12m
                # reciprocal corridor query into a40m clockwise detour.
                penalty = (min(abs(right - self.distance) / self.distance, 6.)
                           if left + right <= self.corridor_width else 0.)
                candidate = accumulated + math.hypot(dy, dx) * self.resolution * (1. + self.weight * penalty)
                if candidate < costs[neighbor]:
                    costs[neighbor] = candidate
                    parents[neighbor] = current
                    heapq.heappush(queue, (candidate + math.dist(neighbor, goal) * self.resolution,
                                          candidate, neighbor))
        path = None
        if math.isfinite(costs[goal]):
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            path = np.asarray(path[::-1], dtype=np.int64)
        if len(self._cache) >= 512:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = path
        return None if path is None else path.copy()
