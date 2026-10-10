"""Independent NumPy social raster; no simulator, model, or capture integration."""
from __future__ import annotations

import numpy as np

RESOLUTION = 64
EXTENT_M = 4.0
ACTOR_SIZE_XY_M = (0.5, 1.0)  # Full sizes in the actor's root-heading X/Y frame.
ACTOR_HEIGHT_M = 1.7
TASK_DIM = 20 + RESOLUTION * RESOLUTION * 3


def _numbers(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"{name} requires finite real values with shape {shape}")
    return array.astype(np.float64, copy=False)


def _integers(value, shape, name, *, minimum=0):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "iu"
            or np.any(array < minimum) or np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} requires integer values >= {minimum} with shape {shape}")
    return array.astype(np.int64, copy=False)


def make_grid(grid_center_xy, grid_heading):
    """World XY[E,64,64,2], ij=X/Y; inclusive -2..2m sample locations.

    The sampling interval is 4/63m. Caller-selected grid heading rotates the spatial grid only.
    Caller-sampled static heights must correspond to these exact locations.
    """
    heads = np.asarray(grid_center_xy)
    if heads.ndim != 2 or heads.shape[1] != 2 or not len(heads):
        raise ValueError("grid_center_xy requires nonempty [E,2]")
    heads = _numbers(heads, heads.shape, "grid_center_xy")
    yaw = _numbers(grid_heading, (len(heads),), "grid_heading")
    axis = np.linspace(-EXTENT_M / 2, EXTENT_M / 2, RESOLUTION, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    cosine, sine = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
    with np.errstate(over="raise", invalid="raise"):
        try:
            return np.stack((cosine*x - sine*y + heads[:, None, None, 0],
                             sine*x + cosine*y + heads[:, None, None, 1]), axis=-1)
        except FloatingPointError as error:
            raise ValueError("Grid arithmetic overflowed finite coordinates") from error


def render(roots_xy, root_headings, world_velocities, actor_ids, scene_ids,
           ego_rows, grid_center_xy, grid_heading, static_heights, height_reference_world_z,
           *, include_ego):
    """Return (float32 raster[E,64,64,3], int64 occupant_ids[E,64,64]).

    Complete actor rows are scene-scoped. Occupant rectangles have full X/Y
    sizes0.5/1m, root-heading orientation, and inclusive boundaries. Among
    overlapping rectangles nearest query-point/root distance wins; exact ties
    choose the smaller stable actor ID. Background ID=-1, world velocity=0.
    Occupied absolute top = caller's sampled static Z +1.7m, once per pixel.
    Channels are caller height reference minus top, and occupant minus ego world
    velocity rotated into ego ROOT-heading X/Y. All three clip[-3,3]*5.
    No input is mutated; this function draws no random numbers.
    """
    if type(include_ego) is not bool:
        raise ValueError("include_ego must be an explicit bool")
    roots = np.asarray(roots_xy)
    if roots.ndim != 2 or roots.shape[1] != 2 or not len(roots):
        raise ValueError("roots_xy requires nonempty [N,2]")
    count = len(roots)
    roots = _numbers(roots, (count, 2), "roots_xy")
    heading = _numbers(root_headings, (count,), "root_headings")
    velocity = _numbers(world_velocities, (count, 2), "world_velocities")
    ids = _integers(actor_ids, (count,), "actor_ids")
    scenes = _integers(scene_ids, (count,), "scene_ids")
    if len(np.unique(ids)) != count:
        raise ValueError("actor_ids must be globally unique stable IDs")
    selected = np.asarray(ego_rows)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("ego_rows requires a nonempty one-dimensional selection")
    selected = _integers(selected, selected.shape, "ego_rows")
    if np.any(selected >= count) or len(np.unique(selected)) != len(selected):
        raise ValueError("ego_rows must be unique valid actor row indices")
    size = len(selected)
    grid = make_grid(_numbers(grid_center_xy, (size, 2), "grid_center_xy"),
                     _numbers(grid_heading, (size,), "grid_heading"))
    static = _numbers(static_heights, (size, RESOLUTION, RESOLUTION), "static_heights")
    ground = _numbers(height_reference_world_z, (size,), "height_reference_world_z")
    occupants = np.full((size, RESOLUTION, RESOLUTION), -1, dtype=np.int64)
    occupant_velocity = np.zeros((size, RESOLUTION, RESOLUTION, 2), dtype=np.float64)
    half_x, half_y = np.asarray(ACTOR_SIZE_XY_M) / 2
    with np.errstate(over="raise", invalid="raise"):
        try:
            for output_row, ego in enumerate(selected):
                peers = np.flatnonzero(scenes == scenes[ego])
                if not include_ego:
                    peers = peers[peers != ego]
                # Sorted IDs plus strict distance comparison resolve exact ties.
                peers = peers[np.argsort(ids[peers], kind="stable")]
                nearest = np.full((RESOLUTION, RESOLUTION), np.inf)
                for actor in peers:
                    delta = grid[output_row] - roots[actor]
                    cosine, sine = np.cos(heading[actor]), np.sin(heading[actor])
                    local_x = cosine*delta[..., 0] + sine*delta[..., 1]
                    local_y = -sine*delta[..., 0] + cosine*delta[..., 1]
                    inside = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
                    distance = np.hypot(delta[..., 0], delta[..., 1])
                    wins = inside & (distance < nearest)
                    nearest[wins] = distance[wins]
                    occupants[output_row, wins] = ids[actor]
                    occupant_velocity[output_row, wins] = velocity[actor]
            top = static + (occupants >= 0)*ACTOR_HEIGHT_M
            height = ground[:, None, None] - top
            relative = occupant_velocity - velocity[selected, None, None, :]
            cosine = np.cos(heading[selected])[:, None, None]
            sine = np.sin(heading[selected])[:, None, None]
            local_vx = cosine*relative[..., 0] + sine*relative[..., 1]
            local_vy = -sine*relative[..., 0] + cosine*relative[..., 1]
            raster = np.stack((height, local_vx, local_vy), axis=-1)
        except FloatingPointError as error:
            raise ValueError("Raster arithmetic overflowed finite inputs") from error
    return (np.clip(raster, -3.0, 3.0)*5.0).astype(np.float32), occupants


def mirror(raster, occupant_ids):
    """Reflect local Y: flip axis2, negate Vy(channel2), preserve Vx and IDs."""
    values = np.asarray(raster)
    if (values.ndim != 4 or values.shape[1:] != (RESOLUTION, RESOLUTION, 3)
            or not len(values) or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(values).all()):
        raise ValueError("raster requires finite float32/64 [E,64,64,3]")
    ids = _integers(occupant_ids, values.shape[:-1], "occupant_ids", minimum=-1)
    reflected = values[:, :, ::-1].copy()
    reflected[..., 2] *= -1
    return reflected, ids[:, :, ::-1].copy()
