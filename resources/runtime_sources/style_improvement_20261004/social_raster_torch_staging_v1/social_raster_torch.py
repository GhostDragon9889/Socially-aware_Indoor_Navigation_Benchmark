"""Pure Tensor implementation of the frozen independent NumPy raster contract."""
from __future__ import annotations

import torch

RESOLUTION = 64
EXTENT_M = 4.0
ACTOR_SIZE_XY_M = (0.5, 1.0)
ACTOR_HEIGHT_M = 1.7
TASK_DIM = 20 + RESOLUTION * RESOLUTION * 3
NUMPY_CONTRACT_SHA = "ff9927e33935318b7e3cafa8cda919a7c8aa2dc77014aeb356bcfc5c405e33c8"
_SIGNED = {torch.int8, torch.int16, torch.int32, torch.int64}
_UNSIGNED = {getattr(torch, name) for name in ("uint8", "uint16", "uint32", "uint64") if hasattr(torch, name)}
_REAL = _SIGNED | {torch.float16, torch.float32, torch.float64}


def _device(device):
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Explicit CPU or indexed CUDA device required") from error
    if target.type == "cpu":
        return torch.device("cpu")
    if target.type != "cuda" or target.index is None:
        raise ValueError("Use cpu or an explicit CUDA index, e.g. cuda:0")
    return target


def _tensor(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.device != device:
        raise ValueError(f"{name} requires Tensor{tuple(shape)} on {device}; no implicit device transfer")
    return value


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Nonfinite input or arithmetic overflow")


def _numbers(value, shape, name, device):
    value = _tensor(value, shape, name, device)
    if value.dtype not in _REAL:
        raise ValueError(name + " requires real floating or signed integer values")
    result = value.to(dtype=torch.float64)
    _finite(result)
    return result


def _integers(value, shape, name, device, *, minimum=0):
    value = _tensor(value, shape, name, device)
    if value.dtype not in _SIGNED | _UNSIGNED:
        raise ValueError(name + " requires integer values")
    result = value.to(dtype=torch.int64)
    if (value.dtype in _UNSIGNED and bool((result < 0).any())) or bool((result < minimum).any()):
        raise ValueError(name + " requires representable int64 values above its minimum")
    return result


def make_grid(grid_center_xy, grid_heading, *, device):
    """FP64 world XY[E,64,64,2], ij=X/Y; explicit center/heading, inclusive±2m."""
    target = _device(device)
    if not isinstance(grid_center_xy, torch.Tensor) or grid_center_xy.ndim != 2 or grid_center_xy.shape[1] != 2 or not len(grid_center_xy):
        raise ValueError("grid_center_xy requires nonempty Tensor[E,2]")
    center = _numbers(grid_center_xy, grid_center_xy.shape, "grid_center_xy", target)
    heading = _numbers(grid_heading, (len(center),), "grid_heading", target)
    # Match NumPy linspace's multiply-then-add order; force the exact endpoint.
    axis = torch.arange(RESOLUTION, dtype=torch.float64, device=target) * (EXTENT_M/(RESOLUTION-1)) - EXTENT_M/2
    axis[-1] = EXTENT_M/2
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    cosine, sine = heading.cos()[:, None, None], heading.sin()[:, None, None]
    grid = torch.stack((cosine*x-sine*y+center[:, None, None, 0], sine*x+cosine*y+center[:, None, None, 1]), -1)
    _finite(grid)
    return grid


def render(roots_xy, root_headings, world_velocities, actor_ids, scene_ids, ego_rows,
           grid_center_xy, grid_heading, static_heights, height_reference_world_z,
           *, include_ego, device, ego_chunk_size=16):
    """Return float32[E,64,64,3] and int64 owners on the declared input device.

    Geometry is FP64, matching NumPy's promotion of supplied numbers. Stream
    one actor into a scene-local ego chunk: no N×N×64² allocation. Exact ties
    use smallest stable actor ID; no tolerance expands rectangle boundaries.
    All inputs remain unchanged. No model, RNG, SDK, or NumPy is used here.
    """
    if type(include_ego) is not bool or type(ego_chunk_size) is not int or ego_chunk_size < 1:
        raise ValueError("Explicit include_ego bool and positive integer ego_chunk_size required")
    target = _device(device)
    if not isinstance(roots_xy, torch.Tensor) or roots_xy.ndim != 2 or roots_xy.shape[1] != 2 or not len(roots_xy):
        raise ValueError("roots_xy requires nonempty Tensor[N,2]")
    count = len(roots_xy)
    roots = _numbers(roots_xy, (count, 2), "roots_xy", target)
    heading = _numbers(root_headings, (count,), "root_headings", target)
    velocity = _numbers(world_velocities, (count, 2), "world_velocities", target)
    ids = _integers(actor_ids, (count,), "actor_ids", target)
    scenes = _integers(scene_ids, (count,), "scene_ids", target)
    if torch.unique(ids).numel() != count:
        raise ValueError("actor_ids must be globally unique stable IDs")
    if not isinstance(ego_rows, torch.Tensor) or ego_rows.ndim != 1 or not len(ego_rows):
        raise ValueError("ego_rows requires nonempty Tensor[E]")
    selected = _integers(ego_rows, ego_rows.shape, "ego_rows", target)
    if bool((selected >= count).any()) or torch.unique(selected).numel() != len(selected):
        raise ValueError("ego_rows must be unique valid actor row indices")
    size = len(selected)
    center = _numbers(grid_center_xy, (size, 2), "grid_center_xy", target)
    grid_yaw = _numbers(grid_heading, (size,), "grid_heading", target)
    static = _numbers(static_heights, (size, RESOLUTION, RESOLUTION), "static_heights", target)
    reference_z = _numbers(height_reference_world_z, (size,), "height_reference_world_z", target)
    raster = torch.empty((size, RESOLUTION, RESOLUTION, 3), dtype=torch.float32, device=target)
    owners = torch.empty((size, RESOLUTION, RESOLUTION), dtype=torch.int64, device=target)
    ego_scenes = scenes.index_select(0, selected)
    for scene in torch.unique(ego_scenes).unbind():
        output_rows = torch.nonzero(ego_scenes == scene, as_tuple=False).flatten()
        peers = torch.nonzero(scenes == scene, as_tuple=False).flatten()
        peers = peers.index_select(0, torch.argsort(ids.index_select(0, peers), stable=True))
        for rows in output_rows.split(ego_chunk_size):
            egos = selected.index_select(0, rows)
            grid = make_grid(center.index_select(0, rows), grid_yaw.index_select(0, rows), device=target)
            shape = (len(rows), RESOLUTION, RESOLUTION)
            owner = torch.full(shape, -1, dtype=torch.int64, device=target)
            nearest = torch.full(shape, torch.inf, dtype=torch.float64, device=target)
            occupant_velocity = torch.zeros((*shape, 2), dtype=torch.float64, device=target)
            for actor in peers.split(1):  # One-dimensional indices avoid scalar actor→Python conversion.
                root = roots.index_select(0, actor)[0]
                yaw = heading.index_select(0, actor)[0]
                delta = grid-root
                local_x = yaw.cos()*delta[..., 0]+yaw.sin()*delta[..., 1]
                local_y = -yaw.sin()*delta[..., 0]+yaw.cos()*delta[..., 1]
                distance = torch.hypot(delta[..., 0], delta[..., 1])
                _finite(delta, local_x, local_y, distance)
                wins = (local_x.abs() <= ACTOR_SIZE_XY_M[0]/2) & (local_y.abs() <= ACTOR_SIZE_XY_M[1]/2) & (distance < nearest)
                if not include_ego:
                    wins = wins & (egos[:, None, None] != actor)
                nearest = torch.where(wins, distance, nearest)
                owner = torch.where(wins, ids.index_select(0, actor)[0], owner)
                occupant_velocity = torch.where(wins[..., None], velocity.index_select(0, actor)[0], occupant_velocity)
            top = static.index_select(0, rows)+(owner >= 0).to(dtype=torch.float64)*ACTOR_HEIGHT_M
            height = reference_z.index_select(0, rows)[:, None, None]-top
            relative = occupant_velocity-velocity.index_select(0, egos)[:, None, None, :]
            cosine, sine = heading.index_select(0, egos).cos()[:, None, None], heading.index_select(0, egos).sin()[:, None, None]
            vx, vy = cosine*relative[..., 0]+sine*relative[..., 1], -sine*relative[..., 0]+cosine*relative[..., 1]
            _finite(top, height, relative, vx, vy)
            values = (torch.stack((height, vx, vy), -1).clamp(-3., 3.)*5.).to(dtype=torch.float32)
            raster.index_copy_(0, rows, values)
            owners.index_copy_(0, rows, owner)
    return raster, owners


def mirror(raster, occupant_ids, *, device):
    """Flip localY/axis2, negate Vy only, preserve floating dtype and stable IDs."""
    target = _device(device)
    if not isinstance(raster, torch.Tensor) or raster.ndim != 4 or raster.shape[1:] != (RESOLUTION, RESOLUTION, 3) or not len(raster):
        raise ValueError("raster requires nonempty Tensor[E,64,64,3]")
    values = _tensor(raster, raster.shape, "raster", target)
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError("raster requires float32 or float64")
    _finite(values)
    ids = _integers(occupant_ids, values.shape[:-1], "occupant_ids", target, minimum=-1)
    reflected = values.flip((2,))
    reflected[..., 2].neg_()
    return reflected, ids.flip((2,))
