"""Independent raw social-task assembly. No policy, normalizer or simulator."""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from types import ModuleType

import torch

TASK_DIM = 12308
RASTER_SHA256 = "234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe"
_RASTER = None


def _backend():
    """Verify the one readonly dependency's complete bytes once, then load them."""
    global _RASTER
    if _RASTER is None:
        path = Path(__file__).parent.parent / "social_raster_torch_staging_v1/social_raster_torch.py"
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
            raise ValueError("Raster dependency must be an ordinary readonly single-link file")
        source = path.read_bytes()
        if len(source) != 9125 or hashlib.sha256(source).hexdigest() != RASTER_SHA256:
            raise ValueError("Frozen raster dependency byte identity mismatch")
        module = ModuleType("_frozen_social_raster_for_task")
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        _RASTER = module
    return _RASTER


def _float(value, shape, name, device, *, dtype=None):
    backend = _backend()
    value = backend._tensor(value, shape, name, device)
    if value.dtype not in (torch.float32, torch.float64) or (dtype is not None and value.dtype != dtype):
        raise ValueError(name + " requires float32/64 and matching physical-source dtype")
    backend._finite(value)
    return value


def heading_from_pelvis_xyzw(quaternion, *, device):
    """Projected world +X heading; unit XYZW tolerance1e-3, singular norm<=1e-12 rejected."""
    target = _backend()._device(device)
    if not isinstance(quaternion, torch.Tensor) or quaternion.ndim != 2 or quaternion.shape[1] != 4 or not len(quaternion):
        raise ValueError("Pelvis quaternion requires nonempty Tensor[E,4]")
    q = _float(quaternion, quaternion.shape, "Pelvis XYZW", target)
    norm = q.norm(dim=-1)
    if not bool(torch.isfinite(norm).all()) or not torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0):
        raise ValueError("Pelvis quaternion must be unit XYZW; no implicit renormalization")
    x, y, z, w = q.unbind(-1)
    # Same projected +X rotation equation as rotations.calc_heading/quat_rotate.
    direction_x = 2*w.square()-1+2*x.square()
    direction_y = 2*w*z+2*x*y
    if bool((torch.hypot(direction_x, direction_y) <= 1e-12).any()):
        raise ValueError("Pelvis projected-X heading is undefined")
    return torch.atan2(direction_y, direction_x)


def trajectory_prefix(trajectory_world_xy, ego_roots_xy, task_heading, trajectory_offsets_s, *, device):
    """Ten same-observation-time future XY samples (+.5..5s), inverse heading, metres."""
    backend = _backend()
    target = backend._device(device)
    if not isinstance(ego_roots_xy, torch.Tensor) or ego_roots_xy.ndim != 2 or ego_roots_xy.shape[1] != 2 or not len(ego_roots_xy):
        raise ValueError("ego_roots_xy requires nonempty Tensor[E,2]")
    size = len(ego_roots_xy)
    roots = _float(ego_roots_xy, (size, 2), "ego_roots_xy", target)
    trajectory = _float(trajectory_world_xy, (size, 10, 2), "trajectory_world_xy", target, dtype=roots.dtype)
    heading = _float(task_heading, (size,), "task_heading", target, dtype=roots.dtype)
    offsets = _float(trajectory_offsets_s, (10,), "trajectory_offsets_s", target)
    expected = torch.arange(1, 11, dtype=offsets.dtype, device=target)/2
    if not torch.equal(offsets, expected):
        raise ValueError("Future offsets must be exactly +0.5,+1,...,+5 seconds")
    delta = trajectory.to(torch.float64)-roots.to(torch.float64)[:, None]
    cosine, sine = heading.to(torch.float64).cos()[:, None], heading.to(torch.float64).sin()[:, None]
    local = torch.stack((cosine*delta[..., 0]+sine*delta[..., 1], -sine*delta[..., 0]+cosine*delta[..., 1]), -1)
    backend._finite(delta, local)
    result = local.flatten(1).to(torch.float32)
    backend._finite(result)
    return result


def assemble_raw_task(roots_xy, root_headings, world_velocities, actor_ids, scene_ids, ego_rows,
                      trajectory_world_xy, task_heading, trajectory_offsets_s,
                      grid_center_xy, grid_heading, height_reference_world_z, terrain_sampler,
                      *, include_ego, device, ego_chunk_size=16):
    """Return float32[E,12308], owners[E,64,64], in ego_rows order.

    Sampler is called once with world XY[E,64,64,2] float64 on device and must
    return absolute static Z[E,64,64] there, without changing its query tensor.
    All poses/trajectories share one source float dtype. Grid and task headings
    are independent explicit contracts. Normalization belongs to the caller.
    """
    backend = _backend()
    target = backend._device(device)
    if type(include_ego) is not bool or type(ego_chunk_size) is not int or ego_chunk_size < 1 or not callable(terrain_sampler):
        raise ValueError("Explicit self choice, positive chunk size and terrain callback required")
    if not isinstance(roots_xy, torch.Tensor) or roots_xy.ndim != 2 or roots_xy.shape[1] != 2 or not len(roots_xy):
        raise ValueError("roots_xy requires nonempty Tensor[N,2]")
    count = len(roots_xy)
    roots = _float(roots_xy, (count, 2), "roots_xy", target)
    headings = _float(root_headings, (count,), "root_headings", target, dtype=roots.dtype)
    velocities = _float(world_velocities, (count, 2), "world_velocities", target, dtype=roots.dtype)
    ids = backend._integers(actor_ids, (count,), "actor_ids", target)
    scenes = backend._integers(scene_ids, (count,), "scene_ids", target)
    if torch.unique(ids).numel() != count:
        raise ValueError("Globally unique stable actor IDs required")
    if not isinstance(ego_rows, torch.Tensor) or ego_rows.ndim != 1 or not len(ego_rows):
        raise ValueError("ego_rows requires nonempty Tensor[E]")
    egos = backend._integers(ego_rows, ego_rows.shape, "ego_rows", target)
    if bool((egos >= count).any()) or torch.unique(egos).numel() != len(egos):
        raise ValueError("ego_rows must be unique valid actor rows")
    size = len(egos)
    center = _float(grid_center_xy, (size, 2), "grid_center_xy", target, dtype=roots.dtype)
    yaw = _float(grid_heading, (size,), "grid_heading", target, dtype=roots.dtype)
    reference = _float(height_reference_world_z, (size,), "height_reference_world_z", target, dtype=roots.dtype)
    prefix = trajectory_prefix(trajectory_world_xy, roots.index_select(0, egos), task_heading, trajectory_offsets_s, device=target)
    grid = backend.make_grid(center, yaw, device=target)
    before = grid.clone()
    heights = terrain_sampler(grid)
    if not torch.equal(grid, before):
        raise ValueError("Terrain callback changed its query grid")
    heights = _float(heights, (size, 64, 64), "sampled absolute static Z", target)
    raster, owners = backend.render(roots, headings, velocities, ids, scenes, egos, center, yaw, heights, reference,
                                    include_ego=include_ego, device=target, ego_chunk_size=ego_chunk_size)
    return torch.cat((prefix, raster.flatten(1)), -1), owners


def _raw_task(value, device):
    target = _backend()._device(device)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != TASK_DIM or not len(value):
        raise ValueError("raw_task requires nonempty Tensor[E,12308]")
    return _float(value, value.shape, "raw_task", target, dtype=torch.float32), target


def mirror_raw_task(raw_task, *, device):
    """New raw task: trajectoryY negated; HWC gridY flipped and Vy negated."""
    task, _ = _raw_task(raw_task, device)
    prefix = task[:, :20].reshape(-1, 10, 2).clone()
    prefix[..., 1].neg_()
    raster = task[:, 20:].reshape(-1, 64, 64, 3).flip((2,))
    raster[..., 2].neg_()
    return torch.cat((prefix.flatten(1), raster.flatten(1)), -1)


def replace_candidate_trajectory(raw_task, candidate_world_xy, ego_roots_xy, task_heading, trajectory_offsets_s, *, device):
    """Replace only20 trajectory values; current body and raster stay fixed outside."""
    task, target = _raw_task(raw_task, device)
    prefix = trajectory_prefix(candidate_world_xy, ego_roots_xy, task_heading, trajectory_offsets_s, device=target)
    if len(prefix) != len(task):
        raise ValueError("Candidate and current task batch mismatch")
    return torch.cat((prefix, task[:, 20:]), -1)
