"""Rasterize the existing navigation occupancy map in the exact ORCA convention."""

import numpy as np
import torch


def rasterize_navigation_map(task, states, *, device="cpu", outside_value=0.5):
    """Return [B,2,224,224] walkable/obstacle; positive y increases raster row.

    Local metres map to (col,row)=(12*x+56,12*y+112), matching TRACE's
    ORCA config and trajdata parser. World→occupancy mapping is delegated to
    NavigationTask, so map origin rotation and cell boundaries remain intact.
    Outside the occupancy domain uses the checkpoint's unknown fill value.
    """
    if not np.isfinite(outside_value) or not 0 <= outside_value <= 1:
        raise ValueError("outside_value must be within [0,1]")
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] not in (4, 5) or not np.isfinite(states).all():
        raise ValueError("states must be finite world[x,y,speed,yaw] [B,4] or [x,y,vx,vy,yaw] [B,5]")
    row, col = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
    local_x, local_y = (col - 56) / 12, (row - 112) / 12
    cosine, sine = np.cos(states[:, -1, None, None]), np.sin(states[:, -1, None, None])
    world = np.stack((states[:, 0, None, None] + cosine * local_x - sine * local_y,
                      states[:, 1, None, None] + sine * local_x + cosine * local_y), axis=-1)
    pixels = np.asarray(task.world_to_pixel(world))
    py, px = pixels[..., 0], pixels[..., 1]
    valid = (py >= 0) & (py < task.free_mask.shape[0]) & (px >= 0) & (px < task.free_mask.shape[1])
    free = np.full(valid.shape, outside_value, dtype=np.float32)
    free[valid] = task.free_mask[py[valid], px[valid]].astype(np.float32)
    occupied = np.full(valid.shape, outside_value, dtype=np.float32)
    occupied[valid] = 1 - free[valid]
    return torch.as_tensor(np.stack((free, occupied), axis=1), device=device)
