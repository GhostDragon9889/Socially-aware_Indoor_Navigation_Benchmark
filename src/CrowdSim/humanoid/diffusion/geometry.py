"""TRACE row-vector point transform, independent of optional geometry packages."""

import torch


def transform_points_tensor(points, transform):
    if points.ndim != 3 or points.shape[-1] != 2 or transform.shape != (points.shape[0], 3, 3):
        raise ValueError("Expected points[B,T,2] and transform[B,3,3]")
    return torch.bmm(points, transform[:, :2, :2].transpose(1, 2)) + transform[:, None, :2, 2]
