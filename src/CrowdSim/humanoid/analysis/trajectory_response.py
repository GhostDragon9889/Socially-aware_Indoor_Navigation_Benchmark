"""Controlled trajectory-prefix probes; action response is not locomotion quality."""
from __future__ import annotations

import numpy as np

from .tracking_error_decomposition import require

CASES = ("observed", "stop", "forward_1m_s", "backward_1m_s", "left_1m_s", "right_1m_s")


def task_cases(raw_task):
    """Own six CPU task copies; only their20 local trajectory values can change."""
    import torch

    require(isinstance(raw_task, torch.Tensor) and raw_task.ndim == 2 and raw_task.shape[1] == 12308
        and len(raw_task) > 0 and raw_task.dtype == torch.float32 and raw_task.device.type == "cpu"
        and not raw_task.requires_grad and bool(torch.isfinite(raw_task).all()), "Require finite non-gradient CPU float32[B,12308] task")
    # repeat copies storage: https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.repeat.html
    tasks = raw_task.repeat((len(CASES), 1)).reshape(len(CASES), len(raw_task), 12308)
    tasks[1:, :, :20] = 0
    offsets = torch.arange(1, 11, dtype=torch.float32)/2
    for case, axis, sign in ((2, 0, 1), (3, 0, -1), (4, 1, 1), (5, 1, -1)):
        tasks[case, :, axis:20:2] = sign*offsets
    return tasks


def action_response(means, applied, mask, std):
    """Per-row active-coordinate differences; Gaussian KL is before clipping."""
    means, applied, mask = np.asarray(means), np.asarray(applied), np.asarray(mask)
    require(means.ndim == 3 and means.shape[0] == len(CASES) and means.shape[1] > 0 and means.shape[2] == 69
        and means.dtype == applied.dtype == np.float32 and applied.shape == means.shape
        and np.isfinite(means).all() and np.isfinite(applied).all(), "Require finite float32[6,B,69] actor outputs")
    require(mask.shape == (69,) and mask.dtype == np.bool_ and int(mask.sum()) == 57,
        "Require explicit canonical57-active Boolean action mask")
    require(type(std) is float and np.isfinite(std) and std > 0, "Require the admitted positive fixed Gaussian std")
    require(np.array_equal(applied, np.clip(means, -1, 1)*mask), "Applied action clipping/mask changed")
    mean = means[..., mask].astype(np.float64)
    action = applied[..., mask].astype(np.float64)
    delta, applied_delta = mean-mean[:1], action-action[:1]
    return {"mean_delta_active_RMS": np.sqrt(np.square(delta).mean(-1)),
        "applied_delta_active_RMS": np.sqrt(np.square(applied_delta).mean(-1)),
        "unclipped_active_equal_std_Gaussian_KL": .5*np.square(delta/std).sum(-1),
        "strictly_clipped_active_mean_coordinates": (np.abs(mean) > 1).sum(-1),
        "active_mean_RMS": np.sqrt(np.square(mean).mean(-1))}


def summarize(arrays):
    return {case: {key: {"rows": value.shape[1], "mean": float(value[i].mean()),
        "maximum": float(value[i].max())} for key, value in arrays.items()} for i, case in enumerate(CASES)}
