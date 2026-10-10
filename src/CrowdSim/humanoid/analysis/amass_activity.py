"""Descriptive velocity statistics, without interpreting low motion as standing."""
from __future__ import annotations

import numpy as np

SPEED_EDGES = (0., .1, .3, .8, 1.5, np.inf)
JOINT_EDGES = (0., .5, 1., 2., 4., np.inf)
PROXIES = ("low_planar_speed", "low_joint_component_RMS", "low_root_angular_speed",
           "low_planar_and_joint", "all_three_low")
METRICS = ("planar_speed_m_s", "joint_component_RMS_rad_s", "root_angular_speed_rad_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clip_statistics(root_velocity, joint_velocity, root_angular_velocity, *, past=9, future=1):
    require(type(past) is int and past >= 0 and type(future) is int and future >= 1,
            "Require nonnegative integer past and positive integer future")
    values = [np.asarray(value) for value in (root_velocity, joint_velocity, root_angular_velocity)]
    count = len(values[0])
    require(count > past + future, "No legal expert center")
    for value, width in zip(values, (3, 57, 3), strict=True):
        require(value.shape == (count, width) and value.dtype.kind == "f" and np.isfinite(value).all(),
                "Require matching finite floating-point velocity arrays")
    root, joints, angular = (value.astype(np.float64) for value in values)
    metrics = np.stack((np.linalg.norm(root[:, :2], axis=1),
                        np.sqrt(np.mean(joints * joints, axis=1)),
                        np.linalg.norm(angular, axis=1)), axis=1)
    low = metrics < np.array((.1, .5, .3))
    proxies = np.column_stack((low, low[:, 0] & low[:, 1], low.all(axis=1)))
    centers = np.arange(past, count - future)
    prefix = np.vstack((np.zeros((1, len(PROXIES)), dtype=np.int64),
                        np.cumsum(proxies, axis=0, dtype=np.int64)))
    history_low = prefix[centers + 1] - prefix[centers - past] == past + 1
    views = []
    for selected in (np.arange(count), centers):
        selected_metrics = metrics[selected]
        histogram = np.histogram2d(selected_metrics[:, 0], selected_metrics[:, 1],
                                   bins=(SPEED_EDGES, JOINT_EDGES))[0].astype(np.int64)
        require(histogram.sum() == len(selected), "Histogram must retain every frame")
        views.append({"count": int(len(selected)), "histogram": histogram,
                      "metric_means": selected_metrics.mean(axis=0),
                      "proxy_counts": proxies[selected].sum(axis=0, dtype=np.int64)})
    return {"views": views, "history_proxy_counts": history_low.sum(axis=0, dtype=np.int64)}


def aggregate(rows):
    """Keep clip-uniform and frame/center-uniform measures distinct."""
    require(len(rows) > 0, "Require at least one clip")
    result = {}
    for view_index, name in enumerate(("all_frames", "legal_expert_centers")):
        views = [row["views"][view_index] for row in rows]
        counts = np.asarray([view["count"] for view in views], dtype=np.int64)
        require((counts > 0).all(), "Every clip must contribute")
        hist = np.stack([view["histogram"] for view in views])
        means = np.stack([view["metric_means"] for view in views])
        proxies = np.stack([view["proxy_counts"] for view in views])
        require(np.array_equal(hist.sum(axis=(1, 2)), counts), "Per-clip histogram accounting differs")
        measures = {}
        for weighting in ("uniform_clip_then_frame", "uniform_frame"):
            weights = np.full(len(rows), 1 / len(rows)) if weighting == "uniform_clip_then_frame" else counts / counts.sum()
            rates = (proxies / counts[:, None] * weights[:, None]).sum(axis=0)
            measures[weighting] = {
                "histogram_probability": (hist / counts[:, None, None] * weights[:, None, None]).sum(axis=0).tolist(),
                "metric_means": dict(zip(METRICS, (means * weights[:, None]).sum(axis=0).tolist(), strict=True)),
                "proxy_probabilities": dict(zip(PROXIES, rates.tolist(), strict=True))}
            if view_index == 1:
                histories = np.stack([row["history_proxy_counts"] for row in rows])
                history_rates = (histories / counts[:, None] * weights[:, None]).sum(axis=0)
                measures[weighting]["all_history_frames_low_probabilities"] = dict(zip(PROXIES, history_rates.tolist(), strict=True))
        result[name] = {"frames_or_centers": int(counts.sum()), "clips_retained": len(rows), "measures": measures}
    return result
