"""Same-instant body angular velocities versus spherical-joint velocities."""
from __future__ import annotations

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct(rotations, angular_velocity, joint_velocity, parents):
    """Reconstruct child and parent bases, without time differencing."""
    q, angular, native = (np.asarray(value) for value in (rotations, angular_velocity, joint_velocity))
    require(q.ndim >= 3 and q.shape[-2:] == (24, 4) and q.size > 0
            and angular.shape == q.shape[:-1] + (3,) and native.shape == q.shape[:-2] + (69,),
            "Require matching nonempty body24/joint69 arrays")
    for value in (q, angular, native):
        require(value.dtype.kind == "f" and np.isfinite(value).all(), "Require finite floating-point states")
    require(len(parents) == 23 and all(type(value) is int and 0 <= value < 24 for value in parents),
            "Require integer parent IDs for every non-root body")
    for child in range(1, 24):
        ancestors, current = set(), child
        while current:
            require(current not in ancestors, "Parent graph must reach the root without cycles")
            ancestors.add(current)
            current = parents[current - 1]
    q, angular, native = (value.astype(np.float64) for value in (q, angular, native))
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.all(np.abs(norm - 1.) <= 1e-3), "Require unit XYZW body quaternions")
    q /= norm
    relative_world = angular[..., 1:, :] - angular[..., parents, :]

    def inverse_rotate(rotation, vector):
        xyz, w = rotation[..., :3], rotation[..., 3:4]
        return vector * (2 * w * w - 1) - 2 * w * np.cross(xyz, vector) + 2 * xyz * np.sum(xyz * vector, axis=-1, keepdims=True)

    child = inverse_rotate(q[..., 1:, :], relative_world)
    parent = inverse_rotate(q[..., parents, :], relative_world)
    return native.reshape(*native.shape[:-1], 23, 3), child, parent


def error_statistics(error):
    values = np.asarray(error, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), "Require finite errors")
    return {"rms_rad_s": float(np.sqrt(np.mean(values * values))),
            "mean_abs_rad_s": float(np.mean(np.abs(values))), "max_abs_rad_s": float(np.max(np.abs(values)))}
