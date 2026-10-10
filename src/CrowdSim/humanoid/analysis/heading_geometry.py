"""Anatomical/root-axis diagnostics; never replace tracking or reward metrics."""
from __future__ import annotations

import numpy as np

from CrowdSim.humanoid.motion_tracking import SMPL_BODY_NAMES
from .tracking_error_decomposition import inverse_rotate, require


def body_axes(positions, rotations):
    """Measure hip lateral axis and projected forwards in COMMON24/XYZW.

    Anatomical forward is (L_Hip-R_Hip) cross measured root +Z. This is a
    geometry diagnostic, not a velocity-based definition: sideways/backwards
    mocap remains valid. Undefined XY projections have an explicit mask; their
    angle slot is unused zero, never an invented heading or a removed frame.
    """
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    require(positions.ndim == 3 and positions.shape[1:] == (24, 3) and len(positions) > 0
        and rotations.shape == (len(positions), 24, 4), "Require nonempty COMMON24 positions/XYZW rotations")
    require(np.issubdtype(positions.dtype, np.floating) and np.issubdtype(rotations.dtype, np.floating)
        and np.isfinite(positions).all() and np.isfinite(rotations).all(), "Require finite floating body geometry")
    positions, rotations = positions.astype(np.float64), rotations.astype(np.float64)
    q = rotations[:, 0]
    lateral = positions[:, SMPL_BODY_NAMES.index("L_Hip")] - positions[:, SMPL_BODY_NAMES.index("R_Hip")]
    local = inverse_rotate(q, lateral[:, None])[:, 0]
    width = np.linalg.norm(local, axis=-1)
    require(np.all(width > 1e-9), "Hip lateral axis is degenerate")
    unit = local / width[:, None]
    # Normalize only for diagnostic geometry; do not alter captured quaternions.
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q.T
    root_x = np.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    root_z = np.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
    anatomical = np.cross(lateral, root_z)
    root_norm = np.linalg.norm(root_x[:, :2], axis=-1)
    anatomical_norm = np.linalg.norm(anatomical[:, :2], axis=-1)
    defined = (root_norm > 1e-12) & (anatomical_norm > 1e-12)
    delta = np.zeros(len(q), np.float64)
    raw = np.arctan2(anatomical[defined, 1], anatomical[defined, 0]) - np.arctan2(root_x[defined, 1], root_x[defined, 0])
    delta[defined] = np.arctan2(np.sin(raw), np.cos(raw))
    return {"hip_local_unit": unit, "hip_width_m": width,
        "hip_to_local_Y_deg": np.rad2deg(np.arccos(np.clip(unit[:, 1], -1, 1))),
        "root_up_z": root_z[:, 2], "projected_root_X_norm": root_norm,
        "projected_anatomical_forward_norm": anatomical_norm,
        "heading_comparison_defined": defined, "anatomical_minus_root_heading_rad": delta}


def summarize(values):
    """Keep every geometry frame; restrict only heading statistics by its mask."""
    def stats(array):
        return {"count": len(array), "mean": float(array.mean()) if len(array) else None,
            "minimum": float(array.min()) if len(array) else None, "maximum": float(array.max()) if len(array) else None}

    mask = values["heading_comparison_defined"]
    result = {key: stats(values[key]) for key in ("hip_width_m", "hip_to_local_Y_deg", "root_up_z", "projected_root_X_norm")}
    result.update(frames=len(mask), hip_local_unit_mean=values["hip_local_unit"].mean(0).tolist(),
        undefined_heading_comparison_frames=int((~mask).sum()),
        signed_anatomical_minus_root_heading_rad=stats(values["anatomical_minus_root_heading_rad"][mask]),
        absolute_anatomical_minus_root_heading_rad=stats(np.abs(values["anatomical_minus_root_heading_rad"][mask])))
    return result
