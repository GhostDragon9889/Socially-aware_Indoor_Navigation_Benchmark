"""Lossless serialization of variable sample counts (DDPM and safety braking)."""

import numpy as np


def pack_candidate_samples(rows):
    if not rows:
        return np.empty((0, 0, 52, 6), dtype=np.float32), np.zeros(1, dtype=np.int64)
    actors = rows[0].shape[0]
    if any(
        row.ndim != 4
        or row.shape[0] != actors
        or row.shape[1] < 1
        or row.shape[2:] != (52, 6)
        or row.dtype != rows[0].dtype
        or not np.isfinite(row).all()
        for row in rows
    ):
        raise ValueError(
            "Every plan requires finite matched actors and physical[S,52,6] samples"
        )
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum([row.shape[1] for row in rows], dtype=np.int64),
        )
    )
    # [sum(S),B,52,6], recover row i by packed[offsets[i]:offsets[i+1]].transpose(1,0,2,3).
    packed = np.concatenate([row.transpose(1, 0, 2, 3) for row in rows], axis=0)
    return packed, offsets
