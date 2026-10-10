"""Pure NumPy four-actor layout and reset masks; no world or runner integration."""
from __future__ import annotations

import numpy as np

GROUP_SIZE = 4


def make_mixed_layout(num_actors):
    """Return group_actor_rows[G,4], scene_ids[N], actor_ids[N] in a dict.

    N is positive and divisible by4. The first N/2 actor rows are AMASS and
    the second N/2 procedural. Group g contains [2g,2g+1,N/2+2g,N/2+2g+1].
    IDs are deterministic nonnegative int64: actor identity equals its initial
    canonical row, scene identity equals the group row. No RNG is consumed.
    """
    if type(num_actors) is not int or num_actors < GROUP_SIZE or num_actors % GROUP_SIZE:
        raise ValueError("num_actors must be an integer >=4 and divisible by4")
    half = num_actors // 2
    first = 2*np.arange(num_actors // GROUP_SIZE, dtype=np.int64)
    groups = np.stack((first, first+1, half+first, half+first+1), axis=1)
    scenes = np.empty(num_actors, dtype=np.int64)
    scenes[groups] = np.arange(len(groups), dtype=np.int64)[:, None]
    return {"group_actor_rows": groups, "scene_ids": scenes,
            "actor_ids": np.arange(num_actors, dtype=np.int64)}


def expand_group_reset(group_actor_rows, terminated, truncated):
    """Return copied terminal/truncation masks and complete group reset rows.

    Explicit group_actor_rows must partition every row of bool[N] into groups
    of four. Its row order need not match contiguous actor-row ranges.
    Any original terminated OR truncated member triggers that complete group.
    Real terminated flags are preserved; collateral members become truncated.
    Original truncated flags remain intact, even when also terminated.

    Outputs: terminated[N], truncated[N], reset_mask[N], collateral_reset[N]
    (all bool), triggered_groups[K] (int64 row indices of the supplied groups).
    No input is mutated. No curriculum, physics, planner, history or GAE runs.
    """
    terminal = np.asarray(terminated)
    truncation = np.asarray(truncated)
    if (terminal.ndim != 1 or not len(terminal) or len(terminal) % GROUP_SIZE
            or terminal.dtype != np.dtype("bool") or truncation.shape != terminal.shape
            or truncation.dtype != np.dtype("bool")):
        raise ValueError("terminated/truncated must be equal bool[N], positive N divisible by4")
    groups = np.asarray(group_actor_rows)
    count = len(terminal)
    if (groups.shape != (count // GROUP_SIZE, GROUP_SIZE) or groups.dtype.kind not in "iu"
            or np.any(groups < 0) or np.any(groups >= count)):
        raise ValueError("group_actor_rows requires in-range integer [N/4,4]")
    groups = groups.astype(np.int64, copy=False)
    if len(np.unique(groups)) != count:
        raise ValueError("group_actor_rows must cover every actor row exactly once")
    original_reset = terminal | truncation
    triggered = np.flatnonzero(original_reset[groups].any(axis=1)).astype(np.int64, copy=False)
    reset = np.zeros(count, dtype=bool)
    reset[groups[triggered]] = True
    collateral = reset & ~original_reset
    return {"terminated": terminal.copy(), "truncated": truncation | collateral,
            "reset_mask": reset, "collateral_reset": collateral,
            "triggered_groups": triggered}
