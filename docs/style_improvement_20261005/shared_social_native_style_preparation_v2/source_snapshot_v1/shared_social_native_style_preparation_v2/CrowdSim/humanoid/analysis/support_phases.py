"""Raw native support transitions; no gait classification or contact filtering."""
from __future__ import annotations

import numpy as np


def support_phases(side_contact, episode_ids, *, physics_dt_s=1 / 120):
    """Summarize [control, substep, actor, L/R] without crossing resets.

    All contact states remain in the exposure. Cross-episode intervals cannot
    describe a physical touchdown; they are counted separately as undefined.
    A 1N native-contact onset may chatter and is not a validated footstep.
    """
    if (side_contact.dtype != np.bool_ or side_contact.ndim != 4
            or side_contact.shape[-1] != 2 or min(side_contact.shape[:3]) < 1
            or episode_ids.shape != (side_contact.shape[0], side_contact.shape[2])
            or episode_ids.dtype.kind not in "iu"
            or not np.isfinite(physics_dt_s) or physics_dt_s <= 0):
        raise ValueError("Invalid support phase shape, dtype or clock")
    controls, substeps, actors, _ = side_contact.shape
    if not np.isin(np.diff(episode_ids, axis=0), (0, 1)).all():
        raise ValueError("Episode clock must advance by zero or one")
    contact = side_contact.reshape(controls * substeps, actors, 2)
    episodes = np.repeat(episode_ids, substeps, axis=0)
    same = episodes[1:] == episodes[:-1]
    onset = (~contact[:-1] & contact[1:]) & same[..., None]
    release = (contact[:-1] & ~contact[1:]) & same[..., None]
    phase = contact[..., 0].astype(np.int8) + 2 * contact[..., 1]
    exposure = controls * substeps * actors
    counts = {name: int((phase == code).sum()) for code, name in
              enumerate(("no_support", "left_only", "right_only", "double_support"))}
    longest = []
    for actor in range(actors):
        longest_run = current = 0
        for index in range(len(contact)):
            if index and episodes[index, actor] != episodes[index - 1, actor]:
                current = 0
            current = current + 1 if phase[index, actor] == 3 else 0
            longest_run = max(longest_run, current)
        longest.append(longest_run * physics_dt_s)
    return {"actor_substeps": exposure, "actor_seconds": exposure * physics_dt_s,
            "phase_actor_substeps": counts,
            "phase_fractions": {key: count / exposure for key, count in counts.items()},
            "same_episode_actor_intervals": int(same.sum()),
            "cross_episode_actor_intervals_excluded_from_transitions": int((~same).sum()),
            "first_actor_samples_without_previous_state": actors,
            "contact_onsets_left_right": onset.sum(axis=(0, 1)).tolist(),
            "contact_releases_left_right": release.sum(axis=(0, 1)).tolist(),
            "longest_double_support_in_one_episode_s_by_actor": longest,
            "all_contact_states_included": True,
            "onsets_are_validated_steps": False,
            "qualification": "Native threshold crossings without temporal debounce; falls retained; run durations stop at episode boundaries and capture end."}
