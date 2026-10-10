"""Explicit motion sampling constraints for the pedestrian inference profile.

The general AMASS validation library is not a walking reset distribution. Keep
the original library intact and constrain the existing MotionManager sampler
using named clips and checked first-frame poses. This does not train or replace
the MaskedMimic policy, and does not establish long-horizon physical stability.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import math

import torch

from protomotions.utils.rotations import quat_rotate


def configure_pedestrian_reset_motions(env, config: dict) -> dict:
    """Select checked reset clips through the existing persistent exclusions.

    Patterns match the source paths recorded by MotionLib, not guessed numeric
    IDs. Missing matches, non-upright starts, incompatible fixed assignments and
    exhausted sampling weights fail before changing the sampler. First-frame
    validation requires resets at time zero; other profiles retain their own
    reset distribution by omitting ``reset_motion_patterns``.
    """
    patterns = config.get("reset_motion_patterns")
    if not isinstance(patterns, (list, tuple)) or not patterns or any(
        not isinstance(pattern, str) or not pattern for pattern in patterns
    ):
        raise ValueError("reset_motion_patterns must be a nonempty list of path patterns")
    max_vertical_speed = float(config.get("reset_motion_max_vertical_speed", 0.15))
    max_tilt = float(config.get("reset_motion_max_tilt_degrees", 15.0))
    if not math.isfinite(max_vertical_speed) or max_vertical_speed < 0.0:
        raise ValueError("reset_motion_max_vertical_speed must be finite and nonnegative")
    if not math.isfinite(max_tilt) or not 0.0 <= max_tilt < 90.0:
        raise ValueError("reset_motion_max_tilt_degrees must be finite and in [0, 90)")

    manager = env.motion_manager
    library = env.motion_lib
    if manager.config.init_start_prob != 1.0:
        raise ValueError("Checked pedestrian reset poses require init_start_prob=1.0")
    if not getattr(manager.config, "resample_on_reset", True):
        raise ValueError("Checked pedestrian reset poses require resample_on_reset=True")
    names = library.motion_files
    if len(names) != library.num_motions():
        raise ValueError("MotionLib source paths must identify every reset motion")
    selected = [
        index for index, name in enumerate(names)
        if any(fnmatchcase(str(name).replace("\\", "/"), pattern) for pattern in patterns)
    ]
    if not selected:
        raise ValueError("No pedestrian reset motions match reset_motion_patterns")
    ids = torch.tensor(selected, dtype=torch.long, device=manager.device)
    state = library.get_motion_state(ids, torch.zeros(len(ids), device=manager.device))
    required = (state.root_pos, state.root_rot, state.root_vel, state.root_ang_vel,
                state.dof_pos, state.dof_vel)
    if any(value is None or not torch.isfinite(value).all() for value in required):
        raise ValueError("Pedestrian reset motion starts must have finite poses and velocities")
    norms = state.root_rot.norm(dim=-1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("Pedestrian reset roots must have nonzero quaternions")
    up = torch.zeros_like(state.root_pos)
    up[:, 2] = 1.0
    rotated_up = quat_rotate(state.root_rot / norms, up, w_last=True)
    tilt_degrees = torch.rad2deg(torch.acos(rotated_up[:, 2].clamp(-1.0, 1.0)))
    height = state.root_pos[:, 2]
    valid_pose = (
        (height >= 0.8) & (height <= 1.1)
        & (state.root_vel[:, 2].abs() <= max_vertical_speed)
        & (tilt_degrees <= max_tilt)
    )
    if not valid_pose.all():
        rejected = ids[~valid_pose].cpu().tolist()
        raise ValueError(f"Pedestrian reset motions have unsuitable upright starts: {rejected}")

    allowed = torch.zeros(library.num_motions(), dtype=torch.bool, device=manager.device)
    allowed[ids] = True
    eligible = manager._sampling_eligible_motion_mask & allowed
    if not torch.any(eligible & (manager.motion_weights > 0)):
        raise ValueError("No eligible pedestrian reset motions retain positive sampling weight")
    assigned = manager.available_motion_ids
    if assigned is not None and not allowed[assigned.to(manager.device)].all():
        raise ValueError("Fixed motion subset contains motions outside pedestrian reset patterns")
    fixed = getattr(manager, "_fixed_motion_ids_per_env", None)
    if fixed is not None:
        fixed = fixed[fixed >= 0].to(manager.device)
        if not allowed[fixed].all():
            raise ValueError("Scene fixed motion IDs conflict with pedestrian reset patterns")

    # Persist via the manager's exclusion contract, including its existing
    # exclusions and short-clip eligibility, so weight updates cannot undo it.
    existing = manager.excluded_motion_ids
    if existing is not None:
        allowed[existing] = False
    excluded = (~allowed).nonzero(as_tuple=False).flatten()
    manager.config.exclude_motion_ids = excluded.cpu().tolist()
    manager.excluded_motion_ids = excluded
    manager._sampling_eligible_motion_mask &= allowed
    manager._apply_motion_exclusions()

    report = {
        "patterns": list(patterns), "selected_motion_ids": selected,
        "selected_motion_files": [str(names[index]) for index in selected],
        "eligible_motion_ids": manager._sampling_eligible_motion_mask.nonzero().flatten().cpu().tolist(),
        "reset_time_s": 0.0, "root_heights_m": height.cpu().tolist(),
        "vertical_velocities_m_s": state.root_vel[:, 2].cpu().tolist(),
        "root_tilt_degrees": tilt_degrees.cpu().tolist(),
        "maximum_vertical_speed_m_s": max_vertical_speed,
        "maximum_tilt_degrees": max_tilt,
        "source_motion_file": str(library.motion_file), "source_unchanged": True,
    }
    env._crowdsim_reset_motion_manifest = report
    print(f"[CrowdSim] Checked walking reset motion IDs: {report['eligible_motion_ids']}")
    return report
