"""NumPy-only storage for optional measured foot support; no SDK initialization."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

PEER_FIELDS = ("static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n")
_METADATA = {"common_body_names", "foot_names", "reference_true_com", "captured_true_com", "extra_physics_steps"}


def _array(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "bifu" or (result.dtype.kind == "f" and not np.isfinite(result).all()):
        raise ValueError(f"{name} requires finite numeric data without object arrays")
    return result.copy()


def _publish_npz(path, arrays):
    """Write one complete archive then publish atomically without overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic NOREPLACE; existing user artifacts survive.
    finally:
        Path(temporary).unlink(missing_ok=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class GroundSupportArchive:
    """Accumulate published controls on CPU, with explicit before/post counters."""

    def __init__(self, num_actors):
        if type(num_actors) is not int or num_actors < 1:
            raise ValueError("Ground archive needs a positive actor count")
        self.num_actors = num_actors
        self.rows = []
        self.common_names = None
        self.foot_names = None

    def append(self, sample, *, reset_mask, terminated, truncated, control_step_after):
        row = {key: _array(value, key) for key, value in sample.items()
               if key not in _METADATA and key not in PEER_FIELDS and value is not None}
        required = {"foot_floor_normal_force_w", "body_com_pos_w", "local_com_pose_b", "body_masses_kg",
                    "mass_weighted_world_com", "physics_step_before", "physics_step", "physics_step_end",
                    "episode_ids", "control_step", "substep_valid_count"}
        if not required.issubset(row) or sample.get("captured_true_com") is not True or sample.get("reference_true_com") is not None:
            raise ValueError("Ground archive requires complete measured force/COM/counter data")
        b = self.num_actors
        force = row["foot_floor_normal_force_w"]
        if force.ndim != 5 or force.shape[:3] != (4, b, 4) or force.shape[3] < 1 or force.shape[-1] != 3:
            raise ValueError("Floor force requires[4,actors,4,floor_colliders,3]")
        expected = {"body_com_pos_w": (4, b, 24, 3), "local_com_pose_b": (4, b, 24, 7),
                    "body_masses_kg": (4, b, 24), "mass_weighted_world_com": (4, b, 3),
                    "physics_step": (4,), "episode_ids": (b,), "control_step": (),
                    "physics_step_before": (), "physics_step_end": (), "substep_valid_count": ()}
        if any(row[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Ground force/COM/clock shapes disagree")
        for name in ("physics_step", "episode_ids", "control_step", "physics_step_before", "physics_step_end", "substep_valid_count"):
            if row[name].dtype.kind not in "iu" or (row[name] < 0).any():
                raise ValueError("Ground clocks and episodes require actual nonnegative integers")
        before = int(row["physics_step_before"])
        if not np.array_equal(row["physics_step"], np.arange(before + 1, before + 5)) or int(row["physics_step_end"]) != before + 4 or int(row["substep_valid_count"]) != 4:
            raise ValueError("Ground archive requires exactly four actual consecutive physics steps")
        if type(control_step_after) is not int or control_step_after != int(row["control_step"]) + 1:
            raise ValueError("Post-control runner counter must equal the prepared before-counter+1")
        row["control_step_after"] = np.asarray(control_step_after, dtype=np.int64)
        for key, value in (("reset_mask", reset_mask), ("terminated", terminated), ("truncated", truncated)):
            row[key] = _array(value, key)
            if row[key].dtype != np.bool_ or row[key].shape != (b,):
                raise ValueError("Reset/termination masks require genuine boolean[actors]")
        names, feet = tuple(sample["common_body_names"]), tuple(sample["foot_names"])
        if len(names) != 24 or len(set(names)) != 24 or len(feet) != 4 or len(set(feet)) != 4:
            raise ValueError("Ground archive needs exact body and foot names")
        row["valid"] = np.asarray(True)
        if self.rows:
            previous = self.rows[-1]
            if names != self.common_names or feet != self.foot_names or set(row) != set(previous):
                raise ValueError("Ground channels or body order changed inside the capture")
            if any(row[key].shape != previous[key].shape or row[key].dtype != previous[key].dtype for key in row):
                raise ValueError("Ground channel shapes/dtypes changed inside the capture")
            if int(row["control_step"]) != int(previous["control_step_after"]):
                raise ValueError("Ground capture controls must remain consecutive")
            if before != int(previous["physics_step_end"]):
                raise ValueError("Missing or extra physics steps between ground captures")
            changed = row["episode_ids"] != previous["episode_ids"]
            if (row["episode_ids"] < previous["episode_ids"]).any() or not np.array_equal(changed, row["reset_mask"]):
                raise ValueError("Before-frame reset masks must match actual episode changes")
        self.common_names, self.foot_names = names, feet
        self.rows.append(row)

    def write(self, path):
        if not self.rows:
            raise ValueError("No measured controls; never publish an empty or zero-filled support capture")
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        arrays.update(common_body_names=np.asarray(self.common_names), foot_names=np.asarray(self.foot_names))
        artifact = _publish_npz(path, arrays)
        return {**artifact, "completed_controls": len(self.rows), "actors": self.num_actors,
                "completed_physics_substeps": len(self.rows) * 4, "all_controls_valid": True,
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()}}


def write_interrupted_ground_support(path, adapter, *, reason):
    """Save available failed partial samples separately, without invented rows."""
    evidence = adapter.discard_interrupted_control(reason=reason)
    arrays = {"valid": np.asarray(False), "control_step": np.asarray(evidence["control_step"], dtype=np.int64),
              "episode_ids": np.asarray(evidence["episode_ids"], dtype=np.int64),
              "physics_step_before": np.asarray(evidence["physics_step_before"], dtype=np.int64),
              "physics_step_at_disposal": np.asarray(evidence["physics_step_at_disposal"], dtype=np.int64)}
    states = evidence["completed_state_samples"]
    if states:
        arrays.update({f"partial_{key}": np.stack([_array(row[key], key) for row in states]) for key in states[0]})
    forces = evidence["floor_force_samples"]
    if forces:
        arrays["partial_foot_floor_normal_force_w"] = np.stack([_array(force, "partial force") for force in forces])
    return {**_publish_npz(path, arrays), "valid": False, "reason": reason,
            "completed_state_samples": len(states), "completed_floor_force_samples": len(forces),
            "missing_substeps_padded": False}
