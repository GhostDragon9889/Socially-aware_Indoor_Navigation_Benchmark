"""PACER no_shape observation contract, using common XYZW physical states.

Reference equations: nv-tlabs/pacer at baeb6241f03fc9d78370a64e3c2dcef9d830d97f,
humanoid.py, humanoid_amp.py and humanoid_pedestrain_terrain.py. No simulator,
AMASS or MaskedMimic policy is imported or selected by these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from protomotions.utils.rotations import (
    calc_heading_quat,
    calc_heading_quat_inv,
    exp_map_to_quat,
    quat_mul,
    quat_to_tan_norm,
)

PACER_BODY_NAMES = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee",
    "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)
PACER_DOF_NAMES = tuple(f"{body}_{axis}" for body in PACER_BODY_NAMES[1:] for axis in "xyz")
_EXCLUDED_DISC_BODIES = {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
_DISC_DOFS = tuple(index for index, name in enumerate(PACER_DOF_NAMES)
                   if name.rsplit("_", 1)[0] not in _EXCLUDED_DISC_BODIES)
_KEY_BODIES = tuple(PACER_BODY_NAMES.index(name) for name in ("R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"))


@dataclass(frozen=True)
class PacerSMPLMapping:
    """Explicit bijections from actual simulator names to the published order."""

    body_names: tuple[str, ...] = PACER_BODY_NAMES
    dof_names: tuple[str, ...] = PACER_DOF_NAMES

    def __post_init__(self):
        for actual, canonical, label in (
            (self.body_names, PACER_BODY_NAMES, "body"),
            (self.dof_names, PACER_DOF_NAMES, "DOF"),
        ):
            if len(actual) != len(canonical) or len(set(actual)) != len(actual) or set(actual) != set(canonical):
                raise ValueError(f"PACER requires an exact named SMPL {label} bijection")

    @property
    def body_indices(self) -> tuple[int, ...]:
        return tuple(self.body_names.index(name) for name in PACER_BODY_NAMES)

    @property
    def dof_indices(self) -> tuple[int, ...]:
        return tuple(self.dof_names.index(name) for name in PACER_DOF_NAMES)

    def canonical_state(self, state):
        bodies = {}
        for name, width in (("rigid_body_pos", 3), ("rigid_body_rot", 4),
                            ("rigid_body_vel", 3), ("rigid_body_ang_vel", 3)):
            value = getattr(state, name, None)
            if value is None or value.ndim != 3 or value.shape[1:] != (24, width):
                raise ValueError(f"PACER requires {name} with shape [B,24,{width}]")
            if value.shape[0] == 0 or not torch.isfinite(value).all():
                raise ValueError(f"PACER requires nonempty finite {name}")
            bodies[name] = value[:, self.body_indices]
        shape = bodies["rigid_body_pos"].shape
        for value in bodies.values():
            if value.shape[0] != shape[0] or value.device != bodies["rigid_body_pos"].device:
                raise ValueError("Body state batch/device mismatch")
        quaternion = bodies["rigid_body_rot"]
        if not torch.allclose(quaternion.norm(dim=-1), torch.ones(shape[:2], device=quaternion.device, dtype=quaternion.dtype), atol=1e-3, rtol=0):
            raise ValueError("PACER requires unit XYZW body quaternions")
        for name in ("dof_pos", "dof_vel"):
            value = getattr(state, name, None)
            if value is None or value.shape != (shape[0], 69) or not torch.isfinite(value).all():
                raise ValueError(f"PACER requires finite {name} with shape [B,69]")
            if value.device != bodies["rigid_body_pos"].device:
                raise ValueError("DOF/body device mismatch")
            bodies[name] = value[:, self.dof_indices]
        bodies.update(root_pos=bodies["rigid_body_pos"][:, 0],
                      root_rot=bodies["rigid_body_rot"][:, 0],
                      root_vel=bodies["rigid_body_vel"][:, 0],
                      root_ang_vel=bodies["rigid_body_ang_vel"][:, 0])
        return SimpleNamespace(**bodies)


def _rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """XYZW rotation, allowing matching arbitrary leading batch dimensions."""
    xyz, w = quaternion[..., :3], quaternion[..., 3:4]
    return vector * (2 * w.square() - 1) + 2 * w * torch.cross(xyz, vector, dim=-1) + 2 * xyz * (xyz * vector).sum(-1, keepdim=True)


def build_self_observation(state, *, mapping: PacerSMPLMapping | None = None) -> torch.Tensor:
    """357 = 23 relative positions + 24 rotations/velocities/angular velocities.

    Root height is excluded, upright=True, localRootObs=True. Position blocks
    and six-dimensional rotations use root heading rather than full root tilt.
    """
    state = (mapping or PacerSMPLMapping()).canonical_state(state)
    inverse = calc_heading_quat_inv(state.root_rot, w_last=True)[:, None].expand(-1, 24, -1)
    positions = _rotate(inverse, state.rigid_body_pos - state.root_pos[:, None])[:, 1:]
    rotations = quat_to_tan_norm(quat_mul(inverse, state.rigid_body_rot, w_last=True), w_last=True)
    velocities = _rotate(inverse, state.rigid_body_vel)
    angular = _rotate(inverse, state.rigid_body_ang_vel)
    return torch.cat(tuple(value.flatten(1) for value in (positions, rotations, velocities, angular)), dim=-1)


def _height_grid(state, *, center: bool) -> torch.Tensor:
    # Official NumPy linspace -> float64 grid -> float32 point assignment.
    count = 3 if center else 32
    x_extent, y_extent = (0.1, 0.2) if center else (2.0, 2.0)
    x = torch.linspace(-x_extent, x_extent, count, dtype=torch.float64, device=state.root_pos.device)
    y = torch.linspace(-y_extent, y_extent, count, dtype=torch.float64, device=state.root_pos.device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten(), torch.zeros(count * count, device=xx.device)), -1).to(state.root_pos.dtype)


def _sample_heights(terrain, xy: torch.Tensor) -> torch.Tensor:
    if terrain is None:
        raise ValueError("An explicit terrain.height_at(xy) sampler is required")
    sample = terrain.height_at(xy) if hasattr(terrain, "height_at") else terrain(xy)
    if not isinstance(sample, torch.Tensor) or sample.shape != xy.shape[:-1] or sample.device != xy.device:
        raise ValueError("Terrain must return tensor heights with matching leading shape/device")
    if not torch.isfinite(sample).all():
        raise ValueError("Terrain heights must be finite")
    return sample


def build_task_observation(state, trajectory: torch.Tensor, terrain, *,
                           trajectory_frame: str = "world",
                           mapping: PacerSMPLMapping | None = None) -> torch.Tensor:
    """20 trajectory XY + 1024 head-centered heading-aligned terrain samples.

    trajectory contains ten positions at t+0,t+0.5,...,t+4.5 seconds
    under the public PACER clock, [B,10,2 or 3]. An explicit legacy runtime
    configuration can shift them by +0.5s; this builder only maps coordinates.
    world means absolute XY; local_heading means root-relative heading-frame
    offsets. Terrain heights are absolute world Z, normalized by the mean of
    nine small root-heading samples, clipped to [-3,3] and multiplied by five.
    """
    state = (mapping or PacerSMPLMapping()).canonical_state(state)
    batch = state.root_pos.shape[0]
    if trajectory.ndim != 3 or trajectory.shape[:2] != (batch, 10) or trajectory.shape[2] not in (2, 3):
        raise ValueError("PACER trajectory must contain ten future XY/XYZ positions")
    if trajectory.device != state.root_pos.device or not torch.isfinite(trajectory).all():
        raise ValueError("Trajectory must be finite and on the physical state device")
    if trajectory_frame == "world":
        delta = torch.zeros((batch, 10, 3), device=trajectory.device, dtype=trajectory.dtype)
        delta[..., :2] = trajectory[..., :2] - state.root_pos[:, None, :2]
        inverse = calc_heading_quat_inv(state.root_rot, w_last=True)[:, None].expand(-1, 10, -1)
        local_xy = _rotate(inverse, delta)[..., :2]
    elif trajectory_frame == "local_heading":
        local_xy = trajectory[..., :2]
    else:
        raise ValueError("trajectory_frame must be world or local_heading")

    head = PACER_BODY_NAMES.index("Head")
    heading = calc_heading_quat(state.rigid_body_rot[:, head], w_last=True)
    grid = _height_grid(state, center=False)[None].expand(batch, -1, -1)
    locations = _rotate(heading[:, None].expand(-1, 1024, -1), grid) + state.rigid_body_pos[:, head, None]
    measured = _sample_heights(terrain, locations[..., :2])
    center_grid = _height_grid(state, center=True)[None].expand(batch, -1, -1)
    # Official get_center_heights uses quat_apply_yaw (quaternion Z/W
    # projection), which differs from projected-X heading when root is tilted.
    root_heading = state.root_rot.clone()
    root_heading[:, :2] = 0
    root_heading = root_heading / root_heading.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centers = _rotate(root_heading[:, None].expand(-1, 9, -1), center_grid) + state.root_pos[:, None]
    center_height = _sample_heights(terrain, centers[..., :2]).mean(-1, keepdim=True)
    heights = (center_height - measured).clamp(-3.0, 3.0) * 5.0
    return torch.cat((local_xy.flatten(1), heights), dim=-1)


def build_amp_observation(state, *, mapping: PacerSMPLMapping | None = None) -> torch.Tensor:
    """Published no_shape D features: 195 values, state-only, no root height.

    PACER treats the 19 selected joint triples as exponential maps; raw DOF
    velocities are used unchanged. This preserves its observation equation,
    including that original convention on a simulator's three joint scalars.
    """
    state = (mapping or PacerSMPLMapping()).canonical_state(state)
    batch = state.root_pos.shape[0]
    inverse = calc_heading_quat_inv(state.root_rot, w_last=True)
    rotation = quat_to_tan_norm(quat_mul(inverse, state.root_rot, w_last=True), w_last=True)
    velocity = _rotate(inverse, state.root_vel)
    angular = _rotate(inverse, state.root_ang_vel)
    dofs = state.dof_pos[:, _DISC_DOFS].reshape(-1, 3)
    dof_observation = quat_to_tan_norm(exp_map_to_quat(dofs, w_last=True), w_last=True).reshape(batch, -1)
    key_delta = state.rigid_body_pos[:, _KEY_BODIES] - state.root_pos[:, None]
    key_observation = _rotate(inverse[:, None].expand(-1, 4, -1), key_delta).flatten(1)
    return torch.cat((rotation, velocity, angular, dof_observation, state.dof_vel[:, _DISC_DOFS], key_observation), dim=-1)
