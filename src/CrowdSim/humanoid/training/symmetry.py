"""Differentiable sagittal reflection of raw public PACER observations.

Contract reference: nv-tlabs/pacer baeb6241f03fc9d78370a64e3c2dcef9d830d97f,
humanoid.py::_compute_flip_humanoid_obs, humanoid_pedestrain_terrain.py::
_compute_flip_task_obs, and amp_continuous.py::_sym_loss. These independently
written operators implement reflection Y -> -Y and bilateral name exchange.
Apply them before observation normalization. They impose a policy regularizer;
an imperfectly bilateral neutral SMPL rig is not an exactly symmetric plant.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..amp.observations import PACER_BODY_NAMES, PACER_DOF_NAMES


def _counterpart(name: str) -> str:
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


def _name_permutation(names, canonical, *, label):
    names = tuple(names)
    if len(names) != len(canonical) or len(set(names)) != len(names) or set(names) != set(canonical):
        raise ValueError(f"Mirror {label} names must be an exact public PACER bijection")
    return tuple(names.index(_counterpart(name)) for name in names)


def _validate_tensor(value, dimension, label):
    if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[-1] != dimension or value.numel() == 0:
        raise ValueError(f"Mirror {label} requires a nonempty batched tensor with final dimension {dimension}")
    if not value.is_floating_point() or value.device.type == "meta" or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Mirror {label} requires finite, materialized floating point values")


def mirror_policy_observations(self_obs: Tensor, task_obs: Tensor, *,
                               body_names=PACER_BODY_NAMES) -> tuple[Tensor, Tensor]:
    """Mirror raw (...,357) state and (...,1044) task without mutation.

    State: 23 root-relative XYZ positions, 24 tangent-X/normal-Z rotations,
    24 linear and 24 axial angular velocities. Task: ten root-heading XY
    waypoints and a head-centered 32x32 height grid flattened in ij order.
    Optional names describe the encoded body order; Pelvis must remain first.
    """
    if isinstance(self_obs, Tensor) and isinstance(task_obs, Tensor):
        if self_obs.device != task_obs.device or self_obs.dtype != task_obs.dtype:
            raise ValueError("Mirror self/task tensors must share device and floating point dtype")
        if self_obs.shape[:-1] != task_obs.shape[:-1]:
            raise ValueError("Mirror self/task tensors must share all batch dimensions")
    _validate_tensor(self_obs, 357, "self observation")
    _validate_tensor(task_obs, 1044, "task observation")
    body_names = tuple(body_names)
    permutation = _name_permutation(body_names, PACER_BODY_NAMES, label="body")
    if body_names[0] != "Pelvis":
        raise ValueError("Mirror body order requires Pelvis first for root-relative positions")
    indices = torch.tensor(permutation, device=self_obs.device)
    positions_index = indices[1:] - 1
    polar = self_obs.new_tensor((1., -1., 1.))
    axial = self_obs.new_tensor((-1., 1., -1.))
    leading = self_obs.shape[:-1]
    positions = self_obs[..., :69].reshape(*leading, 23, 3).index_select(-2, positions_index) * polar
    # For R' = S R S, columns X and Z are both polar vectors under Y reflection.
    rotations = self_obs[..., 69:213].reshape(*leading, 24, 2, 3).index_select(-3, indices) * polar
    velocity = self_obs[..., 213:285].reshape(*leading, 24, 3).index_select(-2, indices) * polar
    angular = self_obs[..., 285:357].reshape(*leading, 24, 3).index_select(-2, indices) * axial
    mirrored_self = torch.cat((positions.flatten(-2), rotations.flatten(-3),
                               velocity.flatten(-2), angular.flatten(-2)), -1)
    trajectory = task_obs[..., :20].reshape(*leading, 10, 2) * task_obs.new_tensor((1., -1.))
    # ij indexing: X is the first grid axis; Y is the last grid axis.
    heights = task_obs[..., 20:].reshape(*leading, 32, 32).flip(-1)
    return mirrored_self, torch.cat((trajectory.flatten(-2), heights.flatten(-2)), -1)


def mirror_actions(actions: Tensor, *, dof_names=PACER_DOF_NAMES) -> Tensor:
    """Mirror (...,69) raw PD action/exp-map triples using axial-vector parity.

    This does not clamp, mask, normalize or detach the input. The trainer must
    explicitly choose its loss scaling and effective physical coordinates.
    """
    _validate_tensor(actions, 69, "actions")
    dof_names = tuple(dof_names)
    permutation = _name_permutation(dof_names, PACER_DOF_NAMES, label="DOF")
    indices = torch.tensor(permutation, device=actions.device)
    signs = actions.new_tensor(tuple(1. if name.endswith("_y") else -1. for name in dof_names))
    return actions.index_select(-1, indices) * signs


__all__ = ["mirror_policy_observations", "mirror_actions"]
