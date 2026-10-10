"""Pure Torch actor-row bridge for four Articulations; no state-format conversion."""
from __future__ import annotations

import torch

ASSET_SLOTS = ("Robot_0", "Robot_1", "Robot_2", "Robot_3")
GROUP_SIZE = 4


def _device(device):
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Explicit cpu or indexed CUDA device required") from error
    if target.type == "cpu":
        return torch.device("cpu")
    if target.type != "cuda" or target.index is None:
        raise ValueError("Use cpu or an explicit CUDA index")
    return target


def _indices(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.device != device or value.dtype != torch.int64:
        raise ValueError(f"{name} requires int64 Tensor{tuple(shape)} on {device}")
    return value


def _partition(group_actor_rows, device):
    if not isinstance(group_actor_rows, torch.Tensor) or group_actor_rows.ndim != 2 or group_actor_rows.shape[1] != GROUP_SIZE or not len(group_actor_rows):
        raise ValueError("group_actor_rows requires nonempty Tensor[G,4]")
    rows = _indices(group_actor_rows, group_actor_rows.shape, "group_actor_rows", device)
    count = rows.numel()
    if bool(((rows < 0) | (rows >= count)).any()) or torch.unique(rows).numel() != count:
        raise ValueError("Groups must partition every canonical actor row0..4G-1 exactly once")
    return rows


def _state(value, count, name, device, *, shape=None, dtype=None):
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != count or not value.numel() or value.device != device:
        raise ValueError(name + " requires a nonempty Tensor with the declared batch and device")
    if value.dtype not in (torch.float32, torch.float64) or (dtype is not None and value.dtype != dtype) or (shape is not None and value.shape != shape):
        raise ValueError(name + " requires matching shape and float32/64 dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " requires finite values")
    return value


def mixed_group_rows(num_actors, *, device):
    """Actor row map only: first-half AMASS, last-half procedural. No stable IDs."""
    if type(num_actors) is not int or num_actors < GROUP_SIZE or num_actors % GROUP_SIZE:
        raise ValueError("num_actors must be an integer>=4 divisible by4")
    target = _device(device)
    first = 2*torch.arange(num_actors//GROUP_SIZE, dtype=torch.int64, device=target)
    return torch.stack((first, first+1, num_actors//2+first, num_actors//2+first+1), -1)


def assets_to_canonical(asset_tensors, group_actor_rows, *, device):
    """Four slot-ordered float[G,...] tensors -> new float[4G,...]; values unchanged."""
    target = _device(device)
    rows = _partition(group_actor_rows, target)
    if not isinstance(asset_tensors, (list, tuple)) or len(asset_tensors) != GROUP_SIZE:
        raise ValueError("Exactly four asset tensors required, in Robot_0..3 slot order")
    first = _state(asset_tensors[0], len(rows), ASSET_SLOTS[0], target)
    values = tuple(_state(value, len(rows), ASSET_SLOTS[slot], target, shape=first.shape, dtype=first.dtype)
                   for slot, value in enumerate(asset_tensors))
    canonical = torch.empty((rows.numel(), *first.shape[1:]), dtype=first.dtype, device=target)
    for slot, value in enumerate(values):
        canonical.index_copy_(0, rows[:, slot], value)
    return canonical


def gather_asset(canonical, group_actor_rows, *, slot, device):
    """New float[G,...] for one explicit integer slot0..3; no joint/quat reordering."""
    if type(slot) is not int or not 0 <= slot < GROUP_SIZE:
        raise ValueError("slot must be an integer0..3")
    target = _device(device)
    rows = _partition(group_actor_rows, target)
    value = _state(canonical, rows.numel(), "canonical actor tensor", target)
    return value.index_select(0, rows[:, slot])


def canonical_to_assets(canonical, group_actor_rows, *, device):
    """Canonical actor rows -> four new asset[G,...] tensors in fixed slot order."""
    target = _device(device)
    rows = _partition(group_actor_rows, target)
    value = _state(canonical, rows.numel(), "canonical actor tensor", target)
    return tuple(value.index_select(0, rows[:, slot]) for slot in range(GROUP_SIZE))


def actions_to_assets(actions, group_actor_rows, *, device):
    """Actor-row-only split of canonical[N,69]; caller owns common->SIMULATOR DOFs/PD."""
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2 or actions.shape[1] != 69:
        raise ValueError("Actions require Tensor[N,69]")
    return canonical_to_assets(actions, group_actor_rows, device=device)


def reset_group_env_ids(canonical_rows, group_actor_rows, *, device):
    """Whole-group actor-row reset request -> four equal copied group env-index lists.

    Partial groups are rejected, never expanded. Empty reset is valid. This
    changes neither termination flags nor histories and makes no SDK writes.
    """
    target = _device(device)
    rows = _partition(group_actor_rows, target)
    if not isinstance(canonical_rows, torch.Tensor) or canonical_rows.ndim != 1:
        raise ValueError("Reset canonical_rows requires int64 Tensor[K]")
    selected = _indices(canonical_rows, canonical_rows.shape, "reset canonical_rows", target)
    if bool(((selected < 0) | (selected >= rows.numel())).any()) or torch.unique(selected).numel() != len(selected):
        raise ValueError("Reset rows must be unique valid canonical row indices")
    mask = torch.zeros(rows.numel(), dtype=torch.bool, device=target)
    mask.index_fill_(0, selected, True)
    counts = mask[rows].sum(-1)
    if bool(((counts > 0) & (counts < GROUP_SIZE)).any()):
        raise ValueError("Reset request must contain every member of each requested group")
    groups = torch.nonzero(counts == GROUP_SIZE, as_tuple=False).flatten()
    return tuple(groups.clone() for _ in range(GROUP_SIZE))
