# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit tensor and state access shared by Isaac Lab 3 SDK revisions."""

from __future__ import annotations

import inspect

import torch
import warp as wp

from protomotions.utils.rotations import quat_apply


def as_torch(value) -> torch.Tensor:
    """Read a Lab ProxyArray explicitly, preserving raw Warp/Torch support.

    SDK asset data is a ProxyArray with a cached, zero-copy ``.torch`` view.
    PhysX view APIs still return plain Warp arrays. Avoid relying on the SDK's
    deprecated implicit tensor bridge; callers clone when they need a snapshot.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    tensor = value.torch
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a Torch view, received {type(tensor).__name__}")
    return tensor


def write_root_link_state(asset, state: torch.Tensor, *, env_ids=None) -> None:
    """Write link pose and link velocity through supported index APIs.

    The ``[position, xyzw rotation, link linear/angular velocity]`` convention
    is unchanged in both the Sim 6.0 and Sim 6.1 Lab 3 backends.
    """
    asset.write_root_link_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
    write_root_link_velocity(asset, state[:, 7:], env_ids=env_ids)


def write_root_link_velocity(asset, velocity: torch.Tensor, *, env_ids=None) -> None:
    """Write world link linear/angular velocity [m/s, rad/s] via the COM API.

    The Lab 3 PhysX articulation link setter currently forwards the link buffer
    to the native COM setter. Convert explicitly and use the public COM API,
    which also invalidates the SDK's derived link/body velocity caches.
    """
    # Project assets keep backend body order, where the root occupies column 0.
    # Rotate the local offset to avoid cancellation at large world positions.
    local_offset = as_torch(asset.data.body_com_pos_b)[:, 0]
    orientation = as_torch(asset.data.root_link_quat_w)
    if env_ids is not None:
        local_offset = local_offset[env_ids]
        orientation = orientation[env_ids]
    world_offset = quat_apply(orientation, local_offset, w_last=True)
    com_velocity = velocity.clone()
    com_velocity[:, :3] += torch.cross(
        velocity[:, 3:], world_offset, dim=-1
    )
    asset.write_root_com_velocity_to_sim_index(
        root_velocity=com_velocity, env_ids=env_ids
    )


def supports_native_usd_only_replication(replicate) -> bool:
    """Whether the SDK can suppress physics replication without queue edits."""
    return "replicate_physics" in inspect.signature(replicate).parameters
