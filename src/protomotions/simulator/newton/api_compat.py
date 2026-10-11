# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton API adapters selected by supported interfaces, without version pins."""

import inspect
import warnings

import newton
import numpy as np
import torch
import warp as wp


if hasattr(newton, "ModelFlags"):
    ModelChangeFlags = newton.ModelFlags
else:
    # Newton <= 1.4 exposes the same model-change bits through its solvers.
    from newton.solvers import SolverNotifyFlags as ModelChangeFlags


def position_target_attribute(control) -> str:
    """Select a live target attribute; the removed 1.5 alias raises on access."""
    if hasattr(control, "joint_target_q"):
        return "joint_target_q"
    return "joint_target_pos"


def mujoco_solver_options(sim_params, device, solver_type) -> dict:
    """Preserve common options and pass optional arguments only when supported."""
    options = {
        name: getattr(sim_params, name)
        for name in (
            "solver", "integrator", "njmax", "nconmax", "iterations",
            "ls_iterations", "impratio", "cone", "ccd_iterations",
        )
    }
    parameters = inspect.signature(solver_type.__init__).parameters
    if "ls_parallel" in parameters:
        options["ls_parallel"] = sim_params.ls_parallel
    elif not sim_params.ls_parallel:
        warnings.warn(
            "This Newton/MuJoCo version has no ls_parallel option; the solver's "
            "line-search implementation is used. ls_parallel=False cannot be enforced.",
            RuntimeWarning,
            stacklevel=2,
        )
    if torch.device(device).type == "cpu":
        if "use_mujoco_cpu" not in parameters:
            raise RuntimeError("This Newton solver does not support the MuJoCo CPU backend.")
        options["use_mujoco_cpu"] = True
    return options


def create_body_contact_sensor(sensor_type, model, patterns):
    parameters = inspect.signature(sensor_type.__init__).parameters
    keyword = "sensing_bodies" if "sensing_bodies" in parameters else "sensing_obj_bodies"
    return sensor_type(model, **{keyword: patterns}, verbose=False)


def count_contact_sensing_objects(sensor) -> int:
    if hasattr(sensor, "sensing_indices"):
        return len(sensor.sensing_indices)
    return sum(len(objects) for objects in getattr(sensor, "sensing_objs", []))


def contact_total_forces(sensor, model, num_envs: int) -> torch.Tensor:
    """Map total world-space contact force to environments, including friction.

    Newton 1.4+ reports one total-force vector per flat sensing index. Legacy
    sensors report a readings dimension through net_force. Explicit world ids
    prevent label selection order from silently swapping environments.
    """
    if hasattr(sensor, "total_force"):
        if sensor.total_force is None:
            raise RuntimeError("The Newton contact sensor has no total-force output.")
        forces = wp.to_torch(sensor.total_force).clone()
        indices = torch.as_tensor(sensor.sensing_indices, device=forces.device, dtype=torch.long)
        worlds = wp.to_torch(model.body_world)[indices].long()
        if num_envs == 1:
            # A model created without explicit worlds uses the global world -1.
            worlds = torch.where(worlds == -1, 0, worlds)
        if torch.any((worlds < 0) | (worlds >= num_envs)):
            raise ValueError("Contact sensing bodies do not belong to the simulation environments.")
        counts = torch.bincount(worlds, minlength=num_envs)
        if not torch.all(counts == 1):
            raise ValueError("A robot contact sensor must match exactly one body in every environment.")
        result = torch.zeros(num_envs, 3, device=forces.device, dtype=forces.dtype)
        return result.index_add_(0, worlds, forces)

    legacy_forces = getattr(sensor, "net_force", None)
    if legacy_forces is None:
        raise RuntimeError("The Newton contact sensor exposes no supported force output.")
    forces = wp.to_torch(legacy_forces).clone()
    if forces.ndim == 3:
        forces = forces.sum(dim=-2)
    if forces.shape != (num_envs, 3):
        raise ValueError(f"Unexpected legacy Newton contact-force shape: {tuple(forces.shape)}")
    return forces


def allocate_contacts(model):
    """Keep the collision pipeline alive and allocate requested sensor fields."""
    if hasattr(newton, "CollisionPipeline"):
        pipeline = newton.CollisionPipeline(model)
        return pipeline, pipeline.contacts()
    return None, model.contacts()


def validate_scene_support(scene_lib, terrain) -> None:
    """Reject geometry that the Newton backend would otherwise silently omit."""
    if scene_lib.num_objects_per_scene > 0:
        raise NotImplementedError(
            "NewtonSimulator does not spawn SceneLib objects. Use the IsaacLab backend "
            "for indoor scenes and object interactions."
        )
    if terrain is not None:
        heights = getattr(terrain, "height_field_raw", None)
        flat = np.all(np.asarray(heights) == 0) if heights is not None else terrain.is_flat()
        if not flat:
            raise NotImplementedError(
                "NewtonSimulator only creates a ground plane; non-flat terrain collision "
                "geometry is not implemented. Use the IsaacLab backend for this terrain."
            )
