"""SFM bridge to the existing four-actor planner call, with no policy ownership.

Metric EDT is a declared projection of the same fixture's cuboids; it is not a
native map gate. The existing driver alone applies the causal AMP reference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .social_force import SFMParameters, SocialForceTrajectoryGenerator, SourceBinding, StaticDistanceGrid

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
SCENE_SHA = "7971206da6fd50d9981dafd068b6005f326c6e7b6a8768e5a717aeb517573e9f"


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_grid(description, *, source, resolution_m=.1):
    """Conservative cell/cuboid intersection; EDT against occupied cells.

    scipy.ndimage.distance_transform_edt uses physical sampling in meters:
    https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html
    """
    from scipy.ndimage import distance_transform_edt
    if description.get("schema") != "navisaaclab.reconstruction_scene.v1":
        raise ValueError("This bridge only supports the explicit cuboid fixture")
    bounds = np.asarray(description["bounds_xy"], np.float64)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or (bounds[2:] <= bounds[:2]).any()
            or isinstance(resolution_m, bool) or not np.isfinite(resolution_m) or resolution_m <= 0):
        raise ValueError("Invalid grid bounds/resolution")
    cells = (bounds[2:] - bounds[:2]) / resolution_m
    if not np.allclose(cells, np.round(cells), rtol=0, atol=1e-9) or (cells < 2).any():
        raise ValueError("Fixture bounds must be whole grid cells")
    width, height = np.round(cells).astype(int)
    x = bounds[0] + (np.arange(width) + .5) * resolution_m
    y = bounds[1] + (height - np.arange(height) - .5) * resolution_m
    occupied = np.zeros((height, width), bool)
    for box in description["cuboids"]:
        center, size = np.asarray(box["center"], np.float64), np.asarray(box["size"], np.float64)
        if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all()
                or not np.isfinite(size).all() or (size <= 0).any()):
            raise ValueError("Invalid registered physical cuboid")
        # Occupy every cell whose closed square intersects the XY footprint.
        occupied |= ((np.abs(x - center[0]) <= size[0] / 2 + resolution_m / 2)[None, :]
                     & (np.abs(y - center[1]) <= size[1] / 2 + resolution_m / 2)[:, None])
    # Exterior participates in metric EDT even if a future fixture omits walls.
    distance = distance_transform_edt(np.pad(~occupied, 1, constant_values=False),
                                      sampling=resolution_m)[1:-1, 1:-1]
    labels = np.where(occupied, 0, 254).astype(np.uint8)
    return StaticDistanceGrid(distance, labels, origin_xy=bounds[:2], resolution_m=resolution_m,
                              source_id=source["path"], source_sha256=source["sha256"])


class SocialForceSharedPlanner:
    """Full-scene current XY/velocity -> raw51, no AMP/model/RNG construction."""

    def __init__(self, *, world, scene_source, seed):
        if (world.num_envs != 4 or world.shared_world is not True or world.scene_provider is not None
                or world.validation_scene is None):
            raise ValueError("Require the existing shared four-actor cuboid World")
        self._initialize(scene_source, seed, world.device)
        if json.dumps(world.validation_scene.description, sort_keys=True) != json.dumps(self.description, sort_keys=True):
            raise ValueError("Live World's registered geometry differs from SFM grid source")
        self.actual_world_boundary_checked = True

    @classmethod
    def for_CPU_snapshot(cls, *, scene_source, seed):
        instance = cls.__new__(cls)
        instance._initialize(scene_source, seed, "cpu")
        instance.actual_world_boundary_checked = False
        return instance

    def _initialize(self, scene_source, seed, device):
        import yaml
        if Path(scene_source).resolve() != SCENE:
            raise ValueError("Use the same fixed cuboid scene as the Diffusion arm")
        self.scene_identity = file_identity(SCENE)
        if self.scene_identity["sha256"] != SCENE_SHA:
            raise ValueError("Registered fixture changed")
        self.description = yaml.safe_load(SCENE.read_text())
        self.device = str(device)
        self.generator = SocialForceTrajectoryGenerator(
            fixture_grid(self.description, source=self.scene_identity), parameters=SFMParameters(), seed=seed,
            sources=(SourceBinding.bind_sfm(ROOT / "CrowdSim/world/sfm.py"),))
        self.compiled_forward_enabled = False
        self.config = SimpleNamespace(diffusion_steps=0)
        self.last_metadata = None

    def describe(self):
        if file_identity(SCENE) != self.scene_identity:
            raise ValueError("Fixture changed after planner admission")
        return {"schema": "local.public_sfm.shared_AMP_fixture_bridge.v1", "scene": self.scene_identity,
                "implementation": file_identity(__file__), "generator": self.generator.describe(),
                "raw_output": "51XY knots at0,.1,..,5s; original AMP driver applies common causal reference",
                "goal": "Registered static A* five-second endpoint, constant during each predicted horizon",
                "map": ".1m cell/cuboid intersection; metric EDT with occupied exterior pad; conservative raster extension",
                "diffusion_weights_loaded": False, "diffusion_guidance_applied": False,
                "selected_sample_zero_semantics": "Compatibility archive sentinel for deterministic SFM; not a Diffusion sample",
                "RNG_consumed": False, "actual_world_boundary_checked": self.actual_world_boundary_checked,
                "physical_validation_established": False, "author_private_protocol": False}

    def predict_snapshot(self, states, waypoints):
        import torch
        self.describe()
        if (not isinstance(states, torch.Tensor) or states.shape != (4, 5) or states.dtype != torch.float32
                or str(states.device) != self.device or not bool(torch.isfinite(states).all())
                or not isinstance(waypoints, torch.Tensor) or waypoints.shape != (4, 10, 2)
                or waypoints.dtype != torch.float32 or waypoints.device != states.device
                or not bool(torch.isfinite(waypoints).all())):
            raise ValueError("Full ordered four-actor FP32 current states and registered ten goals required")
        prediction = self.generator.generate(states[:, :2].detach().cpu().numpy(),
            states[:, 2:4].detach().cpu().numpy(), waypoints[:, -1].detach().cpu().numpy())
        # Own the array before creating a writable Torch output; no input edits.
        xy = torch.from_numpy(np.array(prediction.raw_xy, copy=True)).to(device=states.device, dtype=torch.float32)
        self.last_metadata = {**prediction.metadata, "bridge": self.describe()}
        return SimpleNamespace(xy=xy, selected_sample=torch.zeros(4, dtype=torch.int64, device=states.device))

    def plan(self, *, states, history, neighbors, raster, history_mask, neighbor_mask, waypoints, waypoint_times,
             num_samples=1, **diffusion_only_options):
        import torch
        if (type(num_samples) is not int or num_samples != 1 or history.shape != (4, 31, 5)
                or neighbors.shape != (4, 3, 31, 5) or raster.shape != (4, 2, 224, 224)
                or history_mask.shape != (4, 31) or neighbor_mask.shape != (4, 3, 31)
                or history_mask.dtype != torch.bool or neighbor_mask.dtype != torch.bool
                or not bool(history_mask[:, -1].all()) or not torch.equal(history[:, -1], states)
                or waypoint_times.shape != (10,) or not torch.allclose(
                    waypoint_times, torch.arange(1, 11, device=waypoint_times.device, dtype=waypoint_times.dtype) / 2,
                    rtol=0, atol=1e-6)):
            raise ValueError("SFM requires one complete joint-scene prediction with common history/goal clocks")
        for value in (history, neighbors, raster):
            if value.device != states.device or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid same-scene planner input")
        expected = {"guidance_weight", "guidance_mode", "near_horizon_mass", "scene_ids", "agent_collision_weight",
                    "agent_collision_radius", "scene_collision_weight", "scene_collision_radius", "trajectory_speed_weight",
                    "max_trajectory_speed_mps", "trajectory_accel_weight", "max_trajectory_accel_mps2", "scene_bounds_xy",
                    "obstacle_centers_xy", "obstacle_sizes_xy"}
        if set(diffusion_only_options) != expected:
            raise ValueError("Unexpected Diffusion-call contract; SFM does not silently consume new objectives")
        scene_ids = diffusion_only_options["scene_ids"]
        if scene_ids is None or scene_ids.shape != (4,) or not torch.equal(scene_ids, torch.zeros_like(scene_ids)):
            raise ValueError("A complete single shared scene is required")
        prediction = self.predict_snapshot(states, waypoints)
        self.last_metadata["diffusion_options_accepted_but_not_applied"] = sorted(expected)
        return prediction
