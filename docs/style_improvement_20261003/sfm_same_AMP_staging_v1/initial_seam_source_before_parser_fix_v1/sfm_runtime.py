"""SFM factory for the isolated evaluator; no App/NN at module import."""
from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sfm_target_generator import SFMConfig, SFMTargetGenerator, WorldEDTMap


HERE = Path(__file__).resolve().parent


def prepare_sfm_arguments(args):
    """Convert only the inherited CLI's planner syntax; retain physics options."""
    if (not args.evaluate_steps or not args.resume or not args.scene_config
            or args.trajectory_source not in {"diffusion", "sfm"}
            or args.replan_steps != 15 or not args.synchronized_replanning
            or args.route_guidance != "timed_path" or args.reference_adapter != "causal_accel"
            or args.agent_collision_weight <= 0 or args.compile_diffusion_forward):
        raise ValueError("SFM requires isolated evaluation, saved AMP, scene, same15-step shared clock and causal timed-route recipe")
    args.trajectory_source = "sfm"
    args.diffusion_checkpoint = None
    args.diffusion_samples = 0
    return args


class SFMFactory:
    """Explicit preflight/runtime hooks; never replace the Diffusion class."""

    def __init__(self, config=None, *, flat_resolution_m=.1):
        self.config = SFMConfig() if config is None else config
        if not isinstance(self.config, SFMConfig):
            raise ValueError("SFMConfig required")
        self.config.validate()
        if type(flat_resolution_m) not in (int, float) or not np.isfinite(flat_resolution_m) or flat_resolution_m <= 0:
            raise ValueError("Positive finite flat fixture map resolution required")
        self.flat_resolution_m = float(flat_resolution_m)

    def preflight(self, args, profile):
        """All original resource checks except the unused Diffusion record.

        Compile a private namespace copy of the exact original function. The
        original module/function is never patched; no checkpoint is unpickled.
        """
        prepare_sfm_arguments(args)
        from CrowdSim.humanoid.application import resources

        original = inspect.getsource(resources.preflight_amp_resources)
        before = '''    records = [inspect_resource("profile", args.config),
               inspect_resource("manifest", args.manifest),
               inspect_resource("diffusion_checkpoint", args.diffusion_checkpoint, kind="torch")]
'''
        after = '''    records = [inspect_resource("profile", args.config),
               inspect_resource("manifest", args.manifest)]
'''
        if original.count(before) != 1:
            raise ValueError("Original unused Diffusion resource boundary changed")
        revised = original.replace(before, after)
        if revised.replace(after, before) != original:
            raise ValueError("Only the unused Diffusion record may be removed")
        namespace = dict(vars(resources))
        exec(compile(revised, str(HERE / "sfm_runtime.py"), "exec"), namespace)
        return namespace["preflight_amp_resources"](args, profile)

    def __call__(self, *, world, args, native_scene):
        prepare_sfm_arguments(args)
        if not world.shared_world or world.validation_scene is None:
            raise ValueError("SFM requires the original shared physical validation scene")
        if native_scene is not None:
            provider = native_scene.provider
            # NativeGridScene already admits source-bound .1m/yaw0 ROS export.
            scene_map = WorldEDTMap(provider._map == 254, origin_xy_m=provider._map_origin_xy,
                                    resolution_m=native_scene.resolution, origin_yaw_rad=0.)
            map_description = {"kind": "admitted native raw known-free labels==254",
                               "unknown_cells_blocked": True, "radius_inflation": False}
        else:
            scene = world.validation_scene
            bounds = scene.bounds.detach().cpu().numpy().astype(np.float64)
            extents = (bounds[2:] - bounds[:2]) / self.flat_resolution_m
            dimensions = np.rint(extents).astype(np.int64)
            if np.any(dimensions < 2) or not np.allclose(extents, dimensions, rtol=0, atol=1e-6):
                raise ValueError("Flat bounds must exactly span the declared whole occupancy cells")
            width, height = dimensions
            columns = bounds[0] + (np.arange(width) + .5) * self.flat_resolution_m
            rows = bounds[1] + (height - .5 - np.arange(height)) * self.flat_resolution_m
            yy, xx = np.meshgrid(rows, columns, indexing="ij")
            occupied = np.zeros((height, width), dtype=bool)
            centers, sizes = scene.centers.detach().cpu().numpy(), scene.sizes.detach().cpu().numpy()
            for center, size in zip(centers, sizes):
                occupied |= ((np.abs(xx - center[0]) <= size[0] / 2)
                             & (np.abs(yy - center[1]) <= size[1] / 2))
            scene_map = WorldEDTMap(~occupied, origin_xy_m=bounds[:2], resolution_m=self.flat_resolution_m)
            map_description = {"kind": "same physical cuboid footprints/bounds at cell centers",
                               "unknown_cells_blocked": True, "radius_inflation": False,
                               "bounds_xy_m": bounds.tolist(), "obstacle_count": len(centers)}
        map_description.update(shape_hw=list(scene_map.known_free.shape), origin_xy_m=scene_map.origin_xy_m.tolist(),
                               resolution_m=scene_map.resolution_m, origin_yaw_rad=0.,
                               image_coordinates="top-left row0=maxY,col0=minX",
                               EDT="raw free centre distance in metres; blocked padded exterior; no inflation")
        return _SFMRuntime(SFMTargetGenerator({0: scene_map}, self.config), world.num_envs, map_description)


class _SFMRuntime:
    """Tensor bridge after the parent's App/world exists; CPU predicts full scene."""

    def __init__(self, generator, actors, map_description):
        self.generator, self.actors = generator, actors
        self._description = {
            "trajectory_source": "sfm", "factory": str(Path(__file__).resolve()),
            "target_generator": str(HERE / "sfm_target_generator.py"),
            "configuration": asdict(generator.config), "map": map_description,
            "prediction_dt_s": .1, "prediction_steps": 50, "horizon_s": 5.,
            "nominal_external_replan_dt_s": .5, "FOV": False,
            "same_body_feedback": "world.planar actual full-scene state once before each synchronous prediction",
            "waypoint_progress": "private prediction index only; real arrival remains original post-physics route logic",
            "selected_sample_0": "single deterministic target index, not a Diffusion draw",
            "diffusion_model_constructions": 0, "diffusion_forward_calls": 0, "diffusion_samples": 0,
            "full_scene_prediction_calls": 0, "predicted_actor_plans": 0,
            "predicted_domain_exit_events": [], "initial_overspeed_replans": [],
            "blocked_or_unknown_predicted_knots": 0,
            "author_exact_SFM_parameters": False, "physical_quality_verified": False,
        }

    def describe(self):
        # Shared mutable metadata records only observed calls, not guessed counts.
        return self._description

    def generate(self, states, route_waypoints, *, control_step):
        import torch

        if states.shape != (self.actors, 5) or route_waypoints.shape[0] != self.actors:
            raise ValueError("One actual state and ordered path per original scene actor required")
        plan = self.generator.generate(states.detach().cpu().numpy(), route_waypoints.detach().cpu().numpy(),
                                       scene_ids=np.zeros(self.actors, dtype=np.int64),
                                       actor_ids=np.arange(self.actors, dtype=np.int64))
        description = self._description
        description["full_scene_prediction_calls"] += 1
        description["predicted_actor_plans"] += self.actors
        description["blocked_or_unknown_predicted_knots"] += sum(plan.diagnostics["blocked_or_unknown_knot_counts"])
        description["predicted_domain_exit_events"].extend(
            {"control_step": int(control_step), **event} for event in plan.diagnostics["domain_exit_events"])
        if plan.diagnostics["initial_overspeed_actor_ids"]:
            description["initial_overspeed_replans"].append(
                {"control_step": int(control_step), "actor_ids": plan.diagnostics["initial_overspeed_actor_ids"]})
        return SimpleNamespace(xy=torch.as_tensor(plan.xy, dtype=states.dtype, device=states.device),
                               selected_sample=torch.zeros(self.actors, dtype=torch.long, device=states.device))


def sfm_runtime_metadata(report, target_generator, historical_provenance):
    """Keep complete AMP training provenance untouched; identify actual target source."""
    report["method"] = "SFM targets + the same saved physics AMP controller; local future-target adaptation"
    report["trajectory_source"] = "sfm"
    report["diffusion_guidance"] = None
    report["navigation_guidance"].update(
        source="sfm", mode="ordered existing A* waypoints for private SFM proximity progression",
        guidance_total_weight=None, guidance_per_waypoint_weight=None,
        effective_per_waypoint_goal_weights=None, waypoint_weighting=None)
    report["runtime_execution"].update(
        trajectory_source="sfm", diffusion_steps=0, diffusion_used_for_rollouts=False,
        diffusion_samples_per_actor=0, diffusion_model_constructions=0, diffusion_forward_calls=0,
        compiled_forward=False, compilation=None, compiled_XY_benchmark_max_difference_m=None,
        compiled_bit_identical=None, consumed_planner_output="SFM world XY only",
        sample_filtering="single deterministic target; selected_sample=0 is a plan index, not a diffusion draw")
    report["sfm_target_generator"] = target_generator.describe()
    report["evaluation_target_generator_change"] = {
        "training_trajectory_source": historical_provenance.get("trajectory_source", "diffusion"),
        "historical_training_diffusion_checkpoint": historical_provenance.get("diffusion_checkpoint"),
        "historical_training_diffusion_sha256": historical_provenance.get("diffusion_sha256"),
        "actual_evaluation_trajectory_source": "sfm", "actual_diffusion_checkpoint": None,
        "full_training_provenance_preserved": True,
        "unused_inherited_diffusion_guidance_arguments": True,
    }
