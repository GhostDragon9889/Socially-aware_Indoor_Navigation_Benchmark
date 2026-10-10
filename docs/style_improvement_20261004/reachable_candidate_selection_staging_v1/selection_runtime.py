"""Lazy candidate-ranking callback; module import uses only the standard library."""
from __future__ import annotations

from dataclasses import replace


class ReferenceSelector:
    """Rank one already-generated scene bank; never run a planner or draw RNG."""

    def __init__(self, selection_domain):
        if selection_domain not in ("matched_raw50", "reachable50"):
            raise ValueError("Choose matched_raw50 or reachable50")
        self.selection_domain = selection_domain

    def describe(self):
        return {
            "schema": "local_reference.common50_candidate_selection.v1",
            "selection_domain": self.selection_domain,
            "generation": "one original S10 DDPM call; original52-node clean guidance unchanged",
            "scoring": "same fifty future nodes .1..5.0s; common sample index for all four actors",
            "weights": {"goal": 20., "peer": 5., "static": 10.},
            "radii_m": {"peer": .6, "static": .8},
            "candidate_pair_cost": "unweighted half incident upper-triangle pair cost per actor; sum actors equals unordered scene-pair cost",
            "candidate_joint_cost": "weighted scene total repeated per actor; do not sum those repeated rows",
            "all_generated_candidates_retained": True,
            "unsafe_candidates_filtered": False,
            "historical52_node_selection_unchanged": False,
            "reference_teleport_or_physics_change": False,
            "contact_or_navigation_guarantee": False,
        }

    def __call__(self, prediction, inputs, reference_adapter, previous_xy,
                 previous_velocity, scene_ids, guide, waypoint_times, kwargs):
        # Called only by the existing driver's replan, after its single App starts.
        import torch
        from CrowdSim.humanoid.diffusion.paper import _world_prediction_xy
        from CrowdSim.humanoid.training.reachable_selection import select_reachable_candidates
        from CrowdSim.humanoid.training.reference import DenseReferencePlan

        physical = prediction.sample_states
        if (physical.shape != (4, 10, 52, 5) or physical.dtype != torch.float32
                or not bool(torch.isfinite(physical).all())):
            raise ValueError("The unchanged FP32 four-actor S10 bank is required")
        current = torch.as_tensor(inputs["states"], dtype=physical.dtype, device=physical.device)
        fixed = {"guidance_weight": 20., "agent_collision_weight": 5.,
                 "agent_collision_radius": .6, "scene_collision_weight": 10.,
                 "scene_collision_radius": .8, "near_horizon_mass": 0.}
        if any(kwargs[key] != value for key, value in fixed.items()):
            raise ValueError("Original guide20/peer5/static10/radii/eta0 are required")
        if (kwargs["scene_collision_cost_fn"] is not None
                or reference_adapter.config.max_speed_mps != 1.5
                or reference_adapter.config.max_acceleration_mps2 != 2.
                or reference_adapter.config.initial_speed_policy != "brake"):
            raise ValueError("Use the original fixture geometry and causal-reference recipe")
        batch, samples = physical.shape[:2]
        future = _world_prediction_xy(physical[:, :, :50], current, samples)
        world_xy = torch.cat((current[:, None, None, :2].expand(-1, samples, 1, -1), future), dim=2)
        scored = select_reachable_candidates(
            world_xy, reference_adapter, previous_xy, previous_velocity,
            scene_ids, guide, waypoint_times, selection_domain=self.selection_domain,
            waypoint_weight=kwargs["guidance_weight"],
            agent_collision_weight=kwargs["agent_collision_weight"],
            agent_collision_radius=kwargs["agent_collision_radius"],
            scene_collision_weight=kwargs["scene_collision_weight"],
            scene_collision_radius=kwargs["scene_collision_radius"],
            scene_bounds_xy=kwargs["scene_bounds_xy"],
            obstacle_centers_xy=kwargs["obstacle_centers_xy"],
            obstacle_sizes_xy=kwargs["obstacle_sizes_xy"],
            scene_collision_cost_fn=kwargs["scene_collision_cost_fn"])
        selected = scored["selected_sample"]
        rows = torch.arange(batch, device=physical.device)
        picked = physical[rows, selected, :50]
        cosine, sine = current[:, 4].cos()[:, None], current[:, 4].sin()[:, None]
        velocity = torch.stack((picked[..., 2]*cosine-picked[..., 3]*sine,
                                picked[..., 2]*sine+picked[..., 3]*cosine), dim=-1)
        velocity = torch.cat((current[:, None, 2:4], velocity), dim=1)
        yaw = torch.remainder(picked[..., 4]+current[:, None, 4]+torch.pi, 2*torch.pi)-torch.pi
        revised = replace(prediction, xy=world_xy[rows, selected],
                          yaw=torch.cat((current[:, 4:5], yaw), dim=1),
                          velocity=velocity, vel=velocity.norm(dim=-1), selected_sample=selected)
        # Slice the existing adapter's results, rather than adapting a second time.
        raw_times = torch.arange(51, dtype=physical.dtype, device=physical.device)*reference_adapter.config.raw_knot_dt_s
        dense_times = torch.arange(151, dtype=physical.dtype, device=physical.device)*reference_adapter.config.control_dt_s
        adapted = DenseReferencePlan(
            world_xy[rows, selected], raw_times,
            scored["adapted_xy"][rows, selected], scored["adapted_velocity"][rows, selected],
            dense_times, scored["raw_dense_xy"][rows, selected],
            scored["initial_overspeed"][rows, selected], scored["speed_violation"][rows, selected],
            scored["deviation_m"][rows, selected])
        peer = scored["peer_cost"]
        extras = {
            "candidate_raw_xy": world_xy,
            "candidate_reference_xy": scored["adapted_xy"],
            "candidate_reference_velocity_xy": scored["adapted_velocity"],
            "candidate_goal_cost": scored["goal_cost"],
            "candidate_static_cost": scored["static_cost"],
            "candidate_pair_cost": (peer.sum(0)+peer.sum(1))/2,
            "candidate_joint_cost": scored["scene_sample_cost"],
            "candidate_waypoints": torch.as_tensor(guide, dtype=physical.dtype, device=physical.device),
            "candidate_scene_ids": torch.as_tensor(scene_ids, device=physical.device),
        }
        return revised, adapted, extras
