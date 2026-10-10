"""Experimental joint ranking of existing candidates after reference adaptation.

This changes neither generated candidates nor RNG/control/physical state. The
50 future scoring nodes cover .1..5.0 s, whereas the original planner also
scores its unsaved 5.1/5.2 s nodes. matched_raw50 and reachable50 use the same
scorer to isolate scoring domain; neither is the historical 52-node baseline.
"""
from __future__ import annotations

import math


def select_reachable_candidates(world_xy, adapter, reference_xy, reference_velocity,
                                scene_ids, waypoints, waypoint_times, *,
                                selection_domain="reachable50",
                                waypoint_weight=20., waypoint_weights=None,
                                agent_collision_weight=5., agent_collision_radius=.6,
                                scene_collision_weight=10., scene_collision_radius=.8,
                                scene_bounds_xy=None, obstacle_centers_xy=None,
                                obstacle_sizes_xy=None, scene_collision_cost_fn=None):
    """Adapt worldXY[B,S,51,2], score its future 10Hz nodes, share each scene's S.

    Returns every adapted trajectory and every cost, including unsafe ones.
    matched_raw50 scores the resampled raw geometry; reachable50 scores the
    adapted geometry. Both retain all adaptations and share weights/node times.
    Scores are geometric proxies, not collider contact or feasibility claims;
    there is no fallback. The caller supplies the same continuity state for all
    candidates of each actor. Ordinary replans use the last executed reference;
    only episode initialization/reset uses the physical root. S>1 is restricted
    to scenes with at least two participating actors. S=1 still preserves the
    original adapter output and selects zero, for interface verification.
    """
    import torch
    from ..diffusion.paper import (
        _scene_callback_cost, _validate_scene_collision_callback,
        reduce_waypoint_distances, scene_agent_collision_cost,
        scene_geometry_collision_cost,
    )

    if (not isinstance(world_xy, torch.Tensor) or world_xy.ndim != 4
            or world_xy.shape[2:] != (51, 2) or min(world_xy.shape[:2]) < 1
            or world_xy.dtype not in (torch.float32, torch.float64)):
        raise ValueError("world_xy must be floating[B,S,51,2]")
    batch, samples = world_xy.shape[:2]
    if selection_domain not in ("matched_raw50", "reachable50"):
        raise ValueError("selection_domain must be matched_raw50 or reachable50")
    cfg = adapter.config
    if not all(math.isclose(getattr(cfg, name), expected, abs_tol=1e-10, rel_tol=0)
               for name, expected in (("control_dt_s", 1/30), ("raw_knot_dt_s", .1), ("horizon_s", 5.))):
        raise ValueError("Reachable ranking requires the original 30Hz/10Hz five-second adapter")
    for value in (reference_xy, reference_velocity):
        if (not isinstance(value, torch.Tensor) or value.shape != (batch, 2)
                or value.dtype != world_xy.dtype or value.device != world_xy.device):
            raise ValueError("Continuity states must match world_xy dtype/device and be[B,2]")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (waypoint_weight, agent_collision_weight, scene_collision_weight)):
        raise ValueError("Cost weights must be finite and nonnegative")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (agent_collision_radius, scene_collision_radius)):
        raise ValueError("Geometry radii must be finite and positive")
    ids = torch.as_tensor(scene_ids, device=world_xy.device)
    if ids.shape != (batch,) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("scene_ids must be integer[B]")
    if samples > 1 and any(int((ids == scene).sum()) < 2 for scene in ids.unique()):
        raise ValueError("Multi-candidate ranking needs at least two actors per participating scene")
    guide = torch.as_tensor(waypoints, dtype=world_xy.dtype, device=world_xy.device)
    times = torch.as_tensor(waypoint_times, dtype=world_xy.dtype, device=world_xy.device)
    if (guide.ndim != 3 or guide.shape[0] != batch or guide.shape[-1] != 2 or guide.shape[1] < 1
            or times.shape != (guide.shape[1],) or not bool(torch.isfinite(guide).all())
            or not bool(torch.isfinite(times).all()) or not bool(((times > 0) & (times <= 5)).all())
            or not torch.allclose(times*30, (times*30).round(), atol=1e-4, rtol=0)):
        raise ValueError("World guides[B,K,2] need finite positive exact30Hz times through5s")
    indices = (times*30).round().long()
    adapted = adapter.build(world_xy.detach().reshape(batch*samples, 51, 2),
                            reference_xy.detach().repeat_interleave(samples, 0),
                            reference_velocity.detach().repeat_interleave(samples, 0))
    xy = adapted.xy.reshape(batch, samples, 151, 2)
    raw_dense = adapted.raw_sampled_xy.reshape(batch, samples, 151, 2)
    scoring_xy = raw_dense if selection_domain == "matched_raw50" else xy
    future = scoring_xy[:, :, 3::3]  # .1..5.0s; neverscoret0 as a future collision node.
    goal = reduce_waypoint_distances(torch.linalg.vector_norm(scoring_xy[:, :, indices] - guide[:, None], dim=-1), waypoint_weights)
    peer = scene_agent_collision_cost(future, ids, agent_collision_radius)
    geometry = (scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy)
    _validate_scene_collision_callback(scene_collision_cost_fn, scene_collision_weight, geometry)
    static = world_xy.new_zeros(batch, samples)
    if scene_collision_weight > 0:
        if scene_collision_cost_fn is not None:
            static = _scene_callback_cost(future, scene_collision_cost_fn, scene_collision_radius)
        else:
            if any(value is None for value in geometry):
                raise ValueError("Static scoring requires all world bounds/box fields or a native callback")
            bounds, centers, sizes = (torch.as_tensor(value, dtype=world_xy.dtype, device=world_xy.device) for value in geometry)
            if (bounds.shape != (4,) or centers.ndim != 2 or centers.shape[1] != 2 or sizes.shape != centers.shape
                    or not bool((bounds[2:] > bounds[:2]).all()) or not bool((sizes > 0).all())
                    or not all(bool(torch.isfinite(value).all()) for value in (bounds, centers, sizes))):
                raise ValueError("Expected finite worldbounds[4] and positive boxsizes[M,2]")
            static = scene_geometry_collision_cost(future, bounds, centers, sizes, scene_collision_radius)
    independent = waypoint_weight*goal + scene_collision_weight*static
    costs = torch.empty_like(independent)
    selected = torch.zeros(batch, dtype=torch.long, device=world_xy.device)
    for scene in ids.unique():
        members = ids == scene
        scene_cost = independent[members].sum(0) + agent_collision_weight*peer[members][:, members].sum((0, 1))
        costs[members] = scene_cost
        selected[members] = scene_cost.argmin()
    return {"adapted_xy": xy, "raw_dense_xy": raw_dense, "selection_domain": selection_domain,
            "adapted_velocity": adapted.velocity.reshape(batch, samples, 151, 2),
            "deviation_m": adapted.deviation_m.reshape(batch, samples, 151),
            "initial_overspeed": adapted.initial_overspeed.reshape(batch, samples),
            "speed_violation": adapted.speed_violation.reshape(batch, samples, 151),
            "goal_cost": goal, "peer_cost": peer, "static_cost": static,
            "independent_cost": independent, "scene_sample_cost": costs,
            "selected_sample": selected, "waypoint_indices_30Hz": indices}
