"""Bounded CPU Tensor callback checks: no NN construction, weights or SDK."""
from dataclasses import fields
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.diffusion.paper import PaperDiffusionPlan, scene_agent_collision_cost
from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "docs/style_improvement_20261004/reachable_candidate_selection_staging_v1/selection_runtime.py"
spec = importlib.util.spec_from_file_location("test_common50_runtime", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize("domain", ["matched_raw50", "reachable50"])
def test_real_shape_callback_world_fields_adapter_archive_costs_and_rng(domain):
    current = torch.tensor([[-1.2, -.3, .35, .2, .5], [-.25, .1, .2, -.25, -.8],
                            [.7, -.1, -.15, .3, 1.1], [1.6, .2, .1, .4, -1.4]])
    times = torch.arange(1, 53, dtype=torch.float32)[None, None]*.1
    sample = torch.arange(10, dtype=torch.float32)[None, :, None]
    actor = torch.arange(4, dtype=torch.float32)[:, None, None]
    vx = .3+.06*sample+.04*actor+.07*(sample-4)*times.cos()
    vy = (sample-4.5)*.08+.04*actor*times.cos()
    x = (.3+.06*sample+.04*actor)*times+.07*(sample-4)*times.sin()
    y = (sample-4.5)*.08*times+.04*actor*times.sin()
    physical = torch.stack((x, y, vx, vy, torch.atan2(vy, vx)+.02*times), -1)
    assert physical.shape == (4, 10, 52, 5) and physical.dtype == torch.float32
    cosine, sine = current[:, 4].cos(), current[:, 4].sin()
    rotation = torch.stack((torch.stack((cosine, sine), -1),
                            torch.stack((-sine, cosine), -1)), -2)
    world_xy = physical[:, :, :50, :2] @ rotation[:, None]+current[:, None, None, :2]
    raw = torch.cat((current[:, None, None, :2].expand(-1, 10, 1, -1), world_xy), 2)
    world_uv = physical[:, :, :50, 2:4] @ rotation[:, None]
    all_uv = torch.cat((current[:, None, None, 2:4].expand(-1, 10, 1, -1), world_uv), 2)
    all_yaw = torch.remainder(physical[:, :, :50, 4]+current[:, None, None, 4]+torch.pi, 2*torch.pi)-torch.pi
    all_yaw = torch.cat((current[:, None, 4:5].expand(-1, 10, -1), all_yaw), 2)
    initial = torch.full((4,), 9, dtype=torch.long)
    rows = torch.arange(4)
    prediction = PaperDiffusionPlan(raw[rows, initial], all_yaw[rows, initial],
                                   all_uv[rows, initial].norm(dim=-1), all_uv[rows, initial],
                                   torch.arange(51)*.1, physical, initial)
    previous_xy = current[:, :2]+torch.tensor([[.2, .1], [-.15, .05], [.1, -.2], [-.1, .1]])
    previous_uv = current[:, 2:4]+torch.tensor([[.05, 0.], [0., .03], [.02, 0.], [0., -.04]])
    adapter = CausalReferenceAdapter(ReferenceAdapterConfig(
        max_speed_mps=1.5, max_acceleration_mps2=2., initial_speed_policy="brake"))
    scene_ids = torch.zeros(4, dtype=torch.long)
    guide_times = torch.arange(1, 11)*.5
    guide = current[:, None, :2]+guide_times[None, :, None]*torch.tensor([.4, .05])
    kwargs = {"guidance_weight": 20., "agent_collision_weight": 5., "agent_collision_radius": .6,
              "scene_collision_weight": 10., "scene_collision_radius": .8, "near_horizon_mass": 0.,
              "scene_bounds_xy": torch.tensor([-7., -6., 7., 6.]),
              "obstacle_centers_xy": torch.tensor([[-.1, .8], [2.3, 1.8]]),
              "obstacle_sizes_xy": torch.tensor([[.6, .8], [.8, 1.]]), "scene_collision_cost_fn": None}
    snapshots = [value.clone() for value in (physical, current, previous_xy, previous_uv)]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    revised, adapted, extra = runtime.ReferenceSelector(domain)(
        prediction, {"states": current}, adapter, previous_xy, previous_uv, scene_ids, guide, guide_times, kwargs)
    selected = revised.selected_sample
    assert selected.shape == (4,) and selected.dtype == torch.long
    assert torch.equal(selected, selected[:1].expand(4)) and bool(((selected >= 0) & (selected < 10)).all())
    for actual, expected in ((revised.xy, raw[rows, selected]), (revised.yaw, all_yaw[rows, selected]),
                             (revised.velocity, all_uv[rows, selected]),
                             (revised.vel, all_uv[rows, selected].norm(dim=-1))):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
    assert revised.sample_states is physical and revised.time is prediction.time
    assert revised.reference == prediction.reference and torch.equal(prediction.selected_sample, initial)
    expected_adapter = adapter.build(revised.xy, previous_xy, previous_uv)
    for field in fields(expected_adapter):
        assert torch.equal(getattr(adapted, field.name), getattr(expected_adapter, field.name)), field.name
    shapes = {"candidate_raw_xy": (4, 10, 51, 2), "candidate_reference_xy": (4, 10, 151, 2),
              "candidate_reference_velocity_xy": (4, 10, 151, 2), "candidate_goal_cost": (4, 10),
              "candidate_static_cost": (4, 10), "candidate_pair_cost": (4, 10),
              "candidate_joint_cost": (4, 10), "candidate_waypoints": (4, 10, 2), "candidate_scene_ids": (4,)}
    assert set(extra) == set(shapes)
    assert all(value.shape == shapes[key] for key, value in extra.items())
    assert torch.equal(extra["candidate_scene_ids"], scene_ids) and torch.equal(extra["candidate_waypoints"], guide)
    torch.testing.assert_close(extra["candidate_raw_xy"], raw, rtol=1e-6, atol=2e-6)
    assert torch.equal(adapted.xy, extra["candidate_reference_xy"][rows, selected])
    scoring = (extra["candidate_raw_xy"][:, :, 1:] if domain == "matched_raw50"
               else extra["candidate_reference_xy"][:, :, 3::3])
    upper = scene_agent_collision_cost(scoring, scene_ids, .6)
    assert bool(upper.sum() > 0) and bool(extra["candidate_static_cost"].sum() > 0)
    torch.testing.assert_close(extra["candidate_pair_cost"], (upper.sum(0)+upper.sum(1))/2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(extra["candidate_pair_cost"].sum(0), upper.sum((0, 1)), rtol=1e-5, atol=1e-5)
    joint = (20*extra["candidate_goal_cost"]+10*extra["candidate_static_cost"]).sum(0)+5*upper.sum((0, 1))
    torch.testing.assert_close(extra["candidate_joint_cost"], joint[None].expand(4, -1), rtol=1e-6, atol=1e-5)
    assert int(selected[0]) == int(extra["candidate_joint_cost"][0].argmin())
    assert all(torch.equal(before, after) for before, after in zip(snapshots, (physical, current, previous_xy, previous_uv)))
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), cpu_rng)
    numpy_after = np.random.get_state()
    assert numpy_rng[0] == numpy_after[0] and np.array_equal(numpy_rng[1], numpy_after[1])
    assert numpy_rng[2:] == numpy_after[2:]
