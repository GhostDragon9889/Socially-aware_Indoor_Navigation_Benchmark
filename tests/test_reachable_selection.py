"""CPU geometry and state handoff tests; no NN, weights or simulator objects."""
import random

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.training.reachable_selection import select_reachable_candidates
from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig


def candidates(samples=2, dtype=torch.float64):
    start = torch.tensor([[-2., 0.], [2., 0.]], dtype=dtype)
    raw = start[:, None, None].repeat(1, samples, 51, 1)
    if samples > 1:
        fraction = torch.arange(51, dtype=dtype) / 50
        raw[:, 0, :, 0] *= 1-fraction
    return raw, start, torch.zeros_like(start)


def ranking(raw, position, velocity, **kwargs):
    return select_reachable_candidates(
        raw, CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5)),
        position, velocity, torch.zeros(len(position), dtype=torch.long),
        torch.zeros(len(position), 1, 2, dtype=raw.dtype), torch.tensor([5.], dtype=raw.dtype),
        scene_collision_weight=0., **kwargs)


def test_joint_selection_uses_same_candidate_and_original_cost_function():
    raw, position, velocity = candidates()
    before = raw.clone(), position.clone(), velocity.clone()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = ranking(raw, position, velocity, agent_collision_weight=100.)
    # This is the first helper invocation: include its lazy geometry import.
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and np.array_equal(numpy_rng[1], after_numpy[1])
    assert numpy_rng[2:] == after_numpy[2:]
    from CrowdSim.humanoid.diffusion.paper import scene_agent_collision_cost

    assert result["adapted_xy"].shape == (2, 2, 151, 2)
    assert result["selected_sample"].tolist() == [1, 1]
    expected = scene_agent_collision_cost(result["adapted_xy"][:, :, 3::3], torch.zeros(2, dtype=torch.long), .6)
    assert torch.equal(result["peer_cost"], expected)
    assert torch.equal(result["scene_sample_cost"][0], result["scene_sample_cost"][1])
    assert torch.equal(result["scene_sample_cost"][0], result["independent_cost"].sum(0) + 100.*expected.sum((0, 1)))
    assert all(torch.equal(old, new) for old, new in zip(before, (raw, position, velocity)))


def test_continuity_state_changes_adapted_cost_without_changing_candidates():
    raw, position, velocity = candidates()
    result = ranking(raw, position, velocity)
    shifted = position + torch.tensor([[.5, 0.], [0., 0.]], dtype=position.dtype)
    other = ranking(raw, shifted, velocity)
    assert torch.equal(other["adapted_xy"][:, :, 0], shifted[:, None].expand(-1, 2, -1))
    assert not torch.equal(result["goal_cost"], other["goal_cost"])
    assert not torch.equal(result["scene_sample_cost"], other["scene_sample_cost"])


def test_single_candidate_exact_original_adapter_and_rng_states_unchanged():
    raw, position, velocity = candidates(samples=1)
    adapter = CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5))
    expected = adapter.build(raw[:, 0], position, velocity)
    # Helpers are loaded here without constructing any of their NN classes.
    from CrowdSim.humanoid.diffusion.paper import scene_agent_collision_cost  # noqa: F401

    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    actual = ranking(raw, position, velocity)
    assert torch.equal(actual["adapted_xy"][:, 0], expected.xy)
    assert torch.equal(actual["adapted_velocity"][:, 0], expected.velocity)
    assert actual["selected_sample"].tolist() == [0, 0]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_time_origin_waypoint_indices_speed_and_acceleration():
    raw, position, velocity = candidates()
    raw[:, 0, 26:, 1] = 5.
    times = torch.tensor([.1, .5, 5.], dtype=raw.dtype)
    guide = position[:, None].expand(-1, 3, -1).clone()
    adapter = CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5))
    result = select_reachable_candidates(raw, adapter, position, velocity, [0, 0], guide, times, scene_collision_weight=0.)
    assert result["waypoint_indices_30Hz"].tolist() == [3, 15, 150]
    xy, speed = result["adapted_xy"], result["adapted_velocity"]
    assert torch.equal(xy[:, :, 0], position[:, None].expand(-1, 2, -1))
    assert torch.allclose(torch.diff(xy, dim=2)*30, speed[:, :, 1:], atol=1e-12, rtol=0)
    assert float(speed.norm(dim=-1).max()) <= 1.5 + 1e-12
    assert float((torch.diff(speed, dim=2)*30).norm(dim=-1).max()) <= 2. + 1e-12
    assert not result["speed_violation"].any()
    expected_goal = (xy[:, :, [3, 15, 150]]-guide[:, None]).norm(dim=-1).mean(-1)
    assert torch.equal(result["goal_cost"], expected_goal)
    with pytest.raises(ValueError, match="positive exact30Hz"):
        select_reachable_candidates(raw, adapter, position, velocity, [0, 0], guide, [0., .5, 5.], scene_collision_weight=0.)


def test_native_callback_sees_same_world_future_nodes_once():
    raw, position, velocity = candidates()
    seen = []

    def callback(world, *, radius):
        assert world.shape == (2, 2, 50, 2) and radius == .8
        seen.append(world.clone())
        return world[..., 1].square().sum(-1)

    result = select_reachable_candidates(raw, CausalReferenceAdapter(), position, velocity,
                                        [0, 0], position[:, None], [.5], scene_collision_cost_fn=callback)
    assert len(seen) == 1
    assert torch.equal(seen[0], result["adapted_xy"][:, :, 3::3])
    assert torch.equal(result["static_cost"], seen[0][..., 1].square().sum(-1))


def test_static_cost_reuses_original_geometry_and_scene_partition():
    from CrowdSim.humanoid.diffusion.paper import scene_geometry_collision_cost

    raw, position, velocity = candidates()
    raw, position, velocity = raw.repeat(2, 1, 1, 1), position.repeat(2, 1), velocity.repeat(2, 1)
    bounds = torch.tensor([-10., -10., 10., 10.], dtype=raw.dtype)
    centers, sizes = torch.tensor([[0., 0.]], dtype=raw.dtype), torch.tensor([[1., 1.]], dtype=raw.dtype)
    result = select_reachable_candidates(raw, CausalReferenceAdapter(), position, velocity, [0, 0, 1, 1],
                                        position[:, None], [5.], scene_bounds_xy=bounds,
                                        obstacle_centers_xy=centers, obstacle_sizes_xy=sizes)
    expected = scene_geometry_collision_cost(result["adapted_xy"][:, :, 3::3], bounds, centers, sizes, .8)
    assert torch.equal(result["static_cost"], expected)
    assert result["selected_sample"][0] == result["selected_sample"][1]
    assert result["selected_sample"][2] == result["selected_sample"][3]
    assert not result["peer_cost"][:2, 2:].any()


def test_reject_single_actor_multisample_and_wrong_adapter_clock():
    raw, position, velocity = candidates()
    with pytest.raises(ValueError, match="two actors"):
        select_reachable_candidates(raw[:1], CausalReferenceAdapter(), position[:1], velocity[:1], [0], position[:1, None], [5.], scene_collision_weight=0.)
    with pytest.raises(ValueError, match="30Hz/10Hz"):
        select_reachable_candidates(raw, CausalReferenceAdapter(ReferenceAdapterConfig(control_dt_s=.05)),
                                    position, velocity, [0, 0], position[:, None], [5.], scene_collision_weight=0.)


def test_two_domains_share_candidates_clock_cost_function_and_adaptations():
    from CrowdSim.humanoid.diffusion.paper import scene_agent_collision_cost

    raw, position, velocity = candidates()
    position = position + torch.tensor([[.4, .1], [-.2, .3]], dtype=position.dtype)
    matched = ranking(raw, position, velocity, selection_domain="matched_raw50")
    reachable = ranking(raw, position, velocity, selection_domain="reachable50")
    for key in ("adapted_xy", "adapted_velocity", "raw_dense_xy", "waypoint_indices_30Hz"):
        assert torch.equal(matched[key], reachable[key])
    for result, scored in ((matched, matched["raw_dense_xy"]), (reachable, reachable["adapted_xy"])):
        expected = scene_agent_collision_cost(scored[:, :, 3::3], torch.zeros(2, dtype=torch.long), .6)
        assert torch.equal(result["peer_cost"], expected)
        assert torch.equal(result["goal_cost"], scored[:, :, 150].norm(dim=-1))
    assert not torch.equal(matched["scene_sample_cost"], reachable["scene_sample_cost"])
    assert matched["selection_domain"] == "matched_raw50" and reachable["selection_domain"] == "reachable50"
