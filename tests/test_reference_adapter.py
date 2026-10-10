"""Reference bounds/continuity; no physics, stability or collision-free claims."""
from __future__ import annotations

import json

import pytest
import torch

from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig


DT = 1 / 30


def raw_straight(start, velocity):
    times = torch.arange(51, dtype=start.dtype) * .1
    return start[:, None] + times[None, :, None] * velocity[:, None]


def assert_reference_bounds(plan, max_speed=3., max_acceleration=2., *, allow_initial_overspeed=False):
    speed = torch.linalg.vector_norm(plan.velocity, dim=-1)
    acceleration = torch.linalg.vector_norm(torch.diff(plan.velocity, dim=1) / DT, dim=-1)
    assert acceleration.max() <= max_acceleration + 1e-10
    if not allow_initial_overspeed:
        assert speed.max() <= max_speed + 1e-10
        assert not plan.speed_violation.any()
    # Independently infer interval velocity from exported positions.
    torch.testing.assert_close(torch.diff(plan.xy, dim=1) / DT, plan.velocity[:, 1:], rtol=1e-10, atol=1e-10)


def test_straight_reference_with_matching_initial_velocity_has_no_phase_lead_or_lag():
    xy = torch.tensor([[18., -5.], [-2., 3.]], dtype=torch.float64)
    velocity = torch.tensor([[1., .4], [-.5, .2]], dtype=torch.float64)
    raw = raw_straight(xy, velocity)
    plan = CausalReferenceAdapter().build(raw, xy, velocity)
    assert plan.xy.shape == plan.velocity.shape == (2, 151, 2)
    expected = xy[:, None] + plan.times_s[None, :, None] * velocity[:, None]
    torch.testing.assert_close(plan.xy, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(plan.raw_xy, raw, rtol=0, atol=0)
    torch.testing.assert_close(plan.velocity[:, 0], velocity, rtol=0, atol=0)
    assert_reference_bounds(plan)


def test_acceleration_clips_vector_norm_during_large_turn_not_individual_axes():
    xy = torch.zeros(1, 2, dtype=torch.float64)
    velocity = torch.tensor([[2.8, 0.]], dtype=torch.float64)
    desired = torch.tensor([[0., 100.]], dtype=torch.float64)
    state = CausalReferenceAdapter().step(xy, velocity, xy, desired)
    delta = state.velocity - velocity
    assert torch.linalg.vector_norm(delta) == pytest.approx(2 * DT)
    assert (delta.abs() < 2 * DT).all(), "Diagonal clipping must preserve the vector bound"
    assert torch.linalg.vector_norm(state.velocity) <= 3
    torch.testing.assert_close(state.xy, state.velocity * DT)


def test_violent_learned_positions_are_preserved_while_dense_execution_respects_bounds():
    times = torch.arange(51, dtype=torch.float64) * .1
    raw = torch.stack((20 * times, 15 * torch.sin(8 * times)), dim=-1)[None]
    xy = torch.zeros(1, 2, dtype=torch.float64)
    velocity = torch.zeros_like(xy)
    original = [value.clone() for value in (raw, xy, velocity)]
    plan = CausalReferenceAdapter().build(raw, xy, velocity)
    assert_reference_bounds(plan)
    assert plan.deviation_m.max() > 10
    assert plan.raw_xy.shape == (1, 51, 2) and plan.raw_sampled_xy.shape == (1, 151, 2)
    for actual, untouched in zip((raw, xy, velocity), original):
        torch.testing.assert_close(actual, untouched, rtol=0, atol=0)
    torch.testing.assert_close(plan.raw_xy, raw, rtol=0, atol=0)


def test_ordinary_replan_keeps_executed_reference_state_instead_of_snapping_to_new_raw_origin():
    xy = torch.tensor([[0., 0.]], dtype=torch.float64)
    velocity = torch.tensor([[1., 0.]], dtype=torch.float64)
    adapter = CausalReferenceAdapter()
    old = adapter.build(raw_straight(xy, velocity), xy, velocity)
    executed = old.state_at_step(15)
    # A learned replan can have an arbitrarily different desired origin/heading.
    changed_raw = raw_straight(torch.tensor([[8., -4.]], dtype=torch.float64), -velocity)
    new = adapter.build(changed_raw, executed.xy, executed.velocity)
    torch.testing.assert_close(new.xy[:, 0], old.xy[:, 15], rtol=0, atol=0)
    torch.testing.assert_close(new.velocity[:, 0], old.velocity[:, 15], rtol=0, atol=0)
    assert torch.linalg.vector_norm(new.velocity[:, 1] - old.velocity[:, 15]) <= 2 * DT + 1e-12
    assert_reference_bounds(new)
    assert not torch.equal(new.xy[:, 0], changed_raw[:, 0])


def test_real_episode_reset_is_explicit_new_measured_reference_initialization():
    xy = torch.tensor([[-10., 5.]], dtype=torch.float64)
    measured_velocity = torch.tensor([[.5, -.25]], dtype=torch.float64)
    raw = raw_straight(xy, measured_velocity)
    plan = CausalReferenceAdapter().build(raw, xy, measured_velocity)
    torch.testing.assert_close(plan.xy[:, 0], xy, rtol=0, atol=0)
    torch.testing.assert_close(plan.velocity[:, 0], measured_velocity, rtol=0, atol=0)


def test_initial_overspeed_rejects_with_actor_feedback_without_clipping_physical_inputs():
    xy = torch.zeros(2, 2, dtype=torch.float64)
    velocity = torch.tensor([[1., 0.], [4., 0.]], dtype=torch.float64)
    untouched = velocity.clone()
    with pytest.raises(ValueError, match=r"actors \[1\].*brake"):
        CausalReferenceAdapter().build(raw_straight(xy, velocity), xy, velocity)
    torch.testing.assert_close(velocity, untouched, rtol=0, atol=0)


def test_explicit_brake_preserves_actual_v0_reports_unavoidable_overspeed_and_acceleration():
    xy = torch.zeros(2, 2, dtype=torch.float64)
    velocity = torch.tensor([[4., 0.], [1., 0.]], dtype=torch.float64)
    raw = torch.zeros(2, 51, 2, dtype=torch.float64)
    config = ReferenceAdapterConfig(initial_speed_policy="brake")
    plan = CausalReferenceAdapter(config).build(raw, xy, velocity)
    torch.testing.assert_close(plan.velocity[:, 0], velocity, rtol=0, atol=0)
    assert plan.initial_overspeed.tolist() == [True, False]
    assert plan.speed_violation[0, 0] and plan.speed_violation[0, 1]
    assert plan.velocity[0, 1, 0] == pytest.approx(4 - 2 * DT)
    assert not plan.speed_violation[:, -1].any()
    assert_reference_bounds(plan, allow_initial_overspeed=True)
    torch.testing.assert_close(velocity, torch.tensor([[4., 0.], [1., 0.]], dtype=torch.float64))


def test_explicit_15_mps_ablation_changes_only_reference_bound_and_leaves_raw_plan_available():
    xy = torch.zeros(1, 2, dtype=torch.float64)
    velocity = torch.zeros_like(xy)
    raw = raw_straight(xy, torch.tensor([[10., 0.]], dtype=torch.float64))
    slow = CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5)).build(raw, xy, velocity)
    regular = CausalReferenceAdapter().build(raw, xy, velocity)
    assert_reference_bounds(slow, max_speed=1.5)
    assert slow.xy[0, -1, 0] < regular.xy[0, -1, 0]
    torch.testing.assert_close(slow.raw_xy, regular.raw_xy, rtol=0, atol=0)


def test_per_actor_saved_control_step_state_is_a_copy_and_has_no_hidden_execution_state():
    xy = torch.zeros(2, 2, dtype=torch.float64)
    velocity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    adapter = CausalReferenceAdapter()
    raw = raw_straight(xy, velocity)
    first = adapter.build(raw, xy, velocity)
    state = first.state_at_step(torch.tensor([15, 30]))
    torch.testing.assert_close(state.xy, torch.stack((first.xy[0, 15], first.xy[1, 30])))
    state.xy.add_(100)
    again = adapter.build(raw, xy, velocity)
    torch.testing.assert_close(again.xy, first.xy, rtol=0, atol=0)
    assert torch.linalg.vector_norm(first.xy[0, 15]) < 1


@pytest.mark.parametrize("steps,exception", [
    (-1, ValueError), (151, ValueError), (True, TypeError),
    (torch.tensor([0.5]), TypeError), (torch.tensor([151]), ValueError),
])
def test_state_lookup_cannot_guess_missing_or_outside_reference_steps(steps, exception):
    zero = torch.zeros(1, 2, dtype=torch.float64)
    plan = CausalReferenceAdapter().build(raw_straight(zero, zero), zero, zero)
    with pytest.raises(exception):
        plan.state_at_step(steps)


def test_sparse_grid_is_rejected_instead_of_claiming_30hz_acceleration_after_sparse_interpolation():
    zero = torch.zeros(1, 2, dtype=torch.float64)
    raw = raw_straight(zero, zero)
    with pytest.raises(ValueError, match="151"):
        CausalReferenceAdapter().build(raw, zero, zero, times_s=torch.arange(51, dtype=torch.float64) * .1)
    grid = torch.arange(151, dtype=torch.float64) * DT
    grid[50] += .01
    with pytest.raises(ValueError, match="uniform"):
        CausalReferenceAdapter().build(raw, zero, zero, times_s=grid)


@pytest.mark.parametrize("key,value", [
    ("max_speed_mps", True), ("max_speed_mps", 0.),
    ("max_acceleration_mps2", -1.), ("tracking_time_s", float("nan")),
    ("initial_speed_policy", "clip"), ("control_dt_s", .07),
])
def test_invalid_or_instantaneous_clip_configuration_fails_explicitly(key, value):
    with pytest.raises(ValueError):
        CausalReferenceAdapter(ReferenceAdapterConfig(**{key: value}))


@pytest.mark.parametrize("invalid,exception", [
    ("nan_raw", FloatingPointError), ("nan_velocity", FloatingPointError),
    ("wrong_raw_length", ValueError), ("mixed_dtype", ValueError),
])
def test_invalid_model_or_reference_tensors_are_not_repaired_silently(invalid, exception):
    xy = torch.zeros(1, 2, dtype=torch.float64)
    velocity = torch.zeros_like(xy)
    raw = raw_straight(xy, velocity)
    if invalid == "nan_raw":
        raw[0, 15, 0] = float("nan")
    elif invalid == "nan_velocity":
        velocity[0, 1] = float("nan")
    elif invalid == "wrong_raw_length":
        raw = raw[:, :-1]
    elif invalid == "mixed_dtype":
        velocity = velocity.float()
    with pytest.raises(exception):
        CausalReferenceAdapter().build(raw, xy, velocity)


def test_unsaturated_adaptation_keeps_a_finite_gradient_to_real_model_coordinates():
    xy = torch.zeros(1, 2, dtype=torch.float64)
    velocity = torch.tensor([[.1, .02]], dtype=torch.float64)
    raw = raw_straight(xy, velocity).requires_grad_()
    plan = CausalReferenceAdapter().build(raw, xy, velocity)
    loss = plan.xy[:, 75:, 0].mean() + plan.xy[:, 75:, 1].square().mean()
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_float32_world_coordinates_keep_bounds_with_declared_floating_point_precision():
    xy = torch.tensor([[18., 18.]])
    velocity = torch.tensor([[.5, -.25]])
    raw = raw_straight(xy, torch.tensor([[8., -8.]]))
    plan = CausalReferenceAdapter().build(raw, xy, velocity)
    assert torch.linalg.vector_norm(plan.velocity, dim=-1).max() <= 3 + 1e-6
    assert torch.linalg.vector_norm(torch.diff(plan.velocity, dim=1) / DT, dim=-1).max() <= 2 + 1e-4
    assert not plan.initial_overspeed.any()


def test_runner_description_is_json_safe_and_discloses_reference_only_deployment_limits():
    metadata = json.loads(json.dumps(CausalReferenceAdapter().describe()))
    assert metadata["deployment_adaptation"] is True and metadata["paper_definition"] is False
    assert metadata["hidden_state"] is False
    assert metadata["configuration"]["initial_speed_policy"] == "reject"
    assert "physical speed" in metadata["limitations"][0]
    assert any("no boundary" in limit and "stability guarantee" in limit for limit in metadata["limitations"])
