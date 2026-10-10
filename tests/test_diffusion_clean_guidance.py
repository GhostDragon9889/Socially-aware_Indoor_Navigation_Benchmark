"""Clean-output guidance must differentiate the denoiser, not just goal XY."""

from unittest.mock import patch

import pytest
import torch

from CrowdSim.humanoid.diffusion.paper import (
    PaperDiffusionConfig,
    PaperDiffusionPlanner,
    clean_waypoint_guidance_gradient,
    scene_agent_collision_cost,
    scene_geometry_collision_cost,
    signed_aabb_distance,
    trajectory_dynamics_cost,
)


def test_clean_guidance_uses_dense_denoiser_jacobian_without_parameter_grad():
    noisy = torch.zeros(2, 52, 5, dtype=torch.float64, requires_grad=True)
    parameter = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    parameter.grad = torch.tensor(7., dtype=torch.float64)  # Existing training gradient remains untouched.
    # Endpoint x depends on every earlier frame's velocity and yaw. A direct
    # gradient on output XY cannot produce those input-coordinate gradients.
    clean = noisy.clone()
    clean[..., 0] = parameter * noisy[..., 2].sum(dim=1, keepdim=True) + noisy[..., 4].sum(dim=1, keepdim=True)
    clean[..., 1] = 3 * noisy[..., 3].sum(dim=1, keepdim=True)
    targets = torch.tensor([[[3., 4.], [3., 4.]]], dtype=torch.float64)
    gradient = clean_waypoint_guidance_gradient(noisy, clean, targets, torch.tensor([49, 49]), num_samples=2)
    expected = torch.zeros_like(noisy)
    expected[..., 2] = -1.2
    expected[..., 3] = -2.4
    expected[..., 4] = -.6
    torch.testing.assert_close(gradient, expected, atol=1e-15, rtol=1e-15)
    assert not gradient.requires_grad and noisy.grad is None and parameter.grad == 7.


def test_clean_guidance_is_batch_independent_and_has_finite_exact_target_subgradient():
    def compute(batch):
        noisy = torch.ones(batch, 52, 5, requires_grad=True)
        clean = noisy * 2
        targets = torch.tensor([[[5., 6.], [2., 2.]]]).repeat(batch, 1, 1)
        return clean_waypoint_guidance_gradient(noisy, clean, targets, torch.tensor([49, 2]))

    single, batched = compute(1), compute(4)
    torch.testing.assert_close(single.expand_as(batched), batched)
    assert torch.isfinite(single).all() and single[:, 2].count_nonzero() == 0


def _short_planner():
    planner = PaperDiffusionPlanner(PaperDiffusionConfig(diffusion_steps=3))
    planner.trained_steps = 1  # CPU interface fixture; not a trained artifact.
    states = torch.tensor([[0., 0., .5, .1, .2]])
    history = states[:, None].repeat(1, 31, 1)
    raster = torch.zeros(1, 2, 224, 224)
    raster[:, 0] = 1
    return planner, dict(states=states, history=history, neighbors=None, raster=raster, waypoints=[[1., 1.]], seed=91)


@pytest.mark.parametrize("mode", ["trace_clean", "trace_clean_variance"])
def test_clean_plan_preserves_weights_rng_draws_and_skips_final_guidance(mode):
    planner, inputs = _short_planner()
    before = {name: value.clone() for name, value in planner.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    original = clean_waypoint_guidance_gradient
    with patch("CrowdSim.humanoid.diffusion.paper.clean_waypoint_guidance_gradient", wraps=original) as guidance:
        plan = planner.plan(**inputs, guidance_mode=mode, guidance_weight=.01)
    assert guidance.call_count == 2  # t=2,1 only, never t=0.
    assert plan.xy.shape == (1, 51, 2) and torch.isfinite(plan.sample_states).all()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in planner.model.parameters())
    assert all(torch.equal(before[name], value) for name, value in planner.model.state_dict().items())
    repeated = planner.plan(**inputs, guidance_mode=mode, guidance_weight=.01)
    assert torch.equal(plan.sample_states, repeated.sample_states)
    unguided = planner.plan(**inputs, guidance_weight=0)
    disabled = planner.plan(**inputs, guidance_mode=mode, guidance_weight=0)
    assert torch.equal(unguided.sample_states, disabled.sample_states)


def test_guidance_mode_defaults_to_legacy_and_rejects_typo():
    planner, inputs = _short_planner()
    default = planner.plan(**inputs)
    explicit = planner.plan(**inputs, guidance_mode="legacy_posterior")
    assert torch.equal(default.sample_states, explicit.sample_states)
    with pytest.raises(ValueError, match="guidance_mode"):
        planner.plan(**inputs, guidance_mode="clean_typo")


def test_optional_dynamo_capture_keeps_clean_input_gradient():
    planner, inputs = _short_planner()
    eager = planner.plan(**inputs, guidance_mode="trace_clean", guidance_weight=.1)
    planner.set_compiled_forward(True, backend="eager")
    captured = planner.plan(**inputs, guidance_mode="trace_clean", guidance_weight=.1)
    torch.testing.assert_close(captured.sample_states, eager.sample_states)
    assert all(parameter.grad is None for parameter in planner.model.parameters())


def test_map_feature_grid_rejects_nonfinite_coordinates_before_integer_indexing():
    planner, _ = _short_planner()
    encoded = {"map_grid": torch.zeros(1, 32, 7, 7), "conditioning": torch.zeros(1, 256)}
    with pytest.raises(RuntimeError, match="Nonfinite diffusion map coordinates"):
        planner.model(torch.full((1, 52, 5), torch.nan), torch.tensor([2]), encoded)
    with pytest.raises(RuntimeError, match="Nonfinite diffusion map coordinates"):
        planner.model(torch.full((1, 52, 5), torch.finfo(torch.float32).max), torch.tensor([2]), encoded)


def test_guidance_divergence_reports_the_step_without_returning_repaired_path():
    planner, inputs = _short_planner()
    with patch.object(planner.model, "forward", return_value=torch.full((1, 52, 5), torch.inf)):
        with pytest.raises(ValueError, match="clean prediction at step 2"):
            planner.plan(**inputs)


def test_joint_disk_guidance_uses_only_same_scene_future_predictions():
    noisy = torch.zeros(3, 52, 5, requires_grad=True)
    clean = noisy * 2
    current = torch.tensor([[0., 0., 0., 0., 0.], [.5, 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    gradient = clean_waypoint_guidance_gradient(
        noisy, clean, None, None, current=current, scene_ids=torch.tensor([0, 0, 1]),
        agent_collision_weight=1., agent_collision_radius=.3)
    torch.testing.assert_close(gradient[0, :, 0], torch.full((52,), 2. / .6))
    torch.testing.assert_close(gradient[1, :, 0], torch.full((52,), -2. / .6))
    assert gradient[2].count_nonzero() == 0
    assert gradient[..., 1:].count_nonzero() == 0
    cost = scene_agent_collision_cost(current[:, None, None, :2].expand(-1, 2, 52, -1), torch.tensor([0, 0, 1]))
    assert cost[0, 1, 0] == pytest.approx(52 * (1 - .5 / .6))
    assert cost[:, :, 0].tril().count_nonzero() == 0


def test_joint_sample_filter_does_not_mix_independently_best_colliding_samples():
    planner, _ = _short_planner()
    states = torch.tensor([[0., 0., 0., 0., 0.], [2., 0., 0., 0., 0.]])
    history = states[:, None].repeat(1, 31, 1)
    raster = torch.zeros(2, 2, 224, 224)
    raster[:, 0] = 1
    candidates = torch.zeros(4, 52, 5)
    # Both independently best sample0s occupy the same world point x=1.
    candidates[0, :, 0], candidates[2, :, 0] = .2, -.2

    def fixed_candidates(noisy, *_):
        return candidates + noisy * 0.  # Explicit gradient graph, no oracle.

    inputs = dict(states=states, history=history, neighbors=None, raster=raster,
                  waypoints=[[1., 0.], [1., 0.]], num_samples=2, seed=91, guidance_mode="trace_clean")
    with patch.object(planner.model, "forward", side_effect=fixed_candidates):
        independent = planner.plan(**inputs)
        joint = planner.plan(**inputs, scene_ids=[0, 0], agent_collision_weight=10.)
    assert independent.selected_sample.tolist() == [0, 0]
    assert joint.selected_sample.tolist() == [1, 1]
    torch.testing.assert_close(joint.xy[:, -1], states[:, :2])


def test_joint_guidance_requires_explicit_scene_identity_and_clean_mode():
    planner, inputs = _short_planner()
    with pytest.raises(ValueError, match="clean-output"):
        planner.plan(**inputs, agent_collision_weight=1.)
    with pytest.raises(ValueError, match="scene_ids"):
        planner.plan(**inputs, guidance_mode="trace_clean", agent_collision_weight=1.)
    with pytest.raises(ValueError, match="scene_ids"):
        planner.plan(**inputs, guidance_mode="trace_clean", agent_collision_weight=1., scene_ids=[.5])


def test_signed_box_distance_and_full_embedding_escape_gradient():
    points = torch.tensor([[0., 0.], [2., 0.], [2., 2.]], requires_grad=True)
    distance = signed_aabb_distance(points, torch.tensor([0., 0.]), torch.tensor([2., 2.]))
    torch.testing.assert_close(distance, torch.tensor([-1., 1., 2. ** .5]))
    loss = torch.relu(.5 - distance).sum()
    gradient, = torch.autograd.grad(loss, points)
    assert gradient[0].norm() > 0  # Explicit centre subgradient, not abs(0)'s zero.
    assert torch.all(gradient[0] < 0)
    assert gradient[1:].count_nonzero() == 0  # Free points unchanged.


def test_geometry_hinge_guides_both_full_obstacle_overlap_and_outside_domain():
    points = torch.tensor([[[[0., 0.]]], [[[5., 0.]]]], requires_grad=True)
    loss = scene_geometry_collision_cost(points, torch.tensor([-3., -3., 3., 3.]),
                                         torch.tensor([[0., 0.]]), torch.tensor([[2., 2.]]), .5)
    torch.testing.assert_close(loss[:, 0], torch.tensor([1.5, 2.5]))
    gradient, = torch.autograd.grad(loss.sum(), points)
    assert torch.all(gradient[0, 0, 0] < 0)  # Gradient descent escapes embedded centre.
    assert gradient[1, 0, 0, 0] > 0  # Gradient descent returns x=5 inside xmax=3.
    assert gradient[1, 0, 0, 1] == 0


def test_geometry_guidance_requires_actual_geometry_and_clean_mode():
    planner, inputs = _short_planner()
    with pytest.raises(ValueError, match="clean-output"):
        planner.plan(**inputs, scene_collision_weight=1.)
    with pytest.raises(ValueError, match="Explicit bounds"):
        planner.plan(**inputs, guidance_mode="trace_clean", scene_collision_weight=1.)
    geometry = dict(scene_bounds_xy=[-5., -5., 5., 5.], obstacle_centers_xy=[[0., 0.]], obstacle_sizes_xy=[[1., 1.]])
    plan = planner.plan(**inputs, guidance_mode="trace_clean", scene_collision_weight=.1, **geometry)
    assert torch.isfinite(plan.sample_states).all()
    assert all(parameter.grad is None for parameter in planner.model.parameters())


def test_dynamics_guidance_uses_consumed_xy_and_actual_initial_velocity():
    current = torch.tensor([[0., 0., 4., 0., 0.]])
    xy = torch.zeros(1, 1, 52, 2, requires_grad=True)
    moving = xy + torch.arange(1, 53)[None, None, :, None] * torch.tensor([.4, 0.])
    speed, acceleration = trajectory_dynamics_cost(moving, current)
    assert speed.item() == pytest.approx(52., rel=2e-5)  # Every segment is4m/s, limit3.
    assert acceleration.item() == pytest.approx(0., abs=1e-8)  # Initial physical velocity also4m/s.
    gradient, = torch.autograd.grad(speed.sum(), xy)
    assert gradient[0, 0, -1, 0] > 0  # Gradient descent reduces final position/speed.
    wrong_velocity = current.clone()
    wrong_velocity[:, 2] = 0.
    _, accelerated = trajectory_dynamics_cost(moving.detach(), wrong_velocity)
    assert accelerated.item() == pytest.approx((40. - 2.) ** 2, rel=1e-5)


def test_speed_guidance_requires_clean_mode_and_has_no_post_sample_clip():
    planner, inputs = _short_planner()
    with pytest.raises(ValueError, match="clean-output"):
        planner.plan(**inputs, trajectory_speed_weight=1.)
    with pytest.raises(ValueError, match="limits positive"):
        planner.plan(**inputs, guidance_mode="trace_clean", max_trajectory_speed_mps=0.)
    result = planner.plan(**inputs, guidance_mode="trace_clean", trajectory_speed_weight=.001)
    assert torch.isfinite(result.xy).all()
    assert torch.equal(result.xy[:, 0], inputs["states"][:, :2])
