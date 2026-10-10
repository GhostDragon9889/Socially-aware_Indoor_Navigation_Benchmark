"""World-grid callback contracts; fixtures do not prove native physical quality."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

import CrowdSim.humanoid.diffusion.paper as paper


ROOT = Path(__file__).resolve().parents[1]


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def planner_inputs(batch=1):
    planner = paper.PaperDiffusionPlanner(paper.PaperDiffusionConfig(diffusion_steps=3))
    planner.trained_steps = 1  # CPU contract fixture, not a trained weight.
    states = torch.tensor([[0., 0., .2, 0., .1]]).repeat(batch, 1)
    raster = torch.zeros(batch, 2, 224, 224)
    raster[:, 0] = 1
    return planner, {"states": states, "history": states[:, None].repeat(1, 31, 1),
        "neighbors": None, "raster": raster, "waypoints": states[:, :2] + 1., "seed": 173}


@pytest.fixture
def native_grid(tmp_path):
    # Reuse the persisted raw/CSR/clock admission fixture, not fabricated live
    # native output or a fake AABB substitute. It is explicitly unit evidence.
    fixture = module_at(ROOT / "tests/test_native_grid_scene.py", "native_grid_callback_fixture")
    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 205
    return fixture.scene(fixture.fixture_provider(tmp_path, labels))


@pytest.mark.parametrize("mode,coupled", [("legacy_posterior", False), ("trace_clean", False),
    ("trace_clean_variance", False), ("trace_clean", True), ("trace_clean_variance", True)])
def test_default_none_preserves_pre_callback_seeded_complete_output_bytewise(mode, coupled):
    old = module_at(ROOT / "docs/style_improvement_20261002/native_grid_diffusion_guidance_v1/before/paper.py",
        "CrowdSim.humanoid.diffusion.pre_scene_callback")
    planner, arguments = planner_inputs(2 if coupled else 1)
    before = old.PaperDiffusionPlanner(old.PaperDiffusionConfig(diffusion_steps=3))
    before.model.load_state_dict(planner.model.state_dict())
    before.trained_steps = 1
    arguments.update(num_samples=2, guidance_mode=mode, guidance_weight=.01)
    if coupled:
        arguments.update(scene_ids=[0, 0], agent_collision_weight=.01, scene_collision_weight=.01,
            scene_bounds_xy=[-100., -100., 100., 100.], obstacle_centers_xy=[[40., 40.]], obstacle_sizes_xy=[[1., 1.]],
            trajectory_speed_weight=.0001, trajectory_accel_weight=.0001)
    rng = torch.get_rng_state().clone()
    expected, implicit, explicit = before.plan(**arguments), planner.plan(**arguments), planner.plan(**arguments, scene_collision_cost_fn=None)
    assert torch.equal(rng, torch.get_rng_state())
    for field in ("xy", "yaw", "vel", "velocity", "time", "sample_states", "selected_sample"):
        a, b, c = (getattr(value, field) for value in (expected, implicit, explicit))
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), field
        assert torch.equal(a.contiguous().view(torch.uint8), c.contiguous().view(torch.uint8)), field


def test_callback_receives_once_transformed_world_positions_and_full_input_jacobian_without_parameter_grad():
    noisy = torch.ones(1, 52, 5, dtype=torch.float64, requires_grad=True)
    parameter = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    parameter.grad = torch.tensor(7., dtype=torch.float64)
    clean = noisy * 2
    current = torch.tensor([[-9., 4., 0., 0., np.pi/2]], dtype=torch.float64)
    received = []

    def callback(world_xy, radius):
        received.append(world_xy.detach().clone())
        assert radius == .35
        return parameter * world_xy.square().sum(-1).mean(-1)

    gradient = paper.clean_waypoint_guidance_gradient(noisy, clean, None, None, current=current,
        scene_collision_weight=1., scene_collision_radius=.35, scene_collision_cost_fn=callback)
    torch.testing.assert_close(received[0], torch.tensor([-11., 6.], dtype=torch.float64).expand(1, 1, 52, 2), atol=1e-14, rtol=0)
    expected = torch.zeros_like(noisy)
    expected[..., 0], expected[..., 1] = 48/52, 88/52
    torch.testing.assert_close(gradient, expected, atol=1e-14, rtol=0)
    assert noisy.grad is None and parameter.grad.item() == 7.


def test_real_grid_function_unknown_and_exterior_penalties_have_finite_causal_guidance(native_grid):
    current = torch.tensor([[-.99, -.05, 0., 0., 0.], [4.2, 0., 0., 0., 0.]], dtype=torch.float64)
    noisy = torch.zeros(2, 52, 5, dtype=torch.float64, requires_grad=True)
    clean = noisy * 2
    before = current.clone()
    gradient = paper.clean_waypoint_guidance_gradient(noisy, clean, None, None, current=current,
        scene_collision_weight=1., scene_collision_radius=.35, scene_collision_cost_fn=native_grid.collision_cost)
    world = current[:, None, None, :2].expand(-1, 1, 52, -1)
    cost = native_grid.collision_cost(world, .35)
    assert (cost > 0).all() and (gradient[..., 0] > 0).all()
    improved_world = world - .01 * gradient[:, None, :, :2] * 2
    assert (native_grid.collision_cost(improved_world, .35) < cost).all()
    assert torch.equal(before, current)  # Exterior coordinates are not clipped.


def test_connected_zero_free_cost_is_valid_and_grid_batch_sample_gradients_are_invariant(native_grid):
    def compute(batch, samples, current_xy):
        noisy = torch.zeros(batch*samples, 52, 5, dtype=torch.float64, requires_grad=True)
        current = torch.tensor([[*current_xy, 0., 0., 0.]], dtype=torch.float64).repeat(batch, 1)
        return paper.clean_waypoint_guidance_gradient(noisy, noisy*2, None, None, current=current,
            num_samples=samples, scene_collision_weight=1., scene_collision_radius=.35,
            scene_collision_cost_fn=native_grid.collision_cost)

    assert compute(1, 1, [-3.45, -.05]).count_nonzero() == 0
    single, batched = compute(1, 1, [-.99, -.05]), compute(3, 2, [-.99, -.05])
    assert torch.equal(single.repeat(6, 1, 1), batched)


@pytest.mark.parametrize("change", ["list", "integer", "dtype", "device", "shape", "nan", "negative", "detached", "disconnected", "nan_gradient"])
def test_callback_corrupt_results_rejected_before_returning_guidance(change):
    noisy = torch.zeros(1, 52, 5, requires_grad=True)
    current = torch.zeros(1, 5)
    parameter = torch.nn.Parameter(torch.ones(1, 1))
    parameter.grad = torch.full((1, 1), 11.)

    def callback(world_xy, radius):
        cost = world_xy.square().sum(-1).mean(-1)
        if change == "list":
            return [[0.]]
        if change == "integer":
            return cost.long()
        if change == "dtype":
            return cost.double()
        if change == "device":
            return cost.to("meta")
        if change == "shape":
            return cost[..., None]
        if change == "nan":
            return cost + float("nan")
        if change == "negative":
            return cost - 1
        if change == "detached":
            return cost.detach()
        if change == "disconnected":
            return parameter
        return torch.sqrt(cost)  # Finite0 output but nonfinite backward at0.

    with pytest.raises(ValueError, match="callback"):
        paper.clean_waypoint_guidance_gradient(noisy, noisy*2, None, None, current=current,
            scene_collision_weight=1., scene_collision_cost_fn=callback)
    assert noisy.grad is None and parameter.grad.item() == 11.


@pytest.mark.parametrize("flags", [
    {"guidance_mode": "legacy_posterior", "scene_collision_weight": 1.},
    {"guidance_mode": "trace_clean", "scene_collision_weight": 0.},
    {"guidance_mode": "trace_clean", "scene_collision_weight": 1., "scene_bounds_xy": [-5., -5., 5., 5.]},
    {"guidance_mode": "trace_clean", "scene_collision_weight": 1., "obstacle_centers_xy": []},
    {"guidance_mode": "trace_clean", "scene_collision_weight": 1., "obstacle_sizes_xy": []},
])
def test_legacy_unused_or_ambiguous_geometry_callback_rejected_before_model_or_sampling(flags):
    planner, inputs = planner_inputs()
    rng = torch.get_rng_state().clone()
    with patch.object(planner.model, "encode_context", side_effect=AssertionError("Premature model forward")):
        with pytest.raises(ValueError):
            planner.plan(**inputs, scene_collision_cost_fn=lambda world_xy, radius: world_xy.sum(-1).mean(-1), **flags)
    assert torch.equal(rng, torch.get_rng_state())


def test_noncallable_or_bad_current_shape_dtype_device_rejected():
    planner, inputs = planner_inputs()
    with pytest.raises(ValueError, match="callable"):
        planner.plan(**inputs, guidance_mode="trace_clean", scene_collision_weight=1., scene_collision_cost_fn=1)
    noisy = torch.zeros(1, 52, 5, requires_grad=True)
    for current in (None, torch.zeros(2, 5), torch.zeros(1, 5, dtype=torch.float64), torch.zeros(1, 5, device="meta")):
        with pytest.raises(ValueError, match="current"):
            paper.clean_waypoint_guidance_gradient(noisy, noisy*2, None, None, current=current,
                scene_collision_weight=1., scene_collision_cost_fn=lambda xy, radius: xy.square().mean((-1, -2)))


def test_callback_scores_candidates_and_joint_scene_choice_never_mixes_actor_samples():
    planner, inputs = planner_inputs(2)
    inputs["states"] = torch.tensor([[0., 0., 0., 0., 0.], [10., 0., 0., 0., 0.]])
    inputs["history"] = inputs["states"][:, None].repeat(1, 31, 1)
    inputs.pop("waypoints")
    candidates = torch.zeros(4, 52, 5)
    candidates[:, :, 0] = torch.tensor([1., 2., -2., -1.])[:, None]
    normalized = planner.model.normalize(candidates)
    calls = []

    def cost(world_xy, radius):
        calls.append((world_xy.requires_grad, world_xy.detach().clone()))
        desired = world_xy.new_tensor([1., 9.])[:, None, None]
        return (world_xy[..., 0] - desired).square().mean(-1) * world_xy.new_tensor([1., 3.])[:, None]

    def forward(noisy, *_):
        return normalized + noisy*0.

    first = next(planner.model.parameters())
    first.grad = torch.ones_like(first)
    existing = first.grad.clone()
    model = {name: value.clone() for name, value in planner.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    options = dict(num_samples=2, guidance_mode="trace_clean", scene_collision_weight=1., scene_collision_cost_fn=cost)
    with patch.object(planner.model, "forward", side_effect=forward):
        independent = planner.plan(**inputs, **options)
        assert independent.selected_sample.tolist() == [0, 1]
        assert [grad for grad, _ in calls] == [True, True, False]
        assert all(value.shape == (2, 2, 52, 2) for _, value in calls)
        calls.clear()
        joint = planner.plan(**inputs, scene_ids=[0, 0], agent_collision_weight=.01, **options)
    assert joint.selected_sample.tolist() == [1, 1]
    assert [grad for grad, _ in calls] == [True, True, False]
    assert torch.equal(first.grad, existing) and all(parameter.grad is None for parameter in list(planner.model.parameters())[1:])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(model[name], value) for name, value in planner.model.state_dict().items())


def test_candidate_filter_validates_cost_even_when_guidance_cost_is_valid():
    planner, inputs = planner_inputs()

    def cost(xy, radius):
        value = xy.square().sum(-1).mean(-1)
        return value if xy.requires_grad else torch.full_like(value, float("nan"))

    with pytest.raises(ValueError, match="callback collision cost"):
        planner.plan(**inputs, num_samples=2, guidance_mode="trace_clean", scene_collision_weight=.01,
            scene_collision_cost_fn=cost)
