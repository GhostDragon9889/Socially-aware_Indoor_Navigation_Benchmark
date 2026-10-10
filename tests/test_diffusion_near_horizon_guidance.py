"""Weighted clean guidance and filtering share one explicit objective."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

from CrowdSim.humanoid.diffusion.paper import (
    PaperDiffusionConfig, PaperDiffusionPlanner, clean_waypoint_guidance_gradient,
    reduce_waypoint_distances,
)
from CrowdSim.humanoid.waypoint_weighting import near_horizon_waypoint_weights


ROOT = Path(__file__).resolve().parents[1]
TIMES = [.5*index for index in range(1, 11)]


def before_module():
    name = 'CrowdSim.humanoid.diffusion.near_horizon_before'
    source = ROOT/'docs/style_improvement_20261002/near_horizon_waypoint_before/CrowdSim/humanoid/diffusion/paper.py'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(batch=1):
    states = torch.tensor([[0., 0., .2, 0., .1]]).repeat(batch, 1)
    raster = torch.zeros(batch, 2, 224, 224)
    raster[:, 0] = 1
    goals = torch.stack((torch.tensor(TIMES), torch.zeros(10)), dim=-1).repeat(batch, 1, 1)
    return dict(states=states, history=states[:, None].repeat(1, 31, 1), neighbors=None,
                raster=raster, waypoints=goals, waypoint_times=TIMES, seed=173)


@pytest.mark.parametrize('mode,coupled', [('legacy_posterior', False), ('trace_clean', False),
    ('trace_clean_variance', False), ('trace_clean', True), ('trace_clean_variance', True)])
def test_disabled_full_planner_is_bitwise_identical_to_before_source(mode, coupled):
    old = before_module()
    torch.manual_seed(11)
    previous = old.PaperDiffusionPlanner(old.PaperDiffusionConfig(diffusion_steps=3))
    current = PaperDiffusionPlanner(PaperDiffusionConfig(diffusion_steps=3))
    current.model.load_state_dict(previous.model.state_dict())
    previous.trained_steps = current.trained_steps = 1  # Untrained CPU interface fixtures, not training evidence.
    arguments = dict(**inputs(2 if coupled else 1), num_samples=2, guidance_mode=mode, guidance_weight=.01)
    if coupled:
        arguments.update(scene_ids=[0, 0], agent_collision_weight=.01, scene_collision_weight=.01,
            scene_bounds_xy=[-100., -100., 100., 100.], obstacle_centers_xy=[[40., 40.]], obstacle_sizes_xy=[[1., 1.]],
            trajectory_speed_weight=.0001, trajectory_accel_weight=.0001)
    baseline = previous.plan(**arguments)
    default = current.plan(**arguments)
    explicit_zero = current.plan(**arguments, near_horizon_mass=0.)
    for field in ('xy', 'yaw', 'vel', 'velocity', 'time', 'sample_states', 'selected_sample'):
        assert torch.equal(getattr(baseline, field), getattr(default, field)), field
        assert torch.equal(getattr(baseline, field), getattr(explicit_zero, field)), field


def test_disabled_gradient_keeps_before_mean_bitwise_and_repeated_indices():
    old = before_module()
    targets = torch.tensor([[[3., 4.], [3., 4.], [3., 4.]]], dtype=torch.float64)
    indices = torch.tensor([4, 4, 49])
    left = torch.zeros(1, 52, 5, dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    expected = old.clean_waypoint_guidance_gradient(left, left*2, targets, indices)
    actual = clean_waypoint_guidance_gradient(right, right*2, targets, indices)
    assert torch.equal(expected, actual)
    weighted = clean_waypoint_guidance_gradient(right, right*2, targets, indices,
        waypoint_weights=torch.tensor([.2, .3, .5], dtype=torch.float64))
    analytic = torch.zeros_like(weighted)
    analytic[:, [4, 49], 0] = -.6
    analytic[:, [4, 49], 1] = -.8
    torch.testing.assert_close(weighted, analytic, atol=1e-15, rtol=1e-15)


def test_weighted_gradient_is_batch_and_sample_invariant_preserving_existing_parameter_grad():
    weights = torch.tensor(near_horizon_waypoint_weights(TIMES, .25), dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    parameter.grad = torch.tensor(7., dtype=torch.float64)

    def compute(batch, samples):
        noisy = torch.ones(batch*samples, 52, 5, dtype=torch.float64, requires_grad=True)
        clean = parameter*noisy
        targets = torch.zeros(batch, 10, 2, dtype=torch.float64)
        gradient = clean_waypoint_guidance_gradient(noisy, clean, targets, torch.arange(4, 50, 5),
            num_samples=samples, waypoint_weights=weights)
        assert noisy.grad is None and parameter.grad == 7.
        return gradient

    single = compute(1, 1)
    assert torch.equal(compute(4, 1), single.repeat(4, 1, 1))
    assert torch.equal(compute(1, 3), single.repeat(3, 1, 1))


@pytest.mark.parametrize('weights', [torch.tensor([.2, .2]), torch.tensor([.2, float('nan'), .8]),
    torch.tensor([-.1, .5, .6]), torch.tensor([0., 0., 0.]), torch.tensor([1, 0, 0]),
    torch.tensor([[.2, .3, .5]]), [.2, .3, .5]])
def test_weighted_reduction_rejects_corrupt_or_unnormalized_inputs(weights):
    with pytest.raises(ValueError, match='Waypoint weights'):
        reduce_waypoint_distances(torch.ones(1, 2, 3), weights)


@pytest.mark.parametrize('joint', [False, True])
def test_actual_planner_filter_uses_the_same_weights_for_independent_and_scene_candidates(joint):
    planner = PaperDiffusionPlanner(PaperDiffusionConfig(diffusion_steps=3))
    planner.trained_steps = 1
    arguments = inputs(2)
    arguments['states'][:, 0] = torch.tensor([0., 10.])
    arguments['history'] = arguments['states'][:, None].repeat(1, 31, 1)
    arguments['waypoints'] = arguments['states'][:, None, :2].repeat(1, 10, 1)
    candidates = torch.zeros(4, 52, 5)
    # Uniform goal cost: sample0 .2, sample1 .34. Near-weighted: .4 vs .28.
    candidates[0::2, [4, 9], 0] = 1.
    candidates[1::2, :, 0] = .4
    candidates[1::2, [4, 9], 0] = .1
    normalized = planner.model.normalize(candidates)

    def fixed_candidates(noisy, *_):
        return normalized+noisy*0.

    flags = {'scene_ids': [0, 0], 'agent_collision_weight': 1.} if joint else {}
    with patch.object(planner.model, 'forward', side_effect=fixed_candidates):
        baseline = planner.plan(**arguments, num_samples=2, guidance_mode='trace_clean', guidance_weight=20., **flags)
        weighted = planner.plan(**arguments, num_samples=2, guidance_mode='trace_clean', guidance_weight=20.,
                                near_horizon_mass=.25, **flags)
    assert baseline.selected_sample.tolist() == [0, 0]
    assert weighted.selected_sample.tolist() == [1, 1]
    assert all(parameter.grad is None for parameter in planner.model.parameters())


@pytest.mark.parametrize('mode', ['trace_clean', 'trace_clean_variance'])
def test_weighted_plan_preserves_model_and_rng_and_does_not_guide_t0(mode):
    planner = PaperDiffusionPlanner(PaperDiffusionConfig(diffusion_steps=3))
    planner.trained_steps = 1
    before = {name: value.clone() for name, value in planner.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with patch('CrowdSim.humanoid.diffusion.paper.clean_waypoint_guidance_gradient',
               wraps=clean_waypoint_guidance_gradient) as guided:
        result = planner.plan(**inputs(), guidance_mode=mode, guidance_weight=.01, near_horizon_mass=.25)
    assert guided.call_count == 2
    for call in guided.call_args_list:
        torch.testing.assert_close(call.kwargs['waypoint_weights'], torch.tensor([.2, .2]+[.075]*8))
    assert torch.isfinite(result.sample_states).all() and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[name], value) for name, value in planner.model.state_dict().items())


def test_enabled_plan_requires_clean_mode_explicit_near_goals_and_valid_mass():
    planner = PaperDiffusionPlanner(PaperDiffusionConfig(diffusion_steps=3))
    planner.trained_steps = 1
    with pytest.raises(ValueError, match='clean guidance'):
        planner.plan(**inputs(), near_horizon_mass=.25)
    arguments = inputs()
    arguments.pop('waypoints')
    with pytest.raises(ValueError, match='explicit waypoints'):
        planner.plan(**arguments, near_horizon_mass=.25, guidance_mode='trace_clean')
    arguments = inputs()
    arguments['waypoint_times'] = [2.]*10
    with pytest.raises(ValueError, match='at or before1s'):
        planner.plan(**arguments, near_horizon_mass=.25, guidance_mode='trace_clean')
    with pytest.raises(ValueError, match='mass'):
        planner.plan(**inputs(), near_horizon_mass=True, guidance_mode='trace_clean')
