"""CPU candidate contracts, including real analytic counterexamples."""
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from prototype import (WeightingSpec, classification_loss, command_context,
                       expert_context, expert_weights, renormalize_bounded)


def window(velocity=(1., 0.), batch=2):
    t = torch.arange(11, dtype=torch.float64)/30
    xy = t[None, :, None]*torch.tensor(velocity, dtype=torch.float64)[None, None]
    xy = xy.expand(batch, -1, -1).clone()
    q = torch.zeros(batch, 11, 4, dtype=torch.float64)
    q[..., 3] = 1
    ids = torch.arange(batch)[:, None].expand(-1, 11).clone()
    frames = torch.arange(10, 21)[None].expand(batch, -1).clone()
    counts = torch.full((batch,), 30, dtype=torch.int64)
    return xy, q, ids, frames, counts


def task(velocity=(2., 0.)):
    values = torch.zeros(1, 1044, dtype=torch.float64)
    values[:, :20] = (torch.arange(.5, 5.01, .5)[:, None]*torch.tensor(velocity)).flatten()
    return values


def command(values):
    return command_context(values, capture_phase='pre_action', observation_space='raw_task_1044')


def test_intent_from_future_is_nonzero_without_actual_speed_input():
    values = task((2., -.5))
    result = command(values)
    assert torch.equal(result, torch.tensor([[2., -.5, 0.]], dtype=torch.float64))
    values[:, 20:] = 100
    assert torch.equal(command(values), result)  # Terrain affects no descriptor.


def test_stopped_and_path_turn_semantics_are_exact():
    assert torch.equal(command(task((0., 0.))), torch.zeros(1, 3, dtype=torch.float64))
    values = task()
    values[0, :4] = torch.tensor([.5, 0., .5, .5])
    assert torch.allclose(command(values), torch.tensor([[1., 0., torch.pi]], dtype=torch.float64))


@pytest.mark.parametrize('phase,space', [('post_step','raw_task_1044'), ('pre_action','task_z_clamped')])
def test_normalized_or_post_step_tag_is_rejected(phase, space):
    with pytest.raises(ValueError, match='PRE-ACTION'):
        command_context(task(), capture_phase=phase, observation_space=space)


@pytest.mark.parametrize('bad', [torch.zeros(2, 20), torch.zeros(2, 357), torch.zeros(0, 1044), torch.zeros(2, 1044, dtype=torch.int64), torch.full((1, 1044), float('nan'))])
def test_real_task_shape_dtype_and_finite_gate(bad):
    with pytest.raises(ValueError):
        command(bad)


def test_expert_velocity_is_actual_geometry_in_current_body_heading():
    args = list(window((0., 2.)))
    args[1][..., 2] = np.sin(np.pi/4)
    args[1][..., 3] = np.cos(np.pi/4)
    c, yaw = expert_context(*args)
    assert torch.allclose(c, torch.tensor([[2., 0., 0.], [2., 0., 0.]], dtype=torch.float64), atol=1e-14)
    assert torch.equal(yaw, torch.zeros(2, dtype=torch.float64))


def test_world_translation_and_rotation_leave_expert_context_unchanged():
    args = list(window((1., .3)))
    expected, _ = expert_context(*args)
    theta = .87
    rotation = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    args[0] = args[0]@rotation.T + torch.tensor([100., -60.])
    args[1][..., 2] = np.sin(theta/2)
    args[1][..., 3] = np.cos(theta/2)
    actual, _ = expert_context(*args)
    assert torch.allclose(actual, expected, atol=1e-12)


def test_body_yaw_is_distinct_from_path_turn():
    args = list(window((1., 0.)))
    angle = torch.arange(11, dtype=torch.float64)*.01
    args[1][..., 2] = (angle/2).sin()
    args[1][..., 3] = (angle/2).cos()
    c, body_yaw = expert_context(*args)
    assert torch.equal(c[:, 2], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(body_yaw, torch.full((2,), .3, dtype=torch.float64), atol=1e-14)


@pytest.mark.parametrize('bad_field', ['motion', 'gap', 'count', 'float_frame', 'reset', 'quaternion'])
def test_history_next_boundaries_and_reset_rejected(bad_field):
    args = list(window())
    kwargs = {}
    if bad_field == 'motion':
        args[2][0, -1] += 1
    elif bad_field == 'gap':
        args[3][0, -1] += 1
    elif bad_field == 'count':
        args[4][0] = 20
    elif bad_field == 'float_frame':
        args[3] = args[3].double()
    elif bad_field == 'reset':
        episode = torch.zeros(2, 11, dtype=torch.int64)
        episode[0, -1] = 1
        kwargs['history_episode_ids'] = episode
    else:
        args[1][0, 0, 3] = 2
    with pytest.raises(ValueError):
        expert_context(*args, **kwargs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bounds_and_mean_hold_jointly_in_divide_mean_counterexample(dtype):
    raw = torch.tensor([.5]*19 + [2.], dtype=dtype)
    naive = raw/raw.mean()
    assert naive.max() > 2
    actual = renormalize_bounded(raw)
    assert actual.min() >= .5 and actual.max() <= 2
    assert abs(actual.double().mean().item()-1) < 1e-7


def test_command_relevance_favors_requested_forward_motion_without_sampling():
    c = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
    e = torch.tensor([[0., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    weights, report = expert_weights(c, e)
    assert weights[1] > weights[0]
    assert torch.allclose(weights, torch.tensor([.5, 1.5], dtype=torch.float64), atol=1e-12)
    assert report['sampling_changed'] is False and report['context_conditioned_D'] is False
    assert sum(report['batch_relative_probability']) == pytest.approx(1)


def test_unconditional_global_batch_limitation_is_visible():
    commands = torch.tensor([[0., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    weights, _ = expert_weights(commands, commands)
    assert torch.allclose(weights, torch.ones(2, dtype=torch.float64), atol=1e-14)
    # No per-context weight matrix/D input exists: opposite requests cancel
    # in the marginal while true conditional discrimination could distinguish.


def test_no_supported_batch_does_not_fabricate_matching_data():
    c = torch.tensor([[100., 100., 100.]], dtype=torch.float64)
    e = torch.zeros(4, 3, dtype=torch.float64)
    weights, report = expert_weights(c, e)
    assert torch.equal(weights, torch.ones(4, dtype=torch.float64))
    assert report['no_support_in_sampled_batch'] is True


def test_chunking_permutation_and_input_storage_are_stable():
    c = torch.tensor([[0., 0., 0.], [.5, 0., .4], [1., .2, -.5]], dtype=torch.float64)
    e = torch.tensor([[.2, 0., 0.], [1., .2, 0.], [-.4, 0., 1.]], dtype=torch.float64)
    before_c, before_e = c.clone(), e.clone()
    left, _ = expert_weights(c, e, chunk_size=1)
    right, _ = expert_weights(c.flip(0), e.flip(0), chunk_size=3)
    assert torch.allclose(left, right.flip(0), atol=1e-14)
    assert torch.equal(c, before_c) and torch.equal(e, before_e)


def test_legacy_loss_and_gradients_are_bitwise_exact():
    pos = torch.tensor([-.7, .2, 1.1], dtype=torch.float32, requires_grad=True)
    neg = torch.tensor([-.2, 2.], dtype=torch.float32, requires_grad=True)
    actual = classification_loss(pos, neg)
    legacy = .5*(F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                  + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
    assert torch.equal(actual, legacy)
    assert all(torch.equal(a, b) for a, b in zip(torch.autograd.grad(actual, (pos, neg)), torch.autograd.grad(legacy, (pos, neg))))


def test_positive_weights_change_only_positive_gradient_as_analytic_BCE():
    pos = torch.tensor([-.7, .2], dtype=torch.float64, requires_grad=True)
    neg = torch.tensor([-.2, 2.], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([.5, 1.5], dtype=torch.float64)
    loss = classification_loss(pos, neg, weights=weights)
    gp, gn = torch.autograd.grad(loss, (pos, neg))
    assert torch.allclose(gp, .5*weights*(pos.sigmoid()-1)/2, atol=1e-15)
    assert torch.allclose(gn, .5*neg.sigmoid()/2, atol=1e-15)


@pytest.mark.parametrize('weights', [torch.tensor([.5, 2.]), torch.tensor([.4, 1.6]), torch.tensor([1., float('nan')]), torch.tensor([1., 1.], requires_grad=True)])
def test_unbounded_not_mean_one_or_trainable_loss_weight_rejected(weights):
    with pytest.raises(ValueError):
        classification_loss(torch.zeros(2), torch.zeros(2), weights=weights)


def test_actual_torch_numpy_python_rng_and_no_hidden_sampling():
    torch.manual_seed(114)
    random.seed(114)
    np.random.seed(114)
    before = (torch.get_rng_state().clone(), random.getstate(), repr(np.random.get_state()))
    c = command(task())
    e, _ = expert_context(*window())
    weights, _ = expert_weights(c, e)
    classification_loss(torch.zeros(2, dtype=weights.dtype), torch.zeros(2, dtype=weights.dtype), weights=weights)
    after = (torch.get_rng_state(), random.getstate(), repr(np.random.get_state()))
    assert torch.equal(before[0], after[0]) and before[1:] == after[1:]


@pytest.mark.parametrize('spec', [WeightingSpec(velocity_scale_forward=True), WeightingSpec(path_turn_scale_rad_s=float('inf')), WeightingSpec(control_dt_s=.02)])
def test_unknown_clock_bool_and_nonfinite_configuration_rejected(spec):
    with pytest.raises(ValueError):
        spec.validate()


@pytest.mark.parametrize('raw', [torch.tensor([1e-308, 1.], dtype=torch.float64), torch.tensor([.5, 3.], dtype=torch.float64)])
def test_numeric_bracket_overflow_and_unknown_raw_range_rejected(raw):
    with pytest.raises(ValueError, match='bounded'):
        renormalize_bounded(raw)
