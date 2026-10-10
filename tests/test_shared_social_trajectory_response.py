"""Analytic prefix/storage/action/KL checks independent of actual NN weights."""
import numpy as np
import pytest
import torch

from CrowdSim.humanoid.analysis.trajectory_response import CASES, action_response, task_cases


def test_only_trajectory_changes_and_each_case_owns_storage():
    raw = torch.arange(2*12308, dtype=torch.float32).reshape(2, 12308)
    before = raw.clone()
    cases = task_cases(raw)
    assert torch.equal(cases[0], raw) and torch.equal(raw, before)
    assert all(torch.equal(case[:, 20:], raw[:, 20:]) for case in cases)
    assert torch.equal(cases[1, :, :20], torch.zeros(2, 20))
    offsets = torch.arange(1, 11)/2
    for case, axis, sign in ((2, 0, 1), (3, 0, -1), (4, 1, 1), (5, 1, -1)):
        assert torch.equal(cases[case, :, axis:20:2], (sign*offsets).expand(2, -1))
        assert torch.equal(cases[case, :, 1-axis:20:2], torch.zeros(2, 10))
    cases[2, 0, 20] = -100
    assert torch.equal(raw, before) and cases[3, 0, 20] == raw[0, 20]


@pytest.mark.parametrize("bad", [torch.zeros(0, 12308), torch.zeros(2, 20), torch.zeros(2, 12308, dtype=torch.float64),
    torch.full((2, 12308), float('nan')), torch.zeros(2, 12308, requires_grad=True)])
def test_invalid_raw_input_rejected(bad):
    with pytest.raises(ValueError):
        task_cases(bad)


def outputs():
    means = np.zeros((len(CASES), 2, 69), np.float32)
    mask = np.arange(69) < 57
    means[2, :, :57] = .2
    means[3, :, :57] = 2
    means[4, :, 57:] = 500
    return means, np.clip(means, -1, 1)*mask, mask


def test_known_equal_sigma_KL_and_clipped_actions_exclude_inactive_dimensions():
    means, applied, mask = outputs()
    result = action_response(means, applied, mask, .1)
    assert np.allclose(result['mean_delta_active_RMS'][2], .2)
    assert np.allclose(result['unclipped_active_equal_std_Gaussian_KL'][2], .5*57*4)
    assert np.array_equal(result['applied_delta_active_RMS'][3], [1., 1.])
    assert np.array_equal(result['strictly_clipped_active_mean_coordinates'][3], [57, 57])
    assert np.array_equal(result['mean_delta_active_RMS'][4], [0., 0.])
    assert not result['mean_delta_active_RMS'][0].any()


@pytest.mark.parametrize("bad", ['mask', 'clipping', 'std', 'nonfinite'])
def test_invalid_actor_outputs_rejected(bad):
    means, applied, mask = outputs()
    std = .1
    if bad == 'mask':
        mask[:] = True
    if bad == 'clipping':
        applied[3, 0, 0] = 2
    if bad == 'std':
        std = 0.
    if bad == 'nonfinite':
        means[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        action_response(means, applied, mask, std)
