"""Literal paper equation signs and real differentiable reward boundaries."""

import pytest
import torch

from CrowdSim.humanoid.training.rewards import (
    aligned_position_error_squared, align_trajectory_xy, paper_task_reward, paper_total_reward,
)


def test_published_backward_reward_and_signed_braking_power_are_preserved():
    delta = torch.tensor([0., .5], dtype=torch.float64, requires_grad=True)
    torque = torch.ones(2, 69, dtype=torch.float64, requires_grad=True)
    velocity = torch.stack((torch.ones(69), -torch.ones(69))).double().requires_grad_(True)
    forward = torch.tensor([1., -1.8], dtype=torch.float64, requires_grad=True)
    components = paper_task_reward(delta, torque, velocity, forward)
    torch.testing.assert_close(components["trajectory"], torch.tensor([1., torch.exp(torch.tensor(-1.)).item()], dtype=torch.float64))
    torch.testing.assert_close(components["energy"], torch.tensor([-.0345, .0345], dtype=torch.float64))
    assert components["direction"][0] == 0
    torch.testing.assert_close(components["direction"][1], torch.tensor(1 - torch.exp(torch.tensor(-1.)).item(), dtype=torch.float64))
    total = paper_total_reward(torch.tensor([2., 3.], dtype=torch.float64), components)
    torch.testing.assert_close(total, torch.tensor([2., 3.], dtype=torch.float64) + .5 * components["task"])
    total.sum().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (delta, torque, velocity, forward))
    assert forward.grad[1] < 0  # Printed Eq6 rewards more backward speed.


def test_backwards_penalty_is_an_explicit_different_experiment():
    args = (torch.zeros(1), torch.zeros(1, 69), torch.zeros(1, 69), torch.tensor([-1.]))
    published = paper_task_reward(*args)
    corrected = paper_task_reward(*args, direction_mode="backwards_penalty")
    assert published["direction"] > 0 and corrected["direction"] < 0
    torch.testing.assert_close(corrected["direction"], -published["direction"])


def test_trajectory_alignment_uses_time_not_nearest_future_distance_and_is_differentiable():
    trajectory = torch.tensor([[[0., 0.], [2., 0.], [4., 0.]]], requires_grad=True)
    times = torch.tensor([[0., 2., 5.]])
    target = align_trajectory_xy(trajectory, times, torch.tensor([3.5]))
    torch.testing.assert_close(target, torch.tensor([[3., 0.]]))
    error = aligned_position_error_squared(torch.tensor([[1., 0.]]), target)
    torch.testing.assert_close(error, torch.tensor([4.]))
    error.sum().backward()
    assert torch.count_nonzero(trajectory.grad) == 2
    with pytest.raises(ValueError, match="replan"):
        align_trajectory_xy(trajectory, times, torch.tensor([5.1]))


def test_nonphysical_reward_shapes_and_values_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        paper_task_reward(torch.tensor([-1.]), torch.zeros(1, 69), torch.zeros(1, 69), torch.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        paper_task_reward(torch.zeros(1), torch.full((1, 69), float("nan")), torch.zeros(1, 69), torch.zeros(1))
    with pytest.raises(ValueError, match="69"):
        paper_task_reward(torch.zeros(1), torch.zeros(1, 28), torch.zeros(1, 28), torch.zeros(1))
