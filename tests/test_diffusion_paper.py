"""Actual five-state network CPU training/sampling and coordinate contracts."""

import math

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner, PaperDiffusionTrainBatch, rasterize_navigation_map
from CrowdSim.humanoid.diffusion.paper import world_to_agent_state, waypoint_guidance_gradient


def _inputs():
    current = torch.tensor([[1.0, 2.0, 0.7, 0.2, 0.3]])
    history = current[:, None].repeat(1, 31, 1)
    history[0, :, :2] -= torch.arange(30, -1, -1)[:, None] * 0.1 * current[0, 2:4]
    future = current[:, None].repeat(1, 52, 1)
    future[0, :, :2] += torch.arange(1, 53)[:, None] * 0.1 * current[0, 2:4]
    raster = torch.zeros(1, 2, 224, 224)
    raster[:, 0] = 1
    return current, history, future, raster


def test_five_state_world_transform_preserves_lateral_velocity():
    current = torch.tensor([[4.0, 3.0, 1.0, 0.0, math.pi / 2]])
    other = torch.tensor([[[5.0, 3.0, 1.0, 0.0, math.pi / 2]]])
    local = world_to_agent_state(other, current)
    assert local[0, 0, 0] == pytest.approx(0, abs=1e-6)
    assert local[0, 0, 1] == pytest.approx(-1)
    assert local[0, 0, 2] == pytest.approx(0, abs=1e-6)
    assert local[0, 0, 3] == pytest.approx(-1)
    assert local[0, 0, 4] == pytest.approx(0)


def test_waypoint_guidance_is_independent_of_batch_and_sample_chunk_size():
    trajectory = torch.arange(52 * 5, dtype=torch.float32).reshape(1, 52, 5) / 100
    targets = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    indices = torch.tensor([20, 49])
    single = waypoint_guidance_gradient(trajectory, targets, indices)
    batched = waypoint_guidance_gradient(trajectory.repeat(64, 1, 1), targets.repeat(64, 1, 1), indices)
    multisample = waypoint_guidance_gradient(trajectory.repeat(8, 1, 1), targets, indices, num_samples=8)
    assert torch.equal(single.expand_as(batched), batched)
    assert torch.equal(single.expand_as(multisample), multisample)
    assert single[..., 2:].count_nonzero() == 0


@pytest.mark.parametrize("batch,samples,indices", [(1, 1, [49]), (4, 3, [2, 2, 49]), (2, 2, [1, 2, 2, 49])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_closed_waypoint_gradient_matches_actual_autograd_repeated_indices(batch, samples, indices, dtype):
    generator = torch.Generator().manual_seed(91)
    mean = torch.randn(batch * samples, 52, 5, generator=generator, dtype=dtype, requires_grad=True)
    target = torch.randn(batch, len(indices), 2, generator=generator, dtype=dtype)
    index = torch.tensor(indices)
    residual = mean[:, index, :2].reshape(batch, samples, -1, 2) - target[:, None]
    expected = torch.autograd.grad(residual.square().mean(dim=(-1, -2)).sum(), mean)[0]
    actual = waypoint_guidance_gradient(mean, target, index, num_samples=samples)
    torch.testing.assert_close(actual, expected, atol=2e-7 if dtype == torch.float32 else 1e-15, rtol=1e-6 if dtype == torch.float32 else 1e-15)
    assert not actual.requires_grad
    assert actual[..., 2:].count_nonzero() == 0
    # No gradient computation or modification of the caller's parameter occurs.
    assert mean.grad is None


def test_optional_compiled_forward_keeps_state_config_rng_and_full_sampler():
    planner = PaperDiffusionPlanner()
    planner.trained_steps = 1  # Sampling contract fixture; not a published weight.
    current, history, _, raster = _inputs()
    state_before = {name: value.clone() for name, value in planner.model.state_dict().items()}
    config_before = planner.config
    rng_before = torch.get_rng_state().clone()
    plain = planner.plan(current, history, None, raster, waypoints=[[4., 3.]], seed=91)
    planner.set_compiled_forward(True, backend="eager")
    assert planner.compiled_forward_enabled
    compiled = planner.plan(current, history, None, raster, waypoints=[[4., 3.]], seed=91)
    torch.testing.assert_close(compiled.sample_states, plain.sample_states, atol=1e-6, rtol=1e-6)
    assert torch.equal(compiled.time, plain.time) and len(compiled.time) == 51
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert planner.config == config_before
    assert set(state_before) == set(planner.model.state_dict())
    assert all(torch.equal(value, state_before[name]) for name, value in planner.model.state_dict().items())
    planner.set_compiled_forward(False)
    assert not planner.compiled_forward_enabled
    with pytest.raises(ValueError, match="boolean"):
        planner.set_compiled_forward("yes")


def test_five_state_masks_do_not_invent_neighbors():
    planner = PaperDiffusionPlanner.__new__(PaperDiffusionPlanner)
    planner.device = torch.device("cpu")
    current, history, _, raster = _inputs()
    history[:, :20] = torch.nan
    mask = torch.zeros(1, 31, dtype=torch.bool)
    mask[:, 20:] = True
    batch = planner.prepare_context(current, history, None, raster, history_mask=mask)
    assert torch.isfinite(batch["history"]).all()
    assert not batch["neighbor_mask"].any()
    bad = mask.clone()
    bad[:, -1] = False
    with pytest.raises(ValueError, match="current ego"):
        planner.prepare_context(current, history, None, raster, history_mask=bad)


def test_five_state_real_update_sampler_and_checkpoint_resume(tmp_path):
    planner = PaperDiffusionPlanner()
    current, history, future, raster = _inputs()
    with pytest.raises(RuntimeError, match="no trained weights"):
        planner.plan(current, history, None, raster)
    batch = PaperDiffusionTrainBatch.from_world(planner, current, history, None, raster, future)
    optimizer = torch.optim.Adam(planner.model.parameters(), lr=1e-4)
    weight = planner.model.temporal.final_conv[1].weight
    before = weight.detach().clone()
    values = planner.train_step(batch, optimizer, timesteps=[50], noise=torch.zeros(1, 52, 5), drop_conditioning=False)
    assert np.isfinite(values["loss"]) and values["grad_norm"] > 0
    assert not torch.equal(before, weight)
    assert planner.model.temporal.final_conv[1].out_channels == 5
    assert weight.grad[2:4].norm() > 0
    rng = torch.get_rng_state().clone()
    result = planner.plan(current, history, None, raster, waypoints=[[4.0, 3.0]], seed=7)
    assert result.xy.shape == result.velocity.shape == (1, 51, 2)
    assert result.sample_states.shape == (1, 1, 52, 5)
    assert result.time[-1] == pytest.approx(5.0)
    assert torch.equal(result.velocity[:, 0], current[:, 2:4])
    assert torch.equal(rng, torch.get_rng_state())
    unseeded_first = planner.plan(current, history, None, raster)
    unseeded_second = planner.plan(current, history, None, raster)
    assert not torch.equal(unseeded_first.xy, unseeded_second.xy)
    checkpoint = tmp_path / "five_state.ckpt"
    planner.save_training_checkpoint(checkpoint, optimizer=optimizer, dataset_metadata={"source": "analytic CPU regression fixture"})
    expected_rng = torch.rand(4)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        weight.add_(100)
    restored = planner.restore_training_checkpoint(checkpoint, optimizer=optimizer)
    assert restored["step"] == 1 and planner.trained_steps == 1
    assert torch.equal(torch.rand(4), expected_rng)
    resumed = planner.plan(current, history, None, raster, waypoints=[[4.0, 3.0]], seed=7)
    assert torch.equal(result.xy, resumed.xy)


def test_five_state_validation_does_not_modify_batchnorm():
    planner = PaperDiffusionPlanner()
    current, history, future, raster = _inputs()
    batch = PaperDiffusionTrainBatch.from_world(planner, current, history, None, raster, future)
    before = {name: value.clone() for name, value in planner.model.named_buffers() if "running_" in name}
    with torch.no_grad():
        loss, _ = planner.training_loss(batch.clean_trajectory, batch.context, timesteps=[40], training=False)
    assert torch.isfinite(loss)
    assert all(torch.equal(value, before[name]) for name, value in planner.model.named_buffers() if name in before)


def test_trace_raster_orientation_uses_navigation_transform():
    class Task:
        free_mask = np.ones((100, 100), dtype=bool)

        def world_to_pixel(self, points):
            return np.floor(points[..., [1, 0]] + 50).astype(np.int64)

    task = Task()
    task.free_mask[50, 51] = False
    raster = rasterize_navigation_map(task, [[0, 0, 0, math.pi / 2]])
    # A world obstacle +x=1 is local y=-1 at heading pi/2.
    assert raster[0, 0, 100, 56] == 0
    assert raster[0, 1, 100, 56] == 1
    paper_raster = rasterize_navigation_map(task, [[0, 0, 0, 0, math.pi / 2]])
    assert torch.equal(raster, paper_raster)
