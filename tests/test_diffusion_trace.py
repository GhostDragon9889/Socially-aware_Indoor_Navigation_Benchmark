"""Focused CPU checks using the actual public TRACE weights where available."""

import math

import pytest
import torch

from CrowdSim.humanoid.diffusion import DiffusionPlanner, DiffusionTrainBatch
from CrowdSim.humanoid.diffusion.planner import DEFAULT_CHECKPOINT


def _inputs():
    current = torch.tensor([[4.0, 3.0, 1.0, math.pi / 2]])
    history = current[:, None].repeat(1, 31, 1)
    history[0, :, 1] -= torch.arange(30, -1, -1) * 0.1
    raster = torch.zeros(1, 2, 224, 224)
    raster[:, 0] = 1
    return current, history, raster


def test_trace_current_frame_local_coordinates_and_empty_neighbors():
    planner = DiffusionPlanner.__new__(DiffusionPlanner)
    planner.device = torch.device("cpu")
    current, history, raster = _inputs()
    history[:, 0] = torch.nan
    mask = torch.ones(1, 31, dtype=torch.bool)
    mask[:, 0] = False
    batch = planner.prepare_context(current, history, None, raster, history_mask=mask)
    assert batch["history_positions"][0, 1, 0] == pytest.approx(-2.9)
    assert batch["history_positions"][0, 1, 1] == pytest.approx(0, abs=1e-6)
    assert not batch["all_other_agents_history_availabilities"].any()
    assert batch["raster_from_agent"][0, :, 2].tolist() == [56, 112, 1]
    assert torch.isfinite(batch["history_positions"]).all()


@pytest.mark.needs_lfs
def test_real_trace_weights_complete_cpu_ddpm_and_guidance():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Official TRACE checkpoint is not present; no synthetic replacement")
    planner = DiffusionPlanner()
    assert planner.loaded_state_entries == 406
    current, history, raster = _inputs()
    rng = torch.get_rng_state().clone()
    first = planner.plan(current, history, None, raster, seed=7)
    repeated = planner.plan(current, history, None, raster, seed=7)
    assert torch.equal(first.xy, repeated.xy)
    assert torch.equal(rng, torch.get_rng_state())
    assert first.xy.shape == (1, 51, 2)
    assert first.sample_states.shape == (1, 1, 52, 6)
    assert first.time[-1] == pytest.approx(5.0)
    assert torch.equal(first.xy[:, 0], current[:, :2])
    unseeded_first = planner.plan(current, history, None, raster)
    unseeded_second = planner.plan(current, history, None, raster)
    assert not torch.equal(unseeded_first.xy, unseeded_second.xy)
    guided = planner.plan(current, history, None, raster, waypoints=torch.tensor([[4.0, 8.0]]), guidance_weight=0.1, seed=7)
    assert torch.isfinite(guided.xy).all()
    assert not torch.equal(guided.xy, first.xy)


@pytest.mark.needs_lfs
def test_real_trace_training_backward_and_validation_preserves_stats():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Official TRACE checkpoint is not present")
    planner = DiffusionPlanner()
    current, history, raster = _inputs()
    future = torch.zeros(1, 52, 3)
    future[:, :, :2] = current[:, None, :2]
    future[0, :, 1] += torch.arange(1, 53) * 0.1
    future[:, :, 2] = current[:, 3:4]
    batch = DiffusionTrainBatch.from_world(planner, current, history, None, raster, future)
    before = {key: value.clone() for key, value in planner.model.named_buffers() if "running_" in key}
    loss, _ = planner.training_loss(batch.clean_trajectory, batch.context, [30], noise=torch.zeros(1, 52, 6), training=False)
    assert torch.isfinite(loss)
    assert all(torch.equal(before[key], value) for key, value in planner.model.named_buffers() if key in before)
    loss.backward()
    gradient = planner.model.model.final_conv[1].weight.grad
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
