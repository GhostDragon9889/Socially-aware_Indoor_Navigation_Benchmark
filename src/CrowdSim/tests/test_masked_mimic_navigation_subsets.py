"""Real navigation/context methods with CPU state, without simulator fixtures."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.control.drive import yaw_to_quaternion
from CrowdSim.nav_manager import CrowdNavigationManager
from protomotions.envs.context_views import MaskedMimicContext


def manager_and_context(ids, *, enabled=True):
    manager = object.__new__(CrowdNavigationManager)
    manager.config = NS(num_humanoids=3, humanoid_target_enabled=enabled,
        humanoid_target_min_heading_speed=.05, humanoid_max_yaw_rate=2., humanoid_max_acceleration=1.)
    manager._trajectory_log_file = manager._trajectory_timestamp_file = None
    manager.reached = np.array([True,False,False])
    manager._sfm_desired_velocities = np.array([[1.,0.],[0.,2.],[-1.,0.]], np.float32)
    manager._humanoid_actual_velocities = manager._sfm_desired_velocities.copy()
    manager._humanoid_target_yaws = np.array([0.,np.pi/2,np.pi], np.float32)
    manager._humanoid_future_first_targets = np.full((3,2), -99., np.float32)
    manager._humanoid_future_first_yaws = np.full(3, -99., np.float32)
    manager._humanoid_target_offsets = lambda count: .5 * np.arange(1,count+1,dtype=np.float32)
    manager._frozen_slot_xy = lambda: np.array([-10.,-20.], np.float32)
    rows = len(ids)
    current_pos = torch.zeros(rows,3,3)
    for row, global_id in enumerate(ids):
        current_pos[row,1] = torch.tensor([10.*global_id,10.*global_id+1,global_id+.75])
    current_rot = yaw_to_quaternion(torch.tensor(ids, dtype=torch.float32) * .1)[:,None,:].repeat(1,3,1)
    base = MaskedMimicContext(mimic=None, ref_pos=torch.full((rows,2,3,3),97.),
        ref_rot=torch.tensor([0.,0.,0.,1.]).repeat(rows,2,3,1),
        target_times=torch.full((rows,2),-1.), time_offsets=torch.zeros(rows,2),
        target_poses_masks=torch.ones(rows,2,dtype=torch.bool),
        target_bodies_masks=torch.ones(rows,8,dtype=torch.bool))
    ctx = NS(masked_mimic=base, env_ids=torch.tensor(ids,dtype=torch.long),
        current=NS(rigid_body_pos=current_pos, rigid_body_rot=current_rot))
    component = NS(conditionable_body_ids=torch.tensor([1,2]),num_conditionable_bodies=2,
        env=NS(num_envs=5, robot_config=NS(anchor_body_index=1),
            motion_manager=NS(motion_times=torch.tensor([10.,20.,30.,40.,50.]))))
    return manager, component, ctx


@pytest.mark.parametrize("ids", [[1],[2,0],[4,1,3],[0,1,2,3,4]])
def test_subset_targets_use_global_clock_masks_velocity_and_freeze_ids(ids):
    manager, component, ctx = manager_and_context(ids)
    current = ctx.current
    manager._override_masked_mimic_context(component,ctx)
    result = ctx.masked_mimic
    masks = result.target_bodies_masks.reshape(len(ids),2,2,2)
    for row, global_id in enumerate(ids):
        torch.testing.assert_close(result.target_times[row], torch.tensor([10.*(global_id+1)+.5,10.*(global_id+1)+1.]))
        torch.testing.assert_close(result.time_offsets[row], torch.tensor([.5,1.]))
        torch.testing.assert_close(result.ref_pos[row,:,1,2],current.rigid_body_pos[row,1,2].expand(2))
        assert not masks[row,:,1].any()  # Only the conditionable pelvis is constrained.
        if global_id < 3:
            expected = current.rigid_body_pos[row,1,:2] + torch.tensor([.5,1.])[:,None] * torch.from_numpy(manager._sfm_desired_velocities[global_id])
            torch.testing.assert_close(result.ref_pos[row,:,1,:2],expected)
            assert bool(result.target_poses_masks[row].all()) == (not manager.reached[global_id])
            np.testing.assert_allclose(manager._humanoid_future_first_targets[global_id],expected[0].numpy(), atol=1e-6)
        else:
            torch.testing.assert_close(result.ref_pos[row,:,1,:2],torch.tensor([[-10.,-20.],[-10.,-20.]]))
            torch.testing.assert_close(result.ref_rot[row,:,1],current.rigid_body_rot[row,1].expand(2,4))
            assert result.target_poses_masks[row].all()
    untouched = set(range(3))-set(ids)
    for global_id in untouched:
        np.testing.assert_array_equal(manager._humanoid_future_first_targets[global_id],[-99.,-99.])


def test_global_context_without_ids_preserves_full_batch_behavior():
    manager, component, ctx = manager_and_context([0,1,2,3,4])
    del ctx.env_ids
    manager._override_masked_mimic_context(component,ctx)
    torch.testing.assert_close(ctx.masked_mimic.target_times[:,0],torch.tensor([10.5,20.5,30.5,40.5,50.5]))


def test_disabled_navigation_preserves_active_targets_while_freezing_fillers():
    manager, component, ctx = manager_and_context([4,1,3], enabled=False)
    original_pos = ctx.masked_mimic.ref_pos.clone()
    original_mask = ctx.masked_mimic.target_bodies_masks.clone()
    manager._override_masked_mimic_context(component,ctx)
    torch.testing.assert_close(ctx.masked_mimic.ref_pos[1], original_pos[1])
    torch.testing.assert_close(ctx.masked_mimic.target_bodies_masks[1], original_mask[1])
    assert ctx.masked_mimic.target_poses_masks[[0,2]].all()


@pytest.mark.parametrize("bad_ids", [[-1],[5],[1,2]])
def test_invalid_subset_ids_fail_before_indexing(bad_ids):
    manager, component, ctx = manager_and_context([1])
    ctx.env_ids = torch.tensor(bad_ids)
    with pytest.raises(ValueError, match="env_ids"):
        manager._override_masked_mimic_context(component,ctx)


def test_future_target_cache_uses_non_contiguous_ids():
    manager, _, _ = manager_and_context([1])
    targets, _ = manager._humanoid_future_targets_from_path(
        np.array([[5.,6.],[7.,8.]],np.float32), np.array([.5],np.float32), agent_ids=np.array([2,0]))
    np.testing.assert_allclose(targets[:,0],[[4.5,6.],[7.5,8.]],atol=1e-6)
    np.testing.assert_allclose(manager._humanoid_future_first_targets,[[7.5,8.],[-99.,-99.],[4.5,6.]],atol=1e-6)
