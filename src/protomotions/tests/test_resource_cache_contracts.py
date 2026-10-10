"""Real cache IO regression for tensor-safe terrain/motion subset resources."""
import numpy as np
import pytest
import torch
from protomotions.components.terrains.terrain import Terrain
from protomotions.tests.test_terrain_class import _flat_config
from scripts.subset_motion_lib import subset_motion_lib
from scripts.check_motion_asset_consistency import check_motion_asset


def test_terrain_tensor_cache_roundtrip_uses_real_restricted_loader(tmp_path):
    path = tmp_path / 'terrain.pt'
    terrain = Terrain(_flat_config(save_terrain=True, terrain_path=str(path)), num_envs=1, device=torch.device('cpu'))
    payload = torch.load(path, weights_only=True)
    assert torch.is_tensor(payload['height_field_raw'])
    loaded = Terrain(_flat_config(load_terrain=True, terrain_path=str(path)), num_envs=1, device=torch.device('cpu'))
    np.testing.assert_equal(loaded.height_field_raw, terrain.height_field_raw)


def test_legacy_numpy_terrain_cache_still_loads_safely_and_checks_shape(tmp_path):
    path = tmp_path/'legacy.pt'
    payload = dict(height_field_raw=np.ones((6,12),dtype=np.int16),walkable_field_raw=np.zeros((6,12),dtype=np.int16))
    torch.save(payload,path)
    loaded = Terrain(_flat_config(load_terrain=True,terrain_path=str(path)),num_envs=1,device=torch.device('cpu'))
    assert loaded.height_samples.max() == pytest.approx(.1)
    payload['height_field_raw']=np.zeros((2,2),dtype=np.int16)
    torch.save(payload,path)
    with pytest.raises(ValueError,match='shape.*does not match'):
        Terrain(_flat_config(load_terrain=True,terrain_path=str(path)),num_envs=1,device=torch.device('cpu'))


def test_motion_subset_preserves_goal_states_and_optional_metadata(tmp_path):
    source=tmp_path/'source.pt'
    output=tmp_path/'subset.pt'
    torch.save(dict(gts=torch.arange(18).reshape(6,1,3).float(),length_starts=torch.tensor([0,2,4]),
        motion_num_frames=torch.tensor([2,2,2]),motion_lengths=torch.ones(3),motion_dt=torch.ones(3),
        motion_weights=torch.tensor([.2,.3,.5]),motion_files=('a','b','c'),
        goal_states=torch.tensor([False,True,False,True,True,False]),contacts=None,lrs=None,
        conversion_metadata={'robot':'fixture'}),source)
    subset_motion_lib(source,output,sample_every=2)
    result=torch.load(output,weights_only=True)
    torch.testing.assert_close(result['goal_states'],torch.tensor([False,True,True,False]))
    torch.testing.assert_close(result['length_starts'],torch.tensor([0,2]))
    assert result['motion_files']==('a','c')
    assert result['conversion_metadata']=={'robot':'fixture'}
    assert result['contacts'] is None
    with pytest.raises(ValueError,match='positive integer'):
        subset_motion_lib(source,output,sample_every=0)
    with pytest.raises(ValueError,match='differ'):
        subset_motion_lib(source,source,sample_every=1)


def test_read_only_current_asset_check_keeps_stale_g1_cache_visible():
    report=check_motion_asset('examples/data/g1_crouch_to_run.motion','g1')
    assert report['status']=='mismatch'
    assert report['body_pos_error_max_m'] > .01
    assert report['source_unchanged']


def test_out_of_grid_height_queries_use_boundary_heights_without_extrapolation():
    from protomotions.components.terrains.terrain_utils import get_heights_jit, get_height_maps_jit
    grid = torch.tensor([[0., 2.], [10., 12.]])
    positions = torch.tensor([[-1., -1., 5.], [100., 100., 5.], [-1., .5, 5.]])
    actual = get_heights_jit(positions, grid, 1.)
    torch.testing.assert_close(actual[:, 0], torch.tensor([0., 12., 1.]))
    height_maps = get_height_maps_jit(
        base_rot=torch.tensor([[0.,0.,0.,1.]]).expand(3,-1), base_pos=positions,
        height_points=torch.zeros(3,1,3), height_samples=grid,
        num_height_points=1, terrain_horizontal_scale=1., w_last=True, return_all_dims=False,
    )
    torch.testing.assert_close(height_maps[:, 0], torch.tensor([5., -7., 4.]))
