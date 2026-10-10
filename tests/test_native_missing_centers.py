"""Contract fixtures: absent SDK position rows require exact source labels and all queries."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
R=Path(__file__).resolve().parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=module('native_grid_fixtures',R/'tests/test_native_grid_scene.py')
probe=module('native_probe_contract',R/'scripts/probe_paper_scene_maps.py')

def fixture(tmp_path):
 p=base.fixture_provider(tmp_path)
 def edit(d):
  rawpath=Path(d['native_omap']['path'])
  with np.load(rawpath) as a:raw={k:a[k].copy() for k in a.files}
  raw['free_positions']=raw['free_positions'][1:]
  np.savez(rawpath,**raw);d['native_omap']=base.file_record(rawpath)
  image,coords=probe.reconstruct_native_map(raw['buffer'],raw['occupied_positions'],raw['free_positions'],raw['min_bound'],raw['max_bound'],raw['dimensions'],known5_position_source='official_buffer_centers')
  c=probe.canonical_known5_positions(raw['buffer'],raw['free_positions'],raw['min_bound'],raw['max_bound'],raw['dimensions'],source='official_buffer_centers')
  ix=probe.native_position_buffer_indices(raw['free_positions'],raw['min_bound'],raw['max_bound'],raw['dimensions']);labels=raw['buffer'][ix]
  f=Path(d['coordinate_validation']['canonical_world_positions_artifact']['path'])
  np.savez(f,canonical_free_world_positions=c['positions'],occupied_world_positions=raw['occupied_positions'],canonical_free_API_row=c['API_rows'],canonical_free_position_source=c['position_source'],canonical_known5_native_buffer_indices=c['native_buffer_indices'],missing_known5_buffer_indices=c['missing_known5_buffer_indices'],missing_known5_world_positions=c['missing_known5_world_positions'],API_free_buffer_indices=ix,API_free_buffer_labels=labels,API_free_unknown_world_positions=raw['free_positions'][labels==6],unreported_unknown_buffer_indices=np.array([],dtype=np.int64),unreported_unknown_buffer_labels=np.array([],dtype=raw['buffer'].dtype))
  coords['canonical_world_positions_artifact']=base.file_record(f);d['coordinate_validation']=coords
  for key in ('floor_support','full_height_band_queries'):
   f=Path(d[key]['artifact']['path'])
   with np.load(f) as a:arr={k:a[k].copy() for k in a.files}
   arr['free_world_positions']=c['positions'];
   if key=='floor_support':
    hit=c['positions'].copy();hit[:,2]=0.
    arr['hit_position']=hit;arr['all_hit_positions']=hit
   np.savez(f,**arr);d[key]['artifact']=base.file_record(f)
  d['floor_ownership']['artifact']=d['floor_support']['artifact']
  d.update(native_direct_cell_validation=True,known5_position_source='official_buffer_centers',known5_API_complete=False)
 base.mutate_native(p,edit)
 return p

def test_explicit_missing_center_requires_complete_raw_native_proof(tmp_path):
 grid=base.scene(fixture(tmp_path));assert grid.describe()['admission']['canonical_known5_cells']==7200

@pytest.mark.parametrize('corruption',['source','validation','count','derived','unknown','rows'])
def test_missing_center_corruption_rejected(tmp_path,corruption):
 p=fixture(tmp_path)
 def edit(d):
  if corruption=='source':d['coordinate_validation']['known5_position_source']='api_verified'
  elif corruption=='validation':d['native_direct_cell_validation']=False
  elif corruption=='count':d['coordinate_validation']['missing_known5_API_count']=2
  else:
   record=d['coordinate_validation']['canonical_world_positions_artifact'];f=Path(record['path'])
   with np.load(f) as a:arr={k:a[k].copy() for k in a.files}
   if corruption=='derived':arr['missing_known5_world_positions'][0,0]+=.1
   elif corruption=='unknown':arr['API_free_buffer_labels'][0]=6
   else:arr['canonical_free_API_row'][-1]=0
   np.savez(f,**arr);d['coordinate_validation']['canonical_world_positions_artifact']=base.file_record(f)
 base.mutate_native(p,edit)
 with pytest.raises(ValueError):base.scene(p)

def test_cyclic_lookahead_anticipates_next_leg_without_relocating_root(tmp_path):
 import torch
 p=base.fixture_provider(tmp_path)
 routes=[[[2.55,-.05],[2.55,2.55],[-3.45,2.55],[-3.45,-.05]]]
 start=torch.tensor([[2.15,-.05]])
 g=base.NativeGridScene.from_provider(p,origins_xy=[[-3.45,-.05]],routes_xy=routes,clearance_m=0.,arrival_radius_m=.15,goal_lookahead_m=5.,cyclic_route_lookahead=True)
 points,index,reached=g.route_waypoints(start,torch.tensor([0]),advance=False)
 assert float(points[0,-1,1])>2.
 assert index.tolist()==[0] and not reached.any()
 assert torch.equal(start,torch.tensor([[2.15,-.05]]))
 legacy=base.NativeGridScene.from_provider(p,origins_xy=[[-3.45,-.05]],routes_xy=routes,clearance_m=0.,arrival_radius_m=.15,goal_lookahead_m=5.)
 old,_,_=legacy.route_waypoints(start,torch.tensor([0]),advance=False)
 assert torch.allclose(old[0,-1],torch.tensor([2.55,-.05]))

def test_bounded_connector_plans_from_actual_root_without_teleport(tmp_path):
 import torch
 p=base.fixture_provider(tmp_path)
 g=base.NativeGridScene.from_provider(p,origins_xy=[[-3.45,.05]],routes_xy=[[[2.55,.05],[-3.45,.05]]],clearance_m=.55,start_recovery_radius_m=.8,start_recovery_clearance_m=.4)
 actual=torch.tensor([[-4.45,.05]])
 before=actual.clone();points,_,_=g.route_waypoints(actual,torch.tensor([0]),advance=False)
 assert g.global_path_failures==0 and g.successful_start_recoveries==1
 assert torch.equal(actual,before)
 assert g.last_start_recovery_segments[0]['physical_relocation'] is False
 assert g.last_start_recovery_segments[0]['length_m']<=.8
 assert float(points[0,-1,0])>float(actual[0,0])
 unsafe=torch.tensor([[-4.85,.05]])
 g.route_waypoints(unsafe,torch.tensor([0]),advance=False)
 assert g.global_path_failures==1
