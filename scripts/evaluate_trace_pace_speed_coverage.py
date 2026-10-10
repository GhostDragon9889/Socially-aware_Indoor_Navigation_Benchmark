"""Independent all-frame actual speed and declared reachable-area coverage.

Cruise is selected by planned intent and turn rate, never by actual speed.
No stopped or turning frame is removed from the upper-speed check.
"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from scipy.ndimage import label
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
import argparse
def stats(values):
 a=np.asarray(values,dtype=float).reshape(-1)
 return None if not len(a) else {'frames':len(a),'mean':float(a.mean()),'p05':float(np.quantile(a,.05)),'p50':float(np.quantile(a,.5)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
def evaluate(run,scene_path):
 run=Path(run);result=json.loads((run/'result.json').read_text());e=json.loads(run.with_name(run.name+'_execution.json').read_text())
 assert e['actual_returncode']==0 and e['sources_unchanged'] and result['cleanup_succeeded'] and e['capture_sha256']['physical.npz']==sha(run/'physical.npz') and e['result_sha256']==sha(run/'result.json')
 assert e['inputs_unchanged'] and sha(scene_path)==e['input_snapshot']['--scene']['sha256'] and sha(e['input_snapshot']['--scene']['snapshot'])==e['input_snapshot']['--scene']['sha256'], 'Exact native-tested scene bytes required for coverage'
 c=json.loads(Path(scene_path).read_text());p=DynBenchSceneProvider.from_preflight(c['preflight']['path'],expected_sha256=c['preflight']['sha256'],probe_script_snapshot=c['probe_script_snapshot']['path'])
 p.admit_native_map(c['native_map']['path'],expected_sha256=c['native_map']['sha256']);p.configure_policy_height_feature('reference_floor')
 n=NativeGridScene.from_provider(p,origins_xy=c['origins_xy'],routes_xy=c['routes_xy'],clearance_m=c['clearance_m'],device='cpu')
 with np.load(run/'physical.npz') as z:
  samples=z['samples'].copy();planned=z['planned_xy'].copy();selected=z['selected_samples'].copy()
 planning=json.loads((run/'planning.json').read_text());assert len(planning)==len(planned)==len(selected)
 v=np.linalg.norm(samples[...,2:4],axis=-1);xy=samples[...,:2];smooth=uniform_filter1d(v,size=15,axis=0,mode='nearest')
 cruise=np.zeros(v.shape,bool);intent=[]
 for i,record in enumerate(planning):
  a=record['physical_control'];b=planning[i+1]['physical_control'] if i+1<len(planning) else len(v)
  # TRACE samples span exactly five seconds in 51 points at 10Hz.
  d=np.diff(planned[i,:,:11],axis=1);speed=np.linalg.norm(d,axis=-1)*10
  headings=np.arctan2(d[...,1],d[...,0]);turn=np.abs(np.arctan2(np.sin(np.diff(headings,axis=-1)),np.cos(np.diff(headings,axis=-1)))).sum(-1)
  heading_error=np.abs(np.arctan2(np.sin(headings[:,0]-samples[a,:,4]),np.cos(headings[:,0]-samples[a,:,4])))
  command=(speed[:,:5].mean(-1)>=.55)&(turn<=.35)&(heading_error<=np.pi/4)&(selected[i]>=0)
  cruise[max(a,36):b]=command
  intent.append({'physical_control':a,'first1s_mean_intent_speed_m_s':speed.mean(-1).tolist(),'first0p5s_mean_intent_speed_m_s':speed[:,:5].mean(-1).tolist(),'pelvis_to_intent_heading_error_rad':heading_error.tolist(),'first1s_heading_change_abs_rad':turn.tolist(),'cruise_by_intent':command.tolist()})
 pixels=np.argwhere(n._walkable);world=n.pixel_to_world(pixels);components,k=label(n._walkable,np.array([[0,1,0],[1,1,1],[0,1,0]]))
 ids=components[tuple(n.world_to_pixel(np.asarray(c['origins_xy'])).T)];assigned=np.isin(components,np.unique(ids))&n._walkable
 all_cells=np.argwhere(n._labels==254);all_world=n.pixel_to_world(all_cells);distance=cKDTree(xy.reshape(-1,2)).query(all_world)[0]
 vicinity=.75;covered=distance<=vicinity
 safe= n._walkable[tuple(all_cells.T)];reachable=assigned[tuple(all_cells.T)]
 rows=[]
 for actor in range(v.shape[1]):
  m=cruise[:,actor];rows.append({'actor':actor,'all_frame_speed_m_s':stats(v[:,actor]),'cruise_intent_frames':int(m.sum()),'cruise_actual_speed_m_s':stats(v[m,actor]),'cruise_actual_0p5s_mean_speed_m_s':stats(smooth[m,actor]),'cruise_rolling_below0p5_fraction':float((smooth[m,actor]<.5).mean()) if m.any() else None,'all_frame_above1p5_count':int((v[:,actor]>1.5).sum()),'all_frame_below0p5_fraction':float((v[:,actor]<.5).mean()),'path_length_m':float(np.linalg.norm(np.diff(xy[:,actor],axis=0),axis=-1).sum()),'cruise_minimum5s':int(m.sum())>=150})
 checks={'actual_speed_upper1p5_all_frames':bool((v<=1.5).all()),'each_actor_cruise_at_least5s':all(r['cruise_minimum5s'] for r in rows),'each_actor_cruise_rolling_mean_in_band':all(r['cruise_actual_0p5s_mean_speed_m_s'] is not None and .5<=r['cruise_actual_0p5s_mean_speed_m_s']['mean']<=1.5 for r in rows),'each_actor_cruise_below0p5_fraction_at_most10pct':all(r['cruise_rolling_below0p5_fraction'] is not None and r['cruise_rolling_below0p5_fraction']<=.1 for r in rows)}
 coverage={'vicinity_radius_m':vicinity,'definition':'union of .75m planar neighborhoods around actual root samples, intersected with native known-free cell centers; exploration vicinity, not body contact footprint','known_free_area_m2':len(all_cells)*.01,'safe_cell_area_m2':int(safe.sum())*.01,'spawn_reachable_safe_area_m2':int(reachable.sum())*.01,'actual_vicinity_known_free_area_m2':int(covered.sum())*.01,'actual_vicinity_fraction_known_free':float(covered.mean()),'actual_vicinity_fraction_safe':float(covered[safe].mean()),'actual_vicinity_fraction_spawn_reachable_safe':float(covered[reachable].mean()),'spawn_component_ids':ids.tolist(),'no_claim_whole_USD_coverage':True}
 return {'schema':'crowdsim.actual_speed_coverage.v3','execution_sha256':sha(run.with_name(run.name+'_execution.json')),'evaluator_sha256':sha(__file__),'scene_bound_to_actual_native_input':True,'result_sha256':sha(run/'result.json'),'physical_sha256':sha(run/'physical.npz'),'planning_sha256':sha(run/'planning.json'),'scene_config_sha256':sha(scene_path),'all_frames_included_in_speed_cap':True,'cruise_selection':'planned first0.5s mean speed >=.55m/s, first1s absolute heading increments <=.35rad, pelvis heading within45deg of planned initial direction, selected diffusion sample >=0, after1.2s startup; independent of actual speed','checks':checks,'speed_band_screen_passed':all(checks.values()),'per_actor':rows,'coverage':coverage,'intent_classification':intent,'limits':['Stopping and deceleration for turns or safety can fall below .5m/s and are separately reported.','Measured velocities are physical root instantaneous velocities; .5s smoothing is a diagnostic only and does not change rendered motion.','These are local operational criteria, not author benchmark metrics.']}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Fresh output required')
 result=evaluate(a.run,a.scene);put(a.output,result);print(json.dumps({k:v for k,v in result.items() if k in ('checks','per_actor','coverage')},ensure_ascii=False))
 raise SystemExit(0 if result['speed_band_screen_passed'] else 1)
