"""Actual recorded neutral-SMPL foot surface against the admitted analytical floor."""
from pathlib import Path
import sys,json,hashlib,argparse,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));torch.set_num_threads(4)
from CrowdSim.tools.smpl_mesh_visualizer import HumanMeshConfig,SMPLMeshVisualizer,SMPLRobotPoseMapper,SMPL_JOINT_NAMES,smpl_to_robot_matrix,quat_apply_wxyz
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,required=True);a=p.parse_args()
if a.output.exists():p.error('Fresh audit output required')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((a.run/'result.json').read_text());e=json.loads(a.run.with_name(a.run.name+'_execution.json').read_text());assert e['actual_returncode']==0 and e['sources_unchanged'] and r['status']=='passed' and e['result_sha256']==sha(a.run/'result.json')
z=np.load(a.run/'physical_states.npz');assert e['capture_sha256']['physical_states.npz']==sha(a.run/'physical_states.npz')
floor=json.loads(Path(r['native_scene_provider']['preflight']['path']).read_text())['reference_ground']['height_m']
visual=SMPLMeshVisualizer.from_config(HumanMeshConfig(model_dir=str(R/'data/smpl'),num_envs=z['root_pos'].shape[1],device='cpu',shape_std=0.));mapper=SMPLRobotPoseMapper();foot=[SMPL_JOINT_NAMES.index(s+'_'+j) for s in ('L','R') for j in ('Ankle','Toe')]
weights=visual.smpl_model.lbs_weights;mask=weights[:,foot].sum(-1)>=.6
assert int(mask.sum())>20 and 0<a.steps<=len(z['root_pos'])
dof=torch.from_numpy(z['dof_pos'][:a.steps].reshape(-1,69));root=torch.from_numpy(z['root_pos'][:a.steps].reshape(-1,3));q=torch.from_numpy(z['root_rot'][:a.steps].reshape(-1,4));poses=mapper(dof,r['physics']['dof_names']);mins=[]
with torch.no_grad():
 for i in range(0,len(root),32):
  count=len(root[i:i+32]);o=visual.smpl_model(betas=torch.zeros(count,10),global_orient=torch.zeros(count,3),body_pose=poses[i:i+count],transl=torch.zeros(count,3),return_verts=True)
  vertices=(o.vertices[:,mask]-o.joints[:,0:1])@smpl_to_robot_matrix().T
  world=quat_apply_wxyz(q[i:i+count][:,[3,0,1,2]][:,None,:],vertices)+root[i:i+count,None]
  mins.append(world[:,:,2].amin(-1)-floor)
minimum=torch.cat(mins).numpy();penetration=np.maximum(0,-minimum)
report={'schema':'crowdsim.recorded_SMPL_foot_surface.v1','source_result_sha256':e['result_sha256'],'physical_states_sha256':e['capture_sha256']['physical_states.npz'],'evaluator_source_sha256':sha(Path(__file__)),'source_control_interval':[0,a.steps],'frames':a.steps,'actors':z['root_pos'].shape[1],'foot_vertex_count':int(mask.sum()),'vertex_selection':'sum neutral-SMPL skinning influence on L/R ankle/toe >=0.6','floor_height_m':floor,'minimum_foot_surface_height_m':float(minimum.min()),'foot_penetration_max_m':float(penetration.max()),'foot_penetration_p95_m':float(np.quantile(penetration,.95)),'actor_frame_fraction_foot_vertices_below_minus1cm':float((minimum<-.01).mean()),'actor_frame_fraction_foot_vertices_below_minus2mm':float((minimum<-.002).mean()),'diagnostic_only':True,'physics_or_pose_modified':False,'limits':['Neutral SMPL skinning surface compared to analytical reference floor, not exact contact-patch slip.','Original visual tiles can lie slightly above the analytical support surface; retained geometry is inspected in the video.']}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['frames','foot_vertex_count','minimum_foot_surface_height_m','foot_penetration_max_m','actor_frame_fraction_foot_vertices_below_minus1cm']}))
