"""CPU projection of every recorded neutral SMPL vertex through declared native cameras.

No pose is edited. This is a framing test, not an occlusion or realism certificate.
"""
from pathlib import Path
import sys,json,hashlib,argparse,numpy as np,torch
R=Path(__file__).resolve().parents[1]
if not (R/'CrowdSim').is_dir():R=Path('/project/EXP/SangforAndSaut/ProtoMotions')
sys.path.insert(0,str(R));torch.set_num_threads(4)
from CrowdSim.tools.smpl_mesh_visualizer import HumanMeshConfig,SMPLMeshVisualizer,SMPLRobotPoseMapper,smpl_to_robot_matrix,quat_apply_wxyz
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--camera-plan',type=Path,required=True);p.add_argument('--actors',required=True,help='JSON per-shot visible actor lists');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('Fresh output required')
r=json.loads((a.run/'result.json').read_text());e=json.loads(a.run.with_name(a.run.name+'_execution.json').read_text());assert e['actual_returncode']==0 and e['sources_unchanged'] and e['result_sha256']==sha(a.run/'result.json')
z=np.load(a.run/'physical_states.npz');assert sha(a.run/'physical_states.npz')==e['capture_sha256']['physical_states.npz']
shots=json.loads(a.camera_plan.read_text())['shots'];active=json.loads(a.actors);assert len(shots)==len(active)
visual=SMPLMeshVisualizer.from_config(HumanMeshConfig(model_dir=str(R/'data/smpl'),num_envs=z['root_pos'].shape[1],device='cpu',shape_std=0.));mapper=SMPLRobotPoseMapper();rows=[]
with torch.no_grad():
 for i,shot in enumerate(shots):
  start=shot['start_step'];stop=shots[i+1]['start_step'] if i+1<len(shots) else 1800
  eye=torch.tensor(shot['eye'],dtype=torch.float32);f=torch.tensor(shot['target'],dtype=torch.float32)-eye;f/=f.norm();right=torch.linalg.cross(f,torch.tensor([0.,0.,1.]));right/=right.norm();up=torch.linalg.cross(right,f)
  ids=active[i];root=torch.from_numpy(z['root_pos'][start:stop,ids].reshape(-1,3));q=torch.from_numpy(z['root_rot'][start:stop,ids].reshape(-1,4));dof=torch.from_numpy(z['dof_pos'][start:stop,ids].reshape(-1,69));poses=mapper(dof,r['physics']['dof_names']);count_bad=0;front=float('inf');hmax=0.;vmax=0.;bad_source_steps=set();vh=13.5/shot.get('focal_mm',10);hh=18/shot.get('focal_mm',10)
  for k in range(0,len(root),32):
   n=len(root[k:k+32]);out=visual.smpl_model(betas=torch.zeros(n,10),global_orient=torch.zeros(n,3),body_pose=poses[k:k+n],transl=torch.zeros(n,3),return_verts=True)
   verts=(out.vertices-out.joints[:,0:1])@smpl_to_robot_matrix().T;world=quat_apply_wxyz(q[k:k+n][:,[3,0,1,2]][:,None,:],verts)+root[k:k+n,None];d=world-eye;depth=d@f;h=(d@right)/depth;v=(d@up)/depth;bad=(depth<=0)|(h.abs()>hh)|(v.abs()>vh);per=bad.any(-1);count_bad+=int(per.sum());front=min(front,float(depth.min()));hmax=max(hmax,float(h.abs().max()));vmax=max(vmax,float(v.abs().max()))
   bad_source_steps.update(start+(k+j)//len(ids) for j in per.nonzero().flatten().tolist())
  rows.append({'shot':i,'active_actors':ids,'source_interval':[start,stop],'actor_frames':len(root),'actor_frames_with_any_vertex_outside':count_bad,'outside_source_steps':sorted(bad_source_steps),'minimum_front_depth_m':front,'horizontal_abs_ratio_max':hmax,'horizontal_sensor_half_ratio':hh,'vertical_abs_ratio_max':vmax,'vertical_sensor_half_ratio':vh,'all_neutral_SMPL_vertices_inside':count_bad==0})
report={'schema':'crowdsim.recorded_camera_skin_projection.v1','source_result_sha256':sha(a.run/'result.json'),'source_states_sha256':sha(a.run/'physical_states.npz'),'camera_plan_sha256':sha(a.camera_plan),'evaluator_source_sha256':sha(__file__),'native_camera_aperture_mm':[36,27],'native_camera_source_sha256':sha(R/'scripts/replay_paper_pedestrians.py'),'per_shot':rows,'all_shots_inside':all(x['all_neutral_SMPL_vertices_inside'] for x in rows),'scope':'All actual neutral-SMPL skin vertices. Assigned visible actors only; no changes to actual rendering actor set. Separate native source-geometry occlusion inspection required.'};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows));raise SystemExit(0 if report['all_shots_inside'] else 1)
