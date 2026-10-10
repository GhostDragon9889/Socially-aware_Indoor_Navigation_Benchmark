from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,torch
R=Path(__file__).resolve().parents[1];S=Path(__file__).parent;sys.path.insert(0,str(R));torch.set_num_threads(4)
from CrowdSim.tools.smpl_mesh_visualizer import HumanMeshConfig,SMPLMeshVisualizer,SMPLRobotPoseMapper,SMPL_JOINT_NAMES,smpl_to_robot_matrix,quat_apply_wxyz
parser=argparse.ArgumentParser(description='Verify every rendered body joint against its actual physical pose')
parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--start-step',type=int,default=0);parser.add_argument('--steps',type=int,default=900)
args=parser.parse_args();p=args.run;r=json.loads((p/'result.json').read_text());execution=json.loads(p.with_name(p.name+'_execution.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if execution['actual_returncode']!=0 or execution['actual_returncode_source']!='subprocess.Popen.wait' or execution['result_sha256']!=sha(p/'result.json') or execution['capture_sha256']['physical_states.npz']!=sha(p/'physical_states.npz'):raise ValueError('Completed actual source identity required')
z=np.load(p/'physical_states.npz');indices=np.arange(args.start_step,args.start_step+args.steps)
if args.start_step<0 or args.steps<1 or indices[-1]>=len(z['root_pos']):raise ValueError('Requested interval is outside recorded physical controls')
dof=torch.from_numpy(z['dof_pos'][indices].reshape(-1,69));q=torch.from_numpy(z['root_rot'][indices].reshape(-1,4));root=torch.from_numpy(z['root_pos'][indices].reshape(-1,3));actual=torch.from_numpy(z['body_pos'][indices].reshape(-1,24,3));visual=SMPLMeshVisualizer.from_config(HumanMeshConfig(model_dir=str(R/'data/smpl'),num_envs=z['root_pos'].shape[1],device='cpu',shape_std=0.));mapper=SMPLRobotPoseMapper();poses=mapper(dof,r['physics']['dof_names']);errors=[]
order=[SMPL_JOINT_NAMES.index(n) for n in r['physics']['body_names']]
with torch.no_grad():
 for i in range(0,len(root),32):
  count=len(root[i:i+32]);o=visual.smpl_model(betas=torch.zeros(count,10),global_orient=torch.zeros(count,3),body_pose=poses[i:i+count],transl=torch.zeros(count,3),return_verts=False)
  joints=(o.joints[:,:24]-o.joints[:,0:1])@smpl_to_robot_matrix().T
  world=quat_apply_wxyz(q[i:i+count][:,[3,0,1,2]][:,None,:],joints[:,order])+root[i:i+count,None]
  errors.append((world-actual[i:i+count]).norm(dim=-1))
e=torch.cat(errors).numpy();report={'schema':'crowdsim.every_video_frame_FK_audit.v1','scope':'every requested30Hz source frame, all actors, all24 semantic body joints','source_control_interval':[args.start_step,args.start_step+args.steps],'frames':len(indices),'actors':z['root_pos'].shape[1],'body_joints_mean_error_m':float(e.mean()),'body_joints_p95_error_m':float(np.quantile(e,.95)),'body_joints_max_error_m':float(e.max()),'by_body_mean_m':dict(zip(r['physics']['body_names'],map(float,e.mean(0)))),'passed_under1mm':bool(e.max()<.001),'source_result_sha256':execution['result_sha256'],'physical_states_sha256':execution['capture_sha256']['physical_states.npz'],'evaluator_source_sha256':sha(Path(__file__))}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['frames','actors','body_joints_max_error_m','passed_under1mm']}))
if not report['passed_under1mm']:sys.exit(1)
