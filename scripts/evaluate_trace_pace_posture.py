from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.ndimage import uniform_filter1d
R=Path(__file__).resolve().parents[1];S=Path(__file__).parent
sys.path.insert(0,str(R))
from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES

def stats(x):
 x=np.asarray(x,dtype=float).reshape(-1);return {'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'rms':float(np.sqrt(np.mean(x*x))),'max':float(x.max())}
def measure(body,q,joints,vel,names):
 yaw=np.arctan2(2*(q[...,3]*q[...,2]+q[...,0]*q[...,1]),1-2*(q[...,1]**2+q[...,2]**2))
 wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
 line=body[...,names.index('L_Shoulder'),:]-body[...,names.index('R_Shoulder'),:]
 twist=wrap(np.arctan2(line[...,1],line[...,0])-yaw-np.pi/2)
 axial_twist=abs(twist)*180/np.pi
 raw_yaw=np.unwrap(yaw.astype(float),axis=0)
 yaw_oscillation=raw_yaw-uniform_filter1d(raw_yaw,size=31,axis=0,mode='nearest')
 side=[];sagittal=[];elbow=[]
 for s in ['L','R']:
  shoulder=body[...,names.index(s+'_Shoulder'),:];wrist=body[...,names.index(s+'_Wrist'),:];e=body[...,names.index(s+'_Elbow'),:]
  d=wrist-shoulder;forward=yaw.cos() if hasattr(yaw,'cos') else np.cos(yaw)
  side.append(abs(-np.sin(yaw)*d[...,0]+forward*d[...,1]))
  sagittal.append(np.arctan2(forward*d[...,0]+np.sin(yaw)*d[...,1],-d[...,2])*180/np.pi)
  a=shoulder-e;b=wrist-e;elbow.append(180-np.arccos(np.clip((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)),-1,1))*180/np.pi)
 trunk=body[...,names.index('Head'),:]-body[...,names.index('Pelvis'),:]
 lateral=(-np.sin(yaw)*trunk[...,0]+np.cos(yaw)*trunk[...,1])/np.linalg.norm(trunk,axis=-1)
 return {'pelvis_yaw_1s_detrended_abs_deg':stats(abs(yaw_oscillation)*180/np.pi),'shoulder_line_relative_pelvis_yaw_abs_deg':stats(axial_twist),'head_pelvis_lateral_tilt_abs_deg':stats(abs(np.arcsin(np.clip(lateral,-1,1)))*180/np.pi),'wrist_lateral_offset_from_shoulder_m':stats(side),'arm_sagittal_angle_abs_deg':stats(abs(np.asarray(sagittal))),'elbow_flexion_deg':stats(elbow),'root_speed_m_s':stats(np.linalg.norm(vel[...,:2],axis=-1))}
def actual(path):
 r=json.loads((path/'result.json').read_text());e=json.loads(path.with_name(path.name+'_execution.json').read_text());assert r['status']=='passed' and r['cleanup_succeeded'] and e['result_sha256']==hashlib.sha256((path/'result.json').read_bytes()).hexdigest() and e['actual_returncode_source']=='subprocess.Popen.wait' and e['sources_unchanged'] is True and e['actual_returncode']==0 and e['capture_sha256']['physical.npz']==hashlib.sha256((path/'physical.npz').read_bytes()).hexdigest()
 with np.load(path/'physical.npz') as z:
  body=z['body_positions'];q=z['root_rotations_xyzw'];joints=z['joint_positions'];velocity=z['samples'][...,2:4];names=r['physics']['body_names']
  values=measure(body,q,joints,velocity,names)
  moving=np.linalg.norm(velocity,axis=-1)>=.2
  moving_values=measure(body[moving],q[moving],joints[moving],velocity[moving],names) if moving.any() else None
  if moving_values is not None:
   # Concatenated moving frames do not preserve time. Use the all-frame yaw filter.
   moving_values.pop('pelvis_yaw_1s_detrended_abs_deg')
  per_actor=[]
  for actor in range(body.shape[1]):
   selection=moving[:,actor]
   if not selection.any():
    per_actor.append({'actor':actor,'moving_frames':0,'moving_fraction':0.,'metrics':None});continue
   b=body[selection,actor];q_actor=q[selection,actor]
   metrics=measure(b,q_actor,joints[selection,actor],velocity[selection,actor],names)
   metrics.pop('pelvis_yaw_1s_detrended_abs_deg')
   yaw=np.arctan2(2*(q_actor[:,3]*q_actor[:,2]+q_actor[:,0]*q_actor[:,1]),1-2*(q_actor[:,1]**2+q_actor[:,2]**2))
   arm_details={}
   for side in ('L','R'):
    delta=b[:,names.index(side+'_Wrist')]-b[:,names.index(side+'_Shoulder')]
    forward=np.cos(yaw)*delta[:,0]+np.sin(yaw)*delta[:,1]
    angle=np.arctan2(forward,-delta[:,2])*180/np.pi
    arm_details[side]={'sagittal_abs_deg':stats(abs(angle)),
                      'above_horizontal_fraction':float((abs(angle)>90).mean()),
                      'forward_offset_std_m':float(forward.std())}
   per_actor.append({'actor':actor,'moving_frames':int(selection.sum()),'moving_fraction':float(selection.mean()),'metrics':metrics,'arms':arm_details})
  # Declared local engineering screen, not an author metric or human rating.
  # Set before the YOHO v22 physical test completed. It rejects the raised-arm
  # v19 candidate while preserving moderate natural arm swing and route turns.
  thresholds={'minimum_moving_fraction':.2,'arm_sagittal_p95_max_deg':45.,
              'above_horizontal_fraction_max':.01,'minimum_arm_forward_std_m':.01,
              'head_pelvis_lateral_tilt_p95_max_deg':12.,'shoulder_relative_yaw_p95_max_deg':25.}
  checks={}
  for row in per_actor:
   a=row['actor'];metrics=row['metrics']
   checks[f'actor{a}_moving_fraction']=row['moving_fraction']>=thresholds['minimum_moving_fraction']
   checks[f'actor{a}_lateral_tilt']=metrics is not None and metrics['head_pelvis_lateral_tilt_abs_deg']['p95']<=thresholds['head_pelvis_lateral_tilt_p95_max_deg']
   checks[f'actor{a}_shoulder_twist']=metrics is not None and metrics['shoulder_line_relative_pelvis_yaw_abs_deg']['p95']<=thresholds['shoulder_relative_yaw_p95_max_deg']
   for side in ('L','R'):
    arm=row.get('arms',{}).get(side)
    checks[f'actor{a}_{side}_arm']=arm is not None and arm['sagittal_abs_deg']['p95']<=thresholds['arm_sagittal_p95_max_deg'] and arm['above_horizontal_fraction']<=thresholds['above_horizontal_fraction_max'] and arm['forward_offset_std_m']>=thresholds['minimum_arm_forward_std_m']
  return {'schema':'crowdsim.actual_posture_diagnostics.v2','result_sha256':e['result_sha256'],'physical_sha256':e['capture_sha256']['physical.npz'],'actual_controls':len(body),'actors':body.shape[1],'all_frames':values,'moving_threshold_m_s':.2,'moving_actor_frames':int(moving.sum()),'total_actor_frames':int(moving.size),'moving_frames':moving_values,'per_actor':per_actor,'local_walking_screen':{'thresholds':thresholds,'checks':checks,'passed':all(checks.values()),'author_benchmark':False},'limits':['Geometric angles and body-origin offsets are descriptive proxies; there is no author posture benchmark.','The one-second pelvis yaw detrending includes route turns and is not an isolated gait oscillation estimate.','Moving-frame selection is a declared secondary diagnostic; all-frame reliability remains mandatory.','The local walking screen is a bounded posture filter, not evidence of perceptual realism or dense-crowd generalization.']}


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description='Hash-bound physical posture and speed diagnostics')
 p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--require-walking-screen',action='store_true')
 args=p.parse_args();result=actual(args.run);args.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'output':str(args.output),'moving_actor_frames':result['moving_actor_frames'],'total_actor_frames':result['total_actor_frames']}))
 if args.require_walking_screen and not result['local_walking_screen']['passed']:sys.exit(1)
