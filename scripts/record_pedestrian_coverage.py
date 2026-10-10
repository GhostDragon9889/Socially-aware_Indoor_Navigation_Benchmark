"""Record 60 actual physical seconds through contiguous source-bound camera intervals.

RT2 replays the exact measured poses; cameras never change physical states.
https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/carb/carb.Framework.html
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
if not (R/'CrowdSim').is_dir():R=Path('/project/EXP/SangforAndSaut/ProtoMotions')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def launch(cmd,output):
 receipt=output.with_suffix('.owner.json');log=output.with_suffix('.owner.log')
 if receipt.exists() or log.exists():raise ValueError('Fresh job receipt/log required')
 rec={'command':list(map(str,cmd)),'actual_returncode':None,'actual_returncode_source':'subprocess.Popen.wait'};start=time.monotonic()
 with log.open('x') as stream:
  child=subprocess.Popen(rec['command'],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT);rec['pid']=child.pid;put(receipt,rec);code=child.wait()
 rec.update(actual_returncode=code,elapsed_s=time.monotonic()-start);put(receipt,rec)
 print(json.dumps({'job':output.name,'actual_returncode':code}),flush=True)
 if code:raise RuntimeError('Failed actual job preserved: '+str(output))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--posture',type=Path,required=True);p.add_argument('--camera-plan',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--speed',type=Path,required=True);p.add_argument('--spatial-stasis',type=Path,required=True);p.add_argument('--foot-surface',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--assemble-only',action='store_true');p.add_argument('--spatial-stasis10',type=Path);p.add_argument('--camera-skin',type=Path);p.add_argument('--substep-speed',type=Path);a=p.parse_args()
 if not a.tag.replace('_','').replace('-','').isalnum():p.error('Tag must be simple letters/digits')
 if not a.smoke and a.substep_speed is None:p.error('Full recording requires same-source120Hz speed-cap audit')
 if not a.smoke and a.camera_skin is None:p.error('Full recording requires same-source all-frame skin framing gate')
 if not a.smoke and a.spatial_stasis10 is None:p.error('Full recording requires the strengthened10s spatial stasis gate')
 if a.smoke and a.assemble_only:p.error('Assembly requires full mode')
 s=a.output_dir.resolve();s.mkdir(parents=True,exist_ok=True);run=a.run.resolve();q=a.quality.resolve();posture=json.loads(a.posture.read_text());source=sha(run/'result.json');states=sha(run/'physical_states.npz')
 speed=json.loads(a.speed.read_text());stasis=json.loads(a.spatial_stasis.read_text());foot=json.loads(a.foot_surface.read_text())
 if not(speed['speed_band_screen_passed'] and speed['result_sha256']==source and speed['physical_sha256']==sha(run/'physical.npz') and stasis['zero_spatial_stasis_over20s'] and stasis['result_sha256']==source and stasis['physical_sha256']==sha(run/'physical.npz') and foot['source_result_sha256']==source and foot['physical_states_sha256']==states and foot['frames']==1800 and foot['source_control_interval']==[0,1800] and foot['actor_frame_fraction_foot_vertices_below_minus1cm']==0):raise ValueError('Same-source full speed, spatial stasis and physical foot-surface gates required')
 if a.substep_speed is not None:
  sub=json.loads(a.substep_speed.read_text())
  if not(sub['passed_upper1p5_all_substeps'] and sub['actual_physics_substeps_per_actor']==7200 and sub['upper_speed_m_s']==1.5 and sub['source_result_sha256']==source and sub['ground_support_sha256']==sha(run/'ground_support.npz') and sub['physical_sha256']==sha(run/'physical.npz')):raise ValueError('Same-source all7200substep speed cap required')
 if a.spatial_stasis10 is not None:
  ten=json.loads(a.spatial_stasis10.read_text())
  if not(ten['maximum_allowed_spatial_stasis_s']==10 and ten['zero_spatial_stasis_over10s'] and ten['result_sha256']==source and ten['physical_sha256']==sha(run/'physical.npz')):raise ValueError('Same-source strengthened10s route-progress gate required')
 if not posture['local_walking_screen']['passed'] or posture['result_sha256']!=source:raise ValueError('Same-source local walking screen required')
 if a.camera_skin is not None:
  framing=json.loads(a.camera_skin.read_text())
  if not(framing['all_shots_inside'] and framing['source_result_sha256']==source and framing['source_states_sha256']==states and framing['camera_plan_sha256']==sha(a.camera_plan) and framing['native_camera_source_sha256']==sha(R/'scripts/replay_paper_pedestrians.py')):raise ValueError('Exact native source/scene camera plan and all-frame skin projection required')
 plan=json.loads(a.camera_plan.read_text());shots=plan['shots']
 starts=[x['start_step'] for x in shots]
 if not starts or starts[0]!=0 or any(type(v) is not int or not 0<=v<1800 for v in starts) or any(b<=a for a,b in zip(starts,starts[1:])):raise ValueError('Ordered continuous source camera intervals starting0 and ending1800 required')
 intervals=list(zip(starts,starts[1:]+[1800]))
 fk=s/(a.tag+'_fk.json')
 if not a.assemble_only:launch([sys.executable,R/'scripts/audit_trace_pace_visual_fk.py','--run',run,'--output',fk,'--start-step','0','--steps','1800'],s/(a.tag+'_fk'))
 f=json.loads(fk.read_text());fk_owner=json.loads(fk.with_name(fk.stem+'.owner.json').read_text())
 if fk_owner['actual_returncode']!=0 or fk_owner['actual_returncode_source']!='subprocess.Popen.wait':raise ValueError('Every-frame CPU FK audit must finish normally')
 if not(f['passed_under1mm'] and f['frames']==1800 and f['source_control_interval']==[0,1800] and f['source_result_sha256']==source and f['physical_states_sha256']==states):raise ValueError('Every requested frame must match actual physical FK')
 receipts=[];clips=[]
 for i,shot in enumerate(shots):
  output=s/(a.tag+('_smoke' if a.smoke else '_shot')+str(i));n=min(30,intervals[i][1]-intervals[i][0]) if a.smoke else intervals[i][1]-intervals[i][0]
  if not a.assemble_only:
   cmd=[sys.executable,R/'scripts/record_trace_pace_verified.py','--run',run,'--quality',q,'--output',output,'--start-step',str(shot['start_step']),'--steps',str(n),'--capture-every-steps','1','--native-scene-visuals','--camera-eye',*map(str,shot['eye']),'--camera-target',*map(str,shot['target']),'--camera-focal-length-mm',str(shot.get('focal_mm',10))]
   if shot.get('follow_actor') is not None:cmd.extend(['--follow-actor',str(shot['follow_actor']),'--follow-offset',*map(str,shot['follow_offset'])])
   if shot.get('floor_hidden',False):cmd.append('--native-reference-floor-hidden')
   if shot.get('exposure_ev') is not None:cmd.extend(['--camera-exposure-ev',str(shot['exposure_ev'])])
   launch(cmd,output)
  epath=output.with_name(output.name+'_execution.json');e=json.loads(epath.read_text());result=output/'result.json';r=json.loads(result.read_text());video=output/'pedestrian_motion.mp4'
  if not(e['actual_returncode']==0 and e['sources_unchanged'] and e['result_sha256']==sha(result) and e['video_sha256']==sha(video) and e['physical_result_sha256']==source and e['physical_states_sha256']==states and e['quality_sha256']==sha(q) and r['status']=='passed' and r['frames']==n):raise ValueError('Camera child must finish normally on exact admitted source')
  if [x['source_step_index'] for x in r['frame_records']]!=list(range(shot['start_step'],shot['start_step']+n)):raise ValueError('Source pose/time was interpolated or skipped')
  if r['native_visual_stage']['visual_reference_floor_visible']==shot.get('floor_hidden',False):raise ValueError('Auxiliary floor display differs from explicit plan')
  if r['render_camera'].get('exposure_ev')!=shot.get('exposure_ev'):raise ValueError('Actual camera exposure differs from declared plan')
  receipts.append({'path':str(epath),'sha256':sha(epath)});clips.append(video)
 if a.smoke:return 0
 concat=s/(a.tag+'_concat.txt');concat.write_text(''.join("file '"+str(v).replace("'","'\\''")+"'\n" for v in clips));video=s/(a.tag+'_60s.mp4')
 launch(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',concat,'-c','copy','-movflags','+faststart',video],s/(a.tag+'_concat'))
 cmd=['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(video)];meta=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout);stream=next(x for x in meta['streams'] if x['codec_name']=='h264')
 if int(stream['nb_read_frames'])!=1800 or stream['r_frame_rate']!='30/1' or abs(float(meta['format']['duration'])-60)>.05:raise ValueError('60 seconds/1800 original30Hz frames required')
 launch(['ffmpeg','-nostdin','-v','error','-i',video,'-f','null','-'],s/(a.tag+'_decode'))
 put(s/(a.tag+'_video_receipt.json'),{'schema':'crowdsim.actual_native_60s_video.v1','status':'passed','video':str(video),'sha256':sha(video),'ffprobe':meta,'source_result_sha256':source,'source_states_sha256':states,'camera_receipts':receipts,'camera_plan':str(a.camera_plan.resolve()),'camera_plan_sha256':sha(a.camera_plan),'camera_skin_sha256':sha(a.camera_skin),'spatial_stasis10_sha256':sha(a.spatial_stasis10),'full_MP4_decode_owner':str(s/(a.tag+'_decode.owner.json')),'speed_sha256':sha(a.speed),'substep_speed_sha256':sha(a.substep_speed),'spatial_stasis_sha256':sha(a.spatial_stasis),'foot_surface_sha256':sha(a.foot_surface),'posture_sha256':sha(a.posture),'quality_sha256':sha(q),'visual_fk_sha256':sha(fk),'recorder_source_sha256':sha(Path(__file__)),'source_control_intervals':[list(x) for x in intervals],'source_physics_interval_s':[1/30,60.],'actual_physical_state_render':True,'cuts_only_no_pose_interpolation':True,'native_frozen_original_USD':True,'fresh_physics':False,'author_protocol':False})
 print(json.dumps({'status':'passed','video':str(video)}),flush=True);return 0
if __name__=='__main__':sys.exit(main())
