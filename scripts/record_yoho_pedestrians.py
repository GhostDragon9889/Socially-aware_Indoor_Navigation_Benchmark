"""Two actual-state indoor camera intervals, ordinary SDK exits and one 30Hz video."""
from pathlib import Path
import json,os,subprocess,hashlib,time,sys,argparse,re
R=Path(__file__).resolve().parents[1];S=None
sys.path.insert(0,str(R/'src'))
from social_navigation.external_runtime import selected_python
P=str(selected_python(R))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def launch(command,name):
 log=S/(name+'.owner.log');receipt=S/(name+'.owner.json');start=time.monotonic()
 with log.open('x') as f:
  p=subprocess.Popen(command,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
  receipt.write_text(json.dumps({'command':command,'pid':p.pid,'actual_returncode':None},indent=2)+'\n');code=p.wait()
 receipt.write_text(json.dumps({'command':command,'pid':p.pid,'actual_returncode':code,'actual_returncode_source':'subprocess.Popen.wait','elapsed_s':time.monotonic()-start},indent=2)+'\n')
 print(json.dumps({'job':name,'actual_returncode':code}),flush=True)
 if code:raise RuntimeError('Failed job preserved: '+name)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',type=Path,required=True)
 parser.add_argument('--quality',type=Path,required=True)
 parser.add_argument('--posture',type=Path,required=True)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--tag',default='v1')
 parser.add_argument('--start-step',type=int,default=120,help='First actual30Hz state; full assessments still include initialization')
 parser.add_argument('--smoke',action='store_true')
 parser.add_argument('--female-clip-name')
 parser.add_argument('--male-clip-name')
 parser.add_argument('--assemble-only',action='store_true',help='Resume from completed hash-bound450-frame camera executions without rendering again')
 args=parser.parse_args()
 if args.start_step<0:parser.error('start-step must be nonnegative')
 if args.assemble_only and args.smoke:parser.error('assemble-only requires the full900-frame mode')
 if re.fullmatch(r'[A-Za-z0-9_-]{1,40}',args.tag) is None:parser.error('tag requires1..40 letters, digits, underscores or hyphens')
 for role in ('female','male'):
  name=getattr(args,role+'_clip_name')
  if name is not None and (not args.assemble_only or re.fullmatch('yoho_'+role+'_15s_[A-Za-z0-9_-]{1,40}',name) is None):parser.error('Named clips require assembly mode and the exact camera role prefix')
 S=args.output_dir.resolve();S.mkdir(parents=True,exist_ok=True)
 run=args.run.resolve();quality=args.quality.resolve();mode='smoke' if args.smoke else 'all'
 posture_path=args.posture.resolve();posture=json.loads(posture_path.read_text())
 if not posture['local_walking_screen']['passed'] or posture['result_sha256']!=digest(run/'result.json'):raise ValueError('Local walking screen and physical identity required')
 if args.assemble_only:
  fk=json.loads((S/('yoho_visual_fk_'+args.tag+'.json')).read_text())
  owner=json.loads((S/('yoho_visual_fk_'+args.tag+'.owner.json')).read_text())
  if owner['actual_returncode']!=0 or not fk['passed_under1mm'] or fk['frames']!=900 or fk['source_control_interval']!=[args.start_step,args.start_step+900] or fk['source_result_sha256']!=digest(run/'result.json') or fk['physical_states_sha256']!=digest(run/'physical_states.npz'):raise ValueError('Completed900-frame FK and source identity required for assembly')
 else:
  # Check every requested source frame before rendering either camera.
  launch([P,str(R/'scripts/audit_trace_pace_visual_fk.py'),'--run',str(run),
          '--output',str(S/('yoho_visual_fk_'+args.tag+'.json')),
          '--start-step',str(args.start_step),'--steps','900'],'yoho_visual_fk_'+args.tag)
 jobs=[(args.female_clip_name or 'yoho_female_15s_'+args.tag,args.start_step,[4.4,20.1,2.8],[4.4,17.8,.9],['--native-reference-floor-hidden','--camera-focal-length-mm','16']),
       (args.male_clip_name or 'yoho_male_15s_'+args.tag,args.start_step+450,[0,0,0],[0,0,1],['--follow-actor','2','--follow-offset','-2','0','1.3'])]
 if not args.assemble_only:
  for name,start,eye,target,extra in jobs:
   if mode not in ['all','smoke',name]:continue
   actual_name=name.replace('15s','smoke') if mode=='smoke' else name
   launch([P,str(R/'scripts/record_trace_pace_verified.py'),'--run',str(run),'--quality',str(quality),'--output',str(S/actual_name),'--start-step',str(start),'--steps','30' if mode=='smoke' else '450','--capture-every-steps','1','--native-scene-visuals','--camera-eye',*map(str,eye),'--camera-target',*map(str,target),*extra],actual_name)
 if mode!='all':sys.exit(0)
 clips=[];receipts=[]
 for name,_,_,_,_ in jobs:
  e=json.loads((S/(name+'_execution.json')).read_text());r=json.loads((S/name/'result.json').read_text());v=S/name/'pedestrian_motion.mp4'
  if e['actual_returncode']!=0 or not e['sources_unchanged'] or r['status']!='passed' or r['frames']!=450 or e['video_sha256']!=digest(v):raise ValueError('Every actual render must finish normally')
  if e['physical_result_sha256']!=digest(run/'result.json') or e['physical_states_sha256']!=digest(run/'physical_states.npz') or e['quality_sha256']!=digest(quality) or e['result_sha256']!=digest(S/name/'result.json'):raise ValueError('Every camera must render the same admitted physical source')
  expected_start=args.start_step+(450 if name.startswith('yoho_male_') else 0)
  if [row['source_step_index'] for row in r['frame_records']]!=list(range(expected_start,expected_start+450)):raise ValueError('Continuous exact source camera intervals required')
  if r['native_visual_stage']['visual_reference_floor_visible']!=name.startswith('yoho_male_'):raise ValueError('Recorded auxiliary-floor visibility must match the declared camera')
  clips.append(v);receipts.append({'path':str(S/(name+'_execution.json')),'sha256':digest(S/(name+'_execution.json'))})
 concat=S/('yoho_concat_'+args.tag+'.txt');concat.write_text(''.join("file '"+str(v).replace("'","'\\''")+"'\n" for v in clips))
 final=S/('yoho_diffusion_amp_30s_'+args.tag+'.mp4')
 launch(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(final)],'yoho_concat_'+args.tag)
 probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(final)],capture_output=True,text=True,check=True)
 meta=json.loads(probe.stdout);video=next(x for x in meta['streams'] if x['codec_name']=='h264')
 if int(video['nb_read_frames'])!=900 or video['r_frame_rate']!='30/1' or abs(float(meta['format']['duration'])-30)>.05:raise ValueError('900 actual30Hz frames required')
 out={'schema':'crowdsim.actual_yoho_composite_video.v1','status':'passed','video':str(final),'sha256':digest(final),'ffprobe':meta,'camera_execution_receipts':receipts,'source_result_sha256':digest(run/'result.json'),'source_states_sha256':digest(run/'physical_states.npz'),'recorder_source_sha256':digest(Path(__file__)),'local_walking_screen_sha256':digest(posture_path),'visual_fk_audit_sha256':digest(S/('yoho_visual_fk_'+args.tag+'.json')),'source_control_intervals':[[args.start_step,args.start_step+450],[args.start_step+450,args.start_step+900]],'source_physics_interval_s':[(args.start_step+1)/30,(args.start_step+900)/30],'assembly_only':args.assemble_only,'cuts_only_no_pose_interpolation':True,'camera_follow_uses_same_actual_root_coordinates':True,'fresh_physics':False,'actual_physical_state_render':True,'native_frozen_original_USD':True,'analytical_floor_visibility_by_camera':{'female':False,'male':True},'floor_display':'female original tiles; male admitted analytical floor for visible contact shadows','original_floor_and_physical_pose_unchanged':True,'limits':'Four neutral SMPL bodies in one YOHO scene, two camera shots; First source-control omission is explicitly recorded; full physics/posture reports include initialization. Not the author eight-scene protocol.'}
 (S/('yoho_video_receipt_'+args.tag+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'passed','video':str(final)}),flush=True)
