"""Render an admitted actual AMP scene archive with an ordinary child wait."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--quality', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--start-step', type=int, default=0)
parser.add_argument('--steps', type=int, default=1800)
parser.add_argument('--capture-every-steps', type=int, choices=(1,2), default=2)
parser.add_argument('--camera-eye', type=float, nargs=3, default=[29, -8, 21])
parser.add_argument('--camera-target', type=float, nargs=3, default=[12, 12, .9])
parser.add_argument('--camera-focal-length-mm',type=float,default=20.)
parser.add_argument('--camera-exposure-ev',type=float,default=None)
parser.add_argument("--native-scene-visuals", action="store_true")
parser.add_argument("--native-reference-floor-hidden",action="store_true")
parser.add_argument('--follow-actor',type=int)
parser.add_argument('--follow-offset',type=float,nargs=3,default=[0.,3.,6.])
args = parser.parse_args()
if args.native_reference_floor_hidden and not args.native_scene_visuals:parser.error("Hidden auxiliary floor requires native recorded visuals")
args.output.parent.mkdir(parents=True, exist_ok=True)
physics_receipt = json.loads(args.run.with_name(args.run.name+'_execution.json').read_text())
source_result = args.run/'result.json'
source = json.loads(source_result.read_text())
quality = json.loads(args.quality.read_text())
states = args.run/'physical_states.npz'
if (physics_receipt['actual_returncode'] != 0 or physics_receipt['sources_unchanged'] is not True
        or physics_receipt['result_sha256'] != digest(source_result)
        or physics_receipt['capture_sha256']['physical_states.npz'] != digest(states)
        or source['status'] != 'passed' or source['weights_finetuned_on_local_2848'] is not True
        or quality['quality_status'] != 'passed_local_proxy_gates'
        or not all(quality['checks'].values())
        or quality['result_sha256'] != digest(source_result)):
    raise ValueError('Video requires a hash-bound, actually fine tuned, physically admitted scene run')
receipt = args.output.with_name(args.output.name+'_execution.json')
log = args.output.with_name(args.output.name+'.log')
if any(p.exists() for p in (args.output, receipt, log)):
    parser.error('Fresh video output/receipt/log required')
sources = [ROOT/'scripts/replay_paper_pedestrians.py', ROOT/'CrowdSim/humanoid/simulate.py',
           ROOT/'CrowdSim/tools/smpl_mesh_visualizer.py',
           ROOT/'configs/diffusion_amp/trace_pace_recording.kit', Path(__file__).resolve()]
if args.native_scene_visuals:
    sources.extend(ROOT/name for name in ('CrowdSim/humanoid/runtime/native_visual_scene.py',
        'CrowdSim/humanoid/runtime/dynbench_scene.py','CrowdSim/humanoid/runtime/native_grid_scene.py'))
if args.native_scene_visuals:
    sources.extend((ROOT/'CrowdSim/humanoid/runtime').glob('*.py'))
    sources.append(ROOT/'CrowdSim/utils/map_metadata.py')
sources = sorted(set(sources))
hashes = {str(p): digest(p) for p in sources}
command = [sys.executable, str(ROOT/'scripts/replay_paper_pedestrians.py'), '--states', str(states),
    '--source-result', str(source_result), '--output', str(args.output), '--start-step', str(args.start_step),
    '--steps', str(args.steps), '--capture-every-steps', str(args.capture_every_steps), '--width', '960', '--height', '720',
    '--camera-focal-length-mm',str(args.camera_focal_length_mm),
    '--camera-eye', *map(str, args.camera_eye), '--camera-target', *map(str, args.camera_target)]
if args.camera_exposure_ev is not None:command.extend(['--camera-exposure-ev',str(args.camera_exposure_ev)])
if args.native_scene_visuals:
    command.append('--native-scene-visuals')
    if args.native_reference_floor_hidden:command.append('--native-reference-floor-hidden')
if args.follow_actor is not None:
    command.extend(['--follow-actor',str(args.follow_actor),'--follow-offset',*map(str,args.follow_offset)])
record = {'schema': 'crowdsim.actual_amass_video_execution.v1', 'status': 'running',
          'command': command, 'source_sha256': hashes, 'physical_result_sha256': digest(source_result),
          'physical_states_sha256': digest(states), 'quality_sha256': digest(args.quality),
          'amp_checkpoint_sha256': source['resource_sha256']['amp_checkpoint'],
          'mode': 'faithful_render_of_recorded_actual_physical_states', 'fresh_physics_run': False,
          'actual_returncode': None, 'actual_returncode_source': 'subprocess.Popen.wait'}
import shutil
snapshot = args.output.with_name(args.output.name+'_sources')
snapshot.mkdir(exist_ok=False)
layout = {}
for path in sources:
    relative = path.relative_to(ROOT)
    destination = snapshot/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination)
    if digest(destination) != hashes[str(path)]: raise ValueError('Render source changed while freezing')
    layout[str(path)] = str(relative)
record['source_snapshot_root'] = str(snapshot)
record['source_snapshot_layout'] = layout
receipt.write_text(json.dumps(record, indent=2)+'\n')
started = time.monotonic()
with log.open('w') as stream:
    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1'),
                               stdout=stream, stderr=subprocess.STDOUT)
    record['child_pid'] = process.pid
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    code = process.wait()
record.update(actual_returncode=code, elapsed_wall_s=time.monotonic()-started,
              sources_unchanged=all(digest(Path(p)) == sha for p, sha in hashes.items()),
              status='completed' if code == 0 else 'failed_preserved')
result_path = args.output/'result.json'
if result_path.is_file():
    result = json.loads(result_path.read_text())
    record['result_sha256'] = digest(result_path)
    record['render_result'] = {k: result.get(k) for k in ('status', 'frames', 'video', 'error',
        'simulation_time_advanced_s', 'physics_steps', 'source_interval_s', 'video_duration_at_declared_fps_s')}
    if result.get('video'):
        record['video_sha256'] = digest(Path(result['video']))
        record['video_bytes'] = Path(result['video']).stat().st_size
receipt.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: record.get(k) for k in ('status', 'actual_returncode', 'sources_unchanged', 'elapsed_wall_s', 'render_result')}))
sys.exit(0 if code == 0 and record['sources_unchanged'] and record.get('render_result', {}).get('status') == 'passed' else 1)
