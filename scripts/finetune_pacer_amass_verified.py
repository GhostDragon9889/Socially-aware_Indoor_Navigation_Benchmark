"""Keep real child wait/returncode and exact training source hashes."""
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
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--updates', type=int, default=512)
parser.add_argument('--num-envs', type=int, default=128)
parser.add_argument("--optimization-config", type=Path)
parser.add_argument("--walking-turn-windows",type=Path)
parser.add_argument("--gait-prior", type=Path)
parser.add_argument("--temporal-gait-prior", type=Path)
parser.add_argument('--parent-checkpoint', type=Path)
parser.add_argument("--task-horizons-s",type=float,nargs="+",default=[4.5])
args = parser.parse_args()
import sys
sys.path.insert(0,str(ROOT))
from CrowdSim.humanoid.training.task_horizon import horizon_frame_offsets
horizon_frame_offsets(args.num_envs,args.task_horizons_s)
receipt = args.output.with_name(args.output.name + '_execution.json')
log = args.output.with_name(args.output.name + '.log')
if any(p.exists() for p in (args.output, receipt, log)):
    parser.error('Fresh output/receipt/log required')
sources = {ROOT/'scripts/finetune_pacer_amass.py', Path(__file__).resolve()}
for directory in ('CrowdSim/humanoid/training', 'CrowdSim/humanoid/amp', 'CrowdSim/humanoid/application'):
    sources.update((ROOT/directory).rglob('*.py'))
for name in ('CrowdSim/humanoid/amp_world.py', 'CrowdSim/humanoid/amass_corpus.py',
             'CrowdSim/humanoid/runtime/trace_pace_lifecycle.py',
             'CrowdSim/humanoid/runtime/trace_pace_scope.py',
             'CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py',
             'protomotions/simulator/isaaclab/simulator.py', 'protomotions/robot_configs/smpl.py'):
    sources.add(ROOT/name)
before = {str(p): digest(p) for p in sorted(sources)}
command = [sys.executable, str(ROOT/'scripts/finetune_pacer_amass.py'), '--output', str(args.output),
           '--updates', str(args.updates), '--num-envs', str(args.num_envs)]
command.extend(['--task-horizons-s',*map(str,args.task_horizons_s)])
if args.optimization_config is not None:
    sources.add(args.optimization_config.resolve())
    before = {str(p): digest(p) for p in sorted(sources)}
    command.extend(['--optimization-config', str(args.optimization_config.resolve())])
if args.walking_turn_windows is not None:
    sources.add(args.walking_turn_windows.resolve())
    before={str(p):digest(p) for p in sorted(sources)}
    command.extend(['--walking-turn-windows',str(args.walking_turn_windows.resolve())])
if args.gait_prior is not None:
    sources.add(args.gait_prior.resolve())
    before={str(p):digest(p) for p in sorted(sources)}
    command.extend(['--gait-prior',str(args.gait_prior.resolve())])
if args.temporal_gait_prior is not None:
    sources.add(args.temporal_gait_prior.resolve())
    before={str(p):digest(p) for p in sorted(sources)}
    command.extend(['--temporal-gait-prior',str(args.temporal_gait_prior.resolve())])
if args.parent_checkpoint is not None:
    sources.add(args.parent_checkpoint.resolve())
    before = {str(p): digest(p) for p in sorted(sources)}
    command.extend(['--parent-checkpoint', str(args.parent_checkpoint.resolve())])
record = {'schema': 'crowdsim.actual_finetune_execution.v1', 'command': command,
          'source_sha256': before, 'actual_returncode': None, 'status': 'running'}
args.output.parent.mkdir(parents=True, exist_ok=True)
snapshot = args.output.with_name(args.output.name+'_sources')
snapshot.mkdir(exist_ok=False)
import shutil
layout = {}
for p in sorted(sources):
    relative = p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('recipe')/p.name
    dest = snapshot/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
    if digest(dest) != before[str(p)]: raise ValueError('Training source changed while freezing')
    layout[str(p)] = str(relative)
record['source_snapshot_root'] = str(snapshot)
record['source_snapshot_layout'] = layout
receipt.write_text(json.dumps(record, indent=2)+'\n')
manifest_path=ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
manifest=json.loads(manifest_path.read_text());entry=manifest['splits']['train']
bank=Path(entry['path']);bank=bank if bank.is_absolute() else manifest_path.parent/bank
def bank_fingerprint():
    import hashlib
    h=hashlib.sha256()
    with bank.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    st=bank.stat()
    return {'path':str(bank),'actual_sha256':h.hexdigest(),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'inode':st.st_ino,'manifest_sha256':digest(manifest_path)}
bank_before=bank_fingerprint()
if bank_before['actual_sha256']!=entry['sha256'] or bank_before['bytes']!=entry['bytes']:raise ValueError('Actual fixed training bank SHA/bytes must match the manifest before GPU launch')
record['actual_training_inputs_before']=bank_before
started = time.monotonic()
with log.open('w') as stream:
    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream, stderr=subprocess.STDOUT)
    record['child_pid'] = process.pid
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    code = process.wait()
bank_after=bank_fingerprint()
record['actual_training_inputs_after']=bank_after
record['training_bank_unchanged']=bank_before==bank_after
after = {str(p): digest(p) for p in sorted(sources)}
record.update(actual_returncode=code, status='completed', source_sha256_after=after,
              all_sources_unchanged=before == after, elapsed_wall_s=time.monotonic()-started)
if (args.output/'result.json').is_file():
    result = json.loads((args.output/'result.json').read_text())
    record['child_result'] = {k: result.get(k) for k in ('status', 'updates', 'physical_transitions',
        'cleanup_succeeded', 'error', 'export', 'all_train_clips_seen')}
    record['result_sha256'] = digest(args.output/'result.json')
    for name in ('last.pt', 'Humanoid_AMASS.pth', 'metrics.jsonl'):
        path = args.output/name
        if path.is_file():
            record.setdefault('files', {})[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
receipt.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: record.get(k) for k in ('actual_returncode', 'all_sources_unchanged', 'elapsed_wall_s')}))
sys.exit(0 if code == 0 and before == after and bank_before==bank_after and record.get('child_result', {}).get('status') == 'passed' else 1)
