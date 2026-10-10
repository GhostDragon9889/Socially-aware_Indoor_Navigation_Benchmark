"""Materialize an isolated /tmp source overlay; production stays read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SDKPY='/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python'
parser=argparse.ArgumentParser()
parser.add_argument('--keep-overlay',action='store_true')
args=parser.parse_args()
manifest=json.loads((HERE/'merge_manifest.json').read_text())
for target in manifest['targets']:
    source=ROOT/target['path']; proposed=HERE/'proposal'/target['path']
    assert hashlib.sha256(proposed.read_bytes()).hexdigest()==target['after_sha256']
    if target['before_sha256'] is not None:
        assert hashlib.sha256(source.read_bytes()).hexdigest()==target['before_sha256']
    else: assert not source.exists()
overlay=Path(tempfile.mkdtemp(prefix='action-state-prior-cpu-',dir='/tmp'))
shutil.copytree(ROOT/'CrowdSim',overlay/'CrowdSim',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for target in manifest['targets']:
    destination=overlay/target['path']; destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(HERE/'proposal'/target['path'],destination)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
         PYTHONPATH=f'{overlay}:{ROOT}/src:{ROOT}',ORIGINAL_ROOT=str(ROOT),CANDIDATE_EVIDENCE=str(HERE/'cpu_evidence.json'))
command=[SDKPY,'-m','pytest',str(HERE/'test_action_state_prior.py'),'-q','-p','no:cacheprovider','--junitxml',str(HERE/'cpu_tests.xml')]
result=subprocess.run(command,cwd=overlay,env=env,capture_output=True,text=True)
(HERE/'cpu_tests.log').write_text(result.stdout+'\n'+result.stderr)
report={'command':command,'overlay':str(overlay),'exit_code':result.returncode,'cuda_visible_devices':'','threads':4,
        'targets':manifest['targets'],'production_before_after_sha_equal':all(t['before_sha256'] is None and not (ROOT/t['path']).exists() or t['before_sha256'] is not None and hashlib.sha256((ROOT/t['path']).read_bytes()).hexdigest()==t['before_sha256'] for t in manifest['targets'])}
(HERE/'cpu_gate.json').write_text(json.dumps(report,indent=2)+'\n')
print(result.stdout); print(result.stderr); print('Overlay:',overlay)
if not args.keep_overlay: shutil.rmtree(overlay)
raise SystemExit(result.returncode)
