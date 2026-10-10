"""Drop one new guard in a disposable /tmp overlay and prove the test fails."""
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
gate=json.loads((HERE/'cpu_gate.json').read_text())
source=Path(gate['overlay'])
assert source.is_dir(), 'Run the CPU gate with --keep-overlay first'
overlay=Path(tempfile.mkdtemp(prefix='action-state-prior-mutation-',dir='/tmp'))
shutil.copytree(source/'CrowdSim',overlay/'CrowdSim',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
relative='CrowdSim/humanoid/application/state_prior_initialization.py'
target=overlay/relative
before=target.read_bytes()
assert before== (HERE/'proposal'/relative).read_bytes()
old=b'    if type(enabled) is not bool:\n'
assert before.count(old)==1
mutant=before.replace(old,b'    if False:  # mutation: drop strict boolean guard\n',1)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{overlay}:{ROOT}/src:{ROOT}',ORIGINAL_ROOT=str(ROOT))
command=[SDKPY,'-m','pytest',str(HERE/'test_action_state_prior.py'),'-q','-p','no:cacheprovider','-k','invalid_new_config']
try:
    target.write_bytes(mutant)
    result=subprocess.run(command,cwd=overlay,env=env,capture_output=True,text=True)
finally:
    target.write_bytes(before)
restored=target.read_bytes()==before
(HERE/'guard_mutation.log').write_text(result.stdout+'\n'+result.stderr)
report={'status':'expected_failure_observed' if result.returncode==1 and 'DID NOT RAISE' in result.stdout and restored else 'failed',
        'mutation':'Replace strict new-flag boolean type condition with False in /tmp copy only',
        'path':relative,'source_sha256':hashlib.sha256(before).hexdigest(),'mutant_sha256':hashlib.sha256(mutant).hexdigest(),
        'restored_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'restored_original_bytes':restored,
        'command':command,'exit_code':result.returncode,'production_or_proposal_changed':False}
(HERE/'guard_mutation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2)); print(result.stdout)
shutil.rmtree(overlay)
assert report['status']=='expected_failure_observed'
