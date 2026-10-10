"""Actual admission negative tests only; every GPU/process call is forbidden."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LAUNCHER=ROOT/'docs/style_improvement_20261003/state_prior_actual_runs_v1/launch_registered_run.py'
AUDIT=ROOT/'docs/style_improvement_20261003/state_prior_training_audit_v1/actual_pilot4_final_v2/audit.json'
spec=importlib.util.spec_from_file_location('reviewed_launcher',LAUNCHER); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base=json.loads(AUDIT.read_text())
source_before=LAUNCHER.read_bytes(); rows=[]
folder=Path(tempfile.mkdtemp(prefix='state-prior-launch-negative-',dir='/tmp'))
def forbidden(*a,**k): raise RuntimeError('FORBIDDEN: GPU query or process execution from negative test')
for case in ('status_failed','source_wrong','full_resume_false','state_prior_false','public_D_RMS_false','path_not_plan_resume','size_changed','sha_changed'):
    gate=deepcopy(base)
    if case=='status_failed': gate['status']='failed'
    elif case=='source_wrong': gate['source_manifest']['sha256']='0'*64
    elif case=='full_resume_false': gate['checkpoints'][0]['full_payload_resume_validated']=False
    elif case=='state_prior_false': gate['checkpoints'][0]['state_prior_initialization_validated']=False
    elif case=='public_D_RMS_false': gate['checkpoints'][0]['public_D_or_AMP_RMS_transferred']=False
    elif case=='path_not_plan_resume': gate['checkpoints'][0]['checkpoint']['path']=str(HERE/'restore_actual_step4_cpu.py')
    elif case=='size_changed': gate['checkpoints'][0]['checkpoint']['bytes']+=1
    else: gate['checkpoints'][0]['checkpoint']['sha256']='0'*64
    path=folder/(case+'.json'); path.write_text(json.dumps(gate))
    with patch.object(sys,'argv',[str(LAUNCHER),'--run','resume4to8','--admission',str(path)]),patch.object(module,'HERE',folder),patch.object(module.subprocess,'Popen',forbidden),patch.object(module.subprocess,'check_output',forbidden):
        try: module.main()
        except ValueError as error: rows.append({'case':case,'rejected':True,'error':str(error)})
        else: raise RuntimeError('Negative admission was accepted: '+case)
    if (folder/'resume4to8_actual_launch.json').exists(): raise RuntimeError('A rejected gate wrote an actual launch record')
# Optimization must not remove the ValueError guard. Import is not a launch.
code="import importlib.util; from pathlib import Path; p=Path("+repr(str(LAUNCHER))+ "); s=importlib.util.spec_from_file_location('l',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m);\ntry: m.require(False,'optimized guard retained')\nexcept ValueError: print('explicit_require_retained_under_O')\nelse: raise RuntimeError('guard disappeared')"
result=subprocess.run([sys.executable,'-O','-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
if result.returncode!=0 or result.stdout.strip()!='explicit_require_retained_under_O': raise RuntimeError(result.stdout+result.stderr)
if LAUNCHER.read_bytes()!=source_before: raise RuntimeError('Launcher source changed during review')
report={'schema':'state_prior.launcher_fail_closed_CPU_peer.v1','status':'passed','launcher':{'path':str(LAUNCHER),'bytes':len(source_before),'sha256':hashlib.sha256(source_before).hexdigest()},'actual_admission':{'path':str(AUDIT),'sha256':hashlib.sha256(AUDIT.read_bytes()).hexdigest()},'negative_cases':rows,'explicit_require_active_under_python_O':True,'GPU_queries_or_Popen_calls':0,'actual_launch_records_created':0,'production_or_sealed_checkpoint_written':False}
output=HERE/'launcher_negative_cases.json';
if output.exists(): raise RuntimeError('Refuse to overwrite peer evidence')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','negative_cases':len(rows),'launcher_sha256':report['launcher']['sha256']},indent=2))
