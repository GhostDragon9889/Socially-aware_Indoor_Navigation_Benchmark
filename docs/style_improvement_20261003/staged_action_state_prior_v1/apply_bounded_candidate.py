"""Verify the exact candidate; write only with an explicit --apply option."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
ALLOWED={
    'CrowdSim/humanoid/application/settings.py',
    'CrowdSim/humanoid/application/resources.py',
    'CrowdSim/humanoid/training/trainer.py',
    'CrowdSim/humanoid/runtime/checkpoint_contract.py',
    'CrowdSim/humanoid/application/state_prior_initialization.py',
    'configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml',
}
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=HERE.parents[2])
parser.add_argument('--apply',action='store_true')
args=parser.parse_args()
root=args.root.resolve(strict=True)
manifest=json.loads((HERE/'merge_manifest.json').read_text())
targets=manifest['targets']
assert len(targets)==len(ALLOWED) and {t['path'] for t in targets}==ALLOWED
assert sum(t['after_bytes'] for t in targets)<=150000
payloads=[]
for target in targets:
    relative=Path(target['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path=root/relative
    for parent in [path,*path.parents]:
        if parent==root: break
        assert not parent.is_symlink(), f'Symlink redirect refused: {parent}'
    assert path.resolve().is_relative_to(root)
    source=HERE/'proposal'/relative
    assert source.is_file() and not source.is_symlink()
    payload=source.read_bytes()
    assert len(payload)==target['after_bytes']<=120000
    assert hashlib.sha256(payload).hexdigest()==target['after_sha256']
    if target['before_sha256'] is None:
        assert not path.exists(), f'New target already exists: {path}'
    else:
        assert path.is_file() and path.stat().st_size==target['before_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==target['before_sha256'], f'Before SHA mismatch: {path}'
        snapshot=HERE/'before'/relative
        assert snapshot.read_bytes()==path.read_bytes()
    payloads.append((path,payload))
check=subprocess.run(['patch','--batch','--fuzz=0','-p1','--dry-run','-i',str(HERE/'proposed.patch')],cwd=root,capture_output=True,text=True)
assert check.returncode==0, check.stdout+check.stderr
if args.apply:
    for path,payload in payloads:
        path.parent.mkdir(parents=True,exist_ok=True)
        mode=path.stat().st_mode & 0o777 if path.exists() else 0o644
        fd,temporary=tempfile.mkstemp(prefix=path.name+'.candidate.',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream: stream.write(payload)
            os.chmod(temporary,mode)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    assert all(hashlib.sha256((root/t['path']).read_bytes()).hexdigest()==t['after_sha256'] for t in targets)
print(json.dumps({'status':'applied_exact_targets' if args.apply else 'verified_only_no_writes','root':str(root),'target_count':len(targets),'total_payload_bytes':sum(t['after_bytes'] for t in targets),'fuzz':0},indent=2))
