"""Freeze the seven explicitly authorized source deltas over parent V10."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=ROOT/'docs/style_improvement_20261003/native_runtime_epoch_v10/native_pedestrian_source_v10.json'
STAGED=ROOT/'docs/style_improvement_20261003/staged_action_state_prior_v1'
FIX=ROOT/'docs/style_improvement_20261003/runner_reward_fixture_fix_v1'
BASELINE=ROOT/'output/isolated_training/mixed_state_action_terrain_protected_v1'
ALLOWED_MODIFIED={'CrowdSim/humanoid/application/resources.py','CrowdSim/humanoid/application/settings.py',
                  'CrowdSim/humanoid/runtime/checkpoint_contract.py','CrowdSim/humanoid/training/trainer.py',
                  'tests/test_amp_pacer_adaptation.py'}
ALLOWED_ADDED={'CrowdSim/humanoid/application/state_prior_initialization.py',
               'configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml'}


def identity(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def json_write(path,value):
    assert not path.exists(),f'Refuse to overwrite frozen history: {path}'
    path.write_text(json.dumps(value,indent=2)+'\n')


def junit(path):
    suites=ET.parse(path).getroot().findall('.//testsuite')
    return {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}


parent=json.loads(PARENT.read_text()); assert parent['source_count']==len(parent['files'])==181
parent_by_path={str(Path(row['path']).relative_to(ROOT)):row for row in parent['files']}
assert len(parent_by_path)==181 and ALLOWED_MODIFIED<=set(parent_by_path) and not ALLOWED_ADDED&set(parent_by_path)
contract=json.loads((STAGED/'merge_manifest.json').read_text()); assert len(contract['targets'])==6
payloads={}; comparison=[]; changed=set()
for relative,row in sorted(parent_by_path.items()):
    path=ROOT/relative; assert path.is_file() and not path.is_symlink()
    payload=path.read_bytes(); after=identity(path); payloads[relative]=payload
    change=after['sha256']!=row['sha256'] or after['bytes']!=row['bytes']
    if change: changed.add(relative)
    comparison.append({'path':relative,'before':row,'after':after,'changed':change,'authorized_change':relative in ALLOWED_MODIFIED})
assert changed==ALLOWED_MODIFIED,(changed,ALLOWED_MODIFIED)
for target in contract['targets']:
    relative=target['path']; path=ROOT/relative; actual=identity(path)
    assert actual['sha256']==target['after_sha256'] and actual['bytes']==target['after_bytes']
    assert path.read_bytes()==(STAGED/'proposal'/relative).read_bytes()
    if target['before_sha256'] is None:
        assert relative in ALLOWED_ADDED
        payloads[relative]=path.read_bytes()
    else:
        assert relative in ALLOWED_MODIFIED
        assert parent_by_path[relative]['sha256']==target['before_sha256']
        assert parent_by_path[relative]['bytes']==target['before_bytes']
assert len(payloads)==183
# Preserve both actual fixture test runs and both earlier source snapshots.
fix_first=json.loads((FIX/'manifest.json').read_text())
assert identity(FIX/'before.py')['sha256']==fix_first['before_sha256']==parent_by_path['tests/test_amp_pacer_adaptation.py']['sha256']
assert identity(FIX/'after.py')['sha256']==fix_first['after_sha256']
final_payload=payloads['tests/test_amp_pacer_adaptation.py']
assert final_payload!=(FIX/'after.py').read_bytes()
assert final_payload==(FIX/'after.py').read_bytes().replace(b'for node in loop.body if isinstance(node, ast.Assign)',b'for node in ast.walk(loop) if isinstance(node, ast.Assign)',1)
assert not (FIX/'after_final.py').exists(); (FIX/'after_final.py').write_bytes(final_payload)
first_run=junit(FIX/'actual_tests.xml'); final_run=junit(FIX/'final_actual_tests.xml')
assert first_run=={'tests':2,'failures':2,'errors':0,'skipped':0}
assert final_run=={'tests':2,'failures':0,'errors':0,'skipped':0}
fix_v2={'schema':'runner_reward_fixture_context_fix.v2','path':'tests/test_amp_pacer_adaptation.py',
        'before':identity(FIX/'before.py'),'after_first':identity(FIX/'after.py'),'after_final':identity(FIX/'after_final.py'),
        'historical_manifest':identity(FIX/'manifest.json'),'changes':['Supply training-mode pure_evaluation=False already defined by the runner','Find transition assignment through ast.walk(loop) after runner moved it into a guarded block'],
        'actual_tests':[{'source':identity(FIX/'after.py'),'artifact':identity(FIX/'actual_tests.xml'),'result':first_run,'failure':'StopIteration after first fixture-only repair'},
                        {'source':identity(FIX/'after_final.py'),'artifact':identity(FIX/'final_actual_tests.xml'),'result':final_run}],
        'production_final_sha256':identity(ROOT/'tests/test_amp_pacer_adaptation.py')['sha256'],
        'production_algorithm_or_driver_changed':False,'historical_files_preserved':True}
json_write(FIX/'manifest_v2.json',fix_v2)
# Rehash all original 931 source files and six frozen data/weight inputs.
baseline_manifest=json.loads((BASELINE/'source_manifest.json').read_text())
assert len(baseline_manifest['files'])==931

def verify_baseline(row):
    actual=identity(BASELINE/row['path']); assert actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],row['path']
    return {'path':row['path'],'before':{'sha256':row['sha256'],'bytes':row['bytes']},'after':actual,'unchanged':True}


def verify_resource(row):
    path=Path(row['path']); before_stat=path.stat(); actual=identity(path); after_stat=path.stat()
    assert actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],row['path']
    assert (before_stat.st_size,before_stat.st_mtime_ns,before_stat.st_ino)==(after_stat.st_size,after_stat.st_mtime_ns,after_stat.st_ino)
    return {'before':row,'after':{**actual,'mtime_ns':after_stat.st_mtime_ns,'device':after_stat.st_dev,'inode':after_stat.st_ino},'sha256_and_bytes_unchanged':True,'stat_tuple_matches_frozen':(after_stat.st_mtime_ns,after_stat.st_dev,after_stat.st_ino)==(row['mtime_ns'],row['device'],row['inode'])}

with ThreadPoolExecutor(max_workers=4) as pool:
    baseline_rows=list(pool.map(verify_baseline,baseline_manifest['files']))
    resource_rows=list(pool.map(verify_resource,baseline_manifest['resources']))
protected={'schema':'native_runtime_epoch.v11.protected_fingerprints.v1','baseline_manifest':identity(BASELINE/'source_manifest.json'),
           'readonly_931_source_count':len(baseline_rows),'readonly_931_source_rows':baseline_rows,
           'resource_count':len(resource_rows),'resources':resource_rows,
           'resource_total_bytes':sum(row['after']['bytes'] for row in resource_rows),
           'all_known_weights_data_public_checkpoint_fingerprints_unchanged':True,'resource_contents_deserialized':False}
json_write(HERE/'protected_fingerprints.json',protected)
# Deterministic archive: relative paths sorted; regular files only; metadata fixed.
def archive_bytes():
    out=io.BytesIO()
    with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0,compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for relative,payload in sorted(payloads.items()):
                info=tarfile.TarInfo(relative); info.size=len(payload); info.uid=info.gid=0; info.uname=info.gname=''; info.mode=0o644; info.mtime=0
                archive.addfile(info,io.BytesIO(payload))
    return out.getvalue()
archive=archive_bytes(); second=archive_bytes(); assert archive==second
archive_path=HERE/'native_pedestrian_source_v11.tar.gz'; assert not archive_path.exists(); archive_path.write_bytes(archive)
with tarfile.open(archive_path,'r:gz') as stored:
    members=stored.getmembers(); assert len(members)==183 and [m.name for m in members]==sorted(payloads)
    for member in members:
        assert member.isfile() and member.uid==member.gid==member.mtime==0 and member.mode==0o644
        assert stored.extractfile(member).read()==payloads[member.name]
# Verify production did not change while fingerprinting/archiving.
for relative,payload in payloads.items(): assert (ROOT/relative).read_bytes()==payload
rows=[identity(ROOT/relative) for relative in sorted(payloads)]
receipt={'schema':'action_state_prior.actual_merge_receipt.v1','status':'actual_production_merge_verified',
         'candidate_contract':identity(STAGED/'merge_manifest.json'),'candidate_patch':identity(STAGED/'proposed.patch'),
         'six_exact_candidate_targets':[{'path':t['path'],'before_sha256':t['before_sha256'],'before_bytes':t['before_bytes'],'actual_after':identity(ROOT/t['path']),'matches_frozen_proposal':True} for t in contract['targets']],
         'fixture_final':identity(FIX/'manifest_v2.json'),'parent_source':identity(PARENT),
         'parent_file_comparison':comparison,'parent_existing_files_changed':len(changed),'parent_existing_files_unchanged':181-len(changed),
         'new_files':sorted(ALLOWED_ADDED),'selected_source_count_after':183,
         'allowed_targets_only':True,'freeze_script_changed_production':False,'GPU_or_Kit_started':False}
json_write(HERE/'actual_merge_receipt.json',receipt)
source={'schema':'selected_native_runtime_source.v11','files':rows,'source_count':183,'archive':identity(archive_path),
        'parent_manifest_sha256':identity(PARENT)['sha256'],'parent_manifest':identity(PARENT),'parent_archive':parent['archive'],
        'parent_existing_files_changed':5,'parent_existing_files_unchanged':176,'new_source_files':sorted(ALLOWED_ADDED),
        'approved_changes':[r for r in comparison if r['changed']],
        'scope':'parent181 preserved except four explicitly approved AMP state-prior implementation files and one fixture-only test repair; two explicit helper/profile additions',
        'runtime_or_training_algorithm_changed':True,'change_kind':'explicit optional coupled public D/history-RMS initialization; legacy false numerical/provenance/RNG equivalence verified',
        'actual_merge_receipt':identity(HERE/'actual_merge_receipt.json'),'protected_fingerprints':identity(HERE/'protected_fingerprints.json'),
        'deterministic_tar_proof':{'builds_compared':2,'archive_bytes_equal':True,'archive_members_reverified':183,'sort':'relative path ascending','gzip_mtime':0,'tar_mtime':0,'uid':0,'gid':0,'uname':'','gname':'','mode':'0644','format':'USTAR','symlinks_or_hardlinks':0},
        'isolated931_training_source_changed':False,'known_dataset_weights_fingerprints_unchanged':True,
        'SDK_modified_by_freeze':False,'production_sources_modified_by_freeze':False,'GPU_or_Kit_started':False,
        'production_runtime_effect_not_measured_by_packaging':True,'created_utc':datetime.now(timezone.utc).isoformat()}
json_write(HERE/'native_pedestrian_source_v11.json',source)
print(json.dumps({'status':'frozen_actual_183_source_v11','manifest':identity(HERE/'native_pedestrian_source_v11.json'),'archive':source['archive'],'changed_parent_files':5,'unchanged_parent_files':176,'added_files':2,'protected_931_sources':931,'protected_resources':len(resource_rows),'protected_bytes':protected['resource_total_bytes']},indent=2))
