"""Freeze source hashes and actual gate artifacts without touching production."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def identity(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def junit(path):
    suites=ET.parse(path).getroot().findall('.//testsuite')
    return {name:sum(int(s.get(name,0)) for s in suites) for name in ('tests','failures','errors','skipped')}

manifest=json.loads((HERE/'merge_manifest.json').read_text())
gate=json.loads((HERE/'cpu_gate.json').read_text())
assert gate['exit_code']==0 and gate['targets']==manifest['targets'] and gate['production_before_after_sha_equal'] is True
for target in manifest['targets']:
    assert identity(HERE/'proposal'/target['path'])['sha256']==target['after_sha256']
    current=ROOT/target['path']
    if target['before_sha256'] is None: assert not current.exists()
    else:
        assert identity(current)['sha256']==target['before_sha256']
        assert (HERE/'before'/target['path']).read_bytes()==current.read_bytes()
new_tests=junit(HERE/'cpu_tests.xml'); regression=junit(HERE/'regression_tests.xml'); inherited=junit(HERE/'inherited_failure_baseline.xml')
assert new_tests=={'tests':44,'failures':0,'errors':0,'skipped':0}
assert regression=={'tests':222,'failures':0,'errors':0,'skipped':0}
assert inherited=={'tests':2,'failures':2,'errors':0,'skipped':0}
mutation=json.loads((HERE/'guard_mutation.json').read_text())
helper=next(t for t in manifest['targets'] if t['path'].endswith('state_prior_initialization.py'))
assert mutation['status']=='expected_failure_observed' and mutation['exit_code']==1
assert mutation['source_sha256']==mutation['restored_sha256']==helper['after_sha256'] and mutation['restored_original_bytes'] is True
assert 'All checks passed!' in (HERE/'ruff.log').read_text()
public=identity(ROOT/'data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth')
assert public['sha256']=='bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615'
report={'schema':'navisaaclab.action_state_prior_candidate_cpu_proof.v1',
        'status':'source_bound_CPU_proof_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'candidate_contract':identity(HERE/'merge_manifest.json'),'proposed_patch':identity(HERE/'proposed.patch'),
        'target_digest_binding':manifest['targets'],'new_tests':new_tests,'legacy_regression':regression,
        'known_inherited_runner_fixture_failures':inherited,
        'inherited_failure_reason':'Original runner AST test fixture omits pure_evaluation; both cases reproduce against production source before candidate changes',
        'execution_environment':{'python':'/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python','cuda_visible_devices':'','omp_threads':4,'mkl_threads':4,'SDK_Kit_or_GPU_started':False},
        'evidence':json.loads((HERE/'cpu_evidence.json').read_text()),'mutation':mutation,
        'public_checkpoint_verified_actual':public,
        'protected_base_profile':identity(ROOT/'configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml'),
        'production_target_sha256_unchanged':True,'new_production_targets_absent':True,'candidate_not_merged_or_trained':True,
        'early_pytest_cache_exception':{'paths':['.pytest_cache/v/cache/nodeids','.pytest_cache/v/cache/lastfailed'],'source_or_model_data_changed':False,'parent_notified':True,'future_reproduction_cacheprovider_disabled':True},
        'artifacts':{name:identity(HERE/name) for name in ('cpu_gate.json','cpu_evidence.json','cpu_tests.xml','cpu_tests.log','regression_tests.xml','regression_tests.log','inherited_failure_baseline.xml','inherited_failure_baseline.log','guard_mutation.json','guard_mutation.log','ruff.log','test_action_state_prior.py','run_cpu_gate.py','run_guard_mutation.py','apply_bounded_candidate.py')}}
(HERE/'source_bound_cpu_proof.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'new_tests':new_tests,'regression':regression,'manifest_sha256':report['candidate_contract']['sha256']},indent=2))
