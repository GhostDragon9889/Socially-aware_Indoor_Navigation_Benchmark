"""Bound source-only proof of original raw mathematics and JSON representation fix."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
R=D.parents[2]
O=R/'docs/style_improvement_20261003/state_prior_native_reader_preparation_v1'
REV=R/'docs/style_improvement_20261003/state_prior_native_reader_json_revision_v2'

def record(path):
    path=Path(path).resolve()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    original=(O/'audit_state_prior_native.py').read_text()
    new=(D/'audit_state_prior_native.py').read_text()
    replacements={
        'pacer_reward.native200s_CPU_reader_registration.v1':'public_state_prior.native200s_CPU_reader_registration.v1',
        'pacer_reward.native200s_independent_CPU_audit.v1':'public_state_prior.native200s_independent_CPU_audit.v1',
        'Prepared before actual new4/15 captures; checkpoint SHA and actual terminal/native evidence remain pending. Actual closed owner records are required for execution.':'Reader protocol recorded while native4 may already run; actual preflight and owner prelaunch source snapshots are bound separately. This is not a claim that this CPU reader preceded GPU launch.',
        'proof = {"base_source_archive_index_changes": 1':'return native, {"base_source_archive_index_changes": 1',
        '    return native, json.loads(json.dumps(proof, allow_nan=False))\n':'',
    }
    restored=new
    for before,after in replacements.items():
        assert restored.count(before)>=1
        restored=restored.replace(before,after)
    assert restored==original
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)
    mathematical=[]
    for name in ('frozen_v1.py','frozen_v2.py','frozen_routes.py'):
        assert (D/name).read_bytes()==(O/name).read_bytes()
        mathematical.append({'original':record(O/name),'new_copy':record(D/name),'entire_bytes_exact':True})
    # Actual source/shape adapter execution on the unchanged core; no raw capture is opened.
    sys.path.insert(0,str(D))
    spec=importlib.util.spec_from_file_location('pacer_reward_native_source_proof',D/'audit_state_prior_native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapters=[]
    for actors in (4,15):
        _,proof=module.readers(actors)
        assert proof==json.loads(json.dumps(proof,allow_nan=False))
        assert proof['base_reverse_restored_full_AST_equals_original'] is True
        assert proof['v2_reverse_restored_full_AST_equals_original'] is True
        assert proof['COM_force_fall_route_math_unchanged'] is True
        adapters.append({'actors':actors,'adapter_proof':proof,'JSON_stable':True})
    revision=ast.parse((REV/'audit_native15_json_revision.py').read_text())
    canonical=next(n for n in revision.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_proof')
    expression=canonical.body[-1].value
    expected=ast.parse('json.loads(json.dumps(proof, allow_nan=False))',mode='eval').body
    assert ast.dump(expression,include_attributes=False)==ast.dump(expected,include_attributes=False)
    assert not any(name.split('.')[0] in {'torch','isaacsim','isaaclab','omni','pxr','warp'} for name in sys.modules)
    payload={'schema':'pacer_reward.native_raw_math_identity_proof.v1','status':'passed','new_main':record(D/'audit_state_prior_native.py'),
        'original_main':record(O/'audit_state_prior_native.py'),'original_core_freeze':record(O/'source_freeze_v1.json'),
        'original_JSON_revision_freeze':record(REV/'source_freeze_v2.json'),
        'main_reverse_metadata_and_JSON_representation_full_bytes_exact':True,'main_reverse_full_AST_exact':True,
        'raw_math_files_exact':mathematical,'actual_source_shape_adapter_proofs':adapters,
        'canonical_JSON_expression_exact_original_v2':True,'controls_for_physical_gate':6000,'MMD_is_physical_gate':False,
        'future_checkpoint_SHA_fabricated':False,'actual_checkpoint_read':False,'actual_capture_read':False,'future_effect':False,
        'GPU_used':False,'SDK_started':False,'sealed34_outputs_read':False}
    with (D/'native_math_identity_CPU_proof.json').open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'passed','main_reverse_full_bytes_AST_exact':True,'raw_math_files_exact':3,'actual_case_adapter_proofs':[4,15],'future_effect':False}))

if __name__=='__main__':
    main()
