"""CPU source-only proof. No checkpoint/model or future output is opened."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
R=D.parents[2]
O=R/'docs/style_improvement_20261003/state_prior_training_audit_v1'
B=R/'output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2'

def record(path):
    p=Path(path).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA disabled required')
    spec=json.loads((D/'adaptation_spec.json').read_text())
    assert record(O/'audit_state_prior_training.py')['sha256']=='2c60b6381b17353fd71f69547b499214329cc19c6b8fee57ae4176bd21d84e9f'
    assert record(O/'state_prior_checks.py')['sha256']=='cdb8a07e7ae26ed2f4bc7e36f9a2a1c33a06138c7c18c35ce660c71fcb021a0c'
    oldmain=(O/'audit_state_prior_training.py').read_text()
    newmain=(D/'audit_pacer_reward_training.py').read_text()
    new_metadata='''        reward_contract = {name: expected_config[name] for name in ("amp_reward_scale", "amp_reward_weight", "task_reward_weight", "direction_mode", "amp_probability_epsilon")}
        check("registered_pacer_reward_five_fields_exact", all(type(reward_contract[name]) is type(value) and reward_contract[name] == value for name, value in {"amp_reward_scale": 2.0, "amp_reward_weight": 0.5, "task_reward_weight": 0.5, "direction_mode": "backwards_penalty", "amp_probability_epsilon": 0.0001}.items()))
        report["reward_contract"] = reward_contract
'''
    assert newmain.count(new_metadata)==1
    normalized=newmain.replace(new_metadata,'')
    for before,after in spec['main_replacements'].items():
        assert after in normalized,after
        normalized=normalized.replace(after,before)
    assert normalized==oldmain
    assert ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(oldmain),include_attributes=False)
    oldhelper=(O/'state_prior_checks.py').read_text()
    newhelper=(D/'state_prior_checks.py').read_text()
    normalized_helper=newhelper[:newhelper.index('def verify_source_archive(')]
    normalized_helper=normalized_helper.replace('2e21c1facc64186598ade1d72375f5a0a5dde07ad5f6fd05978ac019383b51c7','00e5e0c8a861f208a7b1eda7d0e7eaf9ef60989dfa29f51596c5bbdaa76c1e1d').replace(spec['source_archive']['sha256'],'f47357ec02e1558bb733d194109f2a1681b122179f6fc9dff5d254596a488695')
    normalized_helper=normalized_helper.replace('state_prior_pacer_reward_training_preparation_v1/root_GPU_launch_commands_v2.json','state_prior_training_preparation_v1/root_GPU_launch_commands.json').replace('state_prior_pacer_reward_actual_runs_v1','state_prior_actual_runs_v1')
    normalized_helper=normalized_helper.replace('\n          and launch["source_manifest"] == record(bundle / "source_manifest.json")\n          and plan["source_manifest"] == record(bundle / "source_manifest.json")','')
    normalized_helper=normalized_helper.replace('\n               and launch["child_exit_observed"] is True and launch["driver_completed"] is True','')
    assert normalized_helper==oldhelper[:oldhelper.index('def verify_source_archive(')]
    old_funcs={node.name:ast.dump(node,include_attributes=False) for node in ast.parse(oldhelper).body if isinstance(node,ast.FunctionDef)}
    new_funcs={node.name:ast.dump(node,include_attributes=False) for node in ast.parse(newhelper).body if isinstance(node,ast.FunctionDef)}
    unchanged=[name for name in old_funcs if name not in {'_verify_launch','verify_source_archive'}]
    assert all(old_funcs[name]==new_funcs[name] for name in unchanged)
    import_spec=importlib.util.spec_from_file_location('pacer_reward_source_proof_helper',D/'state_prior_checks.py')
    helper=importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(helper)
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition: raise ValueError(name)
    source=json.loads((B/'source_manifest.json').read_text())
    archive_proof=helper.verify_source_archive(B,source,record,check)
    launch,_=helper._verify_launch(B,'pilot4',record(B/'source_manifest.json')['sha256'],B/'pilot4_GPU_imports.json',record,check,closed=True)
    negatives=[]
    oldsource=json.loads((R/'output/isolated_training/mixed_state_action_terrain_state_prior_v1/source_manifest.json').read_text())
    wrong=copy.deepcopy(source);wrong['source_epoch']['modified_old_paths']=['code/scripts/train_paper_amp.py']
    for name,bad in [('old934_source_refused',oldsource),('incomplete_resume_guard_derivation_refused',wrong)]:
        try:
            helper.verify_source_archive(B,bad,record,lambda label,ok: None if ok else (_ for _ in ()).throw(ValueError(label)))
        except ValueError as error:
            negatives.append({'name':name,'passed':True,'rejected_at':str(error)})
        else: raise AssertionError(name)
    assert not any(name.split('.')[0] in {'torch','isaacsim','isaaclab','omni','pxr'} for name in sys.modules)
    result={'schema':'pacer_reward.independent_auditor_metadata_adaptation_proof.v1','status':'passed',
        'source_bound_main':record(D/'audit_pacer_reward_training.py'),'source_bound_helper':record(D/'state_prior_checks.py'),
        'original_main':record(O/'audit_state_prior_training.py'),'original_helper':record(O/'state_prior_checks.py'),
        'original_reader_freeze':record(O/'source_freeze_v1.json'),'adaptation_spec':record(D/'adaptation_spec.json'),
        'main_reverse_metadata_exact_original_bytes':True,'main_reverse_metadata_full_AST_exact':True,
        'helper_prefix_reverse_metadata_exact_original_bytes':True,'helper_numerical_functions_AST_exact':unchanged,
        'replaced_metadata_only_functions':['_verify_launch','verify_source_archive'],
        'new_source_archive_derivation':archive_proof,'checks':checks,'check_count':len(checks),'negative_checks':negatives,
        'closed_actual_pilot_launch_metadata_validated':launch['child_exit_observed'] is True,
        'checkpoint_consumed':False,'model_loaded':False,'future_terminal_claim':False,'CPU_only':True,
        'GPU_used':False,'SDK_started':False,'sealed34_outputs_read':False,'final_training_source':record(B/'source_manifest.json')}
    with (D/'metadata_adaptation_CPU_proof.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'passed','source_checks':len(checks),'negative_checks':len(negatives),'unchanged_math_functions':unchanged}))

if __name__=='__main__':
    main()
