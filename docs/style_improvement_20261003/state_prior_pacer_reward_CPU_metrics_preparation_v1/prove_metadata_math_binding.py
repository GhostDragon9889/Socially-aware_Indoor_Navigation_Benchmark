"""Prove only metric identity metadata changed; never process actual captures."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

D=Path(__file__).resolve().parent
R=D.parents[2]
O=R/'docs/style_improvement_20261003/state_prior_CPU_metrics_preparation_v1'

def record(p):
    p=Path(p).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    spec=json.loads((D/'metadata_adaptation_spec.json').read_text())
    old=(O/'run_CPU_metrics.py').read_text();new=(D/'run_CPU_metrics.py').read_text()
    for before,after in spec['runner_metadata_replacements'].items():
        assert after in new
        new=new.replace(after,before)
    assert new==old
    assert ast.dump(ast.parse(new),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
    original=json.loads((O/'commands_CPU_metrics_v1.json').read_text())
    current=json.loads((D/'commands_CPU_metrics_v1.json').read_text())
    assert original['metric_rules']==current['metric_rules']
    changed={'evaluation_source_freeze','evaluation_executor','evaluation_registration','evaluation_evidence','isolated_source_manifest','isolated_launcher','style90_entry','style90_math'}
    exact_sources={k:v for k,v in original['sources'].items() if k not in changed}
    assert all(current['sources'][k]==v for k,v in exact_sources.items())
    for key in ('isolated_launcher','style90_entry','style90_math'):
        assert all(current['sources'][key][k]==original['sources'][key][k] for k in ('bytes','sha256'))
    argv=[]
    for metric,row in current['metrics'].items():
        previous=original['metrics'][metric]
        substitutions={row['run_name']+'='+row['input']:previous['run_name']+'='+previous['input'],row['output']:previous['output']}
        normalized=[substitutions.get(value,value) for value in row['metric_argv']]
        assert normalized==previous['metric_argv']
        assert row['protocol']==previous['protocol']
        if 'requires_prior_metric' in previous:assert row['requires_prior_metric']==previous['requires_prior_metric']
        argv.append({'metric':metric,'original_argv':previous['metric_argv'],'new_argv':row['metric_argv'],'non_identity_metric_argv_exact':True})
    normalized=deepcopy(current)
    for key in ('schema','runner','bundle','source_admission','prepare_only_command'):normalized[key]=original[key]
    normalized['sources']=original['sources'];normalized['metrics']=original['metrics']
    assert normalized==original
    result={'schema':'pacer_reward.CPU_metric_metadata_and_math_binding_proof.v1','status':'passed',
            'runner':record(D/'run_CPU_metrics.py'),'original_runner':record(O/'run_CPU_metrics.py'),
            'catalog':record(D/'commands_CPU_metrics_v1.json'),'original_catalog':record(O/'commands_CPU_metrics_v1.json'),
            'metadata_adaptation_spec':record(D/'metadata_adaptation_spec.json'),
            'runner_reverse_metadata_entire_original_bytes_exact':True,'runner_reverse_metadata_full_AST_exact':True,
            'all_math_feature_source_bytes_exact':True,'reference_basis_source_records_exact':exact_sources,
            'all_primary_secondary_metric_rules_exact':True,'all_non_identity_metric_argv_exact':argv,
            'all_other_catalog_metadata_exact':True,'new_reward_expected_by_terminal_source_admission':{'amp_reward_scale':2.0,'amp_reward_weight':0.5,'task_reward_weight':0.5,'direction_mode':'backwards_penalty','amp_probability_epsilon':0.0001},
            'future_effect':False,'actual_capture_read':False,'actual_checkpoint_read':False,'actual_metrics_computed':False,'GPU_used':False,'SDK_started':False,'sealed34_outputs_read':False}
    with (D/'metadata_math_CPU_proof.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'passed','full_runner_bytes_and_AST_exact_after_metadata_normalization':True,'fixed_non_identity_argv_rows':len(argv),'unchanged_reference_tool_records':len(exact_sources)}))

if __name__=='__main__':
    main()
