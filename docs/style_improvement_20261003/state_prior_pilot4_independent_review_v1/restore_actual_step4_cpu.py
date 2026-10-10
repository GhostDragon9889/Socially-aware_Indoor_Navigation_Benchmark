"""Independent actual sealed step4 restore; no synthetic training or GPU."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'output/isolated_training/mixed_state_action_terrain_state_prior_v1'
CODE=BUNDLE/'code'
PATH=ROOT/'output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_pilot_v1/step_4.pt'
EXPECTED_SHA='01b8b860045943d7c8994d4dbfa9592df0768a1c6d363e82e95c560797f03bd9'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(CODE),str(CODE/'src')]
os.environ['PROTOMOTIONS_ASSET_ROOT']=str(CODE/'protomotions/data/assets')
import torch
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
from CrowdSim.humanoid.amp.controller import PacerAMPController


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def same(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def forbidden(*args,**kwargs): raise AssertionError('Actual recovery must not reopen the public checkpoint')


def reject_check(name,condition,**detail):
    if not condition: raise ValueError(name)


torch.set_num_threads(4)
before=fingerprint(PATH)
assert before['sha256']==EXPECTED_SHA and before['bytes']==388410713 and not PATH.stat().st_mode&0o222
state=torch.load(PATH,map_location='cpu',weights_only=True,mmap=True)
init=state['provenance']['initialization']
assert state['update_count']==4 and state['physical_transition_count']==32768
assert init['normalizer_counts_before_cap']=={'self_normalizer':61931520001.,'task_normalizer':61931520001.,'amp_normalizer':15482880001.,'value_normalizer':6881280001.}
assert init['discriminator_initialization']==init['action_conditioned_discriminator_initialization']=='public_pacer_state_prior_zero_action_extension'
assert init['published_state_only_discriminator_transferred'] is init['published_amp_history_moments_transferred'] is True
counts={name:float(value['count']) for name,value in state['normalizers'].items()}
assert counts=={'self_normalizer':232768.,'task_normalizer':232768.,'amp_normalizer':249152.,'value_normalizer':232768.}
weights=state['model']['discriminator.0.weight']
mask=torch.tensor(state['config']['action_mask'],dtype=torch.bool)
assert weights.shape==(1024,2019) and int(mask.sum())==57
active_columns=int((weights[:,1950:][:,mask]!=0).any(dim=0).sum())
frozen_nonzero=int(torch.count_nonzero(weights[:,1950:][:,~mask]))
assert active_columns==57 and frozen_nonzero==0
original_load=torch.load; loads=[]
def only_sealed_checkpoint(path,*args,**kwargs):
    if not isinstance(path,(Path,str)) or Path(path).resolve()!=PATH.resolve(): raise AssertionError('Recovery attempted a different checkpoint')
    loads.append(str(Path(path).resolve()))
    return original_load(path,*args,**kwargs)
with patch.object(PacerAMPController,'from_checkpoint',forbidden),patch.object(torch,'load',only_sealed_checkpoint):
    loaded,runner=PaperAMPTrainer.load(PATH,device='cpu',restore_rng=False)
assert loads==[str(PATH)]
assert same(loaded.model.state_dict(),state['model'])
assert same(loaded.optimizer.state_dict(),state['optimizer']) and same(loaded.discriminator_optimizer.state_dict(),state['discriminator_optimizer'])
assert all(same(getattr(loaded,name).state_dict(),values) for name,values in state['normalizers'].items())
assert same(loaded._pending_rng_state,state['rng']) and same(runner,state['runner_state'])
assert same(loaded.policy_replay.state_dict(),state['policy_replay'])
assert torch.equal(loaded.model.discriminator[0].weight[:,1950:],weights[:,1950:])
assert loaded.update_count==4 and loaded.physical_transition_count==32768
sys.path.insert(0,str(ROOT/'docs/style_improvement_20261003/state_prior_training_audit_v1'))
import state_prior_checks
state_prior_checks.validate_state_prior(state,reject_check)
negative_results=[]
for problem in ('D_transfer_false','RMS_transfer_false','false_flag','original_AMP_count_is_cap','action_columns_zeroed','frozen_action_column_changed'):
    broken=dict(state)
    broken['config']=dict(state['config'])
    broken['provenance']=dict(state['provenance']); broken['provenance']['initialization']=deepcopy(init)
    broken['model']=dict(state['model'])
    if problem=='D_transfer_false': broken['provenance']['initialization']['published_state_only_discriminator_transferred']=False
    elif problem=='RMS_transfer_false': broken['provenance']['initialization']['published_amp_history_moments_transferred']=False
    elif problem=='false_flag': broken['config']['warm_start_action_discriminator_from_state_prior']=False
    elif problem=='original_AMP_count_is_cap': broken['provenance']['initialization']['normalizer_counts_before_cap']['amp_normalizer']=200000.
    else:
        broken['model']['discriminator.0.weight']=weights.clone()
        if problem=='action_columns_zeroed': broken['model']['discriminator.0.weight'][:,1950:].zero_()
        else: broken['model']['discriminator.0.weight'][0,1950+int((~mask).nonzero()[0])]=.01
    try: state_prior_checks.validate_state_prior(broken,reject_check)
    except ValueError as error: negative_results.append({'case':problem,'rejected':True,'gate':str(error)})
    else: raise AssertionError('Auditor accepted independent tamper: '+problem)
owned=[]
for name,module in tuple(sys.modules.items()):
    if name.startswith(('CrowdSim','protomotions')) and getattr(module,'__file__',None):
        path=Path(module.__file__).resolve(); assert path.is_relative_to(CODE); owned.append({'module':name,'path':str(path)})
after=fingerprint(PATH); assert after==before
assert not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('isaaclab','isaacsim','omni','pxr') for k in sys.modules)
report={'schema':'actual_state_prior_pilot4.independent_cpu_recovery.v1','status':'passed','checkpoint':before,
        'actual_updates':4,'actual_transitions':32768,'public_prior_counts_before_cap':init['normalizer_counts_before_cap'],
        'initial_count_cap':200000.,'actual_saved_RMS_counts':counts,'AMP_count_equation':'200000+4*(8192+4096)=249152',
        'actual_trained_action_columns':active_columns,'actual_frozen_column_nonzero_weights':frozen_nonzero,
        'real_saved_model_optimizers_normalizers_replay_RNG_and_runner_exact':True,
        'trained_action_columns_preserved_during_real_load':True,'checkpoint_load_paths':loads,'public_checkpoint_reopen_calls':0,
        'owned_imports':owned,'actual_payload_tamper_rejections':negative_results,
        'auditor_sources':{name:fingerprint(ROOT/'docs/style_improvement_20261003/state_prior_training_audit_v1'/name) for name in ('audit_state_prior_training.py','state_prior_checks.py')},
        'CPU_only':True,'CUDA_rng_restored':False,'GPU_or_Kit_started':False,'checkpoint_before_after_exact':True,
        'synthetic_rollouts_or_gradient_evidence_used':False,'complete_curriculum_clock_rng_checks_remain_with_primary_2066_check_reader':True}
output=HERE/'actual_step4_restore.json'; assert not output.exists(); output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','actual_RMS_counts':counts,'trained_action_columns':active_columns,'negative_cases':len(negative_results),'checkpoint_sha256':before['sha256']},indent=2))
