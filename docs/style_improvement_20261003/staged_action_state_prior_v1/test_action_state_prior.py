"""CPU integration against the real public checkpoint and frozen original code."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from CrowdSim.humanoid.amp.controller import PacerAMPController
from CrowdSim.humanoid.application.state_prior_initialization import PUBLIC_COUNTS, STRATEGY
from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
from CrowdSim.humanoid.runtime.checkpoint_contract import validate_bundle_identity
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['ORIGINAL_ROOT'])
PUBLIC = ROOT/'data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth'
EVIDENCE = {}


def load_frozen(name, relative):
    spec = importlib.util.spec_from_file_location(name, HERE/'before'/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD_CONFIG = load_frozen('CrowdSim.humanoid.application._frozen_settings', 'CrowdSim/humanoid/application/settings.py').TrainingConfig
OLD_TRAINER = load_frozen('CrowdSim.humanoid.training._frozen_trainer', 'CrowdSim/humanoid/training/trainer.py').PaperAMPTrainer


def rng():
    ns = np.random.get_state()
    return random.getstate(), (ns[0], ns[1].copy(), *ns[2:]), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def assert_tensors_equal(left, right):
    assert set(left) == set(right)
    assert all(torch.equal(value, right[key]) for key, value in left.items())


def make_config(**options):
    defaults = dict(discriminator_input='state_action', amp_normalization='per_history_coordinate',
                    warm_start_action_discriminator_from_state_prior=True,
                    task_terrain_statistics_mode='preserve_declared_suffix',
                    warm_start_normalizer_count_cap=200000., update_epochs=1, minibatch_size=8,
                    discriminator_batch_size=16, discriminator_updates_per_update=1,
                    discriminator_replay_capacity=32, discriminator_replay_keep_probability=1.)
    defaults.update(options)
    return TrainingConfig(**defaults)


def warm(trainer, **kwargs):
    return trainer.warm_start_pacer(PUBLIC, -torch.ones(69), torch.ones(69), **kwargs)


@pytest.fixture(scope='module')
def initialized():
    assert hashlib.sha256(PUBLIC.read_bytes()).hexdigest() == 'bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615'
    reference = PacerAMPController.from_checkpoint(PUBLIC, -torch.ones(69), torch.ones(69))
    trainer = PaperAMPTrainer(make_config())
    warm(trainer)
    return reference, trainer


@pytest.mark.parametrize('changes', [
    {'warm_start_action_discriminator_from_state_prior': 1},
    {'warm_start_action_discriminator_from_state_prior': 'false'},
    {'warm_start_discriminator': True}, {'discriminator_input':'state'},
    {'history_frames':11}, {'amp_frame_dim':194}, {'discriminator_layers':(1024,511)},
    {'amp_normalization':'per_frame'}, {'normalize_observations':False},
    {'normalize_values':False}, {'normalization_clip':4.},
])
def test_invalid_new_config_is_rejected(changes):
    with pytest.raises(ValueError):
        make_config(**changes).validate()


def test_setting_and_checkpoint_admission_remain_stdlib_only():
    code = 'from CrowdSim.humanoid.application.settings import TrainingConfig; from CrowdSim.humanoid.runtime.checkpoint_contract import validate_reconstruction_config; import sys; c=TrainingConfig(warm_start_action_discriminator_from_state_prior=True, amp_normalization="per_history_coordinate"); c.validate(); assert "torch" not in sys.modules; assert not any(s.startswith(("omni", "isaacsim")) for s in sys.modules)'
    result = subprocess.run([sys.executable, '-c', code], cwd=Path.cwd(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_new_profile_has_one_coupled_strategy_variable():
    before = yaml.safe_load((ROOT/'configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml').read_text())
    after = yaml.safe_load((HERE/'proposal/configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml').read_text())
    assert after['training'].pop('warm_start_action_discriminator_from_state_prior') is True
    assert before['training'] == after['training']
    for name in ('reward', 'trajectory_training', 'terrain_statistics', 'evaluation', 'data_manifest', 'mode', 'author_experiment_reproduction'):
        assert before[name] == after[name]


def test_legacy_false_exact_model_outputs_metrics_rng_and_provenance():
    options = dict(self_dim=4, task_dim=7, amp_frame_dim=3, task_layers=(12,6),
                   policy_layers=(16,8), discriminator_layers=(16,8), update_epochs=2,
                   minibatch_size=8, discriminator_batch_size=8)
    original = OLD_TRAINER(OLD_CONFIG(**options), provenance={'fixed':'old'})
    old_rng = rng()
    candidate = PaperAMPTrainer(TrainingConfig(**options), provenance={'fixed':'old'})
    assert_rng_equal(old_rng, rng())
    assert_tensors_equal(original.model.state_dict(), candidate.model.state_dict())
    g = torch.Generator().manual_seed(812)
    own, task = torch.randn(4,2,4,generator=g), torch.randn(4,2,7,generator=g)
    history, expert = torch.randn(4,2,10,3,generator=g), torch.randn(16,10,3,generator=g)
    initial_rng = rng()
    output = original.act(own.flatten(0,1), task.flatten(0,1)); result_rng = rng()
    restore_rng(initial_rng)
    counterpart = candidate.act(own.flatten(0,1), task.flatten(0,1))
    assert_rng_equal(result_rng, rng()); assert_tensors_equal(output, counterpart)
    data = {name:output[name].reshape(4,2,-1).squeeze(-1) for name in ('actions','applied_actions','log_prob','values')}
    data.update(self_obs=own,task_obs=task,amp_history=history,next_values=torch.zeros(4,2),rewards=torch.linspace(0,1,8).reshape(4,2),terminated=torch.zeros(4,2,dtype=torch.bool),truncated=torch.zeros(4,2,dtype=torch.bool))
    initial_rng = rng(); old_metrics = original.update(data, expert, torch.zeros(16,69)); result_rng=rng()
    restore_rng(initial_rng); metrics = candidate.update(data,expert,torch.zeros(16,69))
    assert metrics == old_metrics; assert_rng_equal(result_rng,rng())
    assert_tensors_equal(original.model.state_dict(),candidate.model.state_dict())
    assert original.provenance == candidate.provenance
    assert set(asdict(candidate.config))-set(asdict(original.config)) == {'warm_start_action_discriminator_from_state_prior'}
    EVIDENCE['legacy_exact'] = {'model':True,'outputs':True,'metrics':True,'cpu_python_numpy_rng':True,'provenance':True,'added_serialized_default_field':'warm_start_action_discriminator_from_state_prior=False'}


@pytest.mark.parametrize('cap',[200000.,None])
def test_legacy_public_warm_start_provenance_and_rng_exact(cap):
    options = dict(discriminator_input='state_action',amp_normalization='per_history_coordinate',
                   task_terrain_statistics_mode='preserve_declared_suffix',warm_start_normalizer_count_cap=cap)
    original = OLD_TRAINER(OLD_CONFIG(**options)); warm(original); old_rng=rng()
    candidate = PaperAMPTrainer(TrainingConfig(**options)); warm(candidate)
    assert_rng_equal(old_rng,rng()); assert original.provenance == candidate.provenance
    assert_tensors_equal(original.model.state_dict(),candidate.model.state_dict())
    for name in ('self_normalizer','task_normalizer','value_normalizer','amp_normalizer'):
        assert_tensors_equal(getattr(original,name).state_dict(),getattr(candidate,name).state_dict())


def test_public_state_prior_logits_all_actions_and_coupled_moments(initialized):
    reference, trainer = initialized
    g=torch.Generator().manual_seed(543)
    history=torch.randn(23,10,195,generator=g)
    first=trainer.model.discriminator[0]
    assert_tensors_equal(first.state_dict() | {'weight':first.weight[:,:1950]}, reference.network._disc_mlp[0].state_dict())
    assert torch.count_nonzero(first.weight[:,1950:]) == 0
    assert_tensors_equal(trainer.model.discriminator[2].state_dict(),reference.network._disc_mlp[2].state_dict())
    assert_tensors_equal(trainer.model.discriminator[4].state_dict(),reference.network._disc_logits.state_dict())
    logits=[]
    with torch.no_grad():
        expected=reference.eval_discriminator(history).squeeze(-1)
        for actions in (torch.zeros(23,69),torch.rand(23,69,generator=g)*2-1,torch.ones(23,69),-torch.ones(23,69)):
            actions=actions*trainer.action_mask
            h,a=trainer._amp_inputs(history,actions)
            assert torch.equal(h.flatten(1),reference.discriminator_normalizer(history.flatten(1)))
            logits.append(trainer.model.disc_logits(h,a))
    errors=[float((x-expected).abs().max()) for x in logits]
    assert max(errors)<2e-6
    assert all(torch.equal(logits[0],x) for x in logits[1:])
    assert torch.equal(trainer.amp_normalizer.mean,reference.discriminator_normalizer.running_mean)
    assert torch.equal(trainer.amp_normalizer.variance,reference.discriminator_normalizer.running_var)
    assert trainer.amp_normalizer.count.item()==200000.
    initialization=trainer.provenance['initialization']
    expected_counts={'self_normalizer':float(reference.observation_normalizer.count),'task_normalizer':float(reference.observation_normalizer.count),'amp_normalizer':float(reference.discriminator_normalizer.count),'value_normalizer':float(reference.value_normalizer.count)}
    assert initialization['normalizer_counts_before_cap']==expected_counts
    assert initialization['discriminator_initialization']==STRATEGY
    assert initialization['author_navisaaclab_weights'] is False
    EVIDENCE['initial_logits']={'public_vs_extended_fp32_max_abs':max(errors),'arbitrary_valid_actions_change_abs':0.0,'normalized_history_bitwise_equal':True,'amp_public_original_count':15482880001.0,'amp_capped_count':200000.0}


def test_public_counts_are_recorded_even_without_cap(tmp_path,monkeypatch):
    trainer=PaperAMPTrainer(make_config(warm_start_normalizer_count_cap=None))
    warm(trainer)
    assert trainer.provenance['initialization']['normalizer_counts_before_cap']==PUBLIC_COUNTS
    public=torch.load(PUBLIC,map_location='cpu',weights_only=True,mmap=True)
    assert trainer.amp_normalizer.count.item()==float(public['amp_input_mean_std']['count'])
    expected_counts={'self_normalizer':float(public['running_mean_std']['count']),'task_normalizer':float(public['running_mean_std']['count']),'amp_normalizer':float(public['amp_input_mean_std']['count']),'value_normalizer':float(public['reward_mean_std']['count'])}
    assert trainer.provenance['initialization']['normalizer_counts_before_cap']==expected_counts
    path=tmp_path/'uncapped_public_prior.pt'; trainer.save(path)
    def forbidden(*a,**k): raise AssertionError('uncapped resume must not reopen public checkpoint')
    monkeypatch.setattr(PacerAMPController,'from_checkpoint',forbidden)
    resumed,_=PaperAMPTrainer.load(path,restore_rng=False)
    assert_tensors_equal(trainer.model.state_dict(),resumed.model.state_dict())
    for name in expected_counts: assert_tensors_equal(getattr(trainer,name).state_dict(),getattr(resumed,name).state_dict())
    inference=ReconstructionAMPInference.from_checkpoint(path)
    assert inference.config.warm_start_normalizer_count_cap is None
    EVIDENCE['cap_none_real_public']={'warm_save_resume_inference_pass':True,'public_original_counts':expected_counts,'clamp_applied':False,'public_reopen_calls':0}


@pytest.mark.parametrize('cause',['transitions','replay','existing_initialization','update','policy_optimizer','discriminator_optimizer'])
def test_cold_only_rejection_happens_before_public_file_read(cause, monkeypatch):
    trainer=PaperAMPTrainer(make_config(task_terrain_statistics_mode='legacy'))
    if cause=='transitions': trainer.physical_transition_count=1
    elif cause=='replay': trainer.policy_replay.size=1
    elif cause=='existing_initialization': trainer.provenance['initialization']={}
    elif cause=='update': trainer.update_count=1
    elif cause=='policy_optimizer': trainer.optimizer.state[trainer.model.actor[0].weight]={'step':torch.tensor(1.)}
    else: trainer.discriminator_optimizer.state[trainer.model.discriminator[0].weight]={'step':torch.tensor(1.)}
    def forbidden(*a,**k): raise AssertionError('public file must not be reopened')
    monkeypatch.setattr(PacerAMPController,'from_checkpoint',forbidden)
    before=deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError): warm(trainer)
    assert_tensors_equal(before,trainer.model.state_dict())


def test_wrong_public_hash_rejected_before_any_model_transfer(tmp_path):
    trainer=PaperAMPTrainer(make_config(task_terrain_statistics_mode='legacy'))
    before=deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        warm(trainer,expected_sha256='0'*64)
    public=torch.load(PUBLIC,map_location='cpu',weights_only=True,mmap=True)
    public['extra_test_metadata']='independently changed public artifact'
    changed=tmp_path/'changed_public.pt'; torch.save(public,changed)
    digest=hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='verified public PACER checkpoint SHA'):
        trainer.warm_start_pacer(changed,-torch.ones(69),torch.ones(69),expected_sha256=digest)
    assert_tensors_equal(before,trainer.model.state_dict())
    public['model']['a2c_network._disc_mlp.0.weight']=public['model']['a2c_network._disc_mlp.0.weight'][:,:1949]
    torch.save(public,changed); digest=hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Invalid PACER model tensor'):
        trainer.warm_start_pacer(changed,-torch.ones(69),torch.ones(69),expected_sha256=digest)
    assert_tensors_equal(before,trainer.model.state_dict())


@pytest.fixture(scope='module')
def trained_bundle(initialized,tmp_path_factory):
    _, initialized_trainer=initialized
    trainer=deepcopy(initialized_trainer)
    g=torch.Generator().manual_seed(431)
    # Statistics-shaped inputs exercise classification gradients without saturation.
    h0=trainer.amp_normalizer.mean.float().reshape(1,10,195)
    scale=trainer.amp_normalizer.variance.float().sqrt().reshape(1,10,195)
    agent=h0+torch.randn(16,10,195,generator=g)*scale*.2
    expert=h0+torch.randn(16,10,195,generator=g)*scale*.2
    aa=(torch.rand(16,69,generator=g)*1.6-.8)*trainer.action_mask
    ea=(torch.rand(16,69,generator=g)*1.6-.8)*trainer.action_mask
    own=trainer.self_normalizer.mean.float()[None]+torch.randn(16,357,generator=g)*trainer.self_normalizer.variance.float().sqrt()[None]*.2
    task=trainer.task_normalizer.mean.float()[None]+torch.randn(16,1044,generator=g)*trainer.task_normalizer.variance.float().sqrt()[None]*.2
    output=trainer.act(own,task)
    data={name:output[name].reshape(4,4,-1).squeeze(-1) for name in ('actions','applied_actions','log_prob','values')}
    data.update(self_obs=own.reshape(4,4,357),task_obs=task.reshape(4,4,1044),amp_history=agent.reshape(4,4,10,195),next_values=torch.zeros(4,4),rewards=torch.linspace(0,1,16).reshape(4,4),terminated=torch.zeros(4,4,dtype=torch.bool),truncated=torch.zeros(4,4,dtype=torch.bool))
    metrics=trainer.update(data,expert,ea)
    gradient=trainer.model.discriminator[0].weight.grad[:,1950:]
    norms=gradient.norm(dim=0)
    assert (norms[trainer.action_mask.bool()]>0).all()
    assert torch.count_nonzero(gradient[:,~trainer.action_mask.bool()])==0
    assert torch.count_nonzero(trainer.model.discriminator[0].weight[:,1950:])>0
    assert torch.count_nonzero(trainer.model.discriminator[0].weight[:,1950:][:,~trainer.action_mask.bool()])==0
    assert all(np.isfinite(value) for value in metrics.values())
    path=tmp_path_factory.mktemp('actual_resume')/'trained.pt'
    trainer.save(path,runner_state={'actual_cpu_trainer_updates':1})
    EVIDENCE['actual_trainer_update']={'active_columns_with_nonzero_gradient':int((norms>0).sum()),'frozen_columns_with_nonzero_gradient':int((norms[~trainer.action_mask.bool()]>0).sum()),'active_min_column_gradient_norm':float(norms[trainer.action_mask.bool()].min()),'trained_action_columns_nonzero':int(torch.count_nonzero(trainer.model.discriminator[0].weight[:,1950:])),'finite_metrics':True,'update_count':trainer.update_count,'physical_transition_contract_count':trainer.physical_transition_count,'synthetic_cpu_contract_rollout_not_physics':True}
    return trainer,path


def test_saved_trained_actions_resume_infer_without_public_read(trained_bundle,monkeypatch):
    trainer,path=trained_bundle
    def forbidden(*args,**kwargs): raise AssertionError('resume/inference must not reopen public checkpoint')
    monkeypatch.setattr(PacerAMPController,'from_checkpoint',forbidden)
    loaded,runner=PaperAMPTrainer.load(path,restore_rng=False)
    assert runner=={'actual_cpu_trainer_updates':1}
    assert_tensors_equal(trainer.model.state_dict(),loaded.model.state_dict())
    assert loaded.discriminator_optimizer.state
    before_rng=rng(); inference=ReconstructionAMPInference.from_checkpoint(path); assert_rng_equal(before_rng,rng())
    g=torch.Generator().manual_seed(339)
    own=torch.randn(3,357,generator=g); task=torch.randn(3,1044,generator=g)
    assert torch.equal(inference.act(own,task)['applied_actions'],loaded.act(own,task,deterministic=True)['applied_actions'])
    EVIDENCE['trained_resume']={'all_model_tensors_exact':True,'nonzero_action_columns_preserved':True,'inference_cpu_rng_exact':True,'public_reopen_calls':0,'discriminator_optimizer_state_restored':True}


@pytest.mark.parametrize('case',['missing','strategy','sha','mode','counts','cap','extension_bool','published_amp','flag_removed','flag_false_extension_missing','flag_removed_extension_missing','normalizer_shape','model_shape'])
def test_independently_tampered_checkpoint_rejected_by_resume_and_inference(trained_bundle,tmp_path,case):
    _,path=trained_bundle
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    init=bundle['provenance']['initialization']
    if case=='missing': del bundle['provenance']['initialization']
    elif case=='strategy': init['discriminator_initialization']='independent_random'
    elif case=='sha': init['sha256']='0'*64
    elif case=='mode': bundle['config']['discriminator_input']='state'
    elif case=='counts': init['normalizer_counts_before_cap']['amp_normalizer']=200000.
    elif case=='cap': init['normalizer_count_cap']=True
    elif case=='extension_bool': init['state_prior_action_extension']['action_columns']=True
    elif case=='published_amp': init['published_amp_history_moments_transferred']=False
    elif case=='flag_removed': del bundle['config']['warm_start_action_discriminator_from_state_prior']
    elif case=='flag_false_extension_missing': bundle['config']['warm_start_action_discriminator_from_state_prior']=False; del init['state_prior_action_extension']
    elif case=='flag_removed_extension_missing': del bundle['config']['warm_start_action_discriminator_from_state_prior']; del init['state_prior_action_extension']
    elif case=='normalizer_shape': bundle['normalizers']['amp_normalizer']['mean']=bundle['normalizers']['amp_normalizer']['mean'][:-1]
    else: bundle['model']['discriminator.0.weight']=bundle['model']['discriminator.0.weight'][:,:1950]
    corrupt=tmp_path/f'{case}.pt'; torch.save(bundle,corrupt)
    for loader in (lambda:PaperAMPTrainer.load(corrupt,restore_rng=False),lambda:ReconstructionAMPInference.from_checkpoint(corrupt)):
        with pytest.raises((ValueError,RuntimeError)): loader()
    if case not in ('model_shape','normalizer_shape'):
        with pytest.raises(ValueError): validate_bundle_identity(bundle)


def test_enabled_save_requires_provenance_and_legacy_missing_flag_defaults_false(tmp_path):
    cold=PaperAMPTrainer(make_config(task_terrain_statistics_mode='legacy'))
    with pytest.raises(ValueError,match='initialization provenance'): cold.save(tmp_path/'cold.pt')
    assert not (tmp_path/'cold.pt').exists()
    legacy=PaperAMPTrainer(TrainingConfig(self_dim=4,task_dim=7,amp_frame_dim=3,task_layers=(12,6),policy_layers=(16,8),discriminator_layers=(16,8)))
    path=tmp_path/'old.pt'; legacy.save(path)
    bundle=torch.load(path,weights_only=True); del bundle['config']['warm_start_action_discriminator_from_state_prior']; torch.save(bundle,path)
    loaded,_=PaperAMPTrainer.load(path,restore_rng=False)
    assert loaded.config.warm_start_action_discriminator_from_state_prior is False
    assert loaded.provenance==legacy.provenance
    assert_tensors_equal(legacy.model.state_dict(),loaded.model.state_dict())


def emit_evidence():
    import json
    evidence_file=os.environ.get('CANDIDATE_EVIDENCE')
    if evidence_file:
        Path(evidence_file).write_text(json.dumps(EVIDENCE,indent=2)+'\n')


@pytest.mark.parametrize('terrain_mode',['legacy','preserve_declared_suffix'])
def test_unwarmed_execution_rejected_before_forward_rng_statistics_or_counters(terrain_mode, monkeypatch):
    trainer=PaperAMPTrainer(make_config(task_terrain_statistics_mode=terrain_mode))
    def forbidden(*args,**kwargs): raise AssertionError('uninitialized prior must not execute a model forward')
    monkeypatch.setattr(trainer.model,'policy',forbidden)
    monkeypatch.setattr(trainer.model,'disc_logits',forbidden)
    before=deepcopy(trainer.model.state_dict())
    normalizers={name:deepcopy(getattr(trainer,name).state_dict()) for name in ('self_normalizer','task_normalizer','value_normalizer','amp_normalizer')}
    before_rng=rng()
    own=torch.zeros(1,357); task=torch.zeros(1,1044); history=torch.zeros(1,10,195); actions=torch.zeros(1,69)
    for operation in (lambda:trainer.act(own,task),lambda:trainer.value(own,task),lambda:trainer.amp_reward(history,actions),lambda:trainer.update({},history,actions)):
        with pytest.raises(ValueError,match='initialization provenance'): operation()
    assert_tensors_equal(before,trainer.model.state_dict()); assert_rng_equal(before_rng,rng())
    assert trainer.update_count==trainer.physical_transition_count==0
    assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state
    assert trainer.provenance=={}
    for name,values in normalizers.items(): assert_tensors_equal(values,getattr(trainer,name).state_dict())


def test_from_scratch_state_prior_legacy_rejected_stdlib_before_resources():
    code = "from types import SimpleNamespace; from CrowdSim.humanoid.application.resources import preflight_amp_resources,ResourceError; import sys; args=SimpleNamespace(seed=7,from_scratch=True,trajectory_source='diffusion'); profile={'training':{'task_terrain_statistics_mode':'legacy','warm_start_discriminator':False,'warm_start_action_discriminator_from_state_prior':True}};\ntry: preflight_amp_resources(args,profile)\nexcept ResourceError as e: assert 'From-scratch' in str(e)\nelse: raise AssertionError('from-scratch must fail before resources');\nassert 'torch' not in sys.modules; assert not any(s.startswith(('omni','isaacsim')) for s in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=Path.cwd(),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_emit_evidence():
    emit_evidence()
