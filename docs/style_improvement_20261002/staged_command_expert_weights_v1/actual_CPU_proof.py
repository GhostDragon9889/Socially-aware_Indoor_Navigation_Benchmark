"""Prototype only, using 120 ACTUAL pre-action requests and512 actual v7 windows."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

from prototype import (WeightingSpec, classification_loss, command_context,
                       expert_context, expert_weights, renormalize_bounded)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CODE = ROOT/'output/isolated_training/mixed_state_action_terrain_protected_v1/code'
sys.path.insert(0, str(CODE))
amp_observations = importlib.import_module('CrowdSim.humanoid.amp.observations')
AMASSCorpus = importlib.import_module('CrowdSim.humanoid.amass_corpus').AMASSCorpus


def record(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(x):
    x = x.detach().double().flatten()
    return {'count': x.numel(), 'mean': x.mean().item(), 'min': x.min().item(),
            'max': x.max().item(), 'median': x.median().item()}


def make_D(saved):
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(2019, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    model.load_state_dict({key.removeprefix('discriminator.'): value
                          for key, value in saved['model'].items()
                          if key.startswith('discriminator.')}, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    torch.set_num_threads(4)
    before = (random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone())
    commands, command_files = [], []
    sample_dir = ROOT/'output/diffusion_amp/pre_action_baseline_cuboid_30s_v1/pre_action_observations'
    ledger = [json.loads(line) for line in (sample_dir/'samples.jsonl').read_text().splitlines()]
    assert len(ledger) == 30
    for row in ledger:
        assert row['raw_observation_capture_timing'] == 'immediately before original actor.act and before driver world.step'
        path = Path(row['file']['path'])
        assert record(path) == row['file']
        raw = np.load(path, allow_pickle=False)['task_obs']
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['arrays']['task_obs']['sha256_C_order']
        assert raw.shape == (4, 1044) and raw.dtype == np.float32
        commands.append(command_context(torch.from_numpy(raw.copy()), capture_phase='pre_action', observation_space='raw_task_1044'))
        command_files.append(row['file'])
    commands = torch.cat(commands)
    frozen = ROOT/'docs/style_improvement_20261002/amp_style_bottleneck_review_v1/actual_cpu_probe_v2.json'
    prior = json.loads(frozen.read_text())
    corpus = AMASSCorpus(ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
                         device='cpu', seed=771, dof_velocity_frame='child_joint')
    corpus_rng_before = corpus.generator.get_state().clone()
    ids = torch.tensor(prior['sampled_motion_ids'], dtype=torch.int64)
    frames = torch.tensor(prior['sampled_frames'], dtype=torch.int64)
    chronological = frames[:, None]+torch.arange(-9, 2)
    absolute = corpus.data['length_starts'][ids, None]+chronological
    roots = corpus.data['gts'][absolute, 0, :2]
    rotations = corpus.data['grs'][absolute, 0]
    descriptors, body_yaw = expert_context(roots, rotations,
        ids[:, None].expand(-1, 11), chronological, corpus.data['motion_num_frames'][ids].long())
    weights, description = expert_weights(commands, descriptors)
    assert weights.shape == (512,) and not weights.requires_grad
    assert weights.min() >= .5 and weights.max() <= 2
    assert abs(weights.double().mean().item()-1) <= 5e-7
    assert torch.equal(corpus_rng_before, corpus.generator.get_state())
    manifest_rows = {clip['motion_id']: clip for clip in corpus.entry['clips']}
    group_summary = {}
    for key in ('style_class', 'dataset'):
        groups = {}
        for motion_id, weight in zip(ids.tolist(), weights.tolist()):
            name = manifest_rows[motion_id][key]
            groups.setdefault(name, []).append(weight)
        group_summary[key] = {name: {'sampled_windows': len(values), 'unweighted_share': len(values)/512,
            'positive_loss_weighted_share': sum(values)/weights.sum().item(), 'weight_mean': sum(values)/len(values)}
            for name, values in groups.items()}
    saved_path = ROOT/'output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt'
    saved = torch.load(saved_path, weights_only=True, mmap=True, map_location='cpu')
    assert saved['update_count'] == 2000 and saved['config']['discriminator_input'] == 'state_action'
    physics = saved['provenance']['physics']
    action_to_pd = importlib.import_module('CrowdSim.humanoid.amp.controller').action_to_pd
    history_state = corpus.state_at((corpus.data['length_starts'][ids, None]+frames[:, None]-torch.arange(10)).reshape(-1))
    expert_h = amp_observations.build_amp_observation(history_state).reshape(512, 10, 195)
    next_q = corpus.data['dps'][corpus.data['length_starts'][ids]+frames+1]
    scale = action_to_pd(torch.ones(512, 69), torch.tensor(physics['dof_limits_lower']),
                         torch.tensor(physics['dof_limits_upper']), dof_names=tuple(physics['dof_names']))
    expert_actions = torch.where(scale.abs() > 1e-8, next_q/scale.clamp_min(1e-8), 0).clamp(-1, 1)
    rms = saved['normalizers']['amp_normalizer']
    def inputs(history, actions):
        z = ((history.flatten(1)-rms['mean'].float())/torch.sqrt(rms['variance'].float()+1e-5)).clamp(-5, 5)
        return torch.cat((z, actions), -1)
    positive_inputs = inputs(expert_h, expert_actions).requires_grad_(True)
    replay = saved['policy_replay']
    selection = torch.linspace(0, replay['size']-1, 512).long()
    model = make_D(saved)
    positive = model(positive_inputs).flatten()
    negative = model(inputs(replay['history'][selection], replay['actions'][selection])).flatten()
    original_positive_gradient = torch.autograd.grad(positive.sum(), positive_inputs, create_graph=True, retain_graph=True)[0]
    unweighted_GP = original_positive_gradient.square().sum(-1).mean()
    original_classification = classification_loss(positive, negative)
    weighted_classification = classification_loss(positive, negative, weights=weights)
    subsequent_gradient = torch.autograd.grad(positive.sum(), positive_inputs, create_graph=True, retain_graph=True)[0]
    assert torch.equal(subsequent_gradient, original_positive_gradient)
    assert torch.equal(unweighted_GP, subsequent_gradient.square().sum(-1).mean())
    assert all(p.grad is None for p in model.parameters())
    # A permuted selected subset gets explicit bounded re-normalization;
    # no RNG draw is used, and no new expert is manufactured.
    subset = torch.arange(0, 512, 8)
    subset_weights = renormalize_bounded(weights[subset])
    classification_loss(positive[subset], negative[subset], weights=subset_weights)
    after = (random.getstate(), repr(np.random.get_state()), torch.get_rng_state())
    assert before[:2] == after[:2] and torch.equal(before[2], after[2])
    assert not torch.cuda.is_initialized()
    report = {'schema': 'local.command_expert_weighting_actual_CPU_proof.v1', 'status': 'passed',
              'CPU_only': True, 'SDK_started': False, 'GPU_used': False, 'training_started': False,
              'source': {'script': record(__file__), 'prototype': record(OUT/'prototype.py'),
                         'expert_indices': record(frozen), 'manifest': record(corpus.manifest_path),
                         'bank': {'path': str(corpus.manifest_path.parent/'manifest_labeled_v7_train.pt'),
                                  'bytes': (corpus.manifest_path.parent/'manifest_labeled_v7_train.pt').stat().st_size,
                                  'SHA256_not_recomputed_here': True,
                                  'frozen_bank_sha256': '46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0'},
                         'checkpoint': prior['runs']['mixed_state_action_2000']['checkpoint'],
                         'actual_preaction_files': command_files},
              'commands': {'source_run': str(sample_dir.parent), 'actors': 4, 'samples': 30,
                           'context_rows': len(commands), 'scope': 'ACTUAL selected-baseline cuboid30s requests, not current protected training rollout distribution',
                           'forward_request': stats(commands[:, 0]), 'lateral_request': stats(commands[:, 1]),
                           'path_turn_rate_request': stats(commands[:, 2])},
              'experts': {'rows': 512, 'selected_parent_source_count': len({manifest_rows[i]['parent_source_path'] for i in ids.tolist()}),
                          'source_control_grid': 'real v7 30Hz history10+next, chronological t-9..t+1',
                          'forward': stats(descriptors[:, 0]), 'lateral': stats(descriptors[:, 1]),
                          'path_turn_rate': stats(descriptors[:, 2]), 'actual_body_yaw_rate_DIAGNOSTIC_NOT_WEIGHTED': stats(body_yaw)},
              'weights': description | {'min': weights.min().item(), 'max': weights.max().item()},
              'group_distribution_loss_weights_not_resampling': group_summary,
              'D': {'input_dimensions': 2019, 'effective_actions': 57,
                    'expert_actions_current_proxy_unchanged': True,
                    'classification_unweighted': original_classification.item(),
                    'classification_weighted': weighted_classification.item(),
                    'gradient_penalty_unweighted_unchanged': unweighted_GP.item(),
                    'no_parameter_grad_populated': True, 'raw_logits_or_inference_rewards_changed': False},
              'legacy_corpus_rng_unchanged': True, 'caller_CPU_rng_unchanged': True,
              'prototype_installed_or_activated': False,
              'limitations': ['No model update or quality gain.', 'Only positive-loss batch marginal adapts; D has no context input.',
                              'Command0.5s request and expert0.333s displacement use different windows, explicitly exposed.',
                              'Rawtask XY cannot identify desired body-yaw; path turn is distinct from body-yaw rate.',
                              'Metadata phase validates declared provenance; it cannot authenticate a dishonest caller.'],
              'spec': WeightingSpec().__dict__}
    np.savez(OUT/'actual_descriptors_weights_final.npz', commands=commands.numpy(), expert_descriptors=descriptors.numpy(),
             expert_body_yaw_diagnostic=body_yaw.numpy(), weights=weights.numpy(), motion_ids=ids.numpy(),
             local_frames=chronological.numpy())
    with (OUT/'actual_CPU_proof_final.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'passed', 'commands': len(commands), 'experts': len(weights),
                      'weights': [weights.min().item(), weights.double().mean().item(), weights.max().item()],
                      'RNG_unchanged': True, 'GPU_used': False}))


if __name__ == '__main__':
    main()
