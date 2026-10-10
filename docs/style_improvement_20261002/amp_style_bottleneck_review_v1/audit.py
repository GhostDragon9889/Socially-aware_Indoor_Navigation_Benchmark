"""Bounded, read-only CPU probes of immutable AMP bundles; no training/SDK."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CODE = ROOT / 'output/isolated_training/mixed_state_action_terrain_protected_v1/code'
sys.path.insert(0, str(CODE))
AMASSCorpus = importlib.import_module('CrowdSim.humanoid.amass_corpus').AMASSCorpus
action_to_pd = importlib.import_module('CrowdSim.humanoid.amp.controller').action_to_pd
_observations = importlib.import_module('CrowdSim.humanoid.amp.observations')
PACER_DOF_NAMES = _observations.PACER_DOF_NAMES
PacerSMPLMapping = _observations.PacerSMPLMapping
build_amp_observation = _observations.build_amp_observation


def file_record(path, *, digest=True):
    path = Path(path)
    result = {'path': str(path), 'bytes': path.stat().st_size}
    if digest:
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(chunk)
        result['sha256'] = checksum.hexdigest()
    return result


def summarize(values):
    x = torch.as_tensor(values).detach().double().flatten()
    assert x.numel() and torch.isfinite(x).all()
    return {'count': x.numel(), 'mean': x.mean().item(), 'min': x.min().item(),
            'p10': x.quantile(.1).item(), 'median': x.median().item(),
            'p90': x.quantile(.9).item(), 'max': x.max().item(),
            'rms': x.square().mean().sqrt().item()}


def rng_signature():
    return (random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone())


def metrics(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    keys = ('reward_mean', 'amp_rewards_mean', 'trajectory_rewards_mean',
            'energy_rewards_mean', 'direction_rewards_mean', 'forward_velocity_mean',
            'backwards_fraction', 'disc_expert_probability', 'disc_policy_probability',
            'disc_live_policy_probability', 'disc_replay_policy_probability',
            'disc_gradient_penalty', 'approximate_kl', 'motion_symmetry_loss')
    def window(selection):
        values = {key: summarize([row[key] for row in selection])
                  for key in keys if all(key in row for row in selection)}
        if all('amp_rewards_mean' in row for row in selection):
            values['weighted_amp_mean'] = summarize([2 * row['amp_rewards_mean'] for row in selection])
            values['weighted_task_position_mean'] = summarize([.5 * row['trajectory_rewards_mean'] for row in selection])
        return values
    return {'source': file_record(path), 'rows': len(rows), 'first_update': rows[0]['update_count'],
            'last_update': rows[-1]['update_count'], 'first_100_or_all': window(rows[:100]),
            'last_100_or_all': window(rows[-100:]),
            'reset_counter_scope': 'Cumulative within process, never sum update rows',
            'last_reported_process_counters': {key: rows[-1].get(key) for key in ('fall_resets', 'time_limit_resets', 'resets')}}


def discriminator(saved):
    sizes = saved['config']['discriminator_layers']
    input_size = saved['model']['discriminator.0.weight'].shape[1]
    with torch.random.fork_rng(devices=[]):
        modules = []
        for size in sizes:
            modules += [nn.Linear(input_size, size), nn.ReLU()]
            input_size = size
        modules += [nn.Linear(input_size, 1)]
        model = nn.Sequential(*modules)
    model.load_state_dict({key.removeprefix('discriminator.'): value
                          for key, value in saved['model'].items()
                          if key.startswith('discriminator.')}, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    torch.set_num_threads(4)
    before = rng_signature()
    assert not torch.cuda.is_initialized()
    corpus = AMASSCorpus(ROOT / 'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
                         device='cpu', seed=1719, dof_velocity_frame='child_joint')
    own, history, next_position, motion_ids, frames = corpus.sample_reference(512, 10)
    expert_history = build_amp_observation(history).reshape(512, 10, 195)
    # Source931 physics names/limits are read from each REAL saved provenance.
    checkpoints = {
        'mixed_state_only_2000': ('amp_style_mixed_curriculum_snapshots_v1/step_2000.pt',
                                 'amp_style_mixed_curriculum_v1/metrics.jsonl'),
        'mixed_state_action_2000': ('amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt',
                                   'amp_style_mixed_state_action_isolated_v1/metrics.jsonl'),
        'protected_4': ('amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step_4.pt',
                        'amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step4_metrics.jsonl'),
        'protected_8': ('amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step_8.pt',
                        'amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step8_metrics.jsonl'),
    }
    report = {'schema': 'local.amp_style_bottleneck_cpu_probe.v1', 'CPU_only': True,
              'CUDA_initialized': False, 'SDK_started': False, 'batch': 512,
              'selection': 'Private CPU seed1719, uniform actual train clip, uniform legal history10+next frame; same experts across D',
              'replay_selection': '512 equidistant stored ring-array indices, NOT known latest/live or branch IDs',
              'bank': file_record(corpus.manifest_path.parent / 'manifest_labeled_v7_train.pt', digest=False),
              'manifest': file_record(corpus.manifest_path),
              'sampled_motion_ids': motion_ids.tolist(), 'sampled_frames': frames.tolist(),
              'runs': {}}
    arms = [i for i, name in enumerate(PACER_DOF_NAMES)
            if name.rsplit('_', 1)[0] in {f'{side}_{body}' for side in ('L', 'R')
                                        for body in ('Thorax', 'Shoulder', 'Elbow', 'Wrist')}]
    selected_bodies = [name for name in PacerSMPLMapping().body_names[1:]
                       if name not in {'L_Hand', 'R_Hand', 'L_Toe', 'R_Toe'}]
    arm_features = [12 + 6*i+j for i, name in enumerate(selected_bodies)
                    if name in {f'{side}_{body}' for side in ('L', 'R')
                                for body in ('Thorax', 'Shoulder', 'Elbow', 'Wrist')} for j in range(6)]
    arm_features += [126 + 3*i+j for i, name in enumerate(selected_bodies)
                     if name in {f'{side}_{body}' for side in ('L', 'R')
                                 for body in ('Thorax', 'Shoulder', 'Elbow', 'Wrist')} for j in range(3)]
    arm_features += list(range(189, 195))  # Last two keypoints are R/L Wrist.
    speeds = own.root_vel[:, :2].norm(dim=-1)
    for label, (checkpoint, metric_path) in checkpoints.items():
        path = ROOT / 'output/diffusion_amp' / checkpoint
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        config, physics = saved['config'], saved['provenance']['physics']
        mapping = PacerSMPLMapping(tuple(physics['body_names']), tuple(physics['dof_names']))
        assert mapping.body_names == PacerSMPLMapping().body_names
        assert mapping.dof_names == PacerSMPLMapping().dof_names
        scale = action_to_pd(torch.ones(512, 69), torch.tensor(physics['dof_limits_lower']),
                             torch.tensor(physics['dof_limits_upper']), dof_names=mapping.dof_names)
        assert torch.equal(scale, scale[0].expand_as(scale))
        expert_actions = torch.where(scale.abs() > 1e-8, next_position / scale.clamp_min(1e-8), torch.zeros_like(scale)).clamp(-1, 1)
        current_proxy = torch.where(scale.abs() > 1e-8, own.dof_pos / scale.clamp_min(1e-8), torch.zeros_like(scale)).clamp(-1, 1)
        replay = saved['policy_replay']
        indices = torch.linspace(0, replay['size'] - 1, 512).long()
        policy_history, policy_actions = replay['history'][indices], replay['actions'][indices]
        mask = torch.tensor(config['action_mask'])
        assert int(mask.sum()) == 57
        assert torch.equal(policy_actions[:, mask == 0], torch.zeros(512, 12))
        assert torch.equal(expert_actions[:, mask == 0], torch.zeros(512, 12))
        normalizer = saved['normalizers']['amp_normalizer']
        assert tuple(normalizer['mean'].shape) == (1950,)
        model = discriminator(saved)
        def features(h, a):
            raw_z = (h.flatten(1)-normalizer['mean'].float()) / torch.sqrt(normalizer['variance'].float() + 1e-5)
            z = raw_z.clamp(-config['normalization_clip'], config['normalization_clip'])
            return (torch.cat((z, a), -1) if config['discriminator_input'] == 'state_action' else z), raw_z
        def evaluate(h, a, network=model):
            x, _ = features(h, a)
            with torch.no_grad():
                logits = network(x).flatten()
                reward = torch.nn.functional.softplus(logits).clamp_max(-math.log(config['amp_probability_epsilon'])) * config['amp_reward_scale']
            return {'probability': summarize(logits.sigmoid()), 'amp_reward': summarize(reward)}
        value = {'checkpoint': file_record(path), 'updates': saved['update_count'],
                 'physical_transitions': saved['physical_transition_count'],
                 'config': config, 'initialization': saved['provenance']['initialization'],
                 'amp_reward_timestamp': saved['provenance']['amp_reward_timestamp'],
                 'replay': {key: replay[key] for key in ('size', 'capacity', 'accepted', 'cursor')},
                 'amp_normalizer_count': normalizer['count'].item(),
                 'task_count': saved['normalizers']['task_normalizer']['count'].item(),
                 'metrics': metrics(ROOT / 'output/diffusion_amp' / metric_path),
                 'observed_replay': evaluate(policy_history, policy_actions),
                 'actual_expert_pose_proxy': evaluate(expert_history, expert_actions),
                 'counterfactual_replay_actions_zero': evaluate(policy_history, torch.zeros_like(policy_actions)),
                 'counterfactual_replay_actions_reversed': evaluate(policy_history, policy_actions.flip(0)),
                 'counterfactual_expert_current_pose_not_next': evaluate(expert_history, current_proxy),
                 'counterfactual_expert_policy_action_swap': evaluate(expert_history, policy_actions),
                 'next_current_proxy_delta': summarize((expert_actions-current_proxy)[:, mask.bool()]),
                 'expert_actions_active': summarize(expert_actions[:, mask.bool()]),
                 'policy_actions_active': summarize(policy_actions[:, mask.bool()]),
                 'expert_arms_action': summarize(expert_actions[:, arms]),
                 'policy_arms_action': summarize(policy_actions[:, arms]),
                 'action_saturation_fraction_active': {'expert': (expert_actions[:, mask.bool()].abs() >= .999999).float().mean().item(),
                                                      'policy': (policy_actions[:, mask.bool()].abs() >= .999999).float().mean().item()},
                 'expert_by_current_root_speed': {}}
        for origin, h, a in (('replay', policy_history, policy_actions), ('expert', expert_history, expert_actions)):
            x, raw_z = features(h, a)
            x = x.detach().requires_grad_(True)
            gradient = torch.autograd.grad(model(x).sum(), x)[0]
            total_sq = gradient.square().sum(-1)
            history_gradient = gradient[:, :1950].reshape(512, 10, 195)
            arm_sq = history_gradient[:, :, arm_features].square().sum((1, 2))
            value[f'{origin}_normalized_gradient'] = {
                'squared_gradient_action_fraction_mean': (gradient[:, 1950:].square().sum(-1) / total_sq.clamp_min(1e-12)).mean().item(),
                'squared_gradient_history_arms_fraction_mean': (arm_sq / total_sq.clamp_min(1e-12)).mean().item(),
                'absolute_gradient_rms': summarize(gradient),
                'AMP_unclipped_abs_over5_fraction': (raw_z.abs() > 5).float().mean().item()}
        for name, lower, upper in (('stationary_lt_0_12', 0, .12), ('slow_0_12_0_5', .12, .5),
                                   ('walking_0_5_1_8', .5, 1.8), ('fast_ge_1_8', 1.8, float('inf'))):
            keep = (speeds >= lower) & (speeds < upper)
            value['expert_by_current_root_speed'][name] = {'count': int(keep.sum()),
                'D': evaluate(expert_history[keep], expert_actions[keep]) if keep.any() else None}
        value['model_parameter_grad_populated'] = any(p.grad is not None for p in model.parameters())
        assert not value['model_parameter_grad_populated']
        report['runs'][label] = value
        del model, saved
    after = rng_signature()
    report['caller_RNG_unchanged'] = before[:2] == after[:2] and torch.equal(before[2], after[2])
    assert report['caller_RNG_unchanged'] and not torch.cuda.is_initialized()
    report['imported_helper_sources'] = {name: file_record(sys.modules[name].__file__)
        for name in ('CrowdSim.humanoid.amass_corpus', 'CrowdSim.humanoid.amp.controller', 'CrowdSim.humanoid.amp.observations')}
    report['script'] = file_record(__file__)
    with (OUT / 'actual_cpu_probe_v2.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'passed', 'reports': str(OUT / 'actual_cpu_probe_v2.json'),
                      'runs': list(report['runs']), 'caller_RNG_unchanged': report['caller_RNG_unchanged'],
                      'CUDA_initialized': torch.cuda.is_initialized()}))


if __name__ == '__main__':
    main()
