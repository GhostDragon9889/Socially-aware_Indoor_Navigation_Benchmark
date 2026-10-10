"""Saved D/actor offline expert probes: no online-observation claim or physics."""
from __future__ import annotations

import json

import torch
from torch import nn

from audit import OUT, ROOT, discriminator, file_record, rng_signature, summarize
from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.amp.observations import (
    PACER_DOF_NAMES, build_amp_observation, build_self_observation, build_task_observation,
)
from CrowdSim.humanoid.amp.controller import action_to_pd


def mlp(weights, prefix):
    output_sizes = [weights[f'{prefix}.{index}.weight'].shape[0] for index in (0, 2, 4)
                    if f'{prefix}.{index}.weight' in weights]
    input_size = weights[f'{prefix}.0.weight'].shape[1]
    with torch.random.fork_rng(devices=[]):
        layers = []
        for index, width in enumerate(output_sizes):
            layers.append(nn.Linear(input_size, width))
            if index < len(output_sizes)-1 or prefix == 'task_encoder':
                layers.append(nn.ReLU())
            input_size = width
        result = nn.Sequential(*layers)
    result.load_state_dict({key.removeprefix(prefix + '.'): value for key, value in weights.items()
                           if key.startswith(prefix + '.')}, strict=True)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def main():
    torch.set_num_threads(4)
    before = rng_signature()
    previous = json.loads((OUT/'actual_cpu_probe_v2.json').read_text())
    corpus = AMASSCorpus(ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
                         seed=1719, device='cpu', dof_velocity_frame='child_joint')
    ids = torch.tensor(previous['sampled_motion_ids'])
    frames = torch.tensor(previous['sampled_frames'])
    indices = corpus.data['length_starts'][ids] + frames
    own = corpus.state_at(indices)
    hstates = corpus.state_at((indices[:, None] - torch.arange(10)).reshape(-1))
    h = build_amp_observation(hstates).reshape(512, 10, 195)
    q = own.root_rot
    yaw = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]),
                      1-2*(q[:, 1].square()+q[:, 2].square()))
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    side = torch.stack((-yaw.sin(), yaw.cos()), -1)
    velocity = own.root_vel[:, :2]
    speed = velocity.norm(dim=-1)
    forward_speed = (velocity * forward).sum(-1)
    lateral_speed = (velocity * side).sum(-1)
    counts = corpus.data['motion_num_frames'][ids]
    query = frames[:, None] + torch.arange(15, 151, 15)
    bounded = query.minimum(counts[:, None]-1)
    absolute = corpus.data['length_starts'][ids, None] + bounded
    xy = corpus.data['gts'][absolute, 0, :2]
    tail = (query-counts[:, None]+1).clamp_min(0).float()/30
    xy += corpus.data['gvs'][absolute, 0, :2]*tail[..., None]
    self_obs = build_self_observation(own)
    task_obs = build_task_observation(own, xy, lambda points: torch.zeros(points.shape[:-1]))
    assert torch.equal(task_obs[:, 20:], torch.zeros(512, 1024))
    clip_by_id = {clip['motion_id']: clip for clip in corpus.entry['clips']}
    groups = {}
    for index, motion in enumerate(ids.tolist()):
        row = clip_by_id[motion]
        name = row.get('style_class', row.get('motion_class', row.get('class', 'unknown')))
        groups.setdefault(name, []).append(index)
    heading = {'speed': summarize(speed), 'forward': summarize(forward_speed),
               'lateral': summarize(lateral_speed), 'moving_ge_0_12': int((speed >= .12).sum()),
               'moving_backward_fraction': ((forward_speed < 0) & (speed >= .12)).sum().item() / max(1, (speed >= .12).sum().item()),
               'moving_backward_below_minus_0_12_fraction': ((forward_speed < -.12) & (speed >= .12)).sum().item() / max(1, (speed >= .12).sum().item()),
               'by_metadata_class': {}}
    for group, values in groups.items():
        selection = torch.tensor(values)
        moving = speed[selection] >= .12
        heading['by_metadata_class'][group] = {'sampled_windows': len(values),
            'forward_speed': summarize(forward_speed[selection]),
            'lateral_speed': summarize(lateral_speed[selection]),
            'moving_count': int(moving.sum()),
            'moving_backward_fraction': int(((forward_speed[selection] < 0) & moving).sum()) / max(1, int(moving.sum()))}
    report = {'schema': 'local.amp_heading_actor_offline_probe.v1', 'CPU_only': True,
              'SDK_started': False, 'bank_modified': False, 'batch': 512,
              'basis': file_record(OUT/'actual_cpu_probe_v2.json'), 'heading': heading,
              'task_input_scope': 'Reconstructed actual AMASS source root future+0.5..5s; terminal linear-velocity tail; zero training terrain. NOT recorded online self/task observations.',
              'tail_query_knots': int((tail > 0).sum()), 'runs': {}}
    arms = [index for index, name in enumerate(PACER_DOF_NAMES)
            if name.rsplit('_', 1)[0] in {f'{side}_{body}' for side in ('L', 'R')
                                        for body in ('Thorax', 'Shoulder', 'Elbow', 'Wrist')}]
    for label, record in previous['runs'].items():
        saved = torch.load(record['checkpoint']['path'], weights_only=True, mmap=True, map_location='cpu')
        actor, task_encoder = mlp(saved['model'], 'actor'), mlp(saved['model'], 'task_encoder')
        mask = torch.tensor(saved['config']['action_mask'], dtype=torch.float32)
        def norm(x, label):
            n = saved['normalizers'][label]
            return ((x-n['mean'].float())/torch.sqrt(n['variance'].float()+1e-5)).clamp(-5, 5)
        with torch.no_grad():
            mean = actor(torch.cat((norm(self_obs, 'self_normalizer'),
                                    task_encoder(norm(task_obs, 'task_normalizer'))), -1))
            applied = mean.clamp(-1, 1) * mask
        physics = saved['provenance']['physics']
        scale = action_to_pd(torch.ones(512, 69), torch.tensor(physics['dof_limits_lower']),
                             torch.tensor(physics['dof_limits_upper']), dof_names=tuple(physics['dof_names']))
        target = scale * applied
        next_pose = corpus.data['dps'][indices+1]
        proxy = torch.where(scale.abs() > 1e-8, next_pose/scale.clamp_min(1e-8), torch.zeros_like(scale)).clamp(-1, 1)
        norm_h = norm(h.flatten(1), 'amp_normalizer')
        d = discriminator(saved)
        def probability(actions):
            inputs = torch.cat((norm_h, actions), -1) if saved['config']['discriminator_input'] == 'state_action' else norm_h
            with torch.no_grad():
                return summarize(d(inputs).flatten().sigmoid())
        delta = target[:, mask.bool()] - next_pose[:, mask.bool()]
        value = {'raw_mean_active': summarize(mean[:, mask.bool()]),
                 'applied_minus_next_pose_proxy_active': summarize((applied-proxy)[:, mask.bool()]),
                 'decoded_target_minus_next_mocap_pose_active_rad': summarize(delta),
                 'decoded_target_minus_next_mocap_pose_arms_rad': summarize((target-next_pose)[:, arms]),
                 'D_expert_history_next_pose_proxy': probability(proxy),
                 'D_expert_history_offline_actor_applied': probability(applied),
                 'actor_raw_abs_over1_fraction_active': (mean[:, mask.bool()].abs() > 1).float().mean().item(),
                 'caller_input_dtypes': [str(self_obs.dtype), str(task_obs.dtype)],
                 'physical_tracking_claim': False}
        report['runs'][label] = value
    after = rng_signature()
    report['caller_RNG_unchanged'] = before[:2] == after[:2] and torch.equal(before[2], after[2])
    assert report['caller_RNG_unchanged'] and not torch.cuda.is_initialized()
    report['script'] = file_record(__file__)
    with (OUT/'heading_actor_probe_v2.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'passed', 'output': str(OUT/'heading_actor_probe_v2.json'), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
