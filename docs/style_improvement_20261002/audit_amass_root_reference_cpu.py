"""Finite CPU inventory/proof of real v7 commands and crop-tail risk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum


def main():
    if torch.cuda.is_available():
        raise RuntimeError("This bounded audit requires CUDA_VISIBLE_DEVICES='' and CPU only")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]
    manifest = root/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
    c = AMASSCorpus(manifest, split='train', device='cpu', seed=7, dof_velocity_frame='child_joint')
    n = c.data['motion_num_frames'][c.ids].long()
    low, high = torch.full_like(n, 9), n-32
    legal = high-low+1
    assert len(c.ids) == 2848 and bool((legal > 0).all())
    assert all(int(size) == clip['frames'] for size, clip in zip(n, c.entry['clips'], strict=True))
    assert torch.allclose(c.data['motion_dt'][c.ids], torch.full_like(c.data['motion_dt'][c.ids], 1/30), atol=1e-7, rtol=0)
    minimum_remaining, maximum_remaining = torch.full_like(n, 31), n-10
    mean_remaining = (minimum_remaining+maximum_remaining).double()/2
    initial_hold = []
    pre_counts, post_counts = [], []
    for delta in range(15, 151, 15):
        held = (high-torch.maximum(low, n-delta)+1).clamp_min(0)
        initial_hold.append(float((held.double()/legal).mean()))
        per_clip_pre, per_clip_post = [], []
        for count in n.tolist():
            remaining = torch.arange(31, count-9, dtype=torch.float64)
            per_clip_pre.append(float(remaining.clamp_max(delta-1).mean()))
            per_clip_post.append(float(remaining.clamp_max(delta).mean()))
        pre_counts.append(sum(per_clip_pre)/len(n)/float(mean_remaining.mean()))
        post_counts.append(sum(per_clip_post)/len(n)/float(mean_remaining.mean()))
    speeds, backward_end, terminal_stored = [], [], []
    crop_end, crop_start = 0, 0
    for clip in c.entry['clips']:
        mid = clip['motion_id']
        start, count = int(c.data['length_starts'][mid]), int(c.data['motion_num_frames'][mid])
        path = c.data['gts'][start:start+count, 0, :2]
        vel = (path[1:]-path[:-1])*30
        speeds.append(float(vel[-30:].norm(dim=-1).mean()))
        backward_end.append(float(vel[-1].norm()))
        terminal_stored.append(float(c.data['gvs'][start+count-1, 0, :2].norm()))
        if clip.get('parent_source_frame_stop') is not None and clip.get('parent_source_frames') is not None:
            crop_end += clip['parent_source_frame_stop'] < clip['parent_source_frames']
        crop_start += clip.get('parent_source_frame_start', 0) > 0
    generator = torch.Generator().manual_seed(71)
    frames = low+(torch.rand(len(n), generator=generator)*legal).long()
    absolute = c.data['length_starts'][c.ids]+frames
    source_reset = c.data['gts'][absolute, 0, :2]
    origins = torch.full_like(source_reset, 20.)
    curriculum = AMASSRootTrajectoryCurriculum(c, len(n))
    rng = c.generator.get_state().clone()
    curriculum.bind_reset(torch.arange(len(n)), c.ids.clone(), frames, source_reset, origins, torch.ones(len(n), dtype=torch.long))
    plan = curriculum.build_plan(torch.zeros(len(n), dtype=torch.long))
    assert torch.equal(rng, c.generator.get_state())
    assert torch.equal(plan.native_xy[:, ::3], plan.xy)
    true_frames = (frames[:, None]+torch.arange(151)).minimum(n[:, None]-1)
    true_native = c.data['gts'][c.data['length_starts'][c.ids, None]+true_frames, 0, :2]+curriculum.offset_xy[:, None]
    assert torch.equal(plan.native_xy, true_native)
    steps = torch.arange(151)
    lower, upper = steps.div(3, rounding_mode='floor'), (steps.div(3, rounding_mode='floor')+1).clamp_max(50)
    approximated = torch.lerp(plan.xy[:, lower], plan.xy[:, upper], ((steps%3)/3)[None, :, None])
    error = (approximated-plan.native_xy).norm(dim=-1)
    speed = torch.tensor(speeds)
    ending = torch.tensor(backward_end)
    record = {
        'schema': 'amass_root_reference_cpu_proof.v1', 'status': 'passed',
        'cpu_only': True, 'SDK_started': False, 'GPU_used': False, 'source_bank_modified': False,
        'manifest_sha256': file_sha256(manifest), 'bank_sha256_declared': c.entry['sha256'],
        'bank_sha256_recomputed': False, 'all_manifest_frames_match_actual_bank': True, 'all_motion_dt_30Hz': True,
        'clip_count': len(n), 'sum_frames': int(n.sum()), 'min_frames': int(n.min()), 'max_frames': int(n.max()),
        'median_frames': float(n.float().median()), 'history_frames': 10, 'real_subsequent_frames_reserved': 31,
        'reset_frame_range': '[9,N-32] inclusive', 'minimum_legal_frames_per_clip': int(legal.min()),
        'all2848_retained': True, 'uniform_clip_then_legal_frame_expected_episode_seconds': float(mean_remaining.mean()/30),
        'clips_without_any_full5s_future_after_history': int((n < 160).sum()),
        'tail_analysis': {
            'waypoint_times_s': [.5*i for i in range(1, 11)],
            'initial_ten_waypoint_hold_fraction': initial_hold,
            'initial_ten_waypoint_mean_hold_fraction': sum(initial_hold)/10,
            'all_real_pre_action_phase_hold_fraction': pre_counts,
            'all_real_post_step_nextvalue_phase_hold_fraction': post_counts,
            'aggregation': 'uniform clips, uniform legal reset frames, then transition-weighted over all real episode steps; cap/falls excluded',
            'crop_end_before_parent_end_count': crop_end, 'crop_start_after_parent_start_count': crop_start,
            'last_real_second_backward_difference_speed_mean_mps': float(speed.mean()),
            'last_real_second_backward_difference_speed_median_mps': float(speed.median()),
            'last_real_segment_speed_mean_mps': float(ending.mean()),
            'last_real_segment_speed_gt_point1_fraction': float((ending > .1).float().mean()),
            'last_real_segment_speed_gt_point5_fraction': float((ending > .5).float().mean()),
            'stored_last_frame_velocity_mean_mps': sum(terminal_stored)/len(n),
            'held_future_velocity_mps': 0.,
            'interpretation': 'crop boundaries are not proven physical stops; zero held velocity is an explicit command-only assumption'},
        'source_proof': {
            'seed': 71, 'sampled_actual_clips': len(n), 'canonical_shape': list(plan.xy.shape),
            'native_shape': list(plan.native_xy.shape), 'native_equals_actual_source_frames_plus_fixed_offset': True,
            'canonical_equals_native_every_third': True, 'bind_and_replan_consume_no_corpus_rng': True,
            'frame_sha256': hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
            'source_native_sha256': hashlib.sha256(plan.native_xy.numpy().tobytes()).hexdigest(),
            'all51_native151_plans_finite': bool(torch.isfinite(plan.native_xy).all()),
            '10Hz_linear_interpolation_vs_true30Hz_mean_m': float(error.mean()),
            '10Hz_linear_interpolation_vs_true30Hz_max_m': float(error.max()),
            '10Hz_linear_interpolation_vs_true30Hz_p99_m': float(torch.quantile(error.flatten(), .99)),
            'scope': 'math/source-contract proof, not physical rollout or mocap pose-following'},
        'sources': {str(p.relative_to(root)): file_sha256(p) for p in [
            root/'CrowdSim/humanoid/training/amass_trajectories.py', root/'CrowdSim/humanoid/amass_corpus.py',
            root/'CrowdSim/humanoid/amp_world.py', root/'scripts/train_paper_amp.py']}}
    output = root/'docs/style_improvement_20261002/amass_root_reference_cpu_proof.json'
    if output.exists():
        raise RuntimeError('Do not overwrite a frozen actual-source audit')
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
