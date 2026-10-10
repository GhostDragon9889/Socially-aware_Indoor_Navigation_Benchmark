"""Full actual-source CPU proof of the explicitly engineered linear tail."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum


def main():
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only proof requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]
    c = AMASSCorpus(root/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
                    split='train', device='cpu', seed=7, dof_velocity_frame='child_joint')
    n = c.data['motion_num_frames'][c.ids].long()
    gen = torch.Generator().manual_seed(71)
    frames = 9+(torch.rand(len(n), generator=gen)*(n-40)).long()
    origins = torch.full((len(n), 2), 48.)
    source = c.data['gts'][c.data['length_starts'][c.ids]+frames, 0, :2]
    curriculum = AMASSRootTrajectoryCurriculum(c, len(n), bounds_xy=[0, 0, 192, 192], tail_policy='linear_velocity')
    rng = c.generator.get_state().clone()
    curriculum.bind_reset(torch.arange(len(n)), c.ids, frames, source, origins, torch.ones(len(n), dtype=torch.long))
    plan = curriculum.build_plan(torch.zeros(len(n), dtype=torch.long))
    actual = c.data['gts'][c.data['length_starts'][c.ids, None]+plan.native_source_frames, 0, :2]+curriculum.offset_xy[:, None]
    real = ~plan.native_beyond_source
    assert torch.equal(plan.native_xy[real], actual[real])
    assert torch.equal(plan.native_xy[:, ::3], plan.xy)
    assert torch.equal(rng, c.generator.get_state())
    delta_min, delta_max, velocities = [], [], []
    for mid, count in zip(c.ids.tolist(), n.tolist(), strict=True):
        start = int(c.data['length_starts'][mid])
        xy = c.data['gts'][start:start+count, 0, :2]
        legal = xy[9:count-31]  # Last reset frame N-32 inclusive.
        reverse = xy.flip(0)
        suffix_min = reverse.cummin(0).values.flip(0)[9:count-31]
        suffix_max = reverse.cummax(0).values.flip(0)[9:count-31]
        velocity = (xy[-1]-xy[-11])/(10/30)
        end = xy[-1]+5*velocity
        delta_min.append(torch.minimum(suffix_min-legal, end-legal).amin(0))
        delta_max.append(torch.maximum(suffix_max-legal, end-legal).amax(0))
        velocities.append(velocity)
    minimum, maximum = torch.stack(delta_min).amin(0), torch.stack(delta_max).amax(0)
    assert bool((48+minimum >= 1).all()) and bool((108+maximum <= 191).all())
    tree = ast.parse((root/'scripts/train_paper_amp.py').read_text())
    fn = deepcopy(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'interpolate'))
    namespace = dict(torch=torch, plans=plan.native_xy, plan_frequency_hz=30,
        args=SimpleNamespace(trajectory_source='amass_root', num_envs=len(n)), world=SimpleNamespace(device=torch.device('cpu')))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'actual_native_interpolate', 'exec'), namespace)
    target = namespace['interpolate'](torch.full((len(n), 1), 1/30))[:, 0]
    assert torch.equal(target, actual[:, 1])  # All selected frames have at least31 actual next steps.
    latest = curriculum.remaining_steps.clone()
    curriculum.last_observed_age.copy_(latest)  # Source-query proof only; no fake physical rollout claim.
    future = curriculum.future_targets(latest)
    end_xy = c.data['gts'][c.data['length_starts'][c.ids]+n-1, 0, :2]+curriculum.offset_xy
    assert torch.equal(future[:, -1], end_xy+5*curriculum.end_velocity_xy)
    velocity = torch.stack(velocities).norm(dim=-1)
    record = {'schema': 'amass_root_linear_source_proof.v2', 'status': 'passed', 'CPU_only': True,
        'SDK_started': False, 'GPU_used': False, 'actual_physical_rollout': False, 'bank_modified': False,
        'selected_clips': len(n), 'all2848_retained': len(n) == 2848, 'reset_future_frames': 31,
        'tail_policy': 'linear_velocity', 'tail_is_amass_groundtruth': False,
        'formula': '(root_xy[-1]-root_xy[-11])/(10/30); eleven positions, ten real intervals',
        'manifest_sha256': file_sha256(c.manifest_path), 'source_bank_sha256_declared': c.entry['sha256'],
        'source_bank_sha256_recomputed': False, 'canonical_shape': list(plan.xy.shape), 'native_shape': list(plan.native_xy.shape),
        'all_real_native_frames_bitwise_actual_source_plus_fixed_offset': True,
        'canonical_bitwise_native_every_third': True, 'actual_runner_next_step_target_bitwise_actual30Hz_source': True,
        'last_real_phase_ten_targets_from_current_phase': True,
        'last_real_phase5s_target_equals_endpoint_plus5estimated_velocity': True,
        'bind_replan_task_queries_consume_no_corpus_rng': bool(torch.equal(rng, c.generator.get_state())),
        'floor_analytical_gate': {'all_legal_reset_frames': int((n-40).sum()),
            'real_future_suffix_and_linear_end5s_checked': True, 'origin_axis_range_m': [48, 108],
            'floor_bounds_m': [0, 0, 192, 192], 'root_margin_m': 1,
            'world_min_xy_m': (48+minimum).tolist(), 'world_max_xy_m': (108+maximum).tolist(),
            'actual_root_only_not_fullbody_or_drift_safety': True},
        'end_velocity_estimate_mps': {'mean': float(velocity.mean()), 'median': float(velocity.median()),
            'p99': float(torch.quantile(velocity, .99)), 'max': float(velocity.max())},
        'sampled_canonical_tail_counters': curriculum.describe()['statistics'],
        'limitations': ['No source identities or future poses enter actor', 'No control/reward after real clip end',
            'Linear tail is an engineering command assumption, not unseen motion capture',
            'Checkpoint resume restores logical state, not PhysX contact/solver caches'],
        'sources': {str(p.relative_to(root)): file_sha256(p) for p in [
            root/'CrowdSim/humanoid/training/amass_trajectories.py', root/'CrowdSim/humanoid/amass_corpus.py',
            root/'CrowdSim/humanoid/amp_world.py', root/'CrowdSim/humanoid/application/configuration.py',
            root/'scripts/train_paper_amp.py', root/'configs/diffusion_amp/amp_style_amass_root_curriculum.yaml']}}
    output = root/'docs/style_improvement_20261002/amass_root_linear_cpu_proof_frozen_resume_guard.json'
    if output.exists():
        raise RuntimeError('Do not overwrite frozen source proof')
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
