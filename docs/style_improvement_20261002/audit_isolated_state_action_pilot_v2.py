"""Read actual restricted CPU checkpoints; never replay physics or CUDA RNG."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'output/isolated_training/mixed_state_action_v1'
CODE = BUNDLE / 'code'
PILOT = ROOT / 'output/diffusion_amp/amp_style_mixed_state_action_isolated_pilot_v2'
OUTPUT = ROOT / 'docs/style_improvement_20261002/state_action_isolated_pilot_v2_review'


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (OUTPUT/'audit.json').exists():
        raise RuntimeError('CPU-only fresh output required')
    OUTPUT.mkdir(exist_ok=True)
    # Only copied owned business modules; actual training import maps remain a
    # separate historical artifact and are never invented by this CPU process.
    sys.path[:0] = [str(CODE), str(CODE/'src')]
    os.environ['PROTOMOTIONS_ASSET_ROOT'] = str(CODE/'protomotions/data/assets')
    import numpy as np
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig

    torch.set_num_threads(4)
    checks, rows = [], []

    def check(name, condition, detail=None):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(name)

    def finite_bundle(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_bundle(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_bundle(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    def same(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    report = {'schema': 'paper_amp.isolated_state_action_actual_pilot_audit.v1', 'status': 'failed',
        'CPU_only': True, 'SDK_started': False, 'GPU_used': False, 'checks': checks, 'checkpoints': rows,
        'heldout_outputs_read': False, 'script': record(__file__)}
    try:
        manifest_path = BUNDLE/'source_manifest_v2.json'
        manifest = json.loads(manifest_path.read_text())
        check('source_v2_identity', record(manifest_path)['sha256'] == '429fdd0c23569400e17584f80d84d993374dac2b57006d5035f7740dc628779c')
        check('launcher_v2_identity', record(BUNDLE/'launch_isolated_v2.py')['sha256'] == '31fd40be8a59f23c19b6c07fc258ebbd8a350402f967bb49dcc46ddfe8ef648d')
        check('full_archive_v2_identity', record(BUNDLE/'source_full_v2.tar.gz')['sha256'] == 'a433396eb28da25781705d520a62252994cefccf5836482eadffc4e1642c215e')
        allowed = {r['path']: r for r in manifest['files']}
        actual = {str(p.relative_to(BUNDLE)) for p in CODE.rglob('*') if p.is_file() and not p.is_symlink()}
        check('exact926_copied_file_set', len(allowed) == 926 and actual == set(allowed))
        for path, row in allowed.items():
            p = BUNDLE/path
            check('immutable_copied:'+path, not p.is_symlink() and p.stat().st_mode & 0o222 == 0
                and p.stat().st_size == row['bytes'] and record(p)['sha256'] == row['sha256'])
        original = json.loads(Path(manifest['source_epoch']['authoritative_registered146_manifest']['path']).read_text())
        check('registered146_old_epoch_preserved', len(original['files']) == 146 and all(
            record(CODE/r['path'])['sha256'] == r['sha256'] for r in original['files']))
        for link in manifest['shared_resource_links']:
            p = BUNDLE/link['path']
            check('shared_link:'+link['path'], p.is_symlink() and str(p.resolve()) == link['target'])
        for resource in manifest['resources']:
            stat = Path(resource['path']).stat()
            check('resource_stat:'+resource['path'], stat.st_size == resource['bytes'] and stat.st_mtime_ns == resource['mtime_ns'])
        check('CPU_actual_admission_classes_are_copied', all(Path(sys.modules[c.__module__].__file__).resolve().is_relative_to(CODE)
            for c in (AMASSCorpus, MixedTrajectoryCurriculum, PaperAMPTrainer, PacerTrajectoryConfig)))
        report['source'] = {'manifest': record(manifest_path), 'archive': record(BUNDLE/'source_full_v2.tar.gz'),
            'launcher': record(BUNDLE/'launch_isolated_v2.py'), 'copied_files_hashed': len(allowed), 'old_registered_source_count': 146,
            'large_shared_resources': 'exact declared size/mtime and symlink targets checked; full bank hashes not repeated'}
        metrics = [json.loads(line) for line in (PILOT/'metrics_step8.jsonl').read_text().splitlines() if line.strip()]
        first_metrics = [json.loads(line) for line in (PILOT/'metrics_step4.jsonl').read_text().splitlines() if line.strip()]
        check('metrics_first_segment_preserved_exact', metrics[:4] == first_metrics and [r['update_count'] for r in metrics] == list(range(1, 9)))
        saved_states = []
        for update, result_name in ((4, 'result_step4.json'), (8, 'result_step8.json')):
            path = PILOT/f'step_{update}.pt'
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            saved_states.append(state)
            result = json.loads((PILOT/result_name).read_text())
            runner, cfg = state['runner_state'], state['config']
            prefix = f'step{update}:'
            check(prefix+'complete_restricted_payload', state['format'] == 'navisaaclab_method_reconstruction_training'
                and state['version'] == PaperAMPTrainer.checkpoint_version and state['physical_runner_state_included'] is True)
            check(prefix+'actual_committed_budget', result['status'] == 'passed' and state['update_count'] == result['update_count'] == update
                and state['physical_transition_count'] == result['physical_transition_count'] == update*256*32 == runner['control_step']*256)
            check(prefix+'state_action_2019_no_public_disc_or_RMS_transfer', cfg['discriminator_input'] == 'state_action'
                and cfg['warm_start_discriminator'] is False and cfg['amp_normalization'] == 'per_history_coordinate'
                and tuple(state['model']['discriminator.0.weight'].shape) == (1024, 2019)
                and state['provenance']['initialization']['published_state_only_discriminator_transferred'] is False
                and state['provenance']['initialization']['published_amp_history_moments_transferred'] is False
                and state['provenance']['initialization']['normalizer_counts_before_cap']['amp_normalizer'] == 0.)
            check(prefix+'profile_matches_real_training_JSON', json.loads(json.dumps(cfg)) == result['training_config']
                and cfg['direction_mode'] == 'backwards_penalty' and cfg['expert_sampling'] is None and 'expert_sampling' not in runner)
            check(prefix+'entire_training_tensor_bundle_finite', finite_bundle(state))
            optimizer_steps = {}
            for name, metric in (('optimizer', 'optimizer_minibatches'), ('discriminator_optimizer', 'discriminator_minibatches')):
                optimizer = state[name]
                ids = [i for group in optimizer['param_groups'] for i in group['params']]
                steps = {float(s['step']) for s in optimizer['state'].values()}
                check(prefix+name+'_complete_real_Adam_steps', len(ids) == len(set(ids)) and set(ids) == set(optimizer['state'])
                    and len(steps) == 1 and next(iter(steps)) == sum(r[metric] for r in metrics[:update]))
                check(prefix+name+'_nonzero_moments', any(bool((s['exp_avg'] != 0).any()) for s in optimizer['state'].values()))
                optimizer_steps[name] = next(iter(steps))
            loaded, loaded_runner = PaperAMPTrainer.load(path, device='cpu', restore_rng=False)
            check(prefix+'actual_old_epoch_loader_model_Adam_RMS_exact', same(loaded.model.state_dict(), state['model'])
                and same(loaded.optimizer.state_dict(), state['optimizer'])
                and same(loaded.discriminator_optimizer.state_dict(), state['discriminator_optimizer'])
                and all(same(getattr(loaded, name).state_dict(), value) for name, value in state['normalizers'].items())
                and same(loaded_runner, runner) and same(loaded._pending_rng_state, state['rng']))
            check(prefix+'loader_replay_exact', same(loaded.policy_replay.state_dict(), state['policy_replay']))
            del loaded, loaded_runner
            n = 256
            physical_shapes = {'root_pos': (n, 3), 'root_rot': (n, 4), 'root_vel': (n, 3), 'root_ang_vel': (n, 3), 'dof_pos': (n, 69), 'dof_vel': (n, 69)}
            check(prefix+'common_physical_XYZW_history_episode_shapes', set(runner['physical']) == set(physical_shapes)
                and all(tuple(runner['physical'][k].shape) == s for k, s in physical_shapes.items())
                and torch.allclose(runner['physical']['root_rot'].norm(dim=-1), torch.ones(n), atol=1e-4, rtol=0)
                and tuple(runner['amp_history'].shape) == (n, 10, 195)
                and torch.equal(runner['amp_history_episode_ids'], runner['episode_ids'][:, None].expand(n, 10)))
            expected_counts = {'self_normalizer': 200000+update*8192, 'task_normalizer': 200000+update*8192,
                'value_normalizer': 200000+update*8192, 'amp_normalizer': update*(8192+4096)}
            for name, count in expected_counts.items():
                norm = state['normalizers'][name]
                check(prefix+'RMS_committed_count:'+name, float(norm['count']) == count
                    and bool((norm['variance'] >= 0).all()))
            replay = state['policy_replay']
            mask = torch.tensor(cfg['action_mask'])
            check(prefix+'replay_accepted_and_57_effective_commands', replay['accepted'] == update*8192
                and replay['size'] == min(replay['accepted'], replay['capacity']) and replay['cursor'] == replay['accepted'] % replay['capacity']
                and replay['history'].shape == (replay['size'], 10, 195) and replay['actions'].shape == (replay['size'], 69)
                and mask.count_nonzero() == 57 and not bool((replay['actions'][:, mask == 0] != 0).any()))
            rng = state['rng']
            random.Random().setstate(rng['python'])
            ns = rng['numpy']
            np.random.RandomState().set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            torch.Generator(device='cpu').set_state(rng['torch_cpu'])
            check(prefix+'CPU_Python_Numpy_RNG_accepted_CUDA_bytes_retained', len(rng['torch_cuda']) == 1
                and all(x.dtype == torch.uint8 and x.ndim == 1 and x.numel() > 0 for x in rng['torch_cuda']))
            corpus = AMASSCorpus(result['arguments']['manifest'], 'train', device='cpu', seed=7, dof_velocity_frame='child_joint')
            corpus.generator.set_state(runner['corpus_rng'])
            curriculum = MixedTrajectoryCurriculum(corpus, n, device='cpu', bounds_xy=(0., 0., 192., 192.), tail_policy='linear_velocity',
                procedural_config=PacerTrajectoryConfig(mode='smooth', speed_max_mps=2., scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5), seed=7)
            curriculum.load_state_dict(runner['curriculum_state'])
            ids = torch.arange(n)
            origins = torch.stack((48+(ids//16)*4, 48+(ids % 16)*4), -1).float()
            curriculum.validate_runner_state(runner['age'], runner['plan_age'], runner['plans'], runner['reference_ids'], runner['episode_ids'], replan_steps=15, origins_xy=origins)
            check(prefix+'actual_v7_saved_phase_native151_canonical51_restore', True)
            stats = dict(curriculum.statistics)
            check(prefix+'exact_50percent_group_exposure_cumulative', stats == result['group_transition_exposure']
                and stats['control_steps'] == update*32 and stats['amass_actor_transitions'] == stats['procedural_actor_transitions'] == update*4096)
            check(prefix+'child_private_rng_valid_AND_no_alias', corpus.generator is not curriculum.amass.corpus.generator
                and corpus.generator is not curriculum.procedural.generator and curriculum.amass.corpus.generator is not curriculum.procedural.generator)
            check(prefix+'uniform_expert_and_dataset_identity', state['provenance']['dataset']['manifest_sha256'] == record(corpus.manifest_path)['sha256']
                and state['provenance']['dataset']['source_sha256'] == corpus.entry['sha256'])
            check(prefix+'all_nested_curriculum_state_exact_after_restore', same(curriculum.state_dict(), runner['curriculum_state']))
            global_rng = torch.get_rng_state().clone()
            order = torch.tensor([255, 1, 128, 0])

            def next_draws(sampling_curriculum, sampling_corpus):
                sample = sampling_curriculum.sample_reference(order, history_frames=10)
                experts = sampling_corpus.sample_reference(6, history_frames=10)
                from types import SimpleNamespace
                plan = sampling_curriculum.build_plan(SimpleNamespace(**runner['physical']), runner['age'], order, runner['episode_ids'])
                return ({'reference': vars(sample[0]), 'history': vars(sample[1]), 'next': sample[2], 'ids': sample[3], 'frames': sample[4]},
                    {'reference': vars(experts[0]), 'history': vars(experts[1]), 'next': experts[2], 'ids': experts[3], 'frames': experts[4]},
                    {'xy': plan.xy, 'native_xy': plan.native_xy}, sampling_corpus.generator.get_state(),
                    sampling_curriculum.amass.corpus.generator.get_state(), sampling_curriculum.procedural.generator.get_state())

            continuation_a = next_draws(curriculum, corpus)
            corpus.generator.set_state(runner['corpus_rng'])
            curriculum.load_state_dict(runner['curriculum_state'])
            continuation_b = next_draws(curriculum, corpus)
            check(prefix+'real_v7_next_reset_expert_procedural_plan_three_rng_repeat', same(continuation_a, continuation_b)
                and torch.equal(global_rng, torch.get_rng_state()))
            corpus.generator.set_state(runner['corpus_rng'])
            curriculum.load_state_dict(runner['curriculum_state'])
            rows.append({'checkpoint': record(path), 'result': record(PILOT/result_name), 'updates': update, 'transitions': update*8192,
                'control_steps_cumulative': runner['control_step'], 'group_exposure': stats, 'optimizer_steps': optimizer_steps,
                'RMS_counts': expected_counts, 'segment_falls': result['fall_resets'], 'segment_time_limits': result['time_limit_resets'],
                'replay': {k: replay[k] for k in ('size','capacity','accepted','cursor')},
                'RNG_tensor_SHA256': {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in {
                    'torch_CPU': rng['torch_cpu'], 'torch_CUDA_serialized': rng['torch_cuda'][0], 'corpus': runner['corpus_rng'],
                    'AMASS_private': runner['curriculum_state']['amass_state']['generator_state'], 'procedural_private': runner['curriculum_state']['procedural_state']['generator_state']}.items()}})
            del corpus, curriculum
        a, b = saved_states
        check('resume8_names_exact_sealed4_source', json.loads((PILOT/'result_step8.json').read_text())['arguments']['resume'] == str((PILOT/'step_4.pt').resolve()))
        check('both_optimizers_are_continued_not_reinitialized', rows[1]['optimizer_steps']['optimizer'] > rows[0]['optimizer_steps']['optimizer']
            and rows[1]['optimizer_steps']['discriminator_optimizer'] - rows[0]['optimizer_steps']['discriminator_optimizer'] == 8)
        check('resume_retains_real_training_provenance', same(a['provenance'], b['provenance']))
        check('D_and_policy_genuinely_updated_during_resume', not torch.equal(a['model']['discriminator.0.weight'], b['model']['discriminator.0.weight'])
            and not torch.equal(a['model']['actor.4.weight'], b['model']['actor.4.weight']))
        missing = [name for name in ('pilot4_v2_GPU_imports.json','resume4to8_v2_GPU_imports.json') if not (BUNDLE/name).exists()]
        report['actual_GPU_import_reports'] = {'missing': missing, 'full_actual_import_mapping_verified': not missing,
            'CPU59_import_probe_is_GPU_mapping': False,
            'guard_source_and_command_verified': True, 'guard_limit':'Actual command admits only copied owned imports, but no retrospective module-by-module GPU attestation is fabricated.'}
        check('reviewer_no_CUDA_or_Kit_initialized', not torch.cuda.is_initialized()
            and not any(name.split('.')[0] in {'isaacsim','isaaclab','omni','pxr'} for name in sys.modules))
        report['status'] = 'checkpoint_resume_and_isolated_source_passed_actual_GPU_import_report_missing' if missing else 'passed'
        report['limitations'] = ['No physical replay or bitwise PhysX resume equivalence is claimed: solver/contact caches are not serialized.',
            'RNG tensors and all CPU streams are validated; CUDA RNG bytes are retained but not restored/executed by CPU reviewer.',
            'Pilot falls are segment counters: first4=1, resumed4=0, total two observed segments=1; source groups/counters are cumulative.',
            'Copied source hashes and guard behavior are verified; missing historical full GPU import sidecars require a future before-close attestation fix, not fabricated pilot logs.',
            'No navigation, full22 validation, convergence or author-paper effect follows from8 updates/65536 transitions.']
    finally:
        report['check_count'] = len(checks)
        with (OUTPUT/'audit.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'checkpoints':rows}))


if __name__ == '__main__':
    run()
