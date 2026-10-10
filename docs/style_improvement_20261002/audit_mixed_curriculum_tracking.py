"""CPU-only full22 mixed500/1000/2000 validation; never reads test outputs."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

import audit_amass_root_tracking as paired
import audit_amass_root_tracking_lineage as lineage


SOURCE_MANIFEST_SHA = '8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e'
SOURCE_ARCHIVE_SHA = '5692f2cdc4aed9052a9f3c363b37ba3d8b13167a27c034f78c6486ae56b2ce4f'
WATCH_LAUNCH_SHA = 'ab6a52b85356d3fa83fbdb252976b06feac4814d39c2501b15e4a65fafd268e4'
FULL_PREFLIGHT_SHA = 'e3b2b86a6459691dd6586b92371d0978dae0f055e8c5522ad2774db368652a73'
VALIDATION_PLAN_SHA = '12eb7de873c91186cd24e2b80b0d6bc3cacdd5c41c2cedb41698e52358d6c3f2'
LOAD_SOURCES = (
    'CrowdSim/humanoid/amass_corpus.py', 'CrowdSim/humanoid/training/amass_trajectories.py',
    'CrowdSim/humanoid/training/mixed_trajectories.py', 'CrowdSim/humanoid/training/trajectories.py',
    'CrowdSim/humanoid/amp_world.py', 'CrowdSim/humanoid/application/configuration.py',
    'scripts/train_paper_amp.py', 'configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml',
)


def admitted_budget(step):
    paired.require(type(step) is int and step in (500, 1000, 2000), 'Only the declared mixed500/1000/2000 snapshots are admitted')
    return {'updates': step, 'control_steps': step*32, 'total_transitions': step*256*32,
            'amass_transitions': step*128*32, 'procedural_transitions': step*128*32,
            'baseline_updates': 1000, 'baseline_total_transitions': 8192000,
            'equal_total_budget_to_baseline': step == 1000,
            'each_group_has_baseline_total_exposure': step == 2000,
            'sampling_fraction_is_transition_fraction_not_reset_probability': True}


def normalized_config(config):
    result = dict(config)
    result.setdefault('expert_sampling', None)  # One explicitly disabled legacy field.
    if isinstance(result.get('action_mask'), tuple):
        result['action_mask'] = list(result['action_mask'])
    return result


def admitted_validation_sources(result):
    rows = paired.require_full_validation(result)
    document = paired.read(result['arguments']['manifest'])
    selected = document['splits']['validation']['clips']
    expected = {row['motion_id']: row for row in selected}
    paired.require(len(expected) == 22 and rows.keys() == expected.keys(), 'Full validation IDs differ from actual declared manifest')
    for mid, row in rows.items():
        source = expected[mid]
        window = row['tracking_window']
        paired.require(row['source_path'] == source['source_path'] and window['initial_frame'] == 9
                       and window['last_frame'] == source['frames']-1
                       and row['expected_control_steps'] == source['frames']-10,
                       'Reference window is not the full actual manifest clip')
    return rows


def raw_tracking_rule(capture, row, threshold):
    metric = capture['metric_frame']
    error = np.linalg.norm(capture['rigid_body_pos'].astype(np.float64)
                           - capture['reference_rigid_body_pos'].astype(np.float64), axis=-1).mean((-1, -2))
    failing = np.flatnonzero(metric & (error > threshold))
    paired.require(bool(len(failing)) == row['tracking_failed']
                   and (int(failing[0]) if len(failing) else None) == row['first_tracking_failure_capture_frame'],
                   'Raw mean-body threshold / first failure differs')
    paired.require(row['tracking_success'] == (row['complete_clip'] and not len(failing)), 'Raw full-sequence success rule differs')
    paired.require(capture['reference_frame'][0] == row['tracking_window']['initial_frame']
                   and capture['reference_frame'][-1] == row['tracking_window']['last_frame'], 'Captured full real reference endpoints differ')
    return {'raw_failure_threshold_recomputed': True, 'failure_capture_frames': failing.tolist()}


def admit_seal(document, checkpoint, step, launch):
    """Accept an immutable row or an explicit snapshot ledger; never mutable last."""
    if isinstance(document.get('snapshots'), list):
        selected = [row for row in document['snapshots'] if row.get('update_count') == step]
        paired.require(len(selected) == 1, 'Selected sealed snapshot missing or duplicated')
        row = selected[0]
    elif isinstance(document.get('checkpoint'), dict):
        row = document['checkpoint']
    else:
        row = document
    paired.require(isinstance(row, dict) and row.get('update_count') == step, 'Seal update counter differs')
    paired.require(Path(row['path']).resolve() == Path(checkpoint['path']).resolve()
                   and row['bytes'] == checkpoint['bytes'] and row['sha256'] == checkpoint['sha256'],
                   'Actual evaluated checkpoint differs from its immutable seal')
    paired.require(Path(checkpoint['path']).name == f'step_{step}.pt', 'A named immutable step snapshot is required; mutable last.pt is forbidden')
    expected = admitted_budget(step)
    paired.require(row.get('physical_transition_count') == expected['total_transitions']
                   and row.get('identical_source_bytes_verified') is True, 'Seal counter/byte-transport identity differs')
    paired.require(launch['source_manifest_sha256'] == SOURCE_MANIFEST_SHA and launch['targets'] == [500, 1000, 2000]
                   and launch['full_weight_semantics_checked_by_watch'] is False, 'Transport launch is not the declared frozen135 mixed source')
    paired.require(Path(checkpoint['path']).resolve().parent == (paired.ROOT / launch['snapshot_dir']).resolve(), 'Seal snapshot directory differs from registered watcher')
    if 'watcher_sha256' in document:
        paired.require(document['watcher_sha256'] == launch['watcher_sha256'], 'Ledger watcher code differs from registered launch')
    return dict(row)


def independent_checkpoint_admission(document, checkpoint):
    paired.require(document['status'] == 'passed' and document['CPU_only'] is True
                   and document['GPU_used'] is False and document['SDK_started'] is False,
                   'Full independent CPU payload admission is required before evaluation')
    paired.require(document['schema'] in ('mixed_sealed_milestone_payload_admission.v1',
                   'navisaaclab.mixed_actual_checkpoint_audit.v1'), 'Unknown independent payload audit schema')
    checks = document['checks']
    paired.require(isinstance(checks, list) and checks and document['check_count'] == len(checks)
                   and all(row['passed'] is True for row in checks), 'Independent audit contains missing or failed checks')
    if 'checkpoint' in document:
        records = [document['checkpoint']]
    else:
        records = document['checkpoints']
        records = list(records.values()) if isinstance(records, dict) else records
        records = [row.get('checkpoint', row) for row in records]
    matches = [row for row in records if isinstance(row, dict) and row.get('sha256') == checkpoint['sha256']
               and row.get('bytes') == checkpoint['bytes'] and Path(row.get('path', '')).resolve() == Path(checkpoint['path']).resolve()]
    paired.require(len(matches) == 1, 'Independent full CPU audit does not uniquely admit this actual checkpoint')
    paired.require(document['source_manifest']['sha256'] == SOURCE_MANIFEST_SHA
                   and document['source_archive']['sha256'] == SOURCE_ARCHIVE_SHA, 'Independent payload audit uses different source')
    return {'matching_checkpoint': matches[0], 'independent_CPU_check_count': document['check_count']}


def registered_validation_admission(document, result, step):
    """The registered validation plan is evidence, not a new blind test fold."""
    paired.require(document['status'] == 'registered_before_any_mixed_GPU_evaluation'
                   and document['training_source_manifest_sha256'] == SOURCE_MANIFEST_SHA
                   and document['snapshot_watch_registration_sha256'] == WATCH_LAUNCH_SHA
                   and document['milestones'] == [500, 1000, 2000], 'Mixed validation registration differs')
    budget = admitted_budget(step)
    paired.require(document['budgets'][str(step)] == {
        'total_physical_transitions': budget['total_transitions'],
        'each_branch_physical_transitions': budget['amass_transitions']}, 'Registered group/total budget differs')
    tracking, arguments = document['tracking'], result['arguments']
    paired.require(tracking['split'] == arguments['split'] == 'validation'
                   and tracking['complete_motion_count'] == 22 and tracking['all_post_physics_frames'] == 11982
                   and tracking['frame_cap'] is None and arguments['max_steps'] is None
                   and tracking['motion_subset'] is None and arguments['motion_ids'] is None and arguments['max_clips'] is None
                   and tracking['seed'] == arguments['seed'] == 7
                   and tracking['cpu_threads'] == arguments['cpu_threads'] == 4
                   and tracking['deterministic'] is arguments['deterministic_evaluation'] is True
                   and tracking['failure_threshold_m'] == arguments['failure_threshold_m'] == .5,
                   'Actual evaluation differs from the registered complete22 protocol')
    paired.require(result['selected_source_sha256'] == document['actual_runtime_source_sha256']
                   and document['actual_runtime_source_sha256']['scripts/benchmark_paper_motion_tracking.py']
                   == tracking['benchmark_sha256'], 'Actual captured evaluation runtime source differs from registration')
    history = document['test_history']
    paired.require(history['full34_already_opened_in_previous_root_candidate_stage'] is True
                   and history['not_a_new_blind_test_fold'] is True and history['not_for_recipe_tuning'] is True,
                   'Previously opened test history must remain explicit')
    return {'registered_milestone': step, 'registered_actual_runtime_sources': document['actual_runtime_source_sha256'],
            'previous_full34_already_opened': True, 'new_blind_test_claim': False}


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def admit_saved_mixed(state, baseline_training, step):
    """Actual saved payload, including nested source state, determines exposure."""
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig
    budget = admitted_budget(step)
    config = normalized_config(state['config'])
    paired.require(config == normalized_config(baseline_training['training_config'])
                   and config['expert_sampling'] is None, 'Mixed learned config differs from negative baseline')
    paired.require(state['format'] == 'navisaaclab_method_reconstruction_training'
                   and state['physical_runner_state_included'] is True, 'Incomplete training checkpoint format')
    paired.require(type(state['update_count']) is int and state['update_count'] == step
                   and type(state['physical_transition_count']) is int
                   and state['physical_transition_count'] == budget['total_transitions'], 'Saved actual training budget differs')
    paired.require(all(key in state for key in ('model', 'optimizer', 'discriminator_optimizer', 'normalizers', 'rng', 'policy_replay')),
                   'Complete model/optimizer/replay/RNG resume sections are required')
    paired.require(set(('python', 'numpy', 'torch_cpu', 'torch_cuda')) <= set(state['rng']), 'Incomplete global RNG state')
    provenance = state['provenance']
    paired.require(provenance['trajectory_source'] == 'mixed_amass_procedural'
                   and provenance['diffusion_used_for_training'] is False, 'Snapshot is not the declared mixed source')
    contract = provenance['mixed_command_contract']
    paired.require(contract['assignment'] == 'first_half_amass_second_half_procedural'
                   and contract['actual_transition_fraction'] == .5
                   and contract['future_3D_poses_enter_actor'] is False and contract['branch_ID_enters_actor'] is False,
                   'Unexpected mixed actor/assignment contract')
    expected_initialization, _ = lineage.initialization_record(baseline_training)
    paired.require(provenance['initialization'] == expected_initialization, 'Public transfer initialization differs from baseline')
    runner = state['runner_state']
    paired.require(runner['control_step'] == budget['control_steps'] and runner['runner_contract'] == {
        'replan_steps': 15, 'episode_steps': 300, 'planner_batch_size': 64,
        'trajectory_source': 'mixed_amass_procedural', 'planning_clock': 'synchronized'}, 'Saved mixed runner clock/config differs')
    paired.require('expert_sampling' not in runner, 'Uniform baseline expert sampling must remain disabled')
    dataset = provenance['dataset']
    corpus = AMASSCorpus(dataset['manifest'], 'train', device='cpu', seed=7,
                         dof_velocity_frame=config['expert_dof_velocity_frame'])
    paired.require(corpus.provenance() == dataset, 'Actual loaded training corpus provenance differs')
    generator = runner['corpus_rng']
    paired.require(isinstance(generator, torch.Tensor) and generator.dtype == torch.uint8
                   and generator.device.type == 'cpu' and generator.shape == corpus.generator.get_state().shape,
                   'Original corpus CPU RNG buffer is incomplete')
    corpus.generator.set_state(generator)
    curriculum = MixedTrajectoryCurriculum(corpus, 256, bounds_xy=(0., 0., 192., 192.),
        tail_policy='linear_velocity', procedural_config=PacerTrajectoryConfig(mode='smooth',
        speed_max_mps=2., scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5), seed=7)
    driver = load_file(paired.ROOT / 'scripts/train_paper_amp.py', 'mixed_snapshot_cpu_driver')
    curriculum.load_state_dict(driver.curriculum_rng_to_cpu(runner['curriculum_state'], mixed=True))
    ids = torch.arange(256)
    origins = torch.stack((48+(ids//16)*4, 48+(ids % 16)*4), -1).float()
    curriculum.validate_runner_state(runner['age'], runner['plan_age'], runner['plans'], runner['reference_ids'],
        runner['episode_ids'], replan_steps=15, origins_xy=origins)
    driver.validate_mixed_transition_exposure(curriculum, state['physical_transition_count'], runner['control_step'], 256)
    history_episodes = runner['amp_history_episode_ids']
    paired.require(history_episodes.dtype == torch.long and history_episodes.shape == (256, 10)
                   and torch.equal(history_episodes, runner['episode_ids'][:, None].expand_as(history_episodes)),
                   'AMP newest-first history crosses an episode reset boundary')
    paired.require(curriculum.describe()['config'] == provenance['curriculum']['config'], 'Saved source configuration differs from immutable initialization provenance')
    return {'budget': budget, 'statistics': curriculum.describe()['statistics'],
            'amass_statistics': curriculum.describe()['amass']['statistics'],
            'saved_full_mixed_schema_and_child_RNG_checked': True, 'active_native_and_canonical_plans_match_actual_source': True,
            'reset_episode_history_boundary_checked': True, 'physical_solver_cache_bitwise_resume_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-result', type=Path, required=True)
    parser.add_argument('--candidate-step', type=int, choices=(500, 1000, 2000), required=True)
    parser.add_argument('--checkpoint-seal', type=Path, required=True)
    parser.add_argument('--snapshot-watch-launch', type=Path, default=paired.DOCS / 'mixed_curriculum_snapshot_watch_launch_v1.json')
    parser.add_argument('--validation-plan', type=Path, default=paired.DOCS / 'mixed_curriculum_validation_plan_v1.json')
    parser.add_argument('--independent-payload-audit', type=Path, required=True)
    parser.add_argument('--baseline-result', type=Path, default=paired.RUNS / 'motion_tracking_amass2848_backward_penalty_full_validation_v1/result.json')
    parser.add_argument('--baseline-training-result', type=Path, default=paired.RUNS / 'amp_style_amass2848_backward_penalty_v1/result.json')
    parser.add_argument('--candidate-source', type=Path, default=paired.DOCS / 'mixed_curriculum_training_source_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paired.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    paired.require(not args.output.exists(), 'Evidence output must be a fresh directory')
    results = {'baseline': paired.read(args.baseline_result), 'mixed': paired.read(args.candidate_result)}
    rows = {name: admitted_validation_sources(result) for name, result in results.items()}
    paired.require(paired.sha256(args.validation_plan) == VALIDATION_PLAN_SHA, 'Before-evaluation validation registration changed')
    registration = registered_validation_admission(paired.read(args.validation_plan), results['mixed'], args.candidate_step)
    paired.require(rows['baseline'].keys() == rows['mixed'].keys(), 'All22 motion IDs must pair')
    for key in ('protocol', 'physics', 'layout', 'reference_velocity_contract', 'runtime_numerics'):
        paired.require(results['baseline'][key] == results['mixed'][key], f'Frozen evaluation contract differs:{key}')
    paired.require(normalized_config(results['baseline']['training_config']) == normalized_config(results['mixed']['training_config'])
                   and results['mixed']['training_config'].get('expert_sampling') is None,
                   'Evaluation learning config differs beyond the explicitly disabled legacy expert_sampling field')
    flag_diff = paired.differences(results['baseline']['arguments'], results['mixed']['arguments'])
    paired.require(set(flag_diff) <= {'checkpoint', 'checkpoint_sha256', 'output'}, 'Evaluation flags changed beyond checkpoint/output')
    archives, entries = {}, {}
    for name, path in (('baseline', paired.DOCS / 'backward_penalty_training_source.json'), ('mixed', args.candidate_source)):
        archives[name], entries[name] = paired.verify_archive(path)
        for relative, digest in results[name]['selected_source_sha256'].items():
            paired.require(relative in entries[name] and entries[name][relative]['sha256'] == digest, 'Captured runtime source differs from immutable archive')
    paired.require(archives['mixed']['manifest']['sha256'] == SOURCE_MANIFEST_SHA
                   and archives['mixed']['archive']['sha256'] == SOURCE_ARCHIVE_SHA
                   and archives['mixed']['verified_regular_members'] == 135, 'Mixed frozen135 source identity differs')
    for relative in LOAD_SOURCES:
        paired.require(paired.sha256(paired.ROOT / relative) == entries['mixed'][relative]['sha256'], f'Current CPU admission implementation differs from frozen source:{relative}')
    checkpoints = {name: paired.verified_checkpoint(result) for name, result in results.items()}
    paired.require(paired.sha256(args.snapshot_watch_launch) == WATCH_LAUNCH_SHA, 'Registered before-start watcher launch changed')
    launch = paired.read(args.snapshot_watch_launch)
    preflight_path = paired.ROOT / launch['full_preflight']
    paired.require(paired.sha256(preflight_path) == FULL_PREFLIGHT_SHA == launch['full_preflight_sha256'], 'Actual full-run argv/preflight source changed')
    preflight = paired.read(preflight_path)
    paired.require(preflight['fresh_public_initialization_not_pilot_resume'] is True
                   and preflight['total_planned_transitions'] == 16384000 and preflight['each_group_planned_transitions'] == 8192000,
                   'Full mixed run is not the registered fresh2000 recipe')
    seal = admit_seal(paired.read(args.checkpoint_seal), checkpoints['mixed'], args.candidate_step, launch)
    independent = independent_checkpoint_admission(paired.read(args.independent_payload_audit), checkpoints['mixed'])
    import torch
    baseline_training = paired.read(args.baseline_training_result)
    state = torch.load(checkpoints['mixed']['path'], weights_only=True, map_location='cpu', mmap=True)
    paired.require(state['provenance'] == results['mixed']['controller_provenance'], 'Evaluation actor provenance differs from actual sealed payload')
    admission = admit_saved_mixed(state, baseline_training, args.candidate_step)
    benchmark = load_file(paired.ROOT / 'scripts/benchmark_paper_motion_tracking.py', 'mixed_frozen_benchmark_gate')
    parent_gate = benchmark.verify_actor_training_split(state['provenance'], results['mixed']['arguments']['manifest'], 'validation')
    paired.require(parent_gate == results['mixed']['actor_training_heldout_parent_gate'], 'Actual parent-normalized source admission differs')
    del state
    decomposition = paired.helper()
    decomposed = {name: decomposition.decompose(path.resolve().parent)
                  for name, path in (('baseline', args.baseline_result), ('mixed', args.candidate_result))}
    parents = decomposition.parent_indices(paired.ROOT / 'protomotions/data/assets/mjcf/smpl_humanoid.xml')
    pairs = []
    for mid, old in rows['baseline'].items():
        new = rows['mixed'][mid]
        paired.require(all(old[key] == new[key] for key in ('source_path', 'tracking_window', 'observed_metric_steps', 'initial_translation_xyz_m')),
                       'Frozen source/window/initial translation differs')
        with np.load(old['capture'], allow_pickle=False) as left, np.load(new['capture'], allow_pickle=False) as right:
            exact = paired.exact_capture_pair(left, right)
            raw = {name: raw_tracking_rule(capture, row, results[name]['arguments']['failure_threshold_m'])
                   for name, capture, row in (('baseline', left, old), ('mixed', right, new))}
        pairs.append({'motion_id': mid, **exact, 'independent_tracking_rule': raw, 'fall_diagnostics': {
            name: paired.capture_diagnostics(row['capture'], row, decomposition, parents)
            for name, row in (('baseline', old), ('mixed', new))}})
    success = {name: {mid for mid, row in values.items() if row['tracking_success']} for name, values in rows.items()}
    output = {'schema': 'fixed_slots_mixed.full22_validation_comparison.v1', 'status': 'passed', 'CPU_only': True,
        'Kit_started': False, 'GPU_used': False, 'heldout_test_outputs_read': False, 'style_metrics_recomputed': False,
        'script': paired.record(__file__), 'results': {'baseline': paired.record(args.baseline_result), 'mixed': paired.record(args.candidate_result)},
        'checkpoints': checkpoints, 'checkpoint_seal': paired.record(args.checkpoint_seal), 'sealed_row': seal,
        'watch_launch': paired.record(args.snapshot_watch_launch), 'actual_full_preflight': paired.record(preflight_path),
        'registered_validation_plan': paired.record(args.validation_plan), 'validation_registration_admission': registration,
        'independent_payload_admission': independent, 'independent_payload_audit': paired.record(args.independent_payload_audit),
        'watcher_gate_is_transport_and_scalar_counters_only': True,
        'source_archives': archives, 'current_cpu_admission_sources_verified': list(LOAD_SOURCES),
        'saved_mixed_admission': admission, 'parent_gate': parent_gate, 'paired_clip_count': len(pairs), 'paired_clips': pairs,
        'evaluation_argument_differences': flag_diff, 'aggregate': {name: value['aggregate'] for name, value in results.items()},
        'evaluation_training_config_difference_raw': paired.differences(results['baseline']['training_config'], results['mixed']['training_config']),
        'explicit_config_serialization_normalization': 'legacy absent expert_sampling -> None; action_mask tuple/list only, identical values',
        'same_frame_diagnostics': {name: value['aggregate_diagnostic'] for name, value in decomposed.items()},
        'speed_conditioned_diagnostics': {name: value['by_reference_speed_bin'] for name, value in decomposed.items()},
        'secondary_success_subsets': {'each_run_success_only': {name: paired.subset_primary(values, success[name]) for name, values in rows.items()},
            'common_success_intersection': {name: paired.subset_primary(values, success['baseline'] & success['mixed']) for name, values in rows.items()}},
        'limits': ['All22 real clips and all fall frames retained; no phase/time fitting or failed-frame filtering.',
                   'Mixed source/reset/clock/RNG recipe is an engineering variant; no branch ID or future3D pose enters actor.',
                   'Mixed2000 doubles the old baseline total budget; each group exposure equals the old total, not matched-total-budget causality.',
                   'Full22 validation is repeatedly exposed, one seed; not an independent test or author protocol equivalence.',
                   'Full34 was already opened for the preceding root-curriculum candidate; this is not a new blind test fold.',
                   'No official style/MMD recomputation; XY reference tracking does not establish learned Diffusion closed-loop navigation.']}
    args.output.mkdir(parents=True)
    for name, value in decomposed.items():
        paired.write_new(args.output / f'{name}_decomposition.json', value)
    paired.write_new(args.output / 'comparison.json', output)
    lines = [f'# Mixed step{args.candidate_step} 完整22段验证', '',
             'CPU重算既有真实捕获；完整11982帧和失败帧保留，初始状态与参考/相位/时钟dtype及bitwise配对。', '',
             '| 主指标 | negative1000 | mixed |', '| --- | --- | --- |']
    for key in ('tracking_successes', 'sequences_with_fall', 'frame_weighted_mean_body_position_error_m', 'frame_weighted_mean_body_rotation_error_rad'):
        lines.append(f"| {key} | {results['baseline']['aggregate'][key]:.9g} | {results['mixed']['aggregate'][key]:.9g} |")
    budget = admission['budget']
    lines += ['', f"mixed累计{budget['updates']}updates，总{budget['total_transitions']}transitions，每组{budget['amass_transitions']}。baseline1000总8.192M；budget关系已单独记录。",
              '全22为主分母；success-only/common-success子集仅描述性，不代替主指标或未知的论文error分母。',
              '源码135tar逐member核验；预登记绑定实际评估6份runtime源，当前CPU admission的8份源另列，二者计数/用途不同。',
              '检查点字节seal绑定训练前登记source/argv，独立完整payload门禁与本reader的nestedstate/三RNG、real151/proc51、episode/两组实际budget核查通过。',
              '此前34test已在上一阶段开封；本次不读取它，也不宣称新的blind test。',
              '本报告不读取34test、不改变部署默认、权重或业务源码，不把reference tracking称200秒导航成功。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print({'status': 'passed', 'step': args.candidate_step, 'clips': len(pairs), 'budget': budget, 'aggregate': output['aggregate']})


if __name__ == '__main__':
    main()
