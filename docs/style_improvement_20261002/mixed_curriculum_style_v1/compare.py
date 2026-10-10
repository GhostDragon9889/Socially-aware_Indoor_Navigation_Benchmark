"""Append actual mixed validation captures to nine frozen style dictionaries.

This CPU tool does not train, simulate, change metric parameters, or overwrite
historical evidence. Run names bind genuine500/1000/2000 full checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT/'docs/style_improvement_20261002'
HERE = Path(__file__).resolve().parent
PRIOR = DOCS/'amass_root_style_v3'
REFERENCE = DOCS/'backward_penalty_style_v1/heldout_reference.json'
REFERENCE_SHA = 'c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564'
MANIFEST = ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
MANIFEST_SHA = '68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca'
EXPECTED = {'mixed500': 500, 'mixed1000': 1000, 'mixed2000': 2000}
SOURCE = DOCS/'mixed_curriculum_training_source_v1.json'
SOURCE_SHA = '8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e'
FROZEN = {
    'scripts/evaluate_pedestrian_style.py': '82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f',
    'CrowdSim/humanoid/style_evaluation.py': 'b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2',
    'docs/style_improvement_20261002/decompose_motion_tracking.py': '122404852c3ed1e1275b579f331f129b8709cd4d59c8b7ea73db0739c886ccfe',
    'CrowdSim/humanoid/training/mixed_trajectories.py': 'c79732a675ac737d4ce0aff0d2b76cfa30cebed2d9c115e1a35da39299694d67',
}


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def preflight():
    files = {path: record(ROOT/path) for path in FROZEN}
    require(all(files[path]['sha256'] == sha for path, sha in FROZEN.items()), 'Frozen metric/command implementation changed')
    require(record(REFERENCE)['sha256'] == REFERENCE_SHA, 'Frozen reference artifact changed')
    require(record(MANIFEST)['sha256'] == MANIFEST_SHA, 'v7 training manifest changed')
    require(record(SOURCE)['sha256'] == SOURCE_SHA, 'Mixed prelaunch source identity changed')
    prior, comparison = load(PRIOR/'evaluation.json'), load(PRIOR/'comparison.json')
    require(prior['status'] == 'passed' and len(prior['runs']) == 9, 'Expected frozen nine-model predecessor')
    require(set(prior['runs']) == set(comparison['runs']), 'Frozen predecessor dictionaries disagree')
    require(record(PRIOR/'heldout_reference.json')['sha256'] == REFERENCE_SHA, 'Predecessor reference differs')
    return prior, comparison, {'frozen_files': files, 'reference': record(REFERENCE), 'v7_manifest': record(MANIFEST),
        'prior_evaluation': record(PRIOR/'evaluation.json'), 'prior_comparison': record(PRIOR/'comparison.json'),
        'mixed_prelaunch_source': record(SOURCE)}


def checkpoint_contract(result, name):
    """Read only metadata from an actual immutable CPU-mmap training payload."""
    import torch
    require(not torch.cuda.is_initialized(), 'CPU-only verification cannot use an initialized CUDA context')
    checkpoint = Path(result['arguments']['checkpoint'])
    checkpoint = (ROOT/checkpoint if not checkpoint.is_absolute() else checkpoint).resolve()
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    count = EXPECTED[name]
    transitions = count*256*32
    require(payload.get('format') == 'navisaaclab_method_reconstruction_training', 'Unknown real checkpoint format')
    require(type(payload.get('update_count')) is int and payload['update_count'] == count, 'Named milestone differs from checkpoint update_count')
    require(payload.get('physical_transition_count') == transitions, 'Checkpoint budget differs from256x32 actual rollout protocol')
    require(payload.get('physical_runner_state_included') is True, 'Checkpoint lacks full physical runner state')
    runner = payload['runner_state']
    require(runner['runner_contract']['trajectory_source'] == 'mixed_amass_procedural', 'Checkpoint is not the mixed experiment')
    curriculum = runner['curriculum_state']
    require(curriculum['schema'] == 'fixed_actor_mixed_trajectory_curriculum_v1', 'Mixed source state schema differs')
    config, stats = curriculum['config'], curriculum['statistics']
    require(config['num_envs'] == 256 and config['amass_actor_ids'] == list(range(128)) and config['procedural_actor_ids'] == list(range(128, 256)), 'Mixed fixed global mapping differs')
    require(stats['amass_actor_transitions'] == stats['procedural_actor_transitions'] == transitions//2, 'Actual equal group exposure differs')
    require(stats['control_steps'] == count*32, 'Actual completed control count differs')
    provenance = payload['provenance']
    require(provenance['trajectory_source'] == 'mixed_amass_procedural', 'Controller provenance is not mixed')
    require(provenance['dataset']['manifest_sha256'] == MANIFEST_SHA, 'Controller did not use frozen v7')
    require(payload['config'].get('expert_sampling') is None, 'Expert distribution changed from declared mixed experiment')
    digest = record(checkpoint)
    require(digest['sha256'] == result['arguments']['checkpoint_sha256'], 'Controller bytes changed after tracking capture')
    return {'checkpoint': digest, 'updates': count, 'physical_transitions': transitions,
            'actual_group_transitions': {'AMASS': transitions//2, 'procedural': transitions//2},
            'curriculum_statistics': stats, 'curriculum_config': config,
            'qualification': 'Full checkpoint metadata verified; no PhysX state or optimizer restore executed by this CPU comparison'}


def run_cpu(command, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['PYTHONPATH'] = str(ROOT)+':'+str(ROOT/'src')
    with Path(log).open('x') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def summarize(item, result, decomposition, decomposition_path):
    inclusive = item['inclusive_speed_conditioned']
    require(decomposition['status'] == 'passed' and decomposition['analytic_rigid_transform_check_passed'], 'Existing diagnostic did not pass')
    require(decomposition['primary_aggregate_unchanged'] == result['aggregate'], 'Diagnostic changed primary errors')
    require(decomposition['source_result']['sha256'] == item['sources']['result']['sha256'], 'Diagnostic source differs')
    return {'result': item['sources']['result'],
        'checkpoint': item['sources']['models_resolved_from_evaluation_arguments']['controller'],
        'aggregate': result['aggregate'], 'inclusive_mmd2': inclusive['equal_supported_bin_mean_pose_velocity_mmd2'],
        'upright_mmd2': item['upright_subset']['comparison']['equal_supported_bin_mean_pose_velocity_mmd2'],
        'upright_fraction': item['upright_subset']['generated_fraction_retained'],
        'speed_wasserstein_m_s': inclusive['physical_speed_distribution_wasserstein_m_s'],
        'speed_bins': [{k: row[k] for k in ('lower_m_s', 'upper_m_s', 'generated_count', 'reference_count')}
            | {'mmd2': None if row['pose_velocity_gaussian_mmd2'] is None else row['pose_velocity_gaussian_mmd2']['value']}
            for row in inclusive['speed_bins']],
        'toe_near_ground_body_origin_proxy': item['generated']['toe_slide_near_ground_proxy'],
        'mass_weighted_joint_origin_height_proxy_m': item['generated']['metrics']['mass_weighted_joint_origin_height_proxy_m'],
        'capture_validation': {k: item['capture_validation'][k] for k in ('metric_frames', 'clips',
            'initial_reset_frames_discarded', 'valid_same_clip_style_intervals', 'fall_metric_frames_kept')},
        'decomposition': record(decomposition_path), 'same_frame_decomposition': decomposition['aggregate_diagnostic'],
        'wrists_and_hands': [row for row in decomposition['by_body'] if row['body'] in ('L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')],
        'by_reference_speed_bin': decomposition['by_reference_speed_bin'],
        'heldout_parent_disjointness': item['reference_not_in_declared_training']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='append', default=[], help='mixed500/mixed1000/mixed2000=actual full22 validation directory')
    parser.add_argument('--output', type=Path, help='Fresh descendant directory of mixed_curriculum_style_v1')
    args = parser.parse_args()
    prior, prior_comparison, source_contract = preflight()
    if args.prepare_only:
        require(not args.run and args.output is None, 'Preparation must not imply actual runs or metrics')
        write_new(HERE/'preparation.json', {'schema': 'mixed_curriculum_style_preparation.v1',
            'status': 'ready_waiting_for_actual_validation_captures', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_contract': source_contract, 'tool': record(__file__), 'milestones': EXPECTED,
            'metrics_computed': False, 'sealed_34_test_outputs_read': False,
            'metric_parameters': prior['reference']['metric_spec'], 'prior_nine_run_names': list(prior['runs']),
            'append_rule': 'Evaluate only supplied actual mixed captures; preserve all nine historical run dictionaries exactly',
            'primary': 'Inclusive speed-conditioned90D pose/velocity MMD2, unchanged sigma1/features; all fall frames kept',
            'secondary': 'Upright subset, fixed speed bins, body-origin toe proxy, yaw/wrist decomposition; no contact or true COM inference'})
        print('Prepared frozen comparison; no new metrics computed')
        return
    require(args.run and args.output is not None, 'Actual comparison requires --run and a fresh --output')
    output = args.output.resolve()
    require(HERE in output.parents and not output.exists(), 'Output must be a fresh descendant; historical outputs cannot be overwritten')
    runs, results, contracts = {}, {}, {}
    baseline = load(prior_comparison['runs']['backward_penalty1000']['result']['path'])
    for declaration in args.run:
        name, separator, path = declaration.partition('=')
        require(separator and name in EXPECTED and name not in runs, 'Supply unique real mixed500/1000/2000 names')
        directory = Path(path).resolve()
        result = load(directory/'result.json')
        require(result.get('status') == 'passed' and result['arguments']['split'] == 'validation', 'Only completed validation captures may be compared; no test capture reads')
        require(len(result['sequences']) == 22 and {r['motion_id'] for r in result['sequences']} == {r['motion_id'] for r in baseline['sequences']}, 'Expected the same full22 validation identities')
        require(all(result[k] == baseline[k] for k in ('protocol', 'physics', 'layout', 'reference_velocity_contract')), 'Physical/reference benchmark protocol changed')
        contracts[name] = checkpoint_contract(result, name)
        runs[name], results[name] = directory, result
    output.mkdir(parents=True, exist_ok=False)
    evaluation_command = [sys.executable, str(ROOT/'scripts/evaluate_pedestrian_style.py'), '--reference-artifact', str(REFERENCE),
        '--cpu-threads', '4', '--minimum-bin-samples', '30', '--maximum-motion-samples', '256', '--output', str(output/'new_evaluation')]
    for name, path in runs.items():
        evaluation_command += ['--run', name+'='+str(path)]
    run_cpu(evaluation_command, output/'evaluation.log')
    fresh = load(output/'new_evaluation/evaluation.json')
    require(fresh['sources'] == prior['sources'] and fresh['reference'] == prior['reference'], 'Frozen evaluator/reference contract changed')
    merged = deepcopy(prior)
    merged['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    merged['runs'].update(fresh['runs'])
    merged['append_provenance'] = {'prior': source_contract, 'new_only_evaluation': record(output/'new_evaluation/evaluation.json'),
        'historical_run_dicts_recomputed': False}
    write_new(output/'evaluation.json', merged)
    shutil.copyfile(output/'new_evaluation/heldout_reference.json', output/'heldout_reference.json')
    require(record(output/'heldout_reference.json')['sha256'] == REFERENCE_SHA, 'Written reference bytes differ')
    spec = importlib.util.spec_from_file_location('unchanged_tracking_pair_helper', DOCS/'summarize_stratified_tracking.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    summary_runs = deepcopy(prior_comparison['runs'])
    pairs = {}
    for name, path in runs.items():
        decomposition = output/(name+'_decomposition.json')
        run_cpu([sys.executable, str(DOCS/'decompose_motion_tracking.py'), '--input', str(path), '--output', str(decomposition)], output/(name+'_decomposition.log'))
        summary_runs[name] = summarize(fresh['runs'][name], results[name], load(decomposition), decomposition)
        pairs[name] = helper.exact_initial_and_reference_pairs(baseline, results[name])
    old_eval_exact = {name: merged['runs'][name] == row for name, row in prior['runs'].items()}
    old_comparison_exact = {name: summary_runs[name] == row for name, row in prior_comparison['runs'].items()}
    require(all(old_eval_exact.values()) and all(old_comparison_exact.values()), 'Historical dictionaries changed')
    write_new(output/'comparison.json', {'schema': 'mixed_curriculum_tracking_style_review.v1', 'status': 'passed',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'CPU_only': True, 'Kit_started': False,
        'source_contract': source_contract, 'tool': record(__file__), 'frozen_reference': record(output/'heldout_reference.json'),
        'style_evaluation': record(output/'evaluation.json'), 'prior_nine_evaluation_dicts_exact': old_eval_exact,
        'prior_nine_comparison_dicts_exact': old_comparison_exact, 'runs': summary_runs, 'mixed_checkpoint_contracts': contracts,
        'same_initial_reference_pairs_vs_negative_baseline': pairs, 'sealed_34_test_outputs_read': False,
        'limitations': prior_comparison['limitations']+['Mixed milestones have unequal update budgets; compare physical transitions explicitly.',
            '50% actor-transition exposure does not imply equal resets or plans.',
            'Repeatedly exposed22 validation is not a blind test; no multi-seed significance or author-equivalent threshold.',
            'Tracking commands use true futureXY; this is not learned Diffusion closed-loop navigation.']})
    print(json.dumps({'status': 'passed', 'historical_nine_exact': True, 'new_runs': list(runs), 'output': str(output)}))


if __name__ == '__main__':
    main()
