"""Freeze the bounded CPU audit and distinguish cumulative reset counters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
probe = json.loads((OUT/'actual_cpu_probe.json').read_text())
heading = json.loads((OUT/'heading_actor_probe.json').read_text())
for row in probe['runs'].values():
    row['metrics'].pop('resets_sum_in_this_file_only')
    path = Path(row['metrics']['source']['path'])
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    row['metrics']['reset_counter_scope'] = 'Each metrics row is cumulative within its actual process; never sum these rows; resumed process starts new counters.'
    row['metrics']['last_reported_process_counters'] = {
        name: records[-1].get(name) for name in ('fall_resets', 'time_limit_resets', 'resets')}
configs = {name: row['config'] for name, row in probe['runs'].items()}
def differences(left, right):
    return {key: [left.get(key), right.get(key)] for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)}
report = {'schema': 'local.amp_style_bottleneck_review.v1', 'status': 'bounded_CPU_probes_passed',
          'SDK_started': False, 'GPU_used': False, 'test_split_capture_read': False,
          'business_or_data_modified': False, 'initial_tool_scope_correction':
          'Initial actual_cpu_probe.json reset sum was a reader aggregation error. No conclusion uses it. Final report removes it and retains actual final process counters.',
          'actual_D_probes': probe, 'actual_heading_and_offline_actor_probes': heading,
          'saved_config_differences': {
              'old_state_only_to_state_action': differences(configs['mixed_state_only_2000'], configs['mixed_state_action_2000']),
              'old_state_action_to_protected': differences(configs['mixed_state_action_2000'], configs['protected_8'])},
          'interpretation_limits': [
              '512 frozen private-seed AMASS expert windows; not an entire-bank probability census.',
              '512 equidistant actual stored replay indices; live/recency/curriculum actor identity absent.',
              'D input sensitivities and action swaps are offline counterfactuals, not physical or causal training evidence.',
              'Squared input-gradient fractions depend on normalization; they are not a theorem that actions never matter.',
              'Expert-state self/task inputs are reconstructed from AMASS source futures and zero training terrain, not original online policy observations.',
              '4 and resumed8 pilots validate mechanisms, not matched-budget style or navigation quality.',
              'Current protected full2000 continues unchanged; its future results are not assumed.',
              'Bank SHA uses existing independently verified frozen v7 proof; this audit restats and mmap reads bounded samples, does not rehash the 1.16GB bank.',
              'RNG invariance applies to caller Python/NumPy/Torch CPU; no GPU/Kit execution or CUDA RNG claim.'],
          'reused_measurement_evidence': {},
          'primary_sources': [
              {'name': 'NavIsaacLab', 'url': 'https://arxiv.org/html/2606.26265v1', 'scope': 'IV-B discriminator history/action, reward, IV-C AMASS selection'},
              {'name': 'AMP', 'url': 'https://xbpeng.github.io/projects/AMP/AMP_2021.pdf', 'scope': '5.1 observed state transitions; 5.2 LSGAN bounded reward'},
              {'name': 'CALM', 'url': 'https://arxiv.org/html/2305.02195v1', 'scope': 'conditional motion encoder and latent conditioned policy/D'},
              {'name': 'CMP', 'url': 'https://arxiv.org/html/2608.03234v1', 'scope': 'context-motion relevance and residual adaptation, not an identified CMD method'},
              {'name': 'SMP', 'url': 'https://arxiv.org/html/2512.03028v2', 'scope': 'separate full-body score matching motion prior; not XY planner checkpoint'}]}
for name, path in {
    'coverage': 'docs/style_improvement_20261002/amass_style_coverage_v7_findings_v1.json',
    'frozen_navigation': 'docs/style_improvement_20261002/state_action_navigation_style_v1/comparison.json',
    'wrist_heading': 'docs/style_improvement_20261002/state_action_navigation_style_v1/descriptive_yaw_wrist.json',
    'poststep_RMS_reconstruction': 'docs/style_improvement_20261002/navigation_terrain_rms_review_v1/audit.json',
    'actual_preaction_negative_baseline': 'docs/style_improvement_20261002/pre_action_actual_observation_reader_v1/actual_cuboid30_v1/report.json',
    'protected_step4_seal': 'output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step4_seal.json',
    'protected_step8_seal': 'output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step8_seal.json',
    'isolated931_source': 'output/isolated_training/mixed_state_action_terrain_protected_v1/source_manifest.json',
}.items():
    target = ROOT/path
    report['reused_measurement_evidence'][name] = {'path': str(target), 'bytes': target.stat().st_size,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
with (OUT/'report.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({'status': 'passed', 'report': str(OUT/'report.json')}))
