"""Read actual Diffusion terminal metadata on CPU; no simulation or inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[2]


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only metadata read required')
    import torch

    torch.set_num_threads(4)
    registration = ROOT / 'docs/style_improvement_20261003/orca_maps_diffusion_experiment_v1/registration.json'
    plan = json.loads(registration.read_text())
    receipt_path, seal_path = DOC / 'training_actual_launch.json', DOC / 'terminal5000_root_seal.json'
    receipt, seal = json.loads(receipt_path.read_text()), json.loads(seal_path.read_text())
    assert receipt['status'] == 'closed' and receipt['exit_code'] == 0 and receipt['child_exit_observed']
    assert receipt['selected_source_before'] == receipt['selected_source_after']
    new_path = Path(seal['exclusive_terminal_copy']['path'])
    new = torch.load(new_path, weights_only=True, map_location='cpu')
    old = torch.load(plan['bindings']['old_model']['path'], weights_only=True, map_location='cpu')
    initial = torch.load(plan['bindings']['initial_model']['path'], weights_only=True, map_location='cpu')
    assert new['step'] == old['step'] == 5000 and initial['step'] == 1000
    assert new['schema'] == old['schema'] == initial['schema']
    assert new['config'] == old['config'] == initial['config']
    buffers = ('state_mean', 'state_std')
    for key in buffers:
        assert torch.equal(new['model'][key], old['model'][key])
        assert torch.equal(new['model'][key], initial['model'][key])
    assert all(torch.isfinite(value).all() for value in new['model'].values())
    assert new['optimizer']['state']
    for state in new['optimizer']['state'].values():
        assert state['step'].item() == 5000
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    metadata = new['dataset_metadata']
    assert metadata['train']['manifest_sha256'] == plan['bindings']['new_manifest']['sha256']
    assert metadata['initialization']['sha256'] == plan['bindings']['initial_model']['sha256']
    assert metadata['initialization']['optimizer_reset'] and metadata['initialization']['rng_reset']
    assert all(value > 0 for value in metadata['first_condition_gradient_evidence']['grad_norms_after_clip'].values())
    assert not torch.cuda.is_initialized()
    report = {'schema': 'local_orca_maps.actual_diffusion_terminal_CPU_metadata.v1',
              'status': 'passed', 'CPU_only': True, 'GPU_used': False,
              'simulation_or_inference_executed': False, 'reader': record(__file__),
              'registration': record(registration), 'actual_closed_training': record(receipt_path),
              'terminal_root_seal': record(seal_path), 'new_checkpoint': record(new_path),
              'old_checkpoint': plan['bindings']['old_model'], 'initial_checkpoint': plan['bindings']['initial_model'],
              'steps': {'initial': 1000, 'old': 5000, 'new': 5000},
              'architectures_equal': True, 'state_mean_std_equal_old_new_initial': True,
              'finite_model_and_Adam_moments': True, 'all_Adam_parameter_steps': 5000,
              'optimizer_parameter_states': len(new['optimizer']['state']),
              'new_dataset_manifest_exact': True, 'fresh_optimizer_rng_declared_in_actual_payload': True,
              'map_and_neighbor_first_gradient_nonzero': True, 'training_resume_or_quality_proven': False,
              'sealed34_outputs_read': False}
    target = DOC / 'terminal_CPU_metadata.json'
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    target.chmod(0o444)
    print(json.dumps({'status': report['status'], 'report': record(target)}))


if __name__ == '__main__':
    main()
