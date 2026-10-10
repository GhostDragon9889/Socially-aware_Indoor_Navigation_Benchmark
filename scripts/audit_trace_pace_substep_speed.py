"""Check the actual planar pelvis velocity at every archived120Hz physics substep."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(run):
    run = Path(run)
    execution_path = run.with_name(run.name + '_execution.json')
    execution = json.loads(execution_path.read_text())
    result = json.loads((run / 'result.json').read_text())
    assert execution['actual_returncode'] == 0 and execution['sources_unchanged'] and execution['inputs_unchanged']
    assert result['cleanup_succeeded'] and result['actual_controls'] == 1800
    assert execution['result_sha256'] == sha(run / 'result.json')
    for name in ('physical.npz', 'ground_support.npz'):
        assert execution['capture_sha256'][name] == sha(run / name)
    with np.load(run / 'ground_support.npz') as ground, np.load(run / 'physical.npz') as physical:
        assert ground['valid'].all() and (ground['substep_valid_count'] == 4).all()
        assert np.array_equal(ground['physics_step'].reshape(-1), np.arange(1, 7201))
        pelvis = ground['common_body_names'].tolist().index('Pelvis')
        velocities = ground['body_link_lin_vel_w'][..., pelvis, :2].copy()
        assert velocities.shape[:2] == (1800, 4) and np.isfinite(velocities).all()
        final_error = float(np.abs(velocities[:, -1] - physical['samples'][..., 2:4]).max())
        assert final_error < 1e-6
    speed = np.linalg.norm(velocities, axis=-1)
    rows = [{'actor': actor, 'actual_physics_substeps': 7200,
             'peak_m_s': float(speed[:, :, actor].max()),
             'mean_m_s': float(speed[:, :, actor].mean()),
             'substeps_above1p5': int((speed[:, :, actor] > 1.5).sum()),
             'substeps_below0p5_fraction': float((speed[:, :, actor] < .5).mean())}
            for actor in range(speed.shape[2])]
    return {'schema': 'crowdsim.actual_substep_speed.v1',
            'source_result_sha256': sha(run / 'result.json'),
            'ground_support_sha256': sha(run / 'ground_support.npz'),
            'physical_sha256': sha(run / 'physical.npz'),
            'native_execution_sha256': sha(execution_path), 'evaluator_source_sha256': sha(__file__),
            'actual_controls': 1800, 'actual_physics_substeps_per_actor': 7200,
            'control_final_velocity_max_abs_error': final_error,
            'upper_speed_m_s': 1.5, 'passed_upper1p5_all_substeps': bool((speed <= 1.5).all()),
            'per_actor': rows, 'turn_start_stop_low_speed_samples_retained': True,
            'scope': 'Observed actual pelvis planar velocities at every120Hz native substep; no physical state changes and no global bound on future runs.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh output required')
    report = evaluate(args.run)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed_upper1p5_all_substeps'], 'per_actor': report['per_actor']}))
    return 0 if report['passed_upper1p5_all_substeps'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
