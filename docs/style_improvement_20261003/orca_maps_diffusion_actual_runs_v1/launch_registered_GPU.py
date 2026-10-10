"""Execute the preregistered fresh ORCA Diffusion command on the root's GPU."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
DOC = Path(__file__).resolve().parent
PLAN = ROOT / 'docs/style_improvement_20261003/orca_maps_diffusion_experiment_v1/registration.json'
PLAN_SHA = '89060cfc42d6217bd8b6ca0ab760a67c00fe3d3b44956af0f0cf970a02ed0c4e'


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def selected_source(plan):
    manifest = Path(plan['bindings']['source_manifest']['path'])
    if record(manifest) != plan['bindings']['source_manifest']:
        raise ValueError('Frozen source manifest changed')
    data = json.loads(manifest.read_text())
    for row in data['files']:
        path = manifest.parent / row['path']
        actual = record(path)
        if (actual['bytes'], actual['sha256']) != (row['bytes'], row['sha256']):
            raise ValueError(f'Frozen source changed: {path}')
    return {'manifest': record(manifest), 'regular_files': len(data['files']),
            'all_declared_files_verified': True,
            'complete_runtime_import_ledger_claimed': False}


def main():
    if record(PLAN)['sha256'] != PLAN_SHA:
        raise ValueError('Preregistered command changed')
    plan = json.loads(PLAN.read_text())
    for row in plan['bindings'].values():
        if record(row['path']) != row:
            raise ValueError(f'Registered binding changed: {row["path"]}')
    qc = json.loads(Path(plan['bindings']['dataset_QC']['path']).read_text())
    if qc['status'] != 'passed' or qc['unique_scenes'] != 1000:
        raise ValueError('Actual expanded dataset QC required')
    command = plan['command']
    output = Path(command[command.index('--output-dir') + 1])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('A fresh experiment requires an empty output directory')
    before = selected_source(plan)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                         check=True, text=True, capture_output=True)
    if gpu.stdout.strip():
        raise RuntimeError('Another CUDA compute process is active')
    receipt_path, log = DOC / 'training_actual_launch.json', DOC / 'training_GPU.log'
    if receipt_path.exists() or log.exists():
        raise FileExistsError('Actual launch records are exclusive')
    settings = plan['declared_env']
    env = os.environ.copy()
    env.update(settings)
    receipt = {'schema': 'local_orca_maps.actual_diffusion_training_launch.v1',
               'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
               'command': command, 'cwd': plan['cwd'], 'declared_env': settings,
               'registration': record(PLAN), 'bindings': plan['bindings'],
               'owner': record(__file__), 'log': str(log),
               'selected_source_before': before,
               'no_parallel_GPU_at_launch': True, 'atomic_GPU_lock_claimed': False,
               'author_data_or_weights': False, 'paper_quality_claimed': False,
               'existing_AMP_inputs_changed': False, 'exit_code': None,
               'child_exit_observed': False, 'driver_completed': False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    child = None
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=plan['cwd'], env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt['child_pid'] = child.pid
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            receipt['exit_code'] = child.wait()
        receipt.update(status='closed', child_exit_observed=True, driver_completed=True)
    except BaseException as error:
        receipt.update(status='owner_failed', owner_error=repr(error), driver_completed=False)
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=10)
            except BaseException as cleanup_error:
                receipt['cleanup_error'] = repr(cleanup_error)
            finally:
                receipt.update(exit_code=child.returncode, child_exit_observed=child.returncode is not None)
        raise
    finally:
        receipt['end_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            receipt['selected_source_after'] = selected_source(plan)
        except BaseException as error:
            receipt['source_verification_error'] = repr(error)
            receipt['status'] = 'source_verification_failed'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.chmod(0o444)
    print(json.dumps({'status': receipt['status'], 'exit_code': receipt['exit_code'],
                      'receipt': str(receipt_path)}), flush=True)
    return 1 if 'source_verification_error' in receipt else receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
