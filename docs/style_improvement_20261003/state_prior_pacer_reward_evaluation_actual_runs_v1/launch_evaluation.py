"""Root owner for the two already registered, serial GPU evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
DOC = Path(__file__).resolve().parent
COMMANDS = ROOT / 'docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_v1/commands.json'
CHECKPOINT_SHA = '09e802847caba0b8c1ef836da2299d3468166be65aad127e2249f455326e0066'
ADMISSION = ROOT / 'docs/style_improvement_20261003/state_prior_pacer_reward_training_audit_v1/terminal2000/audit.json'
ADMISSION_SHA = 'c0e93cf94222a6f46a1d3bde8f0202915d0f1ca73dc56539c8941d2d754184c0'


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('tracking', 'navigation'), required=True)
    args = parser.parse_args()
    plan = json.loads(COMMANDS.read_text())
    for row in (plan['registered_protocol'], plan['executor']):
        if record(row['path']) != row:
            raise ValueError('Registered evaluation source changed')
    if record(ADMISSION)['sha256'] != ADMISSION_SHA:
        raise ValueError('Actual terminal admission changed')
    command = [CHECKPOINT_SHA if v == 'ACTUAL_SEALED_STEP2000_SHA256' else v
               for v in plan[args.protocol + '_GPU_root_only']]
    checkpoint = Path(command[command.index('--checkpoint') + 1])
    if checkpoint.is_symlink() or checkpoint.stat().st_nlink != 1 or checkpoint.stat().st_mode & 0o222:
        raise ValueError('Readonly exclusive checkpoint required')
    if record(checkpoint)['sha256'] != CHECKPOINT_SHA:
        raise ValueError('Actual checkpoint differs')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                         check=True, text=True, capture_output=True)
    if gpu.stdout.strip():
        raise RuntimeError('Another CUDA compute process is active')
    path, log = DOC / (args.protocol + '_actual_launch.json'), DOC / (args.protocol + '_GPU.log')
    if path.exists() or log.exists():
        raise FileExistsError('Actual launch records are exclusive')
    settings = plan['GPU_environment_root_single_App_only']
    env = os.environ.copy()
    env.update(settings)
    receipt = {'schema': 'pacer_reward.actual_evaluation_launch.v1', 'protocol': args.protocol,
               'status': 'running', 'start_utc': datetime.now(timezone.utc).isoformat(),
               'command': command, 'cwd': str(ROOT), 'declared_env': settings,
               'amp_checkpoint': record(checkpoint), 'terminal_admission': record(ADMISSION),
               'commands': record(COMMANDS), 'owner': record(__file__), 'log': str(log),
               'no_parallel_GPU_at_launch': True, 'atomic_GPU_lock_claimed': False,
               'paper_quality_claimed': False}
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    child = None
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt['child_pid'] = child.pid
            path.write_text(json.dumps(receipt, indent=2) + '\n')
            receipt['exit_code'] = child.wait()
        receipt.update(status='closed', child_exit_observed=True, driver_completed=True)
    except BaseException as error:
        receipt.update(status='owner_failed', owner_error=repr(error), driver_completed=False)
        if child is not None:
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
            receipt.update(exit_code=child.returncode, child_exit_observed=child.returncode is not None)
        raise
    finally:
        receipt['end_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    path.chmod(0o444)
    print(json.dumps({'protocol': args.protocol, 'status': receipt['status'],
                      'exit_code': receipt['exit_code'], 'receipt': str(path)}))
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
