"""Record actual exit of the separately registered expert-sampling experiment."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def validate_gate(gate, phase, source_record):
    require(gate.get('status') == 'passed', 'Actual preparation or pilot admission failed')
    require(gate.get('CPU_only') is True and gate.get('GPU_used') is False
            and gate.get('SDK_started') is False and gate.get('sealed34_outputs_read') is False,
            'Actual CPU-only admission without sealed test outputs required')
    if phase == 'pilot4':
        require(gate.get('schema') == 'state_only_matched_q.actual_CPU_preparation.v1'
                and gate.get('source_manifest_sha256') == source_record['sha256']
                and gate.get('actual_fresh2000_parser_passed') is True
                and gate.get('full_resource_hashes_verified_by_unchanged_launcher') is True
                and gate.get('source_count') == 936 and gate.get('D_input') == 1950
                and gate.get('q_configuration_and_unchanged_q_sources_exact') is True
                and gate.get('direct_public_D_and_AMP_RMS_config_validated') is True
                and gate.get('matched_pre_step_history_replay_AST_validated') is True
                and gate.get('original_v7_mapping_probability_hashes_reproduced') is True,
                'Actual fixed-bank mapping and private/shared RNG preparation required')
        return
    require(gate.get('schema') == 'state_only_matched_q.actual_training_checkpoint_audit.v1'
            and gate.get('source_manifest') == source_record
            and gate.get('phase') == 'pilot4' and type(gate.get('copied_files_hashed')) is int
            and gate['copied_files_hashed'] == 936,
            'Actual state-only936 pilot4 payload admission required')
    checks = gate.get('checks')
    require(isinstance(checks, list) and bool(checks) and type(gate.get('check_count')) is int
            and gate['check_count'] == len(checks)
            and all(isinstance(row, dict) and row.get('passed') is True for row in checks),
            'Pilot checks must be complete, nonempty and strictly passed')
    rows = gate.get('checkpoints')
    require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict)
            and type(rows[0].get('updates')) is int and rows[0]['updates'] == 4
            and type(rows[0].get('transitions')) is int and rows[0]['transitions'] == 32768
            and rows[0].get('full_payload_resume_validated') is True and rows[0].get('D_input') == 1950
            and rows[0].get('state_prior', {}).get('strategy') == 'public_pacer_state_only'
            and rows[0].get('state_prior', {}).get('action_extension_explicit_flag') is False,
            'Actual complete pilot4 restoration required')
    require(gate.get('state_prior_initialization_validated') is True
            and gate.get('public_D_or_AMP_RMS_transferred') is True,
            'Coupled state/history initialization must be admitted')
    sampling = rows[0].get('actual_expert_sampling_saved_state')
    require(isinstance(sampling, dict)
            and sampling.get('actual_bank_registry_rebuilt') is True
            and sampling.get('private_CPU_RNG_saved_and_actual_loader_exact') is True
            and type(sampling.get('sampled_windows')) is int and sampling['sampled_windows'] == 16384
            and type(sampling.get('sampled_batches')) is int and sampling['sampled_batches'] == 4
            and sampling.get('importance_correction_applied') is False,
            'Actual q sampler state, counters and private RNG must be admitted')


def interrupted(signum, frame):
    raise InterruptedError(f'Root received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=('pilot4', 'fullfresh2000', 'watch'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    args = parser.parse_args()
    require(digest(args.plan) == args.plan_sha256, 'Registered launch plan changed')
    plan = json.loads(args.plan.read_text())
    source_path = Path(plan['source_manifest'])
    source = {'path': str(source_path), 'bytes': source_path.stat().st_size, 'sha256': digest(source_path)}
    require(source['sha256'] == '5c2a22098bf8df60530b5f53963c041a221cc17e62759284d7586f6ea6a05f50',
            'Registered state-only matched q source differs')
    require(digest(source['path']) == source['sha256']
            and Path(source['path']).stat().st_size == source['bytes'], 'Isolated source manifest changed')
    require(plan['single_Kit_GPU_at_a_time'] is True, 'Serial GPU protocol required')
    watch = args.run == 'watch'
    admission = None
    if watch:
        owner = json.loads((HERE / 'fullfresh2000_actual_launch.json').read_text())
        require(owner.get('schema') == 'state_only_matched_q.actual_launch.v1'
                and owner.get('name') == 'fullfresh2000' and owner.get('status') == 'running'
                and owner.get('launch_plan_sha256') == args.plan_sha256
                and owner.get('source_manifest') == source
                and type(owner.get('child_pid')) is int and owner['child_pid'] > 0,
                'Watcher requires actual running training from this registered plan and source')
        os.kill(owner['child_pid'], 0)
        command, env = plan['watch_command'], plan['CPU_watch_env']
        require(env.get('CUDA_VISIBLE_DEVICES') == '', 'Watcher must be CPU only')
    else:
        require(args.admission is not None and args.admission_sha256 is not None, 'Bound actual admission required')
        require(digest(args.admission) == args.admission_sha256, 'Actual admission changed')
        gate = json.loads(args.admission.read_text())
        validate_gate(gate, args.run, source)
        if args.run == 'fullfresh2000':
            admitted = gate['checkpoints'][0]['checkpoint']
            pilot_command = plan['commands']['pilot4']
            expected = Path(pilot_command[pilot_command.index('--output') + 1]) / 'step_4.pt'
            checkpoint = Path(admitted['path'])
            require(checkpoint.resolve() == expected.resolve() and not checkpoint.is_symlink()
                    and not checkpoint.stat().st_mode & 0o222
                    and checkpoint.stat().st_size == admitted['bytes']
                    and digest(checkpoint) == admitted['sha256'], 'Admitted pilot checkpoint changed')
        admission = {'path': str(args.admission.resolve()), 'sha256': args.admission_sha256}
        command, env = plan['commands'][args.run], plan['env']
        require(not any(token == '--resume' or token.startswith('--resume=') for token in command),
                'This expert-sampling experiment requires fresh training')
        require(not Path(command[command.index('--output') + 1]).exists(), 'Fresh output already exists')
        require(not subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(),
            'Another GPU compute process is running')
    path = HERE / (args.run + '_actual_launch.json')
    log_path = HERE / (args.run + ('_CPU.log' if watch else '_GPU.log'))
    require(not path.exists() and not log_path.exists(), 'Exclusive new actual run record required')
    record = {'schema': 'state_only_matched_q.actual_launch.v1', 'name': args.run,
              'status': 'running', 'start_utc': now(), 'command': command, 'cwd': plan['cwd'],
              'env': env, 'launch_plan_sha256': args.plan_sha256, 'source_manifest': source,
              'admission': admission, 'CPU_only': watch, 'no_parallel_GPU_at_launch': not watch,
              'atomic_GPU_lock_claimed': False, 'driver_sha256': digest(__file__), 'log': str(log_path)}
    process, failure = None, None
    write(path, record)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=plan['cwd'], env=dict(os.environ, **env),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['child_pid'] = process.pid
            write(path, record)
            process.wait()
    except BaseException as error:
        failure = error
        record['driver_error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
        except BaseException as error:
            failure = error
            record['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
        code = process.returncode if process is not None else 1
        observed = process is not None and process.poll() is not None
        status = 'closed' if observed else 'start_failed' if process is None else 'child_exit_not_observed'
        record.update(status=status, exit_code=code, end_utc=now(),
                      child_exit_observed=observed,
                      driver_completed=failure is None)
        write(path, record)
    print(json.dumps({'name': args.run, 'exit_code': code, 'record': str(path),
                      'child_exit_observed': record['child_exit_observed'], 'driver_completed': failure is None}))
    raise SystemExit(code if failure is None and code is not None else 1)


if __name__ == '__main__':
    main()

