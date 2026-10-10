"""Root-only execution of frozen local commands after actual recovery admission."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'docs/style_improvement_20261003/state_prior_training_preparation_v1'
PLAN = PREP / 'root_GPU_launch_commands.json'
PLAN_SHA = '00e5e0c8a861f208a7b1eda7d0e7eaf9ef60989dfa29f51596c5bbdaa76c1e1d'
SOURCE_SHA = '3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', choices=('resume4to8', 'fullfresh2000', 'watch'), required=True)
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    require(digest(PLAN) == PLAN_SHA, 'Frozen launch plan changed')
    plan = json.loads(PLAN.read_text())
    watch = args.run == 'watch'
    admission_record = None
    if watch:
        launch = json.loads((HERE / 'fullfresh2000_actual_launch.json').read_text())
        require(launch['status'] == 'running', 'Watcher requires a running full training process')
        os.kill(launch['child_pid'], 0)
        command, env = plan['watch_command'], plan['CPU_watch_env']
    else:
        require(args.admission is not None, 'Actual recovery admission is required')
        gate = json.loads(args.admission.read_text())
        require(gate['schema'] == 'public_state_prior.actual_training_checkpoint_audit.v1', 'Wrong audit schema')
        require(gate['status'] == 'passed', 'Recovery audit did not pass')
        expected = [4] if args.run == 'resume4to8' else [4, 8]
        require(gate['phase'] == ('pilot4' if args.run == 'resume4to8' else 'pilot8'), 'Wrong recovery phase')
        require([row['updates'] for row in gate['checkpoints']] == expected, 'Wrong admitted update set')
        require(gate['source_manifest']['sha256'] == SOURCE_SHA, 'Wrong isolated source manifest')
        resume_command = plan['commands']['resume4to8']
        pilot_step4 = Path(resume_command[resume_command.index('--resume') + 1]).resolve()
        expected_paths = {4: pilot_step4, 8: pilot_step4.parent / 'step_8.pt'}
        for row in gate['checkpoints']:
            require(row['full_payload_resume_validated'] is True, 'Full checkpoint recovery not validated')
            require(row['state_prior_initialization_validated'] is True, 'State-prior initialization not validated')
            require(row['public_D_or_AMP_RMS_transferred'] is True, 'Public D and AMP statistics not admitted')
            checkpoint = Path(row['checkpoint']['path'])
            require(checkpoint.resolve() == expected_paths[row['updates']], 'Admitted checkpoint differs from frozen run input')
            require(checkpoint.stat().st_size == row['checkpoint']['bytes'], 'Checkpoint size changed')
            require(checkpoint.stat().st_mode & 0o222 == 0, 'Checkpoint is not sealed read-only')
            require(digest(checkpoint) == row['checkpoint']['sha256'], 'Checkpoint digest changed')
        admission_record = {'path': str(args.admission.resolve()), 'sha256': digest(args.admission)}
        command, env = plan['commands'][args.run], plan['env']
        require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Another GPU compute process is running')
        if args.run == 'fullfresh2000':
            require('--resume' not in command, 'Fresh2000 must not resume')
            require(not Path(command[command.index('--output') + 1]).exists(), 'Fresh training output already exists')
    record_path = HERE / (args.run + '_actual_launch.json')
    require(not record_path.exists(), 'Actual launch record already exists')
    log_path = HERE / (args.run + ('_CPU.log' if watch else '_GPU.log'))
    record = {'schema': 'state_prior.actual_launch.v1', 'name': args.run,
              'start_utc': now(), 'command': command, 'cwd': plan['cwd'], 'env': env,
              'launch_plan_sha256': PLAN_SHA, 'status': 'running', 'log': str(log_path),
              'admission': admission_record, 'CPU_only': watch, 'no_parallel_GPU_at_launch': True}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=plan['cwd'], env=dict(os.environ, **env),
                                   stdout=log, stderr=subprocess.STDOUT)
        record['child_pid'] = process.pid
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        code = process.wait()
    record.update(status='closed', exit_code=code, end_utc=now())
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    print(log_path.read_text()[-3500:])
    raise SystemExit(code)


if __name__ == '__main__':
    main()
