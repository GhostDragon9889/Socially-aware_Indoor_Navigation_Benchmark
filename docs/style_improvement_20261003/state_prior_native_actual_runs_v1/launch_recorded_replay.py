"""Serial owner launcher for a visual replay of the actual new native4 states."""
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
SOURCE = ROOT/'docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json'
RUN = ROOT/'output/diffusion_amp/native_warehouse_state_prior_clearance08_200s_v1'
OUTPUT = ROOT/'output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1'
SDK = '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python'


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def source_guard():
    require(record(SOURCE)['sha256'] == 'b4770f3f7301af7aed294db3145161453cf22325e66a9f9da0a3d677f5aeb3ee',
            'Exact production source V11 required')
    source = json.loads(SOURCE.read_text())
    require(len(source['files']) == source['source_count'] == 183, 'Exact183 selected sources')
    for row in source['files']:
        require(record(row['path']) == row, 'Selected source changed: '+row['path'])
    require(record(source['archive']['path']) == source['archive'], 'Source archive changed')
    return {'manifest': record(SOURCE), 'archive': source['archive'], 'all183_exact': True,
            'complete_runtime_import_ledger': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source_before = source_guard()
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'New output required')
    owner = json.loads((DOC/'native4_actual_launch.json').read_text())
    require(owner['status'] == 'closed' and owner['exit_code'] == 0 and owner['actors'] == 4
            and owner['prepare_only'] is False, 'Actual native4 closed exit0 required')
    inputs = {name: record(RUN/name) for name in ('result.json', 'physical_states.npz')}
    require(inputs['result.json']['sha256'] == '5b66a60d6b3681d84ab559dad5adea3faaa00efbb44a996ce53b2c05604ff5b1'
            and inputs['physical_states.npz']['sha256'] == 'b7756c40d07c810f0f090e375682fb781eb7bbee6438b90d89083f8ae5261910',
            'Actual completed native4 bytes required')
    result = json.loads((RUN/'result.json').read_text())
    require(result['actual_control_steps'] == 6000 and result['evaluation_actor']['checkpoint_sha256'] ==
            '19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f', 'New controller identity required')
    command = [SDK, str(ROOT/'scripts/replay_paper_pedestrians.py'),
               '--states', str(RUN/'physical_states.npz'), '--source-result', str(RUN/'result.json'),
               '--output', str(OUTPUT), '--steps', '900', '--capture-every-steps', '2',
               '--camera-eye', '-6.75', '-8.95', '3.2', '--camera-target', '0.35', '6.35', '1.0',
               '--native-scene-visuals']
    suffix = 'CPU_preflight' if args.prepare_only else 'actual'
    report, log = DOC/f'recorded_replay_{suffix}_launch.json', DOC/f'recorded_replay_{suffix}.log'
    require(not report.exists() and not log.exists(), 'Preserve prior execution records')
    env = os.environ.copy()
    declared = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                'OPENBLAS_NUM_THREADS': '1'}
    env.update(declared)
    if args.prepare_only:
        command.append('--preflight-only')
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        other = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True)
        require(not other.stdout.strip(), 'Serial GPU use: native15 must really exit first')
        owner15 = json.loads((DOC/'native15_v2_actual_launch.json').read_text())
        require(owner15['status'] == 'closed' and owner15['child_exit_observed'] is True,
                'Actual15 process must be closed before rendering')
    receipt = {'schema': 'state_prior.actual_recorded_replay_launch.v1', 'status': 'running',
               'start_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
               'cwd': str(ROOT), 'declared_env': declared, 'prepare_only': args.prepare_only,
               'source_before': source_before, 'inputs': inputs, 'launcher': record(__file__),
               'visualization_only': True, 'fresh_physics_or_policy_claimed': False,
               'complete_runtime_import_ledger_claimed': False, 'log': str(log)}
    child = None
    report.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            receipt['child_pid'] = child.pid
            report.write_text(json.dumps(receipt, indent=2)+'\n')
            receipt['exit_code'] = child.wait()
        receipt['child_exit_observed'] = True
        receipt['source_after'] = source_guard()
        require(inputs == {name: record(RUN/name) for name in inputs}, 'Recorded source mutated')
        receipt['inputs_unchanged'] = True
        receipt['status'] = 'closed'
    except BaseException as error:
        receipt['status'] = 'owner_launch_failed'
        receipt['owner_error'] = repr(error)
        if child is not None:
            code = child.poll()
            if code is None:
                receipt['cleanup_requested'] = 'SIGTERM_process_group'
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    receipt['cleanup_escalated'] = 'SIGKILL_process_group'
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = child.wait()
            receipt['exit_code'] = code
            receipt['child_exit_observed'] = code is not None
        else:
            receipt['child_exit_observed'] = False
        raise
    finally:
        receipt['end_utc'] = datetime.now(timezone.utc).isoformat()
        report.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: receipt[key] for key in ('schema', 'status', 'prepare_only', 'exit_code',
                                                 'child_exit_observed', 'inputs_unchanged')}))
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
