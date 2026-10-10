"""Bounded owner launcher for the new sealed state-prior Warehouse4/15 trial."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
DOC = Path(__file__).resolve().parent
SDK = '/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python'
SOURCE = ROOT/'docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json'
SOURCE_SHA = 'b4770f3f7301af7aed294db3145161453cf22325e66a9f9da0a3d677f5aeb3ee'
CKPT = ROOT/'output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt'
CKPT_SHA = '19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f'
TERMINAL = ROOT/'docs/style_improvement_20261003/state_prior_training_audit_v1/terminal2000_admission_v1.json'
TERMINAL_SHA = 'd5154251853253a8f866a103af9e51c6fff0e6ee7e1f56c36ff3f250b7124118'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def source_guard():
    require(record(SOURCE)['sha256'] == SOURCE_SHA, 'Exact production V11 required')
    manifest = json.loads(SOURCE.read_text())
    require(manifest['source_count'] == len(manifest['files']) == 183, 'Exact183 production sources')
    for row in manifest['files']:
        require(record(row['path']) == row, 'Production source changed: '+row['path'])
    require(record(manifest['archive']['path']) == manifest['archive'], 'V11 archive changed')
    return {'manifest': record(SOURCE), 'archive': manifest['archive'], 'all183_exact': True,
            'complete_runtime_import_ledger': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actors', type=int, choices=(4, 15), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    before = source_guard()
    require(record(CKPT)['sha256'] == CKPT_SHA and CKPT.stat().st_mode & 0o222 == 0,
            'Exact sealed readonly step2000 required')
    require(record(TERMINAL)['sha256'] == TERMINAL_SHA, 'Actual terminal admission changed')
    terminal = json.loads(TERMINAL.read_text())
    require(terminal['status'] == 'passed' and terminal['terminal_state_admitted'] is True
            and terminal['public_D_and_history_RMS_transfer_validated'] is True
            and terminal['milestone_only_admission'] is False, 'Full true-state-prior terminal admission required')
    completion_path = ROOT/'docs/style_improvement_20261003/state_prior_metrics_actual_runs_v1/navigation_CPU_completion_v1.json'
    completion = json.loads(completion_path.read_text())
    require(completion['status'] == 'passed', 'Complete actual three navCPU metrics required')
    comparison_path = ROOT/'docs/style_improvement_20261003/state_prior_complete_review_v1/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    require(comparison['minimum_gate_passed'] is True and all(comparison['registered_minimum_gate_results'].values()),
            'Registered actual fixture minimum gate required')
    config = ROOT/'configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml'
    suffix = '' if args.actors == 4 else '_15'
    scene = ROOT/f'configs/diffusion_amp/native_warehouse_pedestrians{suffix}_clearance08_v1.yaml'
    output = ROOT/f'output/diffusion_amp/native_warehouse_state_prior{suffix}_clearance08_200s_v1'
    require(not output.exists() and not output.is_symlink(), 'Must use the exact fresh native output')
    diffusion = ROOT/'output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt'
    manifest = ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
    require(record(diffusion)['sha256'] == '63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac', 'Diffusion changed')
    require(record(manifest)['sha256'] == '68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca', 'AMASS manifest changed')
    command = [SDK, str(ROOT/'scripts/simulate_paper_pedestrians.py'),
               '--amp-checkpoint', str(CKPT), '--config', str(config),
               '--diffusion-checkpoint', str(diffusion), '--manifest', str(manifest),
               '--scene-config', str(scene), '--actors', str(args.actors), '--duration-seconds', '200',
               '--seed', '7', '--inference-only', '--native-dynbench', '--arm-style-alpha', '0', '--output', str(output)]
    env = os.environ.copy()
    settings = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    env.update(settings)
    name = f'native{args.actors}'
    launch_path, log = DOC/f'{name}_actual_launch.json', DOC/f'{name}_GPU.log'
    if args.prepare_only:
        launch_path, log = DOC/f'{name}_actual_preflight.json', DOC/f'{name}_CPU_preflight.log'
        command.append('--preflight-only')
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                             check=True, text=True, capture_output=True)
        require(not gpu.stdout.strip(), 'Another CUDA compute process is active')
    require(not launch_path.exists() and not log.exists(), 'Never overwrite an actual launch receipt')
    receipt = {'schema': 'state_prior.actual_native_launch.v1', 'status': 'running',
               'start_utc': datetime.now(timezone.utc).isoformat(), 'actors': args.actors,
               'prepare_only': args.prepare_only, 'command': command, 'cwd': str(ROOT),
               'declared_env': settings, 'selected_source_before': before,
               'registered_fixture_minimum_only': True, 'overall_style_gain_claimed': False,
               'default_promotion': False, 'complete_import_ledger_claimed': False,
               'inputs': [record(p) for p in (CKPT, TERMINAL, completion_path, comparison_path,
                                               config, scene, diffusion, manifest)],
               'launcher': record(__file__), 'log': str(log)}
    launch_path.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt['child_pid'] = child.pid
            launch_path.write_text(json.dumps(receipt, indent=2)+'\n')
            receipt['exit_code'] = child.wait()
        receipt['selected_source_after'] = source_guard()
        receipt['status'] = 'closed'
    except BaseException as error:
        receipt['status'] = 'owner_launch_failed'
        receipt['owner_error'] = repr(error)
        raise
    finally:
        receipt['end_utc'] = datetime.now(timezone.utc).isoformat()
        launch_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
