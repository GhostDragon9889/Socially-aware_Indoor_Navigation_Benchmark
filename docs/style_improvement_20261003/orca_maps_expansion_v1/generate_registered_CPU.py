"""Generate a fresh, local ORCA map corpus using unchanged isolated sources."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
DOC = Path(__file__).resolve().parent
BUNDLE = ROOT / 'output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2'
SOURCE_SHA = '1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21'
OUTPUT = ROOT / 'data/diffusion_amp/orca_maps_expanded_1000_v1'
PYTHON = ROOT / 'data/tools/orca_generator/.venv/bin/python'


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    source = record(BUNDLE / 'source_manifest.json')
    if source['sha256'] != SOURCE_SHA or OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('Exact frozen source and fresh output are required')
    receipt_path = DOC / 'generation_actual_launch.json'
    log = DOC / 'generation_CPU.log'
    if receipt_path.exists() or log.exists():
        raise FileExistsError('Never overwrite actual generation receipts')
    code = BUNDLE / 'code'
    command = [str(PYTHON), str(code / 'scripts/generate_orca_diffusion_data.py'),
               '--output', str(OUTPUT), '--trace-source', str(ROOT / 'third_party/trace'),
               '--rvo2-source', str(ROOT / 'data/tools/orca_generator/Python-RVO2'),
               '--seed', '20261003', '--train-scenes', '800', '--validation-scenes', '100',
               '--test-scenes', '100', '--episodes-per-scene', '1', '--seconds', '10',
               '--max-obstacles', '20']
    settings = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    env = os.environ.copy()
    env.update(settings)
    receipt = {'schema': 'local_orca_maps.actual_CPU_generation.v1', 'status': 'running',
               'start_utc': datetime.now(timezone.utc).isoformat(), 'CPU_only': True,
               'author_dataset': False, 'author_reproduction_claimed': False,
               'existing_AMASS_or_diffusion_inputs_modified': False,
               'command': command, 'cwd': str(code), 'declared_env': settings,
               'source_manifest': source,
               'owned_sources': [record(code / 'scripts/generate_orca_diffusion_data.py'),
                                 record(code / 'CrowdSim/humanoid/orca_corpus.py')],
               'driver': record(__file__), 'log': str(log),
               'known_differences_from_TRACE': [
                   'Map-only nonempty layouts; no ORCA-Interact subset is generated',
                   '4..15 agents and real-neighbor/visible-obstacle window selection',
                   'New seeds, bounded spawn rejection and exact 10Hz sampling',
                   'Velocity-derived heading is a disc proxy, not measured human orientation']}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=code, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        receipt['child_pid'] = child.pid
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        code = child.wait()
    receipt.update(status='closed', exit_code=code, child_exit_observed=True,
                   end_utc=datetime.now(timezone.utc).isoformat())
    if code == 0:
        receipt['manifest'] = record(OUTPUT / 'manifest.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.chmod(0o444)
    print(json.dumps(receipt))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
