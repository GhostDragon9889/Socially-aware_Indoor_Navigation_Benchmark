# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise preference pipeline SDK routing without constructing a simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from protomotions.repository import repository_root


@pytest.mark.parametrize('cuda_device', [None, '2'])
def test_training_stage_uses_configured_sdk_from_an_unrelated_directory(tmp_path, cuda_device):
    bin_dir = tmp_path / 'bin with spaces'
    bin_dir.mkdir()
    calls = tmp_path / 'calls.jsonl'
    fake = bin_dir / 'python3'
    fake.write_text('#!' + sys.executable + '\n' +
                    'import json, os, sys\n' +
                    'with open(os.environ["SDK_CALLS"], "a") as stream:\n' +
                    ' stream.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(), "cuda": os.getenv("CUDA_VISIBLE_DEVICES")}) + "\\n")\n')
    fake.chmod(0o700)
    root = repository_root()
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               SDK_CALLS=str(calls), NUM_ENVS='8', TOTAL_STEPS='64', ROLLOUT_STEPS='4')
    env.pop('CUDA_DEVICE', None)
    if cuda_device is not None:
        env['CUDA_DEVICE'] = cuda_device
    result = subprocess.run(['bash', str(root / 'src/CrowdSim/scripts/run_pref_pipeline.sh'), 'train'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert rows[0]['args'] == ['scripts/bootstrap_reproduction.py', 'check']
    assert all(row['cwd'] == str(root) for row in rows)
    training = rows[-1]
    assert training['cuda'] == (cuda_device or '0')
    assert training['args'][:2] == ['scripts/run_external_python.py', 'CrowdSim/ppo/train_ppo.py']
    assert training['args'][2:] == ['--num-envs', '8', '--headless', '--total-steps', '64',
                                    '--rollout-steps', '4', '--save-interval', '20000']
