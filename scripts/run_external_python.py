#!/usr/bin/env python3
"""Run a script/module with the configured external IsaacLab/Isaac Sim Python."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
# Ubuntu22.04 commonly invokes this entry with Python3.10. Read the binding
# using only stdlib before importing modules that require tomllib/Python3.12.
if sys.version_info < (3, 11):
    if sys.argv[1:2] == ['--sdk-child']:
        raise SystemExit('Select the configured SDK Python3.12 for --sdk-child')
    binding = json.loads((ROOT / '.runtime/external_sdk.json').read_text())
    python = binding.get('python')
    if (binding.get('schema') != 'social_navigation.external_sdk.v1'
            or not isinstance(python, str) or not Path(python).is_absolute() or not Path(python).is_file()):
        raise SystemExit('Configure a valid external SDK Python before launching')
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.execv(python, [python, str(__file__), '--sdk-child', *sys.argv[1:]])
sys.path.insert(0, str(ROOT / 'src'))
from social_navigation.external_runtime import execute, launch

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if arguments[:1] == ['--sdk-child']:
        execute(ROOT, arguments[1:])
    else:
        raise SystemExit(launch(ROOT, arguments))
