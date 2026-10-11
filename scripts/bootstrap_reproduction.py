#!/usr/bin/env python3
"""Install the pinned SDK and the project's isolated runtime dependencies."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'prepare-native', 'prepare-runtime'))
    parser.add_argument('--profile', choices=('isaaclab3_ea_sim61', 'isaaclab3_sim6'),
                        default='isaaclab3_ea_sim61')
    parser.add_argument('--lab-root', type=Path)
    args = parser.parse_args()
    if args.action == 'install' and args.profile == 'isaaclab3_sim6':
        parser.error('The frozen Sim6.0 profile is for existing runtimes only; its stored project extra predates the Lab3EA migration. Use prepare-runtime with a matching installed Lab12/Sim6.0 SDK, or install the default isaaclab3_ea_sim61 profile. Original SDK locks are preserved.')
    if args.action == 'prepare-native':
        from social_navigation.reproduction import prepare_native_runtime
        print(json.dumps(prepare_native_runtime(ROOT), indent=2))
        return 0
    if shutil.which('uv') is None:
        parser.error('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
    lab = (args.lab_root or ROOT / '.runtime' / ('IsaacLab61' if args.profile == 'isaaclab3_ea_sim61' else 'IsaacLab60')).expanduser().resolve()
    profile = json.loads((ROOT / 'env' / args.profile / 'profile.json').read_text())
    if args.action == 'install':
        if (lab / '.venv/bin/python').is_file():
            parser.error('An installed SDK already exists; preserved. Use prepare-runtime --lab-root with this matching SDK instead of install.')
        if not lab.exists():
            lab.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                            'https://github.com/isaac-sim/IsaacLab.git', str(lab)], check=True)
            subprocess.run(['git', '-C', str(lab), 'checkout', '--detach', profile['isaaclab_commit']], check=True)
        os.environ['SOCIAL_NAVIGATION_SDK_PROFILE'] = args.profile
        environment = importlib.import_module('isaaclab3_environment')
        environment.require_checkout(lab)
        subprocess.run(['uv', 'python', 'install', profile['python_recorded']], check=True)
        environment.apply_profile(lab, sync=True, fresh=True)
    if args.profile == 'isaaclab3_ea_sim61':
        from social_navigation.runtime_environment import prepare_runtime_environment
        result = prepare_runtime_environment(ROOT, lab)
        python = Path(result['python'])
    else:
        python = lab / '.venv/bin/python'
        subprocess.run([str(python), '-I', str(ROOT / 'scripts/isaaclab3_environment.py'), '--profile', 'isaaclab3_sim6', 'check'], check=True)
        result = {'status': 'legacy_existing_environment', 'python': str(python)}
    subprocess.run([str(python), str(__file__), 'prepare-native', '--profile', args.profile], cwd=ROOT, check=True)
    link = ROOT / '.venv'
    target = python.parent.parent
    if link.is_symlink():
        if link.resolve() != target.resolve():
            link.unlink()
            link.symlink_to(os.path.relpath(target, ROOT), target_is_directory=True)
    elif not link.exists():
        link.symlink_to(os.path.relpath(target, ROOT), target_is_directory=True)
    elif link.resolve() != target.resolve():
        raise RuntimeError('Existing .venv is a directory; preserved. Select the reported interpreter explicitly.')
    result.update(profile=args.profile, datasets_restored=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
