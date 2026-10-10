#!/usr/bin/env python3
"""Install the exact SDK profile or restore its native overlay (Linux x86_64)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'prepare-native'))
    parser.add_argument('--lab-root', type=Path, default=ROOT / '.runtime/IsaacLab')
    args = parser.parse_args()
    if args.action == 'prepare-native':
        from social_navigation.reproduction import prepare_native_runtime
        print(json.dumps(prepare_native_runtime(ROOT), indent=2))
        return 0
    if shutil.which('uv') is None:
        parser.error('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
    lab = args.lab_root.expanduser().resolve()
    profile = json.loads((ROOT / 'env/isaaclab3_sim6/profile.json').read_text())
    if not lab.exists():
        lab.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/isaac-sim/IsaacLab.git', str(lab)], check=True)
        subprocess.run(['git', '-C', str(lab), 'checkout', '--detach', profile['isaaclab_commit']], check=True)
    # Existing checkout source must pass before any manifest/package modification.
    import isaaclab3_environment as environment
    environment.require_checkout(lab)
    subprocess.run(['uv', 'python', 'install', profile['python_recorded']], check=True)
    environment.apply_profile(lab, sync=True, fresh=True)
    python = lab / '.venv/bin/python'
    subprocess.run([str(python), str(__file__), 'prepare-native'], cwd=ROOT, check=True)
    link = ROOT / '.venv'
    if not link.exists() and not link.is_symlink():
        import os
        link.symlink_to(os.path.relpath(lab / '.venv', ROOT), target_is_directory=True)
    print(json.dumps({'status': 'installed', 'python': str(python),
        'activate': f'source "{lab}/.venv/bin/activate"', 'datasets_restored': False,
        'license': 'Isaac Sim EULA acceptance remains a user installation requirement'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
