"""Restore pinned reference checkouts without modifying existing repositories."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clone', action='store_true', help='Clone missing references; existing checkouts are only checked.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/reference_repositories.json').read_text())
    reports = []
    for entry in manifest['repositories']:
        target = ROOT / entry['path']
        if not target.exists():
            if not args.clone:
                reports.append({'name': entry['name'], 'status': 'missing', 'path': str(target)})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', entry['url'], str(target)], check=True)
            subprocess.run(['git', 'checkout', '--detach', entry['commit']], cwd=target, check=True)
        commit = run('git', 'rev-parse', 'HEAD', cwd=target)
        origin = run('git', 'remote', 'get-url', 'origin', cwd=target)
        if commit != entry['commit'] or origin != entry['url']:
            raise RuntimeError(f"Reference differs from pinned source: {entry['name']}")
        reports.append({'name': entry['name'], 'status': 'matched', 'commit': commit})
    print(json.dumps(reports, indent=2))
    return int(any(row['status'] != 'matched' for row in reports))

if __name__ == '__main__':
    raise SystemExit(main())
