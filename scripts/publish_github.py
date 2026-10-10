"""Create the requested private GitHub repository and verify its main commit."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Socially-aware_Indoor_Navigation_Benchmark'
PREPARED = ROOT / 'output/github_publication_20261010/source_checkout' / NAME
SOURCE_ROOT = PREPARED if (PREPARED / '.git').is_dir() else ROOT

def run(*args, **kwargs):
    return subprocess.run(args, cwd=SOURCE_ROOT, text=True, check=True, **kwargs)

def output(*args):
    return run(*args, capture_output=True).stdout.strip()

def main():
    if ROOT.name != NAME:
        raise RuntimeError('Run this script from the renamed canonical checkout.')
    run('gh', 'auth', 'status')
    owner = output('gh', 'api', 'user', '--jq', '.login')
    repository = owner + '/' + NAME
    local_head = output('git', 'rev-parse', 'main')
    if output('git', 'status', '--porcelain'):
        raise RuntimeError('Commit reviewed source changes before publishing.')
    lookup = subprocess.run(['gh', 'api', 'repos/' + repository], cwd=SOURCE_ROOT, text=True, capture_output=True)
    if lookup.returncode == 0:
        info = json.loads(lookup.stdout)
        if not info['private']:
            raise RuntimeError('An existing public repository has this name; refusing to change its visibility.')
    elif '(HTTP 404)' in lookup.stderr or 'Not Found' in lookup.stderr:
        run('gh', 'repo', 'create', repository, '--private', '--description',
            'Diffusion and AMP pedestrians for a socially-aware indoor navigation benchmark on Isaac Sim 6 / IsaacLab 3.')
    else:
        raise RuntimeError('Cannot inspect the destination repository: ' + lookup.stderr.strip())
    url = 'https://github.com/' + repository
    remote_url = url + '.git'
    current = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=SOURCE_ROOT, text=True, capture_output=True)
    if current.returncode == 0 and current.stdout.strip() != remote_url:
        raise RuntimeError('An unrelated origin already exists; refusing to replace it.')
    if current.returncode != 0:
        run('git', 'remote', 'add', 'origin', remote_url)
    run('gh', 'auth', 'setup-git', '--hostname', 'github.com')
    remote = output('git', 'ls-remote', 'origin')
    if remote and any(line.split()[0] != local_head for line in remote.splitlines() if not line.split()[1].startswith('refs/tags/')):
        raise RuntimeError('Destination contains different commits; refusing to overwrite history.')
    run('git', 'push', '--set-upstream', 'origin', 'main')
    remote_main = output('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    info = json.loads(output('gh', 'repo', 'view', repository, '--json', 'url,nameWithOwner,isPrivate,defaultBranchRef'))
    if remote_main != local_head or not info['isPrivate'] or info['defaultBranchRef']['name'] != 'main':
        raise RuntimeError('GitHub verification failed.')
    result = dict(info, commit=local_head, pushed=True)
    destination = ROOT / 'output/github_publication_20261010/github_result.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    if SOURCE_ROOT != ROOT:
        existing = subprocess.run(['git', 'remote', 'get-url', 'benchmark'], cwd=ROOT, text=True, capture_output=True)
        if existing.returncode != 0:
            subprocess.run(['git', 'remote', 'add', 'benchmark', remote_url], cwd=ROOT, check=True)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
