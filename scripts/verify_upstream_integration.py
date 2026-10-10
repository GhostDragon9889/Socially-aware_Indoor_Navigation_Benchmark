"""Verify every mapped upstream file against the merged Git tree or index."""
import argparse
import json
from pathlib import Path
import subprocess


def git_entries(root, staged):
    args = ['ls-files', '--stage', '-z'] if staged else ['ls-tree', '-r', '-z', 'HEAD']
    raw = subprocess.check_output(['git', '-C', str(root), *args])
    result = {}
    for row in raw.split(b'\0'):
        if not row:
            continue
        metadata, path = row.split(b'\t', 1)
        fields = metadata.decode().split()
        if staged:
            mode, sha, stage = fields
            if stage != '0':
                raise RuntimeError('Unresolved merge conflict: ' + path.decode())
        else:
            mode, kind, sha = fields
            if kind != 'blob':
                continue
        result[path.decode()] = (mode, sha)
    return result


def verify(root, staged=False):
    catalog = json.loads((root / 'configs/upstream/protomotions_merge_20261010.json').read_text())
    rows = catalog['files']
    entries = git_entries(root, staged)
    failures = []
    if len(rows) != catalog['upstream_file_count'] or len({r['upstream_path'] for r in rows}) != len(rows):
        failures.append('Incomplete or duplicated upstream inventory')
    for row in rows:
        target = row['target_path']
        expected = (row['merged_mode'], row['merged_git_blob'])
        if entries.get(target) != expected:
            failures.append(target + ': missing or differs from recorded merge resolution')
    for key, expected in (('lfs.url', catalog['lfs_download_url']),
                          ('lfs.pushurl', catalog['lfs_push_url'])):
        actual = subprocess.check_output(['git', 'config', '-f', str(root / '.lfsconfig'), '--get', key], text=True).strip()
        if actual != expected:
            failures.append(key + ': incorrect LFS endpoint')
    if failures:
        raise RuntimeError('\n'.join(failures))
    return {'upstream_files_verified': len(rows), 'restored_files': catalog['restored_file_count'],
            'upstream_lfs_references': catalog['upstream_lfs_file_count'], 'failures': 0,
            'selection': 'index' if staged else 'HEAD'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.staged), indent=2))


if __name__ == '__main__':
    main()
