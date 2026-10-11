"""Verify the historical upstream merge and explicitly registered maintenance.

The reviewed maintenance policy accompanies this verifier. Target files and
the frozen catalog are read only from HEAD or the index, never the worktree.
Git LFS payload downloads are unnecessary: this verifies Git pointer blobs.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


CATALOG = 'configs/upstream/protomotions_merge_20261010.json'
MAINTENANCE = 'configs/upstream/protomotions_maintenance_20261011.json'
CATALOG_COMMIT = '15dd16bfc40320ae13ec147daa03cc8779d95807'
CATALOG_BLOB = '989602ed0b3454b8c47bb1ae3b29910d411461c2'
MERGE_COMMIT = '23ccdddee7f1d72c08b59f7985a13a1a79084cbe'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if result.returncode:
        raise RuntimeError('Git verification failed (complete history is required): '
                           + result.stderr.decode(errors='replace').strip())
    return result.stdout


def git_entries(root, staged=False, *, ref='HEAD'):
    args = ['ls-files', '--stage', '-z'] if staged else ['ls-tree', '-r', '-z', ref]
    result = {}
    for row in git(root, *args).split(b'\0'):
        if not row:
            continue
        metadata, path = row.split(b'\t', 1)
        fields = metadata.decode().split()
        if staged:
            mode, sha, stage = fields
            require(stage == '0', 'Unresolved merge conflict: ' + path.decode())
        else:
            mode, kind, sha = fields
            if kind != 'blob':
                continue
        result[path.decode()] = (mode, sha)
    return result


def object_id(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{40}', value) is not None,
            'Require a complete Git object identity')
    return value


def relative_path(value):
    require(isinstance(value, str) and bool(value) and '\\' not in value,
            'Require a repository-relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
            'Require a canonical repository-relative path')
    return value


def identity(value):
    require(isinstance(value, dict) and set(value) == {'mode', 'git_blob'},
            'Require an exact mode/blob identity')
    require(value['mode'] in ('100644', '100755', '120000'), 'Invalid Git file mode')
    return value['mode'], object_id(value['git_blob'])


def ancestor(root, earlier, later):
    git(root, 'merge-base', '--is-ancestor', earlier, later)


def verify_history(root, catalog, tree):
    """Authenticate the original mapping separately from current maintenance."""
    rows = catalog['files']
    upstream = tree(catalog['upstream_commit'])
    require(git(root, 'rev-parse', catalog['upstream_commit'] + '^{tree}').decode().strip()
            == catalog['upstream_tree'], 'Upstream tree identity changed')
    require(len(rows) == catalog['upstream_file_count'] == len(upstream),
            'Incomplete upstream inventory')
    require(len({r['upstream_path'] for r in rows}) == len(rows)
            and len({r['target_path'] for r in rows}) == len(rows),
            'Duplicated upstream inventory')
    require({r['upstream_path']: (r['mode'], r['sha']) for r in rows} == upstream,
            'Catalog differs from the complete original upstream tree')
    parents = git(root, 'show', '-s', '--format=%P', MERGE_COMMIT).decode().split()
    require(parents == [catalog['benchmark_base_commit'], catalog['upstream_commit']],
            'Historical merge parent identities changed')
    merged, baseline = tree(MERGE_COMMIT), tree(CATALOG_COMMIT)
    for row in rows:
        target = relative_path(row['target_path'])
        require(merged.get(target) == (row['merged_mode'], row.get('initial_merge_git_blob', row['merged_git_blob'])),
                'Historical merge resolution differs: ' + target)
        require(baseline.get(target) == (row['merged_mode'], row['merged_git_blob']),
                'Catalog baseline differs: ' + target)
    ancestor(root, MERGE_COMMIT, CATALOG_COMMIT)
    ancestor(root, CATALOG_COMMIT, 'HEAD')


def maintenance_expectations(root, catalog, policy, tree):
    """Accept only explicit, connected changes proved by committed Git objects."""
    require(set(policy) == {'schema', 'catalog_commit', 'merge_commit', 'files'}
            and policy['schema'] == 'social_navigation.upstream_maintenance.v1'
            and policy['catalog_commit'] == CATALOG_COMMIT and policy['merge_commit'] == MERGE_COMMIT,
            'Unregistered maintenance policy baseline')
    expected = {r['target_path']: (r['merged_mode'], r['merged_git_blob']) for r in catalog['files']}
    require(type(policy['files']) is list, 'Maintenance inventory must be a list')
    seen = set()
    for row in policy['files']:
        require(set(row) == {'target_path', 'reason', 'updates'}, 'Unexpected maintenance fields')
        target = relative_path(row['target_path'])
        require(target in expected and target not in seen, 'Unknown or duplicate maintenance target: ' + target)
        seen.add(target)
        require(isinstance(row['reason'], str) and bool(row['reason'].strip()), 'Maintenance reason required')
        require(type(row['updates']) is list and bool(row['updates']), 'Maintenance updates required')
        previous_commit = CATALOG_COMMIT
        for update in row['updates']:
            require(set(update) in ({'commit', 'before', 'after'}, {'commit', 'before', 'after', 'upstream_origin'}),
                    'Unexpected maintenance update fields')
            commit = object_id(update['commit'])
            require(commit != previous_commit, 'Repeated maintenance commit')
            before, after = identity(update['before']), identity(update['after'])
            require(before == expected[target] and before != after, 'Broken maintenance identity chain: ' + target)
            ancestor(root, previous_commit, commit + '^')
            ancestor(root, commit, 'HEAD')
            require(tree(commit + '^').get(target) == before and tree(commit).get(target) == after,
                    'Maintenance commit/mode/blob does not prove the declared change: ' + target)
            if 'upstream_origin' in update:
                origin = update['upstream_origin']
                require(set(origin) == {'commit', 'path', 'lfs_oid', 'lfs_bytes'}, 'Unexpected upstream origin fields')
                source_commit, source_path = object_id(origin['commit']), relative_path(origin['path'])
                ancestor(root, source_commit, catalog['upstream_commit'])
                require(tree(source_commit).get(source_path) == after, 'Replacement differs from its upstream origin')
                require(isinstance(origin['lfs_oid'], str) and re.fullmatch('[0-9a-f]{64}', origin['lfs_oid'])
                        and type(origin['lfs_bytes']) is int and origin['lfs_bytes'] > 0, 'Invalid replacement LFS identity')
                pointer = ('version https://git-lfs.github.com/spec/v1\noid sha256:' + origin['lfs_oid']
                           + '\nsize ' + str(origin['lfs_bytes']) + '\n').encode()
                require(git(root, 'cat-file', 'blob', after[1]) == pointer, 'Replacement LFS pointer differs')
            expected[target] = after
            previous_commit = commit
    return expected, len(seen)


def verify(root, staged=False):
    root = Path(root).resolve()
    entries = git_entries(root, staged)
    require(entries.get(CATALOG) == ('100644', CATALOG_BLOB), 'Frozen upstream catalog changed in selected Git state')
    catalog = json.loads(git(root, 'cat-file', 'blob', CATALOG_BLOB))
    # This explicit policy is reviewed with the verifier, not inferred from
    # whichever source files happen to exist in a developer's working tree.
    policy = json.loads((root / MAINTENANCE).read_text())
    cache = {}
    def tree(ref):
        if ref not in cache:
            cache[ref] = git_entries(root, ref=ref)
        return cache[ref]
    verify_history(root, catalog, tree)
    expected, maintenance_count = maintenance_expectations(root, catalog, policy, tree)
    failures = [path + ': missing or differs from registered merge/maintenance identity'
                for path, value in expected.items() if entries.get(path) != value]
    lfs = entries.get('.lfsconfig')
    require(lfs is not None and lfs[0] == '100644', 'Missing or invalid selected .lfsconfig')
    for key, value in (('lfs.url', catalog['lfs_download_url']), ('lfs.pushurl', catalog['lfs_push_url'])):
        actual = git(root, 'config', '--blob', lfs[1], '--get', key).decode().strip()
        if actual != value:
            failures.append(key + ': incorrect LFS endpoint')
    require(not failures, '\n'.join(failures))
    return {'upstream_files_verified': len(expected), 'restored_files': catalog['restored_file_count'],
            'upstream_lfs_references': catalog['upstream_lfs_file_count'], 'failures': 0,
            'historical_merge_verified': MERGE_COMMIT, 'catalog_baseline_verified': CATALOG_COMMIT,
            'registered_maintenance_files': maintenance_count, 'LFS_payloads_verified': False,
            'selection': 'index' if staged else 'HEAD', 'worktree_sources_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.staged), indent=2))


if __name__ == '__main__':
    main()
