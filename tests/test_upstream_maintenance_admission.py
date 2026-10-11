"""Git-object admission rejects unregistered maintenance without downloading LFS."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upstream_maintenance_verifier', ROOT / 'scripts/verify_upstream_integration.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def git(root, *args, data=None):
    return subprocess.check_output(
        ['git', '-C', str(root), '-c', 'user.name=Admission Test',
         '-c', 'user.email=admission@example.invalid', *args], input=data)


@pytest.fixture
def repository(tmp_path):
    # No checkout, smudge filters, downloads, or changes to the real index.
    target = tmp_path / 'repository with spaces'
    subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(ROOT), str(target)], check=True)
    git(target, 'read-tree', 'HEAD')
    policy = target / verifier.MAINTENANCE
    policy.parent.mkdir(parents=True)
    policy.write_bytes((ROOT / verifier.MAINTENANCE).read_bytes())
    return target


def stage_blob(root, path, content, mode='100644'):
    blob = git(root, 'hash-object', '-w', '--stdin', data=content).decode().strip()
    git(root, 'update-index', '--add', '--cacheinfo', mode + ',' + blob + ',' + path)
    return blob


def commit_index(root):
    tree = git(root, 'write-tree').decode().strip()
    commit = git(root, 'commit-tree', tree, '-p', 'HEAD', data=b'Unregistered test change\n').decode().strip()
    git(root, 'update-ref', 'HEAD', commit)


def test_actual_head_and_index_verify_historical_merge_and_four_registered_paths(repository):
    for staged in (False, True):
        result = verifier.verify(repository, staged=staged)
        assert result['upstream_files_verified'] == 1017
        # The four initial paths remain covered when a later, explicitly
        # registered migration adds further committed maintenance chains.
        policy = json.loads((repository / verifier.MAINTENANCE).read_text())
        assert result['registered_maintenance_files'] == len(policy['files']) >= 4
        assert result['historical_merge_verified'] == verifier.MERGE_COMMIT
        assert result['selection'] == ('index' if staged else 'HEAD')
        assert result['LFS_payloads_verified'] is False
        assert result['worktree_sources_verified'] is False


@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('path', ['.dockerignore', 'README.md'])
def test_unregistered_changes_fail_in_both_selected_git_states(repository, staged, path):
    stage_blob(repository, path, b'Unregistered content\n')
    if not staged:
        commit_index(repository)
    with pytest.raises(RuntimeError, match='differs from registered'):
        verifier.verify(repository, staged=staged)


@pytest.mark.parametrize('staged', [False, True])
def test_modified_original_catalog_cannot_redefine_its_own_inventory(repository, staged):
    catalog = json.loads(git(repository, 'cat-file', 'blob', verifier.CATALOG_BLOB))
    catalog['files'].pop()
    catalog['upstream_file_count'] -= 1
    stage_blob(repository, verifier.CATALOG, json.dumps(catalog).encode())
    if not staged:
        commit_index(repository)
    with pytest.raises(RuntimeError, match='Frozen upstream catalog changed'):
        verifier.verify(repository, staged=staged)


@pytest.mark.parametrize('defect', [
    'duplicate', 'unknown_path', 'traversal', 'broken_chain', 'missing_step',
    'wrong_mode', 'wrong_blob', 'wrong_commit', 'unknown_commit', 'wrong_origin', 'wrong_lfs',
])
def test_policy_must_prove_each_complete_maintenance_chain(repository, defect):
    path = repository / verifier.MAINTENANCE
    policy = json.loads(path.read_text())
    row, first = policy['files'][0], policy['files'][0]['updates'][0]
    if defect == 'duplicate':
        policy['files'].append(deepcopy(row))
    elif defect == 'unknown_path':
        row['target_path'] = 'not-an-upstream-file'
    elif defect == 'traversal':
        row['target_path'] = '../.gitattributes'
    elif defect == 'broken_chain':
        row['updates'][1]['before'] = deepcopy(first['before'])
    elif defect == 'missing_step':
        row['updates'].pop(0)
    elif defect == 'wrong_mode':
        first['after']['mode'] = '100755'
    elif defect == 'wrong_blob':
        first['after']['git_blob'] = '0' * 40
    elif defect == 'wrong_commit':
        first['commit'] = row['updates'][1]['commit']
    elif defect == 'unknown_commit':
        first['commit'] = '0' * 40
    elif defect == 'wrong_origin':
        origins = [update['upstream_origin'] for item in policy['files']
                   for update in item['updates'] if 'upstream_origin' in update]
        assert len(origins) == 1
        origins[0]['commit'] = json.loads(
            git(repository, 'cat-file', 'blob', verifier.CATALOG_BLOB))['upstream_commit']
    else:
        origins = [update['upstream_origin'] for item in policy['files']
                   for update in item['updates'] if 'upstream_origin' in update]
        assert len(origins) == 1
        origins[0]['lfs_oid'] = '0' * 64
    path.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError):
        verifier.verify(repository)


def test_mode_changes_and_deletions_are_not_accepted(repository):
    original = verifier.git_entries(repository)['.dockerignore']
    git(repository, 'update-index', '--cacheinfo', '100755,' + original[1] + ',.dockerignore')
    with pytest.raises(RuntimeError, match='differs from registered'):
        verifier.verify(repository, staged=True)
    git(repository, 'update-index', '--force-remove', '.dockerignore')
    with pytest.raises(RuntimeError, match='missing or differs'):
        verifier.verify(repository, staged=True)


def test_worktree_files_cannot_override_selected_content_or_lfs_policy(repository):
    (repository / 'README.md').write_text('Unstaged source is outside HEAD/index verification\n')
    (repository / verifier.CATALOG).write_text('{}\n')
    (repository / '.lfsconfig').write_bytes(git(repository, 'show', 'HEAD:.lfsconfig'))
    assert verifier.verify(repository)['worktree_sources_verified'] is False
    stage_blob(repository, '.lfsconfig', b'[lfs]\nurl = https://invalid.example/lfs\npushurl = https://invalid.example/lfs\n')
    with pytest.raises(RuntimeError, match='incorrect LFS endpoint'):
        verifier.verify(repository, staged=True)
    assert verifier.verify(repository)['selection'] == 'HEAD'


def test_unresolved_index_conflict_is_rejected(repository):
    mode, blob = verifier.git_entries(repository)['.dockerignore']
    git(repository, 'update-index', '--force-remove', '.dockerignore')
    git(repository, 'update-index', '--index-info', data=(mode + ' ' + blob + ' 1\t.dockerignore\n').encode())
    with pytest.raises(RuntimeError, match='Unresolved merge conflict'):
        verifier.verify(repository, staged=True)
