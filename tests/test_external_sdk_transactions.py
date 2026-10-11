"""Fault injection for SDK configuration: preserve the old selection on failure."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import pytest

from social_navigation import external_runtime as external


@pytest.fixture
def sdk_checkout(tmp_path, monkeypatch):
    root = tmp_path / "project's 空 间"
    sdk = tmp_path / 'external SDK'
    root.mkdir(); sdk.mkdir()
    (sdk / '.venv/bin').mkdir(parents=True)
    (sdk / '.venv/bin/python').symlink_to(sys.executable)
    (root / 'pyproject.toml').write_text('[project]\ndependencies=[]\n')
    fingerprint = {key: key + '-identity' for key in (
        'profile_sha256', 'overlay_lock_sha256', 'project_manifest_sha256',
        'sdk_source_tree', 'sdk_python_version')}
    probe = {'python': sys.executable, 'site_packages': str(sdk / 'site'),
             'providers': {}, 'module_providers': {}}
    snapshot = {'python': sys.executable, 'probe': probe}
    monkeypatch.setattr(external.runtime, 'load_runtime_profile', lambda root: ({}, {'example': '1'}))
    monkeypatch.setattr(external.runtime, '_sdk_snapshot', lambda *a, **k: snapshot)
    monkeypatch.setattr(external.runtime, '_fingerprint', lambda *a: fingerprint.copy())
    monkeypatch.setattr(external.shutil, 'which', lambda name: '/uv fixture')
    def install(command, *, log):
        log.write_text('installer diagnostic retained\n')
        (log.parent / 'installed.txt').write_text('new bytes')
    monkeypatch.setattr(external.runtime, '_run', install)
    monkeypatch.setattr(external, '_check_addons', lambda *a: probe)
    return root, sdk, fingerprint


def existing_selection(root, fingerprint):
    addons = root / external.ADDONS
    addons.mkdir(parents=True)
    (addons / 'social_navigation_addons_receipt.json').write_text(json.dumps({'fingerprint': fingerprint}))
    (addons / 'installed.txt').write_text('old bytes')
    (root / external.BINDING).write_text('previous binding bytes\n')
    return addons


def assert_previous_selection(root):
    assert (root / external.BINDING).read_text() == 'previous binding bytes\n'
    assert (root / external.ADDONS / 'installed.txt').read_text() == 'old bytes'
    assert not list((root / '.runtime').glob('.external_sdk_packages-*'))


def test_native_admission_failure_cannot_replace_binding(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    def invalid_native(root):
        raise RuntimeError('invalid optional native receipt')
    monkeypatch.setattr(external, 'native_paths', invalid_native)
    with pytest.raises(RuntimeError, match='invalid optional'):
        external.configure(root, sdk)
    assert_previous_selection(root)


def test_post_publication_probe_failure_rolls_back_and_keeps_log(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    def probe(root, sdk, profile, pins, addons):
        if addons.name == external.ADDONS.name:
            raise RuntimeError('published provider validation failed')
        return sdk['probe']
    monkeypatch.setattr(external, '_check_addons', probe)
    with pytest.raises(RuntimeError, match='published provider'):
        external.configure(root, sdk, rebuild_addons=True)
    assert_previous_selection(root)
    assert 'installer diagnostic' in (root / '.runtime/external_sdk_failed_install.log').read_text()


def test_interrupted_binding_write_does_not_block_next_configuration(sdk_checkout):
    root, sdk, _ = sdk_checkout
    parent = root / '.runtime'; parent.mkdir()
    stale = parent / 'external_sdk.json.partial'
    stale.write_text('incomplete prior crash')
    result = external.configure(root, sdk)
    assert result['status'] == 'external_sdk_configured'
    assert external.read_binding(root)['lab_root'] == str(sdk)
    assert stale.read_text() == 'incomplete prior crash'


def test_rebuild_install_failure_preserves_selection_and_log(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    def fail(command, *, log):
        log.write_text('controlled network failure\n')
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(external.runtime, '_run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        external.configure(root, sdk, rebuild_addons=True)
    assert_previous_selection(root)
    assert 'controlled network failure' in (root / '.runtime/external_sdk_failed_install.log').read_text()


def test_candidate_provider_failure_does_not_publish(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    def fail(*args):
        raise RuntimeError('candidate provider changed')
    monkeypatch.setattr(external, '_check_addons', fail)
    with pytest.raises(RuntimeError, match='candidate provider'):
        external.configure(root, sdk, rebuild_addons=True)
    assert_previous_selection(root)


def test_existing_additions_fingerprint_mismatch_is_retained(sdk_checkout):
    root, sdk, _ = sdk_checkout
    existing_selection(root, {'different': 'SDK'})
    with pytest.raises(RuntimeError, match='preserved'):
        external.configure(root, sdk)
    assert_previous_selection(root)


def test_existing_additions_symlink_cannot_admit_foreign_directory(sdk_checkout):
    root, sdk, fingerprint = sdk_checkout
    foreign = sdk / 'foreign-project-addons'; foreign.mkdir()
    (foreign / 'social_navigation_addons_receipt.json').write_text(json.dumps({'fingerprint': fingerprint}))
    (root / '.runtime').mkdir()
    (root / external.ADDONS).symlink_to(foreign, target_is_directory=True)
    before = list(foreign.iterdir())
    with pytest.raises(RuntimeError, match='inside this checkout'):
        external.configure(root, sdk)
    assert list(foreign.iterdir()) == before
    assert not (root / external.BINDING).exists()


def test_concurrent_configuration_refuses_without_mutation(sdk_checkout):
    root, sdk, _ = sdk_checkout
    parent = root / '.runtime'; parent.mkdir()
    with (parent / '.external_sdk.install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another process'):
            external.configure(root, sdk)
    assert not (root / external.ADDONS).exists()
    assert not (root / external.BINDING).exists()


@pytest.mark.parametrize('previous', [False, True])
def test_failed_rollback_keeps_recovery_bytes_and_original_error(sdk_checkout, monkeypatch, previous):
    root, _, _ = sdk_checkout
    parent = root / '.runtime'; parent.mkdir()
    environment = parent / 'env'
    if previous:
        environment.write_bytes(b'original environment')
        environment.chmod(0o640)
    replace, unlink = Path.replace, Path.unlink
    publication_failed = False

    def fail_replace(source, target):
        nonlocal publication_failed
        if Path(target).name == 'workspace':
            publication_failed = True
            raise OSError('controlled publication failure')
        if publication_failed and Path(target) == environment:
            raise PermissionError('controlled rollback failure')
        return replace(source, target)

    def fail_unlink(path, *args, **kwargs):
        if publication_failed and path == environment:
            raise PermissionError('controlled rollback failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'replace', fail_replace)
    monkeypatch.setattr(Path, 'unlink', fail_unlink)
    with pytest.raises(OSError, match='controlled publication failure') as error:
        external._publish_setup(parent, {'env': 'new environment', 'workspace': 'new workspace'})
    assert 'controlled rollback failure' in '\n'.join(error.value.__notes__)
    recovery = list(parent.glob('.external_sdk_setup-*'))
    if previous:
        assert len(recovery) == 1 and recovery[0].read_bytes() == b'original environment'
        assert recovery[0].stat().st_mode & 0o777 == 0o640
        assert str(recovery[0]) in '\n'.join(error.value.__notes__)
    else:
        assert not recovery
        assert str(environment) in '\n'.join(error.value.__notes__)


@pytest.mark.parametrize('missing', ['check-only', 'uv'])
def test_missing_install_prerequisite_creates_no_binding(sdk_checkout, monkeypatch, missing):
    root, sdk, _ = sdk_checkout
    if missing == 'uv':
        monkeypatch.setattr(external.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='missing|Install uv'):
        external.configure(root, sdk, install_addons=missing != 'check-only')
    assert not (root / external.BINDING).exists()
    assert not (root / external.ADDONS).exists()


def test_successful_rebuild_retains_backup_and_publishes_matching_workspace(sdk_checkout):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    result = external.configure(root, sdk, rebuild_addons=True)
    value = external.read_binding(root)
    assert value['lab_root'] == str(sdk) and not value['sdk_modified']
    assert (root / external.ADDONS / 'installed.txt').read_text() == 'new bytes'
    backups = list((root / '.runtime').glob('.external_sdk_packages_backup-*'))
    assert len(backups) == 1 and (backups[0] / 'installed.txt').read_text() == 'old bytes'
    workspace = json.loads(Path(result['vscode_workspace']).read_text())
    assert workspace['settings']['python.defaultInterpreterPath'] == sys.executable
    assert str(root / external.ADDONS) in workspace['settings']['terminal.integrated.env.linux']['PYTHONPATH']
    assert external.check(root)['status'] == 'external_sdk_verified'


@pytest.mark.parametrize('name', ['external_sdk.env', 'external_sdk.code-workspace', 'external_sdk.json'])
@pytest.mark.parametrize('previous', [False, True])
def test_setup_publication_failure_restores_all_previous_files(sdk_checkout, monkeypatch, name, previous):
    root, sdk, fingerprint = sdk_checkout
    originals = {}
    if previous:
        existing_selection(root, fingerprint)
        for filename in ('external_sdk.env', 'external_sdk.code-workspace', 'external_sdk.json'):
            path = root / '.runtime' / filename
            if not path.exists(): path.write_text('old ' + filename)
            originals[filename] = path.read_bytes()
    replace = Path.replace
    failed = False
    def fail_once(source, target):
        nonlocal failed
        if not failed and Path(target).name == name:
            failed = True
            raise OSError('controlled setup publication failure')
        return replace(source, target)
    monkeypatch.setattr(Path, 'replace', fail_once)
    with pytest.raises(OSError, match='controlled setup'):
        external.configure(root, sdk, rebuild_addons=previous)
    assert failed
    for filename in ('external_sdk.env', 'external_sdk.code-workspace', 'external_sdk.json'):
        path = root / '.runtime' / filename
        if previous: assert path.read_bytes() == originals[filename]
        else: assert not path.exists()
    if previous: assert_previous_selection(root)
    else: assert not (root / external.ADDONS).exists()
    assert not list((root / '.runtime').glob('.external_sdk_setup-*'))


def test_addons_directory_publication_failure_restores_backup(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    rename = Path.rename
    def fail(source, target):
        if source.name.startswith('.external_sdk_packages-') and Path(target) == root / external.ADDONS:
            raise OSError('controlled addons directory rename failure')
        return rename(source, target)
    monkeypatch.setattr(Path, 'rename', fail)
    with pytest.raises(OSError, match='directory rename'):
        external.configure(root, sdk, rebuild_addons=True)
    assert_previous_selection(root)
    assert 'installer diagnostic' in (root / '.runtime/external_sdk_failed_install.log').read_text()


def test_failed_log_retention_does_not_hide_install_failure(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    def fail(command, *, log):
        log.write_text('original install failure')
        raise subprocess.CalledProcessError(9, command)
    monkeypatch.setattr(external.runtime, '_run', fail)
    def fail_copy(*args):
        raise PermissionError('diagnostic destination denied')
    monkeypatch.setattr(external.shutil, 'copyfile', fail_copy)
    with pytest.raises(subprocess.CalledProcessError) as error:
        external.configure(root, sdk, rebuild_addons=True)
    assert error.value.returncode == 9
    assert 'diagnostic destination denied' in str(error.value.__notes__)
    assert_previous_selection(root)


def test_workspace_symlink_cannot_write_into_external_sdk(sdk_checkout):
    root, sdk, fingerprint = sdk_checkout
    existing_selection(root, fingerprint)
    target = sdk / 'untouched.txt'; target.write_text('SDK original bytes')
    (root / '.runtime/external_sdk.code-workspace').symlink_to(target)
    with pytest.raises(RuntimeError, match='inside this checkout'):
        external.configure(root, sdk, rebuild_addons=True)
    assert target.read_text() == 'SDK original bytes'
    assert_previous_selection(root)
    assert not list((root / '.runtime').glob('.external_sdk_setup-*'))


def test_check_rejects_changed_sdk_fingerprint(sdk_checkout):
    root, sdk, fingerprint = sdk_checkout
    external.configure(root, sdk)
    fingerprint['sdk_python_version'] = 'changed SDK Python'
    with pytest.raises(RuntimeError, match='configure again'):
        external.check(root)


def test_matching_additions_are_reprobed_without_reinstallation(sdk_checkout, monkeypatch):
    root, sdk, fingerprint = sdk_checkout
    addons = existing_selection(root, fingerprint)
    def forbidden_install(*args, **kwargs):
        raise AssertionError('Matching additions must be reused')
    monkeypatch.setattr(external.runtime, '_run', forbidden_install)
    result = external.configure(root, sdk)
    assert result['status'] == 'external_sdk_configured'
    assert (addons / 'installed.txt').read_text() == 'old bytes'
    assert external.check(root)['status'] == 'external_sdk_verified'


def test_legacy_chumpy_install_uses_selected_sdk_and_no_resolver(sdk_checkout, monkeypatch):
    root, sdk, _ = sdk_checkout
    monkeypatch.setattr(external.runtime, 'load_runtime_profile', lambda root: ({}, {'example': '1', 'chumpy': '0.70'}))
    calls = []
    def install(command, *, log):
        calls.append(command)
        log.write_text('build diagnostic')
    monkeypatch.setattr(external.runtime, '_run', install)
    external.configure(root, sdk, python=Path(sys.executable))
    assert len(calls) == 2
    assert all(command[command.index('--python') + 1] == sys.executable for command in calls)
    assert all('--no-deps' in command for command in calls)
    assert '--no-build-isolation' in calls[1] and calls[1][-1] == 'chumpy==0.70'


def test_project_additions_cannot_override_sdk_provider(tmp_path, monkeypatch):
    root = tmp_path / 'project'; root.mkdir()
    (root / 'pyproject.toml').write_text('[project]\ndependencies=[]\n')
    sdk = {'python': sys.executable, 'probe': {'site_packages': '/selected/sdk/site'}}
    monkeypatch.setattr(external.runtime, '_probe_python', lambda *a, **k: {'providers': {'torch': str(root / 'torch')}, 'module_providers': {}})
    with pytest.raises(RuntimeError, match='replaced SDK provider torch'):
        external._check_addons(root, sdk, {}, {}, root / 'addons')
