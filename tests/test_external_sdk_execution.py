"""Exercise standard Python modes and reject invalid SDK/native input identities."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from social_navigation import external_runtime as external


@pytest.fixture
def bound_checkout(tmp_path):
    root = tmp_path / "checkout's 目录"; root.mkdir()
    addons = root / external.ADDONS; addons.mkdir(parents=True)
    value = {'schema': 'social_navigation.external_sdk.v1', 'lab_root': '/SDK/source',
             'python': sys.executable, 'addons': str(addons)}
    (root / external.BINDING).write_text(json.dumps(value))
    return root, value


@pytest.mark.parametrize('key,value', [('lab_root', 'relative'), ('python', None), ('addons', 61)])
def test_binding_requires_absolute_string_paths(bound_checkout, key, value):
    root, binding = bound_checkout
    binding[key] = value
    (root / external.BINDING).write_text(json.dumps(binding))
    with pytest.raises(RuntimeError, match='requires absolute ' + key):
        external.read_binding(root)


@pytest.mark.parametrize('value', [[], {'schema': 'unknown'}])
def test_binding_requires_known_schema(bound_checkout, value):
    root, _ = bound_checkout
    (root / external.BINDING).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='schema'): external.read_binding(root)


def test_binding_rejects_missing_sdk_python(bound_checkout):
    root, value = bound_checkout
    value['python'] = '/missing/sdk/python'
    (root / external.BINDING).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='Python is missing'): external.read_binding(root)


def test_binding_refuses_foreign_addons_symlink(bound_checkout, tmp_path):
    root, _ = bound_checkout
    (root / external.ADDONS).rmdir()
    foreign = tmp_path / 'foreign'; foreign.mkdir()
    (root / external.ADDONS).symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RuntimeError, match='inside this checkout'): external.read_binding(root)


def test_unconfigured_python_selection_uses_invoking_executable(tmp_path):
    assert external.selected_python(tmp_path) == Path(sys.executable)


def test_real_provider_ownership_check_accepts_sdk_paths(tmp_path, monkeypatch):
    (tmp_path / 'pyproject.toml').write_text('[project]\ndependencies=[]\n')
    probe = {'providers': {'torch': '/sdk/site/torch.dist-info'}, 'module_providers': {'pxr': '/sdk/site/pxr'}}
    monkeypatch.setattr(external.runtime, '_probe_python', lambda *a, **k: probe)
    assert external._check_addons(tmp_path, {'python': sys.executable, 'probe': {'site_packages': '/sdk/site'}}, {}, {}, tmp_path) == probe


@pytest.mark.parametrize('inherited', ['', '/user supplied path'])
@pytest.mark.parametrize('status,expected', [(0, 0), (7, 7), (-15, 143)])
def test_launch_preserves_environment_and_process_status(bound_checkout, monkeypatch, inherited, status, expected):
    root, value = bound_checkout
    monkeypatch.setenv('PYTHONPATH', inherited)
    observed = {}
    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=status)
    monkeypatch.setattr(external.subprocess, 'run', run)
    assert external.launch(root, ['-c', 'pass', 'quoted argument']) == expected
    assert observed['command'] == [sys.executable, str(root / 'scripts/run_external_python.py'), '--sdk-child', '-c', 'pass', 'quoted argument']
    env = observed['env']
    assert env['PYTHONNOUSERSITE'] == '1'
    assert env['ISAACLAB_ROOT'] == value['lab_root'] and observed['cwd'] == root
    assert (str(root / external.ADDONS) in env['PYTHONPATH'])
    if inherited: assert env['PYTHONPATH'].endswith(inherited)


@pytest.mark.parametrize('mode', ['-c', '-m', '-', 'file'])
def test_execute_python_modes_preserve_arguments_and_children(bound_checkout, monkeypatch, capsys, mode):
    root, value = bound_checkout
    body = ('import sys, json, subprocess\n'
            'child = subprocess.check_output([sys.executable,"-c","import sdk_mode_fixture; print(sdk_mode_fixture.answer)"],text=True)\n'
            'print(json.dumps([sys.argv[1:], sys.executable, child.strip()]))\n')
    (root / external.ADDONS / 'sdk_mode_fixture.py').write_text('answer=61\n')
    if mode == '-c': arguments = ['-c', body]
    elif mode == '-m':
        (root / external.ADDONS / 'sdk_mode_entry.py').write_text(body)
        arguments = ['-m', 'sdk_mode_entry']
    elif mode == '-':
        arguments = ['-']; monkeypatch.setattr(sys, 'stdin', io.StringIO(body))
    else:
        script = root / 'user script.py'; script.write_text(body)
        arguments = [str(script)]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [])
    monkeypatch.setenv('PYTHONPATH', str(root / external.ADDONS))
    # Register every variable touched by execute for automatic test restoration.
    for name in ('ISAACLAB_ROOT', 'SOCIAL_NAVIGATION_ROOT', 'SOCIAL_NAVIGATION_SDK_PYTHON', 'PYTHONNOUSERSITE'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    external.execute(root, [*arguments, 'argument with spaces', '中文'])
    assert json.loads(capsys.readouterr().out) == [['argument with spaces', '中文'], sys.executable, '61']
    sys.modules.pop('sdk_mode_fixture', None)


@pytest.mark.parametrize('arguments,message', [([], 'Provide a Python'), (['-m'], '-m requires'), (['-c'], '-c requires')])
def test_execute_rejects_missing_python_target(bound_checkout, monkeypatch, arguments, message):
    root, _ = bound_checkout
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in ('PYTHONPATH', 'ISAACLAB_ROOT', 'SOCIAL_NAVIGATION_ROOT', 'SOCIAL_NAVIGATION_SDK_PYTHON', 'PYTHONNOUSERSITE'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    with pytest.raises(SystemExit, match=message): external.execute(root, arguments)


def test_execute_rejects_wrong_interpreter(bound_checkout):
    root, value = bound_checkout
    value['sdk_probe'] = {'python': '/different/sdk/python'}
    (root / external.BINDING).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='configured SDK interpreter'):
        external.execute(root, ['-c', 'raise AssertionError("must not run")'])


def native_receipt(root):
    directory = root / '.runtime/rvo2_native'; directory.mkdir(parents=True)
    name = 'rvo2.cpython-312-x86_64-linux-gnu.so'
    (directory / name).write_bytes(b'native identity fixture only')
    value = {'schema': 'social_navigation.rvo2_native.v1', 'source_commit': 'c2c46ba8d59556aa10faf03479293236efea154d',
             'binary': name, 'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
    (directory / 'receipt.json').write_text(json.dumps(value))
    return directory, value


@pytest.mark.parametrize('kind', ['directory', 'receipt', 'binary'])
def test_native_dependency_symlinks_are_rejected(tmp_path, kind):
    root = tmp_path / 'root'; root.mkdir()
    directory, value = native_receipt(root)
    source = directory if kind == 'directory' else directory / ('receipt.json' if kind == 'receipt' else value['binary'])
    relocated = tmp_path / ('foreign-' + source.name)
    source.rename(relocated); source.symlink_to(relocated)
    with pytest.raises(RuntimeError, match='inside|pinned'): external.native_paths(root)


@pytest.mark.parametrize('key,value', [('schema', 'unknown'), ('source_commit', 'changed'), ('binary', 61), ('binary', '../escape.so'), ('binary', 'rvo2.cpython-310-linux.so'), ('binary', 'rvo2.cpython-312-linux.py'), ('sha256', '0'*64)])
def test_native_dependency_rejects_changed_identity(tmp_path, key, value):
    directory, receipt = native_receipt(tmp_path)
    receipt[key] = value
    (directory / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='pinned'): external.native_paths(tmp_path)


def test_native_receipt_must_be_an_object(tmp_path):
    directory, _ = native_receipt(tmp_path)
    (directory / 'receipt.json').write_text('[]')
    with pytest.raises(RuntimeError, match='object'): external.native_paths(tmp_path)
