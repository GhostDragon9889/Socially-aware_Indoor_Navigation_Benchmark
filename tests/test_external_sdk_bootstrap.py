"""Host-Python bootstrap contracts; the SDK execution is tested separately."""
import json
import os
from pathlib import Path
import runpy
import sys

import pytest


@pytest.fixture
def host_entry(tmp_path, monkeypatch):
    root = tmp_path / "host's 中文"; root.mkdir()
    project = Path(__file__).resolve().parents[1]
    entry = project / 'scripts/run_external_python.py'
    (root / '.runtime').mkdir()
    # Execute the actual entry source while substituting only its input binding.
    # No test changes the active checkout's interpreter selection.
    read_text = Path.read_text
    def input_binding(path, *args, **kwargs):
        if path == project / '.runtime/external_sdk.json':
            path = root / '.runtime/external_sdk.json'
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', input_binding)
    monkeypatch.setattr(sys, 'version_info', (3, 10, 20))
    monkeypatch.setattr(sys, 'argv', [str(entry), '-c', 'print(61)', 'argument with spaces'])
    monkeypatch.delenv('PYTHONNOUSERSITE', raising=False)
    return root, entry


def test_host_bootstrap_selects_sdk_before_project_imports(host_entry, monkeypatch):
    root, entry = host_entry
    (root / '.runtime/external_sdk.json').write_text(json.dumps({
        'schema': 'social_navigation.external_sdk.v1', 'python': sys.executable}))
    observed = {}
    class Dispatch(BaseException):
        pass
    def execv(python, arguments):
        observed.update(python=python, arguments=arguments, user_site=os.environ['PYTHONNOUSERSITE'])
        raise Dispatch
    monkeypatch.setattr(os, 'execv', execv)
    with pytest.raises(Dispatch): runpy.run_path(str(entry), run_name='__main__')
    assert observed == {'python': sys.executable,
                        'arguments': [sys.executable, str(entry), '--sdk-child', '-c', 'print(61)', 'argument with spaces'],
                        'user_site': '1'}


def test_old_host_cannot_enter_sdk_child_mode(host_entry, monkeypatch):
    _, entry = host_entry
    monkeypatch.setattr(sys, 'argv', [str(entry), '--sdk-child'])
    with pytest.raises(SystemExit, match='SDK Python3.12'):
        runpy.run_path(str(entry), run_name='__main__')


@pytest.mark.parametrize('binding', [{'schema': 'wrong', 'python': '/missing'},
                                    {'schema': 'social_navigation.external_sdk.v1', 'python': None},
                                    {'schema': 'social_navigation.external_sdk.v1', 'python': 'relative'},
                                    {'schema': 'social_navigation.external_sdk.v1', 'python': '/missing'}])
def test_old_host_invalid_selection_fails_before_exec(host_entry, binding):
    root, entry = host_entry
    (root / '.runtime/external_sdk.json').write_text(json.dumps(binding))
    with pytest.raises(SystemExit, match='valid external SDK'):
        runpy.run_path(str(entry), run_name='__main__')


def test_importing_entry_does_not_launch_a_user_program(monkeypatch):
    from social_navigation import external_runtime
    entry = Path(__file__).resolve().parents[1] / 'scripts/run_external_python.py'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    def forbidden(*args):
        raise AssertionError('Import must not launch a Python program')
    monkeypatch.setattr(external_runtime, 'launch', forbidden)
    monkeypatch.setattr(external_runtime, 'execute', forbidden)
    runpy.run_path(str(entry), run_name='entry_import_contract')
