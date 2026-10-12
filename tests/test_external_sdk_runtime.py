"""External Python, relocated clones, and hash-first frozen input restoration."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from social_navigation.external_runtime import BINDING, ADDONS, native_paths, read_binding, selected_python
from social_navigation.reproduction import prepare_frozen_inputs
from social_navigation.artifact_identity import artifact_path, same_artifact
from social_navigation.sdk_paths import PUBLISHED_PROJECT


def fixture_checkout(tmp_path):
    root = tmp_path / "clone's 目录 with spaces"
    root.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / 'src/social_navigation', root / 'src/social_navigation')
    shutil.copytree(source / 'src/protomotions', root / 'src/protomotions',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'configs').mkdir()
    (root / 'pyproject.toml').write_text('[project]\nname="test"\n')
    (root / 'scripts').mkdir()
    shutil.copyfile(source / 'scripts/run_external_python.py', root / 'scripts/run_external_python.py')
    (root / ADDONS).mkdir(parents=True)
    binding = {'schema':'social_navigation.external_sdk.v1', 'lab_root':'/unused/sdk',
               'python':sys.executable, 'addons':str(root / ADDONS)}
    (root / BINDING).write_text(json.dumps(binding))
    return root


@pytest.mark.parametrize('direct_debugger', [False, True])
def test_real_launch_preserves_sdk_executable_and_children(tmp_path, direct_debugger):
    root = fixture_checkout(tmp_path)
    (root / ADDONS / 'project_extra.py').write_text('answer=61\n')
    script = root / 'user script.py'
    script.write_text('import project_extra, sys, subprocess, json\n'
                      'child=subprocess.check_output([sys.executable,"-c","import project_extra; print(project_extra.answer)"],text=True)\n'
                      'print(json.dumps([sys.executable,sys.argv[1:],project_extra.answer,child.strip()]))\n')
    result = subprocess.run([sys.executable, str(root / 'scripts/run_external_python.py'),
                             *(['--sdk-child'] if direct_debugger else []),
                             str(script), 'argument with spaces', '中文'], capture_output=True, text=True,
                            env={**__import__('os').environ,'PYTHONPATH':''})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [sys.executable, ['argument with spaces','中文'],61,'61']
    assert selected_python(root) == Path(sys.executable)
    assert not (root / '.venv').exists()


def test_debugger_refuses_an_unconfigured_interpreter(tmp_path):
    root = fixture_checkout(tmp_path)
    value = json.loads((root / BINDING).read_text())
    value['sdk_probe'] = {'python':'/wrong/sdk/python'}
    (root / BINDING).write_text(json.dumps(value))
    result = subprocess.run([sys.executable,str(root / 'scripts/run_external_python.py'),
                             '--sdk-child','-c','raise SystemExit(0)'],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'configured SDK interpreter' in result.stderr


@pytest.mark.parametrize('arguments', [['-c', 'raise SystemExit(7)'],
                                      ['-m', 'project_exit']])
def test_launcher_propagates_actual_failure(tmp_path, arguments):
    root = fixture_checkout(tmp_path)
    (root / ADDONS / 'project_exit.py').write_text('raise SystemExit(7)\n')
    assert subprocess.run([sys.executable, str(root / 'scripts/run_external_python.py'),
                           *arguments], capture_output=True).returncode == 7


def test_binding_from_another_clone_is_rejected(tmp_path):
    root = fixture_checkout(tmp_path)
    value = json.loads((root / BINDING).read_text())
    value['addons'] = '/another/clone/.runtime/external_sdk_packages'
    (root / BINDING).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='another checkout'):
        read_binding(root)


def test_frozen_inputs_verify_every_file_before_any_chmod(tmp_path):
    directory = tmp_path / 'configs/reproduction'
    directory.mkdir(parents=True)
    a, b = tmp_path / 'a.py', tmp_path / 'b.json'
    a.write_text('pass\n'); b.write_text('{}\n')
    rows = [{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (a,b)]
    (directory / 'frozen_inputs.json').write_text(json.dumps({'schema':'social_navigation.frozen_inputs.v1','files':rows}))
    b.write_text('bad\n')
    with pytest.raises(ValueError, match='no permissions changed'):
        prepare_frozen_inputs(tmp_path)
    assert a.stat().st_mode & 0o222
    b.write_text('{}\n')
    assert prepare_frozen_inputs(tmp_path)['files'] == 2
    assert not a.stat().st_mode & 0o222
    assert not b.stat().st_mode & 0o222
    assert a.read_bytes() == b'pass\n'


def test_frozen_inputs_reject_external_symlinks(tmp_path):
    directory = tmp_path / 'configs/reproduction'
    directory.mkdir(parents=True)
    target = tmp_path / 'actual.py'; target.write_text('pass\n')
    (tmp_path / 'link.py').symlink_to(target)
    row = {'path':'link.py','bytes':5,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (directory / 'frozen_inputs.json').write_text(json.dumps({'schema':'social_navigation.frozen_inputs.v1','files':[row]}))
    with pytest.raises(ValueError): prepare_frozen_inputs(tmp_path)
    assert target.stat().st_mode & 0o222


def test_published_checkout_prefix_relocates_without_hiding_changed_bytes(tmp_path):
    old = {'path':PUBLISHED_PROJECT+'/src/CrowdSim/x.py','bytes':2,'sha256':'a'*64}
    new = {**old,'path':str(tmp_path/'src/CrowdSim/x.py')}
    assert artifact_path(old['path'],root=tmp_path) == Path(new['path'])
    assert same_artifact(old,new,root=tmp_path)
    assert not same_artifact(old,{**new,'sha256':'b'*64},root=tmp_path)
    assert artifact_path('/foreign/x.py',root=tmp_path) == Path('/foreign/x.py')


def test_nested_clone_relocation_is_idempotent(tmp_path):
    root = tmp_path/'output/clone workspace 中文'
    path = root/'src/CrowdSim/a.py'
    assert artifact_path(path,root=root) == path
    assert artifact_path(artifact_path(path,root=root),root=root) == path


def test_optional_native_extension_requires_pinned_identity(tmp_path):
    directory = tmp_path / '.runtime/rvo2_native'
    assert native_paths(tmp_path) == []
    directory.mkdir(parents=True)
    name = 'rvo2.cpython-312-x86_64-linux-gnu.so'
    binary = directory / name
    binary.write_bytes(b'recorded extension fixture; not loaded')
    value = {'schema':'social_navigation.rvo2_native.v1',
             'source_commit':'c2c46ba8d59556aa10faf03479293236efea154d',
             'binary':name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
    receipt = directory / 'receipt.json'
    receipt.write_text(json.dumps(value))
    assert native_paths(tmp_path) == [str(directory)]
    binary.write_bytes(b'changed extension')
    with pytest.raises(RuntimeError,match='pinned build receipt'):
        native_paths(tmp_path)
    value['binary'] = '../outside.so'
    receipt.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='pinned build receipt'):
        native_paths(tmp_path)
