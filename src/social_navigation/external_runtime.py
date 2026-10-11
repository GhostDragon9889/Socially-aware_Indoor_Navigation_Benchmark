"""Bind a checkout to an external SDK Python without changing that SDK.

Project additions are ordinary packages, not another interpreter or a second
simulator installation. All launches use the selected SDK executable directly.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import tomllib

from . import runtime_environment as runtime

BINDING = Path('.runtime/external_sdk.json')
ADDONS = Path('.runtime/external_sdk_packages')


def native_paths(root: Path) -> list[str]:
    """Admit only the optional pinned, hash-checked RVO2 extension."""
    directory = root / '.runtime/rvo2_native'
    receipt = directory / 'receipt.json'
    if not receipt.is_file():
        return []
    if directory.is_symlink() or receipt.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Optional RVO2 dependency must remain inside the checkout')
    value = json.loads(receipt.read_text())
    name = value.get('binary', '')
    if (value.get('schema') != 'social_navigation.rvo2_native.v1'
            or value.get('source_commit') != 'c2c46ba8d59556aa10faf03479293236efea154d'
            or Path(name).name != name or not name.startswith('rvo2.cpython-312-') or not name.endswith('.so')
            or (directory / name).is_symlink()
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != value.get('sha256')):
        raise RuntimeError('Optional RVO2 dependency differs from its pinned build receipt')
    return [str(directory)]


def read_binding(root: Path) -> dict:
    path = root / BINDING
    value = json.loads(path.read_text())
    if value.get('schema') != 'social_navigation.external_sdk.v1':
        raise RuntimeError('Unknown external SDK binding schema')
    for key in ('lab_root', 'python', 'addons'):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise RuntimeError('External SDK binding requires absolute ' + key)
    if Path(value['addons']).resolve() != (root / ADDONS).resolve():
        raise RuntimeError('Project additions belong to another checkout; configure this clone first')
    if not Path(value['python']).is_file():
        raise RuntimeError('Configured SDK Python is missing; configure an installed SDK')
    return value


def selected_python(root: Path) -> Path:
    """Use explicit project configuration, otherwise the invoking interpreter."""
    if (root / BINDING).is_file():
        return Path(read_binding(root)['python'])
    return Path(sys.executable)


def _check_addons(root: Path, sdk: dict, profile: dict, pins: dict, addons: Path) -> dict:
    requirements = tomllib.loads((root / 'pyproject.toml').read_text())['project']['dependencies']
    probe = runtime._probe_python(Path(sdk['python']), profile, pins=pins,
                                  requirements=requirements, addons=addons)
    sdk_site = Path(sdk['probe']['site_packages'])
    for name, path in {**probe['providers'], **probe['module_providers']}.items():
        if not runtime._inside(path, sdk_site):
            raise RuntimeError(f'Project additions replaced SDK provider {name}: {path}')
    return probe


def configure(root: Path, lab: Path, *, python: Path | None = None,
              install_addons: bool = True, rebuild_addons: bool = False) -> dict:
    """Validate first, atomically install locked additions, then publish binding."""
    root, lab = root.expanduser().resolve(), lab.expanduser().resolve()
    # Do not resolve the Python symlink: its venv spelling selects SDK sys.prefix.
    python = Path(os.path.abspath(python.expanduser())) if python else lab / '.venv/bin/python'
    profile, pins = runtime.load_runtime_profile(root)
    sdk = runtime._sdk_snapshot(lab, profile, python=python)
    parent, addons = root / '.runtime', root / ADDONS
    if not runtime._inside(parent, root) or runtime._inside(parent, lab):
        raise RuntimeError('Project additions must stay inside this checkout and outside the SDK')
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / '.external_sdk.install.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is configuring this checkout') from exc
        fingerprint = runtime._fingerprint(root, profile, sdk)
        receipt = addons / 'social_navigation_addons_receipt.json'
        if addons.exists() and not rebuild_addons:
            previous = json.loads(receipt.read_text()).get('fingerprint', {}) if receipt.is_file() else {}
            # A second matching SDK can live elsewhere. Package bytes belong to
            # the project; actual requirements/provider ownership are reprobed.
            keys = ('profile_sha256', 'overlay_lock_sha256', 'project_manifest_sha256',
                    'sdk_source_tree', 'sdk_python_version')
            if not previous or any(previous.get(key) != fingerprint[key] for key in keys):
                raise RuntimeError('Existing additions differ from selected SDK/profile; preserved. Use --rebuild-addons')
            probe = _check_addons(root, sdk, profile, pins, addons)
        else:
            if not install_addons:
                raise RuntimeError('Project additions missing; run configure without --check-only')
            uv = shutil.which('uv')
            if uv is None:
                raise RuntimeError('Install uv to download project additions')
            candidate = Path(tempfile.mkdtemp(prefix='.external_sdk_packages-', dir=parent))
            backup, published = None, False
            try:
                partial = candidate / 'requirements.txt'
                partial.write_text(''.join(f'{n}=={v}\n' for n, v in pins.items() if n != 'chumpy'))
                runtime._run([uv, 'pip', 'install', '--python', str(python), '--target', str(candidate),
                              '--no-deps', '-r', str(partial)], log=candidate / 'install.log')
                if 'chumpy' in pins:
                    runtime._run([uv, 'pip', 'install', '--python', str(python), '--target', str(candidate),
                                  '--no-deps', '--no-build-isolation', f"chumpy=={pins['chumpy']}"],
                                 log=candidate / 'install.log')
                probe = _check_addons(root, sdk, profile, pins, candidate)
                (candidate / receipt.name).write_text(json.dumps({'fingerprint': fingerprint}, indent=2))
                if addons.exists():
                    backup = parent / ('.external_sdk_packages_backup-' + candidate.name.removeprefix('.external_sdk_packages-'))
                    addons.rename(backup)
                try:
                    candidate.rename(addons)
                    published = True
                except BaseException:
                    if backup is not None: backup.rename(addons)
                    raise
                probe = _check_addons(root, sdk, profile, pins, addons)
            except BaseException:
                if published:
                    shutil.rmtree(addons)
                    if backup is not None: backup.rename(addons)
                if (candidate / 'install.log').is_file():
                    shutil.copyfile(candidate / 'install.log', parent / 'external_sdk_failed_install.log')
                shutil.rmtree(candidate, ignore_errors=True)
                raise
        binding = {'schema': 'social_navigation.external_sdk.v1', 'lab_root': str(lab),
                   'python': str(python), 'addons': str(addons), 'sdk_modified': False,
                   'project_venv_required': False, 'fingerprint': fingerprint,
                   'sdk_probe': sdk['probe'], 'project_probe': probe, 'physics_tested': False}
        pending = parent / 'external_sdk.json.partial'
        with pending.open('x') as stream:
            json.dump(binding, stream, indent=2)
        pending.replace(root / BINDING)
        workspace = parent / 'external_sdk.code-workspace'
        env_file = parent / 'external_sdk.env'
        import_paths = os.pathsep.join([str(root / 'src'), str(root), str(addons), *native_paths(root)])
        env_file.write_text('PYTHONPATH=' + json.dumps(import_paths, ensure_ascii=False)
                            + '\nPYTHONDONTWRITEBYTECODE=1\nPYTEST_DISABLE_PLUGIN_AUTOLOAD=1\n')
        workspace.write_text(json.dumps({'folders': [{'path':str(root),'name':root.name}],
            'settings': {'python.defaultInterpreterPath':sdk['probe']['python'],
                         'python.envFile':str(env_file),
                         'terminal.integrated.env.linux': {'PYTHONPATH':import_paths,
                                                          'ISAACLAB_ROOT':str(lab)}}}, indent=2)+'\n')
        return {'status': 'external_sdk_configured', 'python': str(python),
                'binding': str(root / BINDING), 'vscode_workspace':str(workspace),
                'sdk_modified': False, 'project_venv_required': False}


def check(root: Path) -> dict:
    value = read_binding(root)
    profile, pins = runtime.load_runtime_profile(root)
    sdk = runtime._sdk_snapshot(Path(value['lab_root']), profile, python=Path(value['python']))
    if value['fingerprint'] != runtime._fingerprint(root, profile, sdk):
        raise RuntimeError('External SDK binding differs from current SDK/profile; configure again')
    probe = _check_addons(root, sdk, profile, pins, Path(value['addons']))
    return {'status': 'external_sdk_verified', 'python': value['python'],
            'sdk_modified': False, 'physics_tested': False, 'project_probe': probe}


def launch(root: Path, arguments: list[str]) -> int:
    """Forward ordinary Python arguments and return the real subprocess status."""
    value = read_binding(root)
    environment = dict(os.environ, ISAACLAB_ROOT=value['lab_root'], SOCIAL_NAVIGATION_ROOT=str(root),
                       SOCIAL_NAVIGATION_SDK_PYTHON=value['python'], PYTHONNOUSERSITE='1')
    # Children launched with sys.executable need the same project/additions paths.
    paths = [str(root / 'src'), str(root), value['addons'], *native_paths(root)]
    if os.environ.get('PYTHONPATH'):
        paths.append(os.environ['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    command = [value['python'], str(root / 'scripts/run_external_python.py'),
               '--sdk-child', *arguments]
    result = subprocess.run(command, cwd=root, env=environment)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def execute(root: Path, arguments: list[str]) -> None:
    """Run in the SDK process, preserving sys.executable and standard imports."""
    value = read_binding(root)
    expected = value.get('sdk_probe', {}).get('python', value['python'])
    if Path(sys.executable).absolute() != Path(expected).absolute():
        raise RuntimeError('Select the configured SDK interpreter before using --sdk-child')
    # Debugger launches enter here directly. Populate the child environment as
    # well as sys.path so forwarded CLI tools inherit project additions.
    os.environ.update(ISAACLAB_ROOT=value['lab_root'], SOCIAL_NAVIGATION_ROOT=str(root),
                      SOCIAL_NAVIGATION_SDK_PYTHON=value['python'], PYTHONNOUSERSITE='1')
    paths = [str(root / 'src'), str(root), value['addons'], *native_paths(root)]
    inherited = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths + [p for p in inherited if p]))
    site = __import__('site')
    site.addsitedir(value['addons'])
    for path in (*native_paths(root), value['addons'], str(root), str(root / 'src')):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    if not arguments:
        raise SystemExit('Provide a Python script, -m module, or -c code')
    target, *rest = arguments
    if target in ('-m', '-c'):
        if not rest:
            raise SystemExit(target + ' requires an argument')
        name, *params = rest
        sys.argv = [name if target == '-m' else '-c', *params]
        if target == '-m':
            runpy.run_module(name, run_name='__main__', alter_sys=True)
        else:
            exec(compile(name, '<string>', 'exec'), {'__name__': '__main__', '__builtins__': __builtins__})
    elif target == '-':
        sys.argv = ['-', *rest]
        exec(compile(sys.stdin.read(), '<stdin>', 'exec'), {'__name__': '__main__', '__builtins__': __builtins__})
    else:
        path = Path(target).expanduser().resolve()
        sys.path.insert(0, str(path.parent))
        sys.argv = [str(path), *rest]
        runpy.run_path(str(path), run_name='__main__')
