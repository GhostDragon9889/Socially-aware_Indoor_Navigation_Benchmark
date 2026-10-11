"""Restore the Git release and pinned native runtime in a fresh checkout."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import stat

from .asset_release import contained_file, sha256


def prepare_frozen_inputs(root: Path):
    """Restore permissions lost by Git only after verifying every frozen byte."""
    root = root.resolve()
    manifest = json.loads((root / 'configs/reproduction/frozen_inputs.json').read_text())
    if manifest.get('schema') != 'social_navigation.frozen_inputs.v1':
        raise ValueError('Unknown frozen inputs schema')
    checked = []
    for row in manifest['files']:
        path = root / row['path']
        info = path.lstat()
        if (not path.resolve().is_relative_to(root) or path.is_symlink()
                or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size != row['bytes'] or sha256(path) != row['sha256']):
            raise ValueError('Frozen input differs; no permissions changed: ' + str(path))
        checked.append((path, info.st_mode))
    for path, mode in checked:
        path.chmod(stat.S_IMODE(mode) & ~0o222)
    return {'status': 'frozen_input_permissions_prepared', 'files': len(checked),
            'content_rewritten': False, 'physics_tested': False}


def _copy_exact(source, target, expected):
    if target.exists():
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Existing resource differs; retained: {target}')
        return False
    if sha256(source) != expected:
        raise ValueError(f'Resource source differs: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise ValueError(f'Copy checksum failed: {target}')
    return True


def prepare_resources(root: Path, release: Path):
    """Restore exact tested USD layers; raw evidence and hashes stay unchanged."""
    bindings = json.loads((root / 'configs/reproduction/resource_bindings.json').read_text())
    if sha256(release / 'manifest.json') != bindings['release_manifest_sha256']:
        raise ValueError('Release manifest differs from the committed resource bindings')
    restorations = bindings['restorations']
    copied = 0
    for row in restorations:
        source = contained_file(release, row['release_path'])
        target = (root / row['target']).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Restoration target escapes checkout')
        if row.get('gzip'):
            if target.exists():
                if sha256(target) != row['sha256']:
                    raise ValueError(f'Existing exact USD differs: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            # Exclusive creation retains a failed partial restoration for diagnosis.
            with gzip.open(source, 'rb') as incoming, temporary.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
            if temporary.stat().st_size != row['bytes'] or sha256(temporary) != row['sha256']:
                raise ValueError(f'Exact USD decompression checksum failed: {temporary}')
            temporary.rename(target)
            copied += 1
        else:
            copied += int(_copy_exact(source, target, row['sha256']))
    visuals = root / 'resources/pedestrian_visuals'
    visual_rows = json.loads((visuals / 'manifest.json').read_text())['files']
    for row in visual_rows:
        source = contained_file(visuals, row['path'])
        target = root / row['install_path']
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Visual resource installation escapes checkout')
        copied += int(_copy_exact(source, target, row['sha256']))
    return {'status': 'resources_prepared', 'new_files': copied,
            'original_receipts_rewritten': False, 'original_USD_bytes_restored': True,
            'licensed_motion_banks_included': False, 'physics_rerun': False}


def prepare_native_runtime(root: Path):
    """Copy matching installed SDK support, applying only the bundled native plugin.

    No installed SDK file is changed. The full original manifest stays byte exact;
    SDK and repository locations are derived separately by the terminal adapter.
    """
    from .sdk_paths import sim_root
    from importlib import metadata
    if metadata.version('isaacsim') == '6.1.0.0':
        from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle
        report = {}
        SDK61Lifecycle(report)
        return {'status': 'stock_sim61_runtime', 'files': 0,
                'native_overlay_loaded': False, 'installed_SDK_changed': False,
                'runtime_versions': report['evaluation_lifecycle']['runtime_versions'],
                'physics_rerun': False}
    bundle = root / 'env/isaaclab3_sim6/native_shutdown'
    manifest_path = bundle / 'original_build_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    destination = root / 'output/native_sdk_overlay/simulation_manager_shutdown_v1'
    overlay = destination / 'isaacsim.core.simulation_manager'
    installed = sim_root() / 'exts/isaacsim.core.simulation_manager'
    copied = 0
    expected_paths = set()
    for row in manifest['copied_file_records']:
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Native overlay manifest path escapes extension')
        source = installed / relative
        if not source.is_file() or sha256(source) != row['installed_source']['sha256']:
            raise ValueError(f'Installed Sim 6.0 source/binary differs: {source}')
        if relative.as_posix() == 'bin/libisaacsim.core.simulation_manager.plugin.so':
            source = bundle / relative.name
        target = overlay / relative
        copied += int(_copy_exact(source, target, row['sha256']))
        if target.is_symlink() or target.stat().st_nlink != 1:
            raise ValueError('Native runtime inputs must be independent files')
        target.chmod(0o444)
        expected_paths.add(target.resolve())
    if expected_paths != {p.resolve() for p in overlay.rglob('*') if p.is_file()}:
        raise ValueError('Unexpected file in owned native runtime')
    _copy_exact(manifest_path, destination / 'build_copy_manifest.json', sha256(manifest_path))
    return {'status': 'native_runtime_prepared', 'files': len(expected_paths),
            'new_files': copied, 'installed_SDK_changed': False, 'physics_rerun': False}
