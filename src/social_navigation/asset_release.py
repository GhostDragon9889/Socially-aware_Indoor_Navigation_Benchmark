"""Fetch and verify private model/scene releases without launching Isaac Sim."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


RELEASE_PATH = Path('resources/releases/pedestrians_20261010')
LFS_URL = 'https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git/info/lfs'
BUILTIN_MDL = {'OmniPBR.mdl', 'OmniGlass.mdl', 'OmniSurface.mdl'}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Invalid manifest path: {relative}')
    result = (root / part).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f'Missing or escaping release file: {relative}')
    return result


def verify(release: Path, *, check_usd: bool = False) -> dict:
    release = release.resolve()
    manifest = json.loads((release / 'manifest.json').read_text())
    if manifest.get('schema') != 'social_navigation.github_asset_release.v1':
        raise ValueError('Unsupported release manifest schema')
    rows = manifest['files']
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate release paths')
    original_usd_checked = 0
    for row in rows:
        path = contained_file(release, row['path'])
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f'Checksum mismatch: {row["path"]}; fetch LFS objects before verifying')
        if row['role'] == 'exact_original_usd_gzip':
            checksum = hashlib.sha256()
            count = 0
            with gzip.open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    checksum.update(chunk)
                    count += len(chunk)
            if checksum.hexdigest() != row['source_sha256'] or count != row['uncompressed_bytes']:
                raise ValueError(f'Original USD snapshot differs: {row["path"]}')
            original_usd_checked += 1
    result = {'status': 'verified', 'files': len(rows), 'bytes': sum(row['bytes'] for row in rows),
              'original_usd_snapshots_verified': original_usd_checked,
              'physics_rerun': False}
    if check_usd:
        # OpenUSD is optional; the regular verifier uses Python's standard library.
        from pxr import Sdf, UsdUtils
        catalog = json.loads((release / 'catalog.json').read_text())
        prefix = manifest['repository_relative_root'] + '/'
        entries = [catalog['robot']['stage']]
        for scene in catalog['scenes'].values():
            entries.extend((scene['current_stage'], scene['tested_static_stage_relocated']))
        checks = []
        for entry in entries:
            if not entry.startswith(prefix):
                raise ValueError(f'Invalid catalog stage: {entry}')
            path = contained_file(release, entry[len(prefix):])
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
            local = [Path(layer.realPath).resolve() for layer in layers if layer.realPath]
            local += [Path(asset).resolve() for asset in assets]
            if not all(item.is_relative_to(release) and item.is_file() for item in local):
                raise ValueError(f'USD dependency escapes release or is missing: {entry}')
            if set(unresolved) - BUILTIN_MDL:
                raise ValueError(f'Unresolved USD asset: {entry}: {unresolved}')
            checks.append({'stage': entry, 'local_dependencies': len(local),
                           'sdk_mdl_dependencies': sorted(unresolved)})
        result['usd_stages'] = checks
    return result


def fetch(root: Path, remote: str | None = None) -> None:
    # The original upstream files still use the NVlabs download endpoint.
    # Override it only for this explicitly scoped private release download.
    if remote is None:
        remotes = subprocess.check_output(['git', '-C', str(root), 'remote'], text=True).splitlines()
        for candidate in remotes:
            url = subprocess.check_output(['git', '-C', str(root), 'remote', 'get-url', candidate], text=True).strip()
            if 'GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark' in url:
                remote = candidate
                break
        if remote is None:
            raise ValueError('No project GitHub remote found; specify --remote explicitly')
    subprocess.run(['git', '-C', str(root), '-c', f'lfs.url={LFS_URL}',
                    '-c', 'lfs.fetchexclude=', 'lfs', 'pull', remote,
                    f'--include={RELEASE_PATH.as_posix()}/**,resources/pedestrian_visuals/**', '--exclude='], check=True)


def install_models(root: Path, release: Path) -> dict:
    manifest = json.loads((release / 'manifest.json').read_text())
    rows = [row for row in manifest['files'] if 'install_path' in row]
    targets = []
    for row in rows:
        relative = Path(row['install_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('data',):
            raise ValueError(f'Invalid install path: {relative}')
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError(f'Install path escapes checkout: {relative}')
        if target.exists() and (not target.is_file() or sha256(target) != row['sha256']):
            raise ValueError(f'Existing resource differs; retained without overwrite: {target}')
        targets.append((row, target))
    copied = 0
    for row, target in targets:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained_file(release, row['path']), target)
            copied += 1
        if sha256(target) != row['sha256']:
            raise ValueError(f'Installed model differs: {target}')
    return {'status': 'models_installed', 'resources': len(rows), 'new_files': copied,
            'amass_motion_data_included': False, 'native_map_paths_migrated': False}


def main(argv=None, *, default_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'verify', 'install-models', 'prepare', 'catalog'))
    parser.add_argument('--root', type=Path, default=default_root or Path.cwd())
    parser.add_argument('--check-usd', action='store_true', help='Also check local USD dependency closure (requires pxr)')
    parser.add_argument('--remote', help='Git remote name; detected from URL by default')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    release = root / RELEASE_PATH
    if args.action == 'fetch':
        fetch(root, args.remote)
    if args.action == 'catalog':
        print((release / 'catalog.json').read_text(), end='')
        return 0
    result = verify(release, check_usd=args.check_usd)
    if args.action in ('install-models', 'prepare'):
        result['installation'] = install_models(root, release)
    if args.action == 'prepare':
        from .reproduction import prepare_resources
        result['preparation'] = prepare_resources(root, release)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
