#!/usr/bin/env python3
"""Check or apply pinned optional reference source patches; never fetch or commit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def inside(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Invalid relative path: {relative}')
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f'Path escapes the selected checkout: {relative}')
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'apply'))
    parser.add_argument('--reference-root', type=Path, default=ROOT / 'third_party')
    parser.add_argument('--patch-root', type=Path, default=ROOT / 'configs/compatibility/reference_patches')
    parser.add_argument('--repository', choices=('navdp','flux','pacer'), action='append')
    args = parser.parse_args()
    patch_root = args.patch_root.resolve()
    manifest = json.loads((patch_root / 'manifest.json').read_text())
    if manifest.get('schema') != 'social_navigation.reference_compat_patches.v1':
        raise ValueError('Unsupported patch manifest')
    selected = args.repository or ['navdp','flux','pacer']
    plans=[]
    for row in manifest['repositories']:
        if row['name'] not in selected:
            continue
        checkout = inside(args.reference_root.resolve(),row['name'])
        if not checkout.is_dir():
            raise FileNotFoundError(f'Missing optional checkout: {checkout}')
        head = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        if head != row['base_commit']:
            raise ValueError(f'{row["name"]}: expected base commit {row["base_commit"]}, got {head}')
        patch = inside(patch_root,row['patch'])
        if digest(patch) != row['patch_sha256']:
            raise ValueError(f'{row["name"]}: patch hash mismatch')
        states = [(f,digest(inside(checkout,f['path']))) for f in row['files']]
        patched = all(actual == f['patched_sha256'] for f,actual in states)
        base = all(actual == f['base_sha256'] for f,actual in states)
        if not patched and not base:
            raise ValueError(f'{row["name"]}: mixed or foreign modifications; preserved')
        if base and not patched:
            subprocess.run(['git','-C',str(checkout),'apply','--check',str(patch)],check=True)
        plans.append((row,checkout,patch,'already_patched' if patched else 'ready_to_apply'))
    if len(plans) != len(set(selected)):
        raise ValueError('Missing repository in manifest')
    results=[]
    for row,checkout,patch,status in plans:
        if args.action=='apply' and status=='ready_to_apply':
            subprocess.run(['git','-C',str(checkout),'apply',str(patch)],check=True)
            for f in row['files']:
                if digest(inside(checkout,f['path'])) != f['patched_sha256']:
                    raise RuntimeError(f'{row["name"]}: applied bytes disagree with manifest')
            status='applied_and_verified'
        results.append(dict(repository=row['name'],base_commit=row['base_commit'],status=status,files=len(row['files'])))
    print(json.dumps(dict(action=args.action,repositories=results,commits_or_pushes=False),indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
