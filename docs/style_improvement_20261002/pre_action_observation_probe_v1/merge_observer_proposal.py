"""Bounded three-new-file proposal; dry run by default, no SDK or GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[2]
ALLOWED = (
    'CrowdSim/humanoid/runtime/pre_action_observation.py',
    'scripts/probe_paper_pre_action.py',
    'tests/test_pre_action_observation.py',
)


def record(path):
    payload = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def verify():
    manifest = json.loads((DOC / 'merge_manifest.json').read_text())
    if tuple(row['relative_path'] for row in manifest['files']) != ALLOWED:
        raise ValueError('Exact three-target allowlist mismatch')
    total = 0
    for row in manifest['files']:
        target = ROOT / row['relative_path']
        source = DOC / 'proposed' / row['relative_path']
        if target.resolve() != target or source.resolve() != source:
            raise ValueError('Symlink/redirected payload or target rejected')
        if target.exists() or target.is_symlink() or row['before'] is not None:
            raise ValueError('Only absent new targets are allowed; no overwrite')
        if record(source) != row['after'] or row['after']['bytes'] > 50000:
            raise ValueError('Payload SHA/size mismatch')
        total += row['after']['bytes']
    if total != manifest['total_proposed_bytes'] or total > 100000:
        raise ValueError('Exact total bytes/100KiB hard bound mismatch')
    for guarded in [manifest['protected_source174_manifest'],
                    *manifest['protected_unchanged_files']]:
        if record(Path(guarded['path'])) != guarded:
            raise ValueError('Frozen source174 bytes changed')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    manifest = verify()
    if not args.apply:
        print(json.dumps({'status': 'dry_run_verified_no_production_writes',
                          'targets': list(ALLOWED),
                          'bytes': manifest['total_proposed_bytes']}))
        return 0
    receipt = DOC / 'merge_receipt.json'
    if receipt.exists():
        raise ValueError('No repeated merge or receipt overwrite')
    pending = []
    try:
        for row in manifest['files']:
            target = ROOT / row['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.pre-action-', dir=target.parent)
            temporary = Path(name)
            pending.append((temporary, target, row))
            with os.fdopen(fd, 'wb') as stream:
                stream.write(Path(row['after']['path']).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o644)
        verify()
        for temporary, target, row in pending:
            if record(temporary)['sha256'] != row['after']['sha256']:
                raise ValueError('Prepared payload changed')
            # Exclusive publication: a raced existing file can never be replaced.
            os.link(temporary, target)
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        published = [record(ROOT / name) for name in ALLOWED]
        assert all(a['sha256'] == b['after']['sha256']
                   for a, b in zip(published, manifest['files']))
        with receipt.open('x') as stream:
            json.dump({'status': 'three_new_files_published', 'published': published,
                       'SDK_started': False, 'GPU_used': False,
                       'existing_production_files_modified': False}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({'status': 'three_new_files_published', 'files': 3}))
    finally:
        for temporary, _, _ in pending:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
