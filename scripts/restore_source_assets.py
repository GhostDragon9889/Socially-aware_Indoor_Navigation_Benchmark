"""Copy verified redistributable assets from a materialized upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
POINTER = b'version https://git-lfs.github.com/spec/v1'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-upstream', type=Path, help='Upstream checkout with git-lfs payloads already fetched.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/publication/external_source_assets.json').read_text())
    records = []
    for entry in manifest['assets']:
        target = ROOT / entry['path']
        if target.is_file() and target.stat().st_size == entry['size'] and sha256(target) == entry['oid']:
            records.append({'path': entry['path'], 'status': 'verified'})
            continue
        if args.from_upstream is None:
            records.append({'path': entry['path'], 'status': 'missing_payload'})
            continue
        source = args.from_upstream.resolve() / entry['upstream_path']
        if not source.is_file() or source.stat().st_size != entry['size'] or sha256(source) != entry['oid']:
            raise ValueError('Missing or mismatched upstream payload: ' + entry['upstream_path'])
        if target.exists():
            with target.open('rb') as handle:
                if not handle.read(90).startswith(POINTER):
                    raise ValueError('Refusing to replace different existing asset: ' + entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': entry['path'], 'status': 'restored_verified'})
    print(json.dumps(records, indent=2))
    return int(any(row['status'] == 'missing_payload' for row in records))

if __name__ == '__main__':
    raise SystemExit(main())
