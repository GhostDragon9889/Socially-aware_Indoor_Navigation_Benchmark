"""Record the actual child exit; a child-written passed flag is insufficient."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
output = root / "output/terminal_probe_runs"
output.mkdir(parents=True, exist_ok=True)
name, *command = sys.argv[1:]
assert Path(name).name == name and command
receipt = output / (name + '_execution.json')
assert not receipt.exists(), receipt
sources = {}
for folder in ('src', 'scripts', 'tests', 'examples'):
    for path in (root / folder).rglob('*.py'):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
started = time.monotonic()
record = dict(command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256=sources, actual_returncode=None)
receipt.write_text(json.dumps(record, indent=2) + '\n')
with (output / (name + '.log')).open('w') as stream:
    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
changed = [name for name, sha in sources.items() if not (root / name).exists() or hashlib.sha256((root / name).read_bytes()).hexdigest() != sha]
record.update(actual_returncode=result.returncode, elapsed_s=time.monotonic()-started, source_changes_during_execution=changed)
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: record[key] for key in ('command', 'actual_returncode', 'elapsed_s')}), flush=True)
sys.exit(0 if result.returncode == 0 else 1)
