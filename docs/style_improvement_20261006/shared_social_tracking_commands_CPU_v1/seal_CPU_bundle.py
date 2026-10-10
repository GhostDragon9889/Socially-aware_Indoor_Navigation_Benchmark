"""Seal an actual CPU command observation, retaining exact source snapshots."""
from pathlib import Path
import json
import shutil
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_shared_social_training import identity  # noqa: E402


def main():
    index_path = BASE / 'evidence_seal.json'
    assert not index_path.exists()
    report = json.loads((BASE / 'actual_result.json').read_text())
    execution = json.loads((BASE / 'CPU_tools_execution.json').read_text())
    assert execution['actual_audit']['observed_session_id'] == 26806
    assert execution['actual_audit']['terminal_response']['exit_code'] == 0
    assert execution['final_pytest']['exit_code'] == execution['final_ruff']['exit_code'] == 0
    assert execution['mutation_pytest']['exit_code'] == 1
    assert '38 passed' in (BASE / 'final_pytest.log').read_text()
    assert '3 failed, 35 passed' in (BASE / 'mutation_pytest.log').read_text()
    assert report['status'] == 'passed_CPU_command_consistency'
    assert not report['Torch_imported'] and not report['SDK_started'] and not report['failed_frames_filtered']
    snapshots = []
    for source in report['sources']:
        assert identity(source['path']) == source
        path = BASE / 'source_snapshots' / Path(source['path']).name
        if not path.exists():
            shutil.copyfile(source['path'], path)
        saved = identity(path)
        assert (source['bytes'], source['sha256']) == (saved['bytes'], saved['sha256'])
        snapshots.append({'original': source, 'archive': saved})
    old_test = ROOT / 'tests/test_shared_social_tracking_decomposition.py'
    prior_test = BASE.parent / 'shared_social_tracking_root_body_CPU_v1/source_snapshots' / old_test.name
    assert old_test.read_bytes() == prior_test.read_bytes()
    shutil.copyfile(old_test, BASE / 'source_snapshots' / old_test.name)
    for row in report['inputs']:
        assert identity(row['path']) == row
    assert all(len(row['sequences']) == 22 and row['aggregate']['metric_frames'] == 11982 for row in report['runs'].values())
    files = sorted(path for path in BASE.rglob('*') if path.is_file())
    records = [identity(path) for path in files]
    for path in files:
        path.chmod(0o444)
    index = {'schema': 'shared_social.tracking_PRE_commands_CPU_evidence_seal.v1',
             'files': records, 'source_snapshots': snapshots,
             'actual_result': identity(BASE / 'actual_result.json'),
             'actual_tools': identity(BASE / 'CPU_tools_execution.json'),
             'SDK_started': False, 'quality_improvement_proven': False}
    with index_path.open('x') as stream:
        stream.write(json.dumps(index, indent=2, allow_nan=False) + '\n')
    index_path.chmod(0o444)
    print(json.dumps({'files': len(records), 'seal': identity(index_path), 'SDK_started': False}))


if __name__ == '__main__':
    main()
