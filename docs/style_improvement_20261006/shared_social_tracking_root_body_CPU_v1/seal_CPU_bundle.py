"""Seal the actual CPU observation and exact source snapshots; no SDK import."""
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
    assert not index_path.exists(), 'Preserve an existing evidence seal'
    report = json.loads((BASE / 'actual/result.json').read_text())
    receipt = json.loads((BASE / 'CPU_tools_execution.json').read_text())
    assert receipt['actual_diagnostic']['terminal_response']['exit_code'] == 0
    assert receipt['actual_diagnostic']['observed_session_id'] == 36953
    assert receipt['final_pytest']['exit_code'] == receipt['final_ruff']['exit_code'] == 0
    assert receipt['mutation_test']['exit_code'] == 1
    assert '22 passed' in (BASE / 'final_pytest.log').read_text()
    assert '6 failed, 16 passed' in (BASE / 'mutation_pytest.log').read_text()
    assert report['status'] == 'completed_CPU_observation'
    assert report['Torch_imported'] is False and report['SDK_started'] is False
    assert not report['paper_effect_achieved'] and not report['default_model_promoted']
    archived = []
    for source in report['sources']:
        assert identity(source['path']) == source
        target = BASE / 'source_snapshots' / Path(source['path']).name
        if not target.exists():
            shutil.copyfile(source['path'], target)
        saved = identity(target)
        assert (saved['bytes'], saved['sha256']) == (source['bytes'], source['sha256'])
        archived.append({'original': source, 'archive': saved})
    for row in report['inputs']:
        assert identity(row['path']) == row
    for run in report['runs'].values():
        assert run['all_metric_frames'] == 11982 and len(run['sequences']) == 22
        assert identity(run['all_frame_archive']['path']) == run['all_frame_archive']
    files = sorted(path for path in BASE.rglob('*') if path.is_file())
    records = [identity(path) for path in files]
    for path in files:
        path.chmod(0o444)
    index = {'schema': 'shared_social.tracking_root_body_CPU_evidence_seal.v1',
             'files': records, 'archived_sources': archived,
             'actual_result': identity(BASE / 'actual/result.json'),
             'actual_tools': identity(BASE / 'CPU_tools_execution.json'),
             'SDK_started': False, 'quality_improvement_proven': False}
    with index_path.open('x') as stream:
        stream.write(json.dumps(index, indent=2, allow_nan=False) + '\n')
    index_path.chmod(0o444)
    print(json.dumps({'files': len(records), 'seal': identity(index_path), 'SDK_started': False}))


if __name__ == '__main__':
    main()
