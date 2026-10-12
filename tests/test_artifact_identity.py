from copy import deepcopy
from pathlib import Path

import pytest

from social_navigation.artifact_identity import LEGACY_PROJECT, artifact_path, same_artifact, same_recorded_value


def test_registered_rename_changes_only_path_without_touching_inputs(tmp_path):
    old = {'path': str(LEGACY_PROJECT / 'output/a.pt'), 'bytes': 12, 'sha256': 'a' * 64}
    new = {**old, 'path': str(tmp_path / 'output/a.pt')}
    original = deepcopy((old, new))
    assert same_artifact(old, new, root=tmp_path)
    assert same_artifact(new, old, root=tmp_path)
    assert (old, new) == original
    assert artifact_path(LEGACY_PROJECT, root=tmp_path) == tmp_path


@pytest.mark.parametrize('field,value', [
    ('path', '/foreign/ProtoMotions/output/a.pt'),
    ('path', '/project/EXP/SangforAndSaut/ProtoMotions-other/output/a.pt'),
    ('path', str(LEGACY_PROJECT / 'output/../output/a.pt')),
    ('path', 'output/a.pt'), ('path', None),
    ('bytes', 13), ('bytes', 12.0), ('bytes', True), ('bytes', -1),
    ('sha256', 'b' * 64), ('sha256', 'g' * 64), ('sha256', 'a' * 63),
    ('sha256', None), ('sha256', 'A' * 64),
])
def test_rename_cannot_hide_changed_or_malformed_identity(tmp_path, field, value):
    old = {'path': str(LEGACY_PROJECT / 'output/a.pt'), 'bytes': 12, 'sha256': 'a' * 64}
    new = {**old, 'path': str(tmp_path / 'output/a.pt'), field: value}
    assert not same_artifact(old, new, root=tmp_path)


@pytest.mark.parametrize('value', [None, [], {}, {'path': '/a'}])
def test_incomplete_identity_is_not_admitted(value):
    assert not same_artifact(value, value)


def test_additional_provenance_is_compared_and_sdk_paths_are_not_relocated(tmp_path):
    record = {'path': str(tmp_path.parent / 'external_sdk/a.py'),
              'bytes': 1, 'sha256': 'a' * 64, 'source_version': 'v1'}
    assert artifact_path(record['path'], root=tmp_path) == Path(record['path'])
    assert same_artifact(record, dict(record))
    assert not same_artifact(record, {**record, 'source_version': 'v2'})


def test_source_layout_relocation_and_nested_binding_preserve_unrelated_strings(tmp_path):
    old = {'path': str(LEGACY_PROJECT / 'CrowdSim/a.py'), 'bytes': 12, 'sha256': 'a' * 64}
    new = {**old, 'path': str(tmp_path / 'src/CrowdSim/a.py')}
    assert same_recorded_value({'sources': [old]}, {'sources': [new]}, root=tmp_path)
    assert artifact_path(tmp_path / 'CrowdSim/a.py', root=tmp_path) == Path(new['path'])
    assert not same_recorded_value({'command': old['path']}, {'command': new['path']}, root=tmp_path)
    assert not same_recorded_value([1], [True], root=tmp_path)
    assert not same_recorded_value([1], [1, 2], root=tmp_path)
    assert not same_recorded_value({'a': 1}, {'b': 1}, root=tmp_path)
    assert not same_recorded_value(old, {}, root=tmp_path)
    assert same_recorded_value((1, 'same'), (1, 'same'), root=tmp_path)


@pytest.mark.parametrize('expected,actual', [(True, 1), (0, 0.0), ({'ok': True}, {'ok': 1})])
def test_additional_provenance_retains_exact_value_types(expected, actual):
    record = {'path': '/absolute/a', 'bytes': 1, 'sha256': 'a' * 64, 'metadata': expected}
    assert not same_artifact(record, {**record, 'metadata': actual})


def test_bound_evidence_reads_relocated_bytes_and_rejects_tampering(tmp_path, monkeypatch):
    from scripts import compare_shared_social_evidence as evidence
    from social_navigation import artifact_identity
    monkeypatch.setattr(artifact_identity, 'repository_root', lambda: tmp_path)
    current = tmp_path / 'output/result.json'
    current.parent.mkdir()
    current.write_text('{"actual_exit_code": 0}')
    identity = {**evidence.record(current), 'path': str(LEGACY_PROJECT / 'output/result.json')}
    assert evidence.bound(identity) == {'actual_exit_code': 0}
    current.write_text('{"actual_exit_code": 1}')
    with pytest.raises(ValueError, match='Recorded artifact changed'):
        evidence.bound(identity)
