"""Current reader provenance must retain measurement code, not only new hashes."""
import hashlib
from pathlib import Path

import pytest

from CrowdSim.humanoid.analysis import cpu_dependency_migration as migration


@pytest.mark.parametrize('name', migration.READER_VERSIONS)
def test_current_readers_retain_historical_measurement_and_admission_AST(name):
    historical, _ = migration.READER_VERSIONS[name]
    path = migration.ROOT / 'src/CrowdSim/humanoid/analysis' / (name + '.py')
    proof = migration.verify_reader(path, historical)
    assert proof['measurement_AST_unchanged']
    assert proof['unchanged_functions']
    assert proof['SDK_execution_verified'] is False


def test_even_a_registered_new_hash_cannot_approve_changed_metric_math(monkeypatch):
    name = 'native_navigation_audit'
    path = migration.ROOT / 'src/CrowdSim/humanoid/analysis' / (name + '.py')
    historical, _ = migration.READER_VERSIONS[name]
    original = Path.read_text
    changed = original(path).replace('3 <= actors <= 15', '3 <= actors <= 16')
    assert changed != original(path)
    monkeypatch.setitem(migration.READER_VERSIONS, name,
                        (historical, hashlib.sha256(changed.encode()).hexdigest()))
    monkeypatch.setattr(Path, 'read_text',
                        lambda p, *a, **kw: changed if p == path else original(p, *a, **kw))
    with pytest.raises(ValueError, match='measurement/admission'):
        migration.verify_reader(path, historical)


def test_foreign_path_or_unregistered_parent_cannot_claim_reader_equivalence(tmp_path):
    historical, _ = migration.READER_VERSIONS['native_navigation_audit']
    with pytest.raises(ValueError, match='Unregistered'):
        migration.verify_reader(tmp_path / 'native_navigation_audit.py', historical)
    path = migration.ROOT / 'src/CrowdSim/humanoid/analysis/native_navigation_audit.py'
    with pytest.raises(ValueError, match='Unregistered'):
        migration.verify_reader(path, '0' * 64)


def test_current_lifecycle_record_bytes_cannot_change_even_with_new_file_hash(monkeypatch):
    path = migration.ROOT / 'src/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py'
    original = Path.read_text
    changed = original(path).replace('info.st_nlink != 1', 'info.st_nlink != 2')
    assert changed != original(path)
    monkeypatch.setattr(migration, 'LIFECYCLE_CURRENT_SHA', hashlib.sha256(changed.encode()).hexdigest())
    monkeypatch.setattr(Path, 'read_text',
                        lambda p, *a, **kw: changed if p == path else original(p, *a, **kw))
    with pytest.raises(ValueError, match='dependency changed'):
        migration.verify_dependency(path, migration.LIFECYCLE_HISTORICAL_SHA)


@pytest.mark.parametrize('relative,sha', [
    ('scripts/compare_shared_social_evidence.py', '76e589b05101480ac21ae229e7c8ecdefde049073670deaddb88a4306ed77a35'),
    ('scripts/verify_shared_social_evidence_admission.py', 'f2cf03483f9bab988d9fb293093c644ef97b7cfeda32458e9ba0b2a93c40600a'),
])
def test_artifact_adapters_restore_the_entire_metric_source(relative, sha):
    proof = migration.verify_dependency(migration.ROOT / relative, sha)
    assert proof['artifact_location_adapter_exact_inverse']
    assert proof['all_metric_function_bytes_unchanged']
