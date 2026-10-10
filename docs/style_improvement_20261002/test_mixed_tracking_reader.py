"""CPU admission edge cases; mock seals are never claimed as trained models."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS))
spec = importlib.util.spec_from_file_location('mixed_tracking_reader_contract', DOCS / 'audit_mixed_curriculum_tracking.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.mark.parametrize('step,total,group,equal', [(500, 4096000, 2048000, False),
    (1000, 8192000, 4096000, True), (2000, 16384000, 8192000, False)])
def test_actual_group_exposure_is_separate_from_total_budget(step, total, group, equal):
    budget = reader.admitted_budget(step)
    assert budget['total_transitions'] == total
    assert budget['amass_transitions'] == budget['procedural_transitions'] == group
    assert budget['equal_total_budget_to_baseline'] is equal
    assert budget['each_group_has_baseline_total_exposure'] is (step == 2000)


@pytest.mark.parametrize('step', [True, 4, 8, 750, 500.0])
def test_pilot_continuation_or_undocumented_budget_cannot_enter_fresh_validation_admission(step):
    with pytest.raises(ValueError):
        reader.admitted_budget(step)


def seal(tmp_path):
    path = tmp_path / 'step_500.pt'
    path.write_bytes(b'synthetic seal contract only, not weights')
    checkpoint = reader.paired.record(path)
    row = {**checkpoint, 'update_count': 500, 'physical_transition_count': 4096000,
           'identical_source_bytes_verified': True}
    document = {'snapshots': [row], 'watcher_sha256': 'synthetic-watcher'}
    launch = {'source_manifest_sha256': reader.SOURCE_MANIFEST_SHA, 'targets': [500, 1000, 2000],
              'snapshot_dir': str(tmp_path), 'watcher_sha256': 'synthetic-watcher',
              'full_weight_semantics_checked_by_watch': False}
    return checkpoint, document, launch


def test_unique_snapshot_bound_to_actual_bytes_and_source(tmp_path):
    checkpoint, document, launch = seal(tmp_path)
    admitted = reader.admit_seal(document, checkpoint, 500, launch)
    assert admitted['sha256'] == checkpoint['sha256']
    assert 'control_step' not in admitted  # Real transport rows do not attest a runner clock/source SHA.
    changed = deepcopy(document)
    changed['snapshots'].append(changed['snapshots'][0])
    with pytest.raises(ValueError, match='duplicated'):
        reader.admit_seal(changed, checkpoint, 500, launch)


@pytest.mark.parametrize('field,value', [('sha256', '0'*64), ('bytes', 1),
    ('physical_transition_count', 8192000), ('identical_source_bytes_verified', False)])
def test_wrong_generation_or_counter_rejected_before_deserialization(tmp_path, field, value):
    checkpoint, document, launch = seal(tmp_path)
    document['snapshots'][0][field] = value
    with pytest.raises(ValueError):
        reader.admit_seal(document, checkpoint, 500, launch)


def test_mutable_last_and_unbound_archive_rejected(tmp_path):
    checkpoint, document, launch = seal(tmp_path)
    launch['source_manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='frozen135'):
        reader.admit_seal(document, checkpoint, 500, launch)
    checkpoint, document, launch = seal(tmp_path)
    renamed = Path(checkpoint['path']).with_name('last.pt')
    Path(checkpoint['path']).rename(renamed)
    checkpoint['path'] = str(renamed)
    document['snapshots'][0]['path'] = str(renamed)
    with pytest.raises(ValueError, match='mutable last'):
        reader.admit_seal(document, checkpoint, 500, launch)


def independent_document(checkpoint):
    return {'schema': 'mixed_sealed_milestone_payload_admission.v1', 'status': 'passed',
            'CPU_only': True, 'GPU_used': False, 'SDK_started': False, 'checkpoint': checkpoint,
            'source_manifest': {'sha256': reader.SOURCE_MANIFEST_SHA},
            'source_archive': {'sha256': reader.SOURCE_ARCHIVE_SHA},
            'checks': [{'passed': True}], 'check_count': 1}


def test_transport_seal_does_not_replace_independent_full_payload_admission(tmp_path):
    checkpoint, _, _ = seal(tmp_path)
    document = independent_document(checkpoint)
    assert reader.independent_checkpoint_admission(document, checkpoint)['independent_CPU_check_count'] == 1
    document['checks'][0]['passed'] = False
    with pytest.raises(ValueError, match='failed checks'):
        reader.independent_checkpoint_admission(document, checkpoint)


@pytest.mark.parametrize('change', ['other_checkpoint', 'source', 'GPU_used', 'duplicate'])
def test_independent_admission_is_bound_to_exact_bytes_source_and_cpu(tmp_path, change):
    checkpoint, _, _ = seal(tmp_path)
    document = independent_document(deepcopy(checkpoint))
    if change == 'other_checkpoint':
        document['checkpoint']['sha256'] = '0'*64
    elif change == 'source':
        document['source_archive']['sha256'] = '0'*64
    elif change == 'GPU_used':
        document['GPU_used'] = True
    else:
        row = document.pop('checkpoint')
        document['schema'] = 'navisaaclab.mixed_actual_checkpoint_audit.v1'
        document['checkpoints'] = [{'checkpoint': row}, {'checkpoint': row}]
    with pytest.raises(ValueError):
        reader.independent_checkpoint_admission(document, checkpoint)


def registered_document():
    document = reader.paired.read(DOCS / 'mixed_curriculum_validation_plan_v1.json')
    result = {'arguments': {'split': 'validation', 'max_steps': None, 'motion_ids': None, 'max_clips': None,
              'seed': 7, 'cpu_threads': 4, 'deterministic_evaluation': True, 'failure_threshold_m': .5},
              'selected_source_sha256': deepcopy(document['actual_runtime_source_sha256'])}
    return document, result


def test_registered_protocol_is_complete_validation_and_opened_test_history_explicit():
    document, result = registered_document()
    admitted = reader.registered_validation_admission(document, result, 1000)
    assert admitted['new_blind_test_claim'] is False
    assert len(admitted['registered_actual_runtime_sources']) == 6


@pytest.mark.parametrize('change', ['frame_cap', 'runtime_source', 'blind_test_claim', 'group_budget'])
def test_changed_registered_protocol_or_source_is_rejected(change):
    document, result = registered_document()
    if change == 'frame_cap':
        result['arguments']['max_steps'] = 30
    elif change == 'runtime_source':
        result['selected_source_sha256']['CrowdSim/humanoid/amp_world.py'] = '0'*64
    elif change == 'blind_test_claim':
        document['test_history']['not_a_new_blind_test_fold'] = False
    else:
        document['budgets']['1000']['each_branch_physical_transitions'] = 8192000
    with pytest.raises(ValueError):
        reader.registered_validation_admission(document, result, 1000)


def test_legacy_field_and_tuple_normalization_preserves_values():
    old = {'action_mask': (1., 0., 1.)}
    saved = {'action_mask': [1., 0., 1.], 'expert_sampling': None}
    assert reader.normalized_config(old) == saved
    altered = {'action_mask': [1., 1., 1.], 'expert_sampling': None}
    assert reader.normalized_config(old) != altered
    assert old == {'action_mask': (1., 0., 1.)}  # Caller payload remains intact.


def test_test_split_never_enters_repeated_validation_reader():
    with pytest.raises(ValueError, match='no test split'):
        reader.paired.require_full_validation({'status': 'passed', 'arguments': {'split': 'test'}})


def test_threshold_is_any_complete_step_and_failed_frames_are_not_removed():
    capture = {'metric_frame': np.array([False, True, True]),
               'rigid_body_pos': np.zeros((3, 1, 24, 3)),
               'reference_rigid_body_pos': np.zeros((3, 1, 24, 3)),
               'reference_frame': np.array([9, 10, 11])}
    capture['rigid_body_pos'][2, :, :, 0] = .50001
    row = {'tracking_window': {'initial_frame': 9, 'last_frame': 11}, 'complete_clip': True,
           'tracking_failed': True, 'first_tracking_failure_capture_frame': 2, 'tracking_success': False}
    check = reader.raw_tracking_rule(capture, row, .5)
    assert check['failure_capture_frames'] == [2]
    row['tracking_success'] = True
    with pytest.raises(ValueError, match='success rule'):
        reader.raw_tracking_rule(capture, row, .5)
