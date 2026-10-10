"""Real V11/closed-preflight and actual4 metadata guards; no future15 captures."""
import ast
from copy import deepcopy
import sys

import pytest

import audit_state_prior_native as reader
import native_evidence as evidence


@pytest.fixture
def registered4():
    return evidence.read(evidence.HERE / "native4_registration.json")


@pytest.fixture
def actual4():
    return evidence.read(evidence.run_path(4) / "result.json")


def test_real_v11_absolute_manifest_relative_archive():
    proof = evidence.source_binding()
    assert proof["source_count"] == 183
    assert proof["all_archive_and_current_selected_bytes_exact"] is True
    assert proof["complete_runtime_import_ledger"] is False


@pytest.mark.parametrize("actors", [4, 15])
def test_frozen_original_math_and_only_declared_adapter(actors):
    _, proof = reader.readers(actors)
    assert proof["base_reverse_restored_full_AST_equals_original"]
    assert proof["v2_reverse_restored_full_AST_equals_original"]
    assert proof["COM_force_fall_route_math_unchanged"]
    assert len(proof["origin_shape_constant_changes"]) == (4 if actors == 15 else 0)


def test_archive_index_missing_rejected():
    with pytest.raises(ValueError, match="Exactly one"):
        reader.source_index_adapter(ast.parse("members = {}"))


def test_origin_gate_missing_rejected():
    with pytest.raises(ValueError, match="Exactly four"):
        reader.origin_shape_adapter(ast.parse("pass"))


def test_real_prepared15_stays_pending_without_terminal_read():
    proof = evidence.read(evidence.HERE / "native15_registration.json")
    assert proof["native_execution_audited"] is False
    assert proof["actual_quality"] is None
    assert proof["expected_counts"]["feet_observations"] == 1440000
    assert proof["expected_counts"]["peer_denominator_unique_pair_controls"] == 630000


def test_actual4_trained_prior_metadata(registered4, actual4):
    evidence.validate_result_metadata(actual4, registered4)


@pytest.mark.parametrize("mutation", ["false_flag", "int_flag", "false_public_D", "false_public_RMS", "missing_extension", "wrong_SHA", "wrong_counts", "wrong_actor_checkpoint", "wrong_control_count"])
def test_independently_tampered_actual4_metadata_rejected(registered4, actual4, mutation):
    changed = deepcopy(actual4)
    actor = changed["evaluation_actor"]
    initialization = actor["full_training_provenance"]["initialization"]
    if mutation == "false_flag":
        actor["config"]["warm_start_action_discriminator_from_state_prior"] = False
    elif mutation == "int_flag":
        actor["config"]["warm_start_action_discriminator_from_state_prior"] = 1
    elif mutation == "false_public_D":
        initialization["published_state_only_discriminator_transferred"] = False
    elif mutation == "false_public_RMS":
        initialization["published_amp_history_moments_transferred"] = False
    elif mutation == "missing_extension":
        del initialization["state_prior_action_extension"]
    elif mutation == "wrong_SHA":
        initialization["sha256"] = "0" * 64
    elif mutation == "wrong_counts":
        initialization["normalizer_counts_before_cap"]["amp_normalizer"] = 200000.
    elif mutation == "wrong_actor_checkpoint":
        actor["checkpoint_sha256"] = "0" * 64
    elif mutation == "wrong_control_count":
        changed["actual_control_steps"] = 5999
    with pytest.raises(ValueError):
        evidence.validate_result_metadata(changed, registered4)


def test_nonzero_falls_not_an_integrity_filter(registered4, actual4):
    changed = deepcopy(actual4)
    changed["fall_resets"] = 7
    changed["time_limit_resets"] = 4
    evidence.validate_result_metadata(changed, registered4)
    assert registered4["falls_zero_required_for_integrity_admission"] is False


@pytest.mark.parametrize("field", ["feet_observations", "live_floor_rays"])
def test_registration_cannot_rewrite_actual_observation_denominator(registered4, field):
    changed = deepcopy(registered4)
    changed["expected_counts"][field] += 1
    with pytest.raises(ValueError, match="canonical complete native denominators"):
        evidence.admit_closed_native(4, changed)


def test_registration_and_actor_provenance_cannot_be_jointly_rewritten(registered4, actual4, monkeypatch):
    changed = deepcopy(registered4)
    changed["evidence"]["expected_full_training_provenance"]["initialization"]["published_state_only_discriminator_transferred"] = False
    changed_result = deepcopy(actual4)
    changed_result["evaluation_actor"]["full_training_provenance"] = deepcopy(changed["evidence"]["expected_full_training_provenance"])
    original_read = evidence.read

    def observed_read(path):
        if evidence.Path(path) == evidence.run_path(4) / "result.json":
            return changed_result
        return original_read(path)

    monkeypatch.setattr(evidence, "read", observed_read)
    with pytest.raises(ValueError, match="actual terminal/readiness/preflight"):
        evidence.admit_closed_native(4, changed)


def test_running15_fails_before_result_or_array_read(monkeypatch):
    registered = evidence.read(evidence.HERE / "native15_registration.json")
    original_read = evidence.read
    visited = []

    def observed_read(path):
        path = evidence.Path(path)
        visited.append(str(path))
        if path.name == "native15_v2_actual_launch.json":
            return {"status": "running"}
        assert not (path.parent == evidence.run_path(15) and path.name in {"result.json", "physical_states.npz", "ground_support.npz"})
        return original_read(path)

    monkeypatch.setattr(evidence, "read", observed_read)
    with pytest.raises(ValueError, match="closed native launch"):
        evidence.admit_closed_native(15, registered)
    assert not any(path.endswith("physical_states.npz") for path in visited)


def test_v2_closed15_without_observed_child_exit_rejected(monkeypatch):
    registered = evidence.read(evidence.HERE / "native15_registration.json")
    launch = evidence.read(evidence.RUNS / "native15_v2_actual_preflight.json")
    launch.update(status="closed", exit_code=0, prepare_only=False, child_exit_observed=False)
    launch["command"] = launch["command"][:-1]
    original_read = evidence.read

    def observed_read(path):
        path = evidence.Path(path)
        if path.name == "native15_v2_actual_launch.json":
            return launch
        assert not (path.parent == evidence.run_path(15) and path.name == "result.json")
        return original_read(path)

    monkeypatch.setattr(evidence, "read", observed_read)
    with pytest.raises(ValueError, match="child exit must be observed"):
        evidence.admit_closed_native(15, registered)


def test_no_model_or_SDK_imports():
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
