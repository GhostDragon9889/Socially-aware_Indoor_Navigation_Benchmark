"""CPU counterexamples for runtime-epoch admission and 15-person adaptation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import native_metric_sources as sources
import run_native_CPU_metrics as runner
import compare_native_metrics as comparison


def raw(actors=4, count=183):
    return {"schema": "public_state_prior.native200s_independent_CPU_audit.v1",
            "status": "raw_sensor_source_contracts_passed_bounded_zero_fall_observed",
            "native_200s_artifacts_independently_audited": True,
            "counts": {"controls": 6000, "actors": actors, "duration_s": 200.0, "shared_physics_substeps": 24000},
            "source_epoch": {"source_count": count, "all_archived_source_bytes_verified": True},
            "checks": ["actual source and complete physical channels"]}


@pytest.mark.parametrize("actors", [4, 15])
def test_complete_native_audit_does_not_filter_stability_failures(actors):
    value = raw(actors)
    value["status"] = "raw_sensor_source_contracts_passed_physical_stability_failed"
    value["counts"]["falls"] = 3
    sources.validate_raw_contract(value, actors, 183, value["schema"])


@pytest.mark.parametrize("mutation", ["fixture_schema", "old168", "old174", "partial", "unknown_status", "not_actual", "archive_not_verified"])
def test_new_native_admission_rejects_fixture_old_or_incomplete_proof(mutation):
    value = raw()
    if mutation == "fixture_schema":
        value["schema"] = "public_state_prior.actual_navigation_ground_audit.v1"
    elif mutation in {"old168", "old174"}:
        value["source_epoch"]["source_count"] = int(mutation[3:])
    elif mutation == "partial":
        value["counts"]["controls"] = 5999
    elif mutation == "unknown_status":
        value["status"] = "running"
    elif mutation == "not_actual":
        value["native_200s_artifacts_independently_audited"] = False
    else:
        value["source_epoch"]["all_archived_source_bytes_verified"] = False
    with pytest.raises(ValueError):
        sources.validate_raw_contract(value, 4, 183, "public_state_prior.native200s_independent_CPU_audit.v1")


@pytest.mark.parametrize("actors,count", [(4, 168), (15, 174)])
def test_historical_native_epochs_are_independent_of_new183(actors, count):
    value = raw(actors, count)
    value["schema"] = f"native_Warehouse{actors}_200s_independent_CPU_audit.v1"
    sources.validate_raw_contract(value, actors, count, value["schema"])
    with pytest.raises(ValueError):
        sources.validate_raw_contract(value, actors, 183, value["schema"])


def test_generic_complete15_actor_metadata_accepts15_rejects4_or_partial():
    class ShapeOnly:
        shape = (6000, 15, 24, 3)
    arrays = {"body_pos": ShapeOnly()}
    result = {"status": "passed", "error": None, "actual_control_steps": 6000, "arguments": {"num_envs": 15}}
    runner.validate_complete_arrays(arrays, result, 15)
    with pytest.raises(ValueError):
        runner.validate_complete_arrays(arrays, result, 4)
    result["actual_control_steps"] = 5999
    with pytest.raises(ValueError):
        runner.validate_complete_arrays(arrays, result, 15)


def test_original_sequence_helper_keeps15_actors_and_excludes_only_reset_boundary():
    from CrowdSim.humanoid.style_evaluation import BODY_NAMES, sequence_style_samples
    frames, actors = 6, 15
    body = np.zeros((frames, actors, 24, 3), dtype=np.float64)
    body[..., 0] = np.arange(24)[None, None, :] * .05
    body[..., 2] = np.linspace(.1, 1.6, 24)[None, None, :]
    body[..., 0] += np.arange(frames)[:, None, None] / 30
    rot = np.zeros((frames, actors, 4), dtype=np.float64)
    rot[..., 3] = 1
    episodes = np.zeros((frames, actors), dtype=np.int64)
    episodes[3:, 0] = 1
    resets = np.zeros((frames, actors), dtype=bool)
    resets[0] = True
    resets[3, 0] = True
    result = sequence_style_samples(body, rot, body_names=BODY_NAMES, episode_ids=episodes, reset_mask=resets)
    assert len(result["local_pose"]) == (frames - 1) * actors - 1
    assert result["summary"]["same_episode_transition_count"] == 74
    assert result["summary"]["excluded_reset_transitions"] == 1
    assert len(np.unique(result["actor_indices"])) == actors


def test_owned_import_guard_rejects_unregistered_checkout_or_old_bundle(tmp_path):
    registered = tmp_path / "registered.py"
    registered.write_text("answer = 42\n")
    ledger = sources.CPULedger(tmp_path / "ledger.json", {str(registered): sources.record(registered)}, {})
    ledger.verify_path(registered)
    outside = tmp_path / "old_bundle.py"
    outside.write_text("answer = 42\n")
    with pytest.raises(ValueError):
        ledger.verify_path(outside)
    registered.write_text("answer = 43\n")
    with pytest.raises(ValueError):
        ledger.verify_path(registered)


def test_catalog_rejects_fixture_output_and_native_epoch_swap():
    path = Path(runner.CATALOG)
    if not path.exists():
        pytest.skip("Catalog binds only after independent reader source freeze")
    original = json.loads(path.read_text())
    runner.validate_catalog(original)
    for change in ("input", "output", "epoch", "raw_alias", "parameter"):
        value = deepcopy(original)
        row = value["metrics"]["new15_90"]
        if change == "input":
            row["input"] = str(sources.ROOT / "output/diffusion_amp/state_prior_nav2000_v1")
        elif change == "output":
            row["output"] = value["metrics"]["new4_90"]["output"]
        elif change == "epoch":
            row["source_count"] = 174
        elif change == "raw_alias":
            row["raw_audit"] = value["metrics"]["old15_90"]["raw_audit"]
        else:
            row["metric_argv"][row["metric_argv"].index("30")] = "1"
        with pytest.raises(ValueError):
            runner.validate_catalog(value)


def test_saved_original90_schema_preserves_unknown_bins_and_primary():
    p = sources.ROOT / "docs/style_improvement_20261003/state_prior_metrics_v1/navigation90/evaluation.json"
    value = json.loads(p.read_text())["runs"]["state_prior_navigation2000"]["inclusive_speed_conditioned"]
    before = deepcopy(value)
    old_count, new_count, common = comparison.common_supported_bins(value, value)
    assert old_count == new_count == 3
    assert common["new_equal_bin_mean_MMD2"] == 0.06811695149147463
    assert value == before
    assert sum(b["pose_velocity_gaussian_mmd2"] is None for b in value["speed_bins"]) == 4
