"""Synthetic CPU reader contracts; no captured rollout or SDK is fabricated."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("audit_amass_root_tracking.py")
spec = importlib.util.spec_from_file_location("amass_root_tracking_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def arrays():
    data = {key: np.zeros((6, 1, 3), dtype=np.float32) for key in audit.INITIAL_FIELDS}
    for key in audit.CLOCK_FIELDS:
        data[key] = np.arange(6, dtype=np.int64)
    data["reference_root_pos"] = np.zeros((6, 1, 3), dtype=np.float32)
    return data


def test_pair_requires_exact_reference_but_allows_learned_poststep_motion():
    a, b = arrays(), arrays()
    b["root_pos"][1:] += 5
    checks = audit.exact_capture_pair(a, b)
    assert "reference_root_pos" in checks["reference_and_clock_fields_bitwise_equal"]
    assert "amp_history" in checks["initial_physical_state_and_history_bitwise_equal"]


@pytest.mark.parametrize("key,change", [
    ("trajectory_xy", "value"), ("reference_root_pos", "dtype"),
    ("amp_history", "initial"), ("episode_id", "value"),
])
def test_tiny_reference_change_dtype_initial_history_and_reset_clock_fail(key, change):
    a, b = arrays(), arrays()
    if change == "dtype":
        b[key] = b[key].astype(np.float64)
    elif change == "initial":
        b[key][0] += 1e-6
    else:
        b[key][1] += 1
    with pytest.raises(ValueError, match="differs"):
        audit.exact_capture_pair(a, b)


def test_fall_intervals_keep_multiple_onsets_and_report_elapsed_not_absolute_phase():
    data = {"fall_mask": np.array([False, False, True, True, False, True]),
            "reference_frame": np.arange(9, 15), "reference_time_s": np.arange(9, 15) / 30,
            "root_pos": np.zeros((6, 1, 3))}
    row = {"fall_capture_frames": [2, 3, 5], "fall_onset_events": 2, "sequence_had_fall": True}
    out = audit.fall_intervals(data, row)
    assert [r["capture_frames_inclusive"] for r in out] == [[2, 3], [5, 5]]
    assert [r["source_frames_inclusive"] for r in out] == [[11, 12], [14, 14]]
    assert out[0]["elapsed_seconds_from_reset_inclusive"][0] == pytest.approx(2 / 30)
    bad = dict(row, fall_onset_events=1)
    with pytest.raises(ValueError, match="onset"):
        audit.fall_intervals(data, bad)


def full_result():
    return {"status": "passed", "arguments": {"split": "validation", "motion_ids": None, "max_clips": None, "max_steps": None},
            "sequences": [{"motion_id": i, "complete_clip": True, "in_full_sequence_success_denominator": True,
                           "observed_metric_steps": 100, "expected_control_steps": 100} for i in range(22)]}


def test_validation_only_full22_admission_rejects_test_subset_or_partial_clip():
    result = full_result()
    assert len(audit.require_full_validation(result)) == 22
    result["arguments"]["split"] = "test"
    with pytest.raises(ValueError, match="no test split"):
        audit.require_full_validation(result)
    result = full_result()
    result["arguments"]["max_steps"] = 30
    with pytest.raises(ValueError, match="caps"):
        audit.require_full_validation(result)
    result = full_result()
    result["sequences"][0]["complete_clip"] = False
    with pytest.raises(ValueError, match="denominator"):
        audit.require_full_validation(result)


def test_success_subset_denominators_do_not_hide_length_weighting_or_empty_set():
    rows = {1: {"observed_metric_steps": 1, "mean_body_position_error_m": .1,
                "mean_body_rotation_error_rad": .2, "sequence_had_fall": False},
            2: {"observed_metric_steps": 3, "mean_body_position_error_m": .3,
                "mean_body_rotation_error_rad": .4, "sequence_had_fall": True}}
    out = audit.subset_primary(rows, {1, 2})
    assert out["clips"] == 2 and out["metric_frames"] == 4 and out["sequences_with_fall"] == 1
    assert out["per_clip_mean"]["mean_body_position_error_m"] == pytest.approx(.2)
    assert out["frame_weighted_mean"]["mean_body_position_error_m"] == pytest.approx(.25)
    assert audit.subset_primary(rows, set())["frame_weighted_mean"] is None


def test_snapshot_copy_requires_identical_original_training_artifact(tmp_path):
    train = tmp_path / "train"
    snapshots = tmp_path / "snapshots"
    train.mkdir()
    snapshots.mkdir()
    (train / "step_1000.pt").write_bytes(b"synthetic_checkpoint_contract_bytes")
    snapshot = snapshots / "step_1000.pt"
    snapshot.write_bytes((train / "step_1000.pt").read_bytes())
    training = {"arguments": {"output": str(train)}}
    assert audit.verify_training_checkpoint(audit.record(snapshot), training)["bytes_sha256_identical"]
    snapshot.write_bytes(b"altered synthetic snapshot")
    with pytest.raises(ValueError, match="differs from its original"):
        audit.verify_training_checkpoint(audit.record(snapshot), training)
