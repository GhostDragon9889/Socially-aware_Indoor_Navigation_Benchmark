"""Read-only comparison contracts; synthetic arrays are not simulation results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paper_entry_prefix_contract", HERE/"compare_physical_prefix.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def test_float_drift_and_signed_zero_are_not_promoted_to_bitwise_equality():
    left = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
    right = left.copy()
    right[1, 0] += .125
    report = reader.array_comparison(left, right)
    assert report["bitwise_equal"] is False
    assert report["first_differing_record"] == 1 and report["maximum_absolute_difference"] == .125
    right = left.copy()
    right[0, 0] = -0.
    report = reader.array_comparison(left, right)
    assert report["bitwise_equal"] is False and report["maximum_absolute_difference"] == 0.


def test_dtype_or_shape_changes_are_reported_without_casting_them_to_equality():
    left = np.ones((2, 3), dtype=np.float32)
    for right in (left.astype(np.float64), left[:1]):
        report = reader.array_comparison(left, right)
        assert report["same_shape_and_dtype"] is False and report["bitwise_equal"] is False
        assert report["maximum_absolute_difference"] is None


def test_changed_event_schema_cannot_be_silently_intersected():
    with pytest.raises(ValueError, match="schema differs"):
        reader.compare_fields({"reset_mask": np.zeros((2, 4), bool)}, {})


def arguments():
    old = json.loads((reader.ROOT/"tests/fixtures/paper_deployment_navigation_arguments.json").read_text())["arguments"]
    new = deepcopy(old)
    new.update(evaluate_steps=900, episode_steps=901, output="new_only",
               near_horizon_waypoint_mass=0.0)
    for name in ("config", "manifest", "diffusion_checkpoint", "resume", "scene_config"):
        new[name] = str((reader.ROOT/new[name]).resolve())
    return old, new


def test_recorded56_vs57_normalizes_only_explicit_eta0_paths_and_declared_duration():
    old, new = arguments()
    reader.assert_argument_contract(old, new)
    assert "near_horizon_waypoint_mass" not in old  # Caller data are not mutated.


@pytest.mark.parametrize("name,value", [("guidance_weight", 40.), ("near_horizon_waypoint_mass", .25),
                                      ("evaluate_steps", 901), ("episode_steps", 900),
                                      ("seed", 27), ("physics_only", False)])
def test_unrecorded_method_seed_or_time_changes_cannot_enter_matched_prefix(name, value):
    old, new = arguments()
    new[name] = value
    with pytest.raises(ValueError):
        reader.assert_argument_contract(old, new)


def test_sparse_plan_selection_keeps_actual_control_step_and_actor_identity(tmp_path):
    path = tmp_path/"plans.npz"
    np.savez(path, control_step=np.array([0, 0, 899, 900]), actor_ids=np.array([0, 1, 2, 3]),
             selected_sample=np.zeros(4, dtype=np.int64), raw_xy=np.arange(16).reshape(4, 2, 2))
    result = reader.prefix_arrays(path, 900, plans=True)
    assert result["control_step"].tolist() == [0, 0, 899]
    assert result["actor_ids"].tolist() == [0, 1, 2]
    assert result["raw_xy"][2].tolist() == [[8, 9], [10, 11]]


def test_prepare_state_has_no_candidate_bitwise_or_physics_success_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(reader, "load_helper", lambda: object())
    monkeypatch.setattr(reader, "prepare_sources", lambda: {"scope": "injected test source contract"})
    monkeypatch.setattr(reader, "baseline_oracle", lambda *_: ({}, {}, {"scope": "injected test oracle, not physical data"}))
    out = tmp_path/"new"
    assert reader.main(["--preparation-only", "--output", str(out)]) == 0
    result = json.loads((out/"comparison.json").read_text())
    assert result["status"] == "prepared_waiting_for_actual30s_capture"
    assert result["candidate_capture_available"] is False
    assert result["candidate_comparison_executed"] is False and result["paired_bitwise_equal"] is None


def test_prepare_cannot_also_claim_to_have_compared_a_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="Preparation cannot pretend"):
        reader.main(["--preparation-only", "--candidate-dir", str(tmp_path/"future"),
                     "--output", str(tmp_path/"new")])
    assert not (tmp_path/"new").exists()
