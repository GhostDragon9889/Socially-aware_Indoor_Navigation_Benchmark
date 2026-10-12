"""CPU route audit tests: physical episodes, ordered wraps and reset distance."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "resources/runtime_sources/amp_improvement_20261001/analyze_route_completion.py"
SPEC = importlib.util.spec_from_file_location("route_completion_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def fixture(indices, reached, *, episodes=None, reset=None):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    frames = len(indices)
    arrays = {"route_indices": indices, "route_vertex_reached": np.asarray(reached, dtype=bool).reshape(-1, 1),
              "episode_ids": np.asarray(episodes or [1]*frames, dtype=np.int64).reshape(-1, 1),
              "reset_mask": np.asarray(reset or [True]+[False]*(frames-1), dtype=bool).reshape(-1, 1),
              "terminated": np.zeros((frames, 1), dtype=bool), "truncated": np.zeros((frames, 1), dtype=bool),
              "root_pos": np.zeros((frames, 1, 3), dtype=np.float32)}
    runner = {"status": "passed", "route_completion_contract": {"reset_position_counts_as_progress": False},
              "route_vertices_reached": [int(arrays["route_vertex_reached"].sum())]}
    return arrays, runner


def test_two_partial_attempts_do_not_form_one_cycle():
    arrays, runner = fixture([1, 2, 2, 2, 1, 2, 2, 2], [1, 1, 0, 0, 1, 1, 0, 0],
                             episodes=[1]*4+[2]*4, reset=[1, 0, 0, 0, 1, 0, 0, 0])
    arrays["terminated"][3, 0] = True
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["status"] == "checked" and report["reported_count_match"]
    assert report["route_reached_total_per_actor"] == [4]
    assert report["complete_cycles_per_actor"] == [0]
    assert [episode["route_vertices_reached"] for episode in report["actors"][0]["episodes"]] == [2, 2]


def test_multiple_ordered_cycles_with_dwell_frames_and_later_fall():
    arrays, runner = fixture([1, 2, 3, 0, 0, 1, 2, 3, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    arrays["terminated"][-1, 0] = True
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["complete_cycles_total"] == 2
    episode = report["actors"][0]["episodes"][0]
    assert [event["vertex_index"] for event in episode["reached_events"]] == list(range(4))*2
    assert episode["ended_by_fall"] and not episode["right_censored"]


def test_index_jump_without_real_arrival_invalidates_completion():
    arrays, runner = fixture([1, 2, 0, 1], [1, 1, 1, 1])
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["status"] == "invalid"
    assert report["complete_cycles_total"] is None
    assert any(issue["kind"] == "index_event_disagreement" for issue in report["issues"])


def test_reset_teleport_does_not_contribute_walked_distance():
    arrays, runner = fixture([0]*4, [0]*4, episodes=[1, 1, 2, 2], reset=[1, 0, 1, 0])
    arrays["root_pos"][:, 0, 0] = [0, 1, 101, 103]
    arrays["terminated"][1, 0] = True
    report = ANALYSIS.analyze_arrays(arrays, runner)
    actor = report["actors"][0]
    assert actor["walked_distance_xy_m_excluding_resets"] == 3
    assert actor["excluded_reset_displacements"] == 1
    assert [episode["walked_distance_xy_m_excluding_resets"] for episode in actor["episodes"]] == [1, 2]


def test_arrival_on_fallen_frame_is_not_a_valid_cycle():
    arrays, runner = fixture([1, 2, 3, 0], [1]*4)
    arrays["terminated"][-1, 0] = True
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["status"] == "invalid" and report["complete_cycles_total"] is None
    assert any(issue["kind"] == "arrival_on_fall_frame" for issue in report["issues"])


def test_missing_old_fields_and_unknown_initial_attempt_are_not_guessed():
    arrays, runner = fixture([3, 0], [0, 1], reset=[0, 0])
    partial = ANALYSIS.analyze_arrays(arrays, runner)
    assert partial["complete_cycles_total"] is None
    arrays.pop("route_vertex_reached")
    old = ANALYSIS.analyze_arrays(arrays, runner)
    assert old["status"] == "unavailable"
    assert old["route_reached_total_per_actor"] is old["complete_cycles_per_actor"] is None
    assert old["reported_count_match"] is None


def test_runner_totals_are_checked_independently_of_ordered_cycle_count():
    arrays, runner = fixture([1, 2, 3, 0], [1]*4)
    runner["route_vertices_reached"] = [8]
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["status"] == "invalid" and report["reported_count_match"] is False
    assert report["complete_cycles_per_actor"] == [1]


def test_actor_episodes_and_wraps_are_independent():
    arrays, runner = fixture([1, 2, 3, 0], [1]*4)
    arrays = {name: np.repeat(value, 2, axis=1) for name, value in arrays.items()}
    arrays["route_indices"][:, 1] = [1, 1, 1, 1]
    arrays["route_vertex_reached"][:, 1] = [1, 0, 0, 0]
    runner["route_vertices_reached"] = [4, 1]
    report = ANALYSIS.analyze_arrays(arrays, runner)
    assert report["complete_cycles_per_actor"] == [1, 0]
    assert report["route_reached_total_per_actor"] == [4, 1]


def test_real_npz_cli_and_hashed_provenance(tmp_path):
    arrays, runner = fixture([1, 2, 3, 0], [1]*4)
    np.savez_compressed(tmp_path / "physical_states.npz", **arrays)
    (tmp_path / "result.json").write_text(json.dumps(runner))
    assert ANALYSIS.main(["--run-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "route_completion_analysis.json").read_text())
    assert report["complete_cycles_total"] == 1
    assert len(report["sources"]["capture"]["sha256"]) == 64
    assert "ordered traversal" in (tmp_path / "route_completion_analysis.md").read_text()
    runner["status"] = "running"
    (tmp_path / "result.json").write_text(json.dumps(runner))
    with pytest.raises(ValueError, match="incomplete capture"):
        ANALYSIS.analyze_run(tmp_path)


def test_boolean_route_flag_is_required():
    arrays, runner = fixture([1], [1])
    arrays["route_vertex_reached"] = arrays["route_vertex_reached"].astype(np.int32)
    with pytest.raises(ValueError, match="route_vertex_reached"):
        ANALYSIS.analyze_arrays(arrays, runner)
