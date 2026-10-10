"""Timing and physical direction diagnostics for the heldout guidance sweep."""

import numpy as np
import pytest

from scripts.benchmark_paper_waypoint_guidance import direction_metrics, oracle_waypoints


def straight_case():
    states = np.array([[10., -3., 1., 0., 0.]])
    future = np.zeros((1, 52, 5))
    future[0, :, 0] = 10. + np.arange(1, 53) / 10
    future[0, :, 1] = -3.
    future[0, :, 2] = 1.
    xy = np.concatenate((states[:, None, :2], future[:, :50, :2]), axis=1)
    return states, future, xy


def test_real_time_half_second_and_five_second_indices_on_world_translated_path():
    states, future, xy = straight_case()
    result = direction_metrics(states, future, xy)
    assert result["endpoint_error_mean_m"] == result["ade_mean_m"] == 0
    assert result["first_half_second_goal_progress_mean_m"] == .5
    assert result["first_half_second_goal_direction_cosine_mean"] == 1
    assert result["first_half_second_expert_direction_cosine_mean"] == 1
    assert result["per_case"][0]["expert_goal_distance_m"] == 5
    assert result["max_segment_speed_m_s"] == pytest.approx(1)


def test_small_endpoint_error_does_not_hide_initial_wrong_direction_or_terminal_jump():
    states, future, xy = straight_case()
    xy[:, 1:6, 0] = 10. - np.arange(1, 6) / 10
    xy[:, -2, 0] = -5.
    result = direction_metrics(states, future, xy)
    assert result["endpoint_error_mean_m"] == 0
    assert result["first_half_second_goal_progress_mean_m"] == -.5
    assert result["first_half_second_expert_direction_cosine_mean"] == -1
    assert result["max_segment_speed_m_s"] == pytest.approx(200.)
    assert result["ade_mean_m"] > 0


def test_stationary_and_invalid_current_frame_are_explicit():
    states, future, xy = straight_case()
    xy[:] = states[:, None, :2]
    result = direction_metrics(states, future, xy)
    assert result["first_half_second_goal_progress_mean_m"] == 0
    assert result["first_half_second_goal_direction_cosine_mean"] == 0
    assert result["endpoint_error_mean_m"] == 5
    xy[:, 0, 0] += 1
    with pytest.raises(ValueError, match="frame0"):
        direction_metrics(states, future, xy)


def test_timed_oracle_points_use_next_frame_origin_and_include_exact_five_second_endpoint():
    _, future, _ = straight_case()
    point, time = oracle_waypoints(future, 1)
    np.testing.assert_array_equal(point[0, :, 0], [15.])
    np.testing.assert_array_equal(time, [5.])
    points, times = oracle_waypoints(future, 10)
    np.testing.assert_allclose(times, np.arange(.5, 5.1, .5))
    np.testing.assert_allclose(points[0, :, 0], 10. + times)
    np.testing.assert_array_equal(points[:, -1:], point)
    with pytest.raises(ValueError, match="1or10"):
        oracle_waypoints(future, 2)
