"""Analytic and accounting contracts; synthetic counters are not SDK evidence."""
from copy import deepcopy

import numpy as np
import pytest

from CrowdSim.humanoid.analysis.episode_horizons import group_accounting, reset_prior


def test_shortest_legal_clip_has_one_center_and_keeps_all_source_ids():
    prior = reset_prior(np.array([41, 41], np.int64), 31)
    assert prior["clip_count"] == 2 and prior["all_clips_retained"]
    assert prior["legal_reset_centers"] == 2
    assert prior["ideal_remaining_mean_s"] == pytest.approx(31 / 30)
    assert prior["ideal_two_independent_AMASS_minimum_mean_s"] == pytest.approx(31 / 30)
    assert prior["ideal_reset_ten_point_beyond_source_fraction"] == pytest.approx(.8)
    assert prior["ideal_reset_full_five_second_real_probability"] == 0


def test_enumerated_clip_uniform_then_frame_uniform_is_not_flat_window_sampling():
    prior = reset_prior(np.array([41, 43], np.int64), 31)
    # Half probability on the one-center clip; the other half on three centers.
    p = np.array([2 / 3, 1 / 6, 1 / 6])
    expected = float((p * [31, 32, 33]).sum()) / 30
    minimum = sum(p[i] * p[j] * min(31 + i, 31 + j) for i in range(3) for j in range(3)) / 30
    assert prior["ideal_remaining_mean_s"] == pytest.approx(expected)
    assert prior["ideal_two_independent_AMASS_minimum_mean_s"] == pytest.approx(minimum)
    assert prior["ideal_remaining_cdf"][31] == pytest.approx(2 / 3)


def test_five_second_endpoint_is_real_and_only_later_samples_are_beyond_source():
    prior = reset_prior(np.array([160], np.int64), 150)
    assert prior["legal_reset_centers"] == 1
    assert prior["ideal_reset_full_five_second_real_probability"] == 1
    assert prior["ideal_reset_ten_point_beyond_source_fraction"] == 0


@pytest.mark.parametrize("counts,future", [(np.array([40], np.int64), 31),
    (np.array([91, 160], np.int64), 82), (np.array([91.], np.float64), 31),
    (np.array([], np.int64), 31), (np.array([91], np.int64), True)])
def test_infeasible_prior_fails_instead_of_filtering_short_clips(counts, future):
    with pytest.raises(ValueError):
        reset_prior(counts, future)


def groups():
    rows = np.array([[0, 1, 4, 5], [2, 3, 6, 7]], np.int64)
    age = np.zeros((3, 8), np.int64)
    age[:, rows[0]] = np.array([4, 5, 6])[:, None]
    age[:, rows[1]] = np.array([7, 8, 9])[:, None]
    reset = np.zeros((3, 8), np.bool_)
    stats = {"actor_controls": 8000, "reset_groups": 100, "reset_actor_rows": 400,
             "initial_reset_actor_rows": 8, "source_end_actor_controls": 101}
    return stats, rows, age, reset


def test_completed_episode_mean_removes_right_censored_age_and_retains_overlap_bounds():
    stats, rows, age, reset = groups()
    result = group_accounting(stats, rows, age, reset)
    assert result["right_censored_final_group_controls"] == 15
    assert result["completed_group_episode_mean_s"] == pytest.approx(1985 / 100 / 30)
    assert result["source_associated_group_reset_lower_bound"] == 51
    assert result["source_associated_group_reset_upper_bound"] == 100
    assert result["source_only_group_reset_count_known"] is False


def test_actual_final_reset_censors_zero_age_not_the_terminal_POST_age():
    stats, rows, age, reset = groups()
    reset[-1, rows[0]] = True
    result = group_accounting(stats, rows, age, reset)
    assert result["right_censored_final_group_controls"] == 9
    assert result["terminal_window_group_resets"] == 1


@pytest.mark.parametrize("bad", ["partial_reset", "unequal_age", "duplicated_actor", "wrong_mix",
    "reset_actor_counter", "initial_counter", "noninteger_controls", "source_bound", "short_controls",
    "age_shape", "reset_dtype", "age_exceeds_total"])
def test_capture_and_cumulative_counter_defects_cannot_yield_an_episode_mean(bad):
    stats, rows, age, reset = deepcopy(groups())
    if bad == "partial_reset":
        reset[-1, 0] = True
    elif bad == "unequal_age":
        age[-1, 0] += 1
    elif bad == "duplicated_actor":
        rows[0, 0] = 1
    elif bad == "wrong_mix":
        rows = np.arange(8, dtype=np.int64).reshape(2, 4)
    elif bad == "reset_actor_counter":
        stats["reset_actor_rows"] += 1
    elif bad == "initial_counter":
        stats["initial_reset_actor_rows"] = 0
    elif bad == "noninteger_controls":
        stats["actor_controls"] = True
    elif bad == "short_controls":
        stats["actor_controls"] = 8
    elif bad == "age_shape":
        age = age[-1]
    elif bad == "reset_dtype":
        reset = reset.astype(np.int64)
    elif bad == "age_exceeds_total":
        age[-1] = 1001
    else:
        stats["source_end_actor_controls"] = 201
    with pytest.raises(ValueError):
        group_accounting(stats, rows, age, reset)


def test_zero_resets_and_zero_completed_exposure_has_no_completed_mean():
    stats, rows, age, reset = groups()
    age[:] = np.arange(1, 4)[:, None]
    stats.update(actor_controls=24, reset_groups=0, reset_actor_rows=0, source_end_actor_controls=0)
    result = group_accounting(stats, rows, age, reset)
    assert result["completed_group_controls"] == 0
    assert result["completed_group_episode_mean_s"] is None


def test_completed_mean_reconstructs_known_episode_lengths_and_final_resets():
    stats, rows, age, reset = groups()
    # Group0 completed [3,5] then age2; group1 completed [4,6] and just reset.
    # Both groups have received ten controls. The complete episode sum is18.
    age[:] = 1
    age[-1, rows[0]] = 2
    age[-1, rows[1]] = 6
    reset[-1, rows[1]] = True
    stats.update(actor_controls=80, reset_groups=4, reset_actor_rows=16, source_end_actor_controls=0)
    result = group_accounting(stats, rows, age, reset)
    assert result["completed_group_controls"] == 18
    assert result["completed_group_episode_mean_s"] == pytest.approx(18 / 4 / 30)
