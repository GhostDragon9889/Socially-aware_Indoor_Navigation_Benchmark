"""Field-of-view selection must keep the matching agent velocity rows."""

import numpy as np
import pytest

from CrowdSim.world.sfm import Social_Force


@pytest.mark.parametrize("count", [3, 12])
def test_forward_neighbors_and_their_velocities_use_the_same_selection(count):
    ids = np.arange(count)
    xy = np.stack((np.arange(count) - 1.5, np.ones(count)), axis=1)
    distances = np.linalg.norm(xy, axis=1)
    velocities = np.stack((ids * 10 + 1, ids * 10 + 2), axis=1)
    selected = ids[xy[:, 0] >= 0]
    selected = selected[np.argsort(distances[selected])[:5]]
    observed = Social_Force.fov_filter(
        np.array([1.0, 0.0]), (ids, distances, xy, velocities)
    )
    for actual, expected in zip(
        observed,
        (selected, distances[selected], xy[selected], velocities[selected]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def force():
    return Social_Force(
        np.full((16, 16), 10.0, dtype=np.float32),
        {
            "env": {
                "dt": 0.04,
                "safe_distance": 0.9,
                "neighbor_radius": 4.0,
                "reach_distance": 0.4,
                "prediction_horizon": 2.0,
                "ttc_threshold": 1.5,
                "ttc_gain": 12.0,
            },
            "agent": {"radius": 0.1, "max_vel": 1.5},
            "map": {"resolution": 0.1},
        },
    )


def test_full_sfm_with_filtered_neighbors_matches_explicit_front_only_input():
    state = (
        np.array([0.0, 0.0]),
        np.array([8, 8]),
        np.array([1.0, 0.0]),
        np.array([10.0, 0.0]),
    )
    ids = np.array([1, 2, 3])
    xy = np.array([[2.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    relative_velocity = np.array([[-2.0, 0.0], [3.0, 1.0], [5.0, -2.0]])
    distances = np.linalg.norm(xy, axis=1)
    controller = force()
    controller.fov = True
    actual = controller.get_action(state, (ids, distances, xy, relative_velocity))
    expected = force().get_action(
        state, (ids[:1], distances[:1], xy[:1], relative_velocity[:1])
    )
    np.testing.assert_allclose(actual[0], expected[0])
    for value, reference in zip(actual[1], expected[1], strict=True):
        np.testing.assert_allclose(value, reference)
    assert np.isfinite(actual[0]).all() and np.linalg.norm(actual[0]) <= 1.5 + 1e-6


def test_empty_neighbors_and_no_forward_neighbors_are_finite():
    for xy in (np.empty((0, 2)), np.array([[-2.0, 0.0], [-3.0, 0.0]])):
        count = len(xy)
        controller = force()
        controller.fov = True
        action, terms = controller.get_action(
            (
                np.array([0.0, 0.0]),
                np.array([8, 8]),
                np.array([1.0, 0.0]),
                np.array([10.0, 0.0]),
            ),
            (np.arange(count), np.linalg.norm(xy, axis=1), xy, np.zeros_like(xy)),
        )
        assert np.isfinite(action).all() and all(
            np.isfinite(term).all() for term in terms
        )
