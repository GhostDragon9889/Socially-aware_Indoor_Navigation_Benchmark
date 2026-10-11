import numpy as np
import pytest
from CrowdSim.world.planning import Path_Planner


def planner(**options):
    return Path_Planner(np.zeros((5, 5), np.uint8), map_resolution=1., step_size_m=1., clearance_m=0., verbose=False, **options)


def test_zero_clearance_does_not_inflate_obstacle_cells():
    grid = np.zeros((5, 5), np.uint8)
    grid[2, 2] = 1
    p = Path_Planner(grid, map_resolution=1., clearance_m=0., verbose=False)
    np.testing.assert_array_equal(p.map_dialate, grid)


def test_astar_rejects_diagonal_corner_cutting():
    p = planner()
    p.map_dialate = np.array([[0, 1], [1, 0]], np.uint8)
    p.map_size = (2, 2)
    assert p.get_astar_path((0, 0), (1, 1)) is None


@pytest.mark.parametrize('start,goal', [((2, 2), (2, 3)), ((2, 3), (2, 2)), ((2, 2), (2, 2))])
def test_astar_never_returns_route_with_occupied_endpoint(start, goal):
    p = planner()
    p.map_dialate[2, 2] = 1
    assert p.get_astar_path(start, goal) is None


@pytest.mark.parametrize('start,goal', [((-1, 1), (1, 1)), ((1, 1), (5, 1))])
def test_astar_outside_map_endpoint_is_unreachable(start, goal):
    assert planner().get_astar_path(start, goal) is None


def test_curvature_for_repeated_points_is_zero_and_finite():
    result = planner().calculate_curvature(np.array([[0., 0.], [0., 0.], [1., 1.], [2., 2.]]))
    np.testing.assert_array_equal(result, [0., 0.])


def test_two_point_path_supports_smoothing():
    p = planner(smooth=True)
    np.testing.assert_allclose(p.smooth_with_b_spline(np.array([[0., 0.], [1., 1.]]), 3), [[0., 0.], [.5, .5], [1., 1.]])


def test_control_point_selection_preserves_both_endpoints():
    path = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    chosen = planner().select_control_points(path)
    np.testing.assert_array_equal(chosen[0], path[0])
    np.testing.assert_array_equal(chosen[-1], path[-1])


def test_b_spline_cannot_replace_verified_astar_with_route_through_obstacles():
    rng = np.random.default_rng(3)
    grid = (rng.random((12, 12)) < .15).astype(np.uint8)
    grid[0, 0] = grid[11, 11] = 0
    plain = Path_Planner(grid, 1., step_size_m=1., clearance_m=0., smooth=False, verbose=False)
    checked = plain.get_astar_path((0, 0), (11, 11))
    smooth = Path_Planner(grid, 1., step_size_m=1., clearance_m=0., smooth=True, verbose=False)
    actual = smooth.get_astar_path((0, 0), (11, 11))
    np.testing.assert_array_equal(actual, checked)
    assert all(smooth._edge_is_free(tuple(a), tuple(b)) for a, b in zip(actual[:-1], actual[1:]))
