"""Planner/observation coordinate agreement on CPU, without USD or physics."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from CrowdSim.nav_manager import CrowdNavigationManager
from CrowdSim.world.map import NavigationTask


def _manager(map_yaw):
    config = NS(map_resolution=1., map_origin_xy=(2., -3.), map_origin_yaw=map_yaw,
                rl_map_size=2, rl_map_extent=3.2)
    task = object.__new__(NavigationTask)
    task.config, task.height, task.width = config, 3, 4
    manager = object.__new__(CrowdNavigationManager)
    manager.config, manager.task = config, task
    manager.height, manager.width = 3, 4
    manager.obstacle_map = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]], np.uint8)
    manager._trajectory_log_file = manager._trajectory_timestamp_file = None
    return manager


@pytest.mark.parametrize("map_yaw", [0., math.pi / 2, -.3])
def test_local_patch_tracks_rotated_map_frame(map_yaw):
    manager = _manager(map_yaw)
    xy = manager.task.pixel_to_world(np.array([1, 1]))
    patch = manager._local_obstacle_patch(xy, map_yaw)
    # Robot forward follows map +x; image rows run opposite map +y.
    np.testing.assert_array_equal(patch, [[1., 1.], [1., 0.]])
    reverse = manager._local_obstacle_patch(xy, map_yaw + math.pi)
    np.testing.assert_array_equal(reverse, [[0., 1.], [1., 1.]])


def test_local_patch_keeps_out_of_map_unknown_instead_of_clamping_to_free_border():
    manager = _manager(math.pi / 2)
    manager.obstacle_map[:] = 0
    xy = manager.task.pixel_to_world(np.array([-10, -10]))
    np.testing.assert_array_equal(manager._local_obstacle_patch(xy, math.pi / 2), np.ones((2, 2)))
