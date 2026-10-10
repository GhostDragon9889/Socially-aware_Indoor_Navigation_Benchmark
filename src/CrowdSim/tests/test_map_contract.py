"""ROS occupancy image, rotated coordinate frame and marker regressions."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
from PIL import Image

from CrowdSim.utils.map_metadata import (
    _free_thresh_to_gray_threshold,
    load_occupancy_map_metadata,
)
from CrowdSim.world.map import CollisionDetector, NavigationTask, NavigationTaskMarkers


def _task(config, *, height=4, width=5):
    task = NavigationTask.__new__(NavigationTask)
    task.config = config
    task.height, task.width = height, width
    return task


def _yaml(tmp_path, *, negate=0, yaw=0.0, resolution=0.25, free=0.196):
    image = tmp_path / "map.png"
    Image.fromarray(np.array([[0, 49, 50, 127, 205, 206, 255]], dtype=np.uint8)).save(image)
    path = tmp_path / "map.yaml"
    path.write_text(
        f"image: map.png\nresolution: {resolution}\norigin: [-3.0, 4.0, {yaw}]\n"
        f"negate: {negate}\nfree_thresh: {free}\noccupied_thresh: 0.65\n"
    )
    return path


@pytest.mark.parametrize("negate", [0, 1])
def test_ros_yaml_negate_and_unknown_classification(tmp_path, negate):
    metadata = load_occupancy_map_metadata(_yaml(tmp_path, negate=negate, yaw=0.7))
    assert metadata.origin_yaw == 0.7
    assert metadata.negate == negate
    assert metadata.free_threshold == (49 if negate else 206)
    task = _task(SimpleNamespace(free_threshold=metadata.free_threshold, map_negate=metadata.negate))
    free, obstacles = task._load_map(metadata.image_path)
    expected = [[True, True, False, False, False, False, False]] if negate else [[False, False, False, False, False, True, True]]
    np.testing.assert_array_equal(free, expected)
    np.testing.assert_array_equal(obstacles, np.logical_not(expected))


@pytest.mark.parametrize("free", [0.0, 0.196, 0.2, 1 / 3, 0.5, 2 / 3, 1.0])
def test_gray_boundary_matches_occupancy_probability(free):
    pixels = np.arange(256)
    for negate in (0, 1):
        threshold = _free_thresh_to_gray_threshold(free, negate)
        actual = pixels <= threshold if negate else pixels >= threshold
        probability = pixels / 255.0 if negate else (255 - pixels) / 255.0
        np.testing.assert_array_equal(actual, probability <= free)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, math.pi / 6, -2.3])
def test_rotated_image_cell_centers_round_trip(yaw):
    config = SimpleNamespace(map_origin_xy=(-3.2, 4.4), map_origin_yaw=yaw, map_resolution=0.25)
    task = _task(config)
    pixels = np.stack(np.meshgrid(np.arange(4), np.arange(5), indexing="ij"), axis=-1)
    world = task.pixel_to_world(pixels)
    np.testing.assert_array_equal(task.world_to_pixel(world), pixels)
    bottom_left = task.pixel_to_world(np.array([3, 0]))
    rotated_center = np.array([
        math.cos(yaw) * 0.125 - math.sin(yaw) * 0.125,
        math.sin(yaw) * 0.125 + math.cos(yaw) * 0.125,
    ]) + np.array(config.map_origin_xy)
    np.testing.assert_allclose(bottom_left, rotated_center, atol=1e-12)


def test_world_outside_map_is_not_clamped_and_collision_is_wall():
    config = SimpleNamespace(map_origin_xy=(-3.0, 4.0), map_origin_yaw=math.pi / 2, map_resolution=0.25)
    task = _task(config)
    outside = np.array([[-3.3, 3.9]])  # inverse rotation gives x=-0.1
    pixel = task.world_to_pixel(outside)[0]
    assert pixel[1] == -1
    detector = CollisionDetector(
        np.zeros((4, 5), dtype=np.uint8), 0.25, 0.7, 0.1,
        task.world_to_pixel, 4, 5,
    )
    assert detector.hits_wall(outside).tolist() == [True]


def test_alpha_unknown_is_not_traversable(tmp_path):
    path = tmp_path / "alpha.png"
    Image.fromarray(np.array([[[255, 255, 255, 255], [255, 255, 255, 0]]], dtype=np.uint8)).save(path)
    free, _ = _task(SimpleNamespace(free_threshold=200, map_negate=0))._load_map(path)
    assert free.tolist() == [[True, False]]


def test_plain_image_remains_centered_under_cell_corner_contract(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("L", (5, 3), 255).save(path)
    metadata = load_occupancy_map_metadata(path)
    config = SimpleNamespace(map_origin_xy=metadata.origin_xy, map_origin_yaw=0.0, map_resolution=metadata.resolution)
    task = _task(config, height=3, width=5)
    np.testing.assert_allclose(task.pixel_to_world([1, 2]), [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"negate": 2}, {"resolution": 0.0}, {"yaw": float("nan")}, {"free": -0.1}])
def test_invalid_map_metadata_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_occupancy_map_metadata(_yaml(tmp_path, **kwargs))


def test_marker_identity_matches_pinned_isaaclab_xyzw():
    markers = NavigationTaskMarkers(enabled=False)
    _, orientations, _, _ = markers._static_marker_arrays(
        paths_xy=[np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)],
        goals_xy=np.array([[2, 0]], dtype=np.float32), num_humanoids=1,
    )
    np.testing.assert_array_equal(orientations, [[0, 0, 0, 1], [0, 0, 0, 1]])
