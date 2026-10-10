"""Spawn sampling preserves the same map coordinates and free-space contract."""

import math

import numpy as np
from PIL import Image
import pytest
import torch

from CrowdSim.scene_setup import _pixel_to_world, sample_spawn_xy_from_map


def test_rotated_cell_center_matches_ros_origin_corner():
    point = _pixel_to_world(0, 2, 4, 3, 2., (10., 20.), math.pi / 2)
    np.testing.assert_allclose(point, (9., 21.), atol=1e-6)


def test_spawn_sampling_respects_negate_rotation_clearance_and_spacing(tmp_path):
    pixels = np.zeros((21, 21), dtype=np.uint8)
    pixels[[0, -1], :] = 255
    pixels[:, [0, -1]] = 255
    path = tmp_path / "map.png"
    Image.fromarray(pixels).save(path)
    locations = sample_spawn_xy_from_map(path, 5, torch.device("cpu"), map_resolution=.2,
        map_origin_xy=(2., 3.), humanoid_radius=.22, min_spacing=.43,
        free_threshold=55, map_negate=True, map_origin_yaw=math.pi / 2).numpy()
    delta = locations - np.array([2., 3.])
    local = np.stack((delta[:, 1], -delta[:, 0]), axis=-1)
    columns = np.floor(local[:, 0] / .2).astype(int)
    rows = 20 - np.floor(local[:, 1] / .2).astype(int)
    assert ((columns >= 3) & (columns <= 17) & (rows >= 3) & (rows <= 17)).all()
    distance = np.linalg.norm(locations[:, None] - locations[None, :], axis=-1)
    assert distance[np.triu_indices(5, 1)].min() >= .43 - 1e-6


def test_insufficient_clearance_raises_instead_of_spawning_on_unknown_grid(tmp_path):
    path = tmp_path / "map.png"
    Image.fromarray(np.full((3, 3), 255, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="Only 1/2"):
        sample_spawn_xy_from_map(path, 2, torch.device("cpu"), map_resolution=1., humanoid_radius=.5)


def test_transparent_pixels_are_not_free_and_zero_count_has_correct_shape(tmp_path):
    path = tmp_path / "transparent.png"
    pixels = np.full((3, 3, 4), 255, dtype=np.uint8)
    pixels[..., 3] = 0
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="No free"):
        sample_spawn_xy_from_map(path, 1, torch.device("cpu"))
    assert sample_spawn_xy_from_map(path, 0, torch.device("cpu")).shape == (0, 2)
