"""Real DynBench map wiring and exporter-style YAML coordinate contracts."""

from protomotions.repository import repository_root

from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest
from PIL import Image

from CrowdSim.humanoid.diffusion.raster import rasterize_navigation_map
from CrowdSim.utils.config_loader import cfg_path, load_config
from CrowdSim.utils.map_metadata import load_occupancy_map_metadata
from CrowdSim.world.map import NavigationTask


ROOT = repository_root(__file__)
SCENES = (
    ("office", "Office"),
    ("hospital", "Hospital"),
    ("warehouse", "Warehouse"),
    ("full_warehouse", "Full_Warehouse"),
)


def _task(metadata):
    task = NavigationTask.__new__(NavigationTask)
    task.config = SimpleNamespace(
        map_origin_xy=metadata.origin_xy,
        map_origin_yaw=metadata.origin_yaw,
        map_resolution=metadata.resolution,
        map_negate=metadata.negate,
        free_threshold=metadata.free_threshold,
    )
    task.free_mask, task.obstacle_map = task._load_map(metadata.image_path)
    task.height, task.width = task.free_mask.shape
    return task


@pytest.mark.parametrize("config_name,scene", SCENES)
def test_real_scene_config_resolves_exported_block_origin_map(config_name, scene):
    config = load_config(ROOT / "CrowdSim/config/scenes" / f"dynabench_{config_name}.yaml")
    source = cfg_path(config["scene_usd"], ROOT)
    path = cfg_path(config["scene_map"], ROOT)
    assert source.is_file()
    assert path == ROOT / "data/diffusion_amp/maps_resource_complete" / scene / "occupancy.yaml"
    assert "origin:\n- " in path.read_text()
    expected = load_config(path)
    metadata = load_occupancy_map_metadata(path)
    assert metadata.image_path.is_file()
    assert metadata.origin_xy == tuple(expected["origin"][:2])
    assert metadata.origin_yaw == expected["origin"][2]
    assert metadata.resolution == expected["resolution"] == 0.1
    assert metadata.negate == expected["negate"] == 0
    assert metadata.free_threshold == 206
    task = _task(metadata)
    assert task.free_mask.any() and task.obstacle_map.any()
    pixels = np.array([[0, 0], [task.height - 1, task.width - 1],
                       [task.height // 2, task.width // 2]])
    np.testing.assert_array_equal(task.world_to_pixel(task.pixel_to_world(pixels)), pixels)
    np.testing.assert_allclose(task.pixel_to_world([task.height - 1, 0]),
                               np.asarray(metadata.origin_xy) + 0.05, atol=1e-12)
    np.testing.assert_allclose(task.pixel_to_world([0, 0])[1],
                               metadata.origin_xy[1] + (task.height - 0.5) * 0.1, atol=1e-12)
    free_pixel = np.argwhere(task.free_mask)[len(np.argwhere(task.free_mask)) // 2]
    xy = task.pixel_to_world(free_pixel)
    raster = rasterize_navigation_map(task, [[*xy, 0.0, 0.0, 0.0]])
    assert tuple(raster.shape) == (1, 2, 224, 224)
    np.testing.assert_array_equal(raster[0, :, 112, 56].numpy(), [1.0, 0.0])


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("negate", [0, 1])
@pytest.mark.parametrize("yaw", [math.pi / 2, -0.73])
def test_yaml_styles_preserve_rotated_origin_and_negate(tmp_path, block, negate, yaw):
    image = tmp_path / "quoted # map.png"
    grid = np.array([[0, 49, 50, 127, 205, 206, 255],
                     [255, 206, 205, 127, 50, 49, 0]], dtype=np.uint8)
    Image.fromarray(grid).save(image)
    origin = f"origin:\n- -3.0\n- 4.0\n- {yaw}\n" if block else f"origin: [-3.0, 4.0, {yaw}]\n"
    path = tmp_path / "map.yaml"
    path.write_text(f'image: "{image.name}" # quoted filename\nresolution: 0.25\n'
                    + origin + f"negate: {negate}\nfree_thresh: 0.196\noccupied_thresh: 0.65\n")
    metadata = load_occupancy_map_metadata(path)
    assert metadata.image_path == image
    assert metadata.origin_xy == (-3.0, 4.0)
    assert metadata.origin_yaw == yaw
    assert metadata.free_threshold == (49 if negate else 206)
    task = _task(metadata)
    probabilities = grid.astype(np.float64) / 255 if negate else (255 - grid.astype(np.float64)) / 255
    np.testing.assert_array_equal(task.free_mask, probabilities <= 0.196)
    pixels = np.stack(np.meshgrid(np.arange(2), np.arange(7), indexing="ij"), axis=-1)
    world = task.pixel_to_world(pixels)
    np.testing.assert_array_equal(task.world_to_pixel(world), pixels)
    expected_corner = np.array([-3.0, 4.0]) + np.array([
        math.cos(yaw) * 0.125 - math.sin(yaw) * 0.125,
        math.sin(yaw) * 0.125 + math.cos(yaw) * 0.125,
    ])
    np.testing.assert_allclose(task.pixel_to_world([1, 0]), expected_corner, atol=1e-12)
    outside = np.array([[-0.125, 0.125]]) @ task._map_rotation().T + [-3.0, 4.0]
    assert task.world_to_pixel(outside)[0, 1] == -1

