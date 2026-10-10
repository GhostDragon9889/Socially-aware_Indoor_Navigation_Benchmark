import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene


CONFIG = Path(__file__).resolve().parents[1]/"configs/diffusion_amp/shared_obstacle_validation.yaml"


@pytest.fixture
def scene():
    return ReconstructionScene.load(CONFIG)


def test_map_height_and_geometry_agree(scene):
    points = torch.tensor([[6., 6.], [12., 12.], [2.1, 12.], [21.9, 12.]])
    assert scene.occupied(points).tolist() == [False, True, True, True]
    assert scene.obstacle_height(points).tolist() == [0., 2., 2., 2.]
    raster = scene.raster(torch.cat((points, torch.zeros(4, 3)), -1))
    assert raster[:, 1, 112, 56].tolist() == [0., 1., 1., 1.]
    assert torch.equal(raster[:, 0]+raster[:, 1], torch.ones(4, 224, 224))


def test_heading_rotates_crop(scene):
    states = torch.tensor([[6., 12., 0., 0., 0.], [6., 12., 0., 0., math.pi/2]])
    raster = scene.raster(states)
    # World point(12,12): forward 6m at yaw0; right -6m at yaw+90deg.
    assert raster[0, 1, 112, 128] == 1
    assert raster[1, 1, 40, 56] == 1
    assert raster[1, 1, 112, 128] == 0


def test_unknown_domain_is_distinct_from_obstacles(scene):
    raster = scene.raster(torch.tensor([[1., 1., 0., 0., 0.]]))
    assert raster[0, :, 112, 56].tolist() == [.5, .5]


def test_global_waypoint_path_avoids_box_and_limits_lookahead(scene):
    indices = torch.zeros(4, dtype=torch.long)
    goal, updated, reached = scene.route_goal(scene.origins, indices)
    assert not scene.occupied(goal).any()
    assert ((goal-scene.origins).norm(dim=-1) <= scene.lookahead+1e-5).all()
    assert not reached.any() and torch.equal(updated, indices)
    assert scene.global_path_failures == 0


def test_route_reached_vertex_advances_without_moving_actor(scene):
    positions = scene.routes[:, 0].clone()
    goal, updated, reached = scene.route_goal(positions, torch.zeros(4, dtype=torch.long))
    assert reached.all() and updated.tolist() == [1, 1, 1, 1]
    assert not scene.occupied(goal).any()
    assert torch.equal(positions, scene.routes[:, 0])


def test_invalid_spawn_inside_actual_cuboid_rejected():
    description = yaml.safe_load(CONFIG.read_text())
    description["origins_xy"][0] = [12., 12.]
    with pytest.raises(ValueError, match="free scene"):
        ReconstructionScene(description)


def test_failed_global_path_keeps_actual_position(scene):
    positions = scene.origins.clone()
    positions[0] = torch.tensor([100., 100.])
    original = positions.clone()
    goal, _, _ = scene.route_goal(positions, torch.zeros(4, dtype=torch.long))
    assert torch.equal(goal[0], positions[0])
    assert torch.equal(positions, original)
    assert scene.global_path_failures == 1


def test_author_scene_provenance_is_explicit(scene):
    assert scene.description["author_benchmark_scene"] is False
    other = copy.deepcopy(scene.description)
    other["bounds_xy"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ReconstructionScene(other)
