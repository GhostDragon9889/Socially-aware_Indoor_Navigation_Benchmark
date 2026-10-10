"""Real scene route queries without simulation or learned planner substitutes."""
from pathlib import Path

import pytest
import torch

from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene


SCENE_CONFIG = Path(__file__).resolve().parents[1] / "configs/diffusion_amp/shared_obstacle_validation.yaml"


def test_control_frame_arrival_catches_pass_between_replans():
    scene = ReconstructionScene({
        "schema": "navisaaclab.reconstruction_scene.v1",
        "bounds_xy": [0, 0, 12, 12], "origins_xy": [[2, 2]],
        "routes_xy": [[[6, 6], [10, 6]]], "arrival_radius_m": .75,
        "goal_lookahead_m": 5., "cuboids": [],
    })
    # A real 3 m/s crossing sampled at 30 Hz enters the arrival disk during
    # the .5 s replanning interval, while neither interval endpoint is inside.
    points = torch.stack((torch.linspace(5.25, 6.75, 16), torch.full((16,), 6.3)), -1)
    for position in (points[:1], points[-1:]):
        _, reached = scene.advance_route(position, torch.tensor([0]))
        assert not reached.item()
    route_index = torch.tensor([0])
    count = 0
    for position in points:
        route_index, reached = scene.advance_route(position[None], route_index)
        count += int(reached.item())
    assert count == 1 and route_index.item() == 1
    assert scene._global_planner is None


@pytest.mark.parametrize("method", ["route_goal", "route_waypoints"])
def test_reset_planning_query_does_not_count_spawn_as_traversed_vertex(method):
    scene = ReconstructionScene.load(SCENE_CONFIG)
    actor_ids, old_index = torch.tensor([0]), torch.tensor([3])
    # The last route vertex equals the spawn. Falling before returning there
    # must not turn the reset position into evidence of physical progress.
    _, reached_before = scene.advance_route(torch.tensor([[15., 18.]]), old_index, actor_ids=actor_ids)
    assert not reached_before.item()
    position = scene.origins[:1].clone()
    query = getattr(scene, method)
    _, unchanged, reached = query(position, old_index, actor_ids=actor_ids, advance=False)
    assert unchanged.item() == 3 and not reached.item()
    assert old_index.item() == 3 and torch.equal(position, scene.origins[:1])
    # Starting a new attempt is an explicit runner decision, not advancement.
    _, restarted, reached = query(position, torch.tensor([0]), actor_ids=actor_ids, advance=False)
    assert restarted.item() == 0 and not reached.item()


def test_control_frame_subset_wraps_once_and_does_not_mutate_inputs():
    scene = ReconstructionScene.load(SCENE_CONFIG)
    ids, indices = torch.tensor([3, 1]), torch.tensor([3, 0])
    positions = scene.routes[ids, indices].clone()
    updated, reached = scene.advance_route(positions, indices, actor_ids=ids)
    assert updated.tolist() == [0, 1] and reached.tolist() == [True, True]
    assert indices.tolist() == [3, 0]
    assert torch.equal(positions, scene.routes[ids, indices])
    assert scene._global_planner is None


def test_empty_control_frame_query_is_pure():
    scene = ReconstructionScene.load(SCENE_CONFIG)
    updated, reached = scene.advance_route(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert updated.shape == reached.shape == (0,)
    assert updated.dtype == torch.long and reached.dtype == torch.bool
    assert scene._global_planner is None


@pytest.mark.parametrize("method", ["route_goal", "route_waypoints"])
def test_advance_flag_requires_explicit_boolean(method):
    scene = ReconstructionScene.load(SCENE_CONFIG)
    with pytest.raises(ValueError, match="boolean"):
        getattr(scene, method)(scene.origins, torch.zeros(4, dtype=torch.long), advance=1)
    assert scene._global_planner is None
