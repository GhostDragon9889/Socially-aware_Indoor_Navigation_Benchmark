"""CPU regression for actual repository A* and timed pedestrian guidance."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene


CONFIG = Path(__file__).resolve().parents[1]/"configs/diffusion_amp/shared_obstacle_validation.yaml"
TIMED_CONFIG = CONFIG.with_name("shared_obstacle_timed_validation.yaml")


@pytest.fixture
def scene():
    return ReconstructionScene.load(CONFIG)


def open_scene():
    return ReconstructionScene({
        "schema": "navisaaclab.reconstruction_scene.v1",
        "bounds_xy": [0, 0, 12, 12], "origins_xy": [[2, 2]],
        "routes_xy": [[[3, 2], [10, 2]]], "arrival_radius_m": .2,
        "goal_lookahead_m": 5, "cuboids": [],
    })


@pytest.mark.parametrize("lookahead,bits", [
    (5., [[1092578932, 1091043328], [1086324736, 1095902418],
          [1095761920, 1086324736], [1098383360, 1096990859]]),
    (3.7, [[1091215783, 1091043328], [1086324736, 1097265566],
           [1097125069, 1086324736], [1098383360, 1098354008]]),
])
def test_legacy_frozen_float_bits_and_timed_endpoint_exact(scene, lookahead, bits):
    # Captured from the original source before this change, not recomputed by
    # the new helper. Includes non-grid starts and a reached route vertex.
    positions = torch.tensor([[6., 6.], [6.113, 17.928], [18., 6.], [17.876, 18.057]])
    original = positions.clone()
    indices = torch.tensor([0, 2, 1, 3])
    scene.lookahead = lookahead
    goal, updated, reached = scene.route_goal(positions, indices)
    points, timed_updated, timed_reached = scene.route_waypoints(positions, indices)
    assert goal.view(torch.int32).tolist() == bits
    assert torch.equal(points[:, -1], goal)
    assert torch.equal(timed_updated, updated) and torch.equal(timed_reached, reached)
    assert torch.equal(positions, original)
    assert scene.global_path_failures == scene.successful_start_recoveries == 0


def test_same_one_astar_path_supplies_all_timestamps(scene, monkeypatch):
    planner = scene._get_global_planner()
    actual = planner.get_astar_path
    calls = []

    def count(start, end):
        calls.append((tuple(start), tuple(end)))
        return actual(start, end)

    monkeypatch.setattr(planner, "get_astar_path", count)
    points, _, _ = scene.route_waypoints(scene.origins, torch.zeros(4, dtype=torch.long))
    assert points.shape == (4, 10, 2)
    assert len(calls) == 4  # Real A*; not ten independent endpoints per actor.


def test_route_samples_follow_obstacle_detour(scene):
    scene.lookahead = 20.
    times = torch.linspace(.125, 5, 40)
    points, _, _ = scene.route_waypoints(scene.origins[:1], torch.tensor([0]), times=times)
    assert not scene.occupied(points).any()
    # Check between samples, too: straight endpoint interpolation would cross
    # the central [10,14] x [10,14] physical box on this diagonal route.
    polyline = torch.cat((scene.origins[:1, None], points), 1)
    fractions = torch.linspace(0, 1, 51)[None, None, :, None]
    dense = polyline[:, :-1, None]+fractions*(polyline[:, 1:, None]-polyline[:, :-1, None])
    assert not scene.occupied(dense).any()
    assert torch.equal(points[0, -1], scene.routes[0, 0])
    assert torch.linalg.vector_norm(polyline[:, 1:]-polyline[:, :-1], dim=-1).sum() > 17.0


def test_straight_path_arc_times_and_short_path_holds_endpoint():
    scene = open_scene()
    points, updated, reached = scene.route_waypoints(torch.tensor([[2., 2.]]), torch.tensor([0]))
    expected = torch.tensor([[[2.5, 2.]]+[[3., 2.]]*9])
    assert torch.equal(points, expected)
    assert updated.tolist() == [0] and reached.tolist() == [False]


def test_nondefault_actor_subset_and_route_indices(scene):
    ids, indices = torch.tensor([3, 1]), torch.tensor([2, 3])
    positions = scene.origins[ids].clone()
    points, updated, reached = scene.route_waypoints(positions, indices, actor_ids=ids)
    for row in range(2):
        single, single_updated, single_reached = scene.route_waypoints(
            positions[row:row+1], indices[row:row+1], actor_ids=ids[row:row+1])
        assert torch.equal(points[row], single[0])
        assert updated[row] == single_updated[0] and reached[row] == single_reached[0]
    assert torch.equal(positions, scene.origins[ids])


def test_reached_vertex_wraps_once_and_never_mutates_input(scene):
    ids, indices = torch.tensor([2, 0]), torch.tensor([3, 0])
    positions = scene.routes[ids, indices].clone()
    before = positions.clone()
    _, updated, reached = scene.route_waypoints(positions, indices, actor_ids=ids)
    assert reached.tolist() == [True, True] and updated.tolist() == [0, 1]
    assert torch.equal(positions, before) and indices.tolist() == [3, 0]


@pytest.mark.parametrize("times", [[], [0, 5], [-1, 5], [.5, 5.01], [1, 1],
                                   [2, 1], [[.5, 5]], [float("nan")],
                                   [float("inf")], [True]])
def test_invalid_timestamps_rejected_before_planning(scene, times):
    with pytest.raises(ValueError, match="times"):
        scene.route_waypoints(scene.origins[:1], torch.tensor([0]), times=times)
    assert scene._global_planner is None and scene.global_path_failures == 0


@pytest.mark.parametrize("positions,indices,ids", [
    ([[6., 6., 0.]], [0], None), ([[float("nan"), 6.]], [0], None),
    ([[6., 6.]], [[0]], None), ([[6., 6.]], [0., ], None),
    ([[6., 6.]], [True], None), ([[6., 6.]], [-1], None),
    ([[6., 6.]], [4], None), ([[6., 6.]], [0, 1], None),
    ([[6., 6.]], [0], [4]), ([[6., 6.]], [0], [-1]),
    ([[6., 6.]], [0], [0.]), ([[6., 6.]], [0], [[0]]),
])
def test_invalid_actor_or_route_shapes_rejected(scene, positions, indices, ids):
    with pytest.raises(ValueError):
        scene.route_waypoints(positions, indices, actor_ids=ids)
    assert scene._global_planner is None


def test_empty_batch(scene):
    points, indices, reached = scene.route_waypoints(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert points.shape == (0, 10, 2) and indices.shape == reached.shape == (0,)
    assert scene._global_planner is None


def test_selected_timestamps_and_grad_input_use_same_five_second_endpoint(scene):
    positions = scene.origins[:1].clone().requires_grad_(True)
    points, _, _ = scene.route_waypoints(positions, torch.tensor([0]), times=[.25, 3., 5.])
    endpoint, _, _ = scene.route_goal(positions, torch.tensor([0]))
    assert points.shape == (1, 3, 2) and torch.equal(points[:, -1], endpoint)
    assert not points.requires_grad  # CPU A* is deliberately nondifferentiable.
    assert torch.equal(positions.detach(), scene.origins[:1])


def test_missing_actor_ids_cannot_silently_wrap_a_larger_batch(scene):
    with pytest.raises(ValueError, match="actor_ids"):
        scene.route_waypoints(scene.origins[:1].expand(5, -1), torch.zeros(5, dtype=torch.long))


def test_disconnected_route_holds_root_and_counts_once_per_actor():
    description = copy.deepcopy(open_scene().description)
    description["cuboids"] = [{"center": [6, 6, 1], "size": [1, 12, 2]}]
    description["routes_xy"] = [[[10, 2], [10, 10]]]
    scene = ReconstructionScene(description)
    position = torch.tensor([[2., 2.]])
    points, _, _ = scene.route_waypoints(position, torch.tensor([0]))
    assert torch.equal(points, position[:, None].expand(-1, 10, -1))
    assert scene.global_path_failures == 1 and scene.successful_start_recoveries == 0


def test_recovery_actual_root_segment_and_failure_counters(scene):
    scene.start_recovery_radius = .5
    positions = torch.tensor([[9.6, 12.], [12., 12.], [100., 12.], [2.1, 12.]])
    original = positions.clone()
    points, _, _ = scene.route_waypoints(positions, torch.zeros(4, dtype=torch.long),
                                        actor_ids=torch.zeros(4, dtype=torch.long))
    assert scene.successful_start_recoveries == 1 and scene.global_path_failures == 3
    assert torch.equal(points[1:], positions[1:, None].expand(-1, 10, -1))
    assert torch.equal(positions, original)  # No teleport or in-place root correction.
    assert not scene.occupied(points[0]).any()
    record, = scene.last_start_recovery_segments
    assert record["actor_id"] == record["query_row"] == 0
    assert record["actual_position_xy"] == positions[0].tolist()
    assert record["planner_start_xy"] == [9.25, 12.]
    assert record["initial_segment_clearance_certified"] is False
    # .35m actual-root -> planner-node connector must consume the first .5m
    # lookahead, leaving .15m along the free path, rather than teleporting it.
    assert points[0, 0].tolist() == pytest.approx([9.25, 12.15], abs=1e-6)


def test_too_small_recovery_radius_fails_without_fake_motion(scene):
    scene.start_recovery_radius = .5
    positions = torch.tensor([[9.9, 12.]])
    points, _, _ = scene.route_waypoints(positions, torch.tensor([0]))
    assert torch.equal(points, positions[:, None].expand(-1, 10, -1))
    assert scene.global_path_failures == 1 and scene.successful_start_recoveries == 0
    assert scene.last_start_recovery_segments == []


def test_default_endpoint_never_uses_configured_recovery(scene):
    position, indices = torch.tensor([[9.6, 12.]]), torch.tensor([0])
    before, _, _ = scene.route_goal(position, indices)
    scene.start_recovery_radius = .5
    after, _, _ = scene.route_goal(position, indices)
    assert torch.equal(before, after) and scene.successful_start_recoveries == 0
    scene.route_waypoints(position, indices)
    assert scene.successful_start_recoveries == 1
    scene.route_waypoints(scene.origins[:1], indices)
    assert scene.last_start_recovery_segments == [] and scene.successful_start_recoveries == 1


def test_recovery_connector_cannot_cross_a_physical_thin_wall():
    scene = open_scene()
    scene.centers = scene.tensor([[2.125, 6, 1]])
    scene.sizes = scene.tensor([[.1, 12, 2]])
    assert not scene._connector_avoids_cuboids(torch.tensor([2., 2.]).numpy(),
                                               torch.tensor([2.25, 2.]).numpy())
    assert scene._connector_avoids_cuboids(torch.tensor([2., 2.]).numpy(),
                                          torch.tensor([1.75, 2.]).numpy())


@pytest.mark.parametrize("radius", [-.01, .751, float("nan"), float("inf"), True])
def test_invalid_recovery_radius_rejected(radius):
    description = copy.deepcopy(open_scene().description)
    description["start_recovery_radius_m"] = radius
    with pytest.raises(ValueError, match="recovery radius"):
        ReconstructionScene(description)


def test_timed_variant_changes_only_guidance_declaration():
    original, timed = yaml.safe_load(CONFIG.read_text()), yaml.safe_load(TIMED_CONFIG.read_text())
    assert timed.pop("route_guidance") == "timed_path"
    assert timed.pop("start_recovery_radius_m") == .5
    assert original == timed and timed["author_benchmark_scene"] is False
