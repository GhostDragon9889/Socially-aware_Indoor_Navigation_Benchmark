"""Scene-object geometry must constrain initial routes as well as later resets."""

import json

import numpy as np
import pytest

from navigation_harness import attach_memory_environment, make_manager


def test_authored_scene_box_is_included_before_initial_route_sampling(tmp_path):
    original = make_manager(tmp_path / "original")
    try:
        # Occupy a spawn chosen from the original, unpainted map. A correct
        # construction must sample another free point rather than keep it.
        blocked = original.starts_xy[0].copy()
    finally:
        original._close_trajectory_logs()
    manager = make_manager(
        tmp_path / "painted",
        scene_objects={
            "box": [float(blocked[0]), float(blocked[1]), 0.0, 1.0, 1.0, 1.0],
            "cylinder": [6.0, 6.0, 0.0, 0.6, 0.6, 1.0, "cylinder"],
        },
    )
    try:
        assert not manager.collision.hits_wall(manager.starts_xy).any()
        assert not manager.collision.hits_wall(manager.goals_xy).any()
        for path in manager.paths_xy:
            assert not manager.collision.hits_wall(path).any()
        pixel = manager.world_to_pixel(blocked)
        assert manager.obstacle_map[tuple(pixel)] == 1
        assert not manager.task.planner_free_mask[tuple(pixel)]
        assert not manager.free_mask[tuple(pixel)]
        assert all(
            not manager.obstacle_map[tuple(point)]
            for point in manager.task._free_pixels_cache
        )
    finally:
        manager._close_trajectory_logs()


def test_runtime_obstacle_paint_rebuilds_planner_and_sampler(tmp_path):
    manager = make_manager(tmp_path)
    try:
        manager._paint_scene_objects_to_map({"box": [6.0, 6.0, 0.0, 1.0, 1.0, 1.0]})
        pixel = manager.world_to_pixel(np.array([6.0, 6.0]))
        assert manager.collision.hits_wall(np.array([[6.0, 6.0]])).all()
        assert not manager.task.planner_free_mask[tuple(pixel)]
        assert not manager.free_mask[tuple(pixel)]
        for _ in range(100):
            assert manager.task.planner_free_mask[
                tuple(manager.task._sample_free_pixel())
            ]
    finally:
        manager._close_trajectory_logs()


def test_runtime_wall_split_rebuilds_connected_component_sampling(tmp_path):
    manager = make_manager(tmp_path)
    try:
        task = manager.task
        assert len(task.component_sizes) == 2  # background and one free component
        # Populate the component cache before changing map topology.
        task._pixels_for_component(1)
        manager._paint_scene_objects_to_map({"wall": [6.4, 6.4, 0.0, 0.4, 12.8, 1.0]})
        assert len(task.component_sizes) == 3
        assert not task._component_pixel_cache
        for label in (1, 2):
            pixels = task._pixels_for_component(label)
            assert np.all(task.component_labels[pixels[:, 0], pixels[:, 1]] == label)
            assert np.all(task.planner_free_mask[pixels[:, 0], pixels[:, 1]])
            start = pixels[len(pixels) // 2]
            result = task._sample_goal_and_path_for_start(start, 5, 20, 200)
            assert result is not None
            goal, route = result
            assert task.component_labels[tuple(goal)] == label
            assert not manager.collision.hits_wall(route).any()
    finally:
        manager._close_trajectory_logs()


def test_fixed_flow_rejects_scene_obstacles_across_its_route(tmp_path):
    # Fixed lanes cannot be resampled around an authored obstacle. Reject the
    # configuration before publishing a trajectory containing unsafe spawns.
    with pytest.raises(ValueError, match="traversable|obstacle"):
        make_manager(
            tmp_path,
            num_humanoids=4,
            scenario={
                "type": "bidirectional_flow",
                "point_a": [4.0, 2.0],
                "point_b": [4.0, 4.0],
                "point_c": [4.0, 8.0],
                "point_d": [4.0, 10.0],
                "lane_offset": 0.5,
            },
            scene_objects={"wall": [4.0, 6.0, 0.0, 4.0, 1.0, 1.0]},
        )
    assert not (tmp_path / "run").exists()


def test_runtime_paint_invalidates_previously_cached_astar_routes(tmp_path):
    manager = make_manager(tmp_path)
    try:
        start, goal = np.array([64, 30]), np.array([64, 90])
        planner = manager.task.planner
        assert planner.get_astar_path(start, goal) is not None
        assert (tuple(start), tuple(goal)) in planner._astar_path_cache
        manager._paint_scene_objects_to_map({"wall": [6.4, 6.4, 0.0, 0.4, 12.8, 1.0]})
        assert planner.get_astar_path(start, goal) is None
    finally:
        manager._close_trajectory_logs()


@pytest.mark.parametrize("mirror", [False, True])
def test_clear_fixed_routes_reverse_without_humanoid_teleport_and_restore_on_reset(
    tmp_path, mirror
):
    manager = make_manager(
        tmp_path,
        num_humanoids=4,
        scenario={
            "type": "bidirectional_flow",
            "point_a": [4.0, 2.0],
            "point_b": [4.0, 4.0],
            "point_c": [4.0, 8.0],
            "point_d": [4.0, 10.0],
            "lane_offset": 0.5,
            "mirror": mirror,
            "robot_route": "b_to_c",
        },
    )
    try:
        env, _ = attach_memory_environment(manager)
        original = manager.task.fixed_agent_route(0)
        positions, _ = manager._read_agent_state()
        positions[0] = manager.goals_xy[0]
        manager._update_waypoints_and_goals(positions)
        np.testing.assert_array_equal(manager.goals_xy[0], original[2])
        assert not manager.reached[0] and not env.reset_buf[0]
        assert manager._navigation_done_agent_ids({(0, 4), (1, -1)}).tolist() == [4]
        env.reset(env_ids=[0])
        np.testing.assert_array_equal(manager.starts_xy[0], original[2])
        np.testing.assert_array_equal(manager.goals_xy[0], original[3])
        frame = json.loads(manager.trajectory_log_path.read_text().splitlines()[-1])
        latest = {record["agent_id"]: record for record in frame["path_updates"]}
        assert latest[0]["goal_xy"] == manager.goals_xy[0].tolist()
        for path in manager.paths_xy:
            assert not manager.collision.hits_wall(path).any()
        manager.reset_robot_rl_episodes(np.array([True, False]))
        assert not manager._robot_last_dones.any()
    finally:
        manager._close_trajectory_logs()


@pytest.mark.parametrize("pixel", [[-1, 2], [2, -1], [128, 2], [2, 128]])
def test_out_of_bounds_traversability_never_wraps_numpy_indices(tmp_path, pixel):
    manager = make_manager(tmp_path)
    try:
        assert not manager.task._is_white_traversable(np.array(pixel))
    finally:
        manager._close_trajectory_logs()
