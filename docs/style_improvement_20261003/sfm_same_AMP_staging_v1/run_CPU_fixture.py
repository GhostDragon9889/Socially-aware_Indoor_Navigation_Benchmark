"""One authorized wall/two-body CPU fixture; no simulator/model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from sfm_target_generator import SFMConfig, SFMTargetGenerator, WorldEDTMap


HERE = Path(__file__).resolve().parent


def record(path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    # A horizontal wall in top-left image coordinates; metric cell centres.
    free = np.ones((40, 60), dtype=bool)
    free[15] = False
    wall_map = WorldEDTMap(free, origin_xy_m=(-3., -2.), resolution_m=.1)
    generator = SFMTargetGenerator({0: wall_map, 1: wall_map})
    controller = generator._controllers[0]
    above, below = np.array([-.95, .95]), np.array([-.95, -.05])
    check("top_left_world_Y_flip", np.array_equal(wall_map.pixels(above), [10, 20])
          and np.array_equal(wall_map.pixels(below), [20, 20]))
    check("EDT_pixels_to_metres_without_radius_inflation",
          np.isclose(wall_map.edt_m[10, 20], .5) and np.isclose(wall_map.edt_m[20, 20], .5))
    empty_neighbors = (np.empty(0, dtype=np.int64), np.empty(0), np.empty((0, 2)), np.empty((0, 2)))
    above_force = controller.get_action((above, wall_map.pixels(above), np.zeros(2), above), empty_neighbors)[1][1]
    below_force = controller.get_action((below, wall_map.pixels(below), np.zeros(2), below), empty_neighbors)[1][1]
    check("EDT_gradient_and_static_force_away_from_wall", above_force[1] > 0 and below_force[1] < 0
          and np.isclose(above_force[0], 0) and np.isclose(below_force[0], 0))

    states = np.array([[-.95, -.45, .2, .1, .3], [-.65, -.65, -.1, .1, -.3]], dtype=np.float32)
    routes = np.array([[[-.95, -.45], [-.5, -.45], [1., -.45]],
                       [[-.65, -.65], [-.15, -.65], [1., -.65]]], dtype=np.float32)
    groups, ids = np.array([0, 0]), np.array([7, 23])
    before = [array.copy() for array in (states, routes, groups, ids)]
    pair = generator.generate(states, routes, scene_ids=groups, actor_ids=ids)
    check("51_knots_exact_actual_t0_and_reference_time_axis", pair.xy.shape == (2, 51, 2)
          and np.array_equal(pair.xy[:, 0], states[:, :2])
          and np.array_equal(pair.velocity_xy[:, 0], states[:, 2:4])
          and np.array_equal(pair.times_s, np.arange(51, dtype=np.float64) / 10)
          and pair.times_s[-1] == 5.)
    # First velocities independently queried from the same old full-scene state.
    expected = np.empty((2, 2), dtype=np.float32)
    for row in range(2):
        other = 1 - row
        relative = states[other, :2] - states[row, :2]
        target_index = 0
        while target_index < routes.shape[1] - 1 and np.linalg.norm(routes[row, target_index] - states[row, :2]) <= .45:
            target_index += 1
        expected[row] = controller.get_action(
            (states[row, :2], wall_map.pixels(states[row, :2]), states[row, 2:4], routes[row, target_index]),
            (ids[[other]], np.linalg.norm(relative)[None], relative[None],
             (states[other, 2:4] - states[row, 2:4])[None]))[0]
    check("same_snapshot_neighbor_minus_ego_and_synchronous_first_commit",
          np.array_equal(pair.velocity_xy[:, 1], expected)
          and np.array_equal(pair.xy[:, 1], states[:, :2] + expected * .1))
    reverse = generator.generate(states[::-1], routes[::-1], scene_ids=groups[::-1], actor_ids=ids[::-1])
    check("actor_row_permutation_invariant_synchronous_prediction",
          np.array_equal(pair.xy, reverse.xy[::-1]) and np.array_equal(pair.velocity_xy, reverse.velocity_xy[::-1]))
    separate = generator.generate(states, routes, scene_ids=np.array([0, 1]), actor_ids=ids)
    singles = [generator.generate(states[row:row+1], routes[row:row+1], scene_ids=np.array([row]),
                                  actor_ids=ids[row:row+1]) for row in range(2)]
    check("exclude_self_and_cross_scene_but_same_scene_peer_is_active",
          np.array_equal(separate.xy, np.concatenate([plan.xy for plan in singles]))
          and np.array_equal(separate.velocity_xy, np.concatenate([plan.velocity_xy for plan in singles]))
          and not np.array_equal(pair.velocity_xy[:, 1], separate.velocity_xy[:, 1]))
    check("actual_states_routes_scene_ids_actor_ids_unchanged",
          all(np.array_equal(array, original) for array, original in zip((states, routes, groups, ids), before))
          and pair.private_waypoint_indices.max() > 0)

    open_map = WorldEDTMap(np.ones((200, 200), dtype=bool), origin_xy_m=(-10., -10.), resolution_m=.1)
    close_generator = SFMTargetGenerator({0: open_map}, SFMConfig())
    close_state = np.array([[0., 0., .2, 0., 0.]], dtype=np.float32)
    close = close_generator.generate(close_state, close_state[:, None, :2], scene_ids=np.array([0]))
    check("near_goal_dt_point1_halves_velocity_without_point5_reverse_amplification",
          np.isclose(close.velocity_xy[0, 1, 0], .1) and np.all(close.velocity_xy[0, :, 0] >= 0)
          and close.diagnostics["prediction_dt_s"] == .1 and close.diagnostics["prediction_steps"] == 50
          and close.diagnostics["configuration"]["ttc_prediction_horizon_s"] == 2.)

    small_map = WorldEDTMap(np.ones((20, 20), dtype=bool), origin_xy_m=(0., 0.), resolution_m=.1)
    small_generator = SFMTargetGenerator({0: small_map})
    # Adversarial initial overspeed forces the exit branch despite EDT repulsion;
    # this is a contract fixture, not a plausible physical pedestrian velocity.
    exiting_state = np.array([[1.98, 1., 100., 0., 0.]], dtype=np.float32)
    exiting = small_generator.generate(exiting_state, np.array([[[3., 1.]]], dtype=np.float32), scene_ids=np.array([0]))
    events = exiting.diagnostics["domain_exit_events"]
    check("domain_exit_stop_retains_attempted_XY_without_edge_clamping",
          len(events) == 1 and events[0]["kind"] == "predicted_domain_exit" and events[0]["step"] == 1
          and events[0]["attempted_xy_m"][0] >= 2.
          and np.array_equal(exiting.xy, np.repeat(exiting_state[:, None, :2], 51, axis=1))
          and np.all(exiting.velocity_xy[:, 1:] == 0) and exiting.velocity_xy[0, 0, 0] == 100.)
    outside_state = np.array([[-.1, 1., 0., 0., 0.]], dtype=np.float32)
    outside = small_generator.generate(outside_state, np.array([[[1., 1.]]], dtype=np.float32), scene_ids=np.array([0]))
    check("true_initial_outside_preserved_no_negative_index_wrap",
          outside.diagnostics["domain_exit_events"][0]["kind"] == "actual_initial_outside"
          and np.array_equal(outside.xy, np.repeat(outside_state[:, None, :2], 51, axis=1))
          and outside.diagnostics["minimum_raw_EDT_m"] == [None])

    unknown_free = np.ones((200, 200), dtype=bool)
    unknown_free[99:102, 99:102] = False
    unknown_map = WorldEDTMap(unknown_free, origin_xy_m=(-10., -10.), resolution_m=.1)
    unknown_state = np.array([[.05, -.05, 0., 0., 0.]], dtype=np.float32)
    unknown = SFMTargetGenerator({0: unknown_map}).generate(
        unknown_state, unknown_state[:, None, :2], scene_ids=np.array([0]))
    check("unknown_cells_remain_blocked_and_negative_prediction_is_retained",
          unknown_map.edt_m[100, 100] == 0 and unknown.xy.shape == (1, 51, 2)
          and unknown.diagnostics["blocked_or_unknown_knot_counts"] == [51]
          and unknown.diagnostics["minimum_raw_EDT_m"] == [0.]
          and unknown.diagnostics["geometry_guarantee"] is False)
    try:
        WorldEDTMap(free, origin_xy_m=(-3., -2.), resolution_m=.1, origin_yaw_rad=.1)
    except ValueError:
        check("nonzero_map_yaw_explicitly_rejected", True)
    else:
        check("nonzero_map_yaw_explicitly_rejected", False)

    forbidden_imports = sorted(name for name in sys.modules if name == "torch"
                               or name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"})
    passed = all(item["passed"] for item in checks) and not forbidden_imports
    result = {"status": "passed" if passed else "failed", "CPU_only": True,
              "forbidden_runtime_modules_loaded": forbidden_imports,
              "python_executable": sys.executable, "python_version": sys.version,
              "numpy_version": np.__version__, "author_exact_SFM_baseline_claim": False,
              "fixture": record(__file__), "adapter": record(HERE / "sfm_target_generator.py"),
              "checks": checks, "check_count": len(checks),
              "near_goal_first_velocity_mps": close.velocity_xy[0, 1].tolist(),
              "wall_static_force_above_below": [above_force.tolist(), below_force.tolist()],
              "domain_exit_events": events, "unknown_diagnostics": unknown.diagnostics}
    with (HERE / "actual_CPU_fixture.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [item["name"] for item in checks if not item["passed"]]}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
