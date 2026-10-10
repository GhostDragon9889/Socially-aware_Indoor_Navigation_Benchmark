"""Synthetic interface tests for goal/endpoint/contact-proxy distinctions."""

import json

import numpy as np
import pytest

from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES
from CrowdSim.humanoid.evaluation import rollout_metrics
from scripts.evaluate_paper_pedestrians import geometry_proxy_metrics, read_capture, summarize_goal_metrics


@pytest.fixture
def capture(tmp_path):
    root = np.zeros((4, 2, 3))
    root[..., 2] = 1
    root[:, 0, 0] = [0, 1, 99, 100]
    root[:, 1, 0] = [0, .25, .5, .75]
    body = np.repeat(root[..., None, :], 24, axis=-2)
    body[..., PACER_BODY_NAMES.index("Head"), 2] += .8
    body[..., PACER_BODY_NAMES.index("L_Ankle"), 2] -= .85
    body[..., PACER_BODY_NAMES.index("R_Ankle"), 2] -= .85
    rotation = np.zeros((4, 2, 4))
    rotation[..., 3] = 1
    ids = np.zeros((4, 2), dtype=np.int64)
    ids[2:, 0] = 1
    reset = np.zeros((4, 2), dtype=bool)
    reset[0] = True
    reset[2, 0] = True
    goal = np.zeros((4, 2, 2))
    goal[:, 0, 0] = [10, 10, 109, 109]
    goal[:, 1, 0] = 5
    arrays = dict(root_pos=root, body_pos=body, root_rot=rotation, dof_pos=np.zeros((4, 2, 69)),
                  target_xy=root[..., :2] + [.1, 0.], episode_ids=ids, reset_mask=reset,
                  global_goal_xy=goal, plan_endpoint_goal_error=np.full((4, 2), 2.))
    report = {"status": "passed", "method": "SYNTHETIC UNIT INTERFACE, no physics run",
              "state_capture": {"body_names": list(PACER_BODY_NAMES), "frequency_hz": 30,
                                "coordinates": "world meters, quaternions XYZW, common SMPL names",
                                "reset_mask": "first post-step frame of each new episode"},
              "route_vertices_reached": [2, 1], "global_path_failures": 3,
              "physics": {"isolated_environments": False}}
    state_path, result_path = tmp_path / "physical_states.npz", tmp_path / "result.json"
    np.savez_compressed(state_path, **arrays)
    result_path.write_text(json.dumps(report))
    return arrays, report, state_path, result_path


def test_goal_distance_and_plan_residual_are_not_local_tracking_or_arrival(capture):
    arrays, report, state_path, result_path = capture
    loaded, runner, _ = read_capture(state_path, result_path)
    assert {"global_goal_xy", "plan_endpoint_goal_error"}.issubset(loaded)
    goals = summarize_goal_metrics(loaded, runner)
    physical = rollout_metrics(arrays["root_pos"], arrays["root_rot"], target_xy=arrays["target_xy"],
                               episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"])
    assert physical["plan_tracking"]["mean_xy_error_m"] == pytest.approx(.1)
    assert goals["distance_to_recorded_goal_xy"]["mean_m"] == pytest.approx(7.0625)
    assert goals["recorded_plan_endpoint_goal_residual"]["mean_m"] == 2
    assert goals["route_vertices_reached"] == [2, 1]
    assert goals["route_vertices_reached_total"] == 3
    assert goals["global_path_failures"] == 3
    assert goals["global_arrival_rate"] is None
    # The 98m reset and 99m goal jump never become movement progress.
    assert goals["unchanged_goal_distance_progress"]["comparable_same_goal_episode_transitions"] == 5
    assert goals["unchanged_goal_distance_progress"]["distance_reduction_sum_m"] == pytest.approx(2.75)


def test_optional_missing_goal_fields_remain_unknown(capture):
    arrays, report, state_path, result_path = capture
    arrays = {key: value for key, value in arrays.items() if key not in {"global_goal_xy", "plan_endpoint_goal_error"}}
    report = {key: value for key, value in report.items() if key not in {"route_vertices_reached", "global_path_failures"}}
    np.savez_compressed(state_path, **arrays)
    result_path.write_text(json.dumps(report))
    loaded, runner, _ = read_capture(state_path, result_path)
    goals = summarize_goal_metrics(loaded, runner)
    assert goals["distance_to_recorded_goal_xy"] is None
    assert goals["recorded_plan_endpoint_goal_residual"] is None
    assert goals["route_vertices_reached"] is None and goals["global_path_failures"] is None


@pytest.mark.parametrize("problem", ("goal_shape", "endpoint_shape", "negative_endpoint"))
def test_wrong_optional_goal_field_shapes_or_negative_residual_fail(capture, problem):
    arrays, _, state_path, result_path = capture
    if problem == "goal_shape":
        arrays["global_goal_xy"] = arrays["global_goal_xy"][..., :1]
    elif problem == "endpoint_shape":
        arrays["plan_endpoint_goal_error"] = arrays["plan_endpoint_goal_error"][..., None]
    else:
        arrays["plan_endpoint_goal_error"][0, 0] = -1
    np.savez_compressed(state_path, **arrays)
    with pytest.raises(ValueError):
        read_capture(state_path, result_path)


def test_explicit_scene_proxies_remain_separate_from_unknown_actual_contacts(capture):
    arrays, report, _, _ = capture
    # One root inside the cuboid, a different root outside the declared domain.
    arrays["root_pos"][3, 1, 0] = 120
    report["evaluation_scene"] = {"schema": "navisaaclab.reconstruction_scene.v1", "name": "synthetic_test_scene",
        "author_benchmark_scene": False, "bounds_xy": [-1, -1, 110, 5], "origins_xy": [[0, 0], [0, .5]],
        "routes_xy": [[[3, 3], [4, 3]], [[4, 3], [3, 3]]],
        "cuboids": [{"center": [1, 0, 1], "size": [.2, .4, 2]}]}
    proxy = geometry_proxy_metrics(arrays, report, agent_radius_m=.3)
    map_proxy = proxy["map_root_point_geometry_proxy"]
    assert map_proxy["inside_cuboid_footprint_frame_agents"] == 1
    assert map_proxy["outside_domain_frame_agents"] == 1
    assert map_proxy["observed_flagged_frames"] == 2
    assert proxy["agent_xy_disc_geometry_proxy"]["observed_flagged_frames"] == 2
    assert proxy["actual_pair_contacts_available"] is False
    actual = rollout_metrics(arrays["root_pos"], arrays["root_rot"], episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"])
    assert actual["map_collisions"] is None and actual["agent_collisions"] is None
    assert "lookahead" in summarize_goal_metrics(arrays, report)["goal_semantics"]


def test_isolated_worlds_are_not_assumed_to_share_agent_collision_space(capture):
    arrays, report, _, _ = capture
    report["physics"]["isolated_environments"] = True
    assert geometry_proxy_metrics(arrays, report)["agent_xy_disc_geometry_proxy"] is None
    report.pop("physics")
    assert geometry_proxy_metrics(arrays, report)["agent_xy_disc_geometry_proxy"] is None


def test_route_counter_corruption_and_bad_disc_radius_fail(capture):
    arrays, report, _, _ = capture
    report["route_vertices_reached"] = [-1, 0]
    with pytest.raises(ValueError, match="route_vertices"):
        summarize_goal_metrics(arrays, report)
    with pytest.raises(ValueError, match="proxy-agent-radius"):
        geometry_proxy_metrics(arrays, report, agent_radius_m=0)
