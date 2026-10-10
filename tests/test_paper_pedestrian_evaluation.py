"""Independent invariance, statistical, train/held-out and reset-boundary gates."""

import json
import math

import pytest
import torch

from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES
from CrowdSim.humanoid.evaluation import (
    KEYPOINT_NAMES, PoseClusterModel, fit_pose_clusters, gaussian_mmd,
    occupancy_collision_mask, planar_agent_overlap, pose_embedding, rollout_metrics,
)


def _standing_pose():
    positions = torch.zeros(3, 2, 24, 3, dtype=torch.float64)
    positions[..., 2] = .9
    for name, xyz in {
        "Head": (0., 0., 1.7), "Neck": (0., 0., 1.5),
        "L_Ankle": (0., .12, .05), "R_Ankle": (0., -.12, .05),
        "L_Hip": (0., .12, .85), "R_Hip": (0., -.12, .85),
        "L_Knee": (.05, .12, .45), "R_Knee": (.05, -.12, .45),
        "L_Shoulder": (0., .25, 1.4), "R_Shoulder": (0., -.25, 1.4),
        "L_Elbow": (.08, .35, 1.1), "R_Elbow": (.08, -.35, 1.1),
        "L_Wrist": (.12, .4, .8), "R_Wrist": (.12, -.4, .8),
    }.items():
        positions[..., PACER_BODY_NAMES.index(name), :] = torch.tensor(xyz)
    # Genuine time/batch variation ensures tests are not one symmetric static point.
    positions[1, :, PACER_BODY_NAMES.index("L_Knee"), 0] += .08
    positions[:, 1, PACER_BODY_NAMES.index("R_Wrist"), 1] -= .03
    quaternion = torch.zeros(3, 2, 4, dtype=torch.float64)
    quaternion[..., 3] = 1
    return positions, quaternion


def test_pose_embedding_translation_global_yaw_and_uniform_scale_invariance():
    positions, quaternion = _standing_pose()
    expected = pose_embedding(positions, quaternion)
    assert expected.shape == (3, 2, 45)
    assert len(KEYPOINT_NAMES) == 15
    assert torch.equal(expected[..., :3], torch.zeros_like(expected[..., :3]))  # Pelvis centers exactly.
    yaw = torch.tensor([[.4, -2.], [1.7, .1], [-.8, 2.4]], dtype=torch.float64)
    rotation = torch.stack((
        torch.stack((yaw.cos(), -yaw.sin(), torch.zeros_like(yaw)), -1),
        torch.stack((yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), -1),
        torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)), -1),
    ), -2)
    transformed = torch.einsum("tbij,tbnj->tbni", rotation, positions)
    scale = torch.tensor([[.4, 2.1], [3.7, 1.2], [.8, 1.6]], dtype=torch.float64)
    translation = torch.tensor([3., -8., 2.], dtype=torch.float64)
    transformed = transformed * scale[..., None, None] + translation
    yaw_quaternion = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin(), (yaw / 2).cos()), -1)
    actual = pose_embedding(transformed, yaw_quaternion)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)


def test_pose_embedding_explicit_height_and_named_body_permutation():
    positions, quaternion = _standing_pose()
    expected = pose_embedding(positions, quaternion, height_m=1.8)
    permutation = torch.randperm(24, generator=torch.Generator().manual_seed(19))
    names = tuple(PACER_BODY_NAMES[index] for index in permutation)
    actual = pose_embedding(positions[..., permutation, :], quaternion, body_names=names, height_m=1.8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    scaled = pose_embedding(positions * 3, quaternion, height_m=5.4)
    torch.testing.assert_close(scaled, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("problem", ("quaternion", "nonfinite", "height", "names", "keypoints"))
def test_ambiguous_or_nonphysical_embedding_inputs_fail(problem):
    positions, quaternion = _standing_pose()
    kwargs = {}
    if problem == "quaternion":
        quaternion[..., 3] = 2
    elif problem == "nonfinite":
        positions[0, 0, 0, 0] = torch.nan
    elif problem == "height":
        kwargs["height_m"] = 0
    elif problem == "names":
        kwargs["body_names"] = ("Pelvis",) * 24
    else:
        kwargs["keypoint_names"] = KEYPOINT_NAMES[:-1]
    with pytest.raises(ValueError):
        pose_embedding(positions, quaternion, **kwargs)


def _cluster_features():
    generator = torch.Generator().manual_seed(73)
    centers = torch.zeros(4, 45, dtype=torch.float64)
    centers[:, :2] = torch.tensor([[0., 0.], [4., 0.], [0., 4.], [4., 4.]])
    return centers.repeat_interleave(20, 0) + .015 * torch.randn(80, 45, generator=generator, dtype=torch.float64)


def test_fixed_train_clusters_score_heldout_without_refitting_and_serialize(tmp_path):
    features = _cluster_features()
    first = PoseClusterModel.fit(features, k=4, split="train", source_id="declared-test-training-corpus", seed=31)
    second = PoseClusterModel.fit(features, k=4, split="train", source_id="declared-test-training-corpus", seed=31)
    assert torch.equal(first.centers, second.centers)
    before = first.centers.clone()
    balanced = first.score(first.centers.repeat_interleave(5, 0), split="test")
    assert balanced["coverage"] == 1
    assert balanced["normalized_pose_entropy"] == pytest.approx(1, abs=1e-14)
    single = first.score(first.centers[:1].repeat(12, 1))
    assert single["coverage"] == .25 and single["normalized_pose_entropy"] == 0
    assert single["mean_assigned_center_distance"] < 1e-6
    assert torch.equal(first.centers, before), "Generated and held-out scoring must never alter training centers"
    path = tmp_path / "clusters.json"
    first.save(path)
    loaded = PoseClusterModel.load(path)
    assert torch.equal(loaded.centers, first.centers)
    assert loaded.score(features, split="validation") == first.score(features, split="validation")
    document = json.loads(path.read_text())
    document["training_split"] = "test"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="train"):
        PoseClusterModel.load(path)


@pytest.mark.parametrize("split", ("validation", "test", "generated", "mixed", ""))
def test_heldout_and_generated_cluster_fit_is_forbidden(split):
    with pytest.raises(ValueError, match="held-out/generated"):
        PoseClusterModel.fit(_cluster_features(), k=4, split=split, source_id="held-out")


def test_k50_and_k100_models_are_both_fixed_to_training_reference():
    features = torch.randn(250, 45, generator=torch.Generator().manual_seed(122), dtype=torch.float64)
    models = fit_pose_clusters(features, split="train", source_id="synthetic-CPU-regression-only", max_iterations=8)
    assert set(models) == {50, 100}
    for k, model in models.items():
        assert model.centers.shape == (k, 45)
        assert model.training_sample_count == 250
        assert model.score(features, split="train")["coverage"] == 1
        assert model.score(features[:10], split="test")["covered_clusters"] <= 10


def test_distinct_reference_poses_and_embedding_width_are_required():
    with pytest.raises(ValueError, match="distinct"):
        PoseClusterModel.fit(torch.zeros(50, 45), k=50, split="train", source_id="empty poses")
    with pytest.raises(ValueError, match="width"):
        PoseClusterModel.fit(torch.randn(10, 2), k=2, split="train", source_id="wrong embedding")


def test_true_gaussian_mmd_identical_is_zero_and_known_two_point_value_matches_kernel():
    poses = _cluster_features()[:23]
    identical = gaussian_mmd(poses, poses.clone(), bandwidth=.7, chunk_size=7)
    assert identical["mmd_squared"] == pytest.approx(0, abs=1e-13)
    assert identical["mmd"] == pytest.approx(0, abs=1e-6)
    zero, one = torch.zeros(1, 45), torch.ones(1, 45)
    result = gaussian_mmd(zero, one, bandwidth=2., chunk_size=1)
    expected = 2 * (1 - math.exp(-45 / 8))
    assert result["mmd_squared"] == pytest.approx(expected, abs=1e-14)
    assert result["mmd"] == pytest.approx(math.sqrt(expected), abs=1e-14)
    assert result["bandwidth"] == 2
    assert "biased" in result["estimator"]
    chunked = gaussian_mmd(poses, poses.flip(0) + .1, bandwidth=.7, chunk_size=5)
    dense = gaussian_mmd(poses, poses.flip(0) + .1, bandwidth=.7, chunk_size=64)
    assert chunked["mmd_squared"] == pytest.approx(dense["mmd_squared"], abs=1e-13)
    assert chunked["mmd_squared"] > 0


@pytest.mark.parametrize("bandwidth", (0., -1., float("nan"), float("inf")))
def test_invalid_or_adapted_bandwidth_is_not_silently_used(bandwidth):
    with pytest.raises(ValueError, match="fixed Gaussian"):
        gaussian_mmd(torch.zeros(1, 45), torch.zeros(1, 45), bandwidth=bandwidth)


def _rollout():
    root = torch.zeros(5, 2, 3, dtype=torch.float64)
    root[..., 2] = 1
    root[:, 0, 0] = torch.tensor([0., 1., 99., 100., 101.])
    quaternion = torch.zeros(5, 2, 4, dtype=torch.float64)
    quaternion[..., 3] = 1
    ids = torch.zeros(5, 2, dtype=torch.long)
    ids[2:, 0] = 1
    return root, quaternion, ids


@pytest.mark.parametrize("boundary_source", ("ids", "before", "after", "both"))
def test_reset_teleport_is_excluded_from_path_distance(boundary_source):
    root, quaternion, ids = _rollout()
    kwargs = {}
    resets = torch.zeros(5, 2, dtype=torch.bool)
    if boundary_source in {"ids", "both"}:
        kwargs["episode_ids"] = ids
    if boundary_source in {"before", "both"}:
        resets[2, 0] = True
        kwargs.update(reset_mask=resets, reset_mask_marks="before_frame")
    if boundary_source == "after":
        resets[1, 0] = True
        kwargs.update(reset_mask=resets, reset_mask_marks="after_frame")
    result = rollout_metrics(root, quaternion, **kwargs)
    assert result["walked_distance_xy_m_per_agent"] == [3., 0.]
    assert result["valid_transition_count_per_agent"] == [3, 4]
    assert result["excluded_episode_boundary_transitions"] == 1
    if boundary_source == "both":
        assert result["boundary_source_disagreement_count"] == 0
    assert result["map_collisions"] is None and result["agent_collisions"] is None
    assert result["success_rate"] is None


def test_episode_falls_collision_events_and_time_aligned_tracking_are_distinct():
    root, quaternion, ids = _rollout()
    root[1:4, 0, 2] = .3
    collisions = torch.zeros(5, 2, dtype=torch.bool)
    collisions[1:4, 0] = True
    target = root[..., :2] + torch.tensor([3., 4.])
    result = rollout_metrics(root, quaternion, episode_ids=ids, target_xy=target,
                             map_collision_mask=collisions, map_collision_definition="known synthetic contact fixture")
    assert result["falls"]["observed_event_starts"] == 2
    assert result["falls"]["observed_flagged_frames"] == 3
    assert result["falls"]["observed_episodes"] == 3  # Agent0 has two, Agent1 has one.
    assert result["falls"]["affected_observed_episodes"] == 2
    assert result["falls"]["affected_episode_fraction"] == pytest.approx(2 / 3)
    assert result["map_collisions"]["observed_event_starts"] == 2
    assert result["map_collisions"]["definition"] == "known synthetic contact fixture"
    assert result["agent_collisions"] is None
    assert result["plan_tracking"]["mean_xy_error_m"] == 5
    assert result["plan_tracking"]["rmse_xy_m"] == 5
    assert result["plan_tracking"]["maximum_xy_error_m"] == 5


def test_invalid_frames_do_not_connect_a_missing_interval():
    root, quaternion, ids = _rollout()
    valid = torch.ones(5, 2, dtype=torch.bool)
    valid[2, 0] = False
    result = rollout_metrics(root, quaternion, episode_ids=ids, valid_mask=valid)
    assert result["walked_distance_xy_m_per_agent"] == [2., 0.]
    assert result["valid_transition_count_per_agent"] == [2, 4]


def test_agent_disc_proxy_does_not_collide_with_self_and_map_axis_is_explicit():
    root = torch.tensor([[[0., 0., 1.], [.4, 0., 1.], [3., 0., 1.]],
                         [[0., 0., 1.], [1., 0., 1.], [3., 0., 1.]]])
    assert torch.equal(planar_agent_overlap(root, radius_m=.3), torch.tensor([[True, True, False], [False, False, False]]))
    assert not planar_agent_overlap(root[:, :1]).any()
    grid = torch.tensor([[True, False, False], [False, False, True]])
    points = torch.tensor([[[-.75, -1.75, 1.]], [[.25, -1.25, 1.]], [[-1.01, -2., 1.]]])
    actual = occupancy_collision_mask(points, grid, origin_xy=(-1., -2.), resolution_m=.5)
    assert actual.tolist() == [[True], [True], [True]]
    without_outside = occupancy_collision_mask(points, grid, origin_xy=(-1., -2.), resolution_m=.5, outside_is_collision=False)
    assert without_outside.tolist() == [[True], [True], [False]]
    with pytest.raises(ValueError, match="boolean"):
        occupancy_collision_mask(points, grid.float(), origin_xy=(-1., -2.), resolution_m=.5)


def test_root_up_axis_can_trigger_fall_without_low_pelvis():
    root, quaternion, _ = _rollout()
    quaternion[3, 1] = torch.tensor([1., 0., 0., 0.])
    result = rollout_metrics(root, quaternion)
    assert result["falls"]["observed_flagged_frames"] == 1
    assert result["falls"]["observed_event_starts"] == 1


def test_unknown_reset_or_collision_contracts_fail_instead_of_inventing_rates():
    root, quaternion, _ = _rollout()
    with pytest.raises(ValueError, match="before_frame or after_frame"):
        rollout_metrics(root, quaternion, reset_mask_marks="guess")
    with pytest.raises(ValueError, match="boolean"):
        rollout_metrics(root, quaternion, map_collision_mask=torch.zeros(5, 2))
    with pytest.raises(ValueError, match="time-aligned"):
        rollout_metrics(root, quaternion, target_xy=torch.zeros(4, 2, 2))


def test_inconsistent_reset_metadata_is_visible_in_report():
    root, quaternion, ids = _rollout()
    reset = torch.zeros(5, 2, dtype=torch.bool)
    reset[1, 0] = True  # Wrongly declared before_frame instead of after_frame.
    result = rollout_metrics(root, quaternion, episode_ids=ids, reset_mask=reset)
    assert result["boundary_source_disagreement_count"] == 2
    assert result["excluded_episode_boundary_transitions"] == 2
    assert result["walked_distance_xy_m_per_agent"] == [2., 0.]


def _normal_contact_contract():
    return {"source": "PhysX exact-path rigid_contact_view.get_contact_force_matrix",
            "units": "N", "components": "normal forces only", "threshold_n": 1.,
            "substep_reduction": "maximum body-pair magnitude over every physics substep; strict > threshold",
            "excluded": ["floor", "self articulation"], "author_metric_equivalence": False,
            "physics_substeps_per_control": 4, "physics_dt_s": 1/120, "control_dt_s": 1/30}


def _contact_recording():
    root = torch.zeros(5, 3, 3)
    root[..., 2] = 1
    q = torch.zeros(5, 3, 4)
    q[..., 3] = 1
    ids = torch.zeros(5, 3, dtype=torch.long)
    ids[2:, 0] = 1
    static_force = torch.zeros(5, 3)
    static_force[1:4, 0] = 2
    static_force[4, 2] = 3
    peer_force = torch.zeros(5, 3, 3)
    peer_force[1:4, 0, 1] = peer_force[1:4, 1, 0] = 5
    contacts = {"static_contact": static_force > 1, "peer_contact": peer_force > 1,
                "static_contact_max_force_n": static_force, "peer_contact_max_force_n": peer_force,
                "contact_contract": _normal_contact_contract()}
    return root, q, ids, contacts


def test_actual_contact_actor_frames_pairs_and_reset_incidences_are_distinct():
    root, q, ids, contacts = _contact_recording()
    result = rollout_metrics(root, q, episode_ids=ids, **contacts)
    actual = result["actual_contacts"]
    static = actual["static"]["actor_frame_statistics"]
    peer = actual["peer"]["actor_frame_statistics"]
    pair = actual["peer"]["unordered_pair_statistics"]
    assert static["observed_flagged_frames"] == 4 and static["observed_event_starts"] == 3
    assert static["observed_episodes"] == 4 and static["affected_observed_episodes"] == 3
    assert peer["observed_flagged_frames"] == 6 and peer["observed_event_starts"] == 3
    assert pair["observed_contact_pair_frames"] == 3 and pair["contiguous_pair_contact_incidences"] == 2
    assert pair["observed_pair_frames"] == 15 and pair["contact_pair_frame_fraction"] == pytest.approx(.2)
    assert pair["observed_pair_episode_segments"] == 5 and pair["affected_pair_episode_segments"] == 2
    assert actual["peer"]["unordered_actor_pairs"] == [[0, 1], [0, 2], [1, 2]]
    assert actual["peer"]["contact_pair_frames_per_pair"] == [3, 0, 0]
    assert actual["peer"]["maximum_body_pair_force_n"] == 5.
    assert actual["static"]["maximum_body_pair_force_n"] == 3.
    assert actual["peer"]["directed_boolean_disagreement_entries"] == 0
    assert result["map_collisions"]["source"] == "recorded PhysX pairwise normal contact"
    assert result["agent_collisions"]["observed_flagged_frames"] == 6
    assert actual["author_metric_equivalence"] is False


def test_directed_views_union_does_not_double_count_unordered_pair():
    root, q, ids, contacts = _contact_recording()
    contacts["peer_contact_max_force_n"][:, 1, 0] = 0
    contacts["peer_contact"] = contacts["peer_contact_max_force_n"] > 1
    result = rollout_metrics(root, q, episode_ids=ids, **contacts)["actual_contacts"]["peer"]
    assert result["unordered_pair_statistics"]["observed_contact_pair_frames"] == 3
    assert result["actor_frame_statistics"]["observed_flagged_frames"] == 6
    assert result["directed_boolean_disagreement_entries"] == 6


def test_contact_invalid_gap_and_either_actor_reset_break_pair_incidence():
    root, q, ids, contacts = _contact_recording()
    ids[:] = 0
    valid = torch.ones(5, 3, dtype=torch.bool)
    valid[2, 1] = False
    result = rollout_metrics(root, q, episode_ids=ids, valid_mask=valid, **contacts)["actual_contacts"]["peer"]
    pair = result["unordered_pair_statistics"]
    assert pair["observed_contact_pair_frames"] == 2
    assert pair["contiguous_pair_contact_incidences"] == 2
    assert pair["observed_pair_frames"] == 13
    # Missing actor1 frame does not fabricate contact for actor0 on that pair.
    assert result["actor_frame_statistics"]["observed_flagged_frames"] == 4


def test_strict_normal_force_threshold_and_one_actor_empty_pair_denominator():
    root, q, _, _ = _contact_recording()
    force = torch.tensor([[0.], [1.], [1.001], [.5], [2.]])
    result = rollout_metrics(root[:, :1], q[:, :1],
        static_contact=force > 1, static_contact_max_force_n=force,
        peer_contact=torch.zeros(5, 1, 1, dtype=torch.bool), peer_contact_max_force_n=torch.zeros(5, 1, 1),
        contact_contract=_normal_contact_contract())["actual_contacts"]
    assert result["static"]["actor_frame_statistics"]["observed_flagged_frames"] == 2
    assert result["peer"]["unordered_pair_statistics"]["observed_pair_frames"] == 0
    assert result["peer"]["unordered_pair_statistics"]["contact_pair_frame_fraction"] is None
    assert result["peer"]["maximum_body_pair_force_n"] is None


def test_missing_actual_contacts_remain_unknown_and_explicit_old_proxy_is_preserved():
    root, q, ids, contacts = _contact_recording()
    missing = rollout_metrics(root, q, episode_ids=ids)
    assert missing["actual_contacts"] is None
    assert missing["map_collisions"] is missing["agent_collisions"] is None
    proxy = torch.zeros(5, 3, dtype=torch.bool)
    result = rollout_metrics(root, q, episode_ids=ids, map_collision_mask=proxy,
        map_collision_definition="explicit synthetic root-point proxy", **contacts)
    assert result["map_collisions"]["observed_flagged_frames"] == 0
    assert result["map_collisions"]["definition"] == "explicit synthetic root-point proxy"
    assert result["actual_contacts"]["static"]["actor_frame_statistics"]["observed_flagged_frames"] == 4


@pytest.mark.parametrize("problem", ["missing_contract", "units", "components", "source", "zero_threshold",
    "nan_threshold", "boolean_threshold", "missing_self_exclusion", "wrong_substeps", "wrong_physics_dt", "wrong_control_dt", "author_claim",
    "mask_float", "mask_shape", "force_shape", "force_negative", "force_nan", "force_integer",
    "force_mask_disagreement", "peer_self_mask", "peer_self_force", "force_without_mask"])
def test_recorded_contact_contracts_fail_closed(problem):
    root, q, ids, contacts = _contact_recording()
    contract = contacts["contact_contract"]
    if problem == "missing_contract":
        contacts["contact_contract"] = None
    elif problem in {"units", "components", "source"}:
        contract[problem] = "unknown"
    elif problem in {"zero_threshold", "nan_threshold", "boolean_threshold"}:
        contract["threshold_n"] = {"zero_threshold": 0, "nan_threshold": float("nan"), "boolean_threshold": True}[problem]
    elif problem == "missing_self_exclusion":
        contract["excluded"] = ["floor"]
    elif problem == "wrong_substeps":
        contract["physics_substeps_per_control"] = 3
    elif problem == "wrong_physics_dt":
        contract["physics_dt_s"] = 1/60
    elif problem == "wrong_control_dt":
        contract["control_dt_s"] = 1/60
    elif problem == "author_claim":
        contract["author_metric_equivalence"] = True
    elif problem == "mask_float":
        contacts["static_contact"] = contacts["static_contact"].float()
    elif problem == "mask_shape":
        contacts["peer_contact"] = contacts["peer_contact"][:, :1]
    elif problem == "force_shape":
        contacts["peer_contact_max_force_n"] = contacts["peer_contact_max_force_n"][:, :1]
    elif problem == "force_negative":
        contacts["static_contact_max_force_n"][0, 0] = -1
    elif problem == "force_nan":
        contacts["static_contact_max_force_n"][0, 0] = float("nan")
    elif problem == "force_integer":
        contacts["static_contact_max_force_n"] = contacts["static_contact_max_force_n"].long()
    elif problem == "force_mask_disagreement":
        contacts["static_contact"][0, 0] = True
    elif problem == "peer_self_mask":
        contacts["peer_contact_max_force_n"][0, 0, 0] = 2
        contacts["peer_contact"][0, 0, 0] = True
    elif problem == "peer_self_force":
        contacts["peer_contact_max_force_n"][0, 0, 0] = .5
    else:
        contacts["static_contact"] = None
    with pytest.raises(ValueError):
        rollout_metrics(root, q, episode_ids=ids, **contacts)


def test_contact_npz_optional_contract_is_read_and_checked(tmp_path):
    import json
    import numpy as np
    from scripts.evaluate_paper_pedestrians import read_capture

    root, q, ids, contacts = _contact_recording()
    arrays = {"root_pos": root.numpy(), "root_rot": q.numpy(), "episode_ids": ids.numpy(),
              "reset_mask": np.zeros((5, 3), dtype=bool), "body_pos": root[:, :, None].expand(-1, -1, 24, -1).numpy(),
              "dof_pos": np.zeros((5, 3, 69)), "target_xy": root[..., :2].numpy()}
    arrays.update({key: value.numpy() for key, value in contacts.items() if key != "contact_contract"})
    states, result = tmp_path/"physical_states.npz", tmp_path/"result.json"
    report = {"status": "passed", "method": "SYNTHETIC contact interface fixture; no physics execution",
        "state_capture": {"body_names": list(PACER_BODY_NAMES), "frequency_hz": 30,
            "coordinates": "world meters; XYZW", "reset_mask": "first post-step frame of each new episode"},
        "physical_contact_measurement": contacts["contact_contract"]}
    np.savez_compressed(states, **arrays)
    result.write_text(json.dumps(report))
    loaded, actual_report, validation = read_capture(states, result)
    assert {"static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n"} <= loaded.keys()
    assert "pairwise_contact_fields" in validation
    assert actual_report["physical_contact_measurement"]["threshold_n"] == 1
    report.pop("physical_contact_measurement")
    result.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="contact contract"):
        read_capture(states, result)
    report["physical_contact_measurement"] = contacts["contact_contract"]
    result.write_text(json.dumps(report))
    arrays.pop("peer_contact_max_force_n")
    np.savez_compressed(states, **arrays)
    with pytest.raises(ValueError, match="all four"):
        read_capture(states, result)
