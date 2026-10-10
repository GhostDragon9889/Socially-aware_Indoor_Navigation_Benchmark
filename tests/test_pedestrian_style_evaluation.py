"""Motion diagnostics test physical invariances and known failure patterns on CPU."""
from dataclasses import asdict
import copy
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.style_evaluation import (BODY_NAMES, StyleSpec, compare_speed_conditioned,
    assert_parent_split_disjoint, gaussian_motion_mmd2, parent_source_identity, sequence_style_samples, wasserstein_1d)
from scripts.evaluate_pedestrian_style import assert_heldout_disjoint, reference_selection


def fixture(t=12):
    coordinates = {"Pelvis": (0, 0, 1), "L_Hip": (0, .1, .9), "L_Knee": (0, .1, .5),
        "L_Ankle": (0, .1, .08), "L_Toe": (.1, .1, .05), "R_Hip": (0, -.1, .9),
        "R_Knee": (0, -.1, .5), "R_Ankle": (0, -.1, .08), "R_Toe": (.1, -.1, .05),
        "Torso": (0, 0, 1.1), "Spine": (0, 0, 1.2), "Chest": (0, 0, 1.35),
        "Neck": (0, 0, 1.5), "Head": (0, 0, 1.6), "L_Thorax": (0, .15, 1.35),
        "L_Shoulder": (0, .2, 1.35), "L_Elbow": (0, .35, 1.1), "L_Wrist": (0, .4, .85),
        "L_Hand": (0, .45, .8), "R_Thorax": (0, -.15, 1.35), "R_Shoulder": (0, -.2, 1.35),
        "R_Elbow": (0, -.35, 1.1), "R_Wrist": (0, -.4, .85), "R_Hand": (0, -.45, .8)}
    pos = np.broadcast_to(np.array([coordinates[name] for name in BODY_NAMES]), (t, 1, 24, 3)).copy()
    q = np.broadcast_to([0., 0., 0., 1.], (t, 1, 4)).copy()
    episodes, resets = np.ones((t, 1), dtype=np.int64), np.zeros((t, 1), dtype=bool)
    resets[0] = True
    return pos, q, episodes, resets


def samples(pos, q, episodes, resets, **kwargs):
    return sequence_style_samples(pos, q, episode_ids=episodes, reset_mask=resets, **kwargs)


def test_heading_translation_and_constant_morphology_scale_do_not_change_local_articulation():
    pos, q, episodes, resets = fixture()
    pos[:, :, BODY_NAMES.index("L_Wrist"), 0] += np.linspace(0, .2, len(pos))[:, None]
    original = samples(pos, q, episodes, resets)
    angle = 1.2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformed = (2*pos)@rotation.T + [30, -20, 5]
    q[..., 2], q[..., 3] = np.sin(angle/2), np.cos(angle/2)
    modified = samples(transformed, q, episodes, resets, spec=StyleSpec(ground_height_m=5.))
    np.testing.assert_allclose(original["local_pose"], modified["local_pose"], atol=1e-12)
    np.testing.assert_allclose(original["local_pose_velocity"], modified["local_pose_velocity"], atol=1e-10)


def test_per_frame_global_yaw_removed_without_calling_rigid_turn_articulation_jitter():
    pos, q, episodes, resets = fixture()
    for i, angle in enumerate(np.linspace(0, .7, len(pos))):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        pos[i] = pos[i]@rotation.T
        q[i, :, 2], q[i, :, 3] = np.sin(angle/2), np.cos(angle/2)
    measured = samples(pos, q, episodes, resets)
    assert measured["metrics"]["local_pose_velocity_rms_chain_s"].max() < 1e-12


def test_sliding_rigid_body_detected_but_anchored_feet_have_zero_proxy_slide():
    pos, q, episodes, resets = fixture()
    pos[..., 0] += np.arange(len(pos))[:, None, None]/30
    sliding = samples(pos, q, episodes, resets)
    assert sliding["summary"]["physical_root_speed_m_s"]["mean"] == pytest.approx(1.)
    assert sliding["summary"]["toe_slide_near_ground_proxy"]["speed_m_s"]["mean"] == pytest.approx(1.)
    assert sliding["summary"]["toe_slide_near_ground_proxy"]["sliding_above_threshold_fraction"] == 1.
    for name in ("L_Toe", "R_Toe"):
        pos[:, :, BODY_NAMES.index(name), 0] = .1
    anchored = samples(pos, q, episodes, resets)
    assert anchored["summary"]["physical_root_speed_m_s"]["mean"] == pytest.approx(1.)
    assert anchored["summary"]["toe_slide_near_ground_proxy"]["speed_m_s"]["mean"] == 0.
    assert anchored["summary"]["measured_foot_contact_body_origin_speed_m_s"] is None


def test_hovering_has_no_support_and_no_fabricated_zero_sliding_score():
    pos, q, episodes, resets = fixture()
    pos[..., 2] += .5
    measured = samples(pos, q, episodes, resets)
    assert measured["summary"]["toe_slide_near_ground_proxy"]["stance_foot_intervals"] == 0
    assert measured["summary"]["toe_slide_near_ground_proxy"]["speed_m_s"]["mean"] is None
    assert measured["summary"]["metrics"]["toe_near_ground_proxy_supported"]["mean"] == 0


def test_reset_teleport_excludes_velocity_and_acceleration_not_just_path_distance():
    pos, q, episodes, resets = fixture()
    pos[6:, :, :, 0] += 1000
    episodes[6:] = 2
    resets[6] = True
    measured = samples(pos, q, episodes, resets)
    assert measured["summary"]["excluded_reset_transitions"] == 1
    assert measured["summary"]["same_episode_transition_count"] == 10
    assert measured["summary"]["valid_acceleration_triplets"] == 8
    assert measured["speed_m_s"].max() == 0
    assert measured["metrics"]["local_pose_acceleration_rms_chain_s2"].max() == 0
    resets[6] = False
    with pytest.raises(ValueError, match="reset masks"):
        samples(pos, q, episodes, resets)


def test_body_name_permutation_and_mass_mapping_preserve_metrics():
    pos, q, episodes, resets = fixture()
    masses = np.arange(1, 25)
    original = samples(pos, q, episodes, resets, body_masses_kg=masses)
    permutation = np.arange(23, -1, -1)
    changed = samples(pos[..., permutation, :], q, episodes, resets,
        body_names=tuple(BODY_NAMES[i] for i in permutation), body_masses_kg=masses[permutation])
    for field in original["metrics"]:
        np.testing.assert_allclose(original["metrics"][field], changed["metrics"][field])


def test_true_com_requires_captured_com_positions_and_does_not_substitute_joint_origins():
    pos, q, episodes, resets = fixture()
    origin_only = samples(pos, q, episodes, resets, body_masses_kg=np.ones(24))
    assert origin_only["summary"]["true_body_com_height_available"] is False
    assert origin_only["summary"]["mass_weighted_joint_origin_height_proxy_available"] is True
    actual = samples(pos, q, episodes, resets, body_masses_kg=np.ones(24), body_com_world=pos+[0, 0, .1])
    difference = actual["metrics"]["measured_body_com_weighted_height_m"]-actual["metrics"]["mass_weighted_joint_origin_height_proxy_m"]
    np.testing.assert_allclose(difference, .1)
    with pytest.raises(ValueError, match="matching body masses"):
        samples(pos, q, episodes, resets, body_com_world=pos)


@pytest.mark.parametrize("failure", ["nonfinite", "quaternion", "mask", "names", "boolean_spec", "mass"])
def test_invalid_measurements_fail_closed(failure):
    pos, q, episodes, resets = fixture()
    kwargs = {}
    if failure == "nonfinite":
        pos[1, 0, 3, 0] = np.nan
    elif failure == "quaternion":
        q *= 2
    elif failure == "mask":
        resets = resets.astype(int)
    elif failure == "names":
        kwargs["body_names"] = BODY_NAMES[:-1]+("wrong",)
    elif failure == "boolean_spec":
        kwargs["spec"] = StyleSpec(dt_s=True)
    elif failure == "mass":
        kwargs["body_masses_kg"] = np.zeros(24)
    with pytest.raises(ValueError):
        samples(pos, q, episodes, resets, **kwargs)


def test_wasserstein_exact_and_gaussian_motion_mmd_identical_zero():
    assert wasserstein_1d([0, 1], [1, 2]) == pytest.approx(1.)
    assert wasserstein_1d([0, 0, 0], [0]) == 0
    data = np.arange(60).reshape(10, 6)/20
    assert gaussian_motion_mmd2(data, data) == 0
    assert gaussian_motion_mmd2(data, data+5) > .1


def test_speed_mixture_not_confused_with_within_bin_articulation_distance():
    pos, q, episodes, resets = fixture(t=101)
    generated = samples(pos, q, episodes, resets)
    reference = copy.deepcopy(generated)
    generated["speed_m_s"][:90], generated["speed_m_s"][90:] = .25, 1.25
    reference["speed_m_s"][:10], reference["speed_m_s"][10:] = .25, 1.25
    generated["metrics"]["pelvis_height_m"][:90], generated["metrics"]["pelvis_height_m"][90:] = .8, 1.
    reference["metrics"]["pelvis_height_m"][:10], reference["metrics"]["pelvis_height_m"][10:] = .8, 1.
    compared = compare_speed_conditioned(generated, reference, minimum_bin_samples=5)
    metric = compared["aggregate_metrics"]["pelvis_height_m"]
    assert metric["unconditional_wasserstein_1d"] == pytest.approx(.16)
    assert metric["equal_supported_bin_mean_wasserstein_1d"] == 0
    assert metric["reference_sample_mass_covered"] == 1
    assert compared["equal_supported_bin_mean_pose_velocity_mmd2"] == 0


def test_missing_speed_support_not_zero_error_and_spec_drift_rejected():
    pos, q, episodes, resets = fixture()
    generated, reference = samples(pos, q, episodes, resets), samples(pos, q, episodes, resets)
    generated["speed_m_s"][:] = 2.5
    compared = compare_speed_conditioned(generated, reference, minimum_bin_samples=1)
    assert compared["aggregate_metrics"]["pelvis_height_m"]["equal_supported_bin_mean_wasserstein_1d"] is None
    assert compared["aggregate_metrics"]["pelvis_height_m"]["reference_sample_mass_covered"] == 0
    reference["spec"] = asdict(StyleSpec(dt_s=1/60))
    with pytest.raises(ValueError, match="exactly the same"):
        compare_speed_conditioned(generated, reference)


def test_reference_windows_are_heldout_within_clip_reproducible_and_leakage_rejected():
    document = {"splits": {"train": {"clips": [{"source_path": "train/a.npz"}]}, "validation": {
        "clips": [{"motion_id": 7, "source_path": "validation/walk.npz", "frames": 90, "dt": 1/30},
                  {"motion_id": 8, "source_path": "validation/short.npz", "frames": 32, "dt": 1/30}]}}}
    selected = reference_selection(document, windows=20)
    assert selected == reference_selection(document, windows=20)
    assert selected["excluded_short_clips"] == 1
    assert all(0 <= window["start_frame"] < window["stop_frame_exclusive"] <= 90 for window in selected["windows"])
    assert_heldout_disjoint(selected, document)
    with pytest.raises(ValueError, match="never train"):
        reference_selection(document, split="train")
    document["splits"]["train"]["clips"].append({"source_path": "validation/walk.npz"})
    with pytest.raises(ValueError, match="leaks"):
        assert_heldout_disjoint(selected, document)


def test_numpy_module_and_cli_import_do_not_import_torch_or_kit():
    code = "import sys; import CrowdSim.humanoid.style_evaluation; import scripts.evaluate_pedestrian_style; assert 'torch' not in sys.modules; assert 'isaacsim' not in sys.modules; assert 'omni' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_disjoint_segments_of_one_heldout_parent_cannot_leak_into_training():
    selection = {"split": "validation", "windows": [{"source_path": "/heldout/walk.motion#frames=0:250"}]}
    document = {"splits": {"train": {"clips": [{"source_path": "/heldout/walk.motion#frames=250:500",
        "parent_source_path": "/heldout/walk.motion"}]}}}
    with pytest.raises(ValueError, match="leaks.*parent"):
        assert_heldout_disjoint(selection, document)
    document["splits"]["train"]["clips"][0]["source_path"] = "/train/walk.motion#frames=250:500"
    document["splits"]["train"]["clips"][0]["parent_source_path"] = "/train/walk.motion"
    review = assert_heldout_disjoint(selection, document)
    assert review["original_source_intersection"] == 0


def test_manifest_rejects_unsampled_heldout_parent_and_allows_within_train_segments():
    document = {"splits": {"train": {"clips": [
        {"source_path": "/train/walk.motion#frames=0:250"},
        {"source_path": "/train/walk.motion#frames=250:500", "parent_source_path": "/train/walk.motion"}]},
        "validation": {"clips": [{"source_path": "/validation/walk.motion", "frames": 90,
                                  "motion_id": 0, "dt": 1/30}]},
        "test": {"clips": [{"source_path": "/test/walk.motion"}]}}}
    assert assert_parent_split_disjoint(document)["train"] == {"/train/walk.motion"}
    selected = reference_selection(document, windows=1)
    assert_heldout_disjoint(selected, document)
    document["splits"]["test"]["clips"].append({"source_path": "/train/walk.motion#frames=500:750"})
    with pytest.raises(ValueError, match="train and test"):
        reference_selection(document, windows=1)
    with pytest.raises(ValueError, match="train and test"):
        assert_heldout_disjoint(selected, document)


def test_parent_normalization_handles_windows_separators_and_dot_components():
    assert parent_source_identity({"source_path": r"C:\AMASS\CMU\45\45_01.motion#frames=0:250",
        "parent_source_path": "C:/AMASS/CMU/45/45_01.motion"}) == "C:/AMASS/CMU/45/45_01.motion"
    selected = {"split": "test", "windows": [{"source_path": "/bank/foo/../heldout.motion"}]}
    document = {"splits": {"train": {"clips": [{"source_path": "/bank/heldout.motion#frames=20:270"}]}}}
    with pytest.raises(ValueError, match="leaks"):
        assert_heldout_disjoint(selected, document)


def test_new_training_manifest_cannot_leak_unsampled_frozen_reference_parent():
    frozen = {"splits": {"train": {"clips": [{"source_path": "/original_train.motion"}]},
        "validation": {"clips": [{"source_path": "/validation/sampled.motion"},
                                  {"source_path": "/validation/unsampled.motion"}]},
        "test": {"clips": [{"source_path": "/test/unexposed.motion"}]}}}
    selection = {"split": "validation", "windows": [{"source_path": "/validation/sampled.motion"}]}
    training = {"splits": {"train": {"clips": [{"source_path": "/validation/unsampled.motion#frames=0:250"}]}}}
    with pytest.raises(ValueError, match="held-out validation parent"):
        assert_heldout_disjoint(selection, training, frozen)
    training["splits"]["train"]["clips"][0]["source_path"] = "/test/unexposed.motion#frames=0:250"
    with pytest.raises(ValueError, match="held-out test parent"):
        assert_heldout_disjoint(selection, training, frozen)


@pytest.mark.parametrize("clip", [
    {"source_path": "heldout.motion#frames=0:250", "parent_source_path": "train.motion"},
    {"source_path": "heldout.motion", "parent_source_path": "train.motion"},
    {"source_path": "heldout.motion", "parent_source_path": None},
    {"source_path": "heldout.motion#frames=0:250", "parent_source_path": "heldout.motion#frames=0:250"},
    {"source_path": "heldout.motion#frames=-1:250"},
    {"source_path": "heldout.motion#frames=250:250"},
    {"source_path": "heldout.motion#frames=251:250"},
    {"source_path": "heldout.motion#frames=0:250/../another.motion"},
    {"source_path": "heldout.motion#other=1"},
    {"source_path": "heldout.motion#frames=0:250#frames=250:500"},
    {"source_path": "heldout.motion#frames=0:250", "parent_control_frame_start": 1, "parent_control_frame_stop": 250},
    {"source_path": "heldout.motion#frames=0:250", "parent_control_frame_start": False, "parent_control_frame_stop": 250},
    {"source_path": "heldout.motion#frames=0:250", "parent_control_frame_start": 0},
    {"source_path": "heldout.motion#frames=0:250", "frames": 249},
    {"source_path": ""},
    {"source_path": True},
])
def test_spoofed_parent_and_malformed_segment_identity_fail_closed(clip):
    with pytest.raises(ValueError):
        parent_source_identity(clip)


def test_segmented_reference_carries_parent_without_changing_plain_legacy_selection():
    train = {"clips": [{"source_path": "train.motion"}]}
    legacy = {"splits": {"train": train, "validation": {"clips": [
        {"motion_id": 7, "source_path": "validation.motion", "frames": 90, "dt": 1/30}]}}}
    expected = {"split": "validation", "seed": 7,
        "windows": [{"motion_id": 7, "source_path": "validation.motion", "start_frame": 28, "stop_frame_exclusive": 89}],
        "window_frames": 61, "window_duration_s": 2., "eligible_clips": 1, "excluded_short_clips": 0,
        "sampling": "uniform clip then uniform valid contiguous window, with replacement; overlaps are not independent observations"}
    assert reference_selection(legacy, windows=1) == expected
    segmented = copy.deepcopy(legacy)
    segmented["splits"]["validation"]["clips"][0].update(
        source_path="validation.motion#frames=50:140", parent_source_path="validation.motion",
        parent_control_frame_start=50, parent_control_frame_stop=140, original_motion_id=17)
    window = reference_selection(segmented, windows=1)["windows"][0]
    assert window["parent_source_path"] == "validation.motion"
    assert window["source_path"] == "validation.motion#frames=50:140"
    assert window["parent_control_frame_start"] == 50
    assert window["parent_control_frame_stop"] == 140
    assert window["original_motion_id"] == 17
    assert (window["start_frame"], window["stop_frame_exclusive"]) == (28, 89)


@pytest.mark.parametrize("split", [None, "train", "unknown"])
def test_frozen_selection_must_declare_real_heldout_split(split):
    selection = {"split": split, "windows": [{"source_path": "test.motion"}]}
    document = {"splits": {"train": {"clips": [{"source_path": "train.motion"}]}}}
    with pytest.raises(ValueError, match="held-out"):
        assert_heldout_disjoint(selection, document)
