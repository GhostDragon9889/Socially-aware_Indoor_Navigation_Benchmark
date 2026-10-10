"""CPU regressions for actual curation provenance, quality and rate conversion."""

from dataclasses import replace
import hashlib
import json
import math

import pytest
import torch

from CrowdSim.humanoid.training.amass_curation import (
    CurationConfig, FIELDS, NEGATIVE, assess_clip, canonical_fingerprint,
    curate_manifest, resample_clip, select_balanced, slerp_xyzw,
    split_namespace_review, segment_spans,
    joint_angular_velocity_xyzw, convert_curated_joint_velocities,
)
from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms, compute_kinematics_velocities,
    extract_qpos_from_transforms,
)
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix, quat_mul


@pytest.fixture(scope="module")
def info():
    return SmplRobotConfig().kinematic_info


def make_clip(info, *, count=120, hz=30, speed=1., joint_angle=None):
    times = torch.arange(count)/hz
    root = torch.zeros(count, 3)
    root[:, 0], root[:, 2] = times*speed, 1.
    local = torch.zeros(count, 24, 4)
    local[..., 3] = 1
    if joint_angle is not None:
        local[:, 1, 2] = torch.sin(joint_angle/2)
        local[:, 1, 3] = torch.cos(joint_angle/2)
    matrices = quaternion_to_matrix(local, w_last=True)
    positions, rotations = compute_forward_kinematics_from_transforms(info, root, matrices)
    velocity, angular = compute_kinematics_velocities(positions, rotations, hz, velocity_max_horizon=1)
    qpos = extract_qpos_from_transforms(info, root, matrices, multi_dof_decomposition_method="exp_map")
    return {"gts": positions, "grs": matrix_to_quaternion(rotations, w_last=True),
            "gvs": velocity, "gavs": angular, "dps": qpos[:, 7:],
            "dvs": torch.zeros(count, 69), "lrs": local,
            "contacts": torch.zeros(count, 24, dtype=torch.bool)}


def test_every_frame_finite_and_upright_is_checked(info):
    clip = make_clip(info, count=1000)
    clip["dvs"][497, 68] = float("nan")
    assert "nonfinite" in assess_clip(clip, 1/30, CurationConfig(), body_names=info.body_names)["failures"]
    clip = make_clip(info, count=1000)
    clip["gts"][440:500, 0, 2] = .2
    assert "upright_fraction" in assess_clip(clip, 1/30, CurationConfig(), body_names=info.body_names)["failures"]


def test_local_arm_sway_is_not_moving_navigation(info):
    clip = make_clip(info)
    clip["gts"][:, :, 0] *= .02
    # A cached velocity alone must not certify translated locomotion.
    result = assess_clip(clip, 1/30, CurationConfig(), body_names=info.body_names)
    assert result["median_speed_m_s"] == pytest.approx(1., abs=2e-6)
    assert "moving_translation_extent" in result["failures"]


def test_true_rate_is_checked_not_silently_relabeled(info):
    clip = make_clip(info, count=150, hz=50)
    result = assess_clip(clip, 1/50, CurationConfig(), body_names=info.body_names)
    assert "requires_real_resampling" in result["failures"]


def test_fingerprint_removes_placement_and_quaternion_sign(info):
    clip = make_clip(info, speed=0.)
    relocated = {key: value.clone() for key, value in clip.items()}
    relocated["gts"][..., :2] += torch.tensor([16., -32.])
    relocated["grs"] *= -1
    relocated["lrs"] *= -1
    assert canonical_fingerprint(clip, 1/30) == canonical_fingerprint(relocated, 1/30)
    relocated["gts"][50, 3, 0] += .01
    assert canonical_fingerprint(clip, 1/30) != canonical_fingerprint(relocated, 1/30)


def test_negative_words_do_not_match_transition_substrings():
    assert NEGATIVE.search("throwing_hard")
    assert NEGATIVE.search("sit_to_stand")
    assert not NEGATIVE.search("standing_transition")
    assert not NEGATIVE.search("walking_positions")


def test_subject_namespace_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        split_namespace_review({"train": ["/smpl/CMU/01/01_01.motion"],
                                "test": ["/smpl/CMU/01/01_02.motion"]})
    report = split_namespace_review({"train": ["/smpl/CMU/01/01_01.motion"],
                                     "test": ["/smpl/SSM/01/walk.motion"]})
    assert report["subject_namespace_disjoint"]
    assert not report["physical_identity_across_datasets_verified"]


def test_selection_deterministic_caps_standing_and_does_not_pad():
    rows = [{"motion_id": index, "source_path": f"/smpl/CMU/{index%3}/{index}.motion",
             "dataset": "CMU", "subject_namespace": f"CMU/{index%3}",
             "style_class": "standing_looking_like" if index < 20 else "walking_like"}
            for index in range(30)]
    config = CurationConfig(target_train_clips=20, standing_fraction_maximum=.25)
    selected = select_balanced(rows, config)
    assert selected == select_balanced(list(reversed(rows)), config)
    assert len(selected) == 13  # ten moving, atmost25% actual standing; no padding.
    assert sum(row["style_class"] == "standing_looking_like" for row in selected) == 3
    assert len({row["motion_id"] for row in selected}) == len(selected)


def test_verified_navigation_precedes_unlabeled_proxy_candidates():
    rows = [{"motion_id": index, "source_path": str(index), "dataset": "CMU",
             "subject_namespace": f"CMU/{index}", "style_class": "walking_like",
             "label_evidence": "official_motion_description" if index >= 5 else "kinematic_candidate_only"}
            for index in range(10)]
    selected = select_balanced(rows, CurationConfig(target_train_clips=5))
    assert {row["motion_id"] for row in selected} == set(range(5, 10))


def test_long_segments_do_not_repeat_frames_or_include_full_parent():
    config = CurationConfig()
    spans = segment_spans(3000, config)
    assert len(spans) == 4
    assert all(stop-start == 250 for start, stop in spans)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    assert spans[0][0] == 0 and spans[-1][1] <= 3000
    assert (0, 3000) not in spans
    assert segment_spans(400, config) == [(0, 250), (250, 400)]
    assert segment_spans(200, config) == [(0, 100), (100, 200)]
    assert segment_spans(300, config) == [(0, 300)]
    assert segment_spans(650, config) == [(0, 250), (250, 500), (500, 650)]
    assert all((stop-start-1)/30 >= 3 for start, stop in segment_spans(200, config))


@pytest.mark.parametrize("field", ["target_train_clips", "seed", "control_hz", "maximum_speed_p99_m_s", "standing_fraction_maximum"])
def test_bool_does_not_pass_numeric_config(field):
    with pytest.raises(ValueError):
        replace(CurationConfig(), **{field: True})


def test_slerp_antipodal_and_long_branch_are_finite():
    first = torch.tensor([[0., 0., math.sin(math.radians(170)/2), math.cos(math.radians(170)/2)]])
    second = torch.tensor([[0., 0., math.sin(math.radians(190)/2), math.cos(math.radians(190)/2)]])
    midpoint = slerp_xyzw(first, second, torch.tensor([.5]))
    assert torch.isfinite(midpoint).all()
    assert midpoint[0, 2].abs() == pytest.approx(1.)
    same = slerp_xyzw(first, -first, torch.tensor([.25]))
    assert torch.allclose(same, first, atol=1e-6)


def test_child_velocity_uses_analytic_child_axis_not_parent_or_expmap_derivative():
    # q(t)=Rz(90deg)*Rx(.4*t): the childbody angular velocity is+.4 aboutX,
    # while its parentframe angular velocity is+.4 aboutY. The finite rotation
    # makes these distinguishable, matching the live D6 probe contract.
    count, dt = 120, 1/30
    angle = torch.arange(count)*dt*.4
    x_rotation = torch.stack((torch.sin(angle/2), torch.zeros(count), torch.zeros(count), torch.cos(angle/2)), -1)
    z_rotation = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)]).expand_as(x_rotation)
    local = quat_mul(z_rotation, x_rotation, w_last=True)[:, None]
    child = joint_angular_velocity_xyzw(local, dt, frame="child_joint")
    parent = joint_angular_velocity_xyzw(local, dt, frame="parent_joint")
    assert torch.allclose(child[:, 0], torch.tensor([.4, 0., 0.]).expand(count, 3), atol=3e-5)
    assert torch.allclose(parent[:, 0], torch.tensor([0., .4, 0.]).expand(count, 3), atol=3e-5)
    assert torch.equal(child[-1], child[-2])
    assert not torch.allclose(child, parent)


def test_real_50_to_30_conversion_preserves_fk_and_angular_velocity(info):
    source = make_clip(info, count=76, hz=50,
                       joint_angle=torch.deg2rad(torch.linspace(170., 190., 76)))
    result, report = resample_clip(source, 1/50, 30, info)
    assert result["gts"].shape == (46, 24, 3)
    assert report["source_duration_s"] == report["target_duration_s"] == 1.5
    assert result["gts"][-1, 0, 0] == pytest.approx(1.5)
    assert torch.allclose(result["gvs"][:, 0, 0], torch.ones(46), atol=2e-5)
    assert torch.isfinite(result["dps"]).all()
    assert result["dvs"].abs().max() < 1  # no exp-map +/-pi derivative spike.
    matrices = quaternion_to_matrix(result["lrs"], w_last=True)
    fresh, _ = compute_forward_kinematics_from_transforms(info, result["gts"][:, 0], matrices)
    assert torch.linalg.vector_norm(fresh-result["gts"], dim=-1).max() < 1e-6
    source["gts"][0, 5, 0] += .01
    with pytest.raises(ValueError, match="FK mismatch"):
        resample_clip(source, 1/50, 30, info)


def save_bank(path, clips, sources, dt):
    counts = torch.tensor([len(clip["gts"]) for clip in clips])
    bank = {key: torch.cat([clip[key] for clip in clips]) for key in (*FIELDS, "contacts")}
    bank.update(length_starts=counts.cumsum(0)-counts, motion_num_frames=counts,
                motion_dt=torch.full((len(clips),), dt), motion_files=tuple(sources))
    torch.save(bank, path)
    records = [{"motion_id": index, "source_path": source, "frames": int(counts[index]),
                "dt": float(bank["motion_dt"][index]), "diffusion_eligible": int(counts[index]) >= 247}
               for index, source in enumerate(sources)]
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clips": records, "clip_count": len(records), "diffusion_clip_count": 0}


def test_manifest_preserves_heldout_and_reindexes_real_resampled_bank(tmp_path, info):
    base = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {}}
    for split, dataset, speed in (("train", "KIT", 1.), ("validation", "HumanEva", .8), ("test", "SSM", .7)):
        base["splits"][split] = save_bank(tmp_path/f"{split}.pt",
            [make_clip(info, count=180, hz=50, speed=speed)],
            [f"/smpl/{dataset}/01/walking_01.motion"], 1/50)
    base_path = tmp_path/"base.json"
    base_path.write_text(json.dumps(base))
    output = tmp_path/"curated.json"
    document = curate_manifest(base_path, output, config=CurationConfig(target_train_clips=1, dof_velocity_frame="child_joint"),
                               body_names=info.body_names, kinematic_info=info)
    assert document["splits"]["validation"] == base["splits"]["validation"]
    assert document["splits"]["test"] == base["splits"]["test"]
    assert document["curation"]["selected_resampled_clips"] == 1
    entry = document["splits"]["train"]
    assert entry["path"] != base["splits"]["train"]["path"]
    bank = torch.load(entry["path"], weights_only=True, mmap=True)
    assert bank["dof_velocity_frame"] == entry["dof_velocity_frame"] == entry["clips"][0]["dof_velocity_frame"] == "child_joint"
    assert bank["motion_dt"][0] == pytest.approx(1/30)
    assert bank["motion_num_frames"][0] == entry["clips"][0]["frames"]
    assert entry["clips"][0]["original_motion_id"] == 0
    row = entry["clips"][0]
    assert row["parent_source_path"] == base["splits"]["train"]["clips"][0]["source_path"]
    assert bank["motion_files"][0] == f"{row['parent_source_path']}#frames=0:{row['frames']}"
    assert row["parent_control_frame_start"] == row["parent_source_frame_start"] == 0
    assert row["parent_control_frame_stop"] == row["frames"]
    assert row["parent_source_frame_stop"] <= 180
    with pytest.raises(FileExistsError):
        curate_manifest(base_path, output, body_names=info.body_names, kinematic_info=info)


def test_official_description_overrides_numeric_filename_and_rejects_sports(tmp_path, info):
    train_names = ["/smpl/CMU/86/86_02_poses.motion", "/smpl/CMU/136/136_20_poses.motion"]
    base = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {}}
    base["splits"]["train"] = save_bank(tmp_path/"train.pt",
        [make_clip(info, speed=1.), make_clip(info, speed=1.1)], train_names, 1/30)
    for split, dataset, speed in (("validation", "HumanEva", .8), ("test", "SSM", .7)):
        base["splits"][split] = save_bank(tmp_path/f"{split}.pt", [make_clip(info, speed=speed)],
            [f"/smpl/{dataset}/01/walking_01.motion"], 1/30)
    base_path = tmp_path/"base.json"
    base_path.write_text(json.dumps(base))
    labels = {train_names[0]: {"description": "Walk, run, jumps and punches", "source_url": "http://mocap.cs.cmu.edu/search.php"},
              train_names[1]: {"description": "Normal Walk", "source_url": "http://mocap.cs.cmu.edu/search.php"}}
    result = curate_manifest(base_path, tmp_path/"out.json", config=CurationConfig(target_train_clips=1),
        body_names=info.body_names, kinematic_info=info, labels_by_source=labels)
    row = result["splits"]["train"]["clips"][0]
    assert row["parent_source_path"] == train_names[1]
    assert row["semantic_label_verified"]
    assert row["semantic_annotation"]["description"] == "Normal Walk"
    assert result["curation"]["rejection_reason_counts_not_mutually_exclusive"]["negative_description_label"] == 1


def test_velocity_version_preserves_real_saved_pose_and_holdout_fields(tmp_path, info):
    base = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {}}
    for split, dataset, speed in (("train", "KIT", 1.), ("validation", "HumanEva", .8), ("test", "SSM", .7)):
        base["splits"][split] = save_bank(tmp_path/f"{split}.pt", [make_clip(info, speed=speed)],
            [f"/smpl/{dataset}/01/walking_01.motion"], 1/30)
    base_path = tmp_path/"base.json"
    base_path.write_text(json.dumps(base))
    parent_path, child_path = tmp_path/"parent.json", tmp_path/"child.json"
    parent = curate_manifest(base_path, parent_path, config=CurationConfig(target_train_clips=1),
        body_names=info.body_names, kinematic_info=info)
    child = convert_curated_joint_velocities(parent_path, child_path)
    assert child["splits"]["validation"] == parent["splits"]["validation"]
    assert child["splits"]["test"] == parent["splits"]["test"]
    old = torch.load(parent["splits"]["train"]["path"], weights_only=True, mmap=True)
    new = torch.load(child["splits"]["train"]["path"], weights_only=True, mmap=True)
    for key, value in old.items():
        if key not in {"dvs", "dof_velocity_frame"}:
            assert torch.equal(value, new[key]) if isinstance(value, torch.Tensor) else value == new[key]
    assert child["selection"]["dof_velocity_frame"] == new["dof_velocity_frame"] == "child_joint"
    assert child["curation"]["joint_velocity_conversion"]["child_velocity_quality_gate_all_passed"]
    assert parent_path.read_text() == json.dumps(parent, indent=2) + "\n"
    with pytest.raises(FileExistsError):
        convert_curated_joint_velocities(parent_path, child_path)
