"""Fail-closed tracking input and physical clip-boundary style contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from CrowdSim.humanoid.style_evaluation import BODY_NAMES, sequence_style_samples
from scripts.evaluate_pedestrian_style import (
    assert_heldout_disjoint, read_tracking_capture, tracking_training_manifest,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, child_joint=False):
    directory = tmp_path/"tracking"
    (directory/"captures").mkdir(parents=True)
    entries = [{"motion_id": motion_id, "source_path": f"/heldout/clip_{motion_id}.motion", "frames": 14, "dt": 1/30}
               for motion_id in (13, 27)]
    heldout = {"splits": {"train": {"clips": [{"source_path": "/oldtrain/clip.motion"}]},
                           "validation": {"clips": entries}}}
    heldout_path = tmp_path/"heldout.json"
    heldout_path.write_text(json.dumps(heldout))
    trained = {"splits": {"train": {"clips": [{"source_path": "/actualtrain/clip.motion#frames=0:100"}], "sha256": "banksha"}}}
    trained_path = tmp_path/"trained.json"
    trained_path.write_text(json.dumps(trained))
    mode, frame = ("child_joint", "child_joint") if child_joint else ("legacy_stored", "parent_joint")
    runner = {"status": "passed", "arguments": {"manifest": str(heldout_path), "split": "validation"},
              "preflight": {"manifest_sha256": digest(heldout_path), "selected_clips": []},
              "physics": {"control_hz": 30, "body_names": list(BODY_NAMES)},
              "protocol": {
                  "metric_phase": "post-physics at30Hz vs same reference frame; no pelvis/yaw/frame best-match alignment".replace("at30Hz", "at 30Hz"),
                  "initial_frame_excluded_from_errors": True, "fall_reset_within_sequence": False,
                  "reference_3d_poses_enter_actor_or_actions": False,
                  "commands": "ten true reference root XY positions at +.5..5s; clip-tail hold is command-only"},
              "reference_velocity_contract": {
                  "expert_dof_velocity_mode": mode, "reference_dof_velocity_frame": frame,
                  "initial_reset_dof_velocity_frame": frame, "amp_reference_history_dof_velocity_frame": frame,
                  "physical_native_dof_velocity_frame": "child_joint", "poses_position_rotation_metric_changed": False},
              "controller_provenance": {"dataset": {
                  "manifest": str(trained_path), "manifest_sha256": digest(trained_path), "split": "train",
                  "source_sha256": "banksha", "clips": 1}},
              "sequences": [], "aggregate": {"measured_sequences": 2, "observed_metric_steps": 8}}
    for ordinal, entry in enumerate(entries):
        window = {"initial_frame": 9, "last_frame": 13, "available_steps": 4, "requested_steps": 4,
                  "minimum_steps": 1, "eligible": True, "full_clip_requested": True}
        selected = {**entry, "tracking_window": window}
        runner["preflight"]["selected_clips"].append(selected)
        # Initial reset is deliberately huge, and the two real clips are far
        # apart. Neither jump may contaminate the style speed/acceleration.
        pose = np.zeros((5, 1, 24, 3), dtype=np.float32)
        pose[..., 0] = np.arange(5)[:, None, None]*.03+ordinal*1000
        pose[..., 1] = np.arange(24)[None, None, :]*.01
        pose[..., 2] = 1+np.arange(24)[None, None, :]*.02
        pose[0, ..., 0] = -9999
        quat = np.broadcast_to([0., 0., 0., 1.], (5, 1, 4)).astype(np.float32).copy()
        falls = np.array([False, False, False, ordinal == 1, ordinal == 1])
        path = directory/"captures"/f"motion_{entry['motion_id']:06d}.npz"
        np.savez(path, rigid_body_pos=pose, root_rot=quat, reference_frame=np.arange(9, 14),
            reference_time_s=np.arange(9, 14)/30, metric_frame=np.array([False, True, True, True, True]),
            native_reset_frame=np.array([True, False, False, False, False]),
            episode_id=np.full((5, 1), ordinal+8, dtype=np.int64), fall_mask=falls)
        runner["sequences"].append({"motion_id": entry["motion_id"], "source_path": entry["source_path"],
            "capture": str(path), "capture_sha256": digest(path), "tracking_window": window,
            "observed_metric_steps": 4, "expected_control_steps": 4, "complete_clip": True,
            "fall_capture_frames": np.flatnonzero(falls).tolist(), "sequence_had_fall": bool(falls.any())})
    save_result(directory, runner)
    return directory, runner


def save_result(directory, runner):
    (directory/"result.json").write_text(json.dumps(runner))


def change_archive(runner, **updates):
    row = runner["sequences"][0]
    path = Path(row["capture"])
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    values.update(updates)
    np.savez(path, **values)
    row["capture_sha256"] = digest(path)


def test_true_poststep_only_clip_boundaries_exclude_both_jumps_and_keep_falls(tmp_path):
    directory, _ = fixture(tmp_path)
    arrays, runner, capture = read_tracking_capture(directory)
    assert capture["metric_frames"] == 8 and capture["initial_reset_frames_discarded"] == 2
    assert capture["valid_same_clip_style_intervals"] == 6 and capture["fall_metric_frames_kept"] == 2
    assert arrays["episode_ids"][:, 0].tolist() == [0]*4+[1]*4
    assert np.flatnonzero(arrays["reset_mask"][:, 0]).tolist() == [0, 4]
    assert arrays["body_pos"][:, 0, 0, 0].min() > 0
    measured = sequence_style_samples(arrays["body_pos"], arrays["root_rot"],
        episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"])
    assert measured["summary"]["same_episode_transition_count"] == 6
    assert measured["summary"]["excluded_reset_transitions"] == 1
    np.testing.assert_allclose(measured["speed_m_s"], .9, atol=.002)
    assert capture["reference_velocity_contract"]["reference_dof_velocity_frame"] == "parent_joint"
    path, training = tracking_training_manifest(runner)
    assert path.name == "trained.json"  # Theoracle's oldtrain manifest is not actual training provenance.
    review = assert_heldout_disjoint({"split": "validation", "windows": [{"source_path": "/heldout/clip_13.motion"}]}, training)
    assert review["original_source_intersection"] == 0


def test_explicit_child_reference_contract_kept_and_stray_file_not_globbed(tmp_path):
    directory, _ = fixture(tmp_path, child_joint=True)
    (directory/"captures"/"stray.npz").write_bytes(b"not a usable NPZ")
    _, _, capture = read_tracking_capture(directory)
    assert capture["clips"] == 2
    assert capture["reference_velocity_contract"]["expert_dof_velocity_mode"] == "child_joint"
    assert capture["reference_velocity_contract"]["reference_dof_velocity_frame"] == "child_joint"


@pytest.mark.parametrize("failure", ["missing_sequence", "duplicate_id", "unknown_id", "source_path", "metric_phase",
    "heldout_hash", "training_hash", "training_split", "training_bank_hash", "aggregate_count", "sequence_count",
    "capture_hash", "escaped_path", "partial_clip", "native_frame", "child_mode_parent_frame", "fall_evidence"])
def test_result_and_source_contracts_fail_closed(tmp_path, failure):
    directory, runner = fixture(tmp_path)
    if failure == "missing_sequence":
        runner["sequences"].pop()
    elif failure == "duplicate_id":
        runner["sequences"][1]["motion_id"] = 13
    elif failure == "unknown_id":
        runner["sequences"][0]["motion_id"] = 99
        runner["preflight"]["selected_clips"][0]["motion_id"] = 99
    elif failure == "source_path":
        runner["sequences"][0]["source_path"] = "/heldout/another.motion"
    elif failure == "metric_phase":
        runner["protocol"]["metric_phase"] = "pre-physics"
    elif failure == "heldout_hash":
        runner["preflight"]["manifest_sha256"] = "changed"
    elif failure == "training_hash":
        runner["controller_provenance"]["dataset"]["manifest_sha256"] = "changed"
    elif failure == "training_split":
        runner["controller_provenance"]["dataset"]["split"] = "validation"
    elif failure == "training_bank_hash":
        runner["controller_provenance"]["dataset"]["source_sha256"] = "changed"
    elif failure == "aggregate_count":
        runner["aggregate"]["observed_metric_steps"] = 7
    elif failure == "sequence_count":
        runner["sequences"][0]["observed_metric_steps"] = 3
    elif failure == "capture_hash":
        runner["sequences"][0]["capture_sha256"] = "changed"
    elif failure == "escaped_path":
        runner["sequences"][0]["capture"] = str(tmp_path/"outside.npz")
    elif failure == "partial_clip":
        runner["sequences"][0]["complete_clip"] = False
    elif failure == "native_frame":
        runner["reference_velocity_contract"]["physical_native_dof_velocity_frame"] = "parent_joint"
    elif failure == "child_mode_parent_frame":
        runner["reference_velocity_contract"]["expert_dof_velocity_mode"] = "child_joint"
    elif failure == "fall_evidence":
        runner["sequences"][0]["sequence_had_fall"] = True
    save_result(directory, runner)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_tracking_capture(directory)


@pytest.mark.parametrize("failure", ["phase_gap", "time_offset", "mask_integer", "second_reset", "episode_change",
                                      "nonfinite", "quaternion", "shape", "fall_integer"])
def test_hashed_but_invalid_physical_capture_fails_closed(tmp_path, failure):
    directory, runner = fixture(tmp_path)
    if failure == "phase_gap":
        change_archive(runner, reference_frame=np.array([9, 10, 11, 13, 14]))
    elif failure == "time_offset":
        change_archive(runner, reference_time_s=np.arange(9, 14)/30+.1)
    elif failure == "mask_integer":
        change_archive(runner, metric_frame=np.array([0, 1, 1, 1, 1]))
    elif failure == "second_reset":
        change_archive(runner, native_reset_frame=np.array([True, False, True, False, False]))
    elif failure == "episode_change":
        change_archive(runner, episode_id=np.arange(5)[:, None])
    elif failure == "nonfinite":
        values = np.zeros((5, 1, 24, 3), dtype=np.float32)
        values[1, 0, 2, 0] = np.nan
        change_archive(runner, rigid_body_pos=values)
    elif failure == "quaternion":
        change_archive(runner, root_rot=np.broadcast_to([0., 0., 0., 2.], (5, 1, 4)))
    elif failure == "shape":
        change_archive(runner, rigid_body_pos=np.zeros((5, 1, 23, 3)))
    elif failure == "fall_integer":
        change_archive(runner, fall_mask=np.zeros(5, dtype=int))
    save_result(directory, runner)
    with pytest.raises(ValueError):
        read_tracking_capture(directory)


def test_missing_declared_capture_cannot_be_silently_omitted(tmp_path):
    directory, runner = fixture(tmp_path)
    Path(runner["sequences"][0]["capture"]).unlink()
    with pytest.raises(FileNotFoundError):
        read_tracking_capture(directory)
