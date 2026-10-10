"""Synthetic CPU contracts only; no SDK execution, native capture or quality proof."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.analysis import native_navigation_style as style


def declared_audit(actors=3):
    kernels = style.raw.prepare_kernels(actors=actors, controls=6000)
    owner = {"schema": style.OWNER_SCHEMA, "actual_exit_code": 0, "terminal_tool_chunk": "IN_MEMORY_FIXTURE_ONLY",
             "actors": actors, "control_steps": 6000, "planner": "diffusion"}
    audit = {"schema": style.raw.AUDIT_SCHEMA, "status": "passed", "actors": actors, "actual_controls": 6000,
        "actor_controls": 6000 * actors, "physics_substeps": 24000, "native_actor_substep_samples": 24000 * actors,
        "unordered_pair_controls_exposure": 6000 * actors * (actors - 1) // 2,
        "planner": "diffusion", "scope": "native_local_200s", "duration_s": 200,
        "all_failure_contact_and_reset_frames_retained": True,
        "recorded_body_pose": {"all24_same_control_native_body_positions_equal": True},
        "initial_capture_link": {"status": "live_initial_capture_linked",
            "all_registered_first_raw_reference_position_velocity_anchors_equal": True},
        "reader": style.metric.record(style.ROOT / style.SOURCES[2][0]), "kernel_sources": kernels.sources,
        "kernel_derivation": kernels.derivation, "metric_formulae_changed": False, "Torch_imported": False,
        "SDK_executed_by_reader": False, "execution_admission": {
            "declared_native_stage_floor_source_and_exposure_admitted": True,
            "actual_process_exit_independently_observed_by_reader": False}}
    return owner, audit, kernels


def synthetic_capture(actors):
    frames = 6000
    names = list(style.metric.BODY_NAMES)
    offsets = np.array([[index * .03, index * .015, .01 * index] for index in range(24)], dtype=np.float32)
    offsets[0] = 0
    body = np.broadcast_to(offsets, (frames, actors, 24, 3)).copy()
    body[..., 0] += np.arange(frames, dtype=np.float32)[:, None, None] * (.2 / 30)
    body[..., 2] += 1
    q = np.zeros((frames, actors, 4), dtype=np.float32)
    q[..., 3] = 1
    episodes = np.zeros((frames, actors), np.int64)
    resets = np.zeros((frames, actors), bool)
    resets[0] = True
    episodes[100:, 1] = 1
    resets[100, 1] = True
    # Both failed frames remain in the capture; the initial row has no predecessor.
    terminated = np.zeros((frames, actors), bool)
    terminated[0, 0] = terminated[99, 1] = True
    physical = {"body_pos": body, "root_rot": q, "episode_ids": episodes,
                "reset_mask": resets, "terminated": terminated}
    native_q = np.broadcast_to(q[:, None, :, None], (frames, 4, actors, 24, 4)).copy()
    ground = {"common_body_names": np.asarray(names), "body_link_pos_w": np.repeat(body[:, None], 4, axis=1),
              "body_link_quat_w": native_q, "episode_ids": episodes.copy(), "reset_mask": resets.copy()}
    result = {"state_capture": {"body_names": names}}
    base = {"same_episode_position_intervals": (frames - 1) * actors - 1,
            "fall_termination_actor_controls": 2}
    return physical, ground, result, base


@pytest.mark.parametrize("actors", (3, 15))
def test_native_counts_are_separate_from_four_substeps_quaternion_and_feet(actors):
    owner, audit, kernels = declared_audit(actors)
    style.validate_audit_contract(owner, audit, kernels, planner="diffusion")
    assert audit["physics_substeps"] == 24000
    assert audit["actor_controls"] == 6000 * actors
    assert audit["native_actor_substep_samples"] == 24000 * actors


@pytest.mark.parametrize("code", (None, False, 0., 1, -1))
def test_pending_boolean_float_and_failed_raw_audit_exits_are_rejected(code):
    owner, audit, kernels = declared_audit()
    owner["actual_exit_code"] = code
    with pytest.raises(ValueError, match="externally observed"):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion")


@pytest.mark.parametrize("field,value", (("schema", "shared_social.actual_high_level_CPU_audit_execution.v1"),
    ("actors", 4), ("actors", 3.), ("control_steps", 30), ("planner", "sfm"), ("terminal_tool_chunk", "")))
def test_wrong_owner_protocol_cannot_establish_a_native_style_run(field, value):
    owner, audit, kernels = declared_audit()
    owner[field] = value
    with pytest.raises(ValueError, match="externally observed"):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion")


@pytest.mark.parametrize("field,value", (("actual_controls", 30), ("actors", 15), ("actor_controls", 24000),
    ("physics_substeps", 18000), ("native_actor_substep_samples", 24000), ("unordered_pair_controls_exposure", 1),
    ("scope", "native_smoke30"), ("duration_s", 1), ("all_failure_contact_and_reset_frames_retained", False)))
def test_smoke_wrong_denominators_and_filtered_failures_are_rejected(field, value):
    owner, audit, kernels = declared_audit()
    audit[field] = value
    with pytest.raises(ValueError, match="complete200s"):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion")


@pytest.mark.parametrize("defect", ("body", "initial", "source", "derivation", "metric", "Torch", "SDK", "exit_observer"))
def test_native_body_initial_sources_and_CPU_ownership_are_required(defect):
    owner, audit, kernels = declared_audit()
    if defect == "body":
        audit["recorded_body_pose"]["all24_same_control_native_body_positions_equal"] = False
    elif defect == "initial":
        audit["initial_capture_link"]["all_registered_first_raw_reference_position_velocity_anchors_equal"] = False
    elif defect == "source":
        audit["kernel_sources"] = copy.deepcopy(audit["kernel_sources"])
        audit["kernel_sources"]["native_reader"]["sha256"] = "other"
    elif defect == "derivation":
        audit["kernel_derivation"] = copy.deepcopy(audit["kernel_derivation"])
        audit["kernel_derivation"]["actors"] = 4
    elif defect == "metric":
        audit["metric_formulae_changed"] = True
    elif defect == "exit_observer":
        audit["execution_admission"]["actual_process_exit_independently_observed_by_reader"] = True
    else:
        audit["Torch_imported" if defect == "Torch" else "SDK_executed_by_reader"] = True
    with pytest.raises(ValueError, match="complete200s|unchanged independently audited"):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion")


@pytest.mark.parametrize("actors", (3, 15))
def test_full_native_sample_extraction_preserves_original_math_and_failed_frames(actors):
    physical, ground, result, base = synthetic_capture(actors)
    samples, admission = style.capture_samples(physical, ground, result, base, actors=actors)
    original = style.metric.sequence_style_samples(physical["body_pos"], physical["root_rot"],
        body_names=result["state_capture"]["body_names"], episode_ids=physical["episode_ids"],
        reset_mask=physical["reset_mask"], spec=style.metric.StyleSpec())
    for key in ("local_pose", "local_pose_velocity", "speed_m_s", "end_frame_indices", "actor_indices"):
        np.testing.assert_array_equal(samples[key], original[key])
    assert samples["summary"] == original["summary"]
    assert samples["local_pose"].shape == ((5999 * actors) - 1, 45)
    assert admission["all_capture_frames"] == 6000 * actors
    assert admission["fall_termination_frames"] == 2
    assert admission["fall_termination_interval_endpoints_kept"] == 1
    assert admission["fall_termination_frames_without_legal_predecessor"] == 1
    assert admission["reset_boundary_intervals_excluded"] == 1
    assert admission["failed_episodes_filtered"] is False


@pytest.mark.parametrize("defect", ("nonroot_pose", "quaternion", "substeps", "episode", "names", "terminated_dtype",
                                   "transitions", "failures", "nonunit", "nonfinite"))
def test_native_style_capture_drift_or_missing_population_is_rejected(defect):
    physical, ground, result, base = synthetic_capture(3)
    if defect == "nonroot_pose":
        physical["body_pos"][20, 2, 23, 0] += 1
    elif defect == "quaternion":
        ground["body_link_quat_w"][20, -1, 2, 0, 3] = .5
    elif defect == "substeps":
        ground["body_link_pos_w"] = ground["body_link_pos_w"][:, :3]
    elif defect == "episode":
        ground["episode_ids"][200, 2] += 1
    elif defect == "names":
        result["state_capture"]["body_names"] = list(reversed(result["state_capture"]["body_names"]))
    elif defect == "terminated_dtype":
        physical["terminated"] = physical["terminated"].astype(np.float32)
    elif defect in ("transitions", "failures"):
        base["same_episode_position_intervals" if defect == "transitions" else "fall_termination_actor_controls"] += 1
    elif defect == "nonunit":
        physical["root_rot"][20, 1, 3] = 2
        ground["body_link_quat_w"][20, -1, 1, 0, 3] = 2
    else:
        physical["body_pos"][20, 1, 20, 0] = np.nan
        ground["body_link_pos_w"][20, -1, 1, 20, 0] = np.nan
    with pytest.raises(ValueError):
        style.capture_samples(physical, ground, result, base, actors=3)


def test_old_or_pending_actual_audit_is_rejected_before_resource_paths(tmp_path):
    for schema, code in (("shared_social.actual_high_level_CPU_audit_execution.v1", 0),
                         (style.OWNER_SCHEMA, False), (style.OWNER_SCHEMA, None)):
        owner = tmp_path / "CPU_FIXTURE_owner.json"
        owner.write_text(json.dumps({"schema": schema, "actual_exit_code": code, "terminal_tool_chunk": "FIXTURE_ONLY"}))
        with pytest.raises(ValueError, match="externally observed"):
            style.read_completed_run(owner, actors=3, planner="diffusion", scene_config=tmp_path / "DOES_NOT_EXIST",
                                     scene_sha256="DOES_NOT_EXIST")


def test_source_check_runs_in_fresh_CPU_process_without_Torch_or_SDK():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(style.ROOT / "scripts/evaluate_shared_social_native_style.py"),
               "--actors", "15", "--check-sources"]
    result = subprocess.run(command, env=env, cwd=style.ROOT, capture_output=True, text=True, check=True)
    document = json.loads(result.stdout)
    assert document["actors"] == 15 and document["required_controls"] == 6000
    assert document["Torch_imported"] is False and document["SDK_started"] is False
    assert document["actual_native_capture_or_quality_measured"] is False
    assert document["reference_or_basis_refit"] is False


def test_source_drift_is_rejected(monkeypatch):
    changed = list(style.SOURCES)
    changed[0] = (changed[0][0], "other")
    monkeypatch.setattr(style, "SOURCES", changed)
    with pytest.raises(ValueError, match="frozen style source changed"):
        style.source_bindings()
