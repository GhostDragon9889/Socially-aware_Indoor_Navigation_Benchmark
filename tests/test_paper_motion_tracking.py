"""CPU metric and runner-interface tests; mocks are not physical simulation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.motion_tracking import (
    SMPL_BODY_NAMES, align_reference_positions, body_tracking_errors,
    future_reference_frames, summarize_tracking_sequence, tracking_window,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("motion_tracking_script", ROOT/"scripts/benchmark_paper_motion_tracking.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def pose(count=4):
    positions = np.zeros((count, 24, 3), dtype=np.float64)
    rotations = np.zeros((count, 24, 4), dtype=np.float64)
    rotations[..., 3] = 1
    return positions, rotations


def test_geodesic_antipodes_and_named_rotation_not_euler_component_difference():
    positions, target_q = pose(2)
    actual_q = -target_q.copy()
    actual_q[1, SMPL_BODY_NAMES.index("R_Knee")] = [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]
    errors = body_tracking_errors(positions, actual_q, positions, target_q)
    assert errors["mean_body_position_error_m"].tolist() == [0., 0.]
    np.testing.assert_allclose(errors["body_rotation_error_rad"][0], 0., atol=1e-8)
    assert errors["body_rotation_error_rad"][1, SMPL_BODY_NAMES.index("R_Knee")] == pytest.approx(np.pi/2)
    assert errors["mean_body_rotation_error_rad"][1] == pytest.approx(np.pi/48)


def test_one_bad_body_is_averaged_over_24_and_world_translation_is_not_removed():
    target, rotations = pose(2)
    actual = target.copy()
    actual[0, SMPL_BODY_NAMES.index("L_Hand"), 0] = 24
    actual[1, :, 1] = .75
    errors = body_tracking_errors(actual, rotations, target, rotations)
    assert errors["mean_body_position_error_m"].tolist() == [1., .75]


def test_name_bijection_reorders_two_independent_axes():
    positions, rotations = pose(2)
    positions[..., 0] = np.arange(24)
    actual_names = SMPL_BODY_NAMES[::-1]
    target_names = SMPL_BODY_NAMES[8:]+SMPL_BODY_NAMES[:8]
    expected = body_tracking_errors(positions, rotations, positions, rotations)
    actual = body_tracking_errors(positions[:, ::-1], rotations[:, ::-1],
                                 np.concatenate((positions[:, 8:], positions[:, :8]), axis=1), rotations,
                                 actual_body_names=actual_names, reference_body_names=target_names)
    np.testing.assert_array_equal(actual["body_position_error_m"], expected["body_position_error_m"])


def test_constant_initial_alignment_keeps_later_reference_motion_and_drift():
    source = np.array([[[1., 2., 1.]], [[2., 3., 1.]], [[3., 4., 1.]]])
    shifted = align_reference_positions(source, source[0, 0], np.array([6., 7., 1.]))
    np.testing.assert_array_equal(shifted[:, 0], [[6., 7., 1.], [7., 8., 1.], [8., 9., 1.]])
    np.testing.assert_array_equal(np.diff(shifted, axis=0), np.diff(source, axis=0))
    # A physically stationary pelvis remains two metres away by the last
    # frame. Re-aligning each frame would falsely erase this error.
    assert np.linalg.norm(shifted[-1, 0]-shifted[0, 0]) == pytest.approx(np.sqrt(8))


def test_threshold_is_step_mean_strict_latched_and_initial_failure_excluded():
    errors = {"mean_body_position_error_m": np.array([9., .5, .50001, .1]),
              "mean_body_rotation_error_rad": np.zeros(4)}
    summary = summarize_tracking_sequence(errors, metric_mask=np.array([False, True, True, True]),
        fall_mask=np.array([False, False, False, False]), expected_steps=3)
    assert summary["first_tracking_failure_capture_frame"] == 2
    assert summary["tracking_failed"] and summary["tracking_success"] is False
    assert summary["observed_metric_steps"] == 3
    assert summary["sequence_had_fall"] is False
    assert summary["mean_body_position_error_m"] == pytest.approx((.5+.50001+.1)/3)


def test_falls_are_separate_events_and_partial_runs_cannot_earn_success():
    errors = {"mean_body_position_error_m": np.full(6, .1), "mean_body_rotation_error_rad": np.full(6, .2)}
    result = summarize_tracking_sequence(errors, metric_mask=np.array([False]+[True]*5),
        fall_mask=np.array([False, True, True, False, True, True]), expected_steps=5)
    assert result["fall_onset_events"] == 2 and result["sequence_had_fall"]
    assert result["tracking_success"] is True  # The declared position-only rule.
    result = summarize_tracking_sequence(errors, metric_mask=np.array([False]+[True]*5),
        fall_mask=np.zeros(6, dtype=bool), expected_steps=6, full_clip_requested=False)
    assert result["tracking_success"] is None and not result["in_full_sequence_success_denominator"]


def test_real_history_window_short_clip_and_requested_cap_denominators():
    short = tracking_window(39, 1/30)
    assert short.initial_frame == 9 and short.available_steps == 29 and not short.eligible
    full = tracking_window(40, 1/30)
    assert full.eligible and full.available_steps == 30 and full.full_clip_requested
    partial = tracking_window(500, 1/30, max_steps=30)
    assert partial.eligible and not partial.full_clip_requested and partial.requested_steps == 30
    assert tracking_window(3, 1/30).available_steps == 0
    with pytest.raises(ValueError, match="30Hz"):
        tracking_window(500, 1/50)


def test_future_time_indices_and_tail_hold_do_not_extend_metric_clip():
    indices, held = future_reference_frames(9, 200)
    np.testing.assert_array_equal(indices, np.arange(24, 160, 15))
    assert not held.any()
    indices, held = future_reference_frames(190, 200)
    assert indices.tolist() == [199]*10 and held.all()
    window = tracking_window(200, 1/30)
    assert window.available_steps == 190 and window.last_frame == 199


@pytest.mark.parametrize("bad", [np.full((2, 24, 3), np.nan), np.zeros((2, 23, 3)), np.zeros((0, 24, 3))])
def test_invalid_position_capture_fails_closed(bad):
    positions, rotations = pose(2)
    with pytest.raises(ValueError):
        body_tracking_errors(bad, rotations, positions, rotations)


def test_nonunit_rotation_and_duplicate_rig_names_rejected():
    positions, rotations = pose(2)
    with pytest.raises(ValueError, match="unit"):
        body_tracking_errors(positions, rotations*2, positions, rotations)
    with pytest.raises(ValueError, match="bijection"):
        body_tracking_errors(positions, rotations, positions, rotations, actual_body_names=("Pelvis",)*24)


def fixture_resources(tmp_path):
    bank = tmp_path/"bank.pt"
    bank.write_bytes(b"PK\x03\x04unit-test-header-only")
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"PK\x03\x04unit-test-checkpoint-header-only")
    manifest = {"schema": "navisaaclab.amass_reconstruction.v1", "author_clip_list": False,
                "splits": {split: {"path": str(bank), "bytes": bank.stat().st_size,
                    "sha256": SCRIPT.file_sha256(bank), "clips": [{"motion_id": motion_id,
                      "source_path": f"source/{split}/walking.motion", "frames": 40, "dt": 1/30}]}
                           for split, motion_id in (("train", 1), ("validation", 2), ("test", 3))}}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    return checkpoint, path


def test_preflight_is_header_metadata_only_excludes_train_and_rejects_overwrite(tmp_path):
    checkpoint, manifest = fixture_resources(tmp_path)
    args = SCRIPT.parse_args(["--checkpoint", str(checkpoint), "--manifest", str(manifest),
                              "--output", str(tmp_path/"new")])
    report = SCRIPT.preflight(args)
    assert report["selected_clips"][0]["motion_id"] == 2 and not report["torch_imported"] and not report["sdk_started"]
    args.motion_ids = [1]
    with pytest.raises(ValueError, match="absent"):
        SCRIPT.preflight(args)
    args.motion_ids = None
    args.output.mkdir()
    (args.output/"history.json").write_text("preserved")
    with pytest.raises(ValueError, match="historical"):
        SCRIPT.preflight(args)


def test_segment_aliases_do_not_hide_parent_split_leakage_before_sdk(tmp_path):
    checkpoint, manifest = fixture_resources(tmp_path)
    document = json.loads(manifest.read_text())
    document["splits"]["train"]["clips"][0].update(
        source_path="source/validation/walking.motion#frames=0:40",
        parent_source_path="source/validation/walking.motion")
    manifest.write_text(json.dumps(document))
    args = SCRIPT.parse_args(["--checkpoint", str(checkpoint), "--manifest", str(manifest),
                              "--output", str(tmp_path/"new")])
    with pytest.raises(ValueError, match="leaks parent"):
        SCRIPT.preflight(args)


def test_checkpoint_training_manifest_can_differ_but_must_be_bound_and_parent_disjoint(tmp_path):
    _, heldout = fixture_resources(tmp_path)
    training = tmp_path/"new_curated_training.json"
    document = json.loads(heldout.read_text())
    document["splits"]["train"]["clips"][0]["source_path"] = "source/newtrain/walking.motion#frames=100:140"
    document["splits"]["validation"]["clips"][0]["source_path"] = "source/newval/walking.motion"
    training.write_text(json.dumps(document))
    provenance = {"dataset": {"split": "train", "manifest": str(training),
        "manifest_sha256": SCRIPT.file_sha256(training), "source_sha256": document["splits"]["train"]["sha256"]}}
    gate = SCRIPT.verify_actor_training_split(provenance, heldout, "validation")
    assert gate["overlap_count"] == 0 and gate["training_manifest"] != gate["heldout_manifest"]
    document["splits"]["train"]["clips"][0]["source_path"] = "source\\validation\\walking.motion#frames=100:140"
    training.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changed"):
        SCRIPT.verify_actor_training_split(provenance, heldout, "validation")
    provenance["dataset"]["manifest_sha256"] = SCRIPT.file_sha256(training)
    with pytest.raises(ValueError, match="held-out parent"):
        SCRIPT.verify_actor_training_split(provenance, heldout, "validation")
    with pytest.raises(ValueError, match="actual training dataset"):
        SCRIPT.verify_actor_training_split({}, heldout, "validation")


def test_cli_help_and_preflight_never_import_torch_or_sdk_in_fresh_process(tmp_path):
    checkpoint, manifest = fixture_resources(tmp_path)
    command = [sys.executable, "-c", "import runpy,sys; m=runpy.run_path(sys.argv[1]); r=m['main'](sys.argv[2:]); assert r==0; assert 'torch' not in sys.modules; assert 'isaaclab.app' not in sys.modules",
               str(ROOT/"scripts/benchmark_paper_motion_tracking.py"), "--checkpoint", str(checkpoint),
               "--manifest", str(manifest), "--output", str(tmp_path/"new"), "--preflight-only"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["sdk_started"] is False
    help_result = subprocess.run([sys.executable, str(ROOT/"scripts/benchmark_paper_motion_tracking.py"), "--help"], capture_output=True, text=True, check=False)
    assert help_result.returncode == 0 and "--preflight-only" in help_result.stdout


def test_deterministic_environment_gate_is_read_only_and_opt_in(tmp_path, monkeypatch):
    checkpoint, manifest = fixture_resources(tmp_path)
    base = ["--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(tmp_path/"new")]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    assert SCRIPT.parse_args(base).deterministic_evaluation is False
    with pytest.raises(SystemExit):
        SCRIPT.parse_args(base+["--deterministic-evaluation"])
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    assert SCRIPT.parse_args(base+["--deterministic-evaluation"]).deterministic_evaluation
    import os
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_aggregate_uses_only_completed_sequences_for_success_and_weights_errors_by_frames():
    rows = [dict(in_full_sequence_success_denominator=True, observed_metric_steps=30,
                 tracking_success=True, tracking_failed=False, sequence_had_fall=False,
                 fall_onset_events=0, mean_body_position_error_m=.1, mean_body_rotation_error_rad=.2),
            dict(in_full_sequence_success_denominator=False, observed_metric_steps=10,
                 tracking_success=None, tracking_failed=True, sequence_had_fall=True,
                 fall_onset_events=1, mean_body_position_error_m=.9, mean_body_rotation_error_rad=1.)]
    summary = SCRIPT.aggregate_sequences(rows)
    assert summary["complete_success_denominator"] == 1 and summary["tracking_success_fraction"] == 1
    assert summary["sequences_with_fall"] == 1 and summary["frame_weighted_mean_body_position_error_m"] == pytest.approx(.3)


def test_real_cpu_checkpoint_loader_freezes_model_moments_and_rejects_nonfinite_weights(tmp_path):
    from CrowdSim.humanoid.training import PaperAMPTrainer, TrainingConfig

    config = TrainingConfig(policy_layers=(8, 4), task_layers=(8, 4), discriminator_layers=(8, 4))
    trainer = PaperAMPTrainer(config, device="cpu")
    path = tmp_path/"small_complete.pt"
    trainer.save(path)
    actor = SCRIPT.load_evaluation_actor(path, device="cpu")
    counts = {name: float(getattr(actor, name).count) for name in
              ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")}
    result = actor.act(torch.zeros(2, 357), torch.zeros(2, 1044), deterministic=True)
    assert result["applied_actions"].shape == (2, 69)
    assert torch.count_nonzero(result["applied_actions"][:, actor.action_mask == 0]) == 0
    assert not actor.model.training and not any(parameter.requires_grad for parameter in actor.model.parameters())
    assert all(float(getattr(actor, name).count) == count for name, count in counts.items())
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["model"]["actor.0.weight"][0, 0] = float("inf")
    torch.save(state, path)
    with pytest.raises(ValueError, match="nonfinite model weights"):
        SCRIPT.load_evaluation_actor(path, device="cpu")


@pytest.mark.parametrize("frame", ["legacy_stored", "child_joint"])
def test_corpus_frame_follows_actual_training_config_and_is_recorded(frame, monkeypatch):
    import CrowdSim.humanoid.amass_corpus as corpus_module
    from CrowdSim.humanoid.training import TrainingConfig

    config = TrainingConfig(expert_dof_velocity_frame=frame)
    config.validate()
    calls = []

    class CorpusBoundary:
        def __init__(self, manifest, split, *, device, seed, dof_velocity_frame):
            calls.append((manifest, split, device, seed, dof_velocity_frame))

        def provenance(self):
            return {"dof_velocity": {"requested_mode": frame,
                    "output_frame": "parent_joint" if frame == "legacy_stored" else "child_joint"},
                    "scope": "CPU constructor-boundary test"}

    monkeypatch.setattr(corpus_module, "AMASSCorpus", CorpusBoundary)
    _, contract = SCRIPT.create_tracking_corpus(config, "bound.json", "validation", device="cpu", seed=7)
    assert calls == [("bound.json", "validation", "cpu", 7, frame)]
    for key in ("reference_dof_velocity_frame", "initial_reset_dof_velocity_frame", "amp_reference_history_dof_velocity_frame"):
        assert contract[key] == ("parent_joint" if frame == "legacy_stored" else "child_joint")
    assert contract["expert_dof_velocity_mode"] == frame
    assert contract["physical_native_dof_velocity_frame"] == "child_joint"
    assert contract["body_angular_velocity_frame"] == "world" and contract["poses_position_rotation_metric_changed"] is False
    assert TrainingConfig().expert_dof_velocity_frame == "legacy_stored"
    with pytest.raises(ValueError, match="unsupported"):
        SCRIPT.create_tracking_corpus(SimpleNamespace(expert_dof_velocity_frame="guess"), "bound.json", "test", device="cpu", seed=7)


def synthetic_corpus(frames=42):
    pos = torch.zeros(frames, 24, 3)
    pos[..., 2] = 1
    pos[..., 0] = torch.arange(frames)[:, None]/30
    rot = torch.zeros(frames, 24, 4)
    rot[..., 3] = 1
    vel = torch.zeros_like(pos)
    vel[..., 0] = 1
    data = {"gts": pos, "grs": rot, "gvs": vel, "gavs": torch.zeros_like(pos),
            "dps": torch.zeros(frames, 69), "dvs": torch.zeros(frames, 69),
            "length_starts": torch.tensor([0]), "motion_num_frames": torch.tensor([frames]),
            "motion_dt": torch.tensor([1/30]), "motion_files": ["source/validation/walking.motion"]}

    def state_at(indices):
        result = {name: data[key][indices].clone() for name, key in (
            ("rigid_body_pos", "gts"), ("rigid_body_rot", "grs"), ("rigid_body_vel", "gvs"),
            ("rigid_body_ang_vel", "gavs"), ("dof_pos", "dps"), ("dof_vel", "dvs"))}
        result.update(root_pos=result["rigid_body_pos"][:, 0], root_rot=result["rigid_body_rot"][:, 0],
                      root_vel=result["rigid_body_vel"][:, 0], root_ang_vel=result["rigid_body_ang_vel"][:, 0])
        return SimpleNamespace(**result)
    return SimpleNamespace(data=data, state_at=state_at)


@pytest.mark.parametrize("invalid_actor_actions", [False, True])
def test_cpu_runner_contract_one_initial_reset_no_fall_teleport_and_exact_poststep_phase(tmp_path, invalid_actor_actions):
    from CrowdSim.humanoid.amp.observations import PacerSMPLMapping

    corpus = synthetic_corpus()
    world = SimpleNamespace(device=torch.device("cpu"), origins=torch.tensor([[6., 6.]]),
        age=torch.zeros(1, dtype=torch.long), episode_ids=torch.zeros(1, dtype=torch.long),
        amp_history=torch.zeros(1, 10, 195), amp_history_episode_ids=torch.zeros(1, 10, dtype=torch.long),
        reference_ids=torch.zeros(1, dtype=torch.long), planar_history=torch.zeros(1, 91, 5),
        history_available=torch.zeros(1, 91, dtype=torch.bool), mapping=PacerSMPLMapping())
    calls, commands = [], []
    world.ground_height_at = lambda xy: torch.zeros(xy.shape[:-1])

    def reset_envs(reset, env_ids):
        calls.append(reset)
        state = corpus.state_at(torch.tensor([9]))
        translation = reset.root_pos-state.root_pos
        state.rigid_body_pos += translation[:, None]
        state.root_pos = state.rigid_body_pos[:, 0]
        world.current = state
        world.steps = 0

    world.simulator = SimpleNamespace(reset_envs=reset_envs, is_simulation_running=lambda: True)
    world.state = lambda: world.current
    world.planar = lambda state: torch.cat((state.root_pos[:, :2], state.root_vel[:, :2], torch.zeros(1, 1)), -1)

    def observations(command, state):
        commands.append(command.clone())
        return torch.zeros(1, 357), torch.zeros(1, 1044)

    def step(actions):
        world.steps += 1
        world.current.rigid_body_pos[..., 0] += 1/30
        if world.steps == 5:
            world.current.rigid_body_pos[..., 1] += .8
        world.current.root_pos = world.current.rigid_body_pos[:, 0]
        return world.current, torch.tensor([world.steps >= 5]), torch.tensor([False])

    world.observations, world.step = observations, step
    actor = SimpleNamespace(act=lambda *obs, deterministic: {
        "actions": torch.full((1, 69), float("inf") if invalid_actor_actions else .2),
        "applied_actions": torch.full((1, 69), .2)})
    record = {"motion_id": 0, "source_path": "source/validation/walking.motion", "frames": 42, "dt": 1/30,
              "tracking_window": tracking_window(42, 1/30).as_dict()}
    if invalid_actor_actions:
        with pytest.raises(FloatingPointError, match="before physics step"):
            SCRIPT.run_reference_clip(world, SimpleNamespace(is_running=lambda: True), corpus, actor,
                                       record, tmp_path, .5)
        assert len(calls) == 1 and world.steps == 0
        return
    report = SCRIPT.run_reference_clip(world, SimpleNamespace(is_running=lambda: True), corpus, actor,
                                       record, tmp_path, .5)
    assert len(calls) == 1 and world.steps == 32
    assert report["first_tracking_failure_capture_frame"] == 5 and report["fall_onset_events"] == 1
    assert report["tracking_success"] is False and report["complete_clip"]
    with np.load(report["capture"]) as saved:
        assert saved["reference_frame"].tolist() == list(range(9, 42))
        assert saved["native_reset_frame"].sum() == 1 and saved["metric_frame"].sum() == 32
        assert saved["waypoint_query_phase_frame"].tolist() == [9]+list(range(9, 41))
        np.testing.assert_allclose(saved["mean_body_position_error_m"][1:5], 0, atol=1e-6)
        np.testing.assert_allclose(saved["mean_body_position_error_m"][5:], .8, atol=1e-6)
    assert all(command.shape == (1, 10, 2) for command in commands)
    assert not torch.cuda.is_initialized()
