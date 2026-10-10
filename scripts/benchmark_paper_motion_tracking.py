#!/usr/bin/env python3
"""Closed-loop AMP full-body tracking of held-out AMASS, commanded by root XY.

One initial mocap reset per clip; subsequent actions come only from the fixed
trajectory-conditioned actor. No runtime pose following or kinematic teleport.
The author clip list, actor and tracking implementation remain unpublished.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--motion-ids", type=int, nargs="+")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--max-steps", type=int, help="Diagnostic cap: partial clips cannot earn sequence success")
    parser.add_argument("--minimum-evaluation-seconds", type=float, default=1.)
    parser.add_argument("--failure-threshold-m", type=float, default=.5)
    parser.add_argument("--maximum-terrain-extent-m", type=float, default=256.)
    parser.add_argument("--verify-package-hashes", action="store_true")
    parser.add_argument("--deterministic-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("max_clips", "max_steps", "cpu_threads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("minimum_evaluation_seconds", "failure_threshold_m", "maximum_terrain_extent_m"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.maximum_terrain_extent_m < 64:
        parser.error("Maximum terrain extent must be at least 64m")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("Seed must lie in [0,2**32)")
    if args.motion_ids is not None and (min(args.motion_ids) < 0 or len(set(args.motion_ids)) != len(args.motion_ids)):
        parser.error("Motion IDs must be unique nonnegative IDs in the held-out split")
    if args.checkpoint_sha256 is not None and (len(args.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256)):
        parser.error("Checkpoint SHA256 must be 64 lowercase hex characters")
    from CrowdSim.humanoid.application.configuration import validate_deterministic_evaluation_environment
    try:
        validate_deterministic_evaluation_environment(args.deterministic_evaluation)
    except ValueError as error:
        parser.error(str(error))
    for name in ("checkpoint", "manifest", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def preflight(args):
    """Local headers/manifest only: no Torch deserialization or SDK import."""
    from CrowdSim.humanoid.application.resources import inspect_resource, resolve_local_path
    from CrowdSim.humanoid.motion_tracking import tracking_window
    from CrowdSim.humanoid.style_evaluation import assert_parent_split_disjoint

    checkpoint = inspect_resource("AMP checkpoint", args.checkpoint, kind="torch",
                                  declared_sha256=args.checkpoint_sha256,
                                  recompute_sha256=bool(args.checkpoint_sha256))
    checkpoint_hash = file_sha256(args.checkpoint)
    document = json.loads(args.manifest.read_text())
    if document.get("schema") != "navisaaclab.amass_reconstruction.v1":
        raise ValueError("Unknown AMASS reconstruction manifest")
    splits = document.get("splits", {})
    if not {"train", "validation", "test"}.issubset(splits):
        raise ValueError("Train/validation/test identities must be explicit")
    parents = assert_parent_split_disjoint(document)
    if parents["validation"] & parents["test"]:
        raise ValueError("Original parent-source leakage between validation and test")
    entry = splits[args.split]
    clips = list(entry["clips"])
    ids = [clip["motion_id"] for clip in clips]
    if not clips or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Held-out split requires unique actual motion IDs")
    if args.motion_ids is not None:
        missing = set(args.motion_ids)-set(ids)
        if missing:
            raise ValueError(f"Requested IDs absent from {args.split}: {sorted(missing)}")
        by_id = {clip["motion_id"]: clip for clip in clips}
        clips = [by_id[i] for i in args.motion_ids]
    if args.max_clips is not None:
        clips = clips[:args.max_clips]
    records = []
    for clip in clips:
        window = tracking_window(clip["frames"], clip["dt"], minimum_seconds=args.minimum_evaluation_seconds,
                                 max_steps=args.max_steps)
        records.append({**clip, "tracking_window": window.as_dict()})
    if not any(record["tracking_window"]["eligible"] for record in records):
        raise ValueError("No selected clip has enough real post-history evaluation frames")
    package_path = resolve_local_path(entry["path"], relative_to=args.manifest.parent)
    package = inspect_resource("AMASS held-out package", package_path, kind="torch",
                               expected_bytes=entry["bytes"], declared_sha256=entry["sha256"],
                               recompute_sha256=args.verify_package_hashes)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Use a new empty output directory; historical captures cannot be overwritten")
    return {"status": "SDK-free preflight passed", "checkpoint": asdict(checkpoint),
            "checkpoint_sha256_recomputed": checkpoint_hash,
            "manifest": str(args.manifest), "manifest_sha256": file_sha256(args.manifest),
            "source_package": asdict(package), "split": args.split,
            "selected_clips": records, "author_clip_list": document.get("author_clip_list", False),
            "torch_imported": False, "sdk_started": False}


def verify_actor_training_split(provenance, heldout_manifest, heldout_split):
    """Bind the checkpoint's actual train manifest to parent-disjoint evaluation."""
    from CrowdSim.humanoid.application.resources import resolve_local_path
    from CrowdSim.humanoid.style_evaluation import assert_parent_split_disjoint

    dataset = provenance.get("dataset") if isinstance(provenance, dict) else None
    if not isinstance(dataset, dict) or dataset.get("split") != "train":
        raise ValueError("Fixed actor must declare its actual training dataset provenance")
    training_path = resolve_local_path(dataset.get("manifest"))
    if not training_path.is_file() or file_sha256(training_path) != dataset.get("manifest_sha256"):
        raise ValueError("Checkpoint training manifest missing or changed since training")
    if heldout_split not in ("validation", "test"):
        raise ValueError("Only explicit held-out reference splits are supported")
    training = json.loads(training_path.read_text())
    heldout = json.loads(Path(heldout_manifest).read_text())
    if training.get("schema") != "navisaaclab.amass_reconstruction.v1":
        raise ValueError("Unknown checkpoint training manifest schema")
    train_parents = assert_parent_split_disjoint(training)["train"]
    heldout_parents = assert_parent_split_disjoint(heldout)[heldout_split]
    overlap = train_parents & heldout_parents
    if overlap:
        raise ValueError(f"Checkpoint training leaks selected held-out parent sources: {sorted(overlap)[:3]}")
    if training["splits"]["train"].get("sha256") != dataset.get("source_sha256"):
        raise ValueError("Checkpoint training source SHA differs from its bound manifest")
    return {"status": "checkpoint training / entire reference split parent-disjoint",
            "training_manifest": str(training_path), "training_manifest_sha256": dataset["manifest_sha256"],
            "training_parent_sources": len(train_parents), "heldout_split": heldout_split,
            "heldout_manifest": str(Path(heldout_manifest).resolve()),
            "heldout_manifest_sha256": file_sha256(heldout_manifest),
            "heldout_parent_sources": len(heldout_parents), "overlap_count": 0,
            "training_package_sha256_recomputed": False}


def _reference_layout(corpus, records, maximum_extent):
    """Derive one explicit flat-floor translation large enough for all clips."""
    minimum, maximum = 0., 0.
    for record in records:
        if not record["tracking_window"]["eligible"]:
            continue
        motion_id = record["motion_id"]
        start = int(corpus.data["length_starts"][motion_id])
        if (int(corpus.data["motion_num_frames"][motion_id]) != record["frames"]
                or str(corpus.data["motion_files"][motion_id]).replace("\\", "/") != record["source_path"].replace("\\", "/")
                or abs(float(corpus.data["motion_dt"][motion_id])-record["dt"]) > 1e-7):
            raise ValueError(f"Selected actual source metadata changed: {motion_id}")
        initial = record["tracking_window"]["initial_frame"]
        root_xy = corpus.data["gts"][start+initial, 0, :2]
        relative = corpus.data["gts"][start+initial:start+record["frames"], :, :2]-root_xy
        if not relative.isfinite().all():
            raise ValueError("Nonfinite selected reference path")
        minimum = min(minimum, float(relative.min()))
        maximum = max(maximum, float(relative.max()))
    margin = math.ceil(max(6., 6.-minimum))
    extent = math.ceil(max(64., margin+maximum+6.)/4)*4
    if extent > maximum_extent:
        raise ValueError(f"Reference path needs {extent}m terrain, exceeds declared {maximum_extent}m bound")
    return {"spawn_margin_m": margin, "terrain_extent_m": extent,
            "source_relative_body_xy_min_m": minimum, "source_relative_body_xy_max_m": maximum,
            "method": "one scalar XY spawn offset, >=6m reference-body margin; flat unobstructed physical floor"}


def load_evaluation_actor(path, *, device):
    """Load the fixed model and moments; optimizer/replay stay unused on CPU."""
    import torch
    from CrowdSim.humanoid.training import PaperAMPTrainer

    trainer, _ = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
    if (trainer.config.self_dim, trainer.config.task_dim, trainer.config.history_frames) != (357, 1044, 10):
        raise ValueError("Tracking requires the raw PACER 357/1044 observation and 10-history contract")
    if not math.isclose(trainer.config.control_dt, 1/30, abs_tol=1e-9) or not math.isclose(trainer.config.target_horizon_s, 5., abs_tol=1e-9):
        raise ValueError("Tracking requires 30Hz control and a 5-second XY command horizon")
    if not trainer.config.freeze_toes:
        raise ValueError("This AMPWorld physical decoder freezes toe actions; checkpoint freeze_toes must agree")
    if not all(torch.isfinite(value).all() for value in trainer.model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model weights")
    trainer.device = torch.device(device)
    trainer.model.to(trainer.device).eval().requires_grad_(False)
    trainer.action_mask = trainer.action_mask.to(trainer.device)
    for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
        getattr(trainer, name).to(trainer.device).eval()
    return trainer


def create_tracking_corpus(actor_config, manifest, split, *, device, seed):
    """Use the actor's declared reference velocity frame for reset and history."""
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus

    frame = actor_config.expert_dof_velocity_frame
    if frame not in ("legacy_stored", "child_joint"):
        raise ValueError("Actor declares an unsupported expert DOF velocity frame")
    corpus = AMASSCorpus(manifest, split, device=device, seed=seed, dof_velocity_frame=frame)
    provenance = corpus.provenance()
    velocity = provenance.get("dof_velocity", {})
    output_frame = velocity.get("output_frame")
    if (velocity.get("requested_mode") != frame or output_frame not in ("parent_joint", "child_joint")
            or (frame == "child_joint" and output_frame != "child_joint")):
        raise ValueError("Corpus does not verify the actor's requested DOF velocity coordinates")
    contract = {"source": "loaded actor TrainingConfig.expert_dof_velocity_frame",
                "expert_dof_velocity_mode": frame,
                "reference_dof_velocity_frame": output_frame,
                "initial_reset_dof_velocity_frame": output_frame,
                "amp_reference_history_dof_velocity_frame": output_frame,
                "physical_native_dof_velocity_frame": "child_joint",
                "body_angular_velocity_frame": "world",
                "poses_position_rotation_metric_changed": False,
                "reference_velocity_mode_is_part_of_evaluation_protocol": True,
                "cross_profile_reference_velocity_may_differ": True,
                "corpus_provenance": provenance}
    return corpus, contract


def reset_reference_clip(world, corpus, record):
    """Exactly one physical reset; history contains actual current + past mocap."""
    import torch
    from CrowdSim.humanoid.amp import build_amp_observation
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

    initial = record["tracking_window"]["initial_frame"]
    absolute = int(corpus.data["length_starts"][record["motion_id"]])+initial
    reference = corpus.state_at(torch.tensor([absolute], dtype=torch.long))
    root = reference.root_pos.clone()
    root[:, :2] = world.origins
    root[:, 2] += world.ground_height_at(root[:, :2])
    translation = root-reference.root_pos
    world.simulator.reset_envs(ResetState(root_pos=root, root_rot=reference.root_rot.clone(),
        root_vel=reference.root_vel.clone(), root_ang_vel=reference.root_ang_vel.clone(),
        dof_pos=reference.dof_pos.clone(), dof_vel=reference.dof_vel.clone(),
        state_conversion=StateConversion.COMMON), env_ids=torch.zeros(1, device=world.device, dtype=torch.long))
    actual = world.state()
    history = corpus.state_at(absolute-torch.arange(10))
    world.amp_history.copy_(build_amp_observation(history).reshape(1, 10, 195))
    world.amp_history[:, 0] = build_amp_observation(actual, mapping=world.mapping)
    world.age.zero_()
    world.episode_ids.add_(1)
    world.amp_history_episode_ids.copy_(world.episode_ids[:, None].expand_as(world.amp_history_episode_ids))
    world.reference_ids.fill_(record["motion_id"])
    world.planar_history.copy_(world.planar(actual)[:, None].expand_as(world.planar_history))
    world.history_available.zero_()
    world.history_available[:, -1] = True
    return actual, translation


def run_reference_clip(world, app, corpus, actor, record, output, threshold_m):
    import numpy as np
    import torch
    from CrowdSim.humanoid.motion_tracking import (
        body_tracking_errors, future_reference_frames, summarize_tracking_sequence,
    )

    window = record["tracking_window"]
    actual, translation = reset_reference_clip(world, corpus, record)
    absolute_start = int(corpus.data["length_starts"][record["motion_id"]])
    capture = {}
    pose_fields = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
                   "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel")

    def append_frame(state, frame, command, query_phase, held, actions, applied, *, measured, fall, timeout):
        reference = corpus.state_at(torch.tensor([absolute_start+frame], dtype=torch.long))
        values = {}
        for name in pose_fields:
            value = getattr(state, name).detach().clone()
            target = getattr(reference, name).detach().clone()
            if name in {"root_pos", "rigid_body_pos"}:
                target += translation if name == "root_pos" else translation[:, None]
            if not torch.isfinite(value).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f"Nonfinite physical/reference {name}")
            values[name] = value.cpu().numpy()
            values["reference_"+name] = target.cpu().numpy()
        values.update(reference_frame=np.asarray(frame), reference_time_s=np.asarray(frame/30),
                      metric_frame=np.asarray(measured), native_reset_frame=np.asarray(not measured),
                      fall_mask=np.asarray(fall), time_limit_mask=np.asarray(timeout),
                      episode_id=world.episode_ids.detach().cpu().numpy().copy(),
                      actions=actions.detach().cpu().numpy().copy(),
                      applied_actions=applied.detach().cpu().numpy().copy(),
                      action_valid=np.asarray(measured), trajectory_xy=command.detach().cpu().numpy().copy(),
                      waypoint_query_phase_frame=np.asarray(query_phase), waypoint_held_at_clip_end=held.copy(),
                      amp_history=world.amp_history.detach().cpu().numpy().copy())
        for name, value in values.items():
            capture.setdefault(name, []).append(value)

    def trajectory(phase):
        frames, held = future_reference_frames(phase, record["frames"])
        reference_xy = corpus.data["gts"][absolute_start+torch.from_numpy(frames), 0, :2].to(world.device)
        return (reference_xy+translation[0, :2])[None], held

    first = window["initial_frame"]
    command, held = trajectory(first)
    zero = torch.zeros(1, 69, device=world.device)
    append_frame(actual, first, command, first, held, zero, zero, measured=False, fall=False, timeout=False)
    stopped = False
    with torch.no_grad():
        for step in range(1, window["requested_steps"]+1):
            if not app.is_running() or not world.simulator.is_simulation_running():
                stopped = True
                break
            query_phase = first+step-1
            command, held = trajectory(query_phase)
            observations = world.observations(command, actual)
            prediction = actor.act(*observations, deterministic=True)
            if any(prediction[name].shape != (1, 69) or not torch.isfinite(prediction[name]).all()
                   for name in ("actions", "applied_actions")):
                raise FloatingPointError("Actor emitted invalid actions before physics step")
            actual, terminated, truncated = world.step(prediction["applied_actions"])
            # No fall/time-limit reset within this reference sequence.
            append_frame(actual, first+step, command, query_phase, held,
                         prediction["actions"], prediction["applied_actions"],
                         measured=True, fall=bool(terminated[0]), timeout=bool(truncated[0]))
    arrays = {name: np.stack(values) for name, values in capture.items()}
    errors = body_tracking_errors(arrays["rigid_body_pos"][:, 0], arrays["rigid_body_rot"][:, 0],
                                 arrays["reference_rigid_body_pos"][:, 0], arrays["reference_rigid_body_rot"][:, 0],
                                 actual_body_names=world.mapping.body_names)
    summary = summarize_tracking_sequence(errors, metric_mask=arrays["metric_frame"], fall_mask=arrays["fall_mask"],
        expected_steps=window["available_steps"], eligible=window["eligible"],
        full_clip_requested=window["full_clip_requested"], threshold_m=threshold_m)
    arrays.update(errors)
    path = output/f"motion_{record['motion_id']:06d}.npz"
    np.savez_compressed(path, **arrays)
    return {**summary, "motion_id": record["motion_id"], "source_path": record["source_path"],
            "tracking_window": window, "capture": str(path), "capture_sha256": file_sha256(path),
            "initial_translation_xyz_m": translation[0].cpu().tolist(),
            "initial_mean_body_position_error_m": float(errors["mean_body_position_error_m"][0]),
            "initial_mean_body_rotation_error_rad": float(errors["mean_body_rotation_error_rad"][0]),
            "held_endpoint_command_samples": int(arrays["waypoint_held_at_clip_end"][1:].sum()),
            "all_command_samples": summary["observed_metric_steps"]*10,
            "app_or_simulation_stopped": stopped}


def aggregate_sequences(sequences):
    complete = [row for row in sequences if row["in_full_sequence_success_denominator"]]
    observations = sum(row["observed_metric_steps"] for row in sequences)
    return {"measured_sequences": len(sequences), "complete_success_denominator": len(complete),
            "tracking_successes": sum(row["tracking_success"] is True for row in complete),
            "tracking_success_fraction": (sum(row["tracking_success"] is True for row in complete)/len(complete) if complete else None),
            "sequences_with_observed_tracking_failure": sum(row["tracking_failed"] for row in sequences),
            "sequences_with_fall": sum(row["sequence_had_fall"] for row in sequences),
            "fall_onset_events": sum(row["fall_onset_events"] for row in sequences),
            "observed_metric_steps": observations,
            "frame_weighted_mean_body_position_error_m": (sum(row["mean_body_position_error_m"]*row["observed_metric_steps"] for row in sequences if row["observed_metric_steps"])/observations if observations else None),
            "frame_weighted_mean_body_rotation_error_rad": (sum(row["mean_body_rotation_error_rad"]*row["observed_metric_steps"] for row in sequences if row["observed_metric_steps"])/observations if observations else None)}


def main(argv=None):
    args = parse_args(argv)
    resources = preflight(args)
    if args.preflight_only:
        print(json.dumps(resources, indent=2))
        return 0
    from CrowdSim.humanoid.application.configuration import prepare_deterministic_evaluation_environment
    prepare_deterministic_evaluation_environment(args.deterministic_evaluation)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "preflight": resources, "sequences": [], "author_experiment_reproduced": False,
              "protocol": {"controller": "fixed AMP deterministic actor mean, clamped by original physical action decoder",
                  "commands": "ten true reference root XY positions at +.5..5s; clip-tail hold is command-only",
                  "reference_3d_poses_enter_actor_or_actions": False, "runtime_teleports": 0,
                  "initial_reset": "once per clip at source frame9; source pose and root/DOF velocity; constant initial translation",
                  "initial_frame_excluded_from_errors": True, "fall_reset_within_sequence": False,
                  "metric_phase": "post-physics at 30Hz vs same reference frame; no pelvis/yaw/frame best-match alignment",
                  "reference_body_order_assumption": "current processed AMASS neutral-SMPL PACER canonical names; package lacks per-frame body name tags",
                  "threshold_m": args.failure_threshold_m,
                  "paper_reported_reference_targets": {"mean_position_m": .218, "mean_rotation_rad": .343, "locomotion_sequence_success_fraction": .646},
                  "author_target_comparison_is_not_a_replication_gate": True}}
    app = world = None
    started = time.monotonic()
    try:
        from protomotions.utils.simulator_imports import import_simulator_before_torch
        launcher = import_simulator_before_torch("isaaclab")
        app = launcher({"headless": True, "device": "cuda:0", "enable_cameras": False,
                        "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
        import torch
        from CrowdSim.humanoid.amp_world import AMPWorld
        from CrowdSim.humanoid.application.configuration import configure_deterministic_evaluation_torch, runtime_numerics
        configure_deterministic_evaluation_torch(torch, args.deterministic_evaluation)
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        actor = load_evaluation_actor(args.checkpoint, device="cuda:0")
        if file_sha256(args.checkpoint) != resources["checkpoint_sha256_recomputed"]:
            raise ValueError("Checkpoint changed between preflight and model load; select an immutable step snapshot")
        report["actor_training_heldout_parent_gate"] = verify_actor_training_split(actor.provenance, args.manifest, args.split)
        if report["actor_training_heldout_parent_gate"]["heldout_manifest_sha256"] != resources["manifest_sha256"]:
            raise ValueError("Held-out manifest changed after preflight")
        corpus, velocity_contract = create_tracking_corpus(actor.config, args.manifest, args.split,
                                                          device="cuda:0", seed=args.seed)
        report["reference_velocity_contract"] = velocity_contract
        report["protocol"]["expert_dof_velocity_frame"] = actor.config.expert_dof_velocity_frame
        layout = _reference_layout(corpus, resources["selected_clips"], args.maximum_terrain_extent_m)
        records = [record for record in resources["selected_clips"] if record["tracking_window"]["eligible"]]
        world = AMPWorld(app, corpus, num_envs=1, shared_world=False,
                         episode_steps=max(record["tracking_window"]["available_steps"] for record in records)+1,
                         terrain_extent_m=layout["terrain_extent_m"], spawn_margin_m=layout["spawn_margin_m"])
        torch.manual_seed(args.seed)
        report.update(physics=world.physics_manifest(), layout=layout, training_config=asdict(actor.config),
                      controller_provenance=actor.provenance,
                      runtime_numerics=runtime_numerics(torch, deterministic_evaluation_requested=args.deterministic_evaluation),
                      excluded_short_clips=[row for row in resources["selected_clips"] if not row["tracking_window"]["eligible"]])
        source_paths = (Path(__file__).resolve(), ROOT/"CrowdSim/humanoid/motion_tracking.py",
                        ROOT/"CrowdSim/humanoid/amp_world.py", ROOT/"CrowdSim/humanoid/training/trainer.py",
                        ROOT/"CrowdSim/humanoid/amp/observations.py", ROOT/"protomotions/simulator/isaaclab/simulator.py")
        report["selected_source_sha256"] = {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths}
        captures = args.output/"captures"
        captures.mkdir()
        for record in records:
            summary = run_reference_clip(world, app, corpus, actor, record, captures, args.failure_threshold_m)
            report["sequences"].append(summary)
            print(json.dumps({key: summary[key] for key in ("motion_id", "observed_metric_steps", "tracking_success", "sequence_had_fall", "mean_body_position_error_m", "mean_body_rotation_error_rad")}), flush=True)
            if summary["app_or_simulation_stopped"]:
                raise RuntimeError("Simulator/app stopped before requested reference measurement completed")
        report.update(status="passed", aggregate=aggregate_sequences(report["sequences"]), elapsed_s=time.monotonic()-started)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      aggregate=aggregate_sequences(report["sequences"]), elapsed_s=time.monotonic()-started)
        traceback.print_exc()
    finally:
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"status": report["status"], "aggregate": report.get("aggregate"), "error": report.get("error")}), flush=True)
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1
            world.close()
        elif app is not None:
            app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
