#!/usr/bin/env python3
"""Evaluate a completed shared Social AMP pilot on all fixed validation22 clips.

Reuses the sealed reference timing, physical decoder and tracking metrics.
The isolated evaluator first requires an actual training exit receipt and CPU
audit. --lifecycle-smoke uses zero actions and makes no policy quality claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import IsolatedTrackingLifecycle, record  # noqa: E402
from CrowdSim.humanoid.runtime import social_observation_producer  # noqa: E402
from CrowdSim.humanoid.runtime import social_popart_training  # noqa: E402
from CrowdSim.humanoid.runtime.popart_checkpoint_family import admit_binding  # noqa: E402
from social_navigation.artifact_identity import artifact_path, same_artifact, same_recorded_value  # noqa: E402

PARENT = ROOT / "docs/style_improvement_20261004/social_tracking_evaluation_staging_v1/evaluate_social_tracking.py"
PARENT_SHA = "79fd24b23bb3e4883e24c03532f6af989970140d9e4dfbca156da7dbbe1a40df"


def load_reference_functions():
    identity = record(PARENT)
    if identity["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
        raise ValueError("Sealed full22 evaluation parent changed")
    spec = importlib.util.spec_from_file_location("_shared_tracking_reference", PARENT)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    from CrowdSim.humanoid.runtime.evaluation_source_migration import migrate_pins
    dependency_migration = migrate_pins(parent)
    bindings, sources = parent._sources()
    derived, derivation = parent.derive(sources["original"].decode())
    namespace = {"__file__": str(PARENT), "__name__": "_shared_tracking_functions",
        "_social_root": str(ROOT), "_social_validate_args": parent._validate_args,
        "_social_load_actor": parent._load_actor, "_social_observe": parent._observe,
        "_social_capture_values": parent._capture_values,
        "_social_partial_capture": lambda *args: None}
    exec(compile(derived, str(PARENT) + ":full22_derived", "exec"), namespace)
    return parent, namespace, {"parent": identity, "sources": bindings, "derivation": derivation,
                               "entry": record(Path(__file__)), "dependency_migration": dependency_migration}


def admit_training(receipt_path):
    """Metadata admission only. The receipt records the external process exit."""
    receipt_path = artifact_path(receipt_path, root=ROOT)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "shared_social.actual_training_execution.v1" or (type(receipt.get("terminal_exit_code")) is not int or receipt["terminal_exit_code"] != 0):
        raise ValueError("Require an actual successful training-process exit receipt")
    run = artifact_path(receipt["run_directory"], root=ROOT).resolve()
    result_path = run / "result.json"
    if not same_artifact(record(result_path), receipt["result_identity"], root=ROOT):
        raise ValueError("Training terminal result changed")
    result = json.loads(result_path.read_text())
    if (result["status"] != "passed_local_training" or result["cleanup_succeeded"] is not True
            or result["cleanup_errors"] != [] or result["source_profile_before_after_equal"] is not True
            or type(result["completed_updates"]) is not int or result["completed_updates"] < 1
            or type(result["actual_training_transitions"]) is not int or result["actual_training_transitions"] < 1):
        raise ValueError("Training did not complete with owned cleanup")
    audit_path = artifact_path(receipt["audit_identity"]["path"], root=ROOT)
    if not same_artifact(record(audit_path), receipt["audit_identity"], root=ROOT):
        raise ValueError("Training CPU audit changed")
    audit = json.loads(audit_path.read_text())
    if (audit["schema"] != "shared_social.completed_training_CPU_audit.v1" or audit["status"] != "passed"
            or audit["CUDA_initialized"] is not False
            or type(audit["updates"]) is not int or type(audit["physical_transitions"]) is not int
            or audit["updates"] != result["completed_updates"]
            or audit["physical_transitions"] != result["actual_training_transitions"]
            or not same_artifact(audit["checkpoint"], receipt["checkpoint_identity"], root=ROOT)):
        raise ValueError("Training CPU audit/result/selected checkpoint disagree")
    expected = run / f"step_{result['completed_updates']:06d}.pt"
    checkpoint = artifact_path(audit["checkpoint"]["path"], root=ROOT)
    if checkpoint != expected or checkpoint.is_symlink() or checkpoint.stat().st_mode & 0o222:
        raise ValueError("Select the independent readonly terminal checkpoint of the audited run")
    if result["completed_updates"] != result["profile"]["runtime"]["updates"]:
        raise ValueError("Training budget is incomplete")
    canonical = result.get("canonical_entry")
    strategy = "legacy" if canonical is None else canonical.get("value_normalization")
    numeric = audit.get("explicit_value_normalization")
    if strategy == "popart":
        binding = admit_binding(canonical.get("numeric_migration"))
        from CrowdSim.humanoid.application.historical_recovery import source_record
        entry_record = record(artifact_path(canonical["entry"]["path"], root=ROOT))
        if (binding["checkpoint_format"] == social_popart_training.LEGACY_FORMAT
                and not same_artifact(entry_record, canonical["entry"], root=ROOT)):
            entry_record = source_record(canonical["entry"]["path"])
        legacy = binding["checkpoint_format"] == social_popart_training.LEGACY_FORMAT
        physical_binding = (canonical.get("World_collector_expert_physics_changed") is False if legacy else
            canonical.get("World_collector_expert_algorithms_changed") is False
            and canonical.get("grouped_runtime_migration", {}).get("schema") == "shared_social.grouped_source_migration.sim61.v2")
        if (not same_recorded_value(numeric, binding, root=ROOT)
                or not same_artifact(entry_record, canonical["entry"], root=ROOT)
                or not physical_binding):
            raise ValueError("Pop-Art training entry, numeric source and CPU audit disagree")
    elif strategy != "legacy" or numeric is not None or canonical is not None and canonical.get("numeric_migration") is not None:
        raise ValueError("Unknown or inconsistent value-normalization checkpoint family")
    return checkpoint, audit["checkpoint"]["sha256"], {"receipt": record(receipt_path),
        "result": receipt["result_identity"], "audit": receipt["audit_identity"],
        "checkpoint": audit["checkpoint"], "terminal_exit_code": 0,
        "value_normalization": strategy, "numeric_migration": numeric}


def load_admitted_actor(checkpoint, *, device, expected_sha256, admission, legacy_loader):
    """Explicit family dispatch after independent terminal result/CPU admission."""
    if admission["value_normalization"] == "popart":
        from CrowdSim.humanoid.runtime.social_popart_inference import FrozenSocialPopArtCheckpointInference
        return FrozenSocialPopArtCheckpointInference.from_checkpoint(
            checkpoint, device=device, expected_sha256=expected_sha256)
    if admission["value_normalization"] != "legacy":
        raise ValueError("Unadmitted checkpoint family")
    return legacy_loader(checkpoint, device=device, expected_sha256=expected_sha256)


def create_producer(world, actor):
    """Original N1 producer contract with the explicit scripted-origin adapter."""
    import torch

    if world.num_envs != 1 or world.shared_world or world.scene_provider is not None or world.validation_scene is not None:
        raise ValueError("Full22 uses the original isolated N1 unobstructed flat-floor world")
    ids = torch.zeros(1, dtype=torch.int64, device=world.device)
    producer = social_observation_producer.RawSocialEvaluationObservations.from_world(
        world, ids, ids.clone(), grid_center="root", grid_heading="root", height_reference="static_ground",
        include_ego=False, terrain_sampler=world.height_at,
        terrain_sampler_semantics="original AMPWorld.height_at absolute static flat-ground feature; not measured body surface")
    producer.require_policy_contract(actor.describe())
    return producer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-sources", action="store_true", help="Stdlib source/overlay checks only")
    parser.add_argument("--lifecycle-smoke", action="store_true")
    options = parser.parse_args(argv)
    parent, reference, binding = load_reference_functions()
    report = {"schema": "local.neutral_smpl.social_full22_tracking.v2", "status": "failed",
              "entry_sources": binding, "sequences": [], "author_experiment_reproduced": False,
              "observation_producer_migration": social_observation_producer.describe_migration()}
    lifecycle = IsolatedTrackingLifecycle(report)
    if options.check_sources:
        print(json.dumps({"status": "source_checks_passed", "NN_constructed": False, "SDK_started": False,
                          "entry": binding["entry"], "lifecycle": report["evaluation_lifecycle"]}))
        return 0
    if options.execution_receipt is None or options.output is None:
        parser.error("--execution-receipt and --output are required")
    checkpoint, sha, admission = admit_training(options.execution_receipt.resolve())
    args = reference["parse_args"](["--checkpoint", str(checkpoint), "--checkpoint-sha256", sha,
                                    "--output", str(options.output.resolve())])
    resources = reference["preflight"](args)
    rows = resources["selected_clips"]
    if (resources["manifest_sha256"] != parent.MANIFEST_SHA or len(rows) != 22
            or sum(row["tracking_window"]["available_steps"] for row in rows) != 11982
            or any(not row["tracking_window"]["eligible"] or not row["tracking_window"]["full_clip_requested"] for row in rows)):
        raise ValueError("Require the entire fixed parent-disjoint validation22 population")
    report.update(preflight=resources, training_execution_admission=admission,
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    if options.preflight_only:
        print(json.dumps({"status": "preflight_passed", "NN_constructed": False, "SDK_started": False,
                          "selected_clips": 22, "postphysics_controls": 11982, "training_admission": admission}))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    app = world = producer = None
    started = time.monotonic()
    try:
        if "torch" in sys.modules:
            raise RuntimeError("A fresh process must launch SDK before Torch")
        from protomotions.utils.simulator_imports import import_simulator_before_torch
        launcher = import_simulator_before_torch("isaaclab")
        app = launcher({"headless": True, "device": "cuda:0", "enable_cameras": False,
                        "fast_shutdown": False, "kit_args": lifecycle.kit_args}).app
        lifecycle.verify_loaded()
        import torch
        from CrowdSim.humanoid.amp_world import AMPWorld
        from CrowdSim.humanoid.application.configuration import runtime_numerics
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        actor = None if options.lifecycle_smoke else load_admitted_actor(
            checkpoint, device="cuda:0", expected_sha256=sha, admission=admission, legacy_loader=parent._load_actor)
        config = SimpleNamespace(expert_dof_velocity_frame="child_joint") if actor is None else actor.config
        corpus, velocity = reference["create_tracking_corpus"](config, args.manifest, "validation", device="cuda:0", seed=args.seed)
        layout = reference["_reference_layout"](corpus, rows, args.maximum_terrain_extent_m)
        constructor = getattr(lifecycle._sdk61, "construct_world", lambda factory, *a, **k: factory(*a, **k))
        world = constructor(AMPWorld, app, corpus, num_envs=1, shared_world=False,
            episode_steps=max(row["tracking_window"]["available_steps"] for row in rows) + 1,
            terrain_extent_m=layout["terrain_extent_m"], spawn_margin_m=layout["spawn_margin_m"])
        training_result = json.loads(Path(admission["result"]["path"]).read_text())
        if record(admission["result"]["path"]) != admission["result"]:
            raise RuntimeError("Training result changed before physical rig admission")
        usd = record(world.simulator._robot.cfg.spawn.usd_path)
        training_assets = training_result.get("actual_native_assets", [])
        if (len(training_assets) != 4 or any(asset.get("USD", {}).get("sha256") != usd["sha256"]
                or asset.get("USD", {}).get("bytes") != usd["bytes"] for asset in training_assets)):
            raise ValueError("Evaluation rig differs from the four actual training USD prototypes")
        report["evaluation_USD_training_match"] = {"USD": usd, "matched_training_prototypes": 4,
                                                  "evaluation_vectorized_assets": 1}
        report.update(physics=world.physics_manifest(), layout=layout, reference_velocity_contract=velocity,
                      runtime_numerics=runtime_numerics(torch, deterministic_evaluation_requested=False))
        if options.lifecycle_smoke:
            initial = world.simulator._sim.get_physics_step_count()
            for _ in range(8):
                state, _, _ = world.step(torch.zeros(1, 69, device=world.device))
                if not torch.isfinite(state.rigid_body_pos).all():
                    raise FloatingPointError("Nonfinite physical state in lifecycle smoke")
            final = world.simulator._sim.get_physics_step_count()
            if final - initial != 32:
                raise RuntimeError("Lifecycle smoke did not execute 8 controls / 32 physics substeps")
            report.update(status="passed_lifecycle_smoke", controls=8, physics_substeps=final - initial,
                          NN_constructed=False, quality_evaluated=False)
        else:
            report["actor_training_heldout_parent_gate"] = reference["verify_actor_training_split"](actor.provenance, args.manifest, "validation")
            producer = create_producer(world, actor)
            report.update(training_config=config.to_document(), controller_provenance=actor.provenance,
                          frozen_social_actor=actor.describe(), social_observation_producer=producer.describe())
            report["protocol"] = {"controller": "fixed Social AMP deterministic actor mean",
                "commands": "ten true reference root XY positions at +.5..5s; clip-tail hold is command-only",
                "reference_3d_poses_enter_actor_or_actions": False, "runtime_teleports": 0,
                "initial_frame_excluded_from_errors": True, "fall_reset_within_sequence": False,
                "metric_phase": "post-physics at 30Hz vs same reference frame; no pelvis/yaw/frame best-match alignment",
                "threshold_m": .5, "reference_selection": "all22 validation;11982 postphysics controls",
                "peers_added_to_single_actor_evaluation": 0, "geometry_proxy_is_collision_truth": False}
            partials = report["unexpected_failure_partial_captures"] = []
            reference["_social_partial_capture"] = lambda *values: partials.append(parent._save_partial(*values))
            captures = args.output / "captures"
            captures.mkdir()
            for row in rows:
                summary = reference["run_reference_clip"](world, app, corpus, actor, row, captures, .5, producer)
                report["sequences"].append(summary)
                print(json.dumps({key: summary[key] for key in ("motion_id", "tracking_success", "sequence_had_fall",
                                   "mean_body_position_error_m", "mean_body_rotation_error_rad")}), flush=True)
                if summary["app_or_simulation_stopped"]:
                    raise RuntimeError("Simulation stopped before full22 measurement completed")
            if sum(row["observed_metric_steps"] for row in report["sequences"]) != 11982:
                raise RuntimeError("Full22 measurement is incomplete")
            report.update(status="passed", aggregate=reference["aggregate_sequences"](report["sequences"]))
        if parent._sources()[0] != binding["sources"] or record(Path(__file__)) != binding["entry"]:
            raise RuntimeError("Evaluation source changed during execution")
        if record(args.manifest)["sha256"] != parent.MANIFEST_SHA:
            raise RuntimeError("AMASS manifest changed during evaluation")
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(),
                      aggregate=reference["aggregate_sequences"](report["sequences"]))
        traceback.print_exc()
    finally:
        # Save an honest pre-cleanup result if native teardown itself crashes.
        result_path = args.output / "result.json"
        report["elapsed_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        lifecycle.close(app, world, producer)
        app = world = producer = None
        if not report["cleanup_succeeded"]:
            report["status"] = "failed"
        report["elapsed_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "aggregate": report.get("aggregate"),
                      "cleanup_succeeded": report["cleanup_succeeded"]}), flush=True)
    return 0 if report["status"] in {"passed", "passed_lifecycle_smoke"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
