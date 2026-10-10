#!/usr/bin/env python3
"""Future single-App shared SocialCNN PPO+AMP entry, fresh only.

Prepared source; import is stdlib only. No current NN/SDK/data execution is
authorized. Root schedules a future bounded smoke before a local training pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGES = ROOT / "docs/style_improvement_20261004"
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
PROFILE_SCHEMA = "local.neutral_smpl.shared_social_entry_profile.v1"
PINS = {
    "probe_setup": (STAGES / "shared_group_world_SDK_probe_staging_v1/probe_shared_world.py", 32066,
                    "8760049579983401d064d7c2a29c6fdef4c40fa6b9358829652aea53c6be3554"),
    "trainer": (STAGES / "social_trainer_staging_v1/social_trainer.py", 24078,
                "81a6319f2520b3be547f30454c88b088638d559d492bba52fe850d1c1d5ba96e"),
    "collector": (STAGES / "social_rollout_collector_staging_v1/social_rollout_collector.py", 27252,
                  "f7d2727051d92da5efc0fc4eae37338c88152aca3904971745d4daa7389556fd"),
    "expert": (STAGES / "social_amass_expert_staging_v1/expert_batch.py", 15279,
               "48bc49764f54bbc6afdf2671f0ed6635f2884534359af1aa1b9bf113696d2afb"),
}
ENTRY_IMPORT_BYTES = Path(__file__).read_bytes()  # Lock this actual execution source before main.
ENTRY_IMPORT_SHA = hashlib.sha256(ENTRY_IMPORT_BYTES).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "smoke.json")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Unsupported: SDK/curriculum/history restore is not implemented")
    args = parser.parse_args()
    if args.resume is not None:
        parser.error("Fresh only: no SDK/curriculum/history resume or old1044/warm-start migration is supported")
    if args.manifest.resolve() != MANIFEST:
        parser.error("The expert adapter requires the exact fixed v7 train manifest path")
    return args


def _record(path, *, readonly=False):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
        raise ValueError("Expected exclusive regular source/artifact: " + str(path))
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load(path, name, record):
    data = Path(path).read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Reviewed source changed before execution: " + name)
    if name in sys.modules:
        raise ValueError("Refuse a previously imported entry-owned module: " + name)
    module = ModuleType(name)
    module.__file__, module.__package__ = str(path), name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _sources():
    records = {}
    for key, (path, size, sha) in PINS.items():
        row = _record(path, readonly=True)
        if row["bytes"] != size or row["sha256"] != sha:
            raise ValueError("Sealed component source changed: " + key)
        records[key] = row
    records["entry"] = _record(Path(__file__))
    if records["entry"]["sha256"] != ENTRY_IMPORT_SHA:
        raise ValueError("Entry file changed after its actual execution source was imported")
    return records


def _profile(path):
    row = _record(path, readonly=True)
    document = json.loads(path.read_text())
    if (type(document) is not dict or set(document) != {"schema", "purpose", "runtime", "training"}
            or document["schema"] != PROFILE_SCHEMA or type(document["purpose"]) is not str
            or document["purpose"] not in {"bounded_real_training_smoke_not_performance",
                "local_shared_training_pilot2000_not_author_complete_budget"}):
        raise ValueError("Explicit new smoke or local training profile required")
    runtime = document["runtime"]
    required = {"actors", "updates", "rollout_controls", "episode_controls", "floor_size_m", "env_spacing_m",
                "checkpoint_every_updates", "archive_every_updates", "archive_controls_before_reset", "seed"}
    if type(runtime) is not dict or set(runtime) != required:
        raise ValueError("Complete explicit runtime profile required")
    for key in ("actors", "updates", "rollout_controls", "episode_controls", "checkpoint_every_updates", "archive_every_updates"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError("Runtime positive exact integer required: " + key)
    if runtime["actors"] % 4 or runtime["seed"] != 7 or type(runtime["seed"]) is not int:
        raise ValueError("Four-actor complete groups and seed7 required")
    if type(runtime["archive_controls_before_reset"]) is not bool:
        raise ValueError("Explicit archive-before-reset choice required")
    for key in ("floor_size_m", "env_spacing_m"):
        if type(runtime[key]) not in (float, int) or not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError("Finite physical extent/spacing required: " + key)
    if type(document["training"]) is not dict:
        raise ValueError("Training fields must be an explicit mapping")
    for key in ("actor_count", "seed", "grid_center", "grid_heading", "height_reference", "include_ego"):
        if key in document["training"]:
            raise ValueError("Fixed runtime/geometry key cannot be overridden by training fields: " + key)
    if document["training"].get("expert_sampling") is not None:
        raise ValueError("This first entry uses the explicit uniform clip/center expert; q requires another declared experiment")
    return document, row


def _save_tensors(path, value, torch):
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return _record(path)


def _runner_setup(args, runtime, probe, records, report, torch):
    """Same reviewed local SMPL/Cuboid setup as the probe, scaled explicitly."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab_physx.physics import PhysxCfg
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
    from protomotions.simulator.isaaclab.utils.scene import SceneCfg
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus

    origins = report["actual_import_origins"]
    for key, value in (("corpus", AMASSCorpus), ("robot_config", SmplRobotConfig),
                       ("simulator_config", IsaacLabSimulatorConfig), ("scene_config", SceneCfg)):
        origins[key] = probe._origin(value, key, records)
    converter = sys.modules["protomotions.simulator.isaaclab.utils.mjcf_to_usd"]
    origins["MJCF_converter"] = probe._origin(converter, "MJCF_converter", records)
    probe._origin(SceneCfg.__init__.__globals__["convert_robot_mjcf_to_usd"], "MJCF_converter", records)
    corpus = AMASSCorpus(args.manifest, "train", device=args.device, seed=runtime["seed"], dof_velocity_frame="child_joint")
    world_module = probe._load(probe.PINS["World"][0], "_shared_social_world", records["World"])
    normal = probe._load(probe.PINS["normal_factory"][0], "_shared_social_normal_factory", records["normal_factory"])
    package = ModuleType("_shared_social_original_training")
    package.__path__ = [str(ROOT / "CrowdSim/humanoid/training")]
    if package.__name__ in sys.modules:
        raise ValueError("Refuse a previously imported private curriculum package")
    sys.modules[package.__name__] = package
    for key, filename in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
        path, size, sha, _ = world_module.PINS[key]
        row = _record(path)
        if row["bytes"] != size or row["sha256"] != sha:
            raise ValueError("Original mixed curriculum source changed: " + key)
        probe._load(path, package.__name__ + "." + filename, row)
    procedural = sys.modules[package.__name__ + ".trajectories"]
    mixed = sys.modules[package.__name__ + ".mixed_trajectories"]
    curriculum = mixed.MixedTrajectoryCurriculum(corpus, runtime["actors"], device=args.device,
        bounds_xy=None, tail_policy="linear_velocity", procedural_config=procedural.PacerTrajectoryConfig(), seed=runtime["seed"])
    groups, spacing = runtime["actors"] // 4, runtime["env_spacing_m"]
    config = IsaacLabSimulatorConfig(num_envs=groups, headless=True, experiment_name="diffusion_amp_reconstruction")
    config.sim.fps, config.sim.decimation = 120, 4
    config.sim.physx.num_position_iterations, config.sim.physx.num_velocity_iterations = 4, 4
    robot = SmplRobotConfig()
    robot.contact_bodies = None
    base = SceneCfg(config=config, robot_config=robot, terrain=None, projectile_config=None,
        num_envs=groups, env_spacing=spacing, filter_collisions=True, replicate_physics=True)
    workaround = sys.modules.get("protomotions.simulator.isaaclab.utils.mjcf_d6_workaround")
    origins["D6_workaround"] = ({"loaded": True, **probe._origin(workaround, "D6_workaround", records)} if workaround is not None
        else {"loaded": False, "expected_source": records["D6_workaround"], "note": "Cached USD can bypass converter/workaround execution"})
    base.robot.spawn.activate_contact_sensors = True
    if not Path(base.robot.spawn.usd_path).is_file():
        raise ValueError("Actual local SMPL USD must exist; no new remote asset request")
    extent = runtime["floor_size_m"]
    base.ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1,
        spawn=sim_utils.CuboidCfg(size=(extent, extent, .2),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.1)))
    def height(query):
        if bool((query.abs() > extent / 2).any()):
            raise ValueError("Actual root/grid query left the finite physical floor; preserve failure, never return infinite-ground0")
        return torch.zeros(query.shape[:-1], dtype=query.dtype, device=query.device)
    def geometry(snapshot, root_heading, defined):
        del defined
        xy = snapshot.robot_state.root_pos[:, :2].clone()
        return xy, root_heading.clone(), height(xy)
    # Only the caller invokes create. _run already checked no existing context
    # and will recover a partially created new context if this call fails.
    return world_module, normal, corpus, curriculum, robot, base, geometry, height, sim_utils.SimulationCfg(
        device=args.device, dt=1/120, render_interval=4,
        physics=PhysxCfg(max_position_iteration_count=4, max_velocity_iteration_count=4))


def _run(args, profile, profile_record, report, records):
    if "torch" in sys.modules:
        raise RuntimeError("Fresh process required: caller App bootstrap must precede Torch")
    sys.path.insert(0, str(ROOT))
    probe = _load(PINS["probe_setup"][0], "_shared_social_probe_helpers", records["probe_setup"])
    setup_records = report["setup_sources"] = probe._sources()
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    report["actual_import_origins"] = {"bootstrap": probe._origin(import_simulator_before_torch, "bootstrap", setup_records)}
    launcher = import_simulator_before_torch("isaaclab")
    app, world, trainer, collector, expert, sim_type = None, None, None, None, None, None
    creation_attempted, active_payload = False, None
    runtime = profile["runtime"]
    try:
        app = launcher({"headless": True, "device": args.device, "enable_cameras": False, "fast_shutdown": False,
            "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
        report["app_constructors_returned"] = 1
        import torch
        import isaaclab.sim as sim_utils
        torch.set_num_threads(4)
        random.seed(runtime["seed"])
        torch.manual_seed(runtime["seed"])
        torch.cuda.manual_seed_all(runtime["seed"])
        sim_type = sim_utils.SimulationContext
        if sim_type.instance() is not None:
            raise RuntimeError("Refuse an existing simulation context")
        manifest = probe._record(args.manifest, readonly=True)
        if manifest["sha256"] != probe.MANIFEST_SHA:
            raise ValueError("Fixed v7 manifest changed")
        report["manifest"] = manifest
        setup = _runner_setup(args, runtime, probe, setup_records, report, torch)
        world_module, normal, corpus, curriculum, robot, base, geometry, height, simulation_cfg = setup
        if sim_type.instance() is not None:
            raise RuntimeError("Scene configuration unexpectedly constructed a simulation context")
        creation_attempted = True
        world = world_module.SharedGroupWorld.create_after_app(app, simulation_cfg, base.robot, base,
            num_actors=runtime["actors"], actor_offsets_xy=probe.OFFSETS, env_spacing=runtime["env_spacing_m"],
            robot_config=robot, curriculum=curriculum, ground_height_at=height, terrain_sampler=height,
            grid_geometry=geometry, grid_geometry_contract=probe.GEOMETRY, include_ego=False,
            normal_force_bundle_factory=normal.create_normal_force_bundle, episode_steps=runtime["episode_controls"])
        report["actual_initialized_physics_step"] = world._counter()
        report["actual_world_origins"] = world.scene.env_origins.detach().cpu().tolist()
        report["actual_robot_MJCF"] = probe._asset_identity(sys.modules["protomotions.simulator.isaaclab.utils.mjcf_to_usd"].resolve_robot_mjcf_path(robot.asset))
        report["actual_native_assets"] = [{"name": f"robot_{slot}", "instances": asset.num_instances,
            "body_names": list(asset.body_names), "joint_names": list(asset.joint_names),
            "USD": probe._asset_identity(asset.cfg.spawn.usd_path)} for slot, asset in enumerate(world.assets)]
        trainer_api = _load(PINS["trainer"][0], "_shared_social_trainer", records["trainer"])
        config = trainer_api.SocialTrainingConfig(actor_count=runtime["actors"], seed=runtime["seed"],
            **probe.GEOMETRY, include_ego=False, **profile["training"])
        config.validate()
        provenance = {"experiment": "fresh_shared_neutral_social_CNN_PPO_AMP_local_pilot", "profile": profile_record,
            "entry_sources": records, "setup_sources": setup_records, "AMASS": corpus.provenance(),
            "physical_scene": {"floor_size_m": runtime["floor_size_m"], "floor_top_world_z_m": 0.,
                "env_spacing_m": runtime["env_spacing_m"], "group_actor_rows": world.rows.cpu().tolist(),
                "grid_geometry_contract": probe.GEOMETRY, "include_ego": False},
            "initialization": {"strategy": "fresh_local_social_CNN_and_state_action_D", "seed": runtime["seed"],
                "weights_or_moments_transferred": False, "author_navisaaclab_weights": False}}
        trainer = trainer_api.SocialAMPTrainer.prepare(config, device=args.device, provenance=provenance)
        expert_api = _load(PINS["expert"][0], "_shared_social_expert", records["expert"])
        expert = expert_api.SocialAMASSExpert(world, corpus)
        collector_api = _load(PINS["collector"][0], "_shared_social_collector", records["collector"])
        collector = collector_api.SocialRolloutCollector(world, trainer)
        report["actual_configuration"] = trainer.config.to_document()
        report["expert_contract"] = expert.describe()
        profile_path = args.output / "resolved_configuration.json"
        report["artifacts"].append(probe._write_json(profile_path, {"entry_profile": profile,
            "training_configuration": report["actual_configuration"], "provenance": trainer.provenance}))
        metrics_path = args.output / "metrics.jsonl"
        with metrics_path.open("x") as log:
            for update in range(1, runtime["updates"] + 1):
                report["attempted_update"] = update
                update_start = time.monotonic()
                archive_count = 0
                def archive_control(item):
                    nonlocal archive_count
                    report["artifacts"].append(_save_tensors(args.output / f"update_{update:06d}_control_{archive_count:03d}.pt", item, torch))
                    archive_count += 1
                collected = collector.collect(runtime["rollout_controls"],
                    external_archive=archive_control if runtime["archive_controls_before_reset"] else None)
                active_payload = {"collection": collected, "expert": None}
                expert_count = min(runtime["rollout_controls"] * runtime["actors"], 4096)
                packet = expert.sample(expert_count)  # Idle after all physical terminal states/reset handoffs.
                active_payload["expert"] = {"history": packet.features, "actions": packet.actions,
                    "motion_ids": packet.motion_ids, "frame_indices": packet.frame_indices}
                if update % runtime["archive_every_updates"] == 0 or update == runtime["updates"]:
                    report["artifacts"].append(_save_tensors(args.output / f"rollout_update_{update:06d}.pt", active_payload, torch))
                metrics = trainer.update(collected["rollout"], packet.features, packet.actions)
                expected = update * runtime["rollout_controls"] * runtime["actors"]
                if (trainer.update_count != update or trainer.physical_transition_count != expected
                        or world.control_step != update * runtime["rollout_controls"]):
                    raise RuntimeError("Fresh training/physical control count changed")
                actual_rms = {name: float(getattr(trainer, name).count) for name in
                    ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")}
                per_update_amp = (runtime["rollout_controls"] * runtime["actors"] + expert_count) * (10 if config.amp_normalization == "per_frame" else 1)
                if (any(actual_rms[name] != expected for name in ("self_normalizer", "task_normalizer", "value_normalizer"))
                        or actual_rms["amp_normalizer"] != update * per_update_amp):
                    raise RuntimeError("Actual full-task/value/AMP moments count disagrees with fresh draw exposure")
                metrics.update(update_wall_s=time.monotonic() - update_start, rollout_actor_transitions=runtime["actors"] * runtime["rollout_controls"],
                    actual_expert_windows_drawn=expert_count, cumulative_expert_windows_drawn=expert.windows,
                    actual_normalizer_counts=actual_rms, completed_physical_controls=world.control_step,
                    actual_physics_step=world._counter(), driver_reason_counts=dict(world.statistics),
                    disc_expert_exposures_this_update=metrics["disc_expert_samples"] * metrics["discriminator_minibatches"],
                    disc_live_policy_exposures_this_update=metrics["disc_live_samples"] * metrics["discriminator_minibatches"],
                    disc_replay_exposures_this_update=metrics["disc_replay_samples"] * metrics["discriminator_minibatches"],
                    elapsed_wall_s=time.monotonic() - report["monotonic_start"])
                json.dump(metrics, log, allow_nan=False)
                log.write("\n")
                log.flush()
                os.fsync(log.fileno())
                report["completed_updates"] = update
                if update % runtime["checkpoint_every_updates"] == 0 or update == runtime["updates"]:
                    path = args.output / f"step_{update:06d}.pt"
                    if path.exists() or path.with_name(path.name + ".tmp").exists():
                        raise FileExistsError("Refuse checkpoint or temporary-file overwrite: " + str(path))
                    trainer.save(path, runner_state=None)  # Deliberately excludes SDK/curriculum/history restore state.
                    report["artifacts"].append(_record(path))
                print(json.dumps({"update": update, "physical_transitions": expected,
                    "reward_mean": metrics["reward_mean"], "actual_physics_step": world._counter()}), flush=True)
                active_payload = None
                del collected, packet
        report["artifacts"].append(_record(metrics_path))
        report["final_world_description"] = probe._json(world.describe(), torch)
        report["final_expert_description"] = expert.describe()
        report["actual_training_transitions"] = trainer.physical_transition_count
        report["entry_sources_after"] = _sources()
        report["setup_sources_after"] = probe._sources()
        if (report["entry_sources_after"] != records or report["setup_sources_after"] != setup_records
                or _record(args.config, readonly=True) != profile_record):
            raise ValueError("Owned execution source/profile changed during this run")
        report["training_loop_completed"] = True
    except BaseException:
        if trainer is not None:
            try:
                report["failure_training_state"] = {"update_count": trainer.update_count,
                    "physical_transition_count": trainer.physical_transition_count,
                    "normalizer_counts": {name: float(getattr(trainer, name).count) for name in
                        ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")},
                    "update_is_atomic": False, "retry_performed": False}
            except BaseException as error:
                report["failure_training_observation_error"] = type(error).__name__ + ": " + str(error)
        if collector is not None and collector.failed_steps is not None:
            active_payload = {"failed_collection": collector.failed_steps}
        if active_payload is not None:
            try:
                report["artifacts"].append(_save_tensors(args.output / "failed_rollout_preserved.pt", active_payload, torch))
            except BaseException as error:
                report["failure_payload_save_error"] = type(error).__name__ + ": " + str(error)
        if world is not None:
            try:
                report["failure_world_description"] = probe._json(world.describe(), torch)
                report["failure_actual_physics_step"] = world._counter()
                report["artifacts"].append(probe._write_npz(args.output / "failure_native_snapshot.npz",
                    probe._tensor_arrays(probe._state_arrays(world.adapter.snapshot()))))
                report["failure_snapshot_is_validated_control_endpoint"] = False
            except BaseException as error:
                report["failure_observation_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            report["entry_sources_after"] = _sources()
            report["setup_sources_after"] = probe._sources()
            report["profile_after"] = _record(args.config, readonly=True)
            report["source_profile_before_after_equal"] = (report["entry_sources_after"] == records
                and report["setup_sources_after"] == setup_records and report["profile_after"] == profile_record)
        except BaseException as error:
            report["source_profile_before_after_equal"] = False
            report["source_observation_error"] = type(error).__name__ + ": " + str(error)
        errors, sim = [], None
        try:
            sim = world.sim if world is not None else sim_type.instance() if creation_attempted and sim_type is not None else None
        except BaseException as error:
            errors.append("context lookup: " + type(error).__name__ + ": " + str(error))
        if sim is not None:
            try:
                sim.stop()
                report["simulation_stop_returned"] = True
            except BaseException as error:
                errors.append("sim.stop: " + type(error).__name__ + ": " + str(error))
            try:
                if type(sim).instance() is not sim:
                    raise RuntimeError("Refuse to clear a simulation singleton not owned by this entry")
                sim.clear_instance()
                if type(sim).instance() is not None:
                    raise RuntimeError("Simulation singleton remains after clear_instance returned")
                report["simulation_clear_instance_returned"] = True
            except BaseException as error:
                errors.append("sim.clear_instance: " + type(error).__name__ + ": " + str(error))
        if app is not None:
            try:
                app.close()
                report["app_close_returned"] = True
            except BaseException as error:
                errors.append("app.close: " + type(error).__name__ + ": " + str(error))
        report["cleanup_errors"] = errors
        report["cleanup_succeeded"] = (not errors and report["simulation_stop_returned"]
            and report["simulation_clear_instance_returned"] and report["app_close_returned"])


def main():
    args = parse_args()
    report = {"schema": "local.neutral_smpl.shared_social_training_entry.v1", "status": "running",
        "monotonic_start": time.monotonic(), "started_unix_s": time.time(), "completed_updates": 0,
        "training_loop_completed": False, "app_constructors_returned": 0, "simulation_stop_returned": False,
        "app_close_returned": False, "simulation_clear_instance_returned": False, "cleanup_succeeded": False, "artifacts": [],
        "resume_implemented": False, "SDK_corpus_curriculum_history_checkpoint_state_included": False,
        "initialization": "fresh SocialCNN and D2019; no old weights or moments transferred",
        "author_budget_or_eight_scene_effect_claimed": False, "promotion_performed": False}
    # Source-only helper loading; no SDK/Torch before the App-owned _run path.
    records = _sources()
    probe = _load(PINS["probe_setup"][0], "_shared_social_output_helpers", records["probe_setup"])
    args.output = probe._output(args.output)
    try:
        profile, profile_record = _profile(args.config)
        report["profile"] = profile
        report["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        report["entry_sources"] = records
        _run(args, profile, profile_record, report, records)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["elapsed_wall_s"] = time.monotonic() - report.pop("monotonic_start")
    report["output_inventory_before_result"] = [{"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(args.output.iterdir()) if path.is_file()]
    report["status"] = "passed_local_training" if (report["training_loop_completed"] and report["cleanup_succeeded"]
        and report.get("source_profile_before_after_equal") is True and "error" not in report) else "failed_preserved"
    probe._write_json(args.output / "result.json", report)
    print(json.dumps({"status": report["status"], "completed_updates": report["completed_updates"], "output": str(args.output)}))
    return 0 if report["status"] == "passed_local_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
