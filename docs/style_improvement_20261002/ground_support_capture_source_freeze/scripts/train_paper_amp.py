#!/usr/bin/env python3
"""Train the action-conditioned AMP controller on real IsaacLab SMPL rollouts.

Diffusion is a separately trained five-state x0 model. Plans, PD targets and
declared joint-power measurements enter directly; no SFM or MaskedMimic is used.
"""
from __future__ import annotations

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


def parse_args():
    from CrowdSim.humanoid.application.configuration import parse_amp_args
    return parse_amp_args()


def capture_state_snapshot(world, torch_module, *, native_time_getter=None):
    """Clone observed common state without advancing physics or consuming RNG."""
    clock_metadata = {"source": "injected diagnostic clock getter", "available": callable(native_time_getter),
                      "sdk_distribution_version": None, "binding_module_version": None, "reason": None}
    if native_time_getter is None:
        import omni.physx
        from importlib.metadata import PackageNotFoundError, version
        interface = omni.physx.get_physx_interface()
        native_time_getter = getattr(interface, "get_simulation_time", None)
        try:
            sdk_version = version("isaacsim")
        except PackageNotFoundError:
            sdk_version = None
        clock_metadata = {"source": "omni.physx.get_physx_interface().get_simulation_time()",
                          "available": callable(native_time_getter), "sdk_distribution_version": sdk_version,
                          "binding_module_version": getattr(omni.physx, "__version__", None),
                          "reason": (None if callable(native_time_getter) else
                                     f"{type(interface).__module__}.{type(interface).__name__} does not expose callable get_simulation_time; no native clock observed")}
    state = world.state()
    buffers = {}
    for name in ("root_pos", "root_rot", "root_vel", "dof_pos", "dof_vel", "rigid_body_pos"):
        value = getattr(state, name)
        if not torch_module.is_tensor(value) or value.numel() == 0:
            raise ValueError(f"Capture diagnostic requires a nonempty RobotState.{name} tensor")
        # RobotState properties and Warp views can alias live buffers.
        value = value.detach().clone()
        if not bool(torch_module.isfinite(value).all()):
            raise ValueError(f"Capture diagnostic found nonfinite RobotState.{name}")
        buffers[name] = value
    native_time = float(native_time_getter()) if clock_metadata["available"] else None
    if native_time is not None and not math.isfinite(native_time):
        raise ValueError("Capture diagnostic found nonfinite native PhysX simulation time")
    step_count = world.simulator._sim._physics_step_count
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
        raise ValueError("Capture diagnostic requires a nonnegative native physics step count")
    return {"buffers": buffers, "native_physx_time_s": native_time, "native_clock_metadata": clock_metadata,
            "physics_step_count": step_count,
            "torch_cpu_rng": torch_module.get_rng_state().detach().clone(),
            "torch_cuda_rng": (torch_module.cuda.get_rng_state().detach().clone()
                               if torch_module.cuda.is_initialized() else None)}


def compare_capture_state(before, after, torch_module):
    """Publish finite scalar deltas and RNG hashes, never full state buffers."""
    time_observed = before["native_physx_time_s"] is not None and after["native_physx_time_s"] is not None
    time_delta = after["native_physx_time_s"]-before["native_physx_time_s"] if time_observed else None
    if time_delta is not None and not math.isfinite(time_delta):
        raise ValueError("Capture diagnostic has nonfinite native PhysX time delta")
    deltas = {}
    for name, previous in before["buffers"].items():
        current = after["buffers"][name]
        if previous.shape != current.shape or not bool(torch_module.isfinite(current).all()):
            raise ValueError(f"Capture diagnostic has invalid post-capture RobotState.{name}")
        maximum = float((current-previous).abs().max())
        if not math.isfinite(maximum):
            raise ValueError(f"Capture diagnostic has nonfinite delta for RobotState.{name}")
        deltas[name] = maximum
    rng = {}
    for key in ("torch_cpu_rng", "torch_cuda_rng"):
        previous, current = before[key], after[key]
        if previous is None or current is None:
            rng[key] = {"observed": False, "equal": None,
                        "availability_unchanged": previous is None and current is None}
        else:
            rng[key] = {"observed": True, "equal": bool(torch_module.equal(previous, current)),
                        "before_sha256": hashlib.sha256(previous.cpu().numpy().tobytes()).hexdigest(),
                        "after_sha256": hashlib.sha256(current.cpu().numpy().tobytes()).hexdigest()}
    return {"state_source": "world.state(): common RobotState getter, cloned before/after capture",
            "max_absolute_buffer_deltas": deltas,
            "native_physx_time_s": {"observed": time_observed,
                                    "before": before["native_physx_time_s"], "after": after["native_physx_time_s"],
                                    "delta": time_delta, "before_metadata": before["native_clock_metadata"],
                                    "after_metadata": after["native_clock_metadata"]},
            "physics_step_count": {"before": before["physics_step_count"], "after": after["physics_step_count"],
                                   "delta": after["physics_step_count"]-before["physics_step_count"]},
            "rng": rng}


def main():
    args = parse_args()
    from CrowdSim.humanoid.application.configuration import (
        configure_deterministic_evaluation_torch,
        prepare_deterministic_evaluation_environment,
        runtime_numerics,
    )
    prepare_deterministic_evaluation_environment(args.deterministic_evaluation)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    from CrowdSim.humanoid.application.configuration import load_amp_profile
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    profile = load_amp_profile(args.config)
    preflight = preflight_amp_resources(args, profile)
    if profile.get("mode") != "paper_method_reconstruction":
        raise ValueError("Exact author experiments require the unpublished author manifests; this runner implements the declared method reconstruction")
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    launcher = import_simulator_before_torch("isaaclab")
    launcher_options = {"headless": True, "device": "cuda:0", "enable_cameras": args.capture}
    if args.physics_only:
        launcher_options["kit_args"] = "--/app/renderer/enabled=false --/renderer/enabled=false"
    app = launcher(launcher_options).app
    import torch
    import numpy as np
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner
    from CrowdSim.humanoid.planning_clock import due_agents
    from CrowdSim.humanoid.power import power_contract, power_stage
    from CrowdSim.humanoid.training import PaperAMPTrainer, TrainingConfig, paper_task_reward, paper_total_reward
    from CrowdSim.humanoid.training.manifests import validate_training_manifest
    world = None
    ground_capture = None
    ground_archive = None
    report = {"status": "failed", "method": "Diffusion5D + PPO/action-conditioned AMP",
              "author_experiment_reproduced": False, "arguments": vars(args), "resource_preflight": preflight.as_dict()}
    started = time.monotonic()
    try:
        configure_deterministic_evaluation_torch(torch, args.deterministic_evaluation)
        report["runtime_numerics"] = runtime_numerics(
            torch, deterministic_evaluation_requested=args.deterministic_evaluation)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        if args.resolve_scene_assets:
            import runpy
            try:
                resolver = runpy.run_path(str(ROOT / "scripts/resolve_paper_scene_assets.py"))
                report["scene_asset_resolution"] = resolver["resolve_requests"]()
            except Exception as error:
                report["scene_asset_resolution"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                print(json.dumps({"optional_scene_asset_resolution": "failed", "error": str(error)}), flush=True)
        report["manifest_gate"] = validate_training_manifest(args.manifest)
        reference_velocity_frame = profile["training"].get("expert_dof_velocity_frame", "legacy_stored")
        corpus = AMASSCorpus(args.manifest, "validation" if args.evaluate_steps else "train",
                             device="cuda:0", seed=args.seed, dof_velocity_frame=reference_velocity_frame)
        scene_description = yaml.safe_load(Path(args.scene_config).read_text()) if args.scene_config else None
        world = AMPWorld(app, corpus, num_envs=args.num_envs, history_frames=10,
                         shared_world=bool(args.evaluate_steps), episode_steps=args.episode_steps,
                         scene_description=scene_description, power_source=args.power_source,
                         contact_reporting=args.pairwise_contacts or args.measure_ground_support, spawn_margin_m=args.spawn_margin_m,
                         terrain_extent_m=args.terrain_extent_m)
        curriculum = None
        if args.trajectory_source == "pacer_curriculum":
            from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum
            curriculum = PacerTrajectoryCurriculum(PacerTrajectoryConfig(mode=args.curriculum_mode,
                speed_max_mps=args.curriculum_speed_max_mps,
                scalar_accel_max_mps2=args.curriculum_accel_max_mps2,
                vector_accel_max_mps2=args.curriculum_accel_max_mps2), seed=args.seed)
        reference_adapter = None
        if args.reference_adapter == "causal_accel":
            from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig
            reference_adapter = CausalReferenceAdapter(ReferenceAdapterConfig(
                max_speed_mps=args.reference_max_speed_mps,
                max_acceleration_mps2=args.reference_max_accel_mps2,
                initial_speed_policy=args.reference_initial_speed_policy))
        report["reference_adaptation"] = reference_adapter.describe() if reference_adapter else None
        physics = world.physics_manifest()
        report["physics"] = physics
        if args.pairwise_contacts:
            report["physical_contact_measurement"] = world.enable_pairwise_contacts(threshold_n=args.contact_threshold_n)
        if args.measure_ground_support:
            from isaaclab.sim.utils import get_current_stage
            from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths
            from CrowdSim.humanoid.foot_support import FOOT_NAMES
            from CrowdSim.humanoid.foot_capture import FootCaptureAdapter
            from CrowdSim.humanoid.ground_support_archive import GroundSupportArchive
            relative_feet = resolve_body_prim_paths(world.simulator._robot.cfg.spawn.usd_path, list(FOOT_NAMES))
            actor_feet = [{name: f"/World/envs/env_{actor}/Robot/{relative_feet[name]}" for name in FOOT_NAMES}
                          for actor in range(args.num_envs)]
            ground_capture = FootCaptureAdapter.from_world(world, actor_foot_paths=actor_feet,
                stage=get_current_stage(), floor_roots=["/World/ground"], contact_reporting_enabled=True,
                peer_monitor=getattr(world.simulator, "pedestrian_contacts", None),
                threshold_n=args.ground_support_threshold_n)
            world.simulator.pedestrian_contacts = ground_capture
            ground_archive = GroundSupportArchive(args.num_envs)
            report["ground_support_measurement"] = {
                "contract": ground_capture.manifest["foot_support_capture"],
                "storage": "separate ground_support.npz; existing physical_states.npz schema unchanged",
                "capture_phase": "four completed120Hz physics substeps, after scene.update; published before resets",
                "control_counter": "control_step is explicit pre-world.step runner counter; control_step_after is post-step counter+1",
                "force_units": "N; floor-only normal-force vector sums; friction excluded",
                "reference_true_com": None, "native_capture_probe_passed": False,
            }
        if world.validation_scene is not None:
            report["route_completion_contract"] = {
                "arrival_check": "every 30Hz post-step physical root, excluding fallen actors",
                "on_physical_reset": "restart route index at zero; retain only vertices actually reached before reset",
                "reset_position_counts_as_progress": False,
                "replan_after_vertex": "immediate on the next control frame; joint guidance replans the entire scene"}
        report["power_measurement"] = power_contract(args.power_source)
        report["navigation_guidance"] = {"mode": args.route_guidance,
            "waypoint_times_s": [float(i/2) for i in range(1, 11)] if args.route_guidance == "timed_path" else [5.],
            "guidance_total_weight": 10. if args.route_guidance == "timed_path" else 1.,
            "guidance_per_waypoint_weight": 1.,
            "meaning": "Declared deployment variant; ten points from one static A* path, not future pedestrian oracle or author configuration" if args.route_guidance == "timed_path" else "One five-second sparse endpoint, as used in training"}
        guidance_weight = (float(args.guidance_weight) if args.guidance_weight is not None else
                           (10. if args.diffusion_guidance_mode != "legacy_posterior" else float(10 if args.route_guidance == "timed_path" else 1)))
        guidance_contract = {"mode": args.diffusion_guidance_mode, "total_weight": guidance_weight,
            "gradient_batch_reduction": "sum independent actors; mean waypoints within actor",
            "agent_collision_weight": args.agent_collision_weight, "agent_collision_radius_m": args.agent_collision_radius,
            "scene_collision_weight": args.scene_collision_weight, "scene_collision_radius_m": args.scene_collision_radius,
            "scene_geometry_guidance": "explicit bounded-domain and signed AABB penetration extension" if args.scene_collision_weight > 0 else None,
            "trajectory_speed_weight": args.trajectory_speed_weight, "max_trajectory_speed_mps": args.max_trajectory_speed_mps,
            "trajectory_accel_weight": args.trajectory_accel_weight, "max_trajectory_accel_mps2": args.max_trajectory_accel_mps2,
            "joint_replanning": "entire shared scene whenever any actor is due" if args.agent_collision_weight > 0 else None,
            "author_configuration": False}
        report["diffusion_guidance"] = guidance_contract
        report["trajectory_source"] = args.trajectory_source
        report["curriculum"] = curriculum.describe() if curriculum is not None else None
        report["navigation_guidance"]["guidance_total_weight"] = guidance_weight
        report["navigation_guidance"]["guidance_per_waypoint_weight"] = guidance_weight/(10 if args.route_guidance == "timed_path" else 1)
        provenance = {"dataset": corpus.provenance(), "physics": physics,
                      "diffusion_checkpoint": str(Path(args.diffusion_checkpoint).resolve()),
                      "diffusion_sha256": file_sha256(args.diffusion_checkpoint),
                      "trajectory_error_alignment": "one root XY sample at the physical next-step time, summed over coordinates",
                      "task_tail": "hold the plan endpoint for future observations beyond five seconds; replan every declared interval",
                      "goal_generation": "random sparse 5-second global goal in initial root heading with +/-45deg variation, 0.5..1.5m/s",
                      "scene": "unobstructed flat plane; eight author benchmark scenes unavailable",
                      "expert_action_semantics": "next-frame mocap joint position mapped to normalized PD target, clipped; no inverse dynamics or measured torques",
                      "author_weights": False, "power_measurement": report["power_measurement"],
                      "navigation_guidance": report["navigation_guidance"], "diffusion_guidance": guidance_contract}
        provenance["trajectory_source"] = args.trajectory_source
        provenance["diffusion_used_for_training"] = curriculum is None
        if curriculum is not None:
            provenance["curriculum"] = curriculum.describe()
            provenance["goal_generation"] = "independent PACER-style procedural speed/heading commands, anchored at actual root without physical state modification"
            provenance["navigation_guidance"] = {"mode": "procedural_training_commands", "diffusion_used": False,
                "deployment_planner": "separately trained five-dimensional diffusion"}
        if args.resume:
            trainer, saved = PaperAMPTrainer.load(args.resume, device=world.device, restore_rng=False)
            if trainer.config.expert_dof_velocity_frame != reference_velocity_frame:
                raise ValueError("Profile reference velocity frame differs from the saved controller contract; select its matching profile")
            old_power, new_power, changed_power = power_stage(trainer.provenance.get("power_measurement"), args.power_source,
                explicit_stage=args.new_power_stage, evaluation=bool(args.evaluate_steps))
            report["training_power_measurement"] = old_power
            report["training_navigation_guidance"] = trainer.provenance.get("navigation_guidance", {
                "mode": "endpoint", "waypoint_times_s": [5.], "guidance_total_weight": 1.,
                "guidance_per_waypoint_weight": 1., "identity": "known legacy runner behavior"})
            if args.evaluate_steps:
                report["evaluation_power_measurement_changed"] = changed_power
            elif changed_power:
                trainer.provenance.setdefault("power_measurement_stages", []).append({
                    "from": old_power, "to": new_power, "at_update": trainer.update_count,
                    "source_checkpoint_sha256": file_sha256(args.resume),
                    "reward_measurement_changed": True, "paper_signed_energy_coefficient_preserved": True})
                trainer.provenance["power_measurement"] = new_power
            else:
                trainer.provenance["power_measurement"] = old_power
            if not args.evaluate_steps and not isinstance(saved, dict):
                raise ValueError("Training resume requires saved physical, history, planning and RNG runner state")
            if not args.evaluate_steps and not saved:
                raise ValueError("Training resume runner state is empty")
            if trainer.provenance["diffusion_sha256"] != provenance["diffusion_sha256"]:
                if not (args.evaluate_steps and args.allow_evaluation_planner_change):
                    raise ValueError("Resume diffusion checkpoint changed; evaluation changes require --allow-evaluation-planner-change")
                report["evaluation_planner_change"] = {"training_sha256": trainer.provenance["diffusion_sha256"],
                    "evaluation_sha256": provenance["diffusion_sha256"], "checkpoint": provenance["diffusion_checkpoint"]}
            if not args.evaluate_steps:
                if trainer.provenance.get("trajectory_source", "diffusion") != args.trajectory_source:
                    raise ValueError("Resume trajectory source changed; start a fresh explicit experiment")
                old_guidance = trainer.provenance.get("diffusion_guidance", {"mode": "legacy_posterior", "total_weight": 1.})
                if (old_guidance["mode"], old_guidance["total_weight"]) != (args.diffusion_guidance_mode, guidance_weight):
                    raise ValueError("Resume diffusion guidance changed; start a fresh explicit experiment")
                for key, default in (("agent_collision_weight", 0.), ("agent_collision_radius_m", .3),
                                     ("scene_collision_weight", 0.), ("scene_collision_radius_m", .5),
                                     ("trajectory_speed_weight", 0.), ("max_trajectory_speed_mps", 3.),
                                     ("trajectory_accel_weight", 0.), ("max_trajectory_accel_mps2", 2.)):
                    if old_guidance.get(key, default) != guidance_contract[key]:
                        raise ValueError(f"Resume diffusion guidance {key} changed; start a fresh explicit experiment")
                if any(value is not None for value in (args.learning_rate, args.epochs, args.minibatch_size)):
                    raise ValueError("Resume retains checkpoint optimizer settings; start a new experiment to override them")
                if args.seed != trainer.config.seed:
                    raise ValueError("Resume seed differs from the saved training configuration")
                for key in ("manifest_sha256", "source_sha256", "split"):
                    if trainer.provenance["dataset"][key] != provenance["dataset"][key]:
                        raise ValueError(f"Resume dataset {key} changed; use an explicit new experiment")
                for key in ("rig", "physics_hz", "control_hz", "isolated_environments", "body_names", "dof_names", "dof_limits_lower", "dof_limits_upper", "joint_control", "terrain_material", "live_body_masses_kg", "live_total_mass_kg"):
                    if trainer.provenance["physics"][key] != physics[key]:
                        raise ValueError(f"Resume physical contract {key} changed")
                if trainer.provenance["physics"].get("isolated_spawn_margin_m", 6.) != physics["isolated_spawn_margin_m"]:
                    raise ValueError("Resume isolated spawn margin changed; start a new experiment")
                if trainer.provenance["physics"].get("terrain_extent_m", 64.) != physics["terrain_extent_m"]:
                    raise ValueError("Resume terrain extent changed; start a new experiment")
                if trainer.provenance["physics"].get("scene_geometry") != physics.get("scene_geometry"):
                    raise ValueError("Resume scene geometry changed")
            else:
                report["evaluation_dataset"] = corpus.provenance()
                report["evaluation_scene"] = scene_description
                for key in ("rig", "physics_hz", "control_hz", "body_names", "dof_names", "dof_limits_lower", "dof_limits_upper", "joint_control", "terrain_material", "live_body_masses_kg", "live_total_mass_kg"):
                    if trainer.provenance["physics"][key] != physics[key]:
                        raise ValueError(f"Evaluation physical controller contract {key} changed")
        else:
            config = TrainingConfig(**profile["training"])
            config.seed = args.seed
            if args.learning_rate is not None:
                config.learning_rate = config.discriminator_learning_rate = args.learning_rate
            if args.epochs is not None:
                config.update_epochs = args.epochs
            if args.minibatch_size is not None:
                config.minibatch_size = args.minibatch_size
            config.discriminator_batch_size = min(args.num_envs*args.rollout_steps, config.discriminator_batch_size)
            trainer = PaperAMPTrainer.prepare(config, device=world.device, provenance=provenance)
            saved = None
            if not args.from_scratch:
                report["initialization"] = trainer.warm_start_pacer(preflight.require("warm_start_checkpoint"),
                    world.lower, world.upper, body_names=world.mapping.body_names, dof_names=world.mapping.dof_names)
        planner = (PaperDiffusionPlanner(checkpoint_path=args.diffusion_checkpoint, device=world.device,
                                        compile_forward=args.compile_diffusion_forward) if curriculum is None else None)
        report["runtime_execution"] = {"torch": str(torch.__version__), "physics_only": args.physics_only,
            "compiled_forward": planner.compiled_forward_enabled if planner else False, "compilation": "inductor dynamic=True mode=default" if args.compile_diffusion_forward else None,
            "diffusion_steps": planner.config.diffusion_steps if planner else None, "precision": "FP32", "consumed_planner_output": "XY only" if planner else None,
            "diffusion_used_for_rollouts": planner is not None,
            "diffusion_samples_per_actor": args.diffusion_samples if planner else 0,
            "sample_filtering": "one coherent joint-scene future when joint guidance is active; otherwise per-actor lowest declared objective",
            "compiled_XY_benchmark_max_difference_m": .00064564 if args.compile_diffusion_forward else None,
            "compiled_bit_identical": False if args.compile_diffusion_forward else None}
        report["runtime_execution"]["numerics"] = report["runtime_numerics"]
        report["method"] = (f"Independent PACER-style commands + PPO/AMP ({trainer.config.discriminator_input}); separate Diffusion5D deployment"
                            if curriculum else f"Diffusion5D + PPO/AMP ({trainer.config.discriminator_input})")
        reward_timestamp = "post-step h[t+1..] before resets" if trainer.config.discriminator_input == "state" else "pre-step h[t..]+applied a[t]"
        report["amp_reward_timestamp"] = reward_timestamp
        if not args.resume:
            trainer.provenance["amp_reward_timestamp"] = reward_timestamp
            if trainer.config.discriminator_input == "state":
                trainer.provenance["expert_action_semantics"] = "not consumed by the explicit classic-AMP state-only discriminator"
        if not trainer.config.freeze_toes or trainer.config.history_frames != world.history_frames:
            raise ValueError("Training action/history configuration differs from the direct SMPL world")
        if not math.isclose(trainer.config.control_dt, world.dt, abs_tol=1e-9) or not math.isclose(trainer.config.target_horizon_s, 5., abs_tol=1e-9):
            raise ValueError("Direct SMPL runner requires the declared 30Hz control and five-second trajectory horizon")
        plan_frequency_hz = 30 if reference_adapter else 10
        plans = torch.zeros(args.num_envs, 5*plan_frequency_hz+1, 2, device=world.device)
        raw_plans = torch.zeros(args.num_envs, 51, 2, device=world.device) if reference_adapter else None
        reference_velocities = torch.zeros_like(plans) if reference_adapter else None
        reference_initialized = torch.zeros(args.num_envs, dtype=torch.bool, device=world.device)
        forced_replanning = torch.zeros_like(reference_initialized)
        reference_plan_records = {key: [] for key in ("control_step", "actor_ids", "selected_sample", "raw_xy", "reference_xy", "reference_velocity_xy",
                                                       "initial_overspeed", "speed_violation", "deviation_m")}
        plan_age = torch.full((args.num_envs,), args.replan_steps, dtype=torch.long, device=world.device)
        goals = torch.zeros(args.num_envs, 2, device=world.device)
        route_indices = torch.zeros(args.num_envs, dtype=torch.long, device=world.device)
        route_vertices_reached = torch.zeros_like(route_indices)
        waypoint_times = torch.arange(.5, 5.1, .5, device=world.device) if args.route_guidance == "timed_path" else torch.tensor([5.], device=world.device)
        route_waypoints = torch.zeros(args.num_envs, len(waypoint_times), 2, device=world.device)
        resets = 0
        fall_resets = 0
        time_limit_resets = 0
        replan_count = 0
        planner_forward_calls = 0
        control_step = 0 if args.evaluate_steps else trainer.physical_transition_count//args.num_envs
        completed_lengths = []
        minimum_height = float("inf")

        def new_goals(ids):
            if curriculum is not None:
                return  # The curriculum samples independent commands in replan().
            state = world.planar()[ids]
            if world.validation_scene is not None:
                if args.route_guidance == "timed_path":
                    route_waypoints[ids], route_indices[ids], reached = world.validation_scene.route_waypoints(
                        state[:, :2], route_indices[ids], actor_ids=ids, times=waypoint_times, advance=False)
                    goals[ids] = route_waypoints[ids, -1]
                else:
                    goals[ids], route_indices[ids], reached = world.validation_scene.route_goal(
                        state[:, :2], route_indices[ids], actor_ids=ids, advance=False)
                return
            angle = state[:, 4] + (torch.rand(len(ids), device=world.device)-.5)*np.pi/2
            speed = .5 + torch.rand(len(ids), device=world.device)
            goals[ids] = state[:, :2] + 5*speed[:, None]*torch.stack((angle.cos(), angle.sin()), -1)

        def replan(ids):
            nonlocal replan_count, planner_forward_calls
            if not len(ids):
                return
            if curriculum is not None:
                prediction = curriculum.build_plan(world.state(), actor_ids=ids)
                plans[ids] = prediction.xy
                goals[ids] = prediction.xy[:, -1]
                plan_age[ids] = 0
                replan_count += len(ids)
                return
            if world.validation_scene is not None:
                new_goals(ids)
            # Small inference chunks limit 100-step DDPM memory alongside Kit.
            for chunk in ids.split(args.planner_batch_size):
                inputs = world.planner_inputs(chunk)
                guide = route_waypoints[chunk] if args.route_guidance == "timed_path" else goals[chunk, None]
                prediction = planner.plan(**inputs, waypoints=guide, waypoint_times=waypoint_times,
                    num_samples=args.diffusion_samples,
                    guidance_weight=guidance_weight, guidance_mode=args.diffusion_guidance_mode,
                    scene_ids=torch.zeros(len(chunk), dtype=torch.long, device=world.device) if args.agent_collision_weight > 0 else None,
                    agent_collision_weight=args.agent_collision_weight, agent_collision_radius=args.agent_collision_radius,
                    scene_collision_weight=args.scene_collision_weight, scene_collision_radius=args.scene_collision_radius,
                    trajectory_speed_weight=args.trajectory_speed_weight, max_trajectory_speed_mps=args.max_trajectory_speed_mps,
                    trajectory_accel_weight=args.trajectory_accel_weight, max_trajectory_accel_mps2=args.max_trajectory_accel_mps2,
                    scene_bounds_xy=world.validation_scene.bounds if args.scene_collision_weight > 0 else None,
                    obstacle_centers_xy=world.validation_scene.centers[:, :2] if args.scene_collision_weight > 0 else None,
                    obstacle_sizes_xy=world.validation_scene.sizes[:, :2] if args.scene_collision_weight > 0 else None)
                planner_forward_calls += 1
                if reference_adapter:
                    indices = plan_age[chunk].clamp_max(plans.shape[1]-1)
                    previous_xy = plans[chunk, indices].clone()
                    previous_velocity = reference_velocities[chunk, indices].clone()
                    initialize = ~reference_initialized[chunk]
                    if bool(initialize.any()):
                        physical = world.state()
                        previous_xy[initialize] = physical.root_pos[chunk[initialize], :2]
                        previous_velocity[initialize] = physical.root_vel[chunk[initialize], :2]
                    adapted = reference_adapter.build(prediction.xy, previous_xy, previous_velocity)
                    raw_plans[chunk] = prediction.xy
                    plans[chunk] = adapted.xy
                    reference_velocities[chunk] = adapted.velocity
                    reference_initialized[chunk] = True
                    records = {"control_step": torch.full_like(chunk, control_step), "actor_ids": chunk,
                        "selected_sample": prediction.selected_sample,
                        "raw_xy": adapted.raw_xy, "reference_xy": adapted.xy, "reference_velocity_xy": adapted.velocity,
                        "initial_overspeed": adapted.initial_overspeed, "speed_violation": adapted.speed_violation,
                        "deviation_m": adapted.deviation_m}
                    for key, value in records.items():
                        reference_plan_records[key].append(value.detach().cpu().numpy())
                else:
                    plans[chunk] = prediction.xy
                plan_age[chunk] = 0
                replan_count += len(chunk)

        def interpolate(seconds, trajectory=None, frequency_hz=None):
            trajectory = plans if trajectory is None else trajectory
            frequency_hz = plan_frequency_hz if frequency_hz is None else frequency_hz
            last = trajectory.shape[1]-1
            fractional = (seconds*frequency_hz).clamp(0, last)
            lower = fractional.floor().long().clamp_max(last-1)
            upper = lower+1
            rows = torch.arange(args.num_envs, device=world.device)[:, None]
            return torch.lerp(trajectory[rows, lower], trajectory[rows, upper], (fractional-lower)[..., None])

        def task_trajectory():
            return interpolate(plan_age[:, None]/30 + torch.arange(.5, 5.1, .5, device=world.device)[None])

        def snapshot():
            state = world.state()
            return {"physical": {key: getattr(state, key).detach().cpu() for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")},
                    "amp_history": world.amp_history.cpu(), "amp_history_episode_ids": world.amp_history_episode_ids.cpu(),
                    "planar_history": world.planar_history.cpu(),
                    "history_available": world.history_available.cpu(), "age": world.age.cpu(),
                    "episode_ids": world.episode_ids.cpu(), "reference_ids": world.reference_ids.cpu(),
                    "plans": plans.cpu(), "plan_age": plan_age.cpu(), "goals": goals.cpu(),
                    "corpus_rng": corpus.generator.get_state(),
                    "curriculum_state": curriculum.state_dict() if curriculum else None,
                    "forced_replanning": forced_replanning.cpu(),
                    "runner_contract": {"replan_steps": args.replan_steps, "episode_steps": args.episode_steps,
                                        "planner_batch_size": args.planner_batch_size,
                                        "trajectory_source": args.trajectory_source,
                                        "planning_clock": "synchronized" if args.synchronized_replanning else "per_actor_age"},
                    "control_step": control_step,
                    "runtime_execution": report["runtime_execution"],
                    "physics_resume": "logical state restored; PhysX contact/solver caches are not serialized"}

        if saved and not args.evaluate_steps:
            previous_contract = dict(saved.get("runner_contract", {}))
            previous_contract.setdefault("planning_clock", "per_actor_age")
            previous_contract.setdefault("trajectory_source", "diffusion")
            expected_contract = {"replan_steps": args.replan_steps, "episode_steps": args.episode_steps,
                "trajectory_source": args.trajectory_source,
                "planner_batch_size": args.planner_batch_size, "planning_clock": "synchronized" if args.synchronized_replanning else "per_actor_age"}
            if previous_contract != expected_contract:
                base_matches = ({key: value for key, value in previous_contract.items() if key != "planning_clock"}
                                == {key: value for key, value in expected_contract.items() if key != "planning_clock"})
                if not (args.new_planning_stage and base_matches and previous_contract["planning_clock"] == "per_actor_age"):
                    raise ValueError("Resume planning/episode settings differ from the saved runner contract")
                stage = {"from": previous_contract, "to": expected_contract, "at_update": trainer.update_count,
                    "source_checkpoint_sha256": file_sha256(args.resume), "noise_draw_grouping_changed": True,
                    "newly_reset_actors_may_replan_early_at_next_global_tick": True,
                    "maximum_interval_preserved_steps": args.replan_steps, "reset_planning": "immediate",
                    "meaning": "Explicit planning-clock stage transition, not bit-identical sampling or physical resume"}
                report["planning_clock_transition"] = stage
                trainer.provenance.setdefault("planning_clock_stages", []).append(stage)
            elif args.new_planning_stage:
                raise ValueError("A new planning stage must actually change the saved planning clock")
            if len(saved["age"]) != args.num_envs:
                raise ValueError("Resume environment count differs from checkpoint")
            if trainer.physical_transition_count % args.num_envs:
                raise ValueError("Resume transitions cannot reconstruct an integer control clock")
            ages = saved["plan_age"].detach().cpu()
            if ages.shape != (args.num_envs,) or ages.dtype != torch.long or not bool(((ages >= 0) & (ages <= args.replan_steps)).all()):
                raise ValueError("Resume plan ages violate the saved control interval")
            resumed_counter = saved.get("control_step", control_step)
            if isinstance(resumed_counter, bool) or not isinstance(resumed_counter, int) or resumed_counter < 0:
                raise ValueError("Resume control clock must be a nonnegative integer")
            if resumed_counter != trainer.physical_transition_count//args.num_envs:
                raise ValueError("Resume control clock differs from committed physical transitions")
            from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
            world.simulator.reset_envs(ResetState(**{k:v.to(world.device) for k,v in saved["physical"].items()}, state_conversion=StateConversion.COMMON))
            for key in ("amp_history", "amp_history_episode_ids", "planar_history", "history_available", "age", "episode_ids", "reference_ids"):
                setattr(world, key, saved[key].to(world.device))
            plans.copy_(saved["plans"])
            plan_age.copy_(saved["plan_age"])
            goals.copy_(saved["goals"])
            if "forced_replanning" in saved:
                forced_replanning.copy_(saved["forced_replanning"])
            corpus.generator.set_state(saved["corpus_rng"].cpu())
            if curriculum is not None:
                curriculum_state = saved.get("curriculum_state")
                if not isinstance(curriculum_state, dict):
                    raise ValueError("Curriculum resume requires its saved private RNG and configuration")
                curriculum_state = dict(curriculum_state)
                curriculum_state["generator_state"] = curriculum_state["generator_state"].cpu()
                curriculum.load_state_dict(curriculum_state)
            control_step = resumed_counter
            trainer.restore_rng()
        else:
            new_goals(torch.arange(args.num_envs, device=world.device))
        capture = None
        if args.deterministic_evaluation and args.capture:
            report["capture_state_diagnostics"] = []
        physical_roots = []
        physical_evaluation = {key: [] for key in ("body_pos", "root_rot", "dof_pos", "dof_vel", "dof_forces", "signed_power_w", "target_xy", "global_goal_xy", "global_waypoints_xy", "route_vertex_xy",
                               "route_vertex_reached", "route_indices", "plan_endpoint_goal_error", "episode_ids", "reset_mask", "terminated", "truncated")}
        if args.pairwise_contacts:
            physical_evaluation.update({key: [] for key in ("static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n")})
        if reference_adapter:
            physical_evaluation.update({key: [] for key in ("raw_target_xy", "raw_plan_endpoint_goal_error", "reference_target_velocity_xy", "reference_deviation_m")})
        if args.capture:
            from CrowdSim.protomotions_runtime import enable_human_mesh
            from CrowdSim.tools.smpl_mesh_visualizer import ProtoMotionsHumanMeshAdapter
            from CrowdSim.humanoid.simulate import _capture_setup
            enable_human_mesh(model_dir="data/smpl", hide_humanoid=True, texture_dir="data/smpl/textures", appearance_seed=args.seed, shape_std=0.)
            mesh = ProtoMotionsHumanMeshAdapter.from_simulator(world.simulator)
            mesh.create()
            mesh.update()
            viewer = {"camera_eye": [32, -12, 26], "camera_target": [12, 12, .9]} if world.validation_scene is not None else {"camera_eye": [18, -12, 16], "camera_target": [8, 8, .9]}
            capture = _capture_setup(world, viewer)
            (output / "frames").mkdir(exist_ok=True)
        total_loops = 1 if args.evaluate_steps else args.updates
        timing = {"planner_wall_s": 0., "policy_observation_wall_s": 0., "physics_wall_s": 0.,
                  "reward_capture_wall_s": 0., "reset_wall_s": 0., "expert_sampling_wall_s": 0., "ppo_update_wall_s": 0.}
        for update in range(total_loops):
            collected = {}
            length = args.evaluate_steps if args.evaluate_steps else args.rollout_steps
            for step in range(length):
                due = due_agents(plan_age, control_step, args.replan_steps, synchronized=args.synchronized_replanning)
                due = torch.cat((due, forced_replanning.nonzero().flatten())).unique()
                if args.agent_collision_weight > 0 and len(due):
                    due = torch.arange(args.num_envs, device=world.device)
                section_started = time.perf_counter()
                replan(due)
                forced_replanning[due] = False
                timing["planner_wall_s"] += time.perf_counter()-section_started
                section_started = time.perf_counter()
                state = world.state()
                self_obs, task_obs = world.observations(task_trajectory(), state)
                history = world.amp_history.clone()
                history_episode = world.amp_history_episode_ids.clone()
                episode = world.episode_ids.clone()
                action = trainer.act(self_obs, task_obs, deterministic=bool(args.evaluate_steps))
                amp_reward = (trainer.amp_reward(history, action["applied_actions"])
                              if trainer.config.discriminator_input == "state_action" else None)
                timing["policy_observation_wall_s"] += time.perf_counter()-section_started
                section_started = time.perf_counter()
                if ground_capture is not None:
                    ground_capture.prepare_control_step(control_step)
                state, terminated, truncated = world.step(action["applied_actions"])
                if ground_capture is not None:
                    ground_sample = ground_capture.measurements()
                    ground_archive.append(ground_sample, reset_mask=world.age == 1,
                        terminated=terminated, truncated=truncated, control_step_after=control_step + 1)
                previous_route_indices = route_indices.clone()
                vertex_reached = torch.zeros(args.num_envs, dtype=torch.bool, device=world.device)
                if world.validation_scene is not None:
                    alive = (~terminated).nonzero().flatten()
                    advanced, arrived = world.validation_scene.advance_route(
                        state.root_pos[alive, :2], route_indices[alive], actor_ids=alive)
                    route_indices[alive] = advanced
                    vertex_reached[alive] = arrived
                    route_vertices_reached[alive] += arrived.long()
                if trainer.config.discriminator_input == "state":
                    history = world.amp_history.clone()
                    history_episode = world.amp_history_episode_ids.clone()
                    amp_reward = trainer.amp_reward(history, action["applied_actions"])
                control_step += 1
                timing["physics_wall_s"] += time.perf_counter()-section_started
                section_started = time.perf_counter()
                plan_age += 1
                target = interpolate((plan_age/30)[:, None])[:, 0]
                delta = (state.root_pos[:, :2]-target).square().sum(-1)
                yaw = world.planar(state)[:, 4]
                forward_velocity = state.root_vel[:, 0]*yaw.cos()+state.root_vel[:, 1]*yaw.sin()
                components = paper_task_reward(delta, state.dof_forces, state.dof_vel, forward_velocity,
                    direction_mode=trainer.config.direction_mode, signed_power_w=world.measured_signed_power_w())
                reward = paper_total_reward(amp_reward, components,
                    amp_weight=trainer.config.amp_reward_weight,
                    task_weight=trainer.config.task_reward_weight)
                next_self, next_task = world.observations(task_trajectory(), state)
                transition = {"self_obs": self_obs, "task_obs": task_obs, "amp_history": history,
                    "actions": action["actions"], "applied_actions": action["applied_actions"],
                    "log_prob": action["log_prob"], "values": action["values"],
                    "next_values": trainer.value(next_self, next_task), "rewards": reward,
                    "terminated": terminated, "truncated": truncated,
                    "episode_ids": episode, "history_episode_ids": history_episode,
                    "amp_rewards": amp_reward, "trajectory_rewards": components["trajectory"],
                    "energy_rewards": components["energy"], "direction_rewards": components["direction"],
                    "signed_power_w": components["signed_power_w"], "forward_velocity": forward_velocity}
                for key, value in transition.items():
                    collected.setdefault(key, []).append(value.detach().clone())
                minimum_height = min(minimum_height, float(state.root_pos[:, 2].min()))
                if args.evaluate_steps:
                    physical_roots.append(state.root_pos.cpu().numpy())
                    values = {"body_pos": state.rigid_body_pos, "root_rot": state.root_rot,
                              "dof_pos": state.dof_pos, "dof_vel": state.dof_vel, "dof_forces": state.dof_forces,
                              "signed_power_w": components["signed_power_w"], "target_xy": target, "global_goal_xy": goals,
                              "global_waypoints_xy": route_waypoints if args.route_guidance == "timed_path" else goals[:, None],
                              "route_vertex_xy": goals if world.validation_scene is None else world.validation_scene.routes[torch.arange(args.num_envs, device=world.device), previous_route_indices],
                              "route_vertex_reached": vertex_reached, "route_indices": route_indices,
                              "plan_endpoint_goal_error": torch.linalg.vector_norm(plans[:, -1]-goals, dim=-1), "episode_ids": episode,
                              "reset_mask": world.age == 1, "terminated": terminated, "truncated": truncated}
                    if args.pairwise_contacts:
                        if ground_capture is not None:
                            from CrowdSim.humanoid.ground_support_archive import PEER_FIELDS
                            values.update({key: ground_sample[key] for key in PEER_FIELDS})
                        else:
                            values.update(world.simulator.pedestrian_contacts.measurements())
                    if reference_adapter:
                        raw_target = interpolate((plan_age/30)[:, None], raw_plans, frequency_hz=10)[:, 0]
                        values.update(raw_target_xy=raw_target,
                            raw_plan_endpoint_goal_error=torch.linalg.vector_norm(raw_plans[:, -1]-goals, dim=-1),
                            reference_target_velocity_xy=reference_velocities[torch.arange(args.num_envs, device=world.device), plan_age.clamp_max(plans.shape[1]-1)],
                            reference_deviation_m=torch.linalg.vector_norm(target-raw_target, dim=-1))
                    for key, value in values.items():
                        physical_evaluation[key].append(value.detach().cpu().numpy())
                ids = (terminated | truncated).nonzero().flatten()
                completed_lengths.extend(world.age[ids].cpu().tolist())
                resets += len(ids)
                fall_resets += int(terminated.sum())
                time_limit_resets += int((truncated & ~terminated).sum())
                if capture and step % args.capture_every_steps == 0:
                    from CrowdSim.humanoid.simulate import _capture_frame
                    from PIL import Image
                    mesh.update()
                    diagnostic_before = (capture_state_snapshot(world, torch)
                                         if args.deterministic_evaluation else None)
                    pixels = _capture_frame(world, capture[2], warmup=step==0)
                    if diagnostic_before is not None:
                        diagnostic_after = capture_state_snapshot(world, torch)
                        diagnostic = compare_capture_state(diagnostic_before, diagnostic_after, torch)
                        diagnostic.update(control_step=control_step, frame=f"frame_{step:06d}.png", warmup=step==0)
                        report["capture_state_diagnostics"].append(diagnostic)
                    Image.fromarray(pixels).save(output / "frames" / f"frame_{step:06d}.png")
                timing["reward_capture_wall_s"] += time.perf_counter()-section_started
                section_started = time.perf_counter()
                world.reset(ids)
                if len(ids):
                    if ground_capture is not None:
                        ground_capture.reset_episode_boundary()
                    if world.validation_scene is not None:
                        route_indices[ids] = 0
                    new_goals(ids)
                    plan_age[ids] = args.replan_steps
                    reference_initialized[ids] = False
                forced_replanning[vertex_reached] = True
                timing["reset_wall_s"] += time.perf_counter()-section_started
                if args.evaluate_steps and (step+1) % 150 == 0:
                    progress = {"control_steps": step+1, "simulated_seconds": (step+1)/30,
                        "elapsed_s": time.monotonic()-started, "fall_resets": fall_resets,
                        "route_vertices_reached": route_vertices_reached.cpu().tolist()}
                    with (output / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress)+"\n")
                    print(json.dumps({"evaluation_progress": progress}), flush=True)
            if not args.evaluate_steps:
                rollout = {key: torch.stack(values) for key, values in collected.items()}
                section_started = time.perf_counter()
                expert_history, expert_actions = world.expert_batch(min(length*args.num_envs, 4096))
                timing["expert_sampling_wall_s"] += time.perf_counter()-section_started
                try:
                    section_started = time.perf_counter()
                    metrics = trainer.update(rollout, expert_history, expert_actions)
                    timing["ppo_update_wall_s"] += time.perf_counter()-section_started
                except Exception:
                    torch.save({"rollout": rollout, "expert_history": expert_history,
                                "expert_actions": expert_actions}, output / "failed_rollout.pt")
                    raise
                metrics.update(elapsed_s=time.monotonic()-started, resets=resets,
                               fall_resets=fall_resets, time_limit_resets=time_limit_resets,
                               mean_completed_episode_s=float(np.mean(completed_lengths))/30 if completed_lengths else None,
                               minimum_height_m=minimum_height, diffusion_plans=replan_count if planner else 0,
                               trajectory_plans=replan_count)
                metrics["planner_forward_calls"] = planner_forward_calls
                metrics.update(timing)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics)+"\n")
                print(json.dumps(metrics), flush=True)
                if (update+1) % args.checkpoint_every == 0 or update+1 == total_loops:
                    trainer.save(output / "last.pt", runner_state=snapshot())
        report.update(status="passed", update_count=trainer.update_count,
                      physical_transition_count=trainer.physical_transition_count,
                      actual_control_steps=args.evaluate_steps if args.evaluate_steps else args.updates*args.rollout_steps,
                      elapsed_s=time.monotonic()-started, resets=resets, minimum_height_m=minimum_height,
                      fall_resets=fall_resets, time_limit_resets=time_limit_resets,
                      diffusion_plans=replan_count if planner else 0, trajectory_plans=replan_count,
                      curriculum_plans=curriculum.plans_generated if curriculum else 0,
                      training_config=asdict(trainer.config),
                      completed_episode_lengths=completed_lengths,
                      convergence_claim=False, provenance=trainer.provenance)
        report["timing"] = {**timing, "meaning": "cumulative wall time of sections, including ordinary device synchronization; not kernel-only timings"}
        report["planning_clock"] = {"mode": "synchronized" if args.synchronized_replanning else "per_actor_age",
            "control_step": control_step, "maximum_interval_steps": args.replan_steps,
            "forward_calls": planner_forward_calls, "mean_agents_per_forward": replan_count/planner_forward_calls if planner_forward_calls else None}
        if world.validation_scene is not None:
            report["route_vertices_reached"] = route_vertices_reached.cpu().tolist()
            report["global_path_failures"] = world.validation_scene.global_path_failures
            report["successful_start_recoveries"] = getattr(world.validation_scene, "successful_start_recoveries", 0)
            report["start_recovery_segments_last_query"] = getattr(world.validation_scene, "last_start_recovery_segments", [])
            report["global_path_method"] = "repository A*, 0.25m cells, 0.35m clearance, 5m arclength lookahead; no movement fallback"
        if physical_roots:
            np.savez_compressed(output / "physical_roots.npz", root_pos=np.stack(physical_roots))
            np.savez_compressed(output / "physical_states.npz", root_pos=np.stack(physical_roots),
                                **{key: np.stack(values) for key, values in physical_evaluation.items()})
            report["state_capture"] = {"path": str(output / "physical_states.npz"), "frequency_hz": 30,
                                       "body_names": list(world.mapping.body_names),
                                       "reset_mask": "first post-step frame of each new episode; physical termination captured separately",
                                       "coordinates": "world meters, quaternions XYZW, common SMPL names"}
        if ground_archive is not None:
            if len(ground_archive.rows) != args.evaluate_steps:
                raise RuntimeError("Ground capture did not retain every requested evaluation control")
            report["ground_support_measurement"]["capture"] = ground_archive.write(output / "ground_support.npz")
        if reference_adapter:
            np.savez_compressed(output / "reference_plans.npz",
                                **{key: np.concatenate(values) for key, values in reference_plan_records.items()})
            report["reference_adaptation"]["plan_capture"] = str(output / "reference_plans.npz")
            report["reference_adaptation"]["tracking_target"] = "executed continuous reference; raw_target_xy recorded separately"
        if capture:
            from CrowdSim.humanoid.simulate import _write_video
            report["video"] = _write_video(output, [str(path) for path in sorted((output / "frames").glob("frame_*.png"))], 30/args.capture_every_steps)
            report["video_capture"] = {"fps": 30/args.capture_every_steps, "interval_control_steps": args.capture_every_steps,
                "extra_physics_steps": 0, "width": 960, "height": 720}
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if ground_capture is not None:
            from CrowdSim.humanoid.ground_support_archive import write_interrupted_ground_support
            try:
                if ground_archive.rows and not (output / "ground_support.npz").exists():
                    report["ground_support_measurement"]["completed_capture_before_failure"] = ground_archive.write(output / "ground_support.npz")
                report["ground_support_measurement"]["failed_partial_capture"] = write_interrupted_ground_support(
                    output / "ground_support_partial.npz", ground_capture, reason=report["error"])
            except Exception as capture_error:
                report["ground_support_measurement"]["failed_partial_capture"] = {
                    "valid": False, "data_available": False, "missing_substeps_padded": False,
                    "error": f"{type(capture_error).__name__}: {capture_error}"}
        traceback.print_exc()
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, default=lambda value: getattr(value, "value", str(value)))+"\n")
        print(json.dumps({key:report.get(key) for key in ("status", "update_count", "physical_transition_count", "error")}), flush=True)
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1
            world.close()
        else:
            app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
