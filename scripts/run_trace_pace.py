#!/usr/bin/env python3
"""Physical TRACE DDPM + public PACER AMP, IsaacLab3/IsaacSim6.

Default is a bounded straight/turn/stop reliability fixture. An explicit shared
scene uses the SAME cuboids for PhysX, AMP height observations and TRACE maps.
No pose playback, root teleportation, policy substitution or automatic reset.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from social_navigation.resource_bindings import scene_locations, resolve_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(*, terminal_state=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/diffusion_amp/trace_pace_reliable.json",
    )
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--measure-contacts",
        action="store_true",
        help="Explicit N1/N4 shared fixture: floor support and peer/map force capture",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 6000:
        parser.error("Physical controls must be 1..6000 (up to200s)")
    if args.measure_contacts:
        import yaml

        if (
            args.scene is None
            or len(scene_locations(yaml.safe_load(args.scene.read_text()))["origins_xy"]) not in (1,4)
        ):
            parser.error(
                "Measured-contact terminal ownership is verified only for an explicit N1/N4 shared scene"
            )
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = json.loads(args.config.read_text())
    if configuration.get("schema") != "crowdsim.trace_pace_reliability.v1":
        raise ValueError("Wrong config schema")
    resources = {}
    for name, item in configuration["resources"].items():
        path = resolve_path(item["path"], root=ROOT)
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError("Resource identity changed: " + name)
        resources[name] = path
    manifest = json.loads(resources["amass_manifest"].read_text())
    if manifest["splits"]["train"]["clip_count"] != 2848:
        raise ValueError("Fixed real-AMASS population changed")
    report = {
        "status": "preflight_passed",
        "paper_result": False,
        "model_family": ("public_TRACE_plus_public_PACER_AMP" if configuration["resources"]["amp_checkpoint"]["sha256"]
                         == "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"
                         else "public_TRACE_plus_hash_bound_PACER_candidate"),
        "weights_finetuned_on_local_2848": (False if configuration["resources"]["amp_checkpoint"]["sha256"]
                                          == "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615" else None),
        "local_2848_manifest_fixed": True,
        "AMP_sample_start_s": configuration.get("amp_sample_start_s", 0.0),
        "AMP_sample_offsets_s": [configuration.get("amp_sample_start_s", 0.0) + .5 * i for i in range(10)],
        "public_PACER_native_sampling_clock": configuration.get("amp_sample_start_s", 0.0) == 0.0,
        "sdk_started": False,
        "actual_controls": 0,
        "falls": 0,
        "route_vertices_reached": 0,
        "resets_after_initialization": 0,
        "seed": args.seed,
        "requested_controls": args.steps,
        "config_sha256": sha256(args.config),
        "resource_sha256": {n: sha256(p) for n, p in resources.items()},
        "scene_sha256": None if args.scene is None else sha256(args.scene),
        "safety_scope": "sampled planning clearance and actual falls; force contact metrics require separate evaluation",
        "method_adaptations": [
            "Public TRACE action-space unicycle baseline; Nav private five-state checkpoint unavailable",
            "Public PACER state-only AMP discriminator; Nav history-plus-action policy remains separate",
            "Locally chosen0.5s replanning; Trace and Pace Sec3.3 uses2s",
            "Conservative SDF raster guide and candidate rejection; not a collision guarantee",
        ],
    }
    if args.preflight_only:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "sdk_started": False}))
        return 0
    from social_navigation.sdk_paths import require_cuda_driver
    require_cuda_driver()
    from CrowdSim.humanoid.runtime.trace_pace_lifecycle import TracePaceLifecycle

    if args.measure_contacts:
        from CrowdSim.humanoid.runtime.trace_pace_measured_lifecycle import (
            TracePaceMeasuredLifecycle,
        )

        import yaml
        declared_scene = scene_locations(yaml.safe_load(args.scene.read_text()))
        if declared_scene.get('schema') == 'crowdsim.native_pedestrian_scene.v1':
            from CrowdSim.humanoid.runtime.trace_pace_native_lifecycle import TracePaceNativeLifecycle
            lifecycle = TracePaceNativeLifecycle(report)
        else:
            lifecycle = TracePaceMeasuredLifecycle(report)
    else:
        lifecycle = TracePaceLifecycle(report)
    from protomotions.utils.simulator_imports import import_simulator_before_torch

    launcher = import_simulator_before_torch("isaaclab")
    app = launcher(
        {
            "headless": True,
            "device": "cuda:0",
            "fast_shutdown": False,
            "kit_args": lifecycle.kit_args,
        }
    ).app
    import gc
    from types import SimpleNamespace
    owner = SimpleNamespace(world=None, temporary_refs={}, physics_frame=None)
    try:
        _run_physics(app, args, resources, configuration, report, lifecycle, owner)
    except Exception as error:
        report.update(status="failed_preserved", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # The physics function has returned, so control-state tensors and
        # controller/planner scopes are gone before native ownership is closed.
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        report["CUDA_synchronized_before_control_frame_release"] = True
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        try:
            report["completed_physics_frame_release"] = clear_completed_physics_frame(owner, _run_physics)
        except Exception as error:
            report.update(status="failed_control_ownership", control_frame_release_error=str(error))
        gc.collect()
        live = [name for name, ref in owner.temporary_refs.items() if ref() is not None]
        report["temporary_control_objects_live_before_SDK_close"] = live
        report["physics_function_scope_released_before_SDK_close"] = not live
        if live:
            report["temporary_object_referrer_diagnostic"] = _temporary_referrers(owner)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        report["CUDA_synchronized_after_control_frame_release"] = True
        if live:
            report["status"] = "failed_control_ownership"
        terminal_events=[]
        previous_profile=sys.getprofile()
        def terminal_profile(frame,event,arg):
            if event=='c_call':
                name=getattr(arg,'__name__','')
                if name.startswith('release_physx'):
                    terminal_events.append({'event':event,'function':name,'caller':frame.f_code.co_filename})
            elif event=='call' and frame.f_code.co_name=='on_shutdown' and '/omni/physx/' in frame.f_code.co_filename:
                terminal_events.append({'event':event,'function':'PhysxExtension.on_shutdown','source':frame.f_code.co_filename})
        sys.setprofile(terminal_profile)
        try:
            lifecycle.close(app, owner.world)
        finally:
            sys.setprofile(previous_profile)
            report['physx_terminal_API_events']=terminal_events
        terminal_profile=None
        owner.world = None
        app = None
        gc.collect()
        if args.measure_contacts and declared_scene.get('schema') == 'crowdsim.native_pedestrian_scene.v1':
            if terminal_state is not None:
                # Only primitive result data crosses this boundary. main's
                # launcher/lifecycle/temporary native aliases must be gone
                # before the framework unloads its plugins.
                terminal_state.update(report=report, output=args.output)
                report['native_framework_shutdown_pending'] = True
            else:
                from CrowdSim.humanoid.runtime.native_framework_shutdown import unload_native_plugins
                try:
                    unload_native_plugins(report, full_app_close_returned=report['cleanup_succeeded'])
                except Exception as error:
                    report['cleanup_succeeded']=False
                    report['cleanup_errors'].append('native framework unload: '+str(error))
        # Reinstall diagnostics after native extensions have shut down; native
        # crash handlers may otherwise leave no Python stack at process exit.
        import faulthandler
        faulthandler.disable()
        faulthandler.enable(all_threads=True)
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in (
                        "status",
                        "actual_controls",
                        "falls",
                        "displacement_m",
                        "mean_route_error_m",
                        "error",
                        "cleanup_succeeded",
                    )
                }
            ),
            flush=True,
        )
    return 0 if report["status"] == "passed" else 1


def _temporary_referrers(owner):
    """Describe only reference types/variable names; retain no live objects."""
    import gc
    import types
    output = {}
    for name, weak in owner.temporary_refs.items():
        value = weak()
        if value is None:
            continue
        descriptions = []
        for referrer in gc.get_referrers(value):
            item = {"type": type(referrer).__module__ + "." + type(referrer).__qualname__}
            if isinstance(referrer, dict):
                item["keys"] = [key for key in referrer if isinstance(key, str)][:90]
                item["same_value_names"] = [key for key, v in referrer.items() if v is value and isinstance(key, str)]
                item["owners"] = [
                    {"type": type(parent).__module__ + "." + type(parent).__qualname__,
                     "function": parent.f_code.co_name if isinstance(parent, types.FrameType) else None}
                    for parent in gc.get_referrers(referrer)
                ][:12]
            elif isinstance(referrer, types.FrameType):
                item["function"] = referrer.f_code.co_name
            descriptions.append(item)
        output[name] = descriptions
    return output


def _run_physics(app, args, resources, configuration, report, lifecycle, owner):
    """Keep borrowed control-state lifetimes inside the live SDK World scope."""
    owner.physics_frame = sys._getframe()
    import torch
    import numpy as np
    import yaml
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp import PacerAMPController
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.diffusion.planner import DiffusionPlanner
    from CrowdSim.humanoid.analysis.trace_pace_capture import pack_candidate_samples
    from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidanceConfig
    from CrowdSim.humanoid.runtime.trace_pace import TracePaceClosedLoop

    world = None
    targets = None
    records = []
    plans = []
    poses = []
    actions = []
    forces = []
    joint_positions = []
    episode_ids = []
    reset_masks = []
    replan_seconds = []
    root_rotations = []
    planning_records = []
    runtime = None
    candidate_states = []
    selected_samples = []
    ground_capture = ground_archive = None
    contact_records = []
    report.update(status="failed", sdk_started=True)
    try:
        lifecycle.verify_loaded()
        torch.manual_seed(args.seed)
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        scene = None if args.scene is None else scene_locations(yaml.safe_load(args.scene.read_text()))
        count = 1 if scene is None else len(scene["origins_xy"])
        native_scene = None
        if scene is not None and scene.get('schema') == 'crowdsim.native_pedestrian_scene.v1':
            if not args.measure_contacts: raise ValueError('Native TRACE/PACER requires actual contact measurement')
            from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
            from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
            snap = scene.get('probe_script_snapshot')
            if snap is not None and sha256(Path(snap['path'])) != snap['sha256']:
                raise ValueError('Native probe snapshot changed')
            provider = DynBenchSceneProvider.from_preflight(scene['preflight']['path'],
                expected_sha256=scene['preflight']['sha256'], probe_script_snapshot=None if snap is None else snap['path'])
            provider.admit_native_map(scene['native_map']['path'], expected_sha256=scene['native_map']['sha256'])
            provider.configure_policy_height_feature(scene.get('amp_policy_height_mode', 'native_obstacle_cap'))
            native_scene = NativeGridScene.from_provider(provider,origins_xy=scene['origins_xy'],routes_xy=scene['routes_xy'],
                clearance_m=scene['clearance_m'],arrival_radius_m=scene.get('arrival_radius_m',.75),
                goal_lookahead_m=scene.get('goal_lookahead_m',5.),device='cuda:0',cyclic_route_lookahead=scene.get('cyclic_route_lookahead',False),start_recovery_radius_m=scene.get('start_recovery_radius_m',0.),start_recovery_clearance_m=scene.get('start_recovery_clearance_m',.4),route_clearance_weight=scene.get('route_clearance_weight',0.),route_comfort_clearance_m=scene.get('route_comfort_clearance_m',.6))
        corpus = AMASSCorpus(
            resources["amass_manifest"],
            split="validation",
            device="cuda:0",
            seed=args.seed,
            dof_velocity_frame="child_joint",
        )
        world = AMPWorld(
            app,
            corpus,
            num_envs=count,
            shared_world=scene is not None,
            scene_description=scene if native_scene is None else None,
            native_navigation_scene=native_scene,
            episode_steps=args.steps + 1,
            contact_reporting=args.measure_contacts,
            align_spawn_heading_to_route=False if scene is None else scene.get("align_spawn_heading_to_route",False),
            initial_reference_max_speed_m_s=None if scene is None else scene.get("initial_reference_max_speed_m_s"),
            initialize_at_rest=False if scene is None else scene.get("initialize_at_rest",False),
            initial_reference_arm_cone_deg=None if scene is None else scene.get("initial_reference_arm_cone_deg"),
            scene_physics_replication=None if scene is None else scene.get("scene_physics_replication"),
        )
        owner.world = world
        report["physics"] = world.physics_manifest()
        if world.scene_provider is not None:
            report['native_scene_provider'] = world.scene_provider.describe()
            report['native_scene_live_observation'] = world.native_scene_live_observation
        report["global_route_guidance"] = {
            "right_hand_traffic": False if scene is None else scene.get("right_hand_traffic", False),
            "right_hand_padding_distance_m": None if scene is None else scene.get("right_hand_padding_distance_m", .5),
            "right_hand_cost_weight": None if scene is None else scene.get("right_hand_cost_weight", 2.),
            "fresh_query_at_every_DDPM_replan": True,
            "author_method": False,
        }
        report["initial_reference_motion_ids"] = world.reference_ids.tolist()
        if args.measure_contacts:
            from isaaclab.sim.utils import get_current_stage
            from protomotions.simulator.isaaclab.utils.usd_body_paths import (
                resolve_body_prim_paths,
            )
            from CrowdSim.humanoid.foot_support import FOOT_NAMES
            from CrowdSim.humanoid.foot_capture import FootCaptureAdapter
            from CrowdSim.humanoid.ground_support_archive import (
                GroundSupportArchive,
                PEER_FIELDS,
            )

            report["physical_contact_measurement"] = world.enable_pairwise_contacts(
                threshold_n=1.0
            )
            feet = resolve_body_prim_paths(
                world.simulator._robot.cfg.spawn.usd_path, list(FOOT_NAMES)
            )
            actor_feet = [
                {
                    name: f"/World/envs/env_{actor}/Robot/{feet[name]}"
                    for name in FOOT_NAMES
                }
                for actor in range(count)
            ]
            ground_capture = FootCaptureAdapter.from_world(
                world,
                actor_foot_paths=actor_feet,
                stage=get_current_stage(),
                floor_roots=["/World/ground"] if world.scene_provider is None else list(world.scene_provider.floor_roots),
                contact_reporting_enabled=True,
                peer_monitor=world.simulator.pedestrian_contacts,
                threshold_n=1.0,
            )
            world.simulator.pedestrian_contacts = ground_capture
            ground_archive = GroundSupportArchive(count)
            report["ground_support_contract"] = ground_capture.manifest[
                "foot_support_capture"
            ]
        controller = PacerAMPController.from_checkpoint(
            resources["amp_checkpoint"],
            world.lower,
            world.upper,
            body_names=world.mapping.body_names,
            dof_names=world.mapping.dof_names,
            device="cuda:0",
            expected_sha256=configuration["resources"]["amp_checkpoint"]["sha256"],
        )
        if controller.local_finetune is not None:
            if (controller.local_finetune["manifest_sha256"] != report["resource_sha256"]["amass_manifest"]
                    or set(controller.local_finetune["expert_unique_motion_ids"])
                    != {row["motion_id"] for row in corpus.manifest["splits"]["train"]["clips"]}):
                raise ValueError("Fine tuned checkpoint used a different actual AMASS training population")
            report.update(model_family="public_TRACE_plus_local_AMASS_finetuned_PACER_AMP",
                          weights_finetuned_on_local_2848=True,
                          AMP_finetune_provenance=controller.local_finetune)
        planner = DiffusionPlanner(
            resources["diffusion_checkpoint"],
            resources["diffusion_config"],
            device="cuda:0",
        )
        report["actual_diffusion_steps"] = planner.model.n_timesteps
        guidance = PedestrianGuidanceConfig(**configuration["guidance"])
        runtime = TracePaceClosedLoop(
            planner,
            controller,
            guidance,
            replan_controls=configuration["replan_controls"],
            num_samples=configuration["num_samples"],
            seed=args.seed,
            allow_safety_braking=configuration.get("allow_safety_braking", False),
            stationary_route_heading=configuration.get(
                "stationary_route_heading", False
            ),
            braking_agent_buffer_m=configuration.get("braking_agent_buffer_m", 0.05),
            mixed_safety_brakes=configuration.get("mixed_safety_brakes", False),
            guard_horizon_s=configuration.get("guard_horizon_s", 1.0),
            guard_tracking_margin_m=configuration.get("guard_tracking_margin_m", 0.0),
            map_guard_clearance_m=configuration.get("map_guard_clearance_m"),
            pedestrian_extent_m=configuration.get("pedestrian_extent_m"),
            actual_body_map_guard=configuration.get("actual_body_map_guard",False),
            body_guard_horizon_s=configuration.get("body_guard_horizon_s",.3),
            body_guard_gap_m=configuration.get("body_guard_gap_m",.025),
            persistent_braking_target=configuration.get("persistent_braking_target", False),
            settle_braking_target=configuration.get("settle_braking_target", False),
            amp_sample_start_s=configuration.get("amp_sample_start_s", 0.0),
        )
        start = world.planar().clone()
        base_yaw = start[:, 4]
        # Route vertex0 is the first destination in reconstruction fixtures.
        route_indices = torch.zeros(count, device=world.device, dtype=torch.long)

        def fixture(time):
            first = time.clamp(0, 4)[None]
            arc = (time - 4).clamp(0, 4)[None]
            angle = base_yaw[:, None] + 0.15 * arc
            x = (
                start[:, 0:1]
                + 0.6 * first * base_yaw[:, None].cos()
                + 4 * (angle.sin() - base_yaw[:, None].sin())
            )
            y = (
                start[:, 1:2]
                + 0.6 * first * base_yaw[:, None].sin()
                - 4 * (angle.cos() - base_yaw[:, None].cos())
            )
            return torch.stack((x, y), -1)

        warmup_controls=0 if scene is None else scene.get("initialization_warmup_controls",0)
        if type(warmup_controls) is not int or not 0<=warmup_controls<=150:raise ValueError("Warmup budget must be declared0..150 actual controls")
        report["initialization_warmup_controls"]=warmup_controls
        from dataclasses import replace
        nominal_guidance=runtime.guidance_config
        warmup_guidance=replace(nominal_guidance,target_speed_m_s=0.)
        from CrowdSim.humanoid.runtime.route_dwell import RouteDwell
        dwell=RouteDwell(0 if scene is None else scene.get('route_dwell_controls',0),count)
        report['route_dwell_requested_controls']=dwell.controls
        report['route_dwell_events']=dwell.events
        report["braking_target_settling_events"] = []
        previous_dwell=False
        stationary_controls = np.zeros(count, dtype=np.int64)
        no_arrival_controls = np.zeros(count, dtype=np.int64)
        for control in range(args.steps):
            before = world.planar()
            scene_object = world.validation_scene
            if scene_object is not None and control>=warmup_controls and not dwell.active(control):
                route_indices, reached = scene_object.advance_route(
                    before[:, :2], route_indices
                )
                dwell.arrived(control,reached,before[:,:2])
                report["route_vertices_reached"] += int(reached.sum())
                no_arrival_controls = np.where(reached.cpu().numpy(), 0, no_arrival_controls + 1)
                if int(no_arrival_controls.max()) > 1800:
                    report["controls_since_route_arrival_per_actor"] = no_arrival_controls.tolist()
                    raise RuntimeError("No route vertex reached for60s in bounded fixture; physical wandering is not navigation progress")
            dwelling=dwell.active(control)
            if dwelling!=previous_dwell:
                # Only command intent is invalidated. Physics, pose, history
                # and the contiguous physical control clock remain untouched.
                runtime.plan=None;runtime.plan_control=-1
                runtime.braking_anchor_xy=None;runtime.braking_anchor_mask=None
            previous_dwell=dwelling
            def fresh_targets():
                nonlocal targets
                old_failures = 0 if scene_object is None else scene_object.global_path_failures
                if dwelling:
                    targets=dwell.targets()
                elif control<warmup_controls:
                    targets=world.origins[:,None,:].expand(count,10,2).clone()
                elif scene_object is None:
                    targets = fixture(torch.arange(1, 11, device=world.device) * .5 + control * world.dt)
                else:
                    targets, _, _ = scene_object.route_waypoints(before[:, :2], route_indices, advance=False)
                    if scene_object.global_path_failures > old_failures:
                        report["failed_global_route_queries"] = scene_object.last_path_query_failures
                        report["route_indices_at_failure"] = route_indices.tolist()
                        raise RuntimeError("Global route planning failed; stationary path is not admitted as progress")
                return targets

            runtime.guidance_config=warmup_guidance if control<warmup_controls or dwelling else nominal_guidance
            if control==warmup_controls and warmup_controls:runtime.plan=None
            if runtime.plan is None:
                fresh_targets()
            started = time.perf_counter()
            action, diag = runtime.action(world, control, targets, route_target_provider=fresh_targets)
            report["braking_target_settling_events"].extend(diag["braking_target_settling_events"])
            if diag["replanned"]:
                replan_seconds.append(time.perf_counter() - started)
                plans.append(runtime.plan.xy.detach().cpu().numpy())
                candidate_states.append(
                    runtime.plan.sample_states.detach().cpu().numpy()
                )
                selected_samples.append(
                    runtime.plan.selected_sample.detach().cpu().numpy()
                )
                planning_records.append(
                    {
                        "physical_control": control,
                        "declared_route_dwell": dwelling,
                        "sample_indices": runtime.plan.selected_sample.tolist(),
                        "sampling_dynamics_limits": runtime.plan.sampling_dynamics_limits,
                        "seconds": replan_seconds[-1],
                        "trajectory_source": runtime.plan.reference,
                        "safety_braking": diag["safety_braking"],
                        "candidate_sources": (
                            ["TRACE_DDPM_refined" if guidance.final_refinement_steps else "TRACE_DDPM"] * configuration["num_samples"]
                            + ["analytic_braking"]
                        )
                        if runtime.plan.sample_states.shape[1]
                        == configuration["num_samples"] + 1
                        else ["analytic_braking"]
                        if bool((runtime.plan.selected_sample < 0).all())
                        else ["TRACE_DDPM_refined" if guidance.final_refinement_steps else "TRACE_DDPM"] * runtime.plan.sample_states.shape[1],
                        "rejected_diffusion_diagnostics": runtime.last_rejection_diagnostics,
                        "AMP_braking_target_mode": "initial inertial stopping destination; measured stationary1s holding intent enabled; causal guards unchanged"
                        if runtime.settle_braking_target else "persistent stopping destination; causal admission separately uses current measured velocity"
                        if runtime.persistent_braking_target else "time-sampled braking trajectory",
                        "AMP_braking_anchor_xy": [
                            point.tolist() if bool(braking) else None
                            for point, braking in zip(runtime.braking_anchor_xy, runtime.braking_anchor_mask)
                        ] if runtime.braking_anchor_xy is not None else None,
                        "diagnostics": {
                            key: [
                                [float(x) if np.isfinite(x) else None for x in row]
                                for row in value.cpu().tolist()
                            ]
                            for key, value in runtime.plan.guidance_diagnostics.items()
                        },
                    }
                )
            with torch.no_grad():
                if ground_capture is not None:
                    ground_capture.prepare_control_step(control)
                state, fallen, truncated = world.step(action)
            if ground_capture is not None:
                measurement = ground_capture.measurements()
                ground_archive.append(
                    measurement,
                    reset_mask=world.age == 1,
                    terminated=fallen,
                    truncated=truncated,
                    control_step_after=control + 1,
                )
                contact_records.append(
                    {
                        key: measurement[key].detach().cpu().numpy().copy()
                        for key in PEER_FIELDS
                    }
                )
            actual = world.planar(state)
            target = (
                fixture(torch.tensor([(control + 1) * world.dt], device=world.device))[
                    :, 0
                ]
                if scene is None
                else targets[:, 0]
            )
            record = torch.cat(
                (
                    actual,
                    state.root_pos[:, 2:3],
                    target,
                    fallen[:, None].float(),
                    world.measured_signed_power_w()[:, None],
                ),
                -1,
            )
            records.append(record.cpu().numpy())
            poses.append(state.rigid_body_pos.cpu().numpy())
            root_rotations.append(state.root_rot.cpu().numpy())
            episode_ids.append(world.episode_ids.cpu().numpy().copy())
            reset_masks.append((world.age == 1).cpu().numpy().copy())
            actions.append(action.cpu().numpy())
            forces.append(state.dof_forces.cpu().numpy())
            joint_positions.append(state.dof_pos.cpu().numpy())
            report["actual_controls"] = control + 1
            report["falls"] += int(fallen.sum())
            if control % 30 == 0:
                print(
                    json.dumps(
                        {
                            "control": control + 1,
                            "falls": report["falls"],
                            "route_vertices_reached": report["route_vertices_reached"],
                            "actual_root_xy_m": state.root_pos[:, :2].cpu().tolist(),
                            "active_route_index": None if scene_object is None else route_indices.cpu().tolist(),
                            "mean_speed_m_s": float(
                                state.root_vel[:, :2].norm(dim=-1).mean()
                            ),
                            "terminal_hold": diag["terminal_held_waypoints"],
                        }
                    ),
                    flush=True,
                )
            if bool(fallen.any()):
                raise RuntimeError(
                    "Actual physical fall: stop without reset or substitute controller"
                )
            if bool(truncated.any()):
                raise RuntimeError("Unexpected episode reset boundary")
            if scene is not None:
                moving = state.root_vel[:, :2].norm(dim=-1).cpu().numpy() >= 0.1
                stationary_controls = np.where(moving, 0, stationary_controls + 1)
                if int(stationary_controls.max()) > 600:
                    report["stationary_controls_per_actor"] = (
                        stationary_controls.tolist()
                    )
                    raise RuntimeError(
                        "Persistent cyclic-route stall above20s; not accepted as reliable navigation"
                    )
        data = np.stack(records)
        from CrowdSim.humanoid.runtime.route_progress import physical_progress
        progress = physical_progress(data[:, :, :2], start[:, :2].cpu().numpy(),cyclic_route=scene is not None)
        speed = np.linalg.norm(data[:, :, 2:4], axis=-1)
        report.update(
            status="passed" if progress["spatial_progress_passed"] else "failed_progress",
            **progress,
            mean_speed_m_s=speed.mean(0).tolist(),
            minimum_pelvis_height_m=float(data[:, :, 5].min()),
            mean_route_error_m=float(
                np.linalg.norm(data[:, :, :2] - data[:, :, 6:8], axis=-1).mean()
            )
            if scene is None
            else None,
            stopped_speed_m_s=float(speed[300:].mean())
            if scene is None and len(data) > 300
            else None,
            mean_replan_seconds=float(np.mean(replan_seconds)),
            realtime_planning_supported=float(np.max(replan_seconds))
            <= configuration["replan_controls"] / 30,
            parameter_gradients_accumulated=any(
                p.grad is not None for p in planner.model.parameters()
            ),
            safety_braking_plans=runtime.safety_braking_plans,
            pure_diffusion_planning=runtime.safety_braking_plans == 0 and guidance.final_refinement_steps == 0,
            final_clean_refinement_steps=guidance.final_refinement_steps,
        )
        if scene is None and (
            report["mean_route_error_m"] > 1
            or (
                report["stopped_speed_m_s"] is not None
                and report["stopped_speed_m_s"] > 0.15
            )
        ):
            report["status"] = "failed_quality"
        report["quality_gate"] = {
            "minimum_spatial_progress_m": 0.3,
            "progress_distance_metric": progress["progress_distance_metric"],
            "fixture_mean_route_error_limit_m": 1.0,
            "fixture_stopped_speed_limit_m_s": 0.15,
            "allowed_falls": 0,
        }
    except Exception as error:
        report.update(
            status="failed_preserved",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if hasattr(error, "diagnostics"):
            report["rejected_plan_diagnostics"] = error.diagnostics
        traceback.print_exc()
    finally:
        if world is not None and world.validation_scene is not None:
            report["global_path_failures"] = world.validation_scene.global_path_failures
            report["successful_start_recoveries"] = world.validation_scene.successful_start_recoveries
            report["last_start_recovery_segments"] = world.validation_scene.last_start_recovery_segments
        if world is not None and world.scene_provider is not None:
            report["initial_reference_max_speed_m_s"]=world.initial_reference_max_speed_m_s
            report["initial_reference_arm_cone_deg"]=world.initial_reference_arm_cone_deg
            report["initial_reference_pool"]=getattr(world,"initial_reference_pool_report",None)
            report["initial_reference_selected_frames"]=getattr(world,"initial_reference_selected_frames",None)
            report["initial_velocities_zero_only_at_spawn"]=world.initialize_at_rest
            report["initial_spawn_heading_aligned_to_route"]=world.align_spawn_heading_to_route
        if runtime is not None:
            report["safety_braking_plans"] = runtime.safety_braking_plans
            report["pure_diffusion_planning"] = runtime.safety_braking_plans == 0 and guidance.final_refinement_steps == 0
            report["final_clean_refinement_steps"] = guidance.final_refinement_steps
            report["stationary_route_heading"] = runtime.stationary_route_heading
        try:
            if records:
                packed, offsets = pack_candidate_samples(candidate_states)
                np.savez_compressed(
                    args.output / "physical.npz",
                    samples=np.stack(records),
                    body_positions=np.stack(poses),
                    root_rotations_xyzw=np.stack(root_rotations),
                    actions=np.stack(actions),
                    joint_forces=np.stack(forces),
                    joint_positions=np.stack(joint_positions),
                    planned_xy=np.stack(plans) if plans else np.empty((0,)),
                    candidate_states_local=packed,
                    candidate_plan_offsets=offsets,
                    selected_samples=np.stack(selected_samples)
                    if selected_samples
                    else np.empty((0,)),
                    columns=np.array(
                        [
                            "x",
                            "y",
                            "vx",
                            "vy",
                            "yaw",
                            "pelvis_z",
                            "target_x",
                            "target_y",
                            "fallen",
                            "signed_power_w",
                        ]
                    ),
                )
                # Lossless rendering input from the same measured physical run.
                # Rendering replay is separate and never adds physics/rewards.
                recorded = args.output / "physical_states.npz"
                body_positions = np.stack(poses)
                np.savez_compressed(recorded, root_pos=body_positions[:, :, 0],
                    body_pos=body_positions, root_rot=np.stack(root_rotations),
                    dof_pos=np.stack(joint_positions), episode_ids=np.stack(episode_ids),
                    reset_mask=np.stack(reset_masks))
                report["actual_control_steps"] = len(records)
                report["evaluation_scene"] = scene
                report["state_capture"] = {"path": str(recorded), "frequency_hz": 30,
                    "body_names": list(world.mapping.body_names),
                    "reset_mask": "first post-step frame of each new episode; physical termination captured separately",
                    "coordinates": "world meters, quaternions XYZW, common SMPL names"}
            (args.output / "planning.json").write_text(
                json.dumps(planning_records, indent=2) + "\n"
            )
            if world is not None and world.scene_provider is not None:
                from CrowdSim.humanoid.runtime.native_contact_report import write_static_report_archive
                report["native_static_contact_report"]=write_static_report_archive(world.scene_provider,args.output/"native_static_residual.npz")
            if ground_archive is not None and ground_archive.rows:
                report["ground_support_capture"] = ground_archive.write(
                    args.output / "ground_support.npz"
                )
                np.savez_compressed(
                    args.output / "contact_measurements.npz",
                    **{
                        name: np.stack([row[name] for row in contact_records])
                        for name in PEER_FIELDS
                    },
                )
                report["actual_force_contact_controls"] = len(contact_records)
        except Exception as capture_error:
            report.update(
                status="failed_capture",
                capture_error=f"{type(capture_error).__name__}: {capture_error}",
                capture_traceback=traceback.format_exc(),
            )
            traceback.print_exc()
        finally:
            import weakref
            # Only weak references cross the function boundary. These objects
            # must die while the native simulator and CUDA context are valid.
            owner.temporary_refs = {
                name: weakref.ref(value) for name, value in (
                    ("last_robot_state", locals().get("state")),
                    ("actual_planar_state", locals().get("actual")),
                    ("pre_control_state", locals().get("before")),
                    ("last_AMP_action", locals().get("action")),
                    ("runtime", runtime),
                    ("planner", locals().get("planner")),
                    ("controller", locals().get("controller")),
                ) if value is not None
            }
            # Nested synchronous route/fixture callbacks share closure cells.
            # A retained callback cell can outlive its cleared caller frame.
            # Unbind our own finished query context after capture is complete;
            # the native World is separately and explicitly owned by owner.
            before = start = base_yaw = targets = route_indices = scene_object = None
            fresh_targets = fixture = None
            world = None


def entry():
    """Finalize plugins after the complete source-owned main scope returns."""
    terminal_state = {}
    code = main(terminal_state=terminal_state)
    if terminal_state:
        import gc
        gc.collect()
        report, output = terminal_state['report'], terminal_state['output']
        from CrowdSim.humanoid.runtime.native_framework_shutdown import unload_native_plugins
        try:
            unload_native_plugins(report, full_app_close_returned=report['cleanup_succeeded'])
            report['native_framework_shutdown_pending'] = False
            report['native_framework_shutdown']['main_scope_returned'] = True
        except Exception as error:
            report['status'] = 'failed_cleanup'
            report['cleanup_succeeded'] = False
            report['cleanup_errors'].append('native framework unload after main: '+str(error))
            code = 1
        (output/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        print(json.dumps({'terminal_status':report['status'],
                          'native_framework_shutdown_pending':report.get('native_framework_shutdown_pending'),
                          'original_main_returncode':code}),flush=True)
    return code

if __name__ == "__main__":
    sys.exit(entry())
