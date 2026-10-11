"""Output-only Sim 6.1 RTX probe. Requires a free GPU; never run as a CPU test.

Uses one owned H1-2 articulation and scene-owned cameras so the existing,
version-pinned SDK61Lifecycle can perform complete shutdown without extension.
Run from the repository root. Acceptance also requires the parent exit receipt.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import traceback
from types import SimpleNamespace


class CameraWorld:
    def __init__(self, app, args):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig, IsaacLabSimParams
        from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig
        from protomotions.components.terrains.terrain import Terrain
        from protomotions.components.terrains.config import TerrainConfig
        from protomotions.components.scene_lib import SceneLib

        class CameraSimulator(IsaacLabSimulator):
            def _get_scene_cfg(self):
                cfg = super()._get_scene_cfg()
                cfg.rtx_probe_target = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/RTXProbeTarget",
                    spawn=sim_utils.CuboidCfg(size=(1., 1., 1.),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .1, .1))),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., 1.)),
                )
                cfg.rtx_probe_camera = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/RTXProbeCamera", update_period=0.,
                    height=48, width=64, data_types=["rgb", "distance_to_image_plane"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.,
                        horizontal_aperture=20.955, clipping_range=(.1, 20.)),
                    renderer_cfg=IsaacRtxRendererCfg(depth_clipping_behavior="none"),
                )
                return cfg

        self.simulator = None
        self.device = torch.device(args.device)
        robot_config = H1_2RobotConfig()
        config = IsaacLabSimulatorConfig(sim=IsaacLabSimParams(fps=200, decimation=4),
            headless=True, num_envs=args.num_envs, experiment_name="rtx_camera_probe")
        terrain = Terrain(config=TerrainConfig(), num_envs=args.num_envs, device=self.device)
        scene_lib = SceneLib.empty(num_envs=args.num_envs, device=self.device)
        self.simulator = CameraSimulator.__new__(CameraSimulator)
        CameraSimulator.__init__(self.simulator, config=config, robot_config=robot_config,
            terrain=terrain, scene_lib=scene_lib, device=self.device, simulation_app=app)
        self.simulator._initialize_with_markers({})


def run_probe(app, args, owner, report, lifecycle):
    # Every camera, renderer-output, callback and simulator alias stays in this
    # completed frame; the existing frame helper clears it before native stop.
    owner.physics_frame = inspect.currentframe()
    import numpy as np
    from PIL import Image
    import torch
    from CrowdSim.control.drive import articulation_torch_view
    from CrowdSim.tools.test_camera_depth import add_camera_check_key, record_headless_camera_burst
    from CrowdSim.utils.sensor_stream import RobotCameraRecorder

    owner.world = lifecycle.construct_world(CameraWorld, app, args)
    lifecycle.verify_loaded()
    if args.fail_after_initialize:
        raise RuntimeError("Intentional RTX failure after camera initialization")
    simulator = owner.world.simulator
    camera = simulator._scene.sensors["rtx_probe_camera"]
    origins = articulation_torch_view(simulator._scene.env_origins).clone()
    report["scene_env_origins"] = origins.detach().cpu().tolist()
    state = simulator.get_default_robot_reset_state()
    # Keep the physical articulation away from the analytic visual target.
    state.root_pos[:] = origins + torch.tensor([-2., 0., 1.], device=args.device)
    simulator.reset_envs(state, env_ids=torch.arange(args.num_envs, device=args.device))
    physics_step_start = simulator._sim.get_physics_step_count()
    camera.set_world_poses_from_view(origins + torch.tensor([0., -3., 1.], device=args.device),
                                     origins + torch.tensor([0., 0., 1.], device=args.device))
    zero = torch.zeros(args.num_envs, simulator.robot_config.number_of_actions, device=args.device)

    def step(action):
        simulator.step(action)
        report["completed_control_steps"] += 1

    for _ in range(args.warmup_steps):
        step(zero)
        simulator._sim.render()
        camera.update(simulator.dt, force_recompute=True)
        _ = camera.data.output
    output = camera.data.output
    rgb = articulation_torch_view(output["rgb"]).detach().cpu().numpy().copy()
    depth = articulation_torch_view(output["distance_to_image_plane"]).detach().cpu().numpy().copy()
    center = depth[:, 22:26, 30:34, 0]
    valid = np.isfinite(depth) & (depth > 0)
    report.update(rgb_shape=list(rgb.shape), rgb_dtype=str(rgb.dtype), depth_shape=list(depth.shape),
                  depth_dtype=str(depth.dtype), valid_fraction=valid.mean(axis=(1, 2, 3)).tolist(),
                  expected_center_depth_m=2.5, center_depth_m=center.mean(axis=(1, 2)).tolist(),
                  camera_output_types={name: type(value).__module__ + "." + type(value).__name__
                                       for name, value in output.items()},
                  renderer_type=type(camera._renderer).__module__ + "." + type(camera._renderer).__name__)
    assert rgb.shape == (args.num_envs, 48, 64, 3) and rgb.dtype == np.uint8
    assert depth.shape == (args.num_envs, 48, 64, 1) and np.issubdtype(depth.dtype, np.floating)
    assert "IsaacRtxRenderer" in report["renderer_type"]
    assert np.isfinite(center).all() and np.all(np.abs(center - 2.5) < .05), report
    assert bool((rgb.max(axis=(1, 2, 3)) > rgb.min(axis=(1, 2, 3))).all()), "RGB has no contrast"
    np.save(args.output / "native_rgb.npy", rgb)
    np.save(args.output / "native_depth.npy", depth)

    callbacks = {}
    facade = SimpleNamespace(_sim=simulator._sim, headless=True,
        keyboard_interface=SimpleNamespace(add_callback=lambda key, callback: callbacks.setdefault(key, callback)))
    env = SimpleNamespace(num_envs=args.num_envs, robot_config=simulator.robot_config,
        device=owner.world.device, simulator=facade, crowdsim_robot_camera=camera,
        dt=simulator.dt, step=step)
    # The project diagnostics use fixed relative paths. A fresh per-run cwd
    # isolates their files; simulator construction already resolved its assets.
    original_cwd = Path.cwd()
    os.chdir(args.output)
    try:
        add_camera_check_key(env)
        callbacks["C"]()
        for eid in range(args.num_envs):
            np.testing.assert_array_equal(np.load(f"output/renderings/camera_check/depth_env{eid}.npy"), depth[eid, ..., 0])
            with Image.open(f"output/renderings/camera_check/rgb_env{eid}.png") as saved:
                np.testing.assert_array_equal(np.asarray(saved), rgb[eid])
        record_headless_camera_burst(env, frames=args.frames, save_envs=args.num_envs)
        recorder = RobotCameraRecorder(env, args.output / "recordings", fps=10., env_ids=list(range(args.num_envs)))
        recorder.toggle()
        recorder.step()
        recorder.toggle()
        for eid in range(args.num_envs):
            raw = np.load(f"output/renderings/camera_headless/env{eid:02d}_frame0000.npy")
            np.testing.assert_allclose(raw[22:26, 30:34], 2.5, atol=.05)
            for index in range(args.frames):
                assert Path(f"output/renderings/camera_headless/env{eid:02d}_frame{index:04d}.npy").is_file()
            recorded = torch.load(recorder.session_dir / f"env_{eid:04d}/depth_000000.pt", weights_only=True)
            assert tuple(recorded.shape) == (48, 64, 1)
            assert (recorder.session_dir / f"env_{eid:04d}/rgb_000000.png").is_file()
        report.update(project_key_capture_exact=True, project_burst_raw_frames=args.frames * args.num_envs,
                      project_recorder_env_frames=args.num_envs,
                      completed_physics_steps=simulator._sim.get_physics_step_count() - physics_step_start,
                      expected_control_steps=args.warmup_steps + args.frames, status="passed")
        assert report["completed_control_steps"] == report["expected_control_steps"]
        assert report["completed_physics_steps"] == report["expected_control_steps"] * simulator.decimation
    finally:
        os.chdir(original_cwd)
        # This dictionary owns a closure over env/camera; drop it before the
        # completed run frame is cleared. No native owner escapes in the report.
        callbacks.clear()


def drain_rtx_before_context_release(app):
    """Finish owned camera/Replicator shutdown through public SDK APIs.

    This rendering-owner phase precedes the unchanged physics lifecycle. The
    pinned Replicator's STOP callback schedules a five-update settings restore;
    wait_until_complete alone pumps only one update after it marks STOPPED.
    """
    import asyncio
    import omni.replicator.core as rep
    from isaaclab.sim import SimulationContext

    rep.orchestrator.set_capture_on_play(False)
    context = SimulationContext.instance()
    if context is not None:
        context.stop()  # SDK Camera STOP releases annotators/render products.
    rep.orchestrator.stop()
    rep.orchestrator.wait_until_complete()
    loop = asyncio.get_event_loop()
    targets = ("_Orchestrator._restore_settings.<locals>._delayed_restore_async_rendering",
               "_Orchestrator.stop.<locals>.on_stop")
    pending_names = []
    for updates in range(1, 33):
        app.update()
        pending_names = [task.get_coro().__qualname__ for task in asyncio.all_tasks(loop)
                         if not task.done() and task.get_coro().__qualname__ in targets]
        if updates >= 6 and not pending_names and rep.orchestrator.get_status().name == "STOPPED":
            return {"public_orchestrator_stop_and_wait": True, "app_updates": updates,
                    "remaining_owned_stop_tasks": [], "orchestrator_status": "STOPPED"}
    raise RuntimeError(f"Replicator stop did not drain after {updates} updates: {pending_names}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent per-run directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--fail-after-initialize", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.warmup_steps, args.frames) < 1:
        parser.error("counts must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "num_envs": args.num_envs, "completed_control_steps": 0,
              "cleanup_succeeded": False, "cleanup_errors": [], "app_close_returned": False,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle, uses_sim61
    from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
    root = Path(__file__).resolve().parents[3]
    _drop_exception_tracebacks = runpy.run_path(str(root / "examples/benchmark/isaaclab_bench.py"))["_drop_exception_tracebacks"]
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "examples/benchmark/isaaclab_bench.py", "src/CrowdSim/tools/test_camera_depth.py",
        "src/CrowdSim/utils/sensor_stream.py", "src/protomotions/simulator/isaaclab/simulator.py",
        "src/protomotions/simulator/isaaclab/utils/usd_utils.py",
        "src/CrowdSim/humanoid/runtime/sdk61_lifecycle.py", "src/CrowdSim/humanoid/runtime/trace_pace_scope.py",
    )}
    if not uses_sim61():
        raise RuntimeError("This output probe requires the pinned Sim 6.1 environment")
    lifecycle = SDK61Lifecycle(report)
    from isaaclab.app import AppLauncher
    # SDK61Lifecycle.kit_args disables the renderer for AMP; deliberately omit
    # that physics-only setting here and request the SDK rendering experience.
    app = AppLauncher({"headless": True, "enable_cameras": True, "device": args.device,
                       "fast_shutdown": False}).app
    owner = SimpleNamespace(world=None, physics_frame=None)
    failure = None
    try:
        run_probe(app, args, owner, report, lifecycle)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        _drop_exception_tracebacks(error)
        failure = error
    finally:
        try:
            report["frame_release"] = clear_completed_physics_frame(owner, run_probe)
        except Exception as error:
            report["cleanup_errors"].append(f"frame release: {type(error).__name__}: {error}")
        gc.collect()
        try:
            report["rendering_shutdown"] = drain_rtx_before_context_release(app)
        except Exception as error:
            report["cleanup_errors"].append(f"RTX drain: {type(error).__name__}: {error}")
            traceback.print_exc()
        try:
            lifecycle.close(app, owner.world)
        except Exception as error:
            report["cleanup_errors"].append(f"lifecycle: {type(error).__name__}: {error}")
            traceback.print_exc()
        owner.world = None
        gc.collect()
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RTX camera lifecycle: " + json.dumps(report), flush=True)
    if failure is not None:
        raise failure
    if not report["cleanup_succeeded"]:
        raise RuntimeError("RTX camera cleanup failed: " + "; ".join(report["cleanup_errors"]))


if __name__ == "__main__":
    main()
