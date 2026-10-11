"""Output-only actual production close probe; requires an exclusively available GPU."""
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
        from isaaclab.assets import RigidObjectCfg
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
                cfg.crowdsim_robot = cfg.robot.copy()
                cfg.crowdsim_robot.prim_path = "{ENV_REGEX_NS}/CrowdRobot"
                cfg.crowdsim_robot.init_state.pos = (-4., 0., 1.)
                cfg.rtx_probe_target = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/RTXProbeTarget",
                    spawn=sim_utils.CuboidCfg(size=(1., 1., 1.),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .1, .1))),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 1.)),
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
    import numpy as np
    from PIL import Image
    import torch
    from CrowdSim.control.drive import articulation_torch_view
    from CrowdSim.tools.test_camera_depth import add_camera_check_key, record_headless_camera_burst
    from CrowdSim.utils.sensor_stream import RobotCameraRecorder

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



def main():
    parser = argparse.ArgumentParser(description="Actual BaseEnv.close -> IsaacLabSimulator.close, dual articulations/rigid/camera")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--fail-after-initialize", action="store_true")
    parser.add_argument("--defer-caller-gc", action="store_true")
    parser.add_argument("--initial-full-shutdown", action="store_true")
    parser.add_argument("--fail-component-close", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "completed_control_steps": 0, "num_envs": args.num_envs,
              "scope": "real BaseEnv.close method on a minimal shell; actual dual H1-2/rigid/camera scene, not full CrowdSim task/checkpoint",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle
    from isaaclab.app import AppLauncher
    # Deliberately exercise the production default. Production close must turn
    # quick shutdown off and return from full App.close, not terminate early.
    app = AppLauncher({"headless": True, "enable_cameras": True, "device": args.device,
                       "fast_shutdown": not args.initial_full_shutdown}).app
    from protomotions.envs.base_env.env import BaseEnv
    from CrowdSim.control.drive import DriveController
    from isaaclab_physx.physics import PhysxManager
    from protomotions.simulator.isaaclab.utils.sdk61_owned_close import owned_keyboard_class
    report["initial_fast_shutdown"] = app.config["fast_shutdown"]
    owner = SimpleNamespace(world=None)
    lifecycle = SDK61Lifecycle({"status": "starting"})
    env = None
    try:
        owner.world = lifecycle.construct_world(CameraWorld, app, args)
        simulator = owner.world.simulator
        env = BaseEnv.__new__(BaseEnv)
        env.simulator = simulator
        env.control_manager = None
        env._key_bindings = None
        env.crowdsim_robot = simulator._scene.articulations["crowdsim_robot"]
        env.crowdsim_robot_camera = simulator._scene.sensors["rtx_probe_camera"]
        robot = simulator._scene.articulations["robot"]
        box = simulator._scene.rigid_objects["rtx_probe_target"]
        camera = env.crowdsim_robot_camera
        nav = SimpleNamespace(robot=env.crowdsim_robot,
            drive=DriveController(env.crowdsim_robot, SimpleNamespace(num_robots=args.num_envs), owner.world.device))
        # Real callbacks retain caller objects. Native GUI input itself is not
        # exercised by this headless probe (the explicit keyboard release has CPU tests).
        callbacks = {"camera": lambda: env.crowdsim_robot_camera, "navigation": lambda: nav.drive.robot}
        class CallbackKeyboard:
            def __init__(self):
                self._additional_callbacks = callbacks.copy()
        keyboard = owned_keyboard_class(CallbackKeyboard)()
        simulator.keyboard_interface = keyboard
        retained_component_tracebacks = []
        component_calls = [0]
        def component_close():
            component_calls[0] += 1
            if component_calls[0] > 1:
                return
            native_alias = PhysxManager.get_scene_data_backend().simulation_view
            assert native_alias is not None
            try:
                raise RuntimeError("Intentional component cleanup error with logger-retained traceback")
            except RuntimeError as error:
                retained_component_tracebacks.append(error.__traceback__)
                raise
        if args.fail_component_close:
            env.control_manager = SimpleNamespace(components={"failing_component": SimpleNamespace(close=component_close)})
        report["native_scene_before"] = {
            "articulations": sorted(simulator._scene.articulations),
            "rigid_objects": sorted(simulator._scene.rigid_objects),
            "sensors": sorted(simulator._scene.sensors)}
        def injected_failure():
            # This completed frame owns a real native SimulationView until the
            # active exception chain is diagnosed and released by production.
            native_alias = PhysxManager.get_scene_data_backend().simulation_view
            assert native_alias is not None
            raise RuntimeError("Intentional production failure after camera initialization")
        try:
            if args.fail_after_initialize:
                injected_failure()
            run_probe(app, args, owner, report, lifecycle)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            # Intentionally keep env/nav/assets/camera/callbacks and an active
            # original exception alive while the real production method runs.
            report["active_error_at_close"] = __import__("sys").exception() is not None
            try:
                close_error = None
                try:
                    env.close()
                except BaseException as error:
                    if not args.fail_component_close:
                        raise
                    close_error = error
                    report.update(status="failed", expected_component_close_error=f"{type(error).__name__}: {error}")
                report["production_close_returned"] = close_error is None
                if args.fail_component_close:
                    assert close_error is not None and str(close_error) == "Intentional component cleanup error with logger-retained traceback"
                    report["component_retained_traceback_frame_cleared"] = all(
                        tb.tb_frame.f_locals == {} for tb in retained_component_tracebacks)
                    report["component_callback_calls_before_repeat"] = component_calls[0]
                    assert report["component_retained_traceback_frame_cleared"]
                    assert simulator._sdk61_close_report["cleanup_succeeded"]
                    assert simulator._sdk61_close_report["app_exit_code"] == 1
                first_report = simulator._sdk61_close_report
                env.close()
                report["repeat_close_returned_same_report"] = simulator._sdk61_close_report is first_report
                report["external_aliases_preserved"] = (callbacks["camera"]() is camera
                    and callbacks["navigation"]() is env.crowdsim_robot and nav.robot is env.crowdsim_robot)
                report["native_fields_released"] = {
                    name: {field: vars(asset).get(field) is None for field in ("_root_view", "_physics_sim_view")}
                    for name, asset in (("robot", robot), ("crowdsim_robot", env.crowdsim_robot), ("rigid", box))}
                report["data_backend_aliases_released"] = all(
                    vars(asset._data).get("_physics_sim_view") is None for asset in (robot, env.crowdsim_robot, box))
                report["camera_native_fields_released"] = all(vars(camera).get(field) is None
                    for field in ("_renderer", "_render_data", "_view"))
                report["keyboard_callback_references_released"] = keyboard._additional_callbacks == {}
                report["final_fast_shutdown"] = app.config["fast_shutdown"]
                assert all(all(fields.values()) for fields in report["native_fields_released"].values())
                assert report["data_backend_aliases_released"] and report["camera_native_fields_released"]
                assert report["external_aliases_preserved"] and report["keyboard_callback_references_released"]
                assert not report["final_fast_shutdown"]
            finally:
                report["production_cleanup"] = getattr(simulator, "_sdk61_close_report", {})
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print("Production camera close: " + json.dumps(report), flush=True)
                if args.defer_caller_gc:
                    # Diagnostic only: force caller-owned reference cycles to
                    # survive until interpreter finalization, without dropping
                    # references or skipping native App cleanup.
                    gc.disable()
                    print("DIAGNOSTIC: automatic caller GC disabled after full close", flush=True)
            if close_error is not None:
                raise close_error
    finally:
        if env is None:
            # Constructor failures are not the injected acceptance path. Keep
            # the prior proven owner cleanup so a setup error is diagnosable.
            from importlib.util import spec_from_file_location, module_from_spec
            original = Path(__file__).with_name("validate_rtx_camera.py")
            spec = spec_from_file_location("original_rtx_probe", original)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                module.drain_rtx_before_context_release(app)
            finally:
                lifecycle.close(app, owner.world)


if __name__ == "__main__":
    main()
