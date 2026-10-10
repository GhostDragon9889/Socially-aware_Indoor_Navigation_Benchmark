"""Run real SMPL pedestrian physics with the local MaskedMimic controller."""

from __future__ import annotations

from protomotions.repository import repository_root

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time
import subprocess
import traceback

PROJECT_ROOT = repository_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="CrowdSim/config/pedestrian.yaml")
    parser.add_argument("--num-humanoids", type=int, default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the GUI closes or Ctrl+C")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-mesh", action="store_true", help="Explicit physical-skeleton mode")
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--reset-at", type=int, default=None, help="Force a last-humanoid subset reset at this step")
    parser.add_argument("--output-dir", default="output/pedestrian_simulation")
    parser.add_argument("--capture", action="store_true", help="Save RGB frames and a pedestrian motion MP4")
    parser.add_argument("--capture-every", type=int, default=3, help="Capture one frame per N control steps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error("Headless execution needs a positive --steps bound")
    if args.num_humanoids is not None and args.num_humanoids < 1:
        parser.error("--num-humanoids must be positive")
    if args.capture_every < 1:
        parser.error("--capture-every must be positive")
    return args


def _capture_setup(env, viewer, width=960, height=720):
    import omni.replicator.core as rep
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_rtx_hydra_engine_attached
    # The headless rendering experience has no viewport window to attach the
    # Hydra engine. Follow IsaacLab's camera renderer initialization contract.
    ensure_rtx_hydra_engine_attached()
    rep.orchestrator.set_capture_on_play(True)
    camera = rep.create.camera(
        position=tuple(viewer["camera_eye"]), look_at=tuple(viewer["camera_target"]),
        focal_length=20.0,
    )
    product = rep.create.render_product(camera, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)
    return rep, product, annotator


def _capture_frame(env, annotator, warmup):
    """Pump a bounded number of render updates without advancing physics."""
    import carb
    import numpy as np
    sim = env.simulator._sim
    settings = carb.settings.get_settings()
    previous = settings.get("/app/player/playSimulations")
    sim.physics_manager.forward()
    try:
        settings.set_bool("/app/player/playSimulations", False)
        # Replicator's synchronous reference-time fence can wait indefinitely
        # when IsaacLab owns physics. Its camera backend instead pumps Kit and
        # reads annotators; use that same path with bounded startup warmup.
        for _ in range(12 if warmup else 3):
            env.simulator._simulation_app.update()
        pixels = np.asarray(annotator.get_data())
        if pixels.ndim != 3 or pixels.shape[-1] < 3 or pixels.size == 0:
            raise RuntimeError("Pedestrian RTX annotator did not produce an RGB frame")
        return pixels[:, :, :3].copy()
    finally:
        settings.set_bool("/app/player/playSimulations", bool(previous))


def _lab_source_version():
    import isaaclab
    for candidate in Path(isaaclab.__file__).resolve().parents:
        if (candidate / "VERSION").is_file() and (candidate / "source/isaaclab").is_dir():
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=candidate, capture_output=True, text=True)
            return (candidate / "VERSION").read_text().strip(), result.stdout.strip() or None
    return None, None


def _write_video(output, frames, fps):
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    path = output / "pedestrian_motion.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_videofile(str(path), codec="libx264", audio=False, logger=None, threads=2)
    finally:
        clip.close()
    return str(path)


def main():
    args = parse_args()
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    import_simulator_before_torch("isaaclab")
    import numpy as np
    import torch
    from CrowdSim.utils.config_loader import cfg_path, load_config
    from CrowdSim.world.builder import build_env

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(cfg_path(args.config, PROJECT_ROOT))
    if args.num_humanoids is not None:
        config["navigation"]["num_humanoids"] = args.num_humanoids
    if args.control_hz is not None:
        config["navigation"]["update_hz"] = args.control_hz
    config["navigation"]["num_robots"] = 0
    config["car"].update(usd=None, rl_policy=False)
    sensors = config.setdefault("sensors", {})
    sensors.setdefault("camera", {})["enabled"] = False
    sensors.setdefault("lidar", {})["enabled"] = False
    config["humanoid"]["human_mesh"] = not args.no_mesh
    config.setdefault("runtime", {})["enable_rendering"] = bool(args.capture)
    output = cfg_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = runtime = nav = capture = None
    frames = []
    roots, root_rotations, dofs, sim_times, reset_history = [], [], [], [], []
    motion_id_history, motion_time_history = [], []
    completed = resets = 0
    initial_state = None
    started = time.monotonic()
    status = "failed"
    error = None
    failure_state = None
    normalizer_snapshots = {}
    video_path = None
    try:
        env, agent, nav, runtime = build_env(config, headless=args.headless)
        app = env.simulator._simulation_app
        if args.capture:
            capture = _capture_setup(env, config["viewer"])
        for name, value in agent.model.named_buffers():
            if "running_obs_norm" in name:
                normalizer_snapshots[name] = value.detach().clone()
        done_ids = None
        while app.is_running() and env.simulator.is_simulation_running() and (args.steps == 0 or completed < args.steps):
            if args.reset_at is not None and completed == args.reset_at:
                done_ids = torch.tensor([env.num_envs - 1], device=env.device, dtype=torch.long)
            if done_ids is not None:
                resets += int(done_ids.numel())
            reset_mask = np.ones(env.num_envs, dtype=bool) if done_ids is None else np.zeros(env.num_envs, dtype=bool)
            if done_ids is not None:
                reset_mask[done_ids.detach().cpu().numpy()] = True
            obs, _ = env.reset(done_ids)
            with torch.no_grad():
                obs = agent.add_agent_info_to_obs(obs)
                outputs = agent.model(agent.obs_dict_to_tensordict(obs))
                action = outputs["mean_action"] if "mean_action" in outputs else outputs["action"]
                if not torch.isfinite(action).all():
                    raise RuntimeError("Pedestrian policy produced non-finite actions")
                _, _, dones, _, _ = env.step(action)
            if not args.no_mesh and getattr(nav, "_human_mesh_adapter", None) is None:
                raise RuntimeError("Requested SMPL mesh stopped updating")
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            state = env.simulator.get_robot_state()
            if not all(torch.isfinite(value).all() for value in (state.root_pos, state.root_rot, state.dof_pos)):
                raise RuntimeError("Pedestrian physical state is non-finite")
            if (state.root_pos[:, 2] < 0.35).any():
                failure_state = {
                    "step": completed + 1,
                    "below_height_threshold_env_ids": (state.root_pos[:, 2] < 0.35).nonzero(
                        as_tuple=False
                    ).flatten().detach().cpu().tolist(),
                    "root_pos": state.root_pos.detach().cpu().tolist(),
                    "root_rot_xyzw": state.root_rot.detach().cpu().tolist(),
                    "motion_ids": env.motion_manager.motion_ids.detach().cpu().tolist(),
                    "motion_times": env.motion_manager.motion_times.detach().cpu().tolist(),
                }
                raise RuntimeError("Pedestrian fell or passed through the ground (pelvis below 0.35m)")
            root = state.root_pos.detach().cpu().numpy().copy()
            if initial_state is None:
                initial_state = root.copy()
            roots.append(root)
            root_rotations.append(state.root_rot.detach().cpu().numpy().copy())
            motion_id_history.append(env.motion_manager.motion_ids.detach().cpu().numpy().copy())
            motion_time_history.append(env.motion_manager.motion_times.detach().cpu().numpy().copy())
            dofs.append(state.dof_pos.detach().cpu().numpy().copy())
            sim_times.append((completed + 1) * env.dt)
            reset_history.append(reset_mask)
            if capture is not None and completed % args.capture_every == 0:
                _, _, annotator = capture
                pixels = _capture_frame(env, annotator, warmup=not frames)
                after_capture = env.simulator.get_robot_state()
                if not torch.equal(state.root_pos, after_capture.root_pos) or not torch.equal(state.dof_pos, after_capture.dof_pos):
                    raise RuntimeError("RGB capture advanced the pedestrian physical state")
                if pixels.size == 0 or not np.isfinite(pixels).all() or pixels.mean() < 5:
                    raise RuntimeError("Pedestrian RGB capture is empty or dark")
                from PIL import Image
                Image.fromarray(pixels).save(output / f"frame_{completed:06d}.png")
                frames.append(pixels)
            completed += 1
            if completed % 100 == 0:
                print(f"[Pedestrian] step={completed} t={completed * env.dt:.2f}s resets={resets}", flush=True)
        if args.steps > 0 and completed != args.steps:
            raise RuntimeError(f"App stopped after {completed}/{args.steps} requested steps")
        if len(frames) > 1 and all(np.array_equal(frames[0], frame) for frame in frames[1:]):
            raise RuntimeError("Pedestrian RGB frames did not update")
        for name, value in agent.model.named_buffers():
            if name in normalizer_snapshots and not torch.equal(value, normalizer_snapshots[name]):
                raise RuntimeError(f"Frozen pedestrian observation statistics changed: {name}")
        # Sim 6's fast shutdown terminates Python. Finish all deliverables first.
        if frames:
            video_path = _write_video(output, frames, fps=1.0 / (env.dt * args.capture_every))
        status = "passed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        lab_product, lab_commit = _lab_source_version()
        mesh_adapter = None if nav is None else getattr(nav, "_human_mesh_adapter", None)
        mesh_visualizer = None if mesh_adapter is None else mesh_adapter.visualizer
        report = {
            "status": status, "error": error,
            "failure_state": failure_state,
            "isaacsim": version("isaacsim"), "isaaclab_package": version("isaaclab"),
            "isaaclab_product": lab_product, "isaaclab_commit": lab_commit,
            "steps": completed, "reset_count": resets,
            "control_hz": None if env is None else 1.0 / env.dt,
            "simulated_seconds": None if env is None else completed * env.dt,
            "wall_seconds": time.monotonic() - started,
            "mesh_requested": not args.no_mesh,
            "mesh_created": mesh_adapter is not None,
            "mesh_appearance": None if mesh_visualizer is None else (
                "textured" if mesh_visualizer.has_texture else "solid-color"
            ),
            "texture_pool": [] if mesh_visualizer is None else list(mesh_visualizer.texture_paths),
            "person_textures": [] if mesh_visualizer is None else list(mesh_visualizer.person_texture_paths),
            "reset_motion_manifest": None if env is None else getattr(
                env, "_crowdsim_reset_motion_manifest", None
            ),
            "pelvis_height_above_ground_m": config["humanoid"].get("pelvis_height_above_ground"),
            "normalizer_buffers_checked": len(normalizer_snapshots),
            "controller": "SFM + SMPL MaskedMimic", "physics_backend": "IsaacLab PhysX",
            "shared_world": None if env is None else not env.simulator._scene.cfg.filter_collisions,
            "root_displacement_m": None if not roots else np.linalg.norm(roots[-1][:, :2] - initial_state[:, :2], axis=1).tolist(),
            "minimum_root_height_m": None if not roots else float(np.stack(roots)[:, :, 2].min()),
            "seed": args.seed,
            "captured_frames": len(frames),
            "video": video_path,
            "capture_hz": None if env is None or capture is None else 1.0 / (env.dt * args.capture_every),
            "walked_distance_m": None if len(roots) < 2 else np.where(
                np.stack(reset_history)[1:], 0.0,
                np.linalg.norm(np.diff(np.stack(roots)[:, :, :2], axis=0), axis=2),
            ).sum(axis=0).tolist(),
            "mean_joint_std_rad": None if not dofs else np.stack(dofs).std(axis=0).mean(axis=1).tolist(),
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if roots:
            np.savez_compressed(
                output / "physical_states.npz", times=np.array(sim_times),
                root_pos=np.stack(roots), root_rot_xyzw=np.stack(root_rotations),
                dof_pos=np.stack(dofs), reset_mask=np.stack(reset_history),
                motion_ids=np.stack(motion_id_history), motion_times=np.stack(motion_time_history),
            )
        print(json.dumps(report, indent=2), flush=True)
        if nav is not None:
            nav._close_trajectory_logs()
        if capture is not None:
            capture[0].orchestrator.stop()
        if env is not None:
            env.simulator._shutdown_exit_code = 1 if status == "failed" else 0
            env.close()


if __name__ == "__main__":
    main()
