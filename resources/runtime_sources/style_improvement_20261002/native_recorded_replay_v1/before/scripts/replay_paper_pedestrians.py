#!/usr/bin/env python3
"""Render recorded SMPL poses without policy inference or physics integration.

This is recorded_pose_visualization. It cannot validate a fresh controller run,
reproduce physics contacts, or establish bitwise rendering/physics equivalence.
Preflight imports NumPy and the standard library only; Kit/Torch are lazy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RIG = ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def common_names():
    worldbody = ET.parse(RIG).getroot().find("worldbody")
    bodies = tuple(node.attrib["name"] for node in worldbody.iter("body"))
    dofs = tuple(node.attrib["name"] for node in worldbody.iter("joint") if node.attrib.get("type") == "hinge")
    if len(bodies) != 24 or len(dofs) != 69 or bodies[0] != "Pelvis":
        raise ValueError("Current SMPL common rig is not the expected24-body/69DOF contract")
    return bodies, dofs


def validate_scene(scene):
    if scene is None:
        return None
    if not isinstance(scene, dict) or not isinstance(scene.get("cuboids"), list):
        raise ValueError("Source scene must explicitly declare its cuboids")
    for box in scene["cuboids"]:
        for key in ("center", "size"):
            values = np.asarray(box.get(key), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all() or (key == "size" and (values <= 0).any()):
                raise ValueError("Source cuboid center/positive size must be finite3-vectors")
    return scene


def bind_source_path(states, result):
    declared = result.get("state_capture", {}).get("path")
    if not isinstance(declared, str) or not declared:
        raise ValueError("Source result must identify the recorded physical_states path")
    expected = Path(declared).expanduser()
    expected = (ROOT / expected if not expected.is_absolute() else expected).resolve()
    if expected != Path(states).resolve():
        raise ValueError("Source result identifies a different physical_states file")


def validate_arrays(arrays, result, *, start_step=0, steps=900, capture_every_steps=2):
    for name, value in (("start_step", start_step), ("steps", steps), ("capture_every_steps", capture_every_steps)):
        if type(value) is not int or value < (0 if name == "start_step" else 1):
            raise ValueError(f"{name} must be a valid integer")
    if result.get("status") != "passed" or not isinstance(result.get("state_capture"), dict):
        raise ValueError("Source must be a completed recorded physical rollout")
    capture = result["state_capture"]
    if capture.get("frequency_hz") != 30 or "quaternions XYZW, common SMPL names" not in capture.get("coordinates", ""):
        raise ValueError("Source requires explicit30Hz, world XYZ/XYZW/common SMPL coordinates")
    if capture.get("reset_mask") != "first post-step frame of each new episode; physical termination captured separately":
        raise ValueError("Source reset mask timestamp is not the supported before-frame contract")
    bodies, dofs = common_names()
    if tuple(capture.get("body_names", ())) != bodies:
        raise ValueError("Source common body names/order differs from current neutral SMPL rig")
    physics = result.get("physics", {})
    native = physics.get("dof_names", [])
    semantic = [re.sub(r"_x:([012])$", lambda match: "_" + "xyz"[int(match.group(1))], name) for name in native if isinstance(name, str)]
    if len(native) != 69 or len(set(semantic)) != 69 or set(semantic) != set(dofs):
        raise ValueError("Source declared joint names are not an exact semantic/D6 SMPL bijection")
    root = arrays.get("root_pos")
    if not isinstance(root, np.ndarray) or root.ndim != 3 or root.shape[2] != 3 or min(root.shape[:2]) < 1:
        raise ValueError("root_pos must have shape[T>=1,B>=1,3]")
    frames, actors = root.shape[:2]
    shapes = {"root_pos": (frames, actors, 3), "root_rot": (frames, actors, 4), "dof_pos": (frames, actors, 69),
              "body_pos": (frames, actors, 24, 3), "episode_ids": (frames, actors), "reset_mask": (frames, actors)}
    for name, shape in shapes.items():
        value = arrays.get(name)
        kind = "b" if name == "reset_mask" else "iu" if name == "episode_ids" else "f"
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in kind or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape{shape} and dtype kind{kind}")
    if not np.allclose(np.linalg.norm(arrays["root_rot"], axis=-1), 1., atol=1e-3, rtol=0):
        raise ValueError("Recorded root quaternions must be unit XYZW")
    if not np.allclose(arrays["body_pos"][:, :, 0], root, atol=1e-4, rtol=0):
        raise ValueError("Recorded root must match named Pelvis body position")
    boundaries = arrays["episode_ids"][1:] != arrays["episode_ids"][:-1]
    if not np.array_equal(boundaries, arrays["reset_mask"][1:]) or (np.diff(arrays["episode_ids"], axis=0) < 0).any():
        raise ValueError("Episode IDs and before-frame reset masks disagree")
    if start_step + steps > frames or result.get("actual_control_steps") != frames:
        raise ValueError("Requested source interval exceeds the recorded control steps")
    indices = list(range(start_step, start_step + steps, capture_every_steps))
    return {"frames": frames, "actors": actors, "frequency_hz": 30, "common_body_names": list(bodies), "common_dof_names": list(dofs),
            "captured_step_indices": indices, "captured_source_time_s": [(index + 1) / 30 for index in indices],
            "source_interval_s": [start_step / 30, (start_step + steps) / 30], "video_fps": 30 / capture_every_steps,
            "source_first_pose": "post-control frame0 at1/30s; initial pre-step pose is absent", "scene": validate_scene(result.get("evaluation_scene"))}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--capture-every-steps", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/smpl")
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[32, -12, 26])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[12, 12, .9])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a fresh path; recorded/user files are never overwritten")
    if min(args.width, args.height) < 16 or args.width % 2 or args.height % 2:
        parser.error("Video width/height must be positive even dimensions>=16")
    if not all(math.isfinite(x) for x in args.camera_eye + args.camera_target) or args.camera_eye == args.camera_target:
        parser.error("Camera eye/target must be distinct finite vectors")
    return args


def configure_recorded_capture(settings, timeline):
    """Use ungated Hydra RGB updates while holding the animation clock.

    A constant reference time cannot identify successive USD pose edits for the
    Replicator orchestrator. Its supported disabled mode leaves the RGB render
    product active without the DispatchSync reference-time gate.
    """
    settings.set_bool("/app/player/playSimulations", False)
    settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", False)
    settings.set_bool("/omni/replicator/disableAnnotatorGate", True)
    timeline.set_auto_update(False)
    timeline.play()
    timeline.set_auto_update(False)


def capture_recorded_frame(app, timeline, settings, annotator, *, width, height, initial_time, warmup=False):
    """Pump a bounded number of app updates, without orchestrator/physics steps."""
    minimum_updates, maximum_updates = (48, 120) if warmup else (3, 24)
    deadline = time.monotonic() + (120 if warmup else 30)
    before = float(timeline.get_current_time())
    if not math.isclose(before, initial_time, rel_tol=0., abs_tol=1e-9):
        raise RuntimeError("Render update advanced timeline time; recorded-only replay contract violated")
    last_error = "RGB annotator is empty or incorrectly sized"
    for update in range(maximum_updates):
        if time.monotonic() > deadline:
            break
        timeline.set_auto_update(False)
        settings.set_bool("/app/player/playSimulations", False)
        app.update()
        after = float(timeline.get_current_time())
        if not math.isclose(after, initial_time, rel_tol=0., abs_tol=1e-9):
            raise RuntimeError("Render update advanced timeline time; recorded-only replay contract violated")
        if update + 1 < minimum_updates:
            continue
        pixels = np.asarray(annotator.get_data())
        if pixels.ndim != 3 or pixels.shape[:2] != (height, width) or pixels.shape[-1] < 3 or pixels.dtype != np.uint8:
            last_error = "RGB annotator is empty, incorrectly sized, or not uint8"
            continue
        rgb = pixels[:, :, :3].copy()
        if rgb.max() == 0 or rgb.mean() < 3 or rgb.mean() > 245:
            last_error = "RGB annotator is black or severely under/overexposed"
            continue
        return rgb, {"timeline_before_s": before, "timeline_after_s": after, "app_updates": update + 1,
                     "rgb_mean": float(rgb.mean()), "rgb_max": int(rgb.max())}
    raise RuntimeError(f"Recorded-pose RGB capture exhausted its bounded app updates: {last_error}")


def stamp_recorded_frame(rgb, source_time_s):
    """Put the recorded-state provenance in every exported PNG/MP4 frame."""
    if not math.isfinite(source_time_s) or source_time_s < 0:
        raise ValueError("Replay source timestamp must be finite and nonnegative")
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(rgb)
    font_size = max(14, min(28, rgb.shape[1] // 55))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    text = f"Recorded physical state replay | source t={source_time_s:.3f}s"
    box = draw.textbbox((12, 8), text, font=font)
    draw.rectangle((0, 0, image.width, box[3] + 10), fill=(10, 14, 18))
    draw.text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(image).copy()


def main(argv=None):
    args = parse_args(argv)
    source_result = json.loads(args.source_result.read_text())
    bind_source_path(args.states, source_result)
    with np.load(args.states, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    preflight = validate_arrays(arrays, source_result, start_step=args.start_step, steps=args.steps, capture_every_steps=args.capture_every_steps)
    if not (args.model_dir / "SMPL_NEUTRAL.pkl").is_file():
        raise FileNotFoundError("Licensed neutral SMPL model is required for recorded-pose visualization")
    report = {"status": "preflight_passed", "mode": "recorded_pose_visualization", "rendering_only": True,
              "status_semantics": "renderer execution only; source physics outcomes remain separate",
              "closed_loop_policy_rollout": False, "physics_verification": False, "physical_humanoids_spawned": False,
              "frames": 0, "physics_steps": 0, "policy_forward_calls": 0, "physics_scene_created": False,
              "sources": {"states": file_record(args.states), "result": file_record(args.source_result), "common_rig": file_record(RIG)},
              "source_code": {name: file_record(ROOT / name) for name in (
                  "scripts/replay_paper_pedestrians.py", "CrowdSim/tools/smpl_mesh_visualizer.py",
                  "CrowdSim/humanoid/simulate.py", "protomotions/utils/rotations.py")},
              "preflight": preflight, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "reset_mask": "copied source before-frame flags; rendered poses are never interpolated across resets"}
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    app = None
    started = time.monotonic()
    try:
        from protomotions.utils.simulator_imports import import_simulator_before_torch
        launcher = import_simulator_before_torch("isaaclab")
        app = launcher({"headless": True, "device": "cuda:0", "enable_cameras": True}).app
        import carb
        import omni.timeline
        import omni.usd
        import torch
        torch.set_num_threads(4)
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux
        from importlib.metadata import version
        from CrowdSim.humanoid.simulate import _capture_setup, _write_video, _lab_source_version
        from CrowdSim.tools.smpl_mesh_visualizer import HumanMeshConfig, SMPLMeshVisualizer, SMPLRobotPoseMapper
        settings = carb.settings.get_settings()
        settings.set_bool("/app/player/playSimulations", False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        def cube(path, center, size, color):
            prim = UsdGeom.Cube.Define(stage, path)
            prim.CreateSizeAttr(1.)
            prim.AddTranslateOp().Set(Gf.Vec3d(*center))
            prim.AddScaleOp().Set(Gf.Vec3f(*size))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        extent = float(source_result["physics"].get("terrain_extent_m", 64.))
        cube("/World/RecordedGround", [extent/2, extent/2, -.05], [extent, extent, .1], [.55, .55, .55])
        for index, box in enumerate((preflight["scene"] or {}).get("cuboids", [])):
            cube(f"/World/PaperValidation/Obstacle_{index}", box["center"], box["size"], [.2, .3, .4])
        UsdLux.DomeLight.Define(stage, "/World/ReplayDome").CreateIntensityAttr(400.)
        sun = UsdLux.DistantLight.Define(stage, "/World/ReplaySun")
        sun.CreateIntensityAttr(1500.)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 0))
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        settings.set("/rtx/post/tonemap/op", 4)
        settings.set("/rtx/post/tonemap/filmIso", 200.)
        textures = tuple(str(p) for p in sorted((args.model_dir / "textures").glob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        visual = SMPLMeshVisualizer.from_config(HumanMeshConfig(model_dir=str(args.model_dir), num_envs=preflight["actors"], device="cpu",
            texture_paths=textures, appearance_seed=int(source_result.get("arguments", {}).get("seed", 7)), shape_std=0., hide_humanoid=False))
        visual.create()
        lab_product, lab_commit = _lab_source_version()
        report["runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__,
            "isaacsim": version("isaacsim"), "isaaclab_package": version("isaaclab"),
            "isaaclab_product": lab_product, "isaaclab_commit": lab_commit,
            "mesh_compute_device": "cpu", "render_mode": settings.get("/rtx/rendermode")}
        report["visual_resources"] = {"textures": [file_record(path) for path in dict.fromkeys(visual.person_texture_paths)],
            "actor_texture_paths": list(visual.person_texture_paths),
            "uv": file_record(args.model_dir / "smpl_uv.npz") if (args.model_dir / "smpl_uv.npz").is_file() else None}
        mapper = SMPLRobotPoseMapper()
        # Set this before attaching the RGB annotator: attach() authors its
        # DispatchSync gate from this supported setting in the installed SDK.
        settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", False)
        settings.set_bool("/omni/replicator/disableAnnotatorGate", True)
        capture = _capture_setup(None, {"camera_eye": args.camera_eye, "camera_target": args.camera_target}, args.width, args.height)
        capture[0].orchestrator.set_capture_on_play(False)
        capture[1].hydra_texture.set_updates_enabled(True)
        initial_time = float(timeline.get_current_time())
        configure_recorded_capture(settings, timeline)
        report["capture_contract"] = {"method": "bounded app.update with ungated active Hydra RGB render product",
            "orchestrator_enabled": False, "capture_on_play": False, "annotator_reference_time_gate": False,
            "timeline_playing": True, "timeline_auto_update": False, "play_simulations": False,
            "maximum_app_updates_first_frame": 120, "maximum_app_updates_other_frames": 24,
            "frame_banner": "Recorded physical state replay | source t={post-control source timestamp}s"}
        rendered = []
        frame_stats = []
        report["frame_records"] = frame_stats
        for frame_number, index in enumerate(preflight["captured_step_indices"]):
            q = torch.from_numpy(arrays["root_rot"][index]).float()
            pose = mapper(torch.from_numpy(arrays["dof_pos"][index]).float(), preflight["common_dof_names"])
            visual.update(pose, torch.from_numpy(arrays["root_pos"][index]).float(), q[:, [3, 0, 1, 2]])
            rgb, capture_stats = capture_recorded_frame(app, timeline, settings, capture[2], width=args.width,
                height=args.height, initial_time=initial_time, warmup=frame_number == 0)
            rgb = stamp_recorded_frame(rgb, (index+1)/30)
            path = args.output / "frames" / f"frame_{frame_number:05d}.png"
            Image.fromarray(rgb).save(path)
            rendered.append(str(path))
            report["frames"] = len(rendered)
            frame_stats.append({"frame": frame_number, "source_step_index": index, "source_time_s": (index+1)/30,
                "source_reset_mask": arrays["reset_mask"][index].tolist(), "source_episode_ids": arrays["episode_ids"][index].tolist(),
                **capture_stats})
            if (frame_number + 1) % 150 == 0:
                print(json.dumps({"mode": "recorded_pose_visualization", "rendered_frames": frame_number+1,
                                  "total_frames": len(preflight["captured_step_indices"]), "elapsed_wall_s": time.monotonic()-started}), flush=True)
        report.update(status="passed", video=_write_video(args.output, rendered, preflight["video_fps"]),
                      frames=len(rendered), frame_records=frame_stats, timeline_time_before_s=initial_time,
                      timeline_time_after_s=float(timeline.get_current_time()), simulation_time_advanced_s=0.,
                      physics_steps=0, policy_forward_calls=0, mesh_mode="textured" if visual.has_texture else "solid-color",
                      licensed_neutral_model=file_record(args.model_dir / "SMPL_NEUTRAL.pkl"),
                      video_duration_at_declared_fps_s=len(rendered)/preflight["video_fps"], elapsed_wall_s=time.monotonic()-started)
        report["video_file"] = file_record(report["video"])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close()
    print(json.dumps({k: report.get(k) for k in ("status", "mode", "frames", "video", "error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
