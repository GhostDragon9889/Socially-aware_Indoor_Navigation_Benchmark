"""Bounded live SMPL capture through the existing Diffusion + AMP driver.

Preflight imports no tensor or simulator libraries. Rendering is an explicit
evaluation request; the original driver owns the only App, world and clock.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os

from . import deployment
from .configuration import load_amp_profile, parse_amp_args
from .resources import preflight_amp_resources


def parse_capture_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Also accepts the deployment arguments shown by scripts/simulate_paper_pedestrians.py --help.")
    parser.add_argument("--depth", action="store_true", help="Also save raw camera-plane depth in metres")
    parser.add_argument("--capture-every-steps", type=int, default=2)
    parser.add_argument("--camera-eye", type=float, nargs=3)
    parser.add_argument("--camera-target", type=float, nargs=3)
    capture, remainder = parser.parse_known_args(argv)
    args = deployment.parse_deployment_args(remainder)
    if not 1 <= capture.capture_every_steps <= 30:
        parser.error("Capture interval must be within 1..30 control steps")
    if (capture.camera_eye is None) != (capture.camera_target is None):
        parser.error("Provide both --camera-eye and --camera-target")
    if capture.camera_eye is not None:
        if not all(math.isfinite(value) for value in capture.camera_eye + capture.camera_target):
            parser.error("Camera coordinates must be finite world metres")
        if capture.camera_eye == capture.camera_target:
            parser.error("Camera eye and target must differ")
    elif args.native_dynbench:
        parser.error("Native capture needs explicit --camera-eye and --camera-target in scene coordinates")
    return args, capture


def capture_driver_arguments(args, capture):
    forwarded = deployment.driver_arguments(args)
    forwarded.remove("--physics-only")
    index = forwarded.index("--capture-every-steps")
    forwarded[index + 1] = str(capture.capture_every_steps)
    forwarded.append("--capture")
    return forwarded


@contextmanager
def _visual_asset_environment():
    # The existing driver's enable_human_mesh selects these exact local paths.
    # Preflight must inspect the same files, rather than a stale model override.
    values = {"CROWDSIM_SMPL_MODEL_DIR": str(deployment.ROOT / "data/smpl"),
              "CROWDSIM_SMPL_TEXTURE_DIR": str(deployment.ROOT / "data/smpl/textures")}
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def preflight_capture(args, capture):
    report = deployment.preflight_deployment(args)
    forwarded = capture_driver_arguments(args, capture)
    driver_args = parse_amp_args(forwarded)
    with _visual_asset_environment():
        resources = preflight_amp_resources(driver_args, load_amp_profile(driver_args.config))
    if not resources.optional_capture_textures_present:
        raise ValueError("Textured capture requires data/smpl/smpl_uv.npz and a texture in data/smpl/textures")
    report.update(schema="navisaaclab.live_pedestrian_capture_preflight.v1",
                  entry=str(deployment.ROOT / "scripts/capture_paper_pedestrians.py"),
                  driver_argv=forwarded, parsed_driver_arguments=vars(driver_args),
                  resource_preflight=resources.as_dict())
    report["deployment_contract"]["physics_only"] = False
    report["capture_request"] = {
        "rgb": True, "depth": capture.depth, "width": 960, "height": 720,
        "interval_control_steps": capture.capture_every_steps,
        "video_fps": 30 / capture.capture_every_steps,
        "camera_eye_world_m": capture.camera_eye, "camera_target_world_m": capture.camera_target,
        "default_camera": "existing cuboid fixture camera" if capture.camera_eye is None else None,
        "depth_annotator": "distance_to_image_plane" if capture.depth else None,
        "depth_units": "metres along optical axis" if capture.depth else None,
        "same_render_product_requested": True,
        "sensor_frame_synchronization_verified": False,
        "actual_camera_calibration": None,
        "physics_capture_diagnostics": "existing driver records actual state, step counter and RNG before/after capture",
        "physical_execution_verified": False,
    }
    report["limitations"].extend([
        "Presence checks do not validate UVs, textures, sensor buffers or camera framing.",
        "Depth uses the same product and bounded render updates; actual sensor synchronization remains to be checked.",
        "Rendering is a separate evaluation; it does not inherit a physics-only run's results or prove paper parity.",
    ])
    return report


def main(argv=None, *, driver_main=None):
    args, capture = parse_capture_args(argv)
    report = preflight_capture(args, capture)
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0
    if driver_main is None:
        deployment.require_fresh_sdk_process()
    for name in deployment.THREAD_ENVIRONMENT:
        os.environ[name] = "4"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "deployment_entry.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    from .capture_rgbd import capture_helpers
    with _visual_asset_environment(), capture_helpers(
            args.output, interval=capture.capture_every_steps,
            depth=capture.depth, camera_eye=capture.camera_eye,
            camera_target=capture.camera_target):
        result = deployment.invoke_existing_driver(
            report["driver_argv"], driver_main,
            evaluation_actor_request=report["evaluation_actor_request"],
            native_scene_request=report["native_scene_request"])
    return 0 if result is None else result
