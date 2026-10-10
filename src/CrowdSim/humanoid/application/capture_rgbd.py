"""Optional same-product depth capture around the existing RGB helpers.

This adapter performs no render update of its own. Matching render products and
read order do not establish actual RGB/depth synchronization or calibration.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
import json
import math
from pathlib import Path
import sys


@contextmanager
def capture_helpers(output, *, interval=1, depth=False, camera_eye=None, camera_target=None):
    """Temporarily adapt one evaluation's setup and bounded RGB frame helper.

    The original evaluation captures after controls 1, 1+interval, ... . Depth
    files use the corresponding original zero-based RGB frame filename. Kit's
    process fast exit can bypass Python cleanup; detach is only attempted when
    this context's exit is reached.
    """
    if type(interval) is not int or interval < 1 or type(depth) is not bool:
        raise ValueError("interval must be a positive integer and depth a boolean")
    if (camera_eye is None) != (camera_target is None):
        raise ValueError("camera_eye and camera_target must be supplied together")
    if camera_eye is not None:
        camera_eye, camera_target = tuple(camera_eye), tuple(camera_target)
        if any(len(value) != 3 or not all(math.isfinite(x) for x in value)
               for value in (camera_eye, camera_target)):
            raise ValueError("camera vectors must contain three finite coordinates")
    module = import_module("CrowdSim.humanoid.simulate")
    original_setup, original_frame = module._capture_setup, module._capture_frame
    output = Path(output)
    setup_called = False
    product = rgb = depth_annotator = index_file = None
    frame_index = 0

    def setup(env, viewer, width=960, height=720):
        nonlocal setup_called, product, rgb, depth_annotator, index_file
        if setup_called:
            raise RuntimeError("capture_helpers supports exactly one capture setup")
        setup_called = True
        selected_viewer = dict(viewer)
        if camera_eye is not None:
            selected_viewer.update(camera_eye=camera_eye, camera_target=camera_target)
        rep, product, rgb = original_setup(env, selected_viewer, width, height)
        if depth:
            folder = output / "depth"
            folder.mkdir(exist_ok=False)
            index_file = (folder / "index.jsonl").open("x", encoding="utf-8")
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            depth_annotator.attach(product)
        return rep, product, rgb

    def frame(env, annotator, warmup):
        nonlocal frame_index
        if not setup_called or rgb is None or annotator is not rgb:
            raise RuntimeError("capture frame must use this setup's RGB annotator")
        if type(warmup) is not bool or warmup != (frame_index == 0):
            raise RuntimeError("warmup must be true only for the first captured frame")
        pixels = original_frame(env, annotator, warmup)
        if depth:
            import numpy as np

            values = np.asarray(depth_annotator.get_data(), dtype=np.float32)
            if values.ndim != 2 or values.shape != pixels.shape[:2]:
                raise RuntimeError("Depth shape does not match the same-product RGB frame")
            if np.any(np.isnan(values) | np.isneginf(values) | (values < 0)):
                raise RuntimeError("Depth contains NaN, negative infinity or negative metres")
            control_step = frame_index * interval + 1
            relative = f"depth/frame_{control_step - 1:06d}.npy"
            with (output / relative).open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            valid = np.isfinite(values) & (values > 0)
            selected = values[valid]
            row = {
                "frame_index": frame_index, "control_step": control_step,
                "file": relative, "shape": list(values.shape), "dtype": "float32",
                "units": "metres", "valid_fraction": float(valid.mean()),
                "valid_min_m": float(selected.min()) if selected.size else None,
                "valid_max_m": float(selected.max()) if selected.size else None,
                "actual_synchronization_verified": False,
            }
            index_file.write(json.dumps(row, allow_nan=False) + "\n")
            index_file.flush()
        frame_index += 1
        return pixels

    module._capture_setup, module._capture_frame = setup, frame
    try:
        yield
    finally:
        failure = sys.exc_info()[1]
        module._capture_setup, module._capture_frame = original_setup, original_frame
        if index_file is not None:
            index_file.close()
        if depth_annotator is not None:
            try:
                depth_annotator.detach(product)
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Depth annotator detach also failed: {cleanup_error!r}")
