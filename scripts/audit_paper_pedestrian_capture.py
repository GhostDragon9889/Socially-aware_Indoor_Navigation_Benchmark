#!/usr/bin/env python3
"""Check completed capture files and recorded state invariance on CPU.

This checks payload consistency. The root launch/exit record remains the source
of execution provenance; neither synthetic test data nor file checks prove it.
"""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    import numpy as np
    from PIL import Image

    entry = json.loads((directory / "deployment_entry.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    require(entry["schema"] == "navisaaclab.live_pedestrian_capture_preflight.v1", "Wrong capture entry")
    require(result["status"] == "passed", "The driver did not complete successfully")
    args, request = result["arguments"], entry["capture_request"]
    require(args["capture"] and not args["physics_only"] and args["deterministic_evaluation"],
            "Missing rendered deterministic evaluation")
    require(args == entry["parsed_driver_arguments"], "Runtime arguments differ from preflight")
    steps, interval = args["evaluate_steps"], args["capture_every_steps"]
    require(type(steps) is int and steps > 0 and type(interval) is int and 1 <= interval <= 30,
            "Invalid capture clock")
    expected = [f"frame_{step:06d}" for step in range(0, steps, interval)]
    require(sorted(path.stem for path in (directory / "frames").glob("frame_*.png")) == expected,
            "RGB frame set differs from the requested control clock")
    diagnostics = result["capture_state_diagnostics"]
    require(len(diagnostics) == len(expected), "Missing per-frame state diagnostics")
    unavailable_clocks = 0
    for index, (name, row) in enumerate(zip(expected, diagnostics)):
        require(row["frame"] == name + ".png" and row["control_step"] == index * interval + 1,
                "Capture diagnostic is attached to the wrong control step")
        require(row["warmup"] is (index == 0), "Wrong capture warmup position")
        deltas = row["max_absolute_buffer_deltas"]
        require(set(deltas) == {"root_pos", "root_rot", "root_vel", "dof_pos", "dof_vel", "rigid_body_pos"}
                and all(type(value) in (int, float) and value == 0 for value in deltas.values()),
                "Rendering changed observed physical state")
        counter = row["physics_step_count"]
        require(all(type(counter[key]) is int for key in ("before", "after", "delta"))
                and counter["before"] >= 0 and counter["before"] == counter["after"]
                and counter["delta"] == 0, "Rendering advanced the recorded physics step counter")
        clock = row["native_physx_time_s"]
        require(type(clock["observed"]) is bool, "Invalid native clock availability flag")
        if clock["observed"]:
            require(all(type(clock[key]) in (int, float) and math.isfinite(clock[key])
                        for key in ("before", "after", "delta"))
                    and clock["before"] == clock["after"] and clock["delta"] == 0,
                    "Rendering advanced the observed native physics clock")
        else:
            unavailable_clocks += 1
            require(clock["before"] is None and clock["after"] is None and clock["delta"] is None,
                    "An unobserved clock must remain unknown")
        require(set(row["rng"]) == {"torch_cpu_rng", "torch_cuda_rng"}, "Missing required RNG channels")
        for rng in row["rng"].values():
            require(rng["observed"] is True and rng["equal"] is True
                    and isinstance(rng["before_sha256"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", rng["before_sha256"]) is not None
                    and rng["before_sha256"] == rng["after_sha256"], "RNG invariance was not observed")
        with Image.open(directory / "frames" / (name + ".png")) as image:
            require(image.size == (request["width"], request["height"]), "Wrong RGB resolution")
            image.verify()
    if request["depth"]:
        rows = [json.loads(line) for line in (directory / "depth/index.jsonl").read_text().splitlines()]
        require(len(rows) == len(expected), "Depth index does not cover every RGB frame")
        require(sorted(path.stem for path in (directory / "depth").glob("frame_*.npy")) == expected,
                "Depth file set differs from RGB")
        for index, (name, row) in enumerate(zip(expected, rows)):
            require(row["control_step"] == index * interval + 1, "Depth is attached to the wrong control step")
            depth = np.load(directory / "depth" / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            require(depth.dtype == np.float32 and depth.shape == (request["height"], request["width"]),
                    "Wrong depth type or resolution")
            require(not np.isnan(depth).any() and not (depth < 0).any(), "Invalid negative or NaN depth")
            valid = np.isfinite(depth) & (depth > 0)
            selected = depth[valid]
            expected_row = {"frame_index": index, "file": f"depth/{name}.npy", "shape": list(depth.shape),
                            "dtype": "float32", "units": "metres", "valid_fraction": float(valid.mean()),
                            "valid_min_m": float(selected.min()) if selected.size else None,
                            "valid_max_m": float(selected.max()) if selected.size else None}
            require(all(row.get(key) == value for key, value in expected_row.items())
                    and row.get("actual_synchronization_verified") is False,
                    "Depth index metadata differs from its payload")
    video = directory / "pedestrian_motion.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(video)],
                           check=True, capture_output=True, text=True)
    media = json.loads(probe.stdout)["streams"]
    require(len(media) == 1, "Video must contain one camera stream")
    media = media[0]
    require(media["width"] == request["width"] and media["height"] == request["height"]
            and int(media["nb_frames"]) == len(expected)
            and Fraction(media["avg_frame_rate"]) == Fraction(30, interval), "Video cadence/size/count mismatch")
    return {"status": "passed", "scope": "capture payload and observed state invariance only",
            "frames": len(expected), "depth": request["depth"], "video": media,
            "unavailable_native_clock_frames": unavailable_clocks,
            "sensor_frame_synchronization_verified": False, "camera_calibration_verified": False,
            "camera_framing_quality_verified": False, "paper_effect_proven": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
