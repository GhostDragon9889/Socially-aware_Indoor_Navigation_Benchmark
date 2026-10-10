#!/usr/bin/env python3
"""Descriptive counterflow observations in recorded actual pelvis motion.

A proximity interval is bounded by root distance<3m. Opposing measured
velocities require speed>=.2m/s and unit-velocity dot product<-.5. At least
.2s of opposing frames are required. Crossing of the relative longitudinal
projection is reported as a passing proxy, not an all-scenario safety score.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect(samples, *, dt=1/30):
    xy, velocity = samples[..., :2], samples[..., 2:4]
    speed = np.linalg.norm(velocity, axis=-1)
    direction = velocity / np.maximum(speed[..., None], 1e-12)
    observations = []
    for a in range(samples.shape[1]):
        for b in range(a+1, samples.shape[1]):
            relative = xy[:, b]-xy[:, a]
            distance = np.linalg.norm(relative, axis=-1)
            opposed = ((direction[:, a]*direction[:, b]).sum(-1) < -.5) & (speed[:, a] >= .2) & (speed[:, b] >= .2)
            close = distance < 3.
            boundaries = np.flatnonzero(np.diff(np.r_[False, close, False]))
            for start, end in zip(boundaries[::2], boundaries[1::2]):
                frames = np.flatnonzero(opposed[start:end]) + start
                if len(frames) < 6:
                    continue
                first, last = int(frames[0]), int(frames[-1])
                forward = direction[frames[:6], a].mean(0)
                forward /= max(np.linalg.norm(forward), 1e-12)
                before = float(relative[first] @ forward)
                after = float(relative[last] @ forward)
                observations.append({
                    "actors": [a, b], "proximity_start_s": float((start+1)*dt),
                    "proximity_end_s": float(end*dt),
                    "proximity_reopened_before_capture_end": bool(end < len(samples)),
                    "opposing_motion_control_frames": len(frames),
                    "first_last_opposing_interval_s": float((last-first)*dt),
                    "minimum_actual_root_distance_m": float(distance[start:end].min()),
                    "relative_longitudinal_before_after_m": [before, after],
                    "observed_longitudinal_passing_proxy": bool(before > .1 and after < -.1),
                })
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads(args.execution.read_text())
    result_path = args.run / "result.json"
    physical_path = args.run / "physical.npz"
    result = json.loads(result_path.read_text())
    if not (receipt.get("actual_returncode") == 0
            and receipt.get("actual_returncode_source") == "subprocess.Popen.wait"
            and receipt.get("sources_unchanged") is True and receipt.get("inputs_unchanged") is True
            and receipt.get("result_sha256") == sha(result_path)
            and receipt.get("capture_sha256", {}).get("physical.npz") == sha(physical_path)
            and result.get("cleanup_succeeded") is True and result.get("status") == "passed"
            and result["actual_controls"] == result["requested_controls"]):
        raise ValueError("Full completed source-bound physical run required")
    with np.load(physical_path, allow_pickle=False) as data:
        samples = data["samples"].copy()
    if len(samples) != result["actual_controls"] or not np.isfinite(samples).all() or result["physics"]["control_hz"] != 30:
        raise ValueError("Complete finite30Hz physical samples required")
    events = inspect(samples)
    report = {"schema": "crowdsim.trace_pace_counterflow_observations.v1",
              "diagnostic_only": True, "paper_metric": False,
              "uncoordinated_counterflow_reliability_established": False,
              "input_result_sha256": sha(result_path), "physical_sha256": sha(physical_path),
              "evaluation_source_sha256": sha(Path(__file__)),
              "all_actual_frames_loaded": True, "observations": events,
              "opposed_proximity_intervals": len(events),
              "longitudinal_passing_proxies": sum(e["observed_longitudinal_passing_proxy"] for e in events),
              "definition": __doc__,
              "limitations": ["Pelvis motion and longitudinal projections are proxies; actual contacts must be evaluated separately",
                              "The directed global route convention is part of this run; no inference of free-form social negotiation",
                              "Zero observed encounters can reflect separated routes and does not establish passing reliability"]}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "opposed_proximity_intervals": len(events),
                      "longitudinal_passing_proxies": report["longitudinal_passing_proxies"]}))


if __name__ == "__main__":
    main()
