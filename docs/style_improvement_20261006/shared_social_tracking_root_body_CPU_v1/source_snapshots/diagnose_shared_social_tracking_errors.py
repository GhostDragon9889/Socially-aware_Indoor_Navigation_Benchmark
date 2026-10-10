#!/usr/bin/env python3
"""Read completed validation22 captures and separate root/body diagnostic errors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import tracking_error_decomposition as kernel  # noqa: E402
from CrowdSim.humanoid.motion_tracking import SMPL_BODY_NAMES  # noqa: E402
from scripts import verify_shared_social_evidence_admission as gate  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

FROZEN = ROOT / "CrowdSim/humanoid/motion_tracking.py"
FROZEN_SHA = "da482dcd565d42ef9f6165606428ad510f077bdd68c3fab09c73a840149eac56"
SCHEMA = "shared_social.complete_tracking_root_body_CPU_diagnostic.v1"
SCALAR_FIELDS = ("mean_body_position_error_m", "mean_body_rotation_error_rad", "root_XY_error_m",
    "root_XYZ_error_m", "root_height_error_m", "root_rotation_error_rad", "root_relative_world_axes_mean_body_error_m",
    "own_root_frame_mean_body_error_m", "world_body_MSE_m2", "root_translation_MSE_m2",
    "root_relative_world_axes_MSE_m2", "signed_cross_MSE_m2", "MSE_closure_absolute_error_m2")


def summarize(values):
    return {key: {"frames": len(values[key]), "mean": float(values[key].mean()), "maximum": float(values[key].max())}
            for key in SCALAR_FIELDS}


def analyze(post_receipt):
    bundle, measurements = gate.admit_post(post_receipt)
    admitted_sources = gate.bundle_sources(bundle, measurements)
    report = bundle["tracking"]
    kernel.require(report["physics"]["body_names"] == list(SMPL_BODY_NAMES)
        and report["protocol"]["threshold_m"] == .5, "Require unchanged canonical world-space protocol")
    inputs = [identity(post_receipt), *measurements.values(), *admitted_sources]
    rows, sequences = [], []
    for sequence in report["sequences"]:
        capture_path = Path(sequence["capture"])
        captured = identity(capture_path)
        kernel.require(captured["sha256"] == sequence["capture_sha256"], "Completed tracking capture changed")
        inputs.append(captured)
        with np.load(capture_path, allow_pickle=False) as arrays:
            values = kernel.verify_capture(arrays, sequence)
            metric = arrays["metric_frame"]
            observed = {key: value[metric] for key, value in values.items() if key in SCALAR_FIELDS}
            observed.update(motion_id=np.full(int(metric.sum()), sequence["motion_id"], np.int64),
                capture_frame=np.flatnonzero(metric), fall_mask=arrays["fall_mask"][metric],
                original_world_failure=values["mean_body_position_error_m"][metric] > .5)
        first = sequence["first_tracking_failure_capture_frame"]
        sequences.append({"motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
            "capture": captured, "original_tracking_success": sequence["tracking_success"],
            "first_world_failure_capture_frame": first, "observed": summarize(observed),
            "at_first_world_failure": None if first is None else {key: float(values[key][first]) for key in SCALAR_FIELDS}})
        rows.append(observed)
    complete = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    kernel.require(len(rows) == 22 and len(complete["motion_id"]) == 11982, "Require all validation22 and every11982 metric frame")
    for name, field in (("frame_weighted_mean_body_position_error_m", "mean_body_position_error_m"),
                        ("frame_weighted_mean_body_rotation_error_rad", "mean_body_rotation_error_rad")):
        kernel.require(abs(float(complete[field].mean()) - report["aggregate"][name]) < 1e-12,
                       "Full original unaligned aggregate changed")
    return report, inputs, complete, sequences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-post-receipt", type=Path, required=True)
    parser.add_argument("--candidate-post-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output_directory.exists(),
                   "Fresh CPU-only output directory required")
    sources = [identity(path) for path in (Path(__file__), Path(kernel.__file__), FROZEN,
        Path(gate.__file__), Path(gate.original.__file__), ROOT / "scripts/diagnose_shared_social_training.py")]
    kernel.require(sources[2]["sha256"] == FROZEN_SHA, "Frozen tracking metric changed")
    results, inputs, populations = {}, [], {}
    tracking = []
    for name, receipt in (("baseline", args.baseline_post_receipt), ("candidate", args.candidate_post_receipt)):
        report, bound_inputs, complete, sequences = analyze(receipt)
        tracking.append(report)
        inputs.extend(bound_inputs)
        populations[name] = complete
        results[name] = {"post_receipt": identity(receipt), "original_world_tracking_aggregate": report["aggregate"],
            "full_population": summarize(complete), "sequences": sequences,
            "all_metric_frames": len(complete["motion_id"]), "fall_metric_frames_retained": int(complete["fall_mask"].sum()),
            "world_failure_frames": int(complete["original_world_failure"].sum()),
            "root_XY_over_0_5m_frames_diagnostic_only": int((complete["root_XY_error_m"] > .5).sum())}
    gate.tracking_protocol(*tracking)
    inputs = gate.unique_artifacts(inputs)
    kernel.require(np.array_equal(populations["baseline"]["motion_id"], populations["candidate"]["motion_id"])
        and np.array_equal(populations["baseline"]["capture_frame"], populations["candidate"]["capture_frame"]),
        "Comparison must retain identical reference IDs/frame phases")
    forbidden = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
    kernel.require(not any(name.split(".")[0] in forbidden for name in sys.modules), "CPU diagnostic imported model/SDK")
    kernel.require(all(identity(row["path"]) == row for row in sources + inputs), "Source or actual input changed during analysis")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    for name, values in populations.items():
        path = args.output_directory / (name + "_all_metric_frames.npz")
        with path.open("xb") as stream:
            np.savez(stream, **values)
        results[name]["all_frame_archive"] = identity(path)
    output = {"schema": SCHEMA, "status": "completed_CPU_observation", "sources": sources, "inputs": inputs,
        "runs": results, "definitions": {
            "primary": "Original world-space 24-body errors and strict >.5m latch; exactly reproduced, no alignment applied",
            "root_relative_world_axes": "Subtract each measured Pelvis translation; retain world orientation",
            "own_root_frame": "Additionally inverse-rotate by each measured root quaternion; no fitted rotation or phase",
            "signed_MSE_identity": "world=root_translation+relative+signed_cross; cross can be negative, not causal percentages"},
        "Torch_imported": False, "SDK_started": False, "failed_frames_filtered": False,
        "diagnostic_alignment_is_paper_score": False, "causal_training_failure_proven": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    with (args.output_directory / "result.json").open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": output["status"], "result": identity(args.output_directory / "result.json"),
        "root_XY_error_m": {name: row["full_population"]["root_XY_error_m"]["mean"] for name, row in results.items()},
        "SDK_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
