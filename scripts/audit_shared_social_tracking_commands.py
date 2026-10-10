#!/usr/bin/env python3
"""Read completed tracking captures and audit PRE command/heading consistency."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import tracking_commands as kernel  # noqa: E402
from CrowdSim.humanoid.analysis import tracking_error_decomposition as original_geometry  # noqa: E402
from scripts import verify_shared_social_evidence_admission as gate  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

TASK_SHA = "6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1"
METRIC_SHA = "da482dcd565d42ef9f6165606428ad510f077bdd68c3fab09c73a840149eac56"


def analyze(receipt_path):
    bundle, measurements = gate.admit_post(receipt_path)
    sources = gate.bundle_sources(bundle, measurements)
    report = bundle["tracking"]
    physics = report["physics"]
    kernel.require(physics["control_hz"] == 30 and physics["physics_hz"] == 120
        and physics["isolated_environments"] is True, "Require the original isolated120/30Hz tracking capture")
    producer = report["social_observation_producer"]["sources"]
    kernel.require(producer["task"]["sha256"] == TASK_SHA
        and Path(producer["task"]["path"]) == ROOT / "docs/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py",
        "Require the original positive-future20 task prefix source")
    inputs = [identity(receipt_path), *measurements.values(), *sources, *producer.values()]
    sequences = []
    for sequence in report["sequences"]:
        source = identity(sequence["capture"])
        kernel.require(source["sha256"] == sequence["capture_sha256"], "Actual command capture changed")
        with np.load(source["path"], allow_pickle=False) as arrays:
            observation = kernel.audit_capture(arrays, sequence)
        inputs.append(source)
        sequences.append({"motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
                          "capture": source, **observation})
    summed = {key: sum(row[key] for row in sequences) for key in ("metric_frames", "metric_command_samples",
        "held_endpoint_samples", "all_ten_held_frames", "singular_heading_fallback_frames", "fall_frames_retained",
        "offline_reference_root_reward_sum")}
    kernel.require(len(sequences) == 22 and summed["metric_frames"] == 11982, "Retain all validation22 metric frames")
    summed.update(held_endpoint_sample_fraction=summed["held_endpoint_samples"] / summed["metric_command_samples"],
        offline_reference_root_reward_mean=summed["offline_reference_root_reward_sum"] / summed["metric_frames"],
        maximum_prefix_float64_difference_m=max(row["maximum_prefix_float64_difference_m"] for row in sequences),
        maximum_defined_heading_wrapped_difference_rad=max(row["maximum_defined_heading_wrapped_difference_rad"] for row in sequences))
    return report, inputs, {"post_receipt": identity(receipt_path), "aggregate": summed, "sequences": sequences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-post-receipt", required=True, type=Path)
    parser.add_argument("--candidate-post-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(), "Fresh CPU-only output required")
    sources = [identity(path) for path in (Path(__file__), Path(kernel.__file__), Path(original_geometry.__file__),
        ROOT / "CrowdSim/humanoid/motion_tracking.py", Path(gate.__file__), Path(gate.original.__file__),
        ROOT / "scripts/diagnose_shared_social_training.py")]
    kernel.require(sources[3]["sha256"] == METRIC_SHA, "Original unaligned tracking metric changed")
    reports, inputs, results = [], [], {}
    for name, receipt in (("baseline", args.baseline_post_receipt), ("candidate", args.candidate_post_receipt)):
        report, bound, result = analyze(receipt)
        reports.append(report)
        inputs.extend(bound)
        results[name] = result
    gate.tracking_protocol(*reports)
    inputs = gate.unique_artifacts(inputs)
    kernel.require(all(identity(row["path"]) == row for row in sources + inputs), "Actual input/source changed during command audit")
    forbidden = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
    kernel.require(not any(name.split(".")[0] in forbidden for name in sys.modules), "CPU reader imported model or SDK")
    output = {"schema": "shared_social.complete_tracking_PRE_commands_CPU_audit.v1",
        "status": "passed_CPU_command_consistency", "sources": sources, "inputs": inputs, "runs": results,
        "scope": "Actual isolated PRE clock, reference phase, positive future samples, measured heading and raw20 prefix; not raster or NN sensitivity",
        "prefix_numerical_bound": "half float32 spacing +32 float64 eps times sum(abs(world-command minus PRE root XY))",
        "offline_reference_root_reward": "exp(-2*same-phase POST rootXY squared error), paper Eq.(4) applied offline; not an actual training or total-reward trace",
        "Torch_imported": False, "SDK_started": False, "failed_frames_filtered": False,
        "causal_training_failure_proven": False, "paper_effect_achieved": False, "default_model_promoted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": identity(args.output), "aggregate": {name: row["aggregate"] for name, row in results.items()},
                      "SDK_started": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
