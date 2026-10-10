#!/usr/bin/env python3
"""CPU style diagnostics for shared Social AMP full22 captures.

Reuse the existing 90D metric and frozen held-out selection. The only reader
bridge admits the new explicit AMASS provenance; no reference refit or Test34.
The result is a local diagnostic, not a numerical NavIsaacLab Table III score.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402

TOOL = ROOT / "scripts/evaluate_pedestrian_style.py"
TOOL_SHA = "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f"
REFERENCE = ROOT / "docs/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json"
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
POSE_TOOL = ROOT / "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py"
POSE_TOOL_SHA = "208ff775d765b7d046e7e7e8c331dbb2b8d3bfdc4c1567c1df499a584af30bfa"
BASIS = POSE_TOOL.parent / "actual_evaluation_v1/reference_basis.npz"
BASIS_SHA = "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb"
BASIS_EVALUATION = BASIS.parent / "evaluation.json"
BASIS_EVALUATION_SHA = "9478afe6b16540f7624867e1d736f30f78399c84177c12d6c9b616bc496c3780"


def training_manifest(runner):
    """Select a declared schema; never substitute the evaluation reference."""
    social = runner.get("schema") == "local.neutral_smpl.social_full22_tracking.v2"
    if social and (runner.get("cleanup_succeeded") is not True or runner.get("cleanup_errors") != []
                   or runner.get("training_execution_admission", {}).get("terminal_exit_code") != 0):
        raise ValueError("Shared Social capture lacks completed training admission or evaluation cleanup")
    key = "AMASS" if social else "dataset"
    dataset = runner.get("controller_provenance", {}).get(key)
    if not isinstance(dataset, dict) or dataset.get("split") != "train":
        raise ValueError("Actual controller training provenance is required")
    path = Path(dataset["manifest"])
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    document = json.loads(path.read_text())
    train = document["splits"]["train"]
    if (record(path)["sha256"] != dataset.get("manifest_sha256")
            or train["sha256"] != dataset.get("source_sha256") or len(train["clips"]) != dataset.get("clips")):
        raise ValueError("Actual training AMASS identity changed")
    if social and len(train["clips"]) != 2848:
        raise ValueError("Shared Social evaluation requires fixed2848 real AMASS")
    return path, document


def pose45(tool, runs):
    """Assign all legal poses to saved centers; never fit generated/reference data."""
    import numpy as np
    from CrowdSim.humanoid.style_evaluation import StyleSpec, sequence_style_samples

    for path, sha in ((POSE_TOOL, POSE_TOOL_SHA), (BASIS, BASIS_SHA), (BASIS_EVALUATION, BASIS_EVALUATION_SHA)):
        if record(path)["sha256"] != sha:
            raise ValueError("Frozen pose45 source or basis changed")
    spec = importlib.util.spec_from_file_location("_shared_pose45_tool", POSE_TOOL)
    pose_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_tool)
    artifact = json.loads(REFERENCE.read_text())
    body, rotation, hashes, _ = tool.load_reference(Path(artifact["manifest"]["path"]), artifact["selection"])
    if hashes != artifact["actual_selected_array_sha256"]:
        raise ValueError("Frozen actual reference arrays changed")
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    reference = sequence_style_samples(body, rotation, episode_ids=np.ones(body.shape[:2], dtype=np.int64),
                                       reset_mask=resets, spec=StyleSpec())["local_pose"]
    previous = json.loads(BASIS_EVALUATION.read_text())
    fit = previous["method"]["fit"]
    if len(reference) != fit["available_reference_poses"]:
        raise ValueError("Frozen reference pose population differs")
    with np.load(BASIS, allow_pickle=False) as saved:
        if set(saved.files) != {"centers_50", "centers_100"}:
            raise ValueError("Unknown frozen pose45 basis")
        centers = {k: saved[f"centers_{k}"].copy() for k in (50, 100)}
    for k, value in centers.items():
        if (value.dtype != np.float64 or value.shape != (k, 45) or not np.isfinite(value).all()
                or pose_tool.array_sha(value) != fit["centers"][str(k)]["float64_sha256"]):
            raise ValueError("Saved pose45 centers differ from the frozen reference fit")
    report = {"schema": "shared_social.frozen_pose45_validation.v1", "status": "passed",
        "reference": record(REFERENCE), "basis": record(BASIS), "original_basis_evaluation": record(BASIS_EVALUATION),
        "metric_implementation": record(POSE_TOOL), "reference_refit_performed": False,
        "generated_samples_used_for_fit": False, "numerical_comparison_to_paper_valid": False,
        "MMD_estimator": "biased Gaussian MMD squared; sigma1; deterministic inclusive cap2048",
        "runs": {}}
    for item in runs:
        name, _, directory = item.partition("=")
        arrays, runner, capture = tool.read_tracking_capture(Path(directory))
        generated = sequence_style_samples(arrays["body_pos"], arrays["root_rot"], body_names=capture["body_names"],
            episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"], spec=StyleSpec())
        report["runs"][name] = {"source": record(Path(directory) / "result.json"),
            "capture_validation": capture, "fall_frames_kept": capture["fall_metric_frames_kept"],
            "tracking_aggregate": runner["aggregate"], "failure_filter_applied": False,
            "poses": pose_tool.compare_pose_only(generated["local_pose"], reference, centers)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=full22 directory; repeat for comparisons")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run CPU metrics with CUDA_VISIBLE_DEVICES=''")
    if record(TOOL)["sha256"] != TOOL_SHA or record(REFERENCE)["sha256"] != REFERENCE_SHA:
        raise ValueError("Frozen style tool or held-out reference changed")
    input_records = {}
    for item in options.run:
        name, separator, directory = item.partition("=")
        if not separator or not name or name in input_records:
            raise ValueError("Each run requires a unique NAME=directory")
        directory = Path(directory).resolve()
        result = json.loads((directory / "result.json").read_text())
        if result.get("status") != "passed" or len(result.get("sequences", [])) != 22:
            raise ValueError("Style comparison requires complete full22 captures, including failed motions")
        training_manifest(result)
        input_records[name] = record(directory / "result.json")
    spec = importlib.util.spec_from_file_location("_shared_style90_tool", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    # This changes only the reader in this entry-owned module. Pose extraction,
    # speed bins, sigma, maximum samples and every metric function stay intact.
    tool.tracking_training_manifest = training_manifest
    command = ["--reference-artifact", str(REFERENCE), "--cpu-threads", "4",
               "--minimum-bin-samples", "30", "--maximum-motion-samples", "256",
               "--output", str(options.output)]
    for item in options.run:
        command.extend(["--run", item])
    tool.main(command)
    output = options.output / "evaluation.json"
    document = json.loads(output.read_text())
    pose_report = pose45(tool, options.run)
    pose_path = options.output / "pose45.json"
    with pose_path.open("x") as stream:
        json.dump(pose_report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    document["frozen_pose45_result"] = record(pose_path)
    document["shared_social_reader_bridge"] = {"implementation": record(Path(__file__)),
        "original_style_tool": record(TOOL), "reference": record(REFERENCE), "inputs": input_records,
        "reader_change": "explicit AMASS provenance for social_full22.v2; legacy dataset schema retained",
        "metric_math_changed": False, "failures_filtered": False, "Test34_read": False,
        "numerical_comparison_to_paper_valid": False}
    for name, identity in input_records.items():
        if record(identity["path"]) != identity:
            raise RuntimeError("Tracking result changed during style evaluation: " + name)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU style evaluation unexpectedly initialized CUDA")
    document["CUDA_initialized"] = False
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
