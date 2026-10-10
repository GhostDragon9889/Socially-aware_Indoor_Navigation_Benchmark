#!/usr/bin/env python3
"""Measure paired native N3..15 200s motion with the frozen held-out CPU metric."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_style as style  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--scene-sha256")
    parser.add_argument("--diffusion-audit-execution", type=Path)
    parser.add_argument("--sfm-audit-execution", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    metric, require = style.metric, style.metric.require
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Native style requires CUDA_VISIBLE_DEVICES='' for CPU analysis")
    require(not any(name.split(".")[0] in style.raw.original.FORBIDDEN for name in sys.modules),
            "Start native style in a fresh process without Torch/SDK")
    sources = {"entry": metric.record(__file__), "reader_bindings": style.source_bindings()}
    if args.check_sources:
        print(json.dumps({"status": "native_style_CPU_source_preparation_passed", "actors": args.actors,
            "required_controls": 6000, "sources": sources, "frozen_metric_math_changed": False,
            "reference_or_basis_refit": False, "Torch_imported": False, "SDK_started": False,
            "actual_native_capture_or_quality_measured": False, "paper_effect_achieved": False}))
        return 0
    if any(value is None for value in (args.scene_config, args.scene_sha256, args.diffusion_audit_execution,
                                       args.sfm_audit_execution, args.output)):
        parser.error("Paired native style needs explicit scene/SHA, both actual CPU audit receipts and new output")
    require(not args.output.exists(), "Use a fresh native style output; preserve previous measurements")
    completed = {name: style.read_completed_run(path, actors=args.actors, planner=name,
        scene_config=args.scene_config, scene_sha256=args.scene_sha256)
        for name, path in (("diffusion", args.diffusion_audit_execution), ("sfm", args.sfm_audit_execution))}
    initial = style.raw.initial.compare_initial_conditions(*(Path(completed[name].result["paired_initial_condition"]
        ["document"]["path"]) for name in ("diffusion", "sfm")))
    reference, pose, centers, reference_admission, reference_inputs = metric.load_frozen_reference()
    identities = [sources["entry"], *reference_inputs]
    for item in completed.values():
        identities.extend(item.identities)
    report = {"schema": style.STYLE_SCHEMA, "status": "passed_CPU_measurement", "actors": args.actors,
        "control_steps": 6000, "duration_s": 200, "sources": sources, "initial_pairing": initial,
        "reference": reference_admission,
        "runs": {name: {"admission": item.admission, "inclusive_summary": item.samples["summary"],
            "pose45": pose.compare_pose_only(item.samples["local_pose"], reference["local_pose"], centers),
            "style90_all_population": metric.compare_speed_conditioned(item.samples, reference),
            "physical_route_audit": item.base} for name, item in completed.items()},
        "style90_matched_pair": metric.matched_motion_bins({name: item.samples for name, item in completed.items()}, reference),
        "basis_refit": False, "metric_math_changed": False, "failed_episodes_filtered": False,
        "motion_capture_scope": "Full post-control captures; original metric uses legal same-episode interval endpoints",
        "numerical_comparison_to_paper_valid": False, "author_eight_scene_protocol_recovered": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    require(all(metric.record(row["path"]) == row for row in identities)
            and style.source_bindings() == sources["reader_bindings"], "Native style inputs or sources changed")
    require(not any(name.split(".")[0] in (style.raw.original.FORBIDDEN - {"torch"}) for name in sys.modules),
            "Native CPU style unexpectedly imported the SDK")
    require("torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "Native CPU style unexpectedly initialized CUDA")
    report["CUDA_initialized"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "actors": args.actors,
        "style90_matched_pair": report["style90_matched_pair"]["equal_common_bin_mean_MMD2"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
