#!/usr/bin/env python3
"""Audit every registered N3..15 native control with pinned CPU measurement math."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_audit as native  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)
    parser.add_argument("--control-steps", type=int, choices=(30, 6000), required=True)
    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--scene-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    require = native.original.require
    require(not any(name.split(".")[0] in native.original.FORBIDDEN for name in sys.modules),
            "Native audit requires a fresh CPU process without Torch/SDK")
    reader = record(__file__)
    kernels = native.prepare_kernels(actors=args.actors, controls=args.control_steps)
    if args.check_sources:
        print(json.dumps({"status": "native_raw_audit_source_preparation_passed", "planner": args.planner, "sources": kernels.sources,
            "derivation": kernels.derivation, "Torch_imported": "torch" in sys.modules, "SDK_started": False,
            "actual_native_capture_audited": False, "paper_effect_achieved": False}))
        return 0
    if any(value is None for value in (args.execution_receipt, args.scene_config, args.scene_sha256, args.output)):
        parser.error("Native raw audit needs explicit actual exit receipt, scene/SHA and new output")
    require(not args.output.exists(), "Preserve prior native audits; use a new output")
    base_path = args.output.with_name(args.output.stem + "_base.json")
    require(not base_path.exists(), "Native base audit output already exists")
    selected = native.admit_execution(args.execution_receipt, actors=args.actors, controls=args.control_steps,
        planner=args.planner, scene_config=args.scene_config, scene_sha256=args.scene_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = sys.argv
    try:
        sys.argv = [str(native.original.AUDIT), "--execution-receipt", str(args.execution_receipt), "--output", str(base_path)]
        code = kernels.base.main()
    finally:
        sys.argv = saved
    require(code == 0, "Reused native raw array kernel failed")
    base = json.loads(base_path.read_text())
    body = kernels.body(selected.run / "physical_states.npz", selected.run / "ground_support.npz",
                        args.control_steps, selected.result)
    with np.load(selected.run / "reference_plans.npz", allow_pickle=False) as data:
        plans = {key: data[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    joint = kernels.joint(plans, args.control_steps)
    response = kernels.analyze(selected.result, base, selected.arrays, kernels.helpers)
    link = kernels.initial_link(selected.run / "paired_initial_condition.json", selected.result,
                                selected.run / "ground_support.npz", plans, selected.run)
    require(native.source_bindings() == {key: value for key, value in kernels.sources.items() if key != "frozen_response_helpers"}
            and record(__file__) == reader and all(record(row["path"]) == row for row in selected.identities)
            and all(record(row["path"]) == row for row in kernels.sources["frozen_response_helpers"]),
            "Native inputs or source-bound kernels changed during CPU measurement")
    require(not any(name.split(".")[0] in native.original.FORBIDDEN for name in sys.modules),
            "Native CPU measurement imported Torch/SDK")
    report = {"schema": native.AUDIT_SCHEMA, "status": "passed", "reader": reader,
        "kernel_sources": kernels.sources, "kernel_derivation": kernels.derivation,
        "execution_receipt": record(args.execution_receipt), "result": selected.receipt["result"],
        "native_scene_request": selected.request, "planner": args.planner, "actors": args.actors,
        "actual_controls": args.control_steps, "actor_controls": args.control_steps * args.actors,
        "physics_substeps": args.control_steps * 4, "native_actor_substep_samples": args.control_steps * 4 * args.actors,
        "unordered_pair_controls_exposure": args.control_steps * args.actors * (args.actors - 1) // 2,
        "duration_s": args.control_steps / 30, "scope": "native_smoke30" if args.control_steps == 30 else "native_local_200s",
        "execution_admission": selected.admission, "base_audit": record(base_path),
        "recorded_body_pose": body, "joint_plan_events": joint, "native_response_support": response,
        "initial_capture_link": link, "all_failure_contact_and_reset_frames_retained": True,
        "Torch_imported": False, "SDK_executed_by_reader": False, "actual_CPU_exit_observed_by_reader": False,
        "metric_formulae_changed": False, "paper_rate_denominators_recovered": False,
        "author_eight_scene_protocol_recovered": False, "paper_effect_achieved": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "actors", "actual_controls", "actor_controls", "scope")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
