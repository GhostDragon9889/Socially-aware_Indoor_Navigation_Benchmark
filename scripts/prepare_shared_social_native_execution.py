#!/usr/bin/env python3
"""Prepare registered commands or publish an externally observed native exit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import native_execution_bridge as bridge  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("source-preflight", "SDK-exit", "raw-CPU-exit"), required=True)
    parser.add_argument("--planner", choices=("diffusion", "sfm"))
    parser.add_argument("--control-steps", type=int, choices=(30, 6000))
    parser.add_argument("--tool-observation", type=Path)
    parser.add_argument("--owner-completion", type=Path)
    args = parser.parse_args(argv)
    bridge.require(not any(name.split(".")[0] in bridge.raw.original.FORBIDDEN for name in sys.modules),
                   "Native exit bridge requires a fresh CPU process without Torch/SDK")
    if args.mode == "source-preflight":
        plan = bridge.prepare_case(args.case_id)
        plan["Torch_imported"] = "torch" in sys.modules
        plan["SDK_started"] = False
        print(json.dumps(plan, allow_nan=False))
        return 0
    if any(value is None for value in (args.planner, args.control_steps, args.tool_observation, args.owner_completion)):
        parser.error("Exit publication requires explicit stage, actual tools observation and external S3 owner completion")
    publish = bridge.publish_sdk_exit if args.mode == "SDK-exit" else bridge.publish_raw_CPU_exit
    receipt, passed = publish(args.case_id, planner=args.planner, controls=args.control_steps,
        observation_path=args.tool_observation, owner_completion=args.owner_completion)
    print(json.dumps({"receipt": receipt, "actual_declared_stage_admitted": passed,
                      "SDK_started_by_bridge": False, "paper_effect_achieved": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
