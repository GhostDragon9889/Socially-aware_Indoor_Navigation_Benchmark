#!/usr/bin/env python3
"""Register episode2000 native tasks and bind actual tools API exit observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_native_execution as bridge  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("register", "source-preflight", "SDK-exit", "raw-CPU-exit"), required=True)
    parser.add_argument("--registration", type=Path, default=bridge.REGISTRATION)
    parser.add_argument("--case-id")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--planner", choices=("diffusion", "sfm"))
    parser.add_argument("--control-steps", type=int, choices=(30, 6000))
    parser.add_argument("--tool-observation", type=Path)
    parser.add_argument("--owner-completion", type=Path)
    args = parser.parse_args(argv)
    bridge.require(not any(name.split(".")[0] in bridge.original.raw.original.FORBIDDEN for name in sys.modules),
                   "Episode native bridge requires a fresh CPU process without Torch/SDK")
    if args.mode == "register":
        identity = bridge.write_registration(args.registration)
        print(json.dumps({"status": "episode_native_registration_created", "registration": identity,
                          "Torch_imported": False, "SDK_started": False}))
        return 0
    if args.case_id is None:
        parser.error("--case-id is required")
    if args.mode == "source-preflight":
        plan = bridge.prepare_case(args.case_id, registration_path=args.registration)
        identity = bridge.original._write_new(args.plan_output, plan) if args.plan_output else None
        print(json.dumps({"status": "episode_native_source_preflight_passed", "case_id": args.case_id,
                          "registered_stage_count": len(plan["stages"]), "plan_output": identity,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False,
                          "actual_episode_completion_admitted": False}, allow_nan=False))
        return 0
    if any(value is None for value in (args.planner, args.control_steps, args.tool_observation, args.owner_completion)):
        parser.error("Exit publication requires an explicit stage, actual tools observation and episode owner completion")
    publish = bridge.publish_sdk_exit if args.mode == "SDK-exit" else bridge.publish_raw_CPU_exit
    receipt, passed = publish(args.case_id, planner=args.planner, controls=args.control_steps,
        observation_path=args.tool_observation, owner_completion=args.owner_completion,
        registration_path=args.registration)
    print(json.dumps({"receipt": receipt, "actual_declared_stage_admitted": passed,
                      "SDK_started_by_bridge": False, "paper_effect_achieved": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
