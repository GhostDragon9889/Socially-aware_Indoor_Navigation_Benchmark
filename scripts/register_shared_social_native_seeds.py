#!/usr/bin/env python3
"""Register all fixed local native resource cases with paired seed7/19/42."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_seed_protocol as protocol  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True, help="Future SDK capture root; not created")
    parser.add_argument("--output", type=Path, required=True, help="New pre-registration JSON")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Preserve earlier native registrations; use a new output")
    forbidden = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
    if any(name.split(".")[0] in forbidden for name in sys.modules):
        raise ValueError("Start matrix registration in a fresh CPU-only process")
    source = protocol.entry.record(__file__)
    report = protocol.build_registration(args.output_root)
    if (protocol.entry.record(__file__) != source
            or any(name.split(".")[0] in forbidden for name in sys.modules)):
        raise ValueError("Native seed registration source changed or imported Torch/SDK")
    report["registration_entry"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "registered_cohorts", "resource_admitted_cohorts",
        "resource_rejected_cohorts", "planned_full200s_arm_runs_after_all_gates", "planned_paired_actor_controls")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
