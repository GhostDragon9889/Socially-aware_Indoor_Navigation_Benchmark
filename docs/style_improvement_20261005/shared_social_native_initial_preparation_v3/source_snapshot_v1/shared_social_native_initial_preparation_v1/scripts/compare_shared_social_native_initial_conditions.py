#!/usr/bin/env python3
"""Require exact native pre-planner snapshots; this is not a quality evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.native_paired_initial import compare_initial_conditions  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Native initial matching report already exists")
    report = compare_initial_conditions(args.left, args.right, require_live=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "actors": report["actors"],
                      "channel_count": report["channel_count"], "actual_process_exit_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
