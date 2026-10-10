#!/usr/bin/env python3
"""Validate exact initial pairing, without claiming actual exit or quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--diffusion-initial", type=Path, required=True)
    parser.add_argument("--sfm-initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from CrowdSim.humanoid.analysis.paired_initial import compare_initial_conditions
    report = compare_initial_conditions(args.diffusion_initial, args.sfm_initial)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
