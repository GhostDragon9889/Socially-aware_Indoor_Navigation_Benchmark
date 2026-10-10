#!/usr/bin/env python3
"""Version a curated AMASS bank with measured IsaacLab/PACER child-joint dvs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from CrowdSim.humanoid.training.amass_curation import convert_curated_joint_velocities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    result = convert_curated_joint_velocities(args.manifest, args.output)
    print(json.dumps({"manifest": str(args.output.resolve()), "train": {key: value for key, value in result["splits"]["train"].items() if key != "clips"},
                      "conversion": result["curation"]["joint_velocity_conversion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
