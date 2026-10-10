#!/usr/bin/env python3
"""Build an honest expanded local-AMASS manifest without modifying the banks."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from CrowdSim.humanoid.training.amass_curation import CurationConfig, curate_manifest
from protomotions.robot_configs.smpl import SmplRobotConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", default="data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--config", default="configs/diffusion_amp/amass_curation.yaml")
    parser.add_argument("--output", default="data/diffusion_amp/amass_style_2848/manifest.json")
    parser.add_argument("--labels", help="Optional verified source-keyed description metadata JSON")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    parameters = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(parameters, dict) or set(parameters) - {item.name for item in fields(CurationConfig)}:
        raise ValueError("Curation configuration has unknown fields or is not a mapping")
    if "excluded_datasets" in parameters:
        parameters["excluded_datasets"] = tuple(parameters["excluded_datasets"])
    config = CurationConfig(**parameters)
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    if labels is not None and (not isinstance(labels, dict) or any(not isinstance(value, dict) or not value.get("description") or not value.get("source_url") for value in labels.values())):
        raise ValueError("Labels must map actual source paths to description/source_url records")
    info = SmplRobotConfig().kinematic_info
    document = curate_manifest(args.base_manifest, args.output, config=config,
        body_names=info.body_names, kinematic_info=info, labels_by_source=labels,
        progress=lambda done, total, accepted: print(json.dumps({"scanned_id": done, "total": total, "eligible": accepted}), flush=True))
    print(json.dumps({"manifest": str(Path(args.output).resolve()), "curation": document["curation"],
                      "counts": {key: entry["clip_count"] for key, entry in document["splits"].items()}}, indent=2))
    return 0 if document["curation"]["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
