#!/usr/bin/env python3
"""Build the explicitly labeled AMASS locomotion reconstruction manifest."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", default=str(ROOT.parent / "Assets/motion"))
    parser.add_argument("--output", default=str(ROOT / "data/diffusion_amp/amass_manifest.json"))
    args = parser.parse_args()
    from CrowdSim.humanoid.amass_corpus import prepare_manifest
    sources = {split: Path(args.motion_root) / f"amass_smpl_{split}.pt" for split in ("train", "validation", "test")}
    result = prepare_manifest(sources, args.output)
    print(json.dumps({"output": args.output, "author_clip_list": False,
                      "splits": {k: {key: v[key] for key in ("clip_count", "diffusion_clip_count", "sha256")} for k, v in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
