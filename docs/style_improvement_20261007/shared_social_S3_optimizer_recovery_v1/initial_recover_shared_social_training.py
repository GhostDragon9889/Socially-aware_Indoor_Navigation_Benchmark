#!/usr/bin/env python3
"""Continue S3's optimizer budget from step12000, with a declared World restart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime import social_training_recovery as recovery  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--stage", choices=("smoke", "training"), required=True)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    module, binding, profile = recovery.prepare_entry(args.registration, args.stage)
    rows = module._sources()
    if args.check_sources:
        print(json.dumps({"status": "optimizer_recovery_source_preflight_passed", "binding": binding,
                          "source_count": len(rows), "profile": profile,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    saved = sys.argv
    try:
        sys.argv = [__file__, "--config", binding["configuration"]["path"],
                    "--output", binding["output"], "--device", "cuda:0"]
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
