"""Generate new map/neighbor-conditioned expert trajectories with real RVO2.

Run through scripts/run_external_python.py after install_rvo2_external.py,
passing --rvo2-source .runtime/dependencies/Python-RVO2. The historical
independent generator environment is also supported. Neither SDK files nor
the frozen AMASS checkpoint are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/diffusion_amp/orca_social_pilot_v1")
    parser.add_argument("--trace-source", type=Path, default=ROOT / "third_party/trace")
    parser.add_argument("--rvo2-source", type=Path, default=ROOT / "data/tools/orca_generator/Python-RVO2")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--train-scenes", type=int, default=16)
    parser.add_argument("--validation-scenes", type=int, default=4)
    parser.add_argument("--test-scenes", type=int, default=4)
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--max-obstacles", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.train_scenes, args.validation_scenes, args.test_scenes, args.episodes_per_scene, args.max_obstacles) < 1:
        parser.error("Scene/episode/obstacle counts must be positive")
    sys.path.insert(0, str(ROOT))
    from CrowdSim.humanoid.orca_corpus import generate_corpus

    def progress(split, scene_id, episode):
        print(json.dumps({"split": split, "scene": scene_id, "episode": episode["episode_id"],
                          "agents": episode["agents"], "windows": episode["conditional_window_count"],
                          "qa": episode["qa"]}), flush=True)

    started = time.monotonic()
    manifest = generate_corpus(args.output, args.trace_source, args.rvo2_source, seed=args.seed,
                              split_scenes={"train": args.train_scenes, "validation": args.validation_scenes,
                                            "test": args.test_scenes},
                              episodes_per_scene=args.episodes_per_scene, seconds=args.seconds,
                              max_obstacles=args.max_obstacles, progress=progress)
    print(json.dumps({"manifest": str(manifest), "elapsed_seconds": time.monotonic() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
