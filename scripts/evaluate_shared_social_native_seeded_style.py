#!/usr/bin/env python3
"""Evaluate complete same-seed native DP/SFM pairs with frozen style metrics."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_seeded_style as style  # noqa: E402
from scripts import evaluate_shared_social_native_seeded as entry  # noqa: E402

PARENT = ROOT / "scripts/evaluate_shared_social_native_style.py"
PARENT_SHA = "3a9378bbaa17fcfbfdbed2c112759c004732cfed10889c33dc4dd2d0c6829920"


def prepared_CLI():
    if style.metric.record(PARENT)["sha256"] != PARENT_SHA:
        raise ValueError("Frozen native style CLI changed")
    source, proof = style.raw.frozen._derive(entry.seed_main_source(PARENT.read_text()), [
        ('    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)\n',
         '    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)\n'
         '    parser.add_argument("--seed", type=int, required=True)\n', 1),
        ('    args = parser.parse_args(argv)\n',
         '    args = parser.parse_args(argv)\n    style.raw.entry.validate_seed(args.seed)\n', 1),
        ('style.read_completed_run(path, actors=args.actors, planner=name,',
         'style.read_completed_run(path, actors=args.actors, planner=name, seed=args.seed,', 1),
        ('"required_controls": 6000, "sources": sources,',
         '"required_controls": 6000, "seed": args.seed, "sources": sources,', 1),
        ('"control_steps": 6000, "duration_s": 200, "sources": sources,',
         '"control_steps": 6000, "duration_s": 200, "seed": args.seed, "sources": sources,', 1),
    ], "seeded_style_CLI")
    namespace = ModuleType("_owned_native_seeded_style_CLI")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(PARENT.read_bytes(), str(PARENT) + ":owned_CLI", "exec"), namespace.__dict__)
    namespace.__doc__, namespace.style = __doc__, style
    exec(compile(source, str(PARENT) + ":seeded_main", "exec"), namespace.__dict__)
    return namespace, proof


def main(argv=None):
    namespace, _ = prepared_CLI()
    return namespace.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
