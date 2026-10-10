#!/usr/bin/env python3
"""Audit explicitly seeded native captures without changing measurement math."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_seeded_audit as native  # noqa: E402
from scripts import evaluate_shared_social_native_seeded as entry  # noqa: E402

PARENT = ROOT / "scripts/audit_shared_social_native.py"
PARENT_SHA = "1bdb0b784ceb888b205f87c5ec4352d81afa4faec60b05bca650f49d2026f208"


def prepared_CLI():
    if native.original.record(PARENT)["sha256"] != PARENT_SHA:
        raise ValueError("Frozen native raw audit CLI changed")
    source = entry.seed_main_source(PARENT.read_text())
    source, proof = native.frozen._derive(source, [
        ('    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)\n',
         '    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)\n'
         '    parser.add_argument("--seed", type=int, required=True)\n', 1),
        ('kernels = native.prepare_kernels(actors=args.actors, controls=args.control_steps)',
         'kernels = native.prepare_kernels(actors=args.actors, controls=args.control_steps, seed=args.seed)', 1),
        ('planner=args.planner, scene_config=args.scene_config, scene_sha256=args.scene_sha256)',
         'planner=args.planner, scene_config=args.scene_config, scene_sha256=args.scene_sha256, seed=args.seed)', 1),
        ('"planner": args.planner,', '"planner": args.planner, "seed": args.seed,', 2),
    ], "seeded_raw_CLI")
    namespace = ModuleType("_owned_native_seeded_audit_CLI")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(PARENT.read_bytes(), str(PARENT) + ":owned_CLI", "exec"), namespace.__dict__)
    namespace.__doc__, namespace.native = __doc__, native
    exec(compile(source, str(PARENT) + ":seeded_main", "exec"), namespace.__dict__)
    return namespace, proof


def main(argv=None):
    namespace, _ = prepared_CLI()
    return namespace.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
