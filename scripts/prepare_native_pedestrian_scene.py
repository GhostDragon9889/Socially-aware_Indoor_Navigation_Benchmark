#!/usr/bin/env python3
"""Prepare deterministic known-free spawns/routes from an admitted native map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CrowdSim.humanoid.application.native_scene_preparation import (  # noqa: E402 -- direct-source CLI bootstrap
    prepare_configuration,
    publish_configuration,
    route_shifts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--native-map", type=Path, required=True)
    parser.add_argument("--probe-script-snapshot", type=Path)
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument("--clearance-m", type=float, default=.8)
    parser.add_argument("--spawn-clearance-m", type=float, default=3.)
    parser.add_argument("--minimum-spawn-separation-m", type=float, default=1.6)
    parser.add_argument("--route-shifts", type=int, nargs="+")
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        route_shifts(args.actors, args.route_shifts)
    except ValueError as error:
        parser.error(str(error))
    if args.output_config.resolve() == args.proof.resolve() or args.output_config.exists() or args.proof.exists():
        parser.error("Output configuration/proof must be different new paths; no overwrite")
    return args


def main(argv=None):
    args = parse_args(argv)
    configuration, proof = prepare_configuration(args.preflight, args.native_map, actors=args.actors,
        probe_script_snapshot=args.probe_script_snapshot, clearance_m=args.clearance_m,
        spawn_clearance_m=args.spawn_clearance_m, minimum_spawn_separation_m=args.minimum_spawn_separation_m,
        shifts=args.route_shifts)
    proof = publish_configuration(configuration, proof, args.output_config, args.proof)
    print(json.dumps({"configuration": proof["configuration"], "proof": str(args.proof.resolve()),
                      "actors": args.actors, "route_shifts": proof["route_shifts"],
                      "CPU_Astar_passed": True, "physical_navigation_validated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
