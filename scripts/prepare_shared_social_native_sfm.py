#!/usr/bin/env python3
"""CPU prediction admission on registered native scenes; no physical rollout.

Inputs are declared zero-velocity spawn snapshots with repeated synthetic
history, not measurements from an AMP World. All prediction failures remain in
the report; this command never estimates navigation success, falls or style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Require a new output directory; preserve all prior failures")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0,2**32)")
    selected = [path.resolve() for path in args.scene_config]
    if len(selected) != len(set(selected)):
        parser.error("Each selected scene configuration must occur exactly once")
    return args


def main(argv=None):
    args = parse_args(argv)
    import yaml
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
    from CrowdSim.humanoid.trajectory.native_social_force_runtime import NativeSocialForcePlanner, source_bindings
    from CrowdSim.humanoid.trajectory.social_force_runtime import file_identity

    sources = source_bindings()
    sources["entry"] = file_identity(__file__)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "shared_social.native_SFM_CPU_prediction_preparation.v1", "status": "pending",
              "sources": sources, "seed": args.seed, "selected_configurations":
              [file_identity(path) for path in args.scene_config], "cases": [],
              "input_semantics": "Registered spawn XY; zero velocity/heading; repeated synthetic history; no physical measurements",
              "SDK_started": False, "GPU_used": False, "AMP_checkpoint_loaded": False,
              "actual_world_boundary_checked": False, "physical_navigation_validated": False,
              "motion_style_measured": False, "eight_author_scenes_restored": False,
              "actual_process_exit_observed": False}
    target = output / "CPU_preparation.json"

    def write_report():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write_report()
    # CPU-only tensor snapshots; no App, World, policy or physical measurements.
    import numpy as np
    import torch
    rng, np_rng = torch.get_rng_state().clone(), np.random.get_state()
    for index, path in enumerate(args.scene_config):
        case = {"configuration": file_identity(path), "status": "pending", "failure_retained": True,
                "SDK_executed": False, "physical_navigation_validated": False}
        report["cases"].append(case)
        write_report()
        try:
            document = yaml.safe_load(path.read_text())
            count = len(document["origins_xy"])
            case["actors"] = count
            case["native_resource_request"] = preflight_native_scene(path, actors=count)
            planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=path, actors=count, seed=args.seed)
            case["planner"] = planner.describe()
            state = torch.zeros(count, 5, dtype=torch.float32)
            state[:, :2] = planner._scene.origins
            route_indices = torch.zeros(count, dtype=torch.int64)
            goals, updated, reached = planner._scene.route_waypoints(state[:, :2], route_indices, advance=False)
            if (planner._scene.global_path_failures or not torch.equal(updated, route_indices) or bool(reached.any())):
                raise ValueError("A* snapshot admission failed; no hold-position fallback")
            history = state[:, None].repeat(1, 31, 1)
            neighbors = torch.stack([history[[peer for peer in range(count) if peer != actor]] for actor in range(count)])
            input_archive = output / f"snapshot_input_{index:03d}.npz"
            np.savez_compressed(input_archive, input_states=state.numpy(), Astar_waypoints=goals.numpy(),
                                route_indices=route_indices.numpy())
            case["input_archive"] = file_identity(input_archive)
            write_report()
            prediction = planner.plan(states=state, history=history, neighbors=neighbors,
                raster=planner._scene.raster(state), history_mask=torch.ones(count, 31, dtype=torch.bool),
                neighbor_mask=torch.ones(count, count - 1, 31, dtype=torch.bool),
                waypoints=goals, waypoint_times=torch.arange(1, 11, dtype=torch.float32) / 2,
                num_samples=1, guidance_weight=20., guidance_mode="trace_clean", near_horizon_mass=0.,
                scene_ids=torch.zeros(count, dtype=torch.int64), agent_collision_weight=5., agent_collision_radius=.6,
                scene_collision_weight=10., scene_collision_radius=.5, trajectory_speed_weight=0.,
                max_trajectory_speed_mps=1.5, trajectory_accel_weight=0., max_trajectory_accel_mps2=2.,
                scene_bounds_xy=None, obstacle_centers_xy=None, obstacle_sizes_xy=None,
                scene_collision_cost_fn=planner._scene.collision_cost)
            archive = output / f"snapshot_{index:03d}.npz"
            np.savez_compressed(archive, input_states=state.numpy(), Astar_waypoints=goals.numpy(),
                                raw_xy=prediction.xy.numpy(), times_s=np.arange(51, dtype=np.float64) / 10)
            case.update(status="CPU_joint_prediction_passed", archive=file_identity(archive),
                        output_shape=list(prediction.xy.shape), planner_metadata=planner.last_metadata)
        except Exception as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                        prediction_replaced_with_fallback=False)
        write_report()
    after = np.random.get_state()
    report["RNG_unchanged"] = bool(torch.equal(rng, torch.get_rng_state())
        and np_rng[0] == after[0] and np.array_equal(np_rng[1], after[1]) and np_rng[2:] == after[2:])
    report["CUDA_initialized"] = torch.cuda.is_initialized()
    actual_sources = source_bindings()
    actual_sources["entry"] = file_identity(__file__)
    report["sources_unchanged"] = actual_sources == sources
    report["configurations_unchanged"] = report["selected_configurations"] == [file_identity(path) for path in args.scene_config]
    passed = (all(case["status"] == "CPU_joint_prediction_passed" for case in report["cases"])
              and report["RNG_unchanged"] and not report["CUDA_initialized"]
              and report["sources_unchanged"] and report["configurations_unchanged"])
    report["status"] = "CPU_preparation_passed" if passed else "failed"
    write_report()
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "report": str(target),
                      "physical_navigation_validated": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
