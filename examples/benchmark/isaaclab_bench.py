# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the maintained H1-2 robot configuration in IsaacLab."""

import argparse
import gc
import inspect
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, help="Write benchmark and complete shutdown evidence as JSON.")
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 1 or args.warmup_steps < 0:
        parser.error("--num-envs and --steps must be positive; --warmup-steps must be nonnegative")
    return args


class BenchmarkWorld:
    """Own the one vectorized articulation, including a failed constructor."""

    def __init__(self, app, args):
        import torch
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig, IsaacLabSimParams
        from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig
        from protomotions.components.terrains.terrain import Terrain
        from protomotions.components.terrains.config import TerrainConfig
        from protomotions.components.scene_lib import SceneLib

        self.simulator = None
        self.device = torch.device(args.device)
        # The old h1.xml is absent. This is the maintained H1-2 benchmark.
        robot_config = H1_2RobotConfig()
        config = IsaacLabSimulatorConfig(
            sim=IsaacLabSimParams(fps=200, decimation=4), headless=True,
            num_envs=args.num_envs, experiment_name="h1_2_isaaclab_example",
        )
        terrain = Terrain(config=TerrainConfig(), num_envs=config.num_envs, device=self.device)
        scene_lib = SceneLib.empty(num_envs=config.num_envs, device=self.device)
        self.simulator = IsaacLabSimulator.__new__(IsaacLabSimulator)
        IsaacLabSimulator.__init__(
            self.simulator, config=config, robot_config=robot_config, terrain=terrain,
            scene_lib=scene_lib, device=self.device, simulation_app=app,
        )
        self.simulator._initialize_with_markers({})


def _run_benchmark(app, args, owner, report, lifecycle):
    owner.physics_frame = inspect.currentframe()
    import torch
    from rich.progress import track

    if lifecycle is not None:
        owner.world = lifecycle.construct_world(BenchmarkWorld, app, args)
        lifecycle.verify_loaded()
    else:
        owner.world = BenchmarkWorld.__new__(BenchmarkWorld)
        BenchmarkWorld.__init__(owner.world, app, args)
    simulator = owner.world.simulator
    device = owner.world.device
    config, robot_config = simulator.config, simulator.robot_config
    default_state = simulator.get_default_robot_reset_state()
    root_pos = torch.zeros(config.num_envs, 3, device=device)
    root_pos[:, :2] = simulator.terrain.sample_valid_locations(config.num_envs).view(-1, 2)
    root_pos[:, 2] = 1.0
    default_state.root_pos[:] = root_pos
    simulator.reset_envs(default_state, env_ids=torch.arange(config.num_envs, device=device))

    for _ in track(range(args.warmup_steps), description="Performing warmup steps"):
        simulator.step(torch.randn(config.num_envs, robot_config.number_of_actions, device=device))
        report["completed_warmup_steps"] += 1
    print(f"Warmup complete: {args.warmup_steps} control steps", flush=True)

    start_time = time.perf_counter()
    for _ in track(range(args.steps), description="Running benchmark"):
        simulator.step(torch.randn(config.num_envs, robot_config.number_of_actions, device=device))
        report["completed_measured_steps"] += 1
    total_time = time.perf_counter() - start_time
    report.update(status="passed", measured_seconds=total_time,
                  average_control_step_ms=total_time / args.steps * 1000.)
    print("\nBenchmark Results (H1-2):")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per step: {report['average_control_step_ms']:.2f} ms")
    print(f"Completed {args.steps} measured control steps across {config.num_envs} environments", flush=True)


def _drop_exception_tracebacks(error):
    """Keep the original exception/chain, but release completed native owner frames."""
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        current.__traceback__ = None


def main(argv=None):
    args = parse_args(argv)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle, uses_sim61
    from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame

    report = {"status": "starting", "robot": "h1_2", "num_envs": args.num_envs,
              "completed_warmup_steps": 0, "completed_measured_steps": 0,
              "cleanup_succeeded": False, "cleanup_errors": [], "app_close_returned": False}
    lifecycle = SDK61Lifecycle(report) if uses_sim61() else None
    from isaaclab.app import AppLauncher

    flags = {"headless": True, "device": args.device, "fast_shutdown": False}
    if lifecycle is not None:
        flags["kit_args"] = lifecycle.kit_args
    simulation_app = AppLauncher(flags).app
    owner = SimpleNamespace(world=None, physics_frame=None)
    failure = None
    try:
        _run_benchmark(simulation_app, args, owner, report, lifecycle)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("\nSimulation stopped by user")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
        _drop_exception_tracebacks(error)
        failure = error
    finally:
        try:
            report["frame_release"] = clear_completed_physics_frame(owner, _run_benchmark)
        except Exception as error:
            report["cleanup_errors"].append(f"Benchmark frame release: {type(error).__name__}: {error}")
        gc.collect()
        print("Closing simulator", flush=True)
        if lifecycle is not None:
            try:
                lifecycle.close(simulation_app, owner.world)
            except Exception as error:
                report["cleanup_errors"].append(f"Benchmark lifecycle: {type(error).__name__}: {error}")
                traceback.print_exc()
        else:
            simulator = getattr(owner.world, "simulator", None)
            if simulator is None or not hasattr(simulator, "_simulation_app"):
                simulation_app.close(exit_code=int(failure is not None))
            else:
                simulator._shutdown_exit_code = int(failure is not None)
                simulator.close()
            report.update(cleanup_succeeded=not report["cleanup_errors"], app_close_returned=True,
                          cleanup_protocol="legacy simulator.close")
        owner.world = None
        gc.collect()
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        if args.report is not None:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("Benchmark lifecycle: " + json.dumps(report), flush=True)
        if report["app_close_returned"]:
            print("Simulator close returned", flush=True)
    if failure is not None:
        raise failure
    if not report["cleanup_succeeded"]:
        raise RuntimeError("Benchmark cleanup failed: " + "; ".join(report["cleanup_errors"]))


if __name__ == "__main__":
    main()
