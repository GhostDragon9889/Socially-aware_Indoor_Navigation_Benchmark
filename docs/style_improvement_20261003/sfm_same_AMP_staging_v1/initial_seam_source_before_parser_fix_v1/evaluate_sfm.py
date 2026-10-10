"""Isolated SFM + unchanged saved AMP evaluation; production defaults unchanged.

The inherited pure CLI materializes the existing physics/route recipe. Its
Diffusion syntax is converted explicitly to actual SFM arguments before any
resource check or world launch. An unused DP checkpoint is never opened here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT_DRIVER = ROOT / "scripts/train_paper_amp.py"
PARENT_DRIVER_SHA256 = "d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3"
sys.path.insert(0, str(ROOT))

from sfm_runtime import SFMFactory


def parse_sfm_deployment_args(argv=None):
    """Original physical CLI/gates; SFM does not require an unused DP file."""
    from CrowdSim.humanoid.application import deployment

    original = inspect.getsource(deployment.parse_deployment_args)
    before = '    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)\n'
    after = '    parser.set_defaults(diffusion_checkpoint=Path("__unused_external_SFM_CLI_argument__"))\n'
    if original.count(before) != 1:
        raise ValueError("Original required Diffusion CLI boundary changed")
    revised = original.replace(before, after)
    revised = revised.replace("Deploy trained AMP + five-state Diffusion pedestrians", "Deploy the same trained AMP + external SFM targets")
    namespace = dict(vars(deployment))
    exec(compile(revised, str(__file__), "exec"), namespace)
    return namespace["parse_deployment_args"](argv)


def preflight_sfm_deployment(args, factory):
    """Private namespace reuse of exact original preflight/actor/native gates."""
    from CrowdSim.humanoid.application import deployment

    if not args.inference_only or args.arm_style_alpha != 0. or args.experimental_arm_style_checkpoint is not None:
        raise ValueError("Same-AMP SFM comparison requires explicit inference-only, alpha0 and no donor")
    if hashlib.sha256(PARENT_DRIVER.read_bytes()).hexdigest() != PARENT_DRIVER_SHA256:
        raise ValueError("The reviewed original physical/controller runner changed")
    namespace = {**vars(deployment), "preflight_amp_resources": factory.preflight}
    # Function body is unchanged; only this copy's unused-DP resource callback differs.
    exec(compile(inspect.getsource(deployment.preflight_deployment), str(__file__), "exec"), namespace)
    report = namespace["preflight_deployment"](args)
    report["schema"] = "local_sfm.same_AMP_evaluate_only_preflight.v1"
    report["entry"], report["driver"] = str(Path(__file__).resolve()), str(HERE / "evaluate_runner.py")
    report["parent_driver"] = {"path": str(PARENT_DRIVER), "sha256": PARENT_DRIVER_SHA256}
    report["actual_trajectory_source"] = "sfm"
    report["deployment_contract"].update(
        source="SFM targets + same saved physics AMP", goal_weight=None, near_horizon_waypoint_mass=None,
        inherited_diffusion_CLI_syntax="unused planner options only; actual parsed arguments source=sfm/checkpoint=None/samples0",
        inherited_agent_collision_flag="same full-scene replan trigger; no Diffusion collision objective",
        physical_execution_verified=False, author_exact_SFM_parameters=False)
    report["limitations"].append(
        "This source slice has CPU target-contract evidence only; no SFM physical baseline or paper8scene effect has run.")
    return report


def main(argv=None):
    from CrowdSim.humanoid.application import deployment

    args = parse_sfm_deployment_args(argv)
    factory = SFMFactory()
    report = preflight_sfm_deployment(args, factory)
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0
    deployment.require_fresh_sdk_process()
    for name in deployment.THREAD_ENVIRONMENT:
        os.environ[name] = "4"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "deployment_entry.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    spec = importlib.util.spec_from_file_location("isolated_same_AMP_sfm_runner", HERE / "evaluate_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def invoke(**options):
        return runner.main(target_generator_factory=factory, **options)

    result = deployment.invoke_existing_driver(
        report["driver_argv"], driver_main=invoke,
        evaluation_actor_request=report["evaluation_actor_request"],
        native_scene_request=report["native_scene_request"])
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
