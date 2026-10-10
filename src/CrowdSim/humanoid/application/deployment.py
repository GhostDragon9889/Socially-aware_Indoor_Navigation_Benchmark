"""Explicit deployment shim; all physical control remains in the existing driver.

Imports and preflight use no Torch, NumPy, Kit, or simulator backend. This
interface supports the source-bound native scene bridge and the four-person
cuboid fixture; each run records its execution and quality evidence separately.
"""

from __future__ import annotations

from protomotions.repository import repository_root

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

from .configuration import load_amp_profile, parse_amp_args
from .resources import preflight_amp_resources


ROOT = repository_root(__file__)
DEFAULT_SCENE = ROOT / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
SCENE_SHA256 = "7971206da6fd50d9981dafd068b6005f326c6e7b6a8768e5a717aeb517573e9f"
DRIVER = ROOT / "scripts/train_paper_amp.py"
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def duration_control_steps(value):
    """Accept a finite positive duration representable by whole 30Hz steps."""
    try:
        duration = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Duration must be a finite positive number of seconds") from error
    if not duration.is_finite() or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    exact = duration * 30
    rounded = exact.to_integral_value()
    # Only accommodate decimal representations of a single 1/30s tick.
    if rounded < 1 or abs(exact-rounded) > Decimal("1e-9"):
        raise ValueError("Duration must correspond to a whole number of 30Hz control steps")
    return int(rounded)


def parse_deployment_args(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Deploy trained AMP + five-state Diffusion pedestrians in a source-bound shared physical scene. "
        "The existing evaluator owns the single App/world and control loop."
    ), allow_abbrev=False)
    parser.add_argument("--amp-checkpoint", type=Path, required=True,
                        help="Explicit immutable trained controller snapshot, mapped to evaluator --resume")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE,
                        help="Four-person cuboid fixture, or source-bound native configuration with --native-dynbench")
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--duration-seconds", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inference-only", action="store_true",
                        help="Use the evaluation actor without optimizer/replay/critic/D; preserves the explicit legacy initialization RNG")
    parser.add_argument("--experimental-arm-style-checkpoint", type=Path,
                        help="Experimental arm donor snapshot; requires --inference-only and is not a validated deployment model")
    parser.add_argument("--arm-style-alpha", type=float, default=0.)
    parser.add_argument("--native-dynbench", action="store_true",
                        help="Use a source-bound native scene configuration; requires --inference-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate resources and print forwarded argv without importing Torch/SDK or creating output")
    args = parser.parse_args(argv)
    try:
        args.control_steps = duration_control_steps(args.duration_seconds)
        if args.native_dynbench and (not args.inference_only or not 1 <= args.actors <= 15):
            raise ValueError("Native DynBench requires --inference-only and1..15 explicit actors")
        if not args.native_dynbench and args.actors != 4:
            raise ValueError("This deployment contract supports exactly four actors in the validated cuboid fixture")
        if not 0 <= args.seed < 2**32:
            raise ValueError("Seed must be within [0,2**32)")
        if args.amp_checkpoint.name.lower() in {"last.pt", "last.ckpt", "latest.pt", "latest.ckpt"}:
            raise ValueError("Use an immutable named AMP snapshot; mutable last/latest checkpoints are not deployment inputs")
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Deployment output must be a new directory; existing outputs are never reused")
        if not math.isfinite(args.arm_style_alpha) or not 0. <= args.arm_style_alpha <= 1.:
            raise ValueError("Arm style alpha must be finite and within [0,1]")
        if args.experimental_arm_style_checkpoint is not None:
            if not args.inference_only:
                raise ValueError("Experimental arm style requires --inference-only")
            if args.experimental_arm_style_checkpoint.name.lower() in {"last.pt", "last.ckpt", "latest.pt", "latest.ckpt"}:
                raise ValueError("Use an immutable named arm donor snapshot")
        elif args.arm_style_alpha != 0.:
            raise ValueError("A nonzero arm style alpha requires an explicit donor checkpoint")
    except ValueError as error:
        parser.error(str(error))
    return args


def driver_arguments(args):
    """Materialize the validated guide20/eta0 evaluation, independent of defaults.

    Inert training bounds are the original evaluator values for exact parsed
    compatibility. This shim never exposes a training request or changes weights.
    """
    values = (
        ("config", args.config.resolve()), ("manifest", args.manifest.resolve()),
        ("diffusion-checkpoint", args.diffusion_checkpoint.resolve()), ("output", args.output.resolve()),
        ("num-envs", args.actors), ("updates", 1000), ("rollout-steps", 32), ("replan-steps", 15),
        ("planner-batch-size", max(8, args.actors) if args.native_dynbench else 8),
        ("diffusion-samples", 1), ("episode-steps", args.control_steps+1),
        ("terrain-extent-m", 64.), ("spawn-margin-m", 6.), ("trajectory-source", "diffusion"),
        ("curriculum-mode", "smooth"), ("curriculum-speed-max-mps", 3.), ("curriculum-accel-max-mps2", 2.),
        ("reference-adapter", "causal_accel"), ("reference-max-speed-mps", 1.5),
        ("reference-max-accel-mps2", 2.), ("reference-initial-speed-policy", "brake"),
        ("checkpoint-every", 10), ("resume", args.amp_checkpoint.resolve()),
        ("evaluate-steps", args.control_steps), ("scene-config", args.scene_config.resolve()),
        ("route-guidance", "timed_path"), ("diffusion-guidance-mode", "trace_clean"),
        ("guidance-weight", 20.), ("near-horizon-waypoint-mass", 0.),
        ("agent-collision-weight", 5.), ("agent-collision-radius", .6),
        ("scene-collision-weight", 10.), ("scene-collision-radius", .8),
        ("trajectory-speed-weight", 0.), ("max-trajectory-speed-mps", 3.),
        ("trajectory-accel-weight", 0.), ("max-trajectory-accel-mps2", 2.),
        ("contact-threshold-n", 1.), ("ground-support-threshold-n", 1.),
        ("power-source", "projected_joint_effort_substep_mean"),
        ("capture-every-steps", 15), ("seed", args.seed),
    )
    forwarded = [part for name, value in values for part in (f"--{name}", str(value))]
    forwarded.extend(("--deterministic-evaluation", "--pairwise-contacts", "--measure-ground-support",
                      "--allow-evaluation-planner-change", "--physics-only", "--synchronized-replanning"))
    return forwarded


def preflight_deployment(args):
    """Reuse the same pure parser, profile/source gate and resource preflight."""
    forwarded = driver_arguments(args)
    driver_args = parse_amp_args(forwarded)
    profile = load_amp_profile(driver_args.config)
    resources = preflight_amp_resources(driver_args, profile)
    native_request = None
    if args.native_dynbench:
        from .native_deployment import preflight_native_scene
        native_request = preflight_native_scene(args.scene_config, actors=args.actors)
    elif hashlib.sha256(args.scene_config.resolve().read_bytes()).hexdigest() != SCENE_SHA256:
        raise ValueError("Only the validated cuboid fixture is supported here; arbitrary/DynBench scenes have no validated AMP deployment")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Deployment output must remain new after resource preflight")
    inference = None
    if args.inference_only:
        from CrowdSim.humanoid.runtime.checkpoint_contract import inspect_checkpoint_file
        identities = {}
        for role, checkpoint in (("base", args.amp_checkpoint),
                                 ("style", args.experimental_arm_style_checkpoint)):
            if checkpoint is None:
                continue
            selected = inspect_checkpoint_file(checkpoint)
            digest = hashlib.sha256()
            with selected.path.open("rb") as stream:
                for part in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(part)
            if not selected.unchanged():
                raise RuntimeError("Inference checkpoint changed during preflight")
            identities[role] = {"path": str(selected.path), "sha256": digest.hexdigest()}
        inference = {"base_checkpoint": identities["base"], "style_checkpoint": identities.get("style"),
            "alpha": args.arm_style_alpha, "experimental_arm_style": "style" in identities,
            "physical_execution_verified": False,
            "legacy_initialization_RNG": "explicit seed and CPU network initialization at the old load boundary"}
    return {
        "schema": "navisaaclab.explicit_pedestrian_deployment_preflight.v1",
        "status": "SDK_free_preflight_passed", "SDK_started": False, "Torch_imported_by_entry": False,
        "physical_execution_verified": False, "entry": str(ROOT / "scripts/simulate_paper_pedestrians.py"),
        "driver": str(DRIVER), "driver_argv": forwarded, "parsed_driver_arguments": vars(driver_args),
        "resource_preflight": resources.as_dict(), "control_steps": args.control_steps,
        "physical_transitions_planned": args.control_steps*args.actors,
        "actual_duration_seconds": args.control_steps/30.,
        "evaluation_actor_request": inference,
        "native_scene_request": native_request,
        "deployment_contract": {
            "profile": "explicit user input; checkpoint/config tensor compatibility remains the existing runtime gate",
            "scene": (native_request["configuration"]["scene"] if native_request else "four_actor_bounded_obstacle_fixture"),
            "scene_sha256": (native_request["configuration_file"]["sha256"] if native_request else SCENE_SHA256),
            "actors": args.actors, "control_hz": 30, "physics_hz": 120, "physics_only": True,
            "strict_Torch_determinism": True, "CPU_threads": 4, "PhysX_Warp_bitwise_guarantee": False,
            "source": "learned five-state Diffusion + trained physics AMP", "fallback_controller": None,
            "goal_weight": 20., "near_horizon_waypoint_mass": 0.,
            "shared_clock_steps": 15, "future_3D_pose_actor_inputs": False,
            "contacts": "native filtered static/peer + separate four-substep feet-floor/live COM",
            "reference_adapter": "causal_accel; 1.5m/s, 2m/s2, overspeed brake; no physical teleport",
            "driver_inert_training_arguments": {"updates": 1000, "rollout_steps": 32, "checkpoint_every": 10},
        },
        "limitations": ["Preflight validates files/declared shapes, not weights or navigation success.",
                        ("Native scene resource gates do not prove human navigation or the authors' eight-scene results."
                         if native_request else "This local cuboid fixture is not an author scene or DynBench reproduction."),
                        "One existing evaluator creates/owns App and world; this shim has no control loop."],
    }


def make_evaluation_actor_factory(request):
    """Defer tensor imports until the sole driver's App has started."""
    def load(checkpoint, *, device):
        from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
        base, style = request["base_checkpoint"], request["style_checkpoint"]
        if Path(checkpoint).resolve() != Path(base["path"]):
            raise ValueError("Driver checkpoint differs from the selected inference artifact")
        actor = EvaluationActor.from_checkpoints(base["path"], None if style is None else style["path"],
            alpha=request["alpha"], device=device, base_sha256=base["sha256"],
            style_sha256=None if style is None else style["sha256"])
        actor.legacy_rng_preparation = prepare_legacy_evaluation_rng(actor.config)
        return actor
    return load


def invoke_existing_driver(forwarded, driver_main=None, *, evaluation_actor_request=None, native_scene_request=None):
    """Delegate once and restore the caller's argv even if the driver raises."""
    original = sys.argv
    try:
        sys.argv = [str(DRIVER), *forwarded]
        if driver_main is None:
            require_fresh_sdk_process()
            spec = importlib.util.spec_from_file_location("explicit_pedestrian_existing_driver", DRIVER)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            driver_main = module.main
        if evaluation_actor_request is None:
            return driver_main()
        options = {"evaluation_actor_factory": make_evaluation_actor_factory(evaluation_actor_request)}
        if native_scene_request is not None:
            from .native_deployment import make_native_scene_factory
            options["native_scene_factory"] = make_native_scene_factory(native_scene_request)
        return driver_main(**options)
    finally:
        sys.argv = original


def require_fresh_sdk_process():
    """A standalone entry must not launch a second App in an existing SDK session."""
    if any(name in {"isaacsim", "isaaclab", "omni.kit.app", "omni.physx"}
           or name.startswith(("isaacsim.", "isaaclab.")) for name in sys.modules):
        raise RuntimeError("Run this deployment entry in a fresh Python process; an SDK session is already imported")


def main(argv=None, *, driver_main=None):
    args = parse_deployment_args(argv)
    report = preflight_deployment(args)
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0
    if driver_main is None:
        require_fresh_sdk_process()
    # The driver sets torch.set_num_threads(4); constrain BLAS before imports.
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "4"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "deployment_entry.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    result = invoke_existing_driver(report["driver_argv"], driver_main,
                                   evaluation_actor_request=report["evaluation_actor_request"],
                                   native_scene_request=report["native_scene_request"])
    return 0 if result is None else result


__all__ = ["duration_control_steps", "parse_deployment_args", "driver_arguments",
           "preflight_deployment", "invoke_existing_driver", "main"]
