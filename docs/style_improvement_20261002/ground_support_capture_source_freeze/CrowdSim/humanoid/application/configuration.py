"""Torch/Kit-free AMP CLI and profile loading; existing CLI paths use cwd."""

from __future__ import annotations

import argparse
from dataclasses import fields
import math
import os
from pathlib import Path

from .settings import TrainingConfig


AMP_CLI_DESCRIPTION = 'Train the action-conditioned AMP controller on real IsaacLab SMPL rollouts.\n\nDiffusion is a separately trained five-state x0 model. Plans, PD targets and\ndeclared joint-power measurements enter directly; no SFM or MaskedMimic is used.'


def parse_amp_args(argv=None):
    parser = argparse.ArgumentParser(description=AMP_CLI_DESCRIPTION)
    parser.add_argument("--config", default="configs/diffusion_amp/method_reconstruction.yaml")
    parser.add_argument("--manifest", default="data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--diffusion-checkpoint", default="output/diffusion_amp/diffusion_training/last.ckpt")
    parser.add_argument("--output", default="output/diffusion_amp/amp_training")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=15)
    parser.add_argument("--planner-batch-size", type=int, default=64)
    parser.add_argument("--diffusion-samples", type=int, default=1,
                        help="Explicit evaluation futures per actor; coupled guidance filters one coherent whole-scene sample")
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--terrain-extent-m", type=float, default=64.,
                        help="Length and width of the finite physical floor [m]; recorded in the physics contract")
    parser.add_argument("--spawn-margin-m", type=float, default=6.,
                        help="Explicit isolated training spawn-grid offset inside the finite physical terrain [m]")
    parser.add_argument("--trajectory-source", choices=("diffusion", "pacer_curriculum"), default="diffusion",
                        help="Low-level training commands: diffusion plans or independently generated PACER-style trajectories; deployment uses diffusion")
    parser.add_argument("--curriculum-mode", choices=("smooth", "pacer_knots"), default="smooth",
                        help="Explicit smooth acceleration adaptation, or public procedural knot recurrence; only for curriculum training")
    parser.add_argument("--curriculum-speed-max-mps", type=float, default=3.,
                        help="Explicit procedural command speed range [0,max] in m/s; resume checks the complete saved curriculum")
    parser.add_argument("--curriculum-accel-max-mps2", type=float, default=2.,
                        help="Explicit scalar and vector command acceleration limit; no physical state clipping")
    parser.add_argument("--reference-adapter", choices=("none", "causal_accel"), default="none",
                        help="Explicit evaluation adaptation of learned XY into a continuous 30Hz command; raw model output remains recorded")
    parser.add_argument("--reference-max-speed-mps", type=float, default=3.)
    parser.add_argument("--reference-max-accel-mps2", type=float, default=2.)
    parser.add_argument("--reference-initial-speed-policy", choices=("reject", "brake"), default="reject",
                        help="Reject an overspeed reference initial state or explicitly brake within acceleration limits; never change physical velocity")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-steps", type=int, default=0)
    parser.add_argument("--deterministic-evaluation", action="store_true",
                        help="Evaluation-only deterministic Torch algorithms and cuBLAS workspace; does not guarantee bitwise PhysX/Warp results")
    parser.add_argument("--scene-config", help="Declared shared validation geometry/routes, separate from author benchmark scenes")
    parser.add_argument("--route-guidance", choices=("endpoint", "timed_path"), default="endpoint",
                        help="Evaluation-only timed A* path waypoints, or the training single-endpoint guidance")
    parser.add_argument("--diffusion-guidance-mode", choices=("legacy_posterior", "trace_clean", "trace_clean_variance"),
                        default="legacy_posterior", help="Explicit diffusion guidance method; preserves the legacy default")
    parser.add_argument("--guidance-weight", type=float,
                        help="Total objective weight, divided across K waypoints; default K for legacy or total 10 for TRACE variants")
    parser.add_argument("--agent-collision-weight", type=float, default=0.,
                        help="Explicit joint clean-trajectory collision objective; replans the entire shared scene when any actor is due")
    parser.add_argument("--agent-collision-radius", type=float, default=.3,
                        help="Declared trajectory collision guidance disc radius in metres")
    parser.add_argument("--scene-collision-weight", type=float, default=0.,
                        help="Explicit signed-distance guidance against the declared physical scene geometry")
    parser.add_argument("--scene-collision-radius", type=float, default=.5,
                        help="Declared root-trajectory obstacle clearance radius [m]")
    parser.add_argument("--trajectory-speed-weight", type=float, default=0.)
    parser.add_argument("--max-trajectory-speed-mps", type=float, default=3.)
    parser.add_argument("--trajectory-accel-weight", type=float, default=0.)
    parser.add_argument("--max-trajectory-accel-mps2", type=float, default=2.)
    parser.add_argument("--pairwise-contacts", action="store_true",
                        help="Record exact filtered normal contacts during shared physical evaluation")
    parser.add_argument("--contact-threshold-n", type=float, default=1.,
                        help="Strict normal contact-force threshold in newtons; recorded independently from geometry proxies")
    parser.add_argument("--measure-ground-support", action="store_true",
                        help="Evaluation-only four-substep floor forces/live COM in a separate ground_support.npz; does not change physical_states fields")
    parser.add_argument("--ground-support-threshold-n", type=float, default=1.,
                        help="Strict floor-only normal force threshold [N]; requires --measure-ground-support for an explicit nondefault value")
    parser.add_argument("--allow-evaluation-planner-change", action="store_true",
                        help="Explicitly deploy a different trained diffusion checkpoint during evaluation; training provenance is preserved")
    parser.add_argument("--resolve-scene-assets", action="store_true", help="Write official storage API requests for the two missing local scene dependencies")
    parser.add_argument("--compile-diffusion-forward", action="store_true", help="Compile FP32 diffusion inference; may introduce documented floating-point differences")
    parser.add_argument("--physics-only", action="store_true", help="Disable the Kit renderer at startup for non-capture physics runs")
    parser.add_argument("--synchronized-replanning", action="store_true", help="Use a shared planning clock with immediate reset replanning; maximum interval is unchanged")
    parser.add_argument("--new-planning-stage", action="store_true", help="Explicitly transition an existing per-actor clock checkpoint to a synchronized planning stage")
    parser.add_argument("--power-source", choices=("projected_joint_effort_substep_mean", "legacy_pd_proxy_misaligned"),
                        default="projected_joint_effort_substep_mean", help="Measured active joint effort, or explicitly replay the old misaligned PD estimate")
    parser.add_argument("--new-power-stage", action="store_true", help="Explicitly change a legacy training checkpoint to synchronized measured joint power")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-every-steps", type=int, default=15,
                        help="Capture interval in 30Hz control steps; 2 yields a 15fps video without advancing extra physics")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    for name in ("curriculum_speed_max_mps", "curriculum_accel_max_mps2"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.deterministic_evaluation and args.evaluate_steps <= 0:
        parser.error("Deterministic evaluation requires a positive --evaluate-steps run")
    try:
        validate_deterministic_evaluation_environment(args.deterministic_evaluation)
    except ValueError as error:
        parser.error(str(error))
    if args.diffusion_samples < 1 or args.diffusion_samples > 20:
        parser.error("Diffusion sample count must be within 1..20")
    if args.diffusion_samples != 1 and not args.evaluate_steps:
        parser.error("Multiple diffusion samples are an explicit evaluation deployment variant")
    for name in ("reference_max_speed_mps", "reference_max_accel_mps2"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.reference_adapter != "none" and not args.evaluate_steps:
        parser.error("The continuous reference adaptation is an explicit diffusion deployment experiment")
    if args.reference_adapter == "none" and (args.reference_max_speed_mps != 3. or args.reference_max_accel_mps2 != 2. or args.reference_initial_speed_policy != "reject"):
        parser.error("Reference adaptation parameters require --reference-adapter causal_accel")
    for name in ("trajectory_speed_weight", "trajectory_accel_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    for name in ("max_trajectory_speed_mps", "max_trajectory_accel_mps2"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if max(args.trajectory_speed_weight, args.trajectory_accel_weight) > 0 and args.diffusion_guidance_mode == "legacy_posterior":
        parser.error("Trajectory dynamics guidance requires a clean guidance mode")
    if not math.isfinite(args.scene_collision_weight) or args.scene_collision_weight < 0:
        parser.error("Scene collision weight must be finite and nonnegative")
    if not math.isfinite(args.scene_collision_radius) or args.scene_collision_radius <= 0:
        parser.error("Scene collision radius must be finite and positive")
    if args.scene_collision_weight > 0 and (not args.evaluate_steps or not args.scene_config or args.diffusion_guidance_mode == "legacy_posterior"):
        parser.error("Scene collision guidance requires clean guidance and an explicit shared evaluation scene")
    if not math.isfinite(args.spawn_margin_m) or args.spawn_margin_m < 0:
        parser.error("Spawn margin must be finite and nonnegative")
    if not math.isfinite(args.terrain_extent_m) or args.terrain_extent_m < 32:
        parser.error("Terrain extent must be finite and at least 32 metres")
    if args.trajectory_source == "pacer_curriculum":
        if args.num_envs < 1:
            parser.error("Environment count must be positive")
        if args.evaluate_steps or args.compile_diffusion_forward:
            parser.error("The procedural curriculum is a low-level training source; evaluation requires the diffusion planner")
        if args.diffusion_guidance_mode != "legacy_posterior" or args.guidance_weight is not None or max(args.trajectory_speed_weight, args.trajectory_accel_weight) > 0:
            parser.error("Diffusion guidance options cannot change procedural curriculum training")
        # Leave the full command range plus a five-second horizon inside the
        # floor. This command bound is not a bound on actual physical velocity.
        clearance = args.curriculum_speed_max_mps * (args.episode_steps / 30. + 5.)
        grid_span = 4. * (math.ceil(math.sqrt(args.num_envs)) - 1)
        if min(args.spawn_margin_m, args.terrain_extent_m - args.spawn_margin_m - grid_span) < clearance:
            parser.error(f"Curriculum floor needs {args.curriculum_speed_max_mps:g}*(episode seconds+5) metres around the entire spawn grid; enlarge terrain and spawn margin")
    elif args.curriculum_mode != "smooth" or args.curriculum_speed_max_mps != 3. or args.curriculum_accel_max_mps2 != 2.:
        parser.error("Explicit curriculum parameters require --trajectory-source pacer_curriculum")
    if not math.isfinite(args.agent_collision_weight) or args.agent_collision_weight < 0:
        parser.error("Agent collision weight must be finite and nonnegative")
    if not math.isfinite(args.agent_collision_radius) or args.agent_collision_radius <= 0:
        parser.error("Agent collision radius must be finite and positive")
    if args.agent_collision_weight > 0 and (not args.evaluate_steps or args.diffusion_guidance_mode == "legacy_posterior" or args.planner_batch_size < args.num_envs):
        parser.error("Joint collision guidance requires shared evaluation, clean guidance and one planning batch containing the entire scene")
    if min(args.num_envs, args.updates, args.rollout_steps, args.replan_steps, args.planner_batch_size, args.episode_steps, args.checkpoint_every) < 1:
        parser.error("Training/physics bounds must be positive")
    if args.evaluate_steps < 0 or args.replan_steps > 30:
        parser.error("Evaluation bounds must be nonnegative; replan at least once per second")
    if args.evaluate_steps and not args.resume:
        parser.error("Evaluation requires an actual trained --resume checkpoint")
    if args.capture and not args.evaluate_steps:
        parser.error("Capture is available for bounded evaluation")
    if args.capture and args.physics_only:
        parser.error("Capture needs the renderer; omit --physics-only")
    if not 1 <= args.capture_every_steps <= 30:
        parser.error("Capture interval must be within 1..30 control steps")
    if args.capture_every_steps != 15 and not args.capture:
        parser.error("An explicit capture interval requires --capture")
    if (args.scene_config or args.allow_evaluation_planner_change) and not args.evaluate_steps:
        parser.error("Scene fixtures and planner changes are explicit evaluation options")
    if args.route_guidance == "timed_path" and (not args.evaluate_steps or not args.scene_config):
        parser.error("Timed path guidance requires evaluation and a declared shared scene")
    if args.guidance_weight is not None and (not math.isfinite(args.guidance_weight) or args.guidance_weight < 0):
        parser.error("Guidance weight must be finite and nonnegative")
    if not math.isfinite(args.contact_threshold_n) or args.contact_threshold_n <= 0:
        parser.error("Contact threshold must be finite and positive")
    if args.pairwise_contacts and (not args.evaluate_steps or args.power_source != "projected_joint_effort_substep_mean"):
        parser.error("Pairwise contacts require evaluation and projected joint power substep measurement")
    if not math.isfinite(args.ground_support_threshold_n) or args.ground_support_threshold_n <= 0:
        parser.error("Ground support threshold must be finite and positive")
    if args.measure_ground_support and (not args.evaluate_steps or args.power_source != "projected_joint_effort_substep_mean"):
        parser.error("Ground support requires evaluation and projected joint power substep measurement")
    if not args.measure_ground_support and args.ground_support_threshold_n != 1.:
        parser.error("Explicit ground support threshold requires --measure-ground-support")
    if args.new_planning_stage and (not args.resume or args.evaluate_steps or not args.synchronized_replanning):
        parser.error("A new planning stage requires training --resume and --synchronized-replanning")
    if args.new_power_stage and (not args.resume or args.evaluate_steps or args.power_source != "projected_joint_effort_substep_mean"):
        parser.error("A new power stage requires training --resume and projected joint power")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Learning rate must be finite and positive")
    if any(value is not None and value < 1 for value in (args.epochs, args.minibatch_size)):
        parser.error("Epoch and minibatch overrides must be positive")
    if not Path(args.diffusion_checkpoint).is_file():
        parser.error("Train the five-state diffusion model before starting AMP")
    output = Path(args.output)
    if args.evaluate_steps and any((output/name).exists() for name in ("result.json", "physical_states.npz", "frames")):
        parser.error("Evaluation output already contains artifacts; choose a new output directory")
    if args.measure_ground_support and any((output/name).exists() for name in ("ground_support.npz", "ground_support_partial.npz")):
        parser.error("Ground support output already contains artifacts; choose a new output directory")
    if not args.resume and any((output/name).exists() for name in ("last.pt", "metrics.jsonl")):
        parser.error("Training output already contains artifacts; use --resume or choose a new output directory")
    return args


def validate_deterministic_evaluation_environment(enabled):
    """Read the opt-in cuBLAS policy without mutating the environment."""
    if not enabled:
        return None
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is not None and existing not in {":16:8", ":4096:8"}:
        raise ValueError(
            f"Deterministic evaluation requires CUBLAS_WORKSPACE_CONFIG ':16:8' or ':4096:8'; got {existing!r}"
        )
    return existing if existing is not None else ":4096:8"


def prepare_deterministic_evaluation_environment(enabled):
    """Set an approved cuBLAS workspace before the SDK can initialize CUDA."""
    workspace = validate_deterministic_evaluation_environment(enabled)
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace)


def configure_deterministic_evaluation_torch(torch_module, enabled):
    """Opt in to strict algorithms; unsupported operations raise normally."""
    if enabled:
        torch_module.use_deterministic_algorithms(True, warn_only=False)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def runtime_numerics(torch_module, *, deterministic_evaluation_requested):
    """Observe actual process settings without changing the default run."""
    matmul = torch_module.backends.cuda.matmul
    cudnn = torch_module.backends.cudnn
    return {
        "deterministic_evaluation_requested": bool(deterministic_evaluation_requested),
        "torch_deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch_module.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "cudnn": {"benchmark": cudnn.benchmark, "deterministic": cudnn.deterministic,
                  "allow_tf32": cudnn.allow_tf32},
        "cuda_matmul": {"allow_tf32": matmul.allow_tf32,
                        "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
                        "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction},
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "physx_warp_bitwise_guarantee": False,
    }


class ConfigurationError(ValueError):
    """A profile is malformed or declares a contract the runner cannot honor."""


def load_yaml_mapping(path):
    """Read safe YAML, rejecting empty/scalar/list documents before SDK startup."""
    import yaml

    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Cannot read YAML configuration {path}: {error}") from error
    if not isinstance(document, dict):
        raise ConfigurationError(f"Configuration must be a YAML mapping: {path}")
    return document


def load_amp_profile(path):
    """Validate the single canonical TrainingConfig without importing Torch.

    CLI overrides remain the runner's responsibility. Metadata never silently
    changes learning algorithms, reward coefficients or artifact paths.
    """
    profile = load_yaml_mapping(path)
    if profile.get("mode") != "paper_method_reconstruction":
        raise ConfigurationError(
            "This runner implements paper_method_reconstruction. Exact author "
            "experiments require the unpublished author manifests and settings."
        )
    training = profile.get("training")
    if not isinstance(training, dict):
        raise ConfigurationError(f"Profile training must be a mapping: {path}")
    unknown = set(training) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ConfigurationError(f"Unknown AMP training fields: {sorted(unknown)}")
    try:
        config = TrainingConfig(**training)
        config.validate()
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigurationError(f"Invalid AMP training configuration {path}: {error}") from error
    if config.mode != profile["mode"]:
        raise ConfigurationError("Profile and training provenance modes disagree")
    if (config.self_dim, config.task_dim, config.amp_frame_dim, config.history_frames) != (357, 1044, 195, 10):
        raise ConfigurationError("Direct SMPL world requires self357/task1044/AMP195/history10")
    if not config.freeze_toes:
        raise ConfigurationError("Direct SMPL world requires the frozen hand/toe PD action contract")
    if not math.isclose(config.control_dt, 1 / 30, abs_tol=1e-9, rel_tol=0) or not math.isclose(
        config.target_horizon_s, 5., abs_tol=1e-9, rel_tol=0
    ):
        raise ConfigurationError("Direct SMPL world requires 30Hz control and five-second trajectories")
    reward = profile.get("reward", {})
    if not isinstance(reward, dict):
        raise ConfigurationError("Profile reward metadata must be a mapping")
    fixed = {"amp_weight": config.amp_reward_weight, "task_weight": config.task_reward_weight, "alpha_position": 2.,
             "alpha_energy": .0005, "alpha_direction": 1.8}
    for key, expected in fixed.items():
        if key in reward and (type(reward[key]) not in (int, float) or not math.isclose(
            reward[key], expected, abs_tol=1e-12, rel_tol=0
        )):
            raise ConfigurationError(f"Runner reward {key} is fixed at {expected}; metadata cannot change it")
    initialization = profile.get("initialization", {})
    if not isinstance(initialization, dict):
        raise ConfigurationError("Profile initialization must be a mapping")
    return profile


def validate_training_source(args, profile):
    """A declared training profile cannot silently run a different curriculum.

    Deployment always consumes the separately trained diffusion model. Old
    profiles have no trajectory_training declaration and retain their CLI.
    """
    declaration = profile.get("trajectory_training")
    if declaration is None:
        return
    if not isinstance(declaration, dict) or declaration.get("source") not in {"diffusion", "pacer_curriculum"}:
        raise ConfigurationError("trajectory_training must declare diffusion or pacer_curriculum")
    if getattr(args, "evaluate_steps", 0):
        return
    if declaration["source"] != getattr(args, "trajectory_source", "diffusion"):
        raise ConfigurationError("CLI trajectory source differs from the training profile; specify --trajectory-source explicitly")
    if declaration["source"] == "pacer_curriculum" and declaration.get("mode", "smooth") != getattr(args, "curriculum_mode", "smooth"):
        raise ConfigurationError("CLI curriculum mode differs from the explicit training profile")
