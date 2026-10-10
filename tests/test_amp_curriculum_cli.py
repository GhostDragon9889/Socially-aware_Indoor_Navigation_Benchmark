"""Reject incompatible curriculum runs before Kit and preserve deployment."""
from types import SimpleNamespace

import pytest

from CrowdSim.humanoid.application.configuration import (
    ConfigurationError,
    parse_amp_args,
    validate_training_source,
)


def curriculum_args(tmp_path, *flags):
    checkpoint = tmp_path / "separate_diffusion.ckpt"
    checkpoint.write_bytes(b"CLI-existence-only")
    return parse_amp_args(["--diffusion-checkpoint", str(checkpoint), "--output", str(tmp_path / "fresh"),
                           "--trajectory-source", "pacer_curriculum", "--terrain-extent-m", "128",
                           "--spawn-margin-m", "48", *flags])


def test_independent_curriculum_has_sufficient_command_and_episode_floor_margin(tmp_path):
    args = curriculum_args(tmp_path)
    assert args.curriculum_mode == "smooth" and args.episode_steps == 300
    assert args.terrain_extent_m == 128 and args.spawn_margin_m == 48


@pytest.mark.parametrize("flags", [
    ("--terrain-extent-m", "64"), ("--terrain-extent-m", "nan"),
    ("--spawn-margin-m", "40"), ("--episode-steps", "600"),
    ("--evaluate-steps", "1", "--resume", "trained.pt"),
    ("--compile-diffusion-forward",), ("--guidance-weight", "10"),
    ("--diffusion-guidance-mode", "trace_clean"), ("--num-envs", "-1"),
])
def test_curriculum_configuration_errors_do_not_start_a_simulator(tmp_path, flags):
    with pytest.raises(SystemExit) as failure:
        curriculum_args(tmp_path, *flags)
    assert failure.value.code == 2


def test_explicit_profile_does_not_silently_train_with_default_diffusion_commands():
    profile = {"trajectory_training": {"source": "pacer_curriculum", "mode": "smooth"}}
    args = SimpleNamespace(trajectory_source="diffusion", curriculum_mode="smooth", evaluate_steps=0)
    with pytest.raises(ConfigurationError, match="specify --trajectory-source"):
        validate_training_source(args, profile)
    args.trajectory_source = "pacer_curriculum"
    validate_training_source(args, profile)
    args.curriculum_mode = "pacer_knots"
    with pytest.raises(ConfigurationError, match="mode"):
        validate_training_source(args, profile)


def test_curriculum_trained_controller_deploys_with_diffusion():
    profile = {"trajectory_training": {"source": "pacer_curriculum", "mode": "smooth"}}
    validate_training_source(SimpleNamespace(trajectory_source="diffusion", evaluate_steps=6000), profile)
    validate_training_source(SimpleNamespace(), {})  # Historical profiles keep their contract.


@pytest.mark.parametrize("flags", [
    ("--reference-adapter", "causal_accel"),
    ("--reference-max-speed-mps", "1.5"),
    ("--reference-max-accel-mps2", "nan"),
])
def test_adaptation_cannot_be_silently_applied_to_curriculum_training(tmp_path, flags):
    with pytest.raises(SystemExit) as failure:
        curriculum_args(tmp_path, *flags)
    assert failure.value.code == 2


def test_declared_diffusion_evaluation_accepts_continuous_reference_options(tmp_path):
    checkpoint = tmp_path / "diffusion.ckpt"
    checkpoint.write_bytes(b"CLI-only")
    args = parse_amp_args(["--diffusion-checkpoint", str(checkpoint), "--output", str(tmp_path / "fresh"),
                           "--evaluate-steps", "6000", "--resume", "trained.pt",
                           "--reference-adapter", "causal_accel", "--reference-max-speed-mps", "1.5",
                           "--reference-initial-speed-policy", "brake"])
    assert args.reference_adapter == "causal_accel" and args.reference_initial_speed_policy == "brake"


def test_render_capture_interval_is_explicit_and_renderer_required(tmp_path):
    checkpoint = tmp_path / "diffusion.ckpt"
    checkpoint.write_bytes(b"CLI-only")
    base = ["--diffusion-checkpoint", str(checkpoint), "--output", str(tmp_path / "fresh"),
            "--evaluate-steps", "900", "--resume", "trained.pt", "--capture-every-steps", "2"]
    with pytest.raises(SystemExit):
        parse_amp_args(base)
    assert parse_amp_args([*base, "--capture"]).capture_every_steps == 2
    with pytest.raises(SystemExit):
        parse_amp_args([*base, "--capture", "--physics-only"])


@pytest.mark.parametrize("samples", ["0", "21", "2"])
def test_multi_sample_configuration_is_bounded_and_explicit_deployment(tmp_path, samples):
    with pytest.raises(SystemExit):
        curriculum_args(tmp_path, "--diffusion-samples", samples)


def test_multi_sample_deployment_keeps_same_scene_batch(tmp_path):
    checkpoint = tmp_path / "diffusion.ckpt"
    checkpoint.write_bytes(b"CLI-only")
    args = parse_amp_args(["--diffusion-checkpoint", str(checkpoint), "--output", str(tmp_path / "fresh"),
                           "--evaluate-steps", "900", "--resume", "trained.pt", "--diffusion-samples", "2",
                           "--diffusion-guidance-mode", "trace_clean", "--num-envs", "4", "--agent-collision-weight", "5"])
    assert args.diffusion_samples == 2 and args.planner_batch_size >= args.num_envs
