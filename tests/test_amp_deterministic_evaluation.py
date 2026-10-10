"""Strict evaluation controls verified on CPU, without importing the SDK."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.application.configuration import (
    configure_deterministic_evaluation_torch,
    parse_amp_args,
    prepare_deterministic_evaluation_environment,
    runtime_numerics,
    validate_deterministic_evaluation_environment,
)


@pytest.fixture
def diagnostic_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/train_paper_amp.py"
    spec = importlib.util.spec_from_file_location("paper_amp_capture_diagnostic_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


@pytest.fixture
def diagnostic_world():
    from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion
    state = RobotState(state_conversion=StateConversion.COMMON,
                       rigid_body_pos=torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
                       rigid_body_rot=torch.tensor([[[0., 0., 0., 1.]]]).expand(1, 4, 4).clone(),
                       rigid_body_vel=torch.zeros(1, 4, 3), dof_pos=torch.zeros(1, 69),
                       dof_vel=torch.zeros(1, 69))
    return SimpleNamespace(state=lambda: state, raw_state=state,
                           simulator=SimpleNamespace(_sim=SimpleNamespace(_physics_step_count=12)), native_time=.1)


def test_capture_snapshots_detect_live_buffer_mutation_without_aliasing(diagnostic_runner, diagnostic_world):
    before = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: diagnostic_world.native_time)
    diagnostic_world.raw_state.rigid_body_pos.add_(.25)
    diagnostic_world.raw_state.dof_vel[0, 6] = 2.
    diagnostic_world.simulator._sim._physics_step_count += 1
    diagnostic_world.native_time += 1/120
    after = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: diagnostic_world.native_time)
    report = diagnostic_runner.compare_capture_state(before, after, torch)
    assert report["max_absolute_buffer_deltas"]["root_pos"] == .25
    assert report["max_absolute_buffer_deltas"]["rigid_body_pos"] == .25
    assert report["max_absolute_buffer_deltas"]["dof_vel"] == 2.
    assert report["max_absolute_buffer_deltas"]["root_rot"] == 0.
    assert report["physics_step_count"]["delta"] == 1
    assert report["native_physx_time_s"]["delta"] == pytest.approx(1/120)
    assert report["rng"]["torch_cpu_rng"]["equal"]
    assert not report["rng"]["torch_cuda_rng"]["observed"]
    assert not torch.cuda.is_initialized()


def test_capture_snapshots_report_unchanged_state_and_rng_without_consuming_it(diagnostic_runner, diagnostic_world):
    global_rng = torch.get_rng_state().clone()
    before = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: .1)
    after = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: .1)
    report = diagnostic_runner.compare_capture_state(before, after, torch)
    assert all(value == 0. for value in report["max_absolute_buffer_deltas"].values())
    assert report["native_physx_time_s"]["delta"] == report["physics_step_count"]["delta"] == 0
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert report["rng"]["torch_cpu_rng"]["before_sha256"] == report["rng"]["torch_cpu_rng"]["after_sha256"]


def test_capture_snapshot_without_native_clock_keeps_other_diagnostics(diagnostic_runner, diagnostic_world, monkeypatch):
    # Reproduce the actual public binding contract: an interface exists but
    # get_simulation_time does not. Real RobotState tensors still supply data.
    omni = ModuleType("omni")
    physx = ModuleType("omni.physx")
    physx.get_physx_interface = lambda: SimpleNamespace()
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)
    before = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch)
    diagnostic_world.raw_state.dof_vel[0, 3] = .5
    after = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch)
    report = diagnostic_runner.compare_capture_state(before, after, torch)
    clock = report["native_physx_time_s"]
    assert clock["observed"] is False
    assert clock["before"] is clock["after"] is clock["delta"] is None
    assert not clock["before_metadata"]["available"]
    assert "does not expose callable get_simulation_time" in clock["before_metadata"]["reason"]
    assert clock["before_metadata"]["source"].startswith("omni.physx.get_physx_interface()")
    assert "sdk_distribution_version" in clock["before_metadata"]
    assert report["max_absolute_buffer_deltas"]["dof_vel"] == .5
    assert report["physics_step_count"]["delta"] == 0
    assert report["rng"]["torch_cpu_rng"]["equal"]
    assert not torch.cuda.is_initialized()


def test_capture_snapshots_detect_changed_cpu_and_mock_cuda_rng(diagnostic_runner, diagnostic_world, monkeypatch):
    # CPU math/state remain real; only CUDA's availability/RNG API is mocked.
    fake_rng = torch.arange(8, dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: fake_rng)
    cpu_rng = torch.get_rng_state().clone()
    try:
        before = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: .1)
        fake_rng[0] += 1
        torch.rand(1)
        after = diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: .1)
        report = diagnostic_runner.compare_capture_state(before, after, torch)
        assert not report["rng"]["torch_cpu_rng"]["equal"]
        assert not report["rng"]["torch_cuda_rng"]["equal"]
        assert report["rng"]["torch_cuda_rng"]["before_sha256"] != report["rng"]["torch_cuda_rng"]["after_sha256"]
    finally:
        torch.set_rng_state(cpu_rng)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_capture_snapshot_rejects_nonfinite_state_and_time(diagnostic_runner, diagnostic_world, invalid):
    with pytest.raises(ValueError, match="native PhysX simulation time"):
        diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: invalid)
    diagnostic_world.raw_state.dof_pos[0, 2] = invalid
    with pytest.raises(ValueError, match="RobotState.dof_pos"):
        diagnostic_runner.capture_state_snapshot(diagnostic_world, torch, native_time_getter=lambda: .1)


def cli_args(tmp_path, *extra):
    checkpoint = tmp_path / "diffusion.ckpt"
    checkpoint.write_bytes(b"CPU-parser-file-presence-fixture-not-model-weights")
    return parse_amp_args(["--diffusion-checkpoint", str(checkpoint),
                           "--output", str(tmp_path / "run"), *extra])


def test_flag_is_disabled_by_default_and_evaluation_only(tmp_path):
    assert not cli_args(tmp_path).deterministic_evaluation
    with pytest.raises(SystemExit) as error:
        cli_args(tmp_path, "--deterministic-evaluation")
    assert error.value.code == 2
    args = cli_args(tmp_path, "--deterministic-evaluation", "--evaluate-steps", "30", "--resume", "resume.pt")
    assert args.deterministic_evaluation


def test_parser_validates_opt_in_workspace_read_only_before_sdk(tmp_path, monkeypatch):
    before_sdk = {name for name in sys.modules if name.startswith(("omni", "isaacsim", "isaaclab"))}
    evaluation = ("--deterministic-evaluation", "--evaluate-steps", "30", "--resume", "resume.pt")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unapproved")
    assert not cli_args(tmp_path).deterministic_evaluation
    assert validate_deterministic_evaluation_environment(False) is None
    with pytest.raises(SystemExit) as error:
        cli_args(tmp_path, *evaluation)
    assert error.value.code == 2
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "unapproved"
    for workspace in (":16:8", ":4096:8"):
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
        assert cli_args(tmp_path, *evaluation).deterministic_evaluation
        assert validate_deterministic_evaluation_environment(True) == workspace
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == workspace
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    assert cli_args(tmp_path, *evaluation).deterministic_evaluation
    assert validate_deterministic_evaluation_environment(True) == ":4096:8"
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert {name for name in sys.modules if name.startswith(("omni", "isaacsim", "isaaclab"))} == before_sdk


def test_workspace_default_is_set_before_sdk_and_preserves_rng(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = torch.get_rng_state().clone()
    prepare_deterministic_evaluation_environment(True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("workspace", [":16:8", ":4096:8"])
def test_existing_approved_workspace_is_preserved(monkeypatch, workspace):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    prepare_deterministic_evaluation_environment(True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == workspace


@pytest.mark.parametrize("workspace", ["", ":4096:2", "4096:8", "unapproved"])
def test_invalid_workspace_is_rejected_without_overwrite(monkeypatch, workspace):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        prepare_deterministic_evaluation_environment(True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == workspace


def test_default_run_does_not_validate_or_mutate_existing_environment(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "caller-owned-value")
    prepare_deterministic_evaluation_environment(False)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "caller-owned-value"


@pytest.fixture
def torch_numerics_state():
    state = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    cuda_initialized = torch.cuda.is_initialized()
    try:
        yield state
    finally:
        torch.use_deterministic_algorithms(state[0], warn_only=state[1])
        torch.backends.cudnn.benchmark = state[2]
        torch.backends.cudnn.deterministic = state[3]
        assert torch.cuda.is_initialized() == cuda_initialized


def test_strict_torch_controls_and_actual_report_without_rng_or_precision_changes(monkeypatch, torch_numerics_state):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    before_rng = torch.get_rng_state().clone()
    before_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    before_precision = torch.get_float32_matmul_precision()
    configure_deterministic_evaluation_torch(torch, True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
    assert torch.equal(before_rng, torch.get_rng_state())
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before_tf32
    assert torch.get_float32_matmul_precision() == before_precision
    report = runtime_numerics(torch, deterministic_evaluation_requested=True)
    assert report["torch_deterministic_algorithms"] and not report["torch_deterministic_warn_only"]
    assert report["cublas_workspace_config"] == ":16:8"
    assert report["cuda_matmul"]["allow_tf32"] == before_tf32[0]
    assert report["physx_warp_bitwise_guarantee"] is False
    # Actual CPU math executes under the enabled policy; no GPU fake is used.
    operand = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(operand @ operand.T, operand @ operand.T)


def test_default_torch_controls_preserve_inherited_flags(torch_numerics_state):
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    configure_deterministic_evaluation_torch(torch, False)
    report = runtime_numerics(torch, deterministic_evaluation_requested=False)
    assert not report["deterministic_evaluation_requested"]
    assert not report["torch_deterministic_algorithms"] and report["torch_deterministic_warn_only"]
    assert report["cudnn"]["benchmark"] and not report["cudnn"]["deterministic"]


def test_real_main_rejects_bad_workspace_before_output_or_sdk(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/train_paper_amp.py"
    spec = importlib.util.spec_from_file_location("paper_amp_environment_guard_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unapproved")
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(deterministic_evaluation=True))
    before_sdk = {name for name in sys.modules if name.startswith(("omni", "isaacsim", "isaaclab"))}
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        runner.main()
    after_sdk = {name for name in sys.modules if name.startswith(("omni", "isaacsim", "isaaclab"))}
    assert after_sdk == before_sdk
