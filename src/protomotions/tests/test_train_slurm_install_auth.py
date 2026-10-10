# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise generated installation gates using fake pip, without SLURM/Kit."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest

from protomotions import train_slurm as slurm


def _args(wandb=False):
    args = slurm.create_parser().parse_args([
        "--robot-name=g1", "--simulator=newton", "--num-envs=2",
        "--batch-size=4", "--motion-file=motions.pt", "--experiment-path=exp.py",
        "--experiment-name=unit", "--user=unit",
    ])
    args.use_wandb = wandb
    for name in ("array_size", "startup_max_attempts", "startup_stall_min", "startup_repeat_limit", "startup_retry_backoff_sec"):
        if getattr(args, name) is None:
            setattr(args, name, slurm.SITE_DEFAULTS[name])
    return args


def _bash(body, gate, rank):
    env = dict(os.environ)
    env.update(PROTOMOTIONS_INSTALL_GATE_DIR=str(gate), SLURM_LOCALID=str(rank))
    return subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True, timeout=15)


def test_install_failure_is_not_ready_and_all_ranks_exit_with_same_failure(tmp_path):
    gate = tmp_path / "gate"
    body = slurm._installation_gate_bash("exit 42", timeout_seconds=1, poll_seconds=1)
    for rank in (0, 1):
        result = _bash(body + "echo training-started", gate, rank)
        assert result.returncode == 42, result.stderr
        assert "training-started" not in result.stdout
    assert (gate / "failed").read_text().strip() == "42"
    assert not (gate / "ready").exists()


def test_successful_install_releases_waiting_rank(tmp_path):
    gate = tmp_path / "gate"
    body = slurm._installation_gate_bash("true", timeout_seconds=1, poll_seconds=1)
    for rank in (0, 1):
        result = _bash(body + "echo training-started", gate, rank)
        assert result.returncode == 0, result.stderr
        assert "training-started" in result.stdout
    assert (gate / "ready").is_file()
    assert not (gate / "failed").exists()


def test_waiting_rank_has_bounded_failure_when_leader_never_starts(tmp_path):
    result = _bash(slurm._installation_gate_bash("true", timeout_seconds=1, poll_seconds=1), tmp_path / "missing", 1)
    assert result.returncode == 1
    assert "Timed out" in result.stdout


def test_full_generated_job_never_starts_training_after_failed_pip(tmp_path):
    # The directory includes spaces to exercise the actual generated cd/log paths.
    exp = tmp_path / "experiment folder"
    exp.mkdir()
    executables = tmp_path / "bin"
    executables.mkdir()
    fake_pip = executables / "pip"
    fake_pip.write_text('#!/bin/bash\nif [ "$1" = uninstall ]; then exit 0; fi\nexit 42\n')
    fake_pip.chmod(0o700)
    marker = tmp_path / "training-started"
    fake_python = executables / "fake-python"
    fake_python.write_text(f"#!/bin/bash\ntouch {shlex.quote(str(marker))}\n")
    fake_python.chmod(0o700)
    body = slurm.build_job_command(_args(), str(exp), shlex.quote(str(fake_python)))
    env = dict(os.environ)
    env.update(PATH=f"{executables}:{env['PATH']}", PROTOMOTIONS_INSTALL_GATE_DIR=str(tmp_path / "gate"), SLURM_LOCALID="0")
    result = subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 42, result.stderr
    assert not marker.exists()
    assert not (tmp_path / "gate" / "ready").exists()


def test_generation_does_not_read_or_embed_local_wandb_key(tmp_path, monkeypatch):
    fake_key = "FAKE_AUDIT_API_KEY_NO_REAL_CREDENTIAL"
    monkeypatch.setenv("WANDB_API_KEY", fake_key)

    def refuse_network(*args, **kwargs):
        pytest.fail("credential generation must not invoke SSH or a credential probe")

    monkeypatch.setattr(slurm, "subprocess_run", refuse_network)
    args = _args(wandb=True)
    command = slurm.build_job_command(args, str(tmp_path), "python")
    script, _ = slurm.generate_slurm_script(args, str(tmp_path), command, "/containers/test.sqsh", "unit")
    assert fake_key not in command and fake_key not in script
    assert "export WANDB_API_KEY=" not in script
    assert "${WANDB_API_KEY:+configured}" in script
    assert 'if [ -f "${HOME}/.netrc" ]' in script
    assert "${HOME}/.netrc:/root/.netrc:ro" in script
    # Parse only: never touch a real .netrc or run the generated job.
    result = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_install_attempt_markers_do_not_reuse_a_previous_slurm_step(tmp_path):
    # Execute only the gate with a fake install; the marker includes the actual
    # job/step/restart/node tuple, rather than assuming an old ready is current.
    job = f"audit-{tmp_path.name}"
    env = dict(os.environ)
    env.pop("PROTOMOTIONS_INSTALL_GATE_DIR", None)
    env.update(SLURM_JOB_ID=job, SLURM_RESTART_COUNT="0", SLURMD_NODENAME="test", SLURM_LOCALID="0")
    gates = []
    try:
        for step, install, expected in (("0", "true", 0), ("1", "exit 42", 42)):
            env["SLURM_STEP_ID"] = step
            gate = Path(f"/tmp/protomotions_install_{job}_{step}_0_test")
            gates.append(gate)
            result = subprocess.run(["bash", "-c", slurm._installation_gate_bash(install)], env=env, capture_output=True, text=True)
            assert result.returncode == expected, result.stderr
        assert (gates[0] / "ready").exists()
        assert not (gates[1] / "ready").exists()
        assert (gates[1] / "failed").exists()
    finally:
        for gate in gates:
            for marker in gate.glob("*"):
                marker.unlink()
            gate.rmdir()
