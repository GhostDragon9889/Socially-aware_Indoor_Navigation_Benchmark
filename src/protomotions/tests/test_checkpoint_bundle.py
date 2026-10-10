# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fault and restore tests for complete model/environment generations."""

import os
from datetime import timedelta
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from protomotions.agents.base_agent import agent as agent_module
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils import checkpoint as checkpoint_utils
from protomotions.utils.checkpoint import (
    CHECKPOINT_BUNDLE_KEY,
    atomic_torch_save,
    atomic_write,
    get_checkpoint_bundle,
    make_checkpoint_bundle,
    snapshot_environment_state,
)


def _state(generation, task="clip.pt"):
    return {
        "model": {"weight": torch.tensor([generation], dtype=torch.float)},
        "optimizer": {"momentum": generation},
        CHECKPOINT_BUNDLE_KEY: make_checkpoint_bundle({
            task: {"motion_manager": {"motion_weights": torch.tensor([generation], dtype=torch.float)}}
        }),
    }


def _read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _bare_agent(root_dir, generation=1, task="clip.pt"):
    agent = object.__new__(BaseAgent)
    agent.device = torch.device("cpu")
    agent.root_dir = root_dir
    agent.fabric = SimpleNamespace(world_size=1, global_rank=0, call=lambda *args: None)
    agent.config = SimpleNamespace(save_inference_checkpoint=True)
    agent.best_evaluated_score = 1.0
    agent.generation = generation
    agent.env = SimpleNamespace(
        get_task_id=lambda: task,
        get_state_dict=lambda: {
            "motion_manager": {"motion_weights": torch.tensor([agent.generation], dtype=torch.float)}
        },
        load_state_dict=lambda state: setattr(agent, "loaded_env", state),
    )
    agent.get_state_dict = lambda state: state | {
        "model": {"weight": torch.tensor([agent.generation], dtype=torch.float)},
        "optimizer": {"momentum": agent.generation},
    }
    agent.get_inference_state_dict = lambda state, model_state_dict: state | {"model": model_state_dict}
    agent.load_parameters = lambda state, load_training_state: setattr(agent, "loaded_agent", state)
    return agent


def _local_collectives(monkeypatch):
    monkeypatch.setattr(agent_module.dist, "all_gather_object", lambda target, value: target.__setitem__(0, value))
    monkeypatch.setattr(agent_module.dist, "all_gather", lambda target, value: target[0].copy_(value))
    monkeypatch.setattr(agent_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(agent_module.dist, "barrier", lambda: None)


def test_partial_serialization_preserves_old_checkpoint_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "last.ckpt"
    atomic_torch_save(_state(1), path)
    old_bytes = path.read_bytes()

    def broken_save(state, handle):
        handle.write(b"incomplete archive")
        raise OSError("disk full during serialization")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save(_state(2), path, preserve_previous=True)
    assert path.read_bytes() == old_bytes
    assert not list(tmp_path.glob("*.tmp"))
    assert _read(path)["optimizer"]["momentum"] == 1


def test_failed_publication_preserves_complete_current_and_previous(tmp_path, monkeypatch):
    path = tmp_path / "last.ckpt"
    atomic_torch_save(_state(1), path)
    replace = os.replace

    def fail_final_replace(source, destination):
        if Path(destination) == path:
            raise OSError("rename rejected")
        replace(source, destination)

    monkeypatch.setattr(checkpoint_utils.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="rename rejected"):
        atomic_torch_save(_state(2), path, preserve_previous=True)
    assert _read(path)["optimizer"]["momentum"] == 1
    assert _read(tmp_path / "last.ckpt.previous")["optimizer"]["momentum"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_previous_checkpoint_remains_a_complete_older_generation(tmp_path):
    path = tmp_path / "last.ckpt"
    first = _state(1)
    atomic_torch_save(first, path)
    atomic_torch_save(_state(2), path, preserve_previous=True)
    old = _read(tmp_path / "last.ckpt.previous")
    assert old["optimizer"]["momentum"] == 1
    assert get_checkpoint_bundle(old)["generation_id"] == get_checkpoint_bundle(first)["generation_id"]
    assert get_checkpoint_bundle(_read(path))["generation_id"] != get_checkpoint_bundle(old)["generation_id"]


def test_process_killed_during_write_cannot_truncate_published_file(tmp_path):
    path = tmp_path / "published.bin"
    atomic_write(path, lambda handle: handle.write(b"old complete snapshot"))
    script = """
import sys
from protomotions.utils.checkpoint import atomic_write
def writer(handle):
    handle.write(b'partial replacement')
    handle.flush()
    print('writer-entered', flush=True)
    sys.stdin.read()
atomic_write(sys.argv[1], writer, preserve_previous=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "writer-entered"
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode < 0
        assert path.read_bytes() == b"old complete snapshot"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_save_and_selected_score_resume_never_load_a_later_environment(tmp_path, monkeypatch):
    _local_collectives(monkeypatch)
    agent = _bare_agent(tmp_path)
    BaseAgent.save(agent, new_high_score=True)
    agent.generation = 2
    BaseAgent.save(agent)
    # A legacy sidecar deliberately disagrees with both complete generations.
    torch.save({"motion_manager": {"motion_weights": torch.tensor([999.0])}}, tmp_path / "env_clip.pt.ckpt")

    for filename, expected in (
        ("score_based.ckpt", 1), ("inference_score_based.ckpt", 1),
        ("last.ckpt", 2), ("inference_last.ckpt", 2), ("last.ckpt.previous", 1),
    ):
        # A checkpoint can be restored into a different output directory.
        agent.root_dir = tmp_path / "new-run"
        BaseAgent.load(agent, tmp_path / filename)
        assert agent.loaded_agent["model"]["weight"].item() == expected
        assert agent.loaded_env["motion_manager"]["motion_weights"].item() == expected
    last = _read(tmp_path / "last.ckpt")
    inference = _read(tmp_path / "inference_last.ckpt")
    assert get_checkpoint_bundle(last)["generation_id"] == get_checkpoint_bundle(inference)["generation_id"]
    assert last["optimizer"]["momentum"] == 2
    assert "optimizer" not in inference


def test_missing_new_task_does_not_fall_back_to_stale_sidecar(tmp_path):
    atomic_torch_save(_state(1, "different.pt"), tmp_path / "last.ckpt")
    torch.save({"wrong": True}, tmp_path / "env_clip.pt.ckpt")
    agent = _bare_agent(tmp_path)
    BaseAgent.load(agent, tmp_path / "last.ckpt")
    assert not hasattr(agent, "loaded_env")


def test_inactive_motion_shard_curriculum_survives_later_generations(tmp_path, monkeypatch):
    _local_collectives(monkeypatch)
    agent = _bare_agent(tmp_path, generation=1, task="A.pt")
    BaseAgent.save(agent)
    agent.generation = 2
    agent.env.get_task_id = lambda: "B.pt"
    assert not BaseAgent._load_environment_checkpoint(agent)
    BaseAgent.save(agent)
    agent.env.get_task_id = lambda: "A.pt"
    assert BaseAgent._load_environment_checkpoint(agent)
    assert agent.loaded_env["motion_manager"]["motion_weights"].item() == 1
    bundle = get_checkpoint_bundle(_read(tmp_path / "last.ckpt"))
    assert set(bundle["environment_states"]) == {"A.pt", "B.pt"}
    assert bundle["environment_states"]["B.pt"]["motion_manager"]["motion_weights"].item() == 2


def test_invalid_bundle_rejected_before_model_or_environment_mutation(tmp_path):
    state = _state(1)
    state[CHECKPOINT_BUNDLE_KEY]["version"] = 999
    torch.save(state, tmp_path / "last.ckpt")
    agent = _bare_agent(tmp_path)
    with pytest.raises(ValueError, match="version"):
        BaseAgent.load(agent, tmp_path / "last.ckpt")
    assert not hasattr(agent, "loaded_agent")
    assert not hasattr(agent, "loaded_env")


def test_legacy_sidecar_is_relative_to_selected_checkpoint(tmp_path, caplog):
    original = tmp_path / "original"
    original.mkdir()
    torch.save({"model": {}}, original / "score_based.ckpt")
    torch.save({"correct": True}, original / "env_clip.pt.ckpt")
    new = tmp_path / "new"
    new.mkdir()
    torch.save({"wrong": True}, new / "env_clip.pt.ckpt")
    agent = _bare_agent(new)
    BaseAgent.load(agent, original / "score_based.ckpt")
    assert agent.loaded_env == {"correct": True}
    assert "generation cannot be verified" in caplog.text


def test_scene_curriculum_uses_complete_generation_not_legacy_sidecar(tmp_path):
    atomic_torch_save(_state(2), tmp_path / "last.ckpt")
    torch.save({"motion_manager": {"motion_weights": torch.tensor([999.0])}}, tmp_path / "env_clip.pt.ckpt")
    # Even a future CUDA scene uses CPU-only, lazily mapped checkpoint tensors;
    # this regression intentionally runs on a machine without a CUDA context.
    assert BaseEnv.apply_motion_weights_to_scene_weights(str(tmp_path), "/motions/clip.pt", torch.device("cuda")) == [2.0]
    assert BaseEnv.apply_motion_weights_to_scene_weights(str(tmp_path), "/motions/other.pt", torch.device("cpu")) is None


def test_environment_snapshot_does_not_alias_live_buffers():
    weights = torch.tensor([1.0], requires_grad=True)
    state = snapshot_environment_state({"nested": [weights, (weights,)]})
    with torch.no_grad():
        weights.fill_(2.0)
    assert state["nested"][0].item() == 1.0
    assert state["nested"][1][0].item() == 1.0
    assert not state["nested"][0].requires_grad
    assert state["nested"][0].device.type == "cpu"


def _distributed_save_worker(rank, root, rendezvous, fail_environment):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        agent = _bare_agent(Path(root), generation=7, task=f"task-{rank}")
        agent.fabric.world_size = 2
        agent.fabric.global_rank = rank
        agent.config.save_inference_checkpoint = False
        if rank == 1 and fail_environment:
            def fail():
                raise OSError("rank-1 environment snapshot failed")
            agent.env.get_state_dict = fail
        try:
            BaseAgent.save(agent)
        except RuntimeError as error:
            if not fail_environment or "Environment checkpoint save failed" not in str(error):
                raise
            (Path(root) / f"rank-{rank}-failed").write_text(str(error))
        else:
            assert not fail_environment
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_environment", [False, True])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs CPU Gloo distributed backend")
def test_real_cpu_collectives_save_all_tasks_and_coordinate_remote_failure(tmp_path, fail_environment):
    mp.spawn(_distributed_save_worker, args=(str(tmp_path), str(tmp_path / "rendezvous"), fail_environment), nprocs=2, join=True)
    path = tmp_path / "last.ckpt"
    if fail_environment:
        assert not path.exists()
        assert all((tmp_path / f"rank-{rank}-failed").is_file() for rank in range(2))
    else:
        bundle = get_checkpoint_bundle(_read(path))
        assert set(bundle["environment_states"]) == {"task-0", "task-1"}
        assert all(state["motion_manager"]["motion_weights"].item() == 7 for state in bundle["environment_states"].values())
