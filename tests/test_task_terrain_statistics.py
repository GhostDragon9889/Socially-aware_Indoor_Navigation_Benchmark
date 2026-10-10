"""Actual moment updates and strict saved policy admission, without SDK."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.training.trainer import RunningNormalizer

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / "CrowdSim/humanoid/training/terrain_statistics.py"
spec = importlib.util.spec_from_file_location("candidate_terrain_statistics", MODULE)
policy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_module)
Policy = policy_module.TerrainStatisticsPolicy


def setup_policy():
    normalizer = RunningNormalizer(1044)
    normalizer.mean.copy_(torch.linspace(-.7, .2, 1044, dtype=torch.float64))
    normalizer.variance.copy_(torch.linspace(.5, 3., 1044, dtype=torch.float64))
    normalizer.count.fill_(200000.)
    initialization = {"mode": "transfer_warm_start_public_pacer_actor_critic", "path": "/known/public/Humanoid.pth",
                      "sha256": policy_module.PUBLIC_PACER_SHA256, "actor_critic_task_transferred": True,
                      "observation_value_moments_transferred": True, "normalizer_count_cap": 200000.,
                      "normalizer_counts_before_cap": {"task_normalizer": policy_module.PUBLIC_REFERENCE_COUNT}}
    config = SimpleNamespace(task_terrain_statistics_mode="preserve_declared_suffix", normalize_observations=True,
                             normalization_clip=5., task_dim=1044, self_dim=357, warm_start_normalizer_count_cap=200000.)
    policy = Policy(config.task_terrain_statistics_mode)
    policy.bind_public_warm_start(normalizer, initialization=initialization,
                                 reference_count=policy_module.PUBLIC_REFERENCE_COUNT, warm_start_task_count=200000.)
    return policy, normalizer, initialization, config


def batch():
    values = torch.zeros(32, 1044)
    values[:, :20] = torch.linspace(-.6, 1.3, 32)[:, None]
    return values


def restore(state, normalizer, initialization, config, *, updates=1, transitions=32):
    return Policy.from_checkpoint(state, config=config, initialization=initialization,
                                  saved_task_normalizer=normalizer.state_dict(), update_count=updates,
                                  physical_transition_count=transitions)


def test_legacy_four_updates_and_save_schema_exact_to_original():
    _, original, _, _ = setup_policy()
    legacy = deepcopy(original)
    policy = Policy()
    for _ in range(4):
        original.update(batch())
        legacy.update(batch())
        assert policy.after_task_update(legacy) is None
    assert policy.state_dict() is None
    for key, value in original.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key])
    old_config = SimpleNamespace()
    restored = Policy.from_checkpoint(None, config=old_config, initialization={},
                                      saved_task_normalizer={}, update_count=4, physical_transition_count=128)
    assert restored.mode == "legacy"
    with pytest.raises(ValueError, match="Legacy"):
        Policy.from_checkpoint({}, config=old_config, initialization={}, saved_task_normalizer={},
                               update_count=4, physical_transition_count=128)


def test_only_suffix_is_protected_prefix_and_actual_count_are_original_math():
    policy, normalizer, _, _ = setup_policy()
    unchanged = deepcopy(normalizer)
    control = deepcopy(normalizer)
    for i in range(4):
        normalizer.update(batch())
        control.update(batch())
        count_before_hook = normalizer.count.clone()
        policy.after_task_update(normalizer)
        assert torch.equal(normalizer.count, count_before_hook)
        assert torch.equal(normalizer.mean[:20], control.mean[:20])
        assert torch.equal(normalizer.variance[:20], control.variance[:20])
        assert torch.equal(normalizer.mean[20:], unchanged.mean[20:])
        assert torch.equal(normalizer.variance[20:], unchanged.variance[20:])
        policy.validate_checkpoint(normalizer, update_count=i+1, physical_transition_count=(i+1)*32)
    assert normalizer.count == control.count == 200128.
    assert not torch.equal(control.variance[20:], unchanged.variance[20:])
    assert policy.describe()["reference_count"] == 61931520001.


def test_real_restricted_serialization_resume_next_moment_update_no_external_file(tmp_path):
    policy, norm, initialization, config = setup_policy()
    norm.update(batch())
    policy.after_task_update(norm)
    path = tmp_path / "policy.pt"
    torch.save({"normalizer": norm.state_dict(), "policy": policy.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    resumed = RunningNormalizer(1044)
    resumed.load_state_dict(saved["normalizer"])
    other = restore(saved["policy"], resumed, initialization, config)
    for current, hook in ((norm, policy), (resumed, other)):
        current.update(batch())
        hook.after_task_update(current)
        hook.validate_checkpoint(current, update_count=2, physical_transition_count=64)
    assert other.state_dict()["post_update_calls"] == 2
    for key, value in norm.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    assert other.describe()["external_statistics_file_required"] is False


def test_binding_and_state_copies_do_not_consume_any_CPU_rng():
    python_before, numpy_before = random.getstate(), np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy, norm, initialization, config = setup_policy()
    saved = policy.state_dict()
    saved["suffix_mean"].fill_(10)
    saved["reference_initialization"]["path"] = "tampered"
    assert policy.describe()["reference_initialization"]["path"] != "tampered"
    with pytest.raises(ValueError):
        restore(saved, norm, initialization, config, updates=0, transitions=0)
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


@pytest.mark.parametrize("mode", [None, False, "unknown"])
def test_unknown_or_non_string_mode_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        Policy(mode)


def test_unbound_and_repeated_warm_start_rejected_before_overwrite():
    policy = Policy("preserve_declared_suffix")
    for action in (policy.require_ready, policy.state_dict):
        with pytest.raises(ValueError, match="unbound"):
            action()
    bound, norm, _, _ = setup_policy()
    before = deepcopy(norm.state_dict())
    with pytest.raises(ValueError, match="cannot be repeated"):
        bound.require_fresh_warm_start(0)
    assert all(torch.equal(value, before[key]) for key, value in norm.state_dict().items())


@pytest.mark.parametrize("case", ["cap", "original_count", "source", "bool_cap"])
def test_fresh_bind_cannot_accept_inconsistent_public_count_or_source_provenance(case):
    _, norm, initialization, _ = setup_policy()
    if case == "cap":
        initialization["normalizer_count_cap"] = 200001.
    elif case == "original_count":
        initialization["normalizer_counts_before_cap"]["task_normalizer"] = 200000.
    elif case == "source":
        initialization["sha256"] = "0"*64
    else:
        initialization["normalizer_count_cap"] = True
    policy = Policy("preserve_declared_suffix")
    before = deepcopy(norm.state_dict())
    with pytest.raises(ValueError):
        policy.bind_public_warm_start(norm, initialization=initialization,
                                     reference_count=policy_module.PUBLIC_REFERENCE_COUNT, warm_start_task_count=200000.)
    assert not policy.bound
    assert all(torch.equal(value, before[key]) for key, value in norm.state_dict().items())


@pytest.mark.parametrize("case", ["missing", "extra", "unknown_schema", "slice_bool", "negative_var", "NaN", "dtype", "shape",
                                  "changed_array", "source", "ref_count", "warm_count", "cap", "original_count", "calls_bool",
                                  "calls_low", "calls_high", "normalizer_suffix", "global_count", "float_updates", "missing_normalization"])
def test_corrupted_checkpoint_policy_or_mutual_contract_rejected(case):
    policy, norm, initialization, config = setup_policy()
    norm.update(batch())
    policy.after_task_update(norm)
    saved = policy.state_dict()
    updates = 1
    if case == "missing":
        saved = None
    elif case == "extra":
        saved["future"] = 1
    elif case == "unknown_schema":
        saved["schema"] = "other"
    elif case == "slice_bool":
        saved["task_slice"] = [False, 1044]
    elif case == "negative_var":
        saved["suffix_variance"][0] = -1
    elif case == "NaN":
        saved["suffix_mean"][0] = float("nan")
    elif case == "dtype":
        saved["suffix_mean"] = saved["suffix_mean"].float()
    elif case == "shape":
        saved["suffix_mean"] = saved["suffix_mean"][:-1]
    elif case == "changed_array":
        saved["suffix_variance"][0] += .01
    elif case == "source":
        saved["reference_initialization"]["sha256"] = "0"*64
    elif case == "ref_count":
        saved["reference_count"] = 200000.
    elif case == "warm_count":
        saved["warm_start_task_count"] += 1
    elif case == "cap":
        config.warm_start_normalizer_count_cap += 1
    elif case == "original_count":
        initialization["normalizer_counts_before_cap"]["task_normalizer"] = 200000.
    elif case == "calls_bool":
        saved["post_update_calls"] = True
    elif case == "calls_low":
        saved["post_update_calls"] = 0
    elif case == "calls_high":
        saved["post_update_calls"] = 2
    elif case == "normalizer_suffix":
        norm.mean[20] += .001
    elif case == "global_count":
        norm.count += 1
    elif case == "float_updates":
        updates = 1.
    else:
        config.normalize_observations = False
    before = deepcopy(norm.state_dict())
    with pytest.raises(ValueError):
        restore(saved, norm, initialization, config, updates=updates)
    assert all(torch.equal(value, before[key]) for key, value in norm.state_dict().items())
