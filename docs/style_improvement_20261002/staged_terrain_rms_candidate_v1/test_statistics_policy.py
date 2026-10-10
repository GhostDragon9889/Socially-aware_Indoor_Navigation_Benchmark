"""Real Torch moment-update and source/resume/RNG admission regression."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.training.trainer import RunningNormalizer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("staged_terrain_policy", HERE/"statistics_policy.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    mean = torch.linspace(-.7, .2, 1044, dtype=torch.float64)
    variance = torch.linspace(.5, 3., 1044, dtype=torch.float64)
    p = tmp_path/"reference.npz"
    np.savez(p, task_mean=mean.numpy(), task_variance=variance.numpy(), reference_count=np.array(61931520001., dtype=np.float64))
    norm = RunningNormalizer(1044)
    norm.mean.copy_(mean)
    norm.variance.copy_(variance)
    norm.count.fill_(200000.)
    return norm, module.file_record(p)


def observations():
    x = torch.zeros(32, 1044)
    x[:, :20] = torch.arange(32)[:, None]/30
    return x


def test_default_legacy_and_real_update_remain_byte_exact(tmp_path):
    norm, _ = fixture(tmp_path)
    control = deepcopy(norm)
    policy = module.StagedTerrainRMSPolicy()
    assert policy.attach(norm) is None
    for _ in range(4):
        norm.update(observations())
        control.update(observations())
        assert policy.after_update(norm) is None
    for k, value in norm.state_dict().items():
        assert torch.equal(value, control.state_dict()[k])
    assert policy.state_dict() is None
    policy.load_state_dict(None, norm)


def test_protection_keeps_reference_suffix_only_and_real_count_prefix_update(tmp_path):
    norm, source = fixture(tmp_path)
    control = deepcopy(norm)
    original = deepcopy(norm)
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    for _ in range(4):
        norm.update(observations())
        control.update(observations())
        post_prefix = norm.mean[:20].clone(), norm.variance[:20].clone(), norm.count.clone()
        policy.after_update(norm)
        assert torch.equal(norm.mean[:20], post_prefix[0]) and torch.equal(norm.variance[:20], post_prefix[1])
        assert torch.equal(norm.count, post_prefix[2])
        assert torch.equal(norm.mean[20:], original.mean[20:]) and torch.equal(norm.variance[20:], original.variance[20:])
    assert not torch.equal(control.variance[20:], original.variance[20:])
    assert torch.equal(norm.mean[:20], control.mean[:20]) and torch.equal(norm.variance[:20], control.variance[:20])
    assert norm.count == control.count == 200128.
    assert policy.calls == 4
    assert policy.describe()["reference_statistics_count_not_global_task_count"] == 61931520001.


def test_real_torch_serialization_resumes_next_statistics_update(tmp_path):
    norm, source = fixture(tmp_path)
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    norm.update(observations())
    policy.after_update(norm)
    p = tmp_path/"resume.pt"
    torch.save({"normalizer": norm.state_dict(), "policy": policy.state_dict()}, p)
    saved = torch.load(p, weights_only=True, map_location="cpu")
    resumed = RunningNormalizer(1044)
    resumed.load_state_dict(saved["normalizer"])
    other = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    other.load_state_dict(saved["policy"], resumed, expected_update_count=1)
    for n, hook in ((norm, policy), (resumed, other)):
        n.update(observations())
        hook.after_update(n)
    assert other.calls == policy.calls == 2
    for key, value in norm.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_no_torch_RNG_draw_and_saved_state_cannot_mutate_live_source(tmp_path):
    norm, source = fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    norm.update(observations())
    policy.after_update(norm)
    saved = policy.state_dict()
    saved["reference_record"]["sha256"] = "0"*64
    saved["suffix_mean"].fill_(0)
    assert policy.reference_record["sha256"] == source["sha256"]
    assert not torch.equal(policy.suffix_mean, saved["suffix_mean"])
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="schema/source"):
        policy.load_state_dict(saved, norm, expected_update_count=1)


@pytest.mark.parametrize("case", ["no_source", "not_fresh", "changed_file", "bad_shape", "NaN", "negative_variance"])
def test_invalid_declared_stat_sources_fail_closed(tmp_path, case):
    _, source = fixture(tmp_path)
    fresh = True
    if case == "no_source":
        source = None
    elif case == "not_fresh":
        fresh = False
    elif case == "changed_file":
        Path(source["path"]).write_bytes(b"different")
    else:
        mean = np.zeros(1044, dtype=np.float64)
        variance = np.ones(1044, dtype=np.float64)
        if case == "bad_shape":
            mean = mean[:-1]
        elif case == "NaN":
            mean[50] = np.nan
        else:
            variance[50] = -1
        np.savez(source["path"], task_mean=mean, task_variance=variance, reference_count=np.array(61931520001., dtype=np.float64))
        source = module.file_record(source["path"])
    with pytest.raises(ValueError):
        module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=fresh)


@pytest.mark.parametrize("case", ["modified_moments", "bool_counter", "extra_key", "wrong_mode", "checkpoint_suffix_changed"])
def test_restore_rejects_changed_source_state_and_normalizer(tmp_path, case):
    norm, source = fixture(tmp_path)
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    state = policy.state_dict()
    if case == "modified_moments":
        state["suffix_variance"][0] += .001
    elif case == "bool_counter":
        state["calls"] = False
    elif case == "extra_key":
        state["future"] = 1
    elif case == "wrong_mode":
        state["mode"] = "legacy"
    else:
        norm.mean[20] += .001
    with pytest.raises(ValueError):
        policy.load_state_dict(state, norm, expected_update_count=0)


def test_no_silent_conversion_of_trained_suffix_and_legacy_state(tmp_path):
    norm, source = fixture(tmp_path)
    norm.update(observations())
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    with pytest.raises(ValueError, match="no silent overwrite"):
        policy.attach(norm)
    with pytest.raises(ValueError, match="silently restore"):
        module.StagedTerrainRMSPolicy().load_state_dict({}, norm)


@pytest.mark.parametrize("claimed_count", [None, False, 0, 2])
def test_resume_cannot_accept_missing_or_changed_completed_trainer_updates(tmp_path, claimed_count):
    norm, source = fixture(tmp_path)
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    norm.update(observations())
    policy.after_update(norm)
    policy.validate_checkpoint(norm, expected_update_count=1)
    with pytest.raises(ValueError, match="counters changed"):
        policy.load_state_dict(policy.state_dict(), norm, expected_update_count=claimed_count)
    with pytest.raises(ValueError, match="update_count"):
        policy.validate_checkpoint(norm, expected_update_count=claimed_count)


def test_success_and_failed_resume_preserve_all_CPU_caller_rng(tmp_path):
    norm, source = fixture(tmp_path)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy = module.StagedTerrainRMSPolicy("preserve_declared_suffix", reference_record=source, fresh_experiment=True)
    policy.attach(norm)
    norm.update(observations())
    policy.after_update(norm)
    state = policy.state_dict()
    state["reference_count"] += 1
    with pytest.raises(ValueError, match="counters changed"):
        policy.load_state_dict(state, norm, expected_update_count=1)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
