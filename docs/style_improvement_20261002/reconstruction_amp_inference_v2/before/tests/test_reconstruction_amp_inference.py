"""Inference isolation, effective actions and corrupted real Torch bundles."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.amp.controller import action_to_pd
from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES
from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
from CrowdSim.humanoid.runtime.checkpoint_contract import (
    CANONICAL_ACTION_MASK,
    inspect_checkpoint_file,
    validate_reconstruction_config,
)
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig


def _rng():
    numpy = np.random.get_state()
    return random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), torch.get_rng_state().clone()


def _same_rng(before):
    python, numpy, tensor = _rng()
    assert python == before[0]
    assert numpy[0] == before[1][0] and numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(numpy[1], before[1][1])
    assert torch.equal(tensor, before[2])


def _inputs(batch=4):
    generator = torch.Generator(device="cpu").manual_seed(2718)
    return torch.randn(batch, 357, generator=generator), torch.randn(batch, 1044, generator=generator)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    # This is a synthetic unit fixture saved by the actual trainer, not a
    # replacement published model or a claim of physical training performance.
    trainer = PaperAMPTrainer(TrainingConfig(discriminator_replay_capacity=4), device="cpu")
    own, task = _inputs()
    generator = torch.Generator().manual_seed(17)
    trainer.self_normalizer.update(own)
    trainer.task_normalizer.update(task)
    trainer.amp_normalizer.update(torch.randn(4, 195, generator=generator))
    trainer.value_normalizer.update(torch.arange(4).float()[:, None])
    with torch.no_grad():
        trainer.model.actor[-1].bias[0] = 1.8
        trainer.model.actor[-1].bias[PACER_DOF_NAMES.index("L_Toe_x")] = 2.4
    path = tmp_path_factory.mktemp("reconstruction_inference") / "unit_training_bundle.pt"
    trainer.save(path)
    return path


def test_cold_header_preflight_imports_neither_torch_numpy_nor_sdk():
    root = Path(__file__).resolve().parents[1]
    code = """import sys
from CrowdSim.humanoid.runtime.checkpoint_contract import validate_reconstruction_config
config = validate_reconstruction_config({})
assert config.self_dim == 357 and sum(config.action_mask) == 57
assert not any(name.split('.')[0] in {'torch','numpy','isaacsim','isaaclab','omni'} for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_loader_and_repeated_inference_preserve_all_rng_and_normalizer_buffers(bundle):
    before = _rng()
    inference = ReconstructionAMPInference.from_checkpoint(bundle)
    _same_rng(before)
    state = {name: tensor.clone() for name, tensor in inference.state_dict().items()}
    expected = inference.act(*_inputs())
    for _ in range(3):
        actual = inference.act(*_inputs())
        for key in expected:
            assert torch.equal(actual[key], expected[key])
    _same_rng(before)
    for name, tensor in inference.state_dict().items():
        assert torch.equal(tensor, state[name])
    assert not inference.training and not inference.model.training
    assert not any(parameter.requires_grad for parameter in inference.parameters())
    assert torch.cuda.is_initialized() is False


def test_no_training_objects_created_and_unused_resume_payload_not_retained(bundle, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference constructed a training object")

    from CrowdSim.humanoid.training.replay import AMPPolicyReplay
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(AMPPolicyReplay, "__init__", forbidden)
    monkeypatch.setattr(PaperAMPTrainer, "__init__", forbidden)
    inference = ReconstructionAMPInference.from_checkpoint(bundle)
    assert set(dict(inference.named_children())) == {"model", "self_normalizer", "task_normalizer"}
    for name in ("optimizer", "discriminator_optimizer", "policy_replay", "rng", "runner_state",
                 "amp_normalizer", "value_normalizer", "expert_sampler"):
        assert not hasattr(inference, name)
    description = inference.describe()
    assert not description["optimizer_created"] and not description["replay_created"]
    description["config"]["control_dt"] = 4.
    assert inference.config.control_dt == 1 / 30


def test_effective_normalized_actions_match_real_trainer_and_single_pd_decode(bundle):
    before = _rng()
    with torch.random.fork_rng(devices=[]):
        trainer, _ = PaperAMPTrainer.load(bundle, device="cpu", restore_rng=False)
    random.setstate(before[0])
    np.random.set_state(before[1])
    inference = ReconstructionAMPInference.from_checkpoint(bundle)
    own, task = _inputs()
    expected = trainer.act(own, task, deterministic=True)
    actual = inference.act(own, task)
    for key in actual:
        assert torch.equal(actual[key], expected[key])
    frozen = [index for index, name in enumerate(PACER_DOF_NAMES)
              if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]
    assert sum(CANONICAL_ACTION_MASK) == 57
    assert tuple(CANONICAL_ACTION_MASK[index] for index in frozen) == (0,) * 12
    assert torch.count_nonzero(actual["actions"][:, frozen]) > 0
    assert torch.count_nonzero(actual["applied_actions"][:, frozen]) == 0
    assert actual["actions"][0, 0] > 1 and actual["applied_actions"][0, 0] == 1
    lower, upper = -torch.ones(69), torch.ones(69)
    torch.testing.assert_close(action_to_pd(actual["applied_actions"], lower, upper),
                               action_to_pd(actual["actions"], lower, upper), atol=0, rtol=0)
    assert not torch.equal(action_to_pd(actual["applied_actions"], lower, upper), actual["applied_actions"])
    _same_rng(before)


@pytest.mark.parametrize("change", ["extra_key", "missing_key", "shape", "dtype", "nan_actor", "nan_discriminator",
                                  "missing_normalizer", "negative_variance", "negative_count", "nan_unused_amp",
                                  "normalizer_dtype", "invalid_gaussian_scale"])
def test_corrupted_saved_tensor_bundles_fail_without_mutating_rng(bundle, tmp_path, change):
    artifact = torch.load(bundle, weights_only=True, map_location="cpu", mmap=True)
    model = artifact["model"]
    name = "actor.0.weight"
    if change == "extra_key":
        model["unrecognized.weight"] = torch.ones(1)
    elif change == "missing_key":
        del model[name]
    elif change == "shape":
        model[name] = model[name][:1].clone()
    elif change == "dtype":
        model[name] = model[name].double()
    elif change.startswith("nan_") and change != "nan_unused_amp":
        name = "discriminator.0.weight" if change == "nan_discriminator" else name
        model[name] = model[name].clone()
        model[name][0, 0] = float("nan")
    elif change == "missing_normalizer":
        del artifact["normalizers"]["value_normalizer"]
    elif change == "negative_variance":
        artifact["normalizers"]["task_normalizer"]["variance"] = -torch.ones(1044, dtype=torch.float64)
    elif change == "negative_count":
        artifact["normalizers"]["self_normalizer"]["count"] = torch.tensor(-1., dtype=torch.float64)
    elif change == "nan_unused_amp":
        artifact["normalizers"]["amp_normalizer"]["mean"] = torch.full((195,), float("nan"), dtype=torch.float64)
    elif change == "normalizer_dtype":
        artifact["normalizers"]["self_normalizer"]["count"] = torch.tensor(4., dtype=torch.float32)
    elif change == "invalid_gaussian_scale":
        model["log_std"] = torch.full((69,), 1000.)
    else:
        raise AssertionError(change)
    path = tmp_path / f"corrupt_{change}.pt"
    torch.save(artifact, path)
    before = _rng()
    with pytest.raises(ValueError):
        ReconstructionAMPInference.from_checkpoint(path)
    _same_rng(before)


@pytest.mark.parametrize("override", [
    {"self_dim": 356}, {"task_dim": 1043}, {"history_frames": 11}, {"control_dt": 1 / 60},
    {"target_horizon_s": 4.}, {"normalization_clip": 6.}, {"freeze_toes": False},
    {"action_mask": (1,) * 69}, {"action_mask": tuple(bool(value) for value in CANONICAL_ACTION_MASK)},
    {"policy_layers": (1000000000, 2)}, {"unknown_field": True},
])
def test_unsupported_feature_physics_or_mask_contract_fails_before_network_allocation(override, monkeypatch):
    import CrowdSim.humanoid.runtime.amp_inference as module
    monkeypatch.setattr(module, "PaperAMPNetwork", lambda *args: pytest.fail("Invalid config reached model allocation"))
    with pytest.raises(ValueError):
        validate_reconstruction_config({**asdict(TrainingConfig()), **override})


@pytest.mark.parametrize("field,value", [("format", "public_pacer"), ("version", True), ("version", 2),
                                       ("update_count", -1), ("physical_transition_count", True)])
def test_wrong_checkpoint_identity_and_counts_are_rejected(bundle, tmp_path, field, value):
    artifact = torch.load(bundle, map_location="cpu", weights_only=True, mmap=True)
    artifact[field] = value
    path = tmp_path / "wrong_schema.pt"
    torch.save(artifact, path)
    before = _rng()
    with pytest.raises(ValueError):
        ReconstructionAMPInference.from_checkpoint(path)
    _same_rng(before)


def test_hash_gate_precedes_deserialization_and_local_pointer_header_fails(bundle, tmp_path, monkeypatch):
    before = _rng()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Wrong hash reached deserialization"))
    with pytest.raises(ValueError, match="SHA256"):
        ReconstructionAMPInference.from_checkpoint(bundle, expected_sha256="0" * 64)
    _same_rng(before)
    for content in (b"version https://git-lfs.github.com/spec/v1\n", b"not a checkpoint"):
        path = tmp_path / "pointer.pt"
        path.write_bytes(content)
        with pytest.raises(ValueError):
            inspect_checkpoint_file(path)


def test_load_uses_restricted_cpu_mmap_and_refuses_atomic_file_replacement(bundle, tmp_path, monkeypatch):
    path = tmp_path / "replaceable.pt"
    shutil.copyfile(bundle, path)
    original = torch.load
    observed = []

    def replaced(*args, **kwargs):
        observed.append(kwargs)
        result = original(*args, **kwargs)
        new = path.with_suffix(".new")
        new.write_bytes(b"PK\x03\x04changed")
        new.replace(path)
        return result

    monkeypatch.setattr(torch, "load", replaced)
    before = _rng()
    with pytest.raises(RuntimeError, match="changed during"):
        ReconstructionAMPInference.from_checkpoint(path)
    assert observed == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    _same_rng(before)


def test_input_contract_and_stochastic_inference_rejected(bundle):
    inference = ReconstructionAMPInference.from_checkpoint(bundle)
    own, task = _inputs()
    for self_obs, task_obs in ((own.double(), task), (own[:0], task[:0]), (own, task[:1]),
                               (torch.full_like(own, float("nan")), task)):
        with pytest.raises(ValueError):
            inference.act(self_obs, task_obs)
    with pytest.raises(ValueError, match="deterministic"):
        inference.act(own, task, deterministic=False)


def test_expected_hash_and_config_descriptions_cannot_mutate_live_controller(bundle):
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    inference = ReconstructionAMPInference.from_checkpoint(bundle, expected_sha256=digest)
    config = inference.config
    config.normalize_observations = False
    description = deepcopy(inference.describe())
    description["config"]["action_mask"] = [0] * 69
    assert inference.config.normalize_observations is True
    assert inference.describe()["checkpoint_sha256"] == digest
    assert inference.describe()["action_contract"]["PD_decoded"] is False

