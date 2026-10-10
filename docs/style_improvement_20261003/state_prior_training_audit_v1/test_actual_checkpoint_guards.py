"""Actual immutable old/new CPU payload guards; never simulate or save a model."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest
import torch

from state_prior_checks import validate_state_prior

ROOT = Path(__file__).resolve().parents[3]
NEW = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_pilot_v1/step_4.pt"
OLD = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_pilot_v1/step_4.pt"


def check(name, condition, **_detail):
    if not condition:
        raise ValueError(name)


@pytest.fixture(scope="module")
def actual():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(4)
    assert not NEW.stat().st_mode & 0o222
    with NEW.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == "01b8b860045943d7c8994d4dbfa9592df0768a1c6d363e82e95c560797f03bd9"
    return torch.load(NEW, map_location="cpu", weights_only=True, mmap=True)


def candidate(actual):
    # Only the small changed structures are private; never write mmap tensors.
    state = dict(actual)
    state["config"] = deepcopy(actual["config"])
    state["provenance"] = deepcopy(actual["provenance"])
    state["model"] = dict(actual["model"])
    state["normalizers"] = {key: dict(value) for key, value in actual["normalizers"].items()}
    return state


def test_actual_new_trained_columns_and_public_history_are_accepted(actual):
    proof = validate_state_prior(actual, check)
    assert proof["current_AMP_count"] == 249152.
    assert proof["trained_active_columns_nonzero"] == 57
    assert proof["public_AMP_count"] == 15482880001.
    assert not torch.cuda.is_initialized()


def test_actual_frozen_old_independent_D_checkpoint_is_rejected():
    assert not OLD.stat().st_mode & 0o222
    old = torch.load(OLD, map_location="cpu", weights_only=True, mmap=True)
    assert old["provenance"]["initialization"]["published_state_only_discriminator_transferred"] is False
    with pytest.raises(ValueError, match="public_D_and_history_RMS_config"):
        validate_state_prior(old, check)


@pytest.mark.parametrize("change", ["false", "missing", "integer"])
def test_new_flag_tamper_rejected(actual, change):
    state = candidate(actual)
    if change == "missing":
        state["config"].pop("warm_start_action_discriminator_from_state_prior")
    else:
        state["config"]["warm_start_action_discriminator_from_state_prior"] = False if change == "false" else 1
    with pytest.raises(ValueError, match="public_D_and_history_RMS_config"):
        validate_state_prior(state, check)


@pytest.mark.parametrize("field,value", [
    ("published_state_only_discriminator_transferred", False),
    ("published_amp_history_moments_transferred", False),
    ("discriminator_initialization", "independent_random"),
    ("sha256", "0" * 64), ("author_navisaaclab_weights", True)])
def test_provenance_tamper_rejected(actual, field, value):
    state = candidate(actual)
    state["provenance"]["initialization"][field] = value
    with pytest.raises(ValueError, match="provenance_exact"):
        validate_state_prior(state, check)


@pytest.mark.parametrize("change", ["extension_missing", "wrong_original_count", "wrong_cap", "cap_as_count"])
def test_coupled_moments_and_initialization_tamper_rejected(actual, change):
    state = candidate(actual)
    initialization = state["provenance"]["initialization"]
    if change == "extension_missing":
        initialization.pop("state_prior_action_extension")
    elif change == "wrong_original_count":
        initialization["normalizer_counts_before_cap"]["amp_normalizer"] = 200000.
    elif change == "wrong_cap":
        initialization["normalizer_count_cap"] = 0.
    else:
        state["normalizers"]["amp_normalizer"]["count"] = torch.tensor(49152., dtype=torch.float64)
    with pytest.raises(ValueError, match="provenance_exact|capped_public_plus_actual_updates"):
        validate_state_prior(state, check)


@pytest.mark.parametrize("change", ["active_reset", "frozen_nonzero", "D1950", "RMS1949", "negative_variance"])
def test_training_tensor_tamper_rejected(actual, change):
    state = candidate(actual)
    mask = torch.tensor(state["config"]["action_mask"])
    if change in {"active_reset", "frozen_nonzero"}:
        weights = state["model"]["discriminator.0.weight"].clone()
        if change == "active_reset":
            weights[:, 1950:] = 0.
        else:
            weights[0, 1950 + int((mask == 0).nonzero()[0])] = 1.
        state["model"]["discriminator.0.weight"] = weights
    elif change == "D1950":
        state["model"]["discriminator.0.weight"] = state["model"]["discriminator.0.weight"][:, :1950]
    elif change == "RMS1949":
        state["normalizers"]["amp_normalizer"]["mean"] = state["normalizers"]["amp_normalizer"]["mean"][:1949]
    else:
        variance = state["normalizers"]["amp_normalizer"]["variance"].clone()
        variance[0] = -1.
        state["normalizers"]["amp_normalizer"]["variance"] = variance
    with pytest.raises(ValueError, match="trained57|D_shapes|history_RMS"):
        validate_state_prior(state, check)
