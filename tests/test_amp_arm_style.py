"""Experimental blend contracts; synthetic unit actors are not physical QA."""
from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, PACER_DOF_NAMES
from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
from CrowdSim.humanoid.runtime.arm_style import (
    ARM_ACTION_INDICES, ARM_ACTION_NAMES, BASE_ACTION_INDICES, ExperimentalArmStyleInference,
)
from CrowdSim.humanoid.runtime.checkpoint_contract import CANONICAL_ACTION_MASK
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig


def unit_physics():
    # Explicit synthetic metadata, not a physical asset or native measurement.
    return {"engine": "PhysX", "physics_hz": 120, "control_hz": 30, "rig": "unit canonical rig only",
        "body_names": list(PACER_BODY_NAMES), "dof_names": list(PACER_DOF_NAMES),
        "dof_limits_lower": [-3.] * 69, "dof_limits_upper": [3.] * 69,
        "joint_control": {name: {"stiffness": 500., "damping": 50., "armature": .02,
            "effort_limit": 500., "velocity_limit": 100., "friction": None} for name in PACER_DOF_NAMES},
        "terrain_material": {"static_friction": 1., "dynamic_friction": 1., "restitution": 0.,
            "height_offset": 0., "combine_mode": "average"},
        "terrain_extent_m": 192., "isolated_spawn_margin_m": 48.}


@pytest.fixture(scope="module")
def pair_paths(tmp_path_factory):
    folder = tmp_path_factory.mktemp("arm_style_contract")
    paths = []
    for label, bias, moment in (("base", 3., -2.), ("style", -1., 4.)):
        trainer = PaperAMPTrainer(TrainingConfig(discriminator_replay_capacity=0), device="cpu",
                                 provenance={"physics": unit_physics()})
        with torch.no_grad():
            trainer.model.actor[-1].weight.zero_()
            trainer.model.actor[-1].bias.fill_(bias)
            trainer.self_normalizer.mean.fill_(moment)
            trainer.task_normalizer.mean.fill_(moment / 2)
        path = folder / f"unit_{label}.pt"
        trainer.save(path)
        paths.append(path)
    return paths


def load_pair(paths, *, alpha=0.):
    return ExperimentalArmStyleInference.from_checkpoints(*paths, device="cpu", alpha=alpha)


def observations():
    generator = torch.Generator().manual_seed(11)
    return torch.randn(3, 357, generator=generator), torch.randn(3, 1044, generator=generator)


def rng_state():
    numpy = np.random.get_state()
    return random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), torch.get_rng_state().clone()


def assert_rng(before):
    current = rng_state()
    assert current[0] == before[0] and current[1][0] == before[1][0] and current[1][2:] == before[1][2:]
    np.testing.assert_array_equal(current[1][1], before[1][1])
    assert torch.equal(current[2], before[2])


def byte_equal(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def test_only_named_upper_limb24_active_axes_are_available():
    names = {name.rsplit("_", 1)[0] for name in ARM_ACTION_NAMES}
    assert names == {f"{side}_{joint}" for side in ("L", "R") for joint in ("Thorax", "Shoulder", "Elbow", "Wrist")}
    assert len(ARM_ACTION_INDICES) == 24 and len(BASE_ACTION_INDICES) == 45
    assert all(CANONICAL_ACTION_MASK[index] for index in ARM_ACTION_INDICES)
    assert sum(CANONICAL_ACTION_MASK[index] for index in BASE_ACTION_INDICES) == 33
    assert [PACER_DOF_NAMES[index] for index in ARM_ACTION_INDICES] == list(ARM_ACTION_NAMES)
    assert not set(ARM_ACTION_INDICES) & set(BASE_ACTION_INDICES)


def test_default_alpha0_returns_baseline_exactly_without_style_forward(pair_paths, monkeypatch):
    blend = load_pair(pair_paths)
    inputs = observations()
    expected = blend.baseline.act(*inputs)

    def prohibited_style_forward(*args, **kwargs):
        raise AssertionError("alpha0 must never forward the style policy")

    monkeypatch.setattr(blend.style, "act", prohibited_style_forward)
    actual = blend.act(*inputs)
    assert set(actual) == {"actions", "mean_actions", "applied_actions"}
    assert all(byte_equal(actual[key], expected[key]) for key in actual)


def test_alpha1_exact_style24_and_baseline_other45_raw_and_effective_actions(pair_paths):
    blend = load_pair(pair_paths, alpha=1.)
    inputs = observations()
    base, style, result = blend.baseline.act(*inputs), blend.style.act(*inputs), blend.act(*inputs)
    assert byte_equal(result["mean_actions"][:, list(ARM_ACTION_INDICES)], style["mean_actions"][:, list(ARM_ACTION_INDICES)])
    assert byte_equal(result["mean_actions"][:, list(BASE_ACTION_INDICES)], base["mean_actions"][:, list(BASE_ACTION_INDICES)])
    assert byte_equal(result["applied_actions"][:, list(BASE_ACTION_INDICES)], base["applied_actions"][:, list(BASE_ACTION_INDICES)])
    assert byte_equal(result["actions"], result["mean_actions"])
    frozen = [index for index, active in enumerate(CANONICAL_ACTION_MASK) if not active]
    assert not result["applied_actions"][:, frozen].any()


def test_raw_blend_precedes_projection_not_already_clamped_policy_mix(pair_paths):
    blend = load_pair(pair_paths, alpha=.5)
    inputs = observations()
    result = blend.act(*inputs)
    arms = list(ARM_ACTION_INDICES)
    # Raw means +3 and -1: correct raw blend=+1, versus clamped-action mix=0.
    assert torch.equal(result["mean_actions"][:, arms], torch.ones(3, 24))
    assert torch.equal(result["applied_actions"][:, arms], torch.ones(3, 24))
    base, style = blend.baseline.act(*inputs), blend.style.act(*inputs)
    clipped_mix = .5 * base["applied_actions"][:, arms] + .5 * style["applied_actions"][:, arms]
    assert torch.equal(clipped_mix, torch.zeros(3, 24))
    assert not torch.equal(result["applied_actions"][:, arms], clipped_mix)


def test_each_actor_uses_own_frozen_rms_and_all_rng_unchanged(pair_paths, monkeypatch):
    before = rng_state()
    blend = load_pair(pair_paths, alpha=.25)
    assert_rng(before)
    buffers = {name: value.clone() for name, value in blend.named_buffers()}
    calls = []
    for label, actor in (("base", blend.baseline), ("style", blend.style)):
        for channel in ("self_normalizer", "task_normalizer"):
            normalizer = getattr(actor, channel)
            original = normalizer.normalize

            def spy(values, *, label=label, channel=channel, normalizer=normalizer, original=original):
                calls.append((label, channel, float(normalizer.mean[0])))
                return original(values)

            monkeypatch.setattr(normalizer, "normalize", spy)
    first = blend.act(*observations())
    second = blend.act(*observations())
    assert all(byte_equal(first[key], second[key]) for key in first)
    assert calls[:4] == [("base", "self_normalizer", -2.), ("base", "task_normalizer", -1.),
                         ("style", "self_normalizer", 4.), ("style", "task_normalizer", 2.)]
    assert all(torch.equal(value, buffers[name]) for name, value in blend.named_buffers())
    assert_rng(before)


@pytest.mark.parametrize("alpha", [True, False, -.1, 1.1, float("nan"), float("inf"), ".25", None])
def test_bad_alpha_fails_before_checkpoint_load(alpha):
    before = rng_state()
    with pytest.raises(ValueError, match="alpha"):
        ExperimentalArmStyleInference.from_checkpoints("missing_base.pt", "missing_style.pt", alpha=alpha)
    assert_rng(before)


@pytest.mark.parametrize("change", ["mask", "double", "partial_device", "different_device", "schema", "config"])
def test_actor_contract_mutation_is_rejected_even_alpha0(pair_paths, change):
    blend = load_pair(pair_paths)
    if change == "mask":
        blend.style.action_mask[0] = 0
    elif change == "double":
        blend.style.double()
    elif change == "partial_device":
        blend.style.model.actor[0].to("meta")
    elif change == "different_device":
        blend.style.to("meta")
    elif change == "schema":
        blend.style._metadata["format"] = "public_pacer"
    else:
        blend.style._config.control_dt = 1 / 60
    before = rng_state()
    with pytest.raises(ValueError):
        blend.act(*observations())
    assert_rng(before)


@pytest.mark.parametrize("change", ["body_order", "limits", "PD", "material", "missing_physics", "nonfinite_physics"])
def test_same69_dimensions_cannot_hide_different_physical_contract(pair_paths, tmp_path, change):
    data = torch.load(pair_paths[1], weights_only=True, map_location="cpu", mmap=True)
    physics = data["provenance"]["physics"]
    if change == "body_order":
        physics["body_names"][1:3] = physics["body_names"][2:0:-1]
    elif change == "limits":
        physics["dof_limits_upper"][0] = 2.
    elif change == "PD":
        physics["joint_control"]["L_Shoulder_x"]["stiffness"] = 1000.
    elif change == "material":
        physics["terrain_material"]["dynamic_friction"] = .5
    elif change == "missing_physics":
        del data["provenance"]["physics"]
    else:
        physics["terrain_material"]["static_friction"] = float("nan")
    path = tmp_path / "altered_physics.pt"
    torch.save(data, path)
    before = rng_state()
    with pytest.raises(ValueError):
        ExperimentalArmStyleInference.from_checkpoints(pair_paths[0], path)
    assert_rng(before)


def test_only_declared_training_domain_fields_can_differ_and_are_reported(pair_paths, tmp_path):
    data = torch.load(pair_paths[1], weights_only=True, map_location="cpu", mmap=True)
    data["provenance"]["physics"]["terrain_extent_m"] = 64.
    data["provenance"]["physics"]["isolated_spawn_margin_m"] = 6.
    path = tmp_path / "different_training_domain.pt"
    torch.save(data, path)
    blend = ExperimentalArmStyleInference.from_checkpoints(pair_paths[0], path)
    domains = blend.describe()["physical_compatibility"]["training_domains"]
    assert domains == {"baseline": {"terrain_extent_m": 192., "isolated_spawn_margin_m": 48.},
                       "style": {"terrain_extent_m": 64., "isolated_spawn_margin_m": 6.}}


def test_invalid_schema_and_dtype_inputs_no_stochastic_or_training_api(pair_paths, tmp_path):
    bad = tmp_path / "public.pt"
    torch.save({"format": "public_pacer"}, bad)
    with pytest.raises(ValueError, match="reconstruction"):
        ExperimentalArmStyleInference.from_checkpoints(pair_paths[0], bad)
    blend = load_pair(pair_paths)
    own, task = observations()
    before = rng_state()
    buffers = {name: value.clone() for name, value in blend.named_buffers()}
    for inputs in ((own.double(), task.double()), (own[:0], task[:0]), (own, task[:1]),
                   (own * float("nan"), task)):
        with pytest.raises(ValueError):
            blend.act(*inputs)
        assert_rng(before)
        assert all(torch.equal(value, buffers[name]) for name, value in blend.named_buffers())
    with pytest.raises(ValueError, match="deterministic"):
        blend.act(own, task, deterministic=False)
    blend.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        blend.act(own, task)
    assert_rng(before)


def test_metadata_is_explicit_experiment_without_composite_training_claim(pair_paths):
    blend = load_pair(pair_paths, alpha=.25)
    report = blend.describe()
    assert report["blended_active_coordinates"] == 24 and report["baseline_other_active_coordinates"] == 33
    assert report["baseline_frozen_coordinates"] == 12
    assert not report["PD_decoded"] and not report["paper_single_actor_reproduction"]
    assert not report["physical_validation_performed"] and not report["physical_safety_guaranteed"]
    assert not report["composite_log_prob_available"] and not report["composite_critic_available"]
    assert not report["composite_AMP_reward_available"] and not report["main_discriminator_changed"]
    assert set(blend.act(*observations())) == {"actions", "mean_actions", "applied_actions"}
    original = asdict(blend.config)
    owned = blend.config
    owned.seed = -1
    assert asdict(blend.config) == original
    assert all(not parameter.requires_grad for parameter in blend.parameters())
    assert not any(hasattr(blend, name) for name in ("optimizer", "replay", "trainer", "runner_state"))


def test_constructor_rejects_unloaded_actor_and_alternate_controller(pair_paths):
    base = ReconstructionAMPInference.from_checkpoint(pair_paths[0])
    with pytest.raises(ValueError, match="fully loaded"):
        ExperimentalArmStyleInference(base, object())
    unloaded = ReconstructionAMPInference(base.config)
    with pytest.raises(RuntimeError, match="validated"):
        ExperimentalArmStyleInference(base, unloaded)
    broken = deepcopy(base)
    broken.action_mask = torch.ones(70)
    with pytest.raises(ValueError, match="canonical"):
        ExperimentalArmStyleInference(base, broken)

