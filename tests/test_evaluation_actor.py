"""Actor-only evaluation and deliberately separate legacy global RNG setup."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig


def rng_state():
    value = np.random.get_state()
    return random.getstate(), (value[0], value[1].copy(), *value[2:]), torch.get_rng_state().clone()


def assert_rng_equal(expected):
    actual = rng_state()
    assert expected[0] == actual[0]
    assert expected[1][0] == actual[1][0] and expected[1][2:] == actual[1][2:]
    np.testing.assert_array_equal(expected[1][1], actual[1][1])
    assert torch.equal(expected[2], actual[2])


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    trainer = PaperAMPTrainer(TrainingConfig(discriminator_replay_capacity=4))
    trainer.provenance = {"physics": {"control_hz": 30, "names": ["unit_fixture"]},
        "diffusion_sha256": "unit_fixture_not_a_published_model", "extra_nested_training": {"stage": [1, 2, 3]}}
    trainer.update_count, trainer.physical_transition_count = 4, 128
    # Unit saved RNG differs deliberately from the constructor seed boundary.
    random.random()
    np.random.rand(5)
    torch.rand(9)
    path = tmp_path_factory.mktemp("evaluation_actor") / "unit_bundle.pt"
    trainer.save(path)
    return path


def inputs():
    generator = torch.Generator().manual_seed(77)
    return torch.randn(5, 357, generator=generator), torch.randn(5, 1044, generator=generator)


def test_cold_facade_and_helper_definition_import_no_torch_numpy_or_sdk():
    code = """
import sys
from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
assert not any(n.split('.')[0] in {'torch','numpy','isaaclab','isaacsim','omni'} for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_facade_preserves_complete_small_provenance_rng_and_exactly_three_actor_outputs(bundle, monkeypatch):
    import CrowdSim.humanoid.training.replay as replay
    monkeypatch.setattr(PaperAMPTrainer, "__init__", lambda *a, **kw: pytest.fail("Facade created a trainer"))
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("Facade created an optimizer"))
    monkeypatch.setattr(replay, "AMPPolicyReplay", lambda *a, **kw: pytest.fail("Facade created replay"))
    before = rng_state()
    actor = EvaluationActor.from_checkpoint(bundle)
    output = actor.act(*inputs())
    assert_rng_equal(before)
    assert set(output) == {"actions", "mean_actions", "applied_actions"}
    assert actor.update_count == 4 and actor.physical_transition_count == 128
    assert actor.provenance["extra_nested_training"] == {"stage": [1, 2, 3]}
    assert actor.device == torch.device("cpu")
    for name in ("optimizer", "policy_replay", "discriminator_optimizer", "amp_reward", "value", "restore_rng", "save", "update"):
        assert not hasattr(actor, name)
    assert actor.describe()["saved_training_rng_restored"] is False


def test_facade_descriptions_provenance_and_config_do_not_alias_live_state(bundle):
    actor = EvaluationActor.from_checkpoint(bundle)
    provenance = actor.provenance
    provenance["extra_nested_training"]["stage"].append(100)
    config = actor.config
    config.seed = 100
    description = actor.describe()
    description["full_training_provenance"]["physics"]["names"].append("mutated")
    assert actor.config.seed == 7
    assert actor.provenance["extra_nested_training"]["stage"] == [1, 2, 3]
    assert actor.provenance["physics"]["names"] == ["unit_fixture"]


def test_helper_matches_actual_legacy_load_rng_and_does_not_restore_saved_rng(bundle):
    before = rng_state()
    try:
        oracle, _ = PaperAMPTrainer.load(bundle, device="cpu", restore_rng=False)
        expected = rng_state()
        saved = torch.load(bundle, map_location="cpu", weights_only=True, mmap=True)["rng"]
        assert not torch.equal(expected[2], saved["torch_cpu"])
        report = prepare_legacy_evaluation_rng(oracle.config)
        assert_rng_equal(expected)
        assert report["saved_checkpoint_rng_restored"] is False
        assert report["GPU_rng_equivalence_verified_by_this_helper"] is False
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


def test_helper_creates_only_temporary_network_not_trainer_optimizer_or_replay(bundle, monkeypatch):
    import CrowdSim.humanoid.training.replay as replay
    actor = EvaluationActor.from_checkpoint(bundle)
    monkeypatch.setattr(PaperAMPTrainer, "__init__", lambda *a, **kw: pytest.fail("RNG helper created a trainer"))
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("RNG helper created an optimizer"))
    monkeypatch.setattr(replay, "AMPPolicyReplay", lambda *a, **kw: pytest.fail("RNG helper created replay"))
    before = rng_state()
    try:
        result = prepare_legacy_evaluation_rng(actor.config)
        assert result["temporary_CPU_PaperAMPNetwork_parameters"] > 1000000
        assert not result["trainer_created"] and not result["optimizer_created"] and not result["replay_created"]
        assert random.random() == random.Random(7).random()
        expected_numpy = np.random.RandomState(7)
        assert np.random.rand() == expected_numpy.rand()
        assert not torch.equal(before[2], torch.get_rng_state())
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


@pytest.mark.parametrize("bad", [{"seed": True}, {"control_dt": .1}, {"unknown": 1}])
def test_bad_rng_configuration_fails_before_reseeding(bad):
    before = rng_state()
    with pytest.raises(ValueError):
        prepare_legacy_evaluation_rng({**asdict(TrainingConfig()), **bad})
    assert_rng_equal(before)


def test_nonstandard_default_dtype_rejected_before_seeding():
    before = rng_state()
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="default CPU float32"):
            prepare_legacy_evaluation_rng(TrainingConfig())
        assert_rng_equal(before)
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize("change", ["NaN", "tensor", "oversize"])
def test_bad_full_provenance_rejected_even_if_selected_inference_metadata_is_valid(bundle, tmp_path, change):
    artifact = torch.load(bundle, map_location="cpu", weights_only=True, mmap=True)
    artifact["provenance"]["new_full_only"] = (
        float("nan") if change == "NaN" else torch.ones(2) if change == "tensor" else "x" * (2*1024*1024))
    path = tmp_path / "bad_provenance.pt"
    torch.save(artifact, path)
    before = rng_state()
    with pytest.raises(ValueError, match="provenance"):
        EvaluationActor.from_checkpoint(path)
    assert_rng_equal(before)


def test_second_metadata_load_rejects_counter_mismatch(bundle, monkeypatch):
    original = torch.load
    calls = []

    def changed(*args, **kwargs):
        artifact = original(*args, **kwargs)
        calls.append(kwargs)
        if len(calls) == 2:
            artifact = deepcopy(artifact)
            artifact["physical_transition_count"] += 1
        return artifact

    monkeypatch.setattr(torch, "load", changed)
    before = rng_state()
    with pytest.raises(ValueError, match="counters differ"):
        EvaluationActor.from_checkpoint(bundle)
    assert len(calls) == 2 and all(value == {"map_location": "cpu", "weights_only": True, "mmap": True} for value in calls)
    assert_rng_equal(before)


def test_facade_rejects_stochastic_actions_instead_of_fabricating_training_outputs(bundle):
    actor = EvaluationActor.from_checkpoint(bundle)
    before = rng_state()
    with pytest.raises(ValueError, match="deterministic"):
        actor.act(*inputs(), deterministic=False)
    assert_rng_equal(before)


@pytest.fixture(scope="module")
def arm_pair(tmp_path_factory):
    from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, PACER_DOF_NAMES

    physics = {"engine": "PhysX", "physics_hz": 120, "control_hz": 30, "rig": "unit fixture not physics validation",
        "body_names": list(PACER_BODY_NAMES), "dof_names": list(PACER_DOF_NAMES),
        "dof_limits_lower": [-3.] * 69, "dof_limits_upper": [3.] * 69,
        "joint_control": {name: {"stiffness": 500., "damping": 50., "armature": .02,
            "effort_limit": 500., "velocity_limit": 100.} for name in PACER_DOF_NAMES},
        "terrain_material": {"static_friction": 1., "dynamic_friction": 1., "restitution": 0.,
            "height_offset": 0., "combine_mode": "average"}}
    folder = tmp_path_factory.mktemp("evaluation_arm")
    paths = []
    for label, count, bias in (("baseline", 8, 3.), ("style", 19, -1.)):
        trainer = PaperAMPTrainer(TrainingConfig(discriminator_replay_capacity=0),
            provenance={"physics": physics, "complete_only": {"label": label}})
        trainer.update_count, trainer.physical_transition_count = count, count * 32
        with torch.no_grad():
            trainer.model.actor[-1].weight.zero_()
            trainer.model.actor[-1].bias.fill_(bias)
        path = folder / f"unit_{label}.pt"
        trainer.save(path)
        paths.append(path)
    return paths


def test_composite_facade_constructs_only_two_networks_and_keeps_baseline_metadata(arm_pair, monkeypatch):
    from CrowdSim.humanoid.training.trainer import PaperAMPNetwork

    baseline = EvaluationActor.from_checkpoint(arm_pair[0])
    expected = baseline.act(*inputs())
    original = PaperAMPNetwork.__init__
    calls = []

    def count(self, *args, **kwargs):
        calls.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(PaperAMPNetwork, "__init__", count)
    before = rng_state()
    actor = EvaluationActor.from_checkpoints(*arm_pair, alpha=0.)
    output = actor.act(*inputs())
    assert_rng_equal(before)
    assert len(calls) == 2
    assert actor.update_count == 8 and actor.physical_transition_count == 256
    assert actor.provenance["complete_only"] == {"label": "baseline"}
    assert all(torch.equal(output[key].view(torch.uint8), expected[key].view(torch.uint8)) for key in output)
    report = actor.describe()
    assert report["checkpoint_sha256"] == baseline.describe()["checkpoint_sha256"]
    assert report["experimental_arm_style"]["blended_active_coordinates"] == 24
    assert report["experimental_arm_style"]["alpha"] == 0.
    report["experimental_arm_style"]["arm_action_indices"].clear()
    assert len(actor.describe()["experimental_arm_style"]["arm_action_indices"]) == 24
    assert not any(hasattr(actor, name) for name in ("value", "amp_reward", "log_prob", "optimizer", "replay"))


def test_composite_facade_exposes_only_actual_blended_actor_outputs(arm_pair):
    from CrowdSim.humanoid.runtime.arm_style import ARM_ACTION_INDICES, BASE_ACTION_INDICES

    before = rng_state()
    actor = EvaluationActor.from_checkpoints(*arm_pair, alpha=.5)
    result = actor.act(*inputs())
    assert set(result) == {"actions", "mean_actions", "applied_actions"}
    assert torch.equal(result["mean_actions"][:, list(ARM_ACTION_INDICES)], torch.ones(5, 24))
    assert torch.equal(result["mean_actions"][:, list(BASE_ACTION_INDICES)], torch.full((5, 45), 3.))
    assert actor.provenance["complete_only"] == {"label": "baseline"}
    assert_rng_equal(before)


@pytest.mark.parametrize("alpha,style", [(True, "missing"), (float("nan"), "missing"), (.1, None)])
def test_composite_invalid_alpha_or_missing_style_rejected_before_loading(alpha, style):
    before = rng_state()
    with pytest.raises(ValueError):
        EvaluationActor.from_checkpoints("missing", style, alpha=alpha)
    assert_rng_equal(before)
