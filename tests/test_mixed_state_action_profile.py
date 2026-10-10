"""CPU contracts for the untrained fixed-half state-action AMP variant."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from CrowdSim.humanoid.application.configuration import ConfigurationError, load_amp_profile
from CrowdSim.humanoid.application.settings import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/diffusion_amp/amp_style_mixed_state_action.yaml"
BASELINE = ROOT / "configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml"
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
PACER = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"


def _inputs():
    generator = torch.Generator().manual_seed(20261002)
    return torch.randn(6, 357, generator=generator), torch.randn(6, 1044, generator=generator)


def test_variant_preserves_every_baseline_training_field_except_discriminator_input_and_transfer():
    actual, baseline = load_amp_profile(PROFILE), load_amp_profile(BASELINE)
    changed = {name for name in set(actual["training"]) | set(baseline["training"])
               if actual["training"].get(name) != baseline["training"].get(name)}
    assert changed == {"discriminator_input", "warm_start_discriminator"}
    assert actual["training"]["discriminator_input"] == "state_action"
    assert actual["training"]["warm_start_discriminator"] is False
    assert actual["training"]["amp_normalization"] == "per_history_coordinate"
    assert actual["initialization"]["discriminator"] == "independent_random_history_and_action"
    for key in ("data_manifest", "reward", "trajectory_training", "evaluation"):
        assert actual[key] == baseline[key]
    assert actual["author_experiment_reproduction"] is False


def test_profile_load_remains_torch_and_sdk_free_in_a_fresh_python_process():
    code = """
import sys
from CrowdSim.humanoid.application.configuration import load_amp_profile
p = load_amp_profile('configs/diffusion_amp/amp_style_mixed_state_action.yaml')
assert p['training']['discriminator_input'] == 'state_action'
assert not any(n.split('.')[0] in {'torch', 'numpy', 'omni', 'isaaclab', 'isaacsim'} for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("mutation", [
    "unknown_training", "wrong_mode", "wrong_history", "transfer_state_only_discriminator",
    "boolean_initialization", "bad_initialization_mapping", "reward_metadata_mismatch",
])
def test_existing_profile_validator_rejects_bad_schema_or_untransferable_initialization(tmp_path, mutation):
    profile = deepcopy(load_amp_profile(PROFILE))
    if mutation == "unknown_training":
        profile["training"]["discriminator_action_dims"] = 69
    elif mutation == "wrong_mode":
        profile["mode"] = "strict_author_experiment"
    elif mutation == "wrong_history":
        profile["training"]["history_frames"] = 11
    elif mutation == "transfer_state_only_discriminator":
        profile["training"]["warm_start_discriminator"] = True
    elif mutation == "boolean_initialization":
        profile["training"]["warm_start_discriminator"] = 1
    elif mutation == "bad_initialization_mapping":
        profile["initialization"] = ["public_pacer"]
    elif mutation == "reward_metadata_mismatch":
        profile["reward"]["amp_weight"] = 1.
    else:
        raise AssertionError(mutation)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(ConfigurationError):
        load_amp_profile(path)


@pytest.fixture(scope="module")
def real_setup():
    from CrowdSim.humanoid.amp.controller import PacerAMPController
    from CrowdSim.humanoid.amp.observations import PacerSMPLMapping
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
    from protomotions.robot_configs.factory import robot_config

    profile = load_amp_profile(PROFILE)
    rig = robot_config("smpl")
    lower, upper = rig.kinematic_info.dof_limits_lower, rig.kinematic_info.dof_limits_upper
    mapping = PacerSMPLMapping(body_names=tuple(rig.kinematic_info.body_names), dof_names=tuple(rig.kinematic_info.dof_names))
    trainer = PaperAMPTrainer(TrainingConfig(**profile["training"]), device="cpu")
    random_discriminator = {name: value.clone() for name, value in trainer.model.discriminator.state_dict().items()}
    initialization = trainer.warm_start_pacer(PACER, lower, upper, body_names=mapping.body_names, dof_names=mapping.dof_names)
    reference = PacerAMPController.from_checkpoint(PACER, lower, upper, body_names=mapping.body_names, dof_names=mapping.dof_names)
    corpus = AMASSCorpus(MANIFEST, device="cpu", seed=7, dof_velocity_frame="child_joint")
    return SimpleNamespace(profile=profile, trainer=trainer, reference=reference, corpus=corpus,
        mapping=mapping, lower=lower, upper=upper, initialization=initialization, random_discriminator=random_discriminator)


def test_actual_public_warmstart_transfers_actor_and_leaves_new_2019_discriminator_random(real_setup):
    setup, trainer = real_setup, real_setup.trainer
    own, task = _inputs()
    raw = torch.cat((own, task), -1)
    with torch.no_grad():
        expected_mean, expected_log_std = setup.reference.network.eval_actor(setup.reference.observation_normalizer(raw))
    action = trainer.act(own, task, deterministic=True)
    torch.testing.assert_close(action["actions"], expected_mean, atol=0, rtol=0)
    torch.testing.assert_close(trainer.model.log_std, expected_log_std[0], atol=0, rtol=0)
    for name, value in trainer.model.discriminator.state_dict().items():
        assert torch.equal(value, setup.random_discriminator[name])
    assert trainer.model.discriminator[0].weight.shape == (1024, 2019)
    assert trainer.amp_normalizer.mean.shape == (1950,)
    assert trainer.amp_normalizer.count.item() == 0
    assert torch.count_nonzero(trainer.amp_normalizer.mean) == 0
    assert torch.all(trainer.amp_normalizer.variance == 1)
    assert setup.initialization["published_state_only_discriminator_transferred"] is False
    assert setup.initialization["published_amp_history_moments_transferred"] is False
    assert setup.initialization["action_conditioned_discriminator_initialization"] == "independent_random"


def test_real_gaussian_six_quantity_contract_and_57_effective_coordinates(real_setup):
    trainer = real_setup.trainer
    own, task = _inputs()
    normalized_self, normalized_task = trainer._observations(own, task)
    with torch.no_grad():
        gaussian, _ = trainer.model.policy(normalized_self, normalized_task)
    sampled = trainer.act(own, task, deterministic=False)
    # Six quantities: Gaussian mean/std, raw sample, effective action,
    # scalar behavior likelihood, scalar denormalized value.
    quantities = (gaussian.mean, gaussian.stddev, sampled["actions"], sampled["applied_actions"],
                  sampled["log_prob"], sampled["values"])
    assert [tuple(value.shape) for value in quantities] == [(6, 69)] * 4 + [(6,), (6,)]
    assert all(torch.isfinite(value).all() for value in quantities)
    assert torch.equal(sampled["mean_actions"], gaussian.mean)
    torch.testing.assert_close(sampled["log_prob"],
        (gaussian.log_prob(sampled["actions"]) * trainer.action_mask).sum(-1), atol=0, rtol=0)
    assert int(trainer.action_mask.sum()) == 57
    assert torch.equal(sampled["applied_actions"], sampled["actions"].clamp(-1, 1) * trainer.action_mask)
    assert torch.count_nonzero(sampled["applied_actions"][:, trainer.action_mask == 0]) == 0
    assert torch.count_nonzero(sampled["actions"][:, trainer.action_mask == 0]) > 0


def test_real_v7_expert_batch_uses_ten_newest_first_states_and_effective_next_pose_proxy(real_setup):
    from CrowdSim.humanoid.amp.controller import action_to_pd
    from CrowdSim.humanoid.amp.observations import build_amp_observation

    setup, corpus = real_setup, real_setup.corpus
    # Execute the existing expert_batch body without importing its Kit module.
    source = ast.parse((ROOT / "CrowdSim/humanoid/amp_world.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    method = deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "expert_batch"))
    isolated = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "action_to_pd": action_to_pd, "build_amp_observation": build_amp_observation}
    exec(compile(ast.fix_missing_locations(isolated), "actual_AMPWorld_expert_batch", "exec"), namespace)
    world = SimpleNamespace(corpus=corpus, history_frames=10, device=torch.device("cpu"),
        mapping=setup.mapping, lower=setup.lower, upper=setup.upper)
    original_rng = corpus.generator.get_state().clone()
    batch = 32
    actual_history, actual_actions = namespace["expert_batch"](world, batch)
    corpus.generator.set_state(original_rng)
    indices, ids, frames = corpus._indices(batch, past=9, future=1)
    expected_indices = indices[:, None] - torch.arange(10)
    history = corpus.state_at(expected_indices.reshape(-1))
    expected_history = build_amp_observation(history, mapping=setup.mapping).reshape(batch, 10, 195)
    assert torch.equal(actual_history, expected_history)
    assert torch.equal(indices, corpus.data["length_starts"][ids] + frames)
    assert torch.all(frames >= 9) and torch.all(frames + 1 < corpus.data["motion_num_frames"][ids])
    decoded = action_to_pd(actual_actions, setup.lower, setup.upper, dof_names=setup.mapping.dof_names)
    target = corpus.data["dps"][indices + 1]
    scale = action_to_pd(torch.ones(batch, 69), setup.lower, setup.upper, dof_names=setup.mapping.dof_names)
    limits = scale.abs()
    target = target.clamp(-limits, limits)
    target[:, [i for i, name in enumerate(setup.mapping.dof_names)
               if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]] = 0
    torch.testing.assert_close(decoded, target, atol=3e-7, rtol=0)
    assert actual_actions.shape == (batch, 69)
    assert torch.count_nonzero(actual_actions[:, setup.trainer.action_mask == 0]) == 0
    with torch.no_grad():
        reward = setup.trainer.amp_reward(actual_history, actual_actions)
    assert reward.shape == (batch,) and torch.isfinite(reward).all()
    bad = actual_actions.clone()
    bad[0, torch.where(setup.trainer.action_mask == 0)[0][0]] = .2
    with pytest.raises(ValueError, match="frozen coordinates"):
        setup.trainer.amp_reward(actual_history, bad)


def test_actual_v7_history_discriminator_has_an_action_gradient_on_effective_coordinates(real_setup):
    from CrowdSim.humanoid.amp.observations import build_amp_observation

    setup = real_setup
    _, history, _, _, _ = setup.corpus.sample_reference(8, 10)
    features = build_amp_observation(history, mapping=setup.mapping).reshape(8, 10, 195)
    actions = torch.zeros(8, 69, requires_grad=True)
    normalized, _ = setup.trainer._amp_inputs(features, actions)
    logits = setup.trainer.model.disc_logits(normalized, actions)
    gradient = torch.autograd.grad(logits.sum(), actions)[0]
    active = gradient[:, setup.trainer.action_mask == 1]
    assert torch.isfinite(active).all() and torch.count_nonzero(active) > 0


def test_actual_pacer_wrong_identity_and_schema_fail_before_partial_training_transfer(real_setup, tmp_path):
    setup = real_setup
    before = {name: value.clone() for name, value in setup.trainer.model.state_dict().items()}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup.trainer.warm_start_pacer(PACER, setup.lower, setup.upper, expected_sha256="0" * 64)
    # A valid serialized mapping and matching hash still lack the public schema.
    bad = tmp_path / "invalid_public_schema.pth"
    torch.save({"model": {}}, bad)
    digest = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="lacks actor/critic/discriminator/statistics"):
        setup.trainer.warm_start_pacer(bad, setup.lower, setup.upper, expected_sha256=digest)
    assert all(torch.equal(value, before[name]) for name, value in setup.trainer.model.state_dict().items())
    assert not setup.trainer.optimizer.state and not setup.trainer.discriminator_optimizer.state
    assert asdict(setup.trainer.config)["warm_start_discriminator"] is False
    assert not torch.cuda.is_initialized()
