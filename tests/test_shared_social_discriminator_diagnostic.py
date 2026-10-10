"""Meaningful CPU sensitivity fixtures; no SDK, AMASS loading or training."""
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_shared_social_discriminator as diagnostic


def populations():
    generator = torch.Generator().manual_seed(123)
    policy = torch.randn(9, 2019, generator=generator)
    expert = torch.randn(7, 2019, generator=generator)
    # Coherent action-only fixtures: donors preserve each population's marginal,
    # while cross-population donors exchange known offsets.
    policy[:, 1950:] = -.4
    expert[:, 1950:] = .6
    return policy, expert


def linear_discriminator(group):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2019, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        if group == "action":
            model.weight[:, 1950:] = 1 / 69
        elif group == "history":
            model.weight[:, :1950] = 1 / 1950
    return model.eval().requires_grad_(False)


def test_action_only_fixture_identifies_action_channel_and_cross_exchange():
    policy, expert = populations()
    report, raw = diagnostic.analyze(linear_discriminator("action"), policy, expert, seed=7, batch_size=3)
    original = report["conditions"]["original"]["separation"]
    assert original["rank_AUC"] == 1
    assert original["balanced_accuracy_at_logit0"] == 1
    zero = report["conditions"]["actions_zero"]["separation"]
    assert zero["rank_AUC"] == .5
    swapped = report["conditions"]["actions_other_population"]["separation"]
    assert swapped["rank_AUC"] == 0
    assert swapped["balanced_accuracy_at_logit0"] == 0
    assert torch.allclose(raw["original_policy_logit"], torch.full((9,), -.4))
    assert torch.allclose(raw["original_expert_logit"], torch.full((7,), .6))
    for cohort in ("policy", "expert"):
        assert report["input_gradients"][cohort]["history_L2"]["max"] == 0
        assert report["input_gradients"][cohort]["action_RMS"]["mean"] == pytest.approx(1 / 69)


def test_history_only_fixture_is_invariant_to_every_action_intervention():
    policy, expert = populations()
    report, raw = diagnostic.analyze(linear_discriminator("history"), policy, expert, seed=7, batch_size=2)
    for condition in ("actions_zero", "actions_within_population_shuffled", "actions_other_population"):
        for cohort in ("policy", "expert"):
            assert torch.equal(raw[f"{condition}_{cohort}_logit"], raw[f"original_{cohort}_logit"])
            assert report["conditions"][condition][cohort]["absolute_logit_delta"]["max"] == 0
    for cohort in ("policy", "expert"):
        assert report["input_gradients"][cohort]["action_L2"]["max"] == 0
        assert report["input_gradients"][cohort]["history_RMS"]["mean"] == pytest.approx(1 / 1950)


def test_preserves_inputs_rng_and_all_rows_and_repeats_private_seed():
    policy, expert = populations()
    old_policy, old_expert = policy.clone(), expert.clone()
    rng = torch.get_rng_state().clone()
    model = linear_discriminator("action")
    first, arrays = diagnostic.analyze(model, policy, expert, seed=29, batch_size=4)
    second, again = diagnostic.analyze(model, policy, expert, seed=29, batch_size=4)
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(policy, old_policy) and torch.equal(expert, old_expert)
    assert arrays.keys() == again.keys()
    for key in arrays:
        assert torch.equal(arrays[key], again[key])
    assert len(arrays["original_policy_logit"]) == 9
    assert len(arrays["original_expert_logit"]) == 7
    assert first["caller_CPU_RNG_preserved"] and first["input_storage_unchanged"]


def test_tie_correct_auc_and_balanced_population_size():
    values = diagnostic.separation(torch.tensor([0., 0., 1.]), torch.tensor([0., 1.]))
    # Six expert/policy pairs: two wins, one loss, three ties.
    assert values["rank_AUC"] == pytest.approx(3.5 / 6)
    assert diagnostic.separation(torch.zeros(9), torch.zeros(7))["rank_AUC"] == .5


def test_real_builder_constructs_only_D_and_RMS_without_consuming_rng():
    _, _, reader, _ = diagnostic.prepare_sources()
    module = reader._owned_module("_D_fixture_source_trainer", "trainer")
    config = module.SocialTrainingConfig(actor_count=8, grid_center="root", grid_heading="root",
        height_reference="static_center_ground_z", include_ego=False)
    with torch.random.fork_rng(devices=[]):
        saved_D = module._base._mlp(2019, config.discriminator_layers, 1)
        saved_RMS = module._base.RunningNormalizer(195, 5.)
    saved_RMS.count.fill_(120)
    state = {"model": {"discriminator." + key: value for key, value in saved_D.state_dict().items()},
             "normalizers": {"amp_normalizer": saved_RMS.state_dict()}}
    # A full network/trainer constructor would fail this fixture immediately.
    original_factory = module.SocialAMPNetwork
    module.SocialAMPNetwork = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Full NN constructed"))
    rng = torch.get_rng_state().clone()
    try:
        discriminator, normalizer = diagnostic.build_discriminator(module, reader, state, config)
    finally:
        module.SocialAMPNetwork = original_factory
    assert torch.equal(rng, torch.get_rng_state())
    assert not discriminator.training and not normalizer.training
    assert all(not parameter.requires_grad for parameter in discriminator.parameters())
    for name, value in discriminator.state_dict().items():
        assert torch.equal(value, saved_D.state_dict()[name])
        assert value.data_ptr() != saved_D.state_dict()[name].data_ptr()
    assert normalizer.count.item() == 120
    assert not torch.cuda.is_initialized()
    state["model"].pop("discriminator.4.bias")
    with pytest.raises(ValueError, match="Complete original D keys"):
        diagnostic.build_discriminator(module, reader, state, config)


def test_normalization_applies_exact_original_RMS_to_history_only():
    class RMS:
        def normalize(self, values):
            return ((values - 2) / 3).clamp(-5, 5)
    history = torch.full((2, 10, 195), 5.)
    actions = torch.full((2, 69), .25)
    features = diagnostic.normalize_features(history, actions, RMS(), SimpleNamespace(amp_normalization="per_frame"))
    assert torch.equal(features[:, :1950], torch.ones(2, 1950))
    assert torch.equal(features[:, 1950:], actions)
    assert torch.equal(history, torch.full_like(history, 5.))
