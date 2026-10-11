"""Real optimizer/checkpoint tests; synthetic rollouts make no physics claim."""
from copy import deepcopy
import uuid

import pytest
import torch

from CrowdSim.humanoid.runtime import social_popart_training as adapter


def definitions():
    return adapter.load_trainer('_test_social_sim61_' + uuid.uuid4().hex)


def rollout(trainer):
    device = trainer.device
    self_obs = torch.zeros(2, 4, 357, device=device)
    task_obs = torch.zeros(2, 4, 12308, device=device)
    sampled = {key: value.unflatten(0, (2, 4)) for key, value in
               trainer.act(self_obs.flatten(0, 1), task_obs.flatten(0, 1)).items()}
    return dict(self_obs=self_obs, task_obs=task_obs,
                amp_history=torch.zeros(2, 4, 10, 195, device=device),
                **sampled, next_values=sampled['values'].clone(),
                rewards=torch.full((2, 4), .5, device=device),
                terminated=torch.zeros(2, 4, dtype=torch.bool, device=device),
                truncated=torch.zeros(2, 4, dtype=torch.bool, device=device))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_both_adams_resume_preserves_CPU_runner_rng_and_updates_all_networks(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required for the actual CUDA optimizer restoration test')
    device = 'cuda:0' if device == 'cuda' else device
    torch.set_num_threads(4)
    module = definitions()
    config = module.SocialTrainingConfig(actor_count=4, grid_center='root', grid_heading='root',
        height_reference='static_center_ground_z', include_ego=False, target_kl=None,
        update_epochs=2, minibatch_size=4, discriminator_updates_per_update=1,
        discriminator_batch_size=8, discriminator_replay_capacity=16)
    trainer = module.SocialAMPTrainer.prepare(config, device=device)
    expert = torch.zeros(8, 10, 195, device=device)
    actions = torch.zeros(8, 69, device=device)
    initial = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    metrics = trainer.update(rollout(trainer), expert, actions)
    assert metrics['optimizer_minibatches'] == 4
    assert metrics['policy_epochs_started'] == 2
    assert metrics['value_normalization']['new_count'] == 8
    assert trainer.optimizer.state and trainer.discriminator_optimizer.state
    for group in ('actor.', 'critic.', 'task_encoder.', 'discriminator.'):
        assert any(not torch.equal(value, initial[name]) for name, value in trainer.model.state_dict().items()
                   if name.startswith(group))
    rng = torch.Generator(device='cpu').manual_seed(73).get_state()
    path = tmp_path / 'social.pt'
    trainer.save(path, runner_state={'sampler_rng': rng})
    restored, runner = module.SocialAMPTrainer.load(path, device=device)
    assert runner['sampler_rng'].device.type == 'cpu'
    assert torch.equal(runner['sampler_rng'], rng)
    assert restored.update_count == 1 and restored.physical_transition_count == 8
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    for before, after in ((trainer.optimizer, restored.optimizer),
                          (trainer.discriminator_optimizer, restored.discriminator_optimizer)):
        expected, actual = before.state_dict(), after.state_dict()
        assert expected['param_groups'] == actual['param_groups']
        for key, fields in expected['state'].items():
            for field, value in fields.items():
                assert torch.equal(value, actual['state'][key][field])
                if field != 'step':
                    assert actual['state'][key][field].device == restored.device
    saved = torch.load(path, map_location='cpu', weights_only=True)
    saved['model']['log_std'][0] += .1
    with pytest.raises(ValueError, match='log_std changed'):
        restored._validate_checkpoint(saved)
    saved['model']['log_std'][0] = config.log_std
    frozen_column = int((restored.action_mask == 0).nonzero()[0])
    saved['policy_replay']['actions'][0, frozen_column] = .2
    with pytest.raises(ValueError, match='frozen PD action columns'):
        restored._validate_checkpoint(saved)
    del saved
    next_metrics = restored.update(rollout(restored), expert, actions)
    assert restored.update_count == 2 and restored.physical_transition_count == 16
    assert next_metrics['value_normalization']['new_count'] == 16
    assert all(torch.isfinite(value).all() for value in restored.model.state_dict().values())


def test_runtime_binding_is_a_distinct_family_and_historical_evidence_stays_legacy():
    historical, current = adapter.describe(), adapter.describe_runtime()
    assert historical['checkpoint_format'] == adapter.LEGACY_FORMAT
    assert current['checkpoint_format'] == adapter.FORMAT != adapter.LEGACY_FORMAT
    assert current['derivation']['source_selector_exact_inverse']
    assert current['derivation']['load_runner_tensors_on_CPU']
    assert not current['derivation']['historical_checkpoint_implicitly_admitted']
    module = definitions()
    state = {name: None for name in module.CHECKPOINT_KEYS}
    state.update(format=adapter.LEGACY_FORMAT, version=1, provenance={})
    with pytest.raises(ValueError, match='schema'):
        module.SocialAMPTrainer._admit_metadata(state)


@pytest.mark.parametrize('dependency', tuple(adapter.CURRENT_SOURCES))
def test_unknown_current_dependencies_are_rejected(monkeypatch, dependency):
    pins = deepcopy(adapter.CURRENT_SOURCES)
    pins[dependency] = '0' * 64
    monkeypatch.setattr(adapter, 'CURRENT_SOURCES', pins)
    with pytest.raises(ValueError, match='dependency changed'):
        adapter.describe_runtime()


def test_tampered_historical_numerical_source_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'trainer.py'
    source.write_text(adapter.NUMERICAL_PARENT.read_text() + '\n# drift\n')
    monkeypatch.setattr(adapter, 'NUMERICAL_PARENT', source)
    with pytest.raises(ValueError, match='numerical trainer snapshot changed'):
        adapter.describe_runtime()


def test_unknown_edits_to_sealed_derivation_input_are_rejected():
    # A caller must not be able to supply unregistered functions beside the
    # verified on-disk parent to the public source derivation helper.
    with pytest.raises(ValueError, match='Social trainer changed'):
        adapter.derive_runtime(adapter.TRAINER.read_text() + '\n# unregistered\n')
