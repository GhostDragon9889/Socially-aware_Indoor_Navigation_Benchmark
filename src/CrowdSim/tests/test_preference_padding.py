from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from CrowdSim.pref import pref_train_reward as training


def _episode(length):
    return {"steps": [{"sys_state": np.array([float(i), .2], np.float32),
                       "action": np.array([.5, -.1], np.float32), "depth": np.full((8, 8), i / 10, np.float32)}
                      for i in range(length)]}


def _model():
    torch.manual_seed(42)
    return training.CrowdSimRewardNet(system_dim=4, vec_dim=2, depth_size=8,
        depth_embed_dim=8, state_embed_dim=16, action_embed_dim=16, max_len=8).eval()


def test_real_reward_network_is_invariant_to_batch_padding_and_zeroes_padded_gradients():
    model = _model()
    short, long = _episode(2), _episode(4)
    alone = model([short])
    together = model([short, long])
    torch.testing.assert_close(alone[0], together[0], atol=1e-6, rtol=1e-6)
    per_step = model.forward_per_step([short, long])
    assert torch.equal(per_step[0, 2:], torch.zeros(2))
    system, depth, valid = model._build_batches([short, long])
    system.requires_grad_()
    # Arbitrary padding must neither contribute rewards nor affect real frames.
    noisy = system.detach().clone()
    noisy[~valid] = 1_000
    depth[~valid] = 1_000
    torch.testing.assert_close(model.net(noisy, depth_feats=depth, valid_mask=valid)[0, :2], per_step[0, :2], atol=1e-6, rtol=1e-6)
    model.net(system, depth_feats=depth, valid_mask=valid).sum().backward()
    assert torch.equal(system.grad[~valid], torch.zeros_like(system.grad[~valid]))
    assert system.grad[valid].isfinite().all()


def test_reward_batch_rejects_empty_episode_without_shifting_pair_labels():
    model = _model()
    with pytest.raises(ValueError, match="at least one step"):
        model([{"steps": []}, _episode(2)])
    assert model([]).shape == (0,)
    with pytest.raises(ValueError, match="max_len"):
        model([_episode(9)])


def test_actual_small_preference_training_uses_eval_validation_and_saves_first_epoch(monkeypatch, tmp_path):
    modes, instances = [], []
    original = training.CrowdSimRewardNet
    class TrackingModel(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, state_embed_dim=16, action_embed_dim=16, depth_embed_dim=8)
            instances.append(self)
        def forward(self, *args, **kwargs):
            modes.append((self.training, torch.is_grad_enabled()))
            return super().forward(*args, **kwargs)
    monkeypatch.setattr(training, "CrowdSimRewardNet", TrackingModel)
    buffer = NS(pairs=[(_episode(2), _episode(3), 1), (_episode(3), _episode(2), 0)])
    result = training.train(buffer, batch_size=2, epochs=2, device="cpu", output=str(tmp_path / "reward.pt"))
    assert result is instances[0] and result.training
    assert modes == [(True, True), (True, True), (False, False), (False, False)] * 2
    assert len(list(tmp_path.glob("reward_*.pt"))) == 1
    assert all(parameter.isfinite().all() for parameter in result.parameters())


def test_one_pair_cannot_create_train_validation_split():
    with pytest.raises(ValueError, match="at least two pairs"):
        training.train(NS(pairs=[(_episode(2), _episode(3), 1)]), device="cpu")


def test_nonfinite_preference_input_cannot_publish_or_corrupt_weights(monkeypatch, tmp_path):
    instances, initial = [], []
    original = training.CrowdSimRewardNet
    class TrackingModel(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, state_embed_dim=16, action_embed_dim=16, depth_embed_dim=8)
            instances.append(self)
            initial.append({key: value.clone() for key, value in self.state_dict().items()})
    monkeypatch.setattr(training, "CrowdSimRewardNet", TrackingModel)
    bad = _episode(2)
    bad["steps"][0]["sys_state"][0] = float("inf")
    buffer = NS(pairs=[(bad, _episode(2), 1), (bad, _episode(2), 0)])
    with pytest.raises(FloatingPointError, match="Non-finite preference loss"):
        training.train(buffer, batch_size=1, epochs=1, device="cpu", output=str(tmp_path / "reward.pt"))
    assert not list(tmp_path.glob("*.pt"))
    for key, value in instances[0].state_dict().items():
        torch.testing.assert_close(value, initial[0][key])
