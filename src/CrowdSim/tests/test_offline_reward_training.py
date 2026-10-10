"""Execute the offline preference optimizer on small, real CPU clips."""

from types import SimpleNamespace as NS

import pytest
import torch

from CrowdSim.pref import train_offline_reward as training


def _args(tmp_path):
    return NS(dataset=str(tmp_path / "dataset.pt"), output_dir=str(tmp_path / "output"),
              epochs=1, batch_size=1, num_workers=0, learning_rate=1e-4,
              weight_decay=1e-4, calibration_weight=.1, seed=42, device="cpu")


def _dataset():
    return {"metadata": {"schema": training.SCHEMA},
            "clips": {"obs": torch.randn(4, 2, 2), "action": torch.zeros(4, 2, 2),
                      "teacher_score": torch.tensor([0., 1., 2., 3.])},
            "pairs": torch.tensor([[0, 1], [2, 3]]), "labels": torch.tensor([0., 0.]),
            "pair_split": torch.tensor([0, 1])}


@pytest.mark.parametrize("field,value", [("epochs", 0), ("batch_size", 0), ("num_workers", -1),
                                        ("learning_rate", float("nan"))])
def test_invalid_offline_training_bounds_fail_before_loading_dataset(tmp_path, field, value):
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        training.train(args)


def test_actual_offline_training_publishes_finite_loadable_best_checkpoint(tmp_path):
    args = _args(tmp_path)
    torch.save(_dataset(), args.dataset)
    checkpoint = training.train(args)
    assert checkpoint.exists()
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["epoch"] == 1
    assert payload["teacher_normalization"] == {"mean": .5, "std": .5}
    assert all(value.isfinite().all() for value in payload["model"].values())
    assert all(torch.isfinite(torch.tensor(value)) for value in payload["metrics"].values())


def test_nonfinite_offline_input_fails_before_weights_are_updated_or_published(monkeypatch, tmp_path):
    args = _args(tmp_path)
    dataset = _dataset()
    dataset["clips"]["obs"][0, 0, 0] = float("inf")
    torch.save(dataset, args.dataset)
    initial, instances = [], []
    original = training.OfflineRewardModel
    class TrackingModel(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            initial.append({key: value.clone() for key, value in self.state_dict().items()})
            instances.append(self)
    monkeypatch.setattr(training, "OfflineRewardModel", TrackingModel)
    with pytest.raises(FloatingPointError, match="Non-finite offline preference loss"):
        training.train(args)
    assert not list((tmp_path / "output").rglob("*.pt"))
    for key, value in instances[0].state_dict().items():
        torch.testing.assert_close(value, initial[0][key])
