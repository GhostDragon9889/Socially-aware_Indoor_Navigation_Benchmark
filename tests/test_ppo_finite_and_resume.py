from copy import deepcopy
import importlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

from crowdsim_ppo import train as training, evaluate as evaluation


@pytest.mark.parametrize("module_name", ["CrowdSim.ppo.ppo_policy", "crowdsim_ppo.ppo_policy"])
@pytest.mark.parametrize("method", ["update", "update_with_kl"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_loss_refuses_optimizer_mutation(module_name, method, invalid):
    module = importlib.import_module(module_name)
    trainer = module.RobotPPOTrainer(module.RobotPPOConfig(obs_dim=5, num_neighbors=0,
        map_enabled=False, depth_enabled=False, hidden_dims=(8,), ppo_epochs=1, minibatch_size=1), torch.device("cpu"))
    reference = deepcopy(trainer.model).eval()
    buffer = module.RobotRolloutBuffer(1, 1, 5, 2, torch.device("cpu"))
    obs = torch.zeros(1, 5)
    _, raw, logp, value = trainer.act(obs)
    buffer.add(obs, raw, logp, torch.ones(1), torch.ones(1, dtype=torch.bool), value)
    buffer.compute_returns_and_advantages(torch.zeros(1), 0., 0.)
    buffer.returns[:] = invalid
    before = {key: tensor.clone() for key, tensor in trainer.model.state_dict().items()}
    with pytest.raises(FloatingPointError, match="Non-finite PPO loss"):
        if method == "update":
            trainer.update(buffer)
        else:
            trainer.update_with_kl(buffer, reference, .1)
    assert trainer.optimizer.state == {}
    for key, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, before[key])


@pytest.mark.parametrize("module_name", ["CrowdSim.ppo.ppo_policy", "crowdsim_ppo.ppo_policy"])
@pytest.mark.parametrize("method", ["update", "update_with_kl"])
def test_finite_loss_but_infinite_backward_gradient_refuses_adam_step(module_name, method):
    module = importlib.import_module(module_name)
    trainer = module.RobotPPOTrainer(module.RobotPPOConfig(obs_dim=5, num_neighbors=0,
        map_enabled=False, depth_enabled=False, hidden_dims=(8,), ppo_epochs=1, minibatch_size=1), torch.device("cpu"))
    reference = deepcopy(trainer.model).eval()
    buffer = module.RobotRolloutBuffer(1, 1, 5, 2, torch.device("cpu"))
    obs = torch.zeros(1, 5)
    _, raw, logp, value = trainer.act(obs)
    buffer.add(obs, raw, logp, torch.ones(1), torch.ones(1, dtype=torch.bool), value)
    buffer.compute_returns_and_advantages(torch.zeros(1), 0., 0.)
    before = {key: tensor.clone() for key, tensor in trainer.model.state_dict().items()}
    parameter = next(trainer.model.parameters())
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            if method == "update":
                trainer.update(buffer)
            else:
                trainer.update_with_kl(buffer, reference, .1)
    finally:
        handle.remove()
    assert trainer.optimizer.state == {}
    for key, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, before[key])


@pytest.mark.parametrize("module_name", ["CrowdSim.ppo.ppo_policy", "crowdsim_ppo.ppo_policy"])
@pytest.mark.parametrize("field", ["rollout_steps", "num_envs"])
def test_empty_rollout_is_rejected_before_nonprogressing_training(module_name, field):
    module = importlib.import_module(module_name)
    kwargs = dict(rollout_steps=1, num_envs=1, obs_dim=5, action_dim=2, device=torch.device("cpu"))
    kwargs[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        module.RobotRolloutBuffer(**kwargs)


def _values():
    return {"environment": {"num_envs": 2, "num_neighbors": 0, "max_episode_steps": 1},
            "network": {"hidden_dims": [8], "vector_hidden_dims": [8], "actor_hidden_dims": [8], "critic_hidden_dims": [8]},
            "algorithm": {"ppo_epochs": 1, "minibatch_size": 4},
            "training": {"rollout_steps": 2, "total_steps": 4, "save_interval": 0}}


def test_real_standalone_zero_neighbors_train_save_resume_and_evaluate(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_values()))
    args = NS(seed=42, config=str(config), device="cpu", total_steps=None, output=str(tmp_path / "first"), resume=None)
    first = training.train(args)
    assert first.exists()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 4
    # An already-complete resume still publishes a valid requested output.
    args.resume, args.output = first, str(tmp_path / "resumed")
    resumed = training.train(args)
    assert resumed.exists()
    assert torch.load(resumed, weights_only=False)["step"] == 4
    result = evaluation.evaluate(NS(config=str(config), device="cpu", checkpoint=resumed, num_envs=2, episodes=3))
    assert result["episodes"] == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


@pytest.mark.parametrize("field,value", [("rollout_steps", 0), ("total_steps", 0), ("save_interval", -1)])
def test_invalid_training_loop_bounds_fail_before_pipeline(monkeypatch, tmp_path, field, value):
    values = _values()
    values["training"][field] = value
    monkeypatch.setattr(training, "load_config", lambda path: values)
    monkeypatch.setattr(training, "build_pipeline", lambda *args: pytest.fail("Invalid bounds must not initialize training"))
    with pytest.raises(ValueError):
        training.train(NS(seed=1, config=None, device="cpu", total_steps=None, output=str(tmp_path), resume=None))


def test_zero_evaluation_episodes_has_clear_error_before_checkpoint_load():
    with pytest.raises(ValueError, match="must be positive"):
        evaluation.evaluate(NS(episodes=0, num_envs=2))
