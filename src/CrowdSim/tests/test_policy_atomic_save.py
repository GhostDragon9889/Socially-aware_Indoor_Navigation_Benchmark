import pytest
import torch

from CrowdSim.ppo import ppo_policy as simulation_ppo
from crowdsim_ppo import ppo_policy as standalone_ppo


@pytest.mark.parametrize("module", [simulation_ppo, standalone_ppo], ids=["CrowdSim", "standalone"])
def test_policy_save_interruption_preserves_prior_checkpoint_and_cleans_temporary(monkeypatch, tmp_path, module):
    config = module.RobotPPOConfig(obs_dim=5, vector_obs_dim=5, num_neighbors=0,
        hidden_dims=(8,), vector_hidden_dims=(8,), actor_hidden_dims=(8,), critic_hidden_dims=(8,),
        map_enabled=False, depth_enabled=False)
    trainer = module.RobotPPOTrainer(config, torch.device("cpu"))
    target = tmp_path / "latest.pt"
    trainer.save(target, step=17, extra_state={"curriculum": "original"})
    original = target.read_bytes()
    original_model = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    with torch.no_grad():
        next(trainer.model.parameters()).add_(10)
    def partial_write(state, handle):
        handle.write(b"incomplete archive")
        raise OSError("simulated interrupted serialization")
    monkeypatch.setattr(torch, "save", partial_write)
    with pytest.raises(OSError, match="interrupted serialization"):
        trainer.save(target, step=18, extra_state={"curriculum": "new"})
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
    assert trainer.load(target) == 17
    assert trainer.extra_state == {"curriculum": "original"}
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, original_model[key])
