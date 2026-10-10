"""CPU fixture gate for the explicit full-epoch Pop-Art experiment.

No AMASS, SDK or physics is executed. Algorithm counters are fixture exposure.
The official PACER loop runs all mini_epochs with PPO clipping; the local
target_kl early stop is an additional choice, so this candidate sets it None.
Source: https://github.com/nv-tlabs/pacer/blob/main/pacer/learning/amp_continuous.py
"""
from dataclasses import asdict
import unittest

import torch

from CrowdSim.humanoid.runtime import social_popart_training

GEOMETRY = {"grid_center": "root", "grid_heading": "root",
            "height_reference": "static_center_ground_z", "include_ego": False}


class FullEpochFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.module = social_popart_training.load_trainer("_test_social_full_epochs")

    def test_complete_six_epochs_and_strict_metadata_without_KL_stop(self):
        module = self.module
        config = module.SocialTrainingConfig(actor_count=8, **GEOMETRY,
            target_kl=None, update_epochs=6, minibatch_size=8,
            discriminator_updates_per_update=1, discriminator_batch_size=16,
            discriminator_replay_capacity=32)
        trainer = module.SocialAMPTrainer.prepare(config, device="cpu")
        self_obs = torch.zeros(2, 8, 357)
        task_obs = torch.zeros(2, 8, 12308)
        action = {key: value.unflatten(0, (2, 8)) for key, value in
                  trainer.act(self_obs.flatten(0, 1), task_obs.flatten(0, 1)).items()}
        rollout = {"self_obs": self_obs, "task_obs": task_obs,
            "amp_history": torch.zeros(2, 8, 10, 195), "actions": action["actions"],
            "applied_actions": action["applied_actions"], "log_prob": action["log_prob"],
            "values": action["values"], "next_values": action["values"].clone(),
            "rewards": torch.full((2, 8), .5), "terminated": torch.zeros(2, 8, dtype=torch.bool),
            "truncated": torch.zeros(2, 8, dtype=torch.bool)}
        metrics = trainer.update(rollout, torch.zeros(16, 10, 195), torch.zeros(16, 69))
        self.assertEqual(metrics["policy_epochs_started"], 6)
        self.assertEqual(metrics["optimizer_minibatches"], 12)
        self.assertFalse(metrics["policy_target_kl_stopped"])
        self.assertEqual(metrics["value_normalization"]["new_count"], 16)
        self.assertTrue(all(torch.isfinite(value).all() for value in trainer.model.state_dict().values()))
        state = {key: None for key in module.CHECKPOINT_KEYS}
        state.update(format=social_popart_training.FORMAT, version=1, config=asdict(config), provenance={},
                     social_training_contract=module._contract(config))
        self.assertIsNone(module.SocialAMPTrainer._admit_metadata(state).target_kl)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
