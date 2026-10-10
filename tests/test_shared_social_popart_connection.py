"""CPU fixture verification of the opt-in full Social trainer connection.

No AMASS, SDK, real physics or locomotion outcome is exercised here.
"""
from dataclasses import asdict
import unittest

import torch

from CrowdSim.humanoid.runtime import social_popart_training as adapter

GEOMETRY = {"grid_center": "root", "grid_heading": "root",
            "height_reference": "static_center_ground_z", "include_ego": False}


class SocialPopArtConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.module = adapter.load_trainer("_test_shared_popart_trainer")

    def test_metadata_declares_new_format_and_rejects_legacy(self):
        module = self.module
        config = module.SocialTrainingConfig(actor_count=8, **GEOMETRY)
        state = {name: None for name in module.CHECKPOINT_KEYS}
        state.update(format=adapter.FORMAT, version=1, config=asdict(config), provenance={},
                     social_training_contract=module._contract(config))
        admitted = module.SocialAMPTrainer._admit_metadata(state)
        self.assertEqual(admitted.to_document(), config.to_document())
        self.assertEqual(state["social_training_contract"]["value_normalization"]["strategy"],
                         "scalar_popart_output_preservation_v1")
        state["format"] = "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"
        with self.assertRaisesRegex(ValueError, "schema"):
            module.SocialAMPTrainer._admit_metadata(state)

    def test_full_fixture_update_applies_preservation_and_correct_counts(self):
        module = self.module
        config = module.SocialTrainingConfig(actor_count=8, **GEOMETRY, update_epochs=1, minibatch_size=16,
            discriminator_updates_per_update=1, discriminator_batch_size=16, discriminator_replay_capacity=32)
        trainer = module.SocialAMPTrainer.prepare(config, device="cpu")
        # Fixtures check numerical connection only, never simulated transitions.
        self_obs = torch.zeros(2, 8, 357)
        task_obs = torch.zeros(2, 8, 12308)
        action = {key: value.unflatten(0, (2, 8)) for key, value in
                  trainer.act(self_obs.flatten(0, 1), task_obs.flatten(0, 1)).items()}
        values = action["values"]
        rollout = {"self_obs": self_obs, "task_obs": task_obs, "amp_history": torch.zeros(2, 8, 10, 195),
            "actions": action["actions"], "applied_actions": action["applied_actions"],
            "log_prob": action["log_prob"], "values": values, "next_values": values.clone(),
            "rewards": torch.full((2, 8), .5), "terminated": torch.zeros(2, 8, dtype=torch.bool),
            "truncated": torch.zeros(2, 8, dtype=torch.bool)}
        result = trainer.update(rollout, torch.zeros(16, 10, 195), torch.zeros(16, 69))
        preservation = result["value_normalization"]
        self.assertEqual(preservation["old_count"], 0)
        self.assertEqual(preservation["new_count"], 16)
        self.assertFalse(preservation["actor_encoder_modified"])
        self.assertFalse(preservation["physical_quality_verified"])
        self.assertEqual(trainer.update_count, 1)
        self.assertEqual(trainer.physical_transition_count, 16)
        self.assertTrue(all(torch.isfinite(value).all() for value in trainer.model.state_dict().values()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
