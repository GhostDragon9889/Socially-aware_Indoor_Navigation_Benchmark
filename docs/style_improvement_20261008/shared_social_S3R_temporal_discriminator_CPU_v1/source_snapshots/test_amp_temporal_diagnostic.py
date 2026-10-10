"""CPU contract fixtures; synthetic networks are never actual model evidence."""
from copy import deepcopy
import unittest

import torch

from CrowdSim.humanoid.analysis import amp_temporal_diagnostic as diagnostic
from CrowdSim.humanoid.training import trainer as original


class TemporalDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        d = original._mlp(2019, (1024, 512), 1)
        cfg = {"history_frames": 10, "amp_frame_dim": 195, "discriminator_input": "state_action",
            "amp_normalization": "per_frame", "discriminator_layers": [1024, 512],
            "normalize_observations": True, "normalization_clip": 5., "action_mask": [1] * 69,
            "amp_probability_epsilon": .0001, "amp_reward_scale": 2.}
        cls.state = {"config": cfg, "model": {"discriminator." + key: value for key, value in d.state_dict().items()},
                     "normalizers": {"amp_normalizer": original.RunningNormalizer(195).state_dict()},
                     "provenance": {"initialization": {"synthetic_fixture": True}}}
        cls.history = torch.randn(8, 10, 195)
        cls.actions = torch.randn(8, 69).clamp(-1, 1)

    def test_past_permutations_keep_current_action_and_complete_frame_multiset(self):
        data = diagnostic.interventions(self.history, self.actions)
        for name, order in diagnostic.ORDERS.items():
            with self.subTest(name=name):
                self.assertEqual(order[0], 0)
                self.assertEqual(sorted(order), list(range(10)))
                self.assertTrue(torch.equal(data[name][0][:, 0], self.history[:, 0]))
                self.assertTrue(torch.equal(data[name][1], self.actions))
        self.assertTrue(torch.equal(data["cyclic_action_mismatch"][0], self.history))
        self.assertTrue(torch.equal(data["cyclic_action_mismatch"][1], self.actions.roll(1, 0)))

    def test_full_D_remains_order_sensitive_with_shared_per_frame_moments(self):
        proxy = diagnostic.load_discriminator(self.state)
        summary, _ = diagnostic.measure(proxy, self.history, self.actions, batch_size=4)
        self.assertEqual(summary["identity"]["mean_absolute_paired_logit_delta"], 0.)
        self.assertGreater(summary["reverse_past_keep_current"]["mean_absolute_paired_logit_delta"], 1e-4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_scores_use_exact_original_logits_and_reward_with_frozen_moments(self):
        proxy = diagnostic.load_discriminator(self.state)
        moments = {key: value.clone() for key, value in proxy.amp_normalizer.state_dict().items()}
        _, scores = diagnostic.measure(proxy, self.history, self.actions, batch_size=8)
        with torch.no_grad():
            normalized, actions = proxy._amp_inputs(self.history, self.actions)
            expected = original.PaperAMPNetwork.disc_logits(proxy.model, normalized, actions)
            reward = original.PaperAMPTrainer.amp_reward(proxy, self.history, self.actions)
        self.assertTrue(torch.equal(scores["identity"]["logits"], expected))
        self.assertTrue(torch.equal(scores["identity"]["rewards"], reward))
        self.assertTrue(all(torch.equal(value, moments[key]) for key, value in proxy.amp_normalizer.state_dict().items()))

    def test_saved_family_and_complete_D_keys_are_required(self):
        for mode in ("RMS_family", "missing_weight"):
            with self.subTest(mode=mode):
                wrong = deepcopy(self.state)
                if mode == "RMS_family":
                    wrong["config"]["amp_normalization"] = "per_history_coordinate"
                    error = ValueError
                else:
                    wrong["model"].pop("discriminator.4.weight")
                    error = RuntimeError
                with self.assertRaises(error):
                    diagnostic.load_discriminator(wrong)

    def test_nonfinite_and_ineffective_physical_actions_rejected_by_original_contract(self):
        proxy = diagnostic.load_discriminator(self.state)
        wrong = self.history.clone()
        wrong[0, 3, 5] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnostic.measure(proxy, wrong, self.actions)
        proxy.action_mask[0] = 0
        with self.assertRaisesRegex(ValueError, "frozen coordinates zero"):
            diagnostic.measure(proxy, self.history, self.actions)

    def test_degenerate_population_and_noninteger_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete materialized"):
            diagnostic.interventions(self.history[:1], self.actions[:1])
        with self.assertRaisesRegex(ValueError, "integer CPU"):
            diagnostic.measure(diagnostic.load_discriminator(self.state), self.history, self.actions, batch_size=True)


if __name__ == "__main__":
    unittest.main()
