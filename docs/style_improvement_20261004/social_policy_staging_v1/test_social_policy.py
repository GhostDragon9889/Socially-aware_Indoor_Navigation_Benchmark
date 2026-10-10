"""Future bounded CPU tests; authored only, not executed during live training."""
import unittest

import torch

from social_policy import (
    ACTION_DIM,
    ENCODED_TASK_DIM,
    POLICY_INPUT_DIM,
    SELF_DIM,
    TASK_DIM,
    SocialAMPActorCritic,
    SocialTaskEncoder,
)


class SocialPolicyTests(unittest.TestCase):
    def test_shapes_shared_encoder_and_separate_heads(self):
        model = SocialAMPActorCritic(task_encoder_sharing="shared")
        self_obs, task_obs = torch.zeros(2, SELF_DIM), torch.zeros(2, TASK_DIM)
        calls = []
        hook = model.task_encoder.register_forward_hook(lambda *unused: calls.append(1))
        distribution, value = model.policy(self_obs, task_obs)
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(distribution.mean.shape, (2, ACTION_DIM))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(model.actor[0].in_features, POLICY_INPUT_DIM)
        self.assertEqual(model.critic[0].in_features, POLICY_INPUT_DIM)
        self.assertIsNot(model.actor[0].weight, model.critic[0].weight)
        self.assertEqual(model.task_encoder(task_obs).shape, (2, ENCODED_TASK_DIM))
        self.assertFalse(model.log_std.requires_grad)

    def test_hwc_packing_and_trajectory_identity(self):
        encoder = SocialTaskEncoder()
        task = torch.zeros(1, TASK_DIM)
        task[:, :20] = torch.arange(20)
        image = task[:, 20:].reshape(1, 64, 64, 3)
        image[0, 3, 7] = torch.tensor((1., 2., 3.))
        seen = []
        hook = encoder.cnn[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        encoded = encoder(task)
        hook.remove()
        self.assertEqual(seen[0].shape, (1, 3, 64, 64))
        self.assertTrue(torch.equal(seen[0][0, :, 3, 7], torch.tensor((1., 2., 3.))))
        self.assertTrue(torch.equal(encoded[:, -20:], task[:, :20]))

    def test_value_gradient_and_incompatible_inputs(self):
        model = SocialAMPActorCritic(task_encoder_sharing="shared")
        task = torch.randn(1, TASK_DIM, requires_grad=True)
        value = model.value(torch.randn(1, SELF_DIM), task)
        value.sum().backward()
        self.assertIsNotNone(task.grad)
        self.assertTrue(torch.isfinite(task.grad).all())
        self.assertTrue(torch.any(task.grad[:, :20] != 0))
        with self.assertRaises(ValueError):
            model.policy(torch.zeros(1, SELF_DIM), torch.zeros(1, 1044))
        with self.assertRaises(ValueError):
            model.load_state_dict({}, strict=False)
        with self.assertRaises(ValueError):
            SocialAMPActorCritic(task_encoder_sharing="separate")


if __name__ == "__main__":
    unittest.main()
