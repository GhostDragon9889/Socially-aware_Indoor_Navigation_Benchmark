"""CPU numerical contracts; no SDK or locomotion-quality evidence."""
import unittest

import torch

from CrowdSim.humanoid.training.value_normalization import preserve_unnormalized_critic


def rms(mean, variance, count):
    return {"mean": torch.tensor([mean], dtype=torch.float64),
            "variance": torch.tensor([variance], dtype=torch.float64),
            "count": torch.tensor(count, dtype=torch.float64)}


def raw_value(head, features, state):
    return head(features).double() * (state["variance"] + 1e-5).sqrt() + state["mean"]


class ScalarPopArtTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = torch.nn.Linear(8, 1)
        self.features = torch.randn(64, 8)

    def test_preserves_raw_values_for_shift_and_scale_change(self):
        old, new = rms(2., 4., 100), rms(-10., 400., 200)
        expected = raw_value(self.head, self.features, old)
        previous = {key: value.clone() for key, value in old.items()}
        current = {key: value.clone() for key, value in new.items()}
        declaration = preserve_unnormalized_critic(self.head, old, new)
        torch.testing.assert_close(raw_value(self.head, self.features, new), expected, rtol=2e-6, atol=2e-6)
        for key in old:
            self.assertTrue(torch.equal(old[key], previous[key]))
            self.assertTrue(torch.equal(new[key], current[key]))
        self.assertFalse(declaration["physical_quality_verified"])

    def test_large_scale_shift_is_quantified_in_float32(self):
        old = rms(48332.437018675664, 17708344474.38586, 16375808)
        new = rms(48200., 17699000000., 16384000)
        expected = raw_value(self.head, self.features, old)
        preserve_unnormalized_critic(self.head, old, new)
        actual = raw_value(self.head, self.features, new)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=.03)

    def test_preserves_existing_Adam_state_and_parameter_identity(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=2e-5)
        self.head(self.features).square().mean().backward()
        optimizer.step()
        identities = [id(parameter) for parameter in self.head.parameters()]
        state = {parameter: {key: value.clone() for key, value in fields.items()}
                 for parameter, fields in optimizer.state.items()}
        preserve_unnormalized_critic(self.head, rms(0., 1., 0), rms(1., 2., 64))
        self.assertEqual(identities, [id(parameter) for parameter in self.head.parameters()])
        for parameter, fields in state.items():
            for key, value in fields.items():
                self.assertTrue(torch.equal(optimizer.state[parameter][key], value))

    def test_invalid_statistics_leave_both_parameters_unchanged(self):
        old, new = rms(0., 1., 1), rms(0., -1., 2)
        weight, bias = self.head.weight.detach().clone(), self.head.bias.detach().clone()
        with self.assertRaises(ValueError):
            preserve_unnormalized_critic(self.head, old, new)
        self.assertTrue(torch.equal(self.head.weight, weight))
        self.assertTrue(torch.equal(self.head.bias, bias))

    def test_rejects_actor_head_and_absent_new_exposure(self):
        with self.assertRaises(TypeError):
            preserve_unnormalized_critic(torch.nn.Linear(8, 69), rms(0., 1., 1), rms(0., 1., 2))
        with self.assertRaises(ValueError):
            preserve_unnormalized_critic(self.head, rms(0., 1., 1), rms(0., 1., 1))


if __name__ == "__main__":
    unittest.main()
