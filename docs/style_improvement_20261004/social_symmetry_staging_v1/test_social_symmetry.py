"""Four CPU tensor checks; no policy, SDK, weights or prior-suite execution."""
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU tests require explicitly hidden CUDA")

import torch

import social_symmetry as symmetry


class SocialSymmetryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_coupled_closed_form_body_vector_parity_and_HWC(self):
        self_obs = torch.arange(357, dtype=torch.float32)[None]
        task_obs = (torch.arange(12308, dtype=torch.float32)%31-15)[None]
        mirrored_self, mirrored_task = symmetry.mirror_raw_observations(self_obs, task_obs, device="cpu")
        source, _ = symmetry._dependencies()
        left, right = source.PACER_BODY_NAMES.index("L_Hip"), source.PACER_BODY_NAMES.index("R_Hip")
        polar, axial = torch.tensor([1., -1., 1.]), torch.tensor([-1., 1., -1.])
        self.assertTrue(torch.equal(mirrored_self[:, :69].reshape(1, 23, 3)[0, left-1], self_obs[:, :69].reshape(1, 23, 3)[0, right-1]*polar))
        self.assertTrue(torch.equal(mirrored_self[:, 69:213].reshape(1, 24, 2, 3)[0, left], self_obs[:, 69:213].reshape(1, 24, 2, 3)[0, right]*polar))
        self.assertTrue(torch.equal(mirrored_self[:, 213:285].reshape(1, 24, 3)[0, left], self_obs[:, 213:285].reshape(1, 24, 3)[0, right]*polar))
        self.assertTrue(torch.equal(mirrored_self[:, 285:].reshape(1, 24, 3)[0, left], self_obs[:, 285:].reshape(1, 24, 3)[0, right]*axial))
        self.assertTrue(torch.equal(mirrored_task[:, :20].reshape(1, 10, 2), task_obs[:, :20].reshape(1, 10, 2)*torch.tensor([1., -1.])))
        raster, reflected = task_obs[:, 20:].reshape(1, 64, 64, 3), mirrored_task[:, 20:].reshape(1, 64, 64, 3)
        self.assertTrue(torch.equal(reflected[:, :, 0], raster[:, :, 63]*torch.tensor([1., 1., -1.])))
        actions = torch.arange(69, dtype=torch.float32)[None]
        reflected_actions = symmetry.mirror_actions(actions)
        for axis, sign in (("x", -1), ("y", 1), ("z", -1)):
            li, ri = source.PACER_DOF_NAMES.index("L_Hip_"+axis), source.PACER_DOF_NAMES.index("R_Hip_"+axis)
            self.assertEqual(float(reflected_actions[0, li]), float(actions[0, ri])*sign)

    def test_involution_and_input_rng_preservation(self):
        own = torch.arange(2*357, dtype=torch.float32).reshape(2, 357)/100
        task = (torch.arange(2*12308, dtype=torch.float32).reshape(2, 12308)%31-15)
        action = torch.arange(2*69, dtype=torch.float32).reshape(2, 69)/100
        before = (own.clone(), task.clone(), action.clone())
        rng = torch.random.get_rng_state().clone()
        mirrored = symmetry.mirror_raw_observations(own, task)
        twice = symmetry.mirror_raw_observations(*mirrored)
        self.assertTrue(torch.equal(twice[0], own))
        self.assertTrue(torch.equal(twice[1], task))
        self.assertTrue(torch.equal(symmetry.mirror_actions(symmetry.mirror_actions(action)), action))
        for value, original in zip((own, task, action), before):
            self.assertTrue(torch.equal(value, original))
        self.assertNotEqual(own.data_ptr(), mirrored[0].data_ptr())
        self.assertNotEqual(task.data_ptr(), mirrored[1].data_ptr())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_both_branch_gradients_and57_mask_coefficient50(self):
        own = (torch.arange(357, dtype=torch.float32)[None]/100).requires_grad_()
        task = (torch.arange(12308, dtype=torch.float32)[None]%31-15).requires_grad_()
        mirrored = symmetry.mirror_raw_observations(own, task)
        (mirrored[0].square().sum()+mirrored[1].square().sum()).backward()
        self.assertTrue(torch.equal(own.grad, 2*own))
        self.assertTrue(torch.equal(task.grad, 2*task))
        mask = symmetry.canonical_action_mask(device="cpu")
        self.assertEqual(int(mask.sum()), 57)
        self.assertTrue(torch.equal(symmetry.mirror_actions(mask[None]).abs()[0], mask))
        mean = torch.zeros((2, 69), requires_grad=True)
        alternate = torch.where(mask[None].bool(), torch.full((2, 69), 2.), torch.full((2, 69), 1000.)).requires_grad_()
        loss = symmetry.effective_symmetry_loss(mean, alternate, mask)
        self.assertTrue(torch.equal(loss, torch.tensor(4.)))
        weighted = 50*loss  # Trainer multiplies exactly once, outside the helper.
        self.assertTrue(torch.equal(weighted, torch.tensor(200.)))
        weighted.backward()
        self.assertTrue(bool((mean.grad[:, mask.bool()].abs() > 0).all()))
        self.assertTrue(bool((alternate.grad[:, mask.bool()].abs() > 0).all()))
        self.assertTrue(bool((mean.grad[:, ~mask.bool()] == 0).all()))
        self.assertTrue(bool((alternate.grad[:, ~mask.bool()] == 0).all()))

    def test_batch_dtype_device_shape_finite_and_mask_rejections(self):
        own, task = torch.zeros((2, 357)), torch.zeros((2, 12308))
        for bad_self, bad_task, device in ((own.double(), task, "cpu"), (own, task[:1], "cpu"),
                                           (own, torch.zeros((2, 1044)), "cpu"), (own, torch.full_like(task, float("nan")), "cpu"),
                                           (own.to("meta"), task, "cpu"), (own, task, "cuda"), (own, task, "cuda:0")):
            with self.assertRaises(ValueError):
                symmetry.mirror_raw_observations(bad_self, bad_task, device=device)
        mask = symmetry.canonical_action_mask(device="cpu")
        altered = mask.clone()
        altered[0], altered[9] = 0., 1.  # Still57; exact physical contract must reject it.
        for invalid in (altered, mask.double(), mask[:68]):
            with self.assertRaises(ValueError):
                symmetry.effective_symmetry_loss(torch.zeros((2, 69)), torch.zeros((2, 69)), invalid)
        with self.assertRaises(ValueError):
            symmetry.mirror_actions(torch.zeros((2, 68)))


if __name__ == "__main__":
    unittest.main()
