"""Bounded CPU tensor checks; no SDK/contact/termination experiment."""
import os
import sys
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU checks require explicitly hidden CUDA")

import torch

from nonfoot_contact import _canonical_bodies, nonfoot_net_normal_contact


class ContactPrimitiveCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must stay uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_named_foot_exclusion_and_reordered_body_axis(self):
        names = _canonical_bodies()
        force = torch.zeros((2, 24, 3))
        force[:, names.index("L_Ankle"), 2] = 800
        force[:, names.index("R_Toe"), 0] = 300
        force[1, names.index("Chest"), 0] = 60
        before = force.clone()
        progress = torch.tensor([2, 2])
        result = nonfoot_net_normal_contact(force, progress, body_names=names)
        order = tuple(reversed(range(24)))
        reordered = nonfoot_net_normal_contact(force[:, order], progress,
                                               body_names=tuple(names[index] for index in order))
        self.assertTrue(torch.equal(result.contact_flag, torch.tensor([False, True])))
        self.assertTrue(torch.equal(result.resultant_force_w, torch.tensor([[0., 0., 0.], [60., 0., 0.]])))
        self.assertTrue(torch.equal(result.resultant_force_w, reordered.resultant_force_w))
        self.assertTrue(torch.equal(force, before))

    def test_vector_cancellation_and_strict_threshold(self):
        force = torch.zeros((4, 24, 3))
        force[0, 0, 0], force[0, 1, 0] = 40, -40
        force[1, 0, 0], force[1, 1, 0] = 30, 30
        force[2, 0, :2] = torch.tensor([30., 40.])
        force[3, 0, 0] = 50.01
        result = nonfoot_net_normal_contact(force, torch.full((4,), 2), body_names=_canonical_bodies())
        self.assertTrue(torch.equal(result.magnitude_n[:3], torch.tensor([0., 60., 50.])))
        self.assertTrue(torch.equal(result.contact_flag, torch.tensor([False, True, False, True])))

    def test_progress_guard_outputs_owned_and_rng_unchanged(self):
        force = torch.zeros((3, 24, 3))
        force[:, 0, 0] = 60
        progress = torch.tensor([0, 1, 2])
        rng = torch.random.get_rng_state().clone()
        result = nonfoot_net_normal_contact(force, progress, body_names=_canonical_bodies())
        self.assertTrue(torch.equal(result.above_threshold, torch.ones(3, dtype=torch.bool)))
        self.assertTrue(torch.equal(result.contact_flag, torch.tensor([False, False, True])))
        force.zero_()
        progress.fill_(4)
        self.assertTrue(torch.equal(result.magnitude_n, torch.full((3,), 60.)))
        self.assertTrue(torch.equal(result.progress_eligible, torch.tensor([False, False, True])))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(any(name in sys.modules for name in ("isaaclab", "isaaclab_physx", "warp", "mujoco", "dm_control")))

    def test_invalid_source_shape_names_and_dtype_rejected(self):
        names = _canonical_bodies()
        force, progress = torch.zeros((2, 24, 3)), torch.tensor([2, 2])
        for bad in (force.double(), force[:, :23], torch.full_like(force, float("nan")), torch.zeros((0, 24, 3))):
            with self.assertRaises(ValueError):
                nonfoot_net_normal_contact(bad, progress, body_names=names)
        for bad in (progress.float(), progress[:1], torch.tensor([-1, 2])):
            with self.assertRaises(ValueError):
                nonfoot_net_normal_contact(force, bad, body_names=names)
        for bad in (names[:-1], ("Unknown", *names[1:]), (names[1], *names[1:])):
            with self.assertRaises(ValueError):
                nonfoot_net_normal_contact(force, progress, body_names=bad)


if __name__ == "__main__":
    unittest.main()
