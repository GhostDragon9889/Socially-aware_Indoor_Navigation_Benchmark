"""Bounded synthetic CPU tensor tests; run only after source review approval."""
import math
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("These tests require explicitly hidden CUDA")

import torch

import social_task_observation as task


def fixture():
    roots = torch.tensor([[1., 2.], [1.12, 2.1], [40., -8.]], dtype=torch.float64)
    headings = torch.tensor([math.pi/2, 0., -math.pi/4], dtype=torch.float64)
    egos = torch.tensor([2, 0])
    local = torch.stack((torch.arange(1, 11, dtype=torch.float64)/4, torch.arange(1, 11, dtype=torch.float64)/8), -1)
    yaw = headings[egos]
    cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
    world_delta = torch.stack((cosine*local[:, 0]-sine*local[:, 1], sine*local[:, 0]+cosine*local[:, 1]), -1)
    return dict(roots_xy=roots, root_headings=headings,
                world_velocities=torch.tensor([[1., 2.], [0., 1.], [3., -1.]], dtype=torch.float64),
                actor_ids=torch.tensor([4, 9, 12]), scene_ids=torch.tensor([2, 2, 7]), ego_rows=egos,
                trajectory_world_xy=roots[egos, None]+world_delta,
                task_heading=yaw.clone(), trajectory_offsets_s=torch.arange(1, 11, dtype=torch.float64)/2,
                grid_center_xy=roots[egos]+torch.tensor([[.1, .2], [.2, -.1]], dtype=torch.float64),
                grid_heading=torch.tensor([0., math.pi/4], dtype=torch.float64),
                height_reference_world_z=torch.tensor([.2, .9], dtype=torch.float64),
                include_ego=False, device="cpu", ego_chunk_size=1)


class AssemblyCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_single_batched_sampler_layout_closed_form_and_no_mutation(self):
        inputs = fixture()
        copies = {key: value.clone() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        rng = torch.random.get_rng_state().clone()
        calls = []

        def sample(grid):
            self.assertEqual(grid.shape, (2, 64, 64, 2))
            self.assertEqual(grid.dtype, torch.float64)
            self.assertEqual(grid.device, torch.device("cpu"))
            calls.append(grid.clone())
            return .3+.005*grid[..., 0]+.01*grid[..., 1]

        result, owners = task.assemble_raw_task(**inputs, terrain_sampler=sample)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.shape, (2, 12308))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(owners.dtype, torch.int64)
        local = torch.stack((torch.arange(1, 11)/4, torch.arange(1, 11)/8), -1)
        torch.testing.assert_close(result[:, :20].reshape(2, 10, 2), local.expand(2, -1, -1), atol=1e-6, rtol=0)
        # Explicit grid heading differs from root/task heading; endpoints use grid frame.
        c, s = math.cos(math.pi/4), math.sin(math.pi/4)
        expected_corner = inputs["grid_center_xy"][1]+torch.tensor([-2*c+2*s, -2*s-2*c])
        torch.testing.assert_close(calls[0][1, 0, 0], expected_corner, atol=1e-7, rtol=0)
        raster = result[:, 20:].reshape(2, 64, 64, 3)
        self.assertEqual(int(owners[0, 0, 0]), -1)
        static = .3+.005*calls[0][0, 0, 0, 0]+.01*calls[0][0, 0, 0, 1]
        vx = -(3*math.cos(-math.pi/4)-math.sin(-math.pi/4))
        vy = -(-3*math.sin(-math.pi/4)-math.cos(-math.pi/4))
        expected = torch.tensor([float((.2-static).clamp(-3, 3))*5, max(-3, min(3, vx))*5, max(-3, min(3, vy))*5])
        torch.testing.assert_close(raster[0, 0, 0], expected, atol=2e-6, rtol=0)
        self.assertTrue(bool((owners[0] == -1).all()))  # Lone actor; no peer from other scene.
        for key, original in copies.items():
            self.assertTrue(torch.equal(inputs[key], original), key)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_mirror_and_candidate_only_prefix(self):
        inputs = fixture()
        original, _ = task.assemble_raw_task(**inputs, terrain_sampler=lambda grid: torch.zeros(grid.shape[:-1], dtype=grid.dtype))
        before = original.clone()
        mirrored = task.mirror_raw_task(original, device="cpu")
        expected_prefix = original[:, :20].reshape(-1, 10, 2).clone()
        expected_prefix[..., 1] *= -1
        expected_map = original[:, 20:].reshape(-1, 64, 64, 3).flip((2,))
        expected_map[..., 2] *= -1
        self.assertTrue(torch.equal(mirrored[:, :20], expected_prefix.flatten(1)))
        self.assertTrue(torch.equal(mirrored[:, 20:], expected_map.flatten(1)))
        self.assertTrue(torch.equal(task.mirror_raw_task(mirrored, device="cpu"), original))
        candidate = inputs["trajectory_world_xy"]+torch.tensor([.1, -.2], dtype=torch.float64)
        candidate_before = candidate.clone()
        revised = task.replace_candidate_trajectory(original, candidate, inputs["roots_xy"][inputs["ego_rows"]],
                                                   inputs["task_heading"], inputs["trajectory_offsets_s"], device="cpu")
        self.assertTrue(torch.equal(revised[:, 20:], original[:, 20:]))
        self.assertFalse(torch.equal(revised[:, :20], original[:, :20]))
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(candidate, candidate_before))
        self.assertNotEqual(original.data_ptr(), revised.data_ptr())
        self.assertNotEqual(original.data_ptr(), mirrored.data_ptr())

    def test_pelvis_projected_x_heading_and_rejection(self):
        yaw, pitch, roll = .7, .4, -.3
        cy, sy, cp, sp, cr, sr = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2), math.cos(roll/2), math.sin(roll/2)
        q = torch.tensor([[sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]], dtype=torch.float64)
        observed = task.heading_from_pelvis_xyzw(torch.cat((q, -q)), device="cpu")
        torch.testing.assert_close(observed, torch.full((2,), yaw, dtype=torch.float64), atol=1e-12, rtol=0)
        singular = torch.tensor([[0., math.sqrt(.5), 0., math.sqrt(.5)]], dtype=torch.float64)
        for invalid in (q*2, torch.zeros_like(q), q*float("nan"), singular):
            with self.assertRaises(ValueError):
                task.heading_from_pelvis_xyzw(invalid, device="cpu")

    def test_contract_rejects_invalid_sources_before_sampler(self):
        calls = []

        def sample(grid):
            calls.append(True)
            return torch.zeros(grid.shape[:-1], dtype=grid.dtype)

        for key, value in (
            ("trajectory_world_xy", fixture()["trajectory_world_xy"].float()),
            ("task_heading", torch.zeros(2, dtype=torch.bool)),
            ("trajectory_offsets_s", torch.arange(10, dtype=torch.float64)/2),
            ("world_velocities", torch.full((3, 2), float("inf"), dtype=torch.float64)),
            ("actor_ids", torch.tensor([4, 4, 12])),
            ("ego_rows", torch.tensor([2, 2])),
            ("grid_center_xy", torch.zeros((2, 2), dtype=torch.float64, device="meta")),
            ("device", "cuda:0"),
            ("include_ego", None),
        ):
            inputs = fixture()
            inputs[key] = value
            with self.assertRaises(ValueError, msg=key):
                task.assemble_raw_task(**inputs, terrain_sampler=sample)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            task.assemble_raw_task(**fixture(), terrain_sampler=lambda grid: torch.zeros((2, 64, 64), device="meta"))
        with self.assertRaises(ValueError):
            task.assemble_raw_task(**fixture(), terrain_sampler=lambda grid: torch.full(grid.shape[:-1], float("nan")))

        def mutate(grid):
            grid.add_(1)
            return torch.zeros(grid.shape[:-1], dtype=grid.dtype)

        with self.assertRaises(ValueError):
            task.assemble_raw_task(**fixture(), terrain_sampler=mutate)


if __name__ == "__main__":
    unittest.main()
