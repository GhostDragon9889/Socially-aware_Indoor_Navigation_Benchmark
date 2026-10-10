"""Bounded synthetic CPU cross-backend checks; never a policy/SDK/GPU test."""

import hashlib
import os
from pathlib import Path
import random
import types
import unittest

import numpy as np
import torch

from social_raster_torch import NUMPY_CONTRACT_SHA, TASK_DIM, make_grid, mirror, render


def reference():
    path = (
        Path(__file__).resolve().parent.parent
        / "social_raster_staging_v1/social_raster.py"
    )
    source, info = path.read_bytes(), path.stat()
    if (
        path.is_symlink()
        or not path.is_file()
        or info.st_nlink != 1
        or info.st_mode & 0o222
        or len(source) != 7218
        or hashlib.sha256(source).hexdigest() != NUMPY_CONTRACT_SHA
    ):
        raise ValueError(
            "Exact frozen NumPy source required before loading the reference"
        )
    module = types.ModuleType("pinned_numpy_raster_reference")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def inputs():
    return dict(
        roots_xy=np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        root_headings=np.array([np.pi / 2, 0.0, 0.0]),
        world_velocities=np.array([[0.2, 0.4], [1.2, -0.6], [2.0, 2.0]]),
        actor_ids=np.array([20, 7, 8]),
        scene_ids=np.array([3, 3, 4]),
        ego_rows=np.array([0]),
        grid_center_xy=np.zeros((1, 2)),
        grid_heading=np.zeros(1),
        static_heights=np.full((1, 64, 64), 2.0),
        height_reference_world_z=np.array([2.0]),
    )


class TensorRasterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("Only explicit CUDA-hidden CPU verification is authorized")
        cls.numpy = reference()

    def compare(self, data, include_ego=False, chunk=16):
        tensors = {key: torch.from_numpy(value.copy()) for key, value in data.items()}
        expected, expected_ids = self.numpy.render(**data, include_ego=include_ego)
        actual, ids = render(
            **tensors, include_ego=include_ego, device="cpu", ego_chunk_size=chunk
        )
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertEqual(ids.device, actual.device)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int64)
        np.testing.assert_array_equal(ids.numpy(), expected_ids)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=0)
        for key, tensor in tensors.items():
            np.testing.assert_array_equal(tensor.numpy(), data[key])
        return actual, ids

    def test_closed_form_self_background_root_velocity_frame_and_rng(self):
        data = inputs()
        python_rng, numpy_rng, torch_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        self.assertEqual(TASK_DIM, 12308)
        grid = make_grid(
            torch.from_numpy(data["grid_center_xy"]),
            torch.from_numpy(data["grid_heading"]),
            device="cpu",
        )
        np.testing.assert_array_equal(
            grid.numpy(),
            self.numpy.make_grid(data["grid_center_xy"], data["grid_heading"]),
        )
        raster, ids = self.compare(data)
        self.assertEqual(ids[0, 48, 32], 7)
        self.assertEqual(ids[0, 16, 32], -1)
        self.assertEqual(ids[0, 32, 32], -1)
        np.testing.assert_allclose(
            raster[0, 48, 32].numpy(), [-8.5, -5.0, -5.0], atol=1e-6
        )
        np.testing.assert_allclose(
            raster[0, 16, 32].numpy(), [0.0, -2.0, 1.0], atol=1e-6
        )
        raster, ids = self.compare(data, include_ego=True)
        self.assertEqual(ids[0, 32, 32], 20)
        np.testing.assert_allclose(
            raster[0, 32, 32].numpy(), [-8.5, 0.0, 0.0], atol=1e-6
        )
        self.assertEqual(random.getstate(), python_rng)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy_rng[0])
        np.testing.assert_array_equal(now[1], numpy_rng[1])
        self.assertEqual(now[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_overlap_nearest_stable_id_row_permutation_and_height_once(self):
        data = inputs()
        data.update(
            roots_xy=np.array([[0.0, -1.5], [0.0, 0.0], [0.0, 0.0]]),
            root_headings=np.zeros(3),
            scene_ids=np.zeros(3, dtype=np.int64),
            actor_ids=np.array([100, 18, 7]),
        )
        raster, ids = self.compare(data)
        self.assertEqual(ids[0, 32, 32], 7)
        self.assertEqual(raster[0, 32, 32, 0], -8.5)
        data["roots_xy"][1:, 0] = [-0.125, 0.125]
        raster, ids = self.compare(data)
        self.assertEqual(ids[0, 30, 32], 18)
        order = np.array([2, 0, 1])
        shuffled = dict(data, ego_rows=np.array([1]))
        for key in (
            "roots_xy",
            "root_headings",
            "world_velocities",
            "actor_ids",
            "scene_ids",
        ):
            shuffled[key] = data[key][order]
        actual, owner = self.compare(shuffled, chunk=1)
        self.assertTrue(torch.equal(actual, raster))
        self.assertTrue(torch.equal(owner, ids))

    def test_rigid_covariance_reflection_mirror_twice(self):
        data = inputs()
        data.update(
            root_headings=np.array([0.6, -0.3, 0.1]),
            grid_center_xy=np.array([[0.1, -0.2]]),
            grid_heading=np.array([-0.45]),
            static_heights=np.arange(4096).reshape(1, 64, 64) / 4096 + 0.1,
            height_reference_world_z=np.array([0.8]),
        )
        raster, ids = self.compare(data, include_ego=True)
        angle = 0.71
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shifted = dict(data)
        for key in ("roots_xy", "grid_center_xy"):
            shifted[key] = data[key] @ rotation.T + [7.0, -3.0]
        shifted["world_velocities"] = data["world_velocities"] @ rotation.T
        shifted["root_headings"] = data["root_headings"] + angle
        shifted["grid_heading"] = data["grid_heading"] + angle
        actual, owner = self.compare(shifted, include_ego=True)
        self.assertTrue(torch.equal(owner, ids))
        torch.testing.assert_close(actual, raster, atol=2e-6, rtol=0)
        reflected = dict(data)
        for key in ("roots_xy", "grid_center_xy", "world_velocities"):
            reflected[key] = data[key] * [1.0, -1.0]
        reflected["root_headings"] = -data["root_headings"]
        reflected["grid_heading"] = -data["grid_heading"]
        reflected["static_heights"] = data["static_heights"][:, :, ::-1].copy()
        actual, owner = self.compare(reflected, include_ego=True)
        expected, expected_ids = mirror(raster, ids, device="cpu")
        self.assertTrue(torch.equal(owner, expected_ids))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
        twice, twice_ids = mirror(expected, expected_ids, device="cpu")
        self.assertTrue(torch.equal(twice, raster))
        self.assertTrue(torch.equal(twice_ids, ids))

    def test_inclusive_boundary_fp32_input_rounding_clipping_and_lonely_ego(self):
        data = inputs()
        data.update(
            roots_xy=np.array([[-1.5, -1.5], [1.75, 1.5], [20.0, 20.0]]),
            root_headings=np.zeros(3),
        )
        _, ids = self.compare(data)
        self.assertEqual(ids[0, 63, 63], 7)  # Exact half-size boundary included.
        data["roots_xy"][1, 0] = np.nextafter(1.75, -np.inf)
        _, ids = self.compare(data)
        self.assertEqual(ids[0, 63, 63], -1)
        rounded = {
            key: value.astype(np.float32) if value.dtype.kind == "f" else value
            for key, value in data.items()
        }
        _, ids = self.compare(rounded)
        self.assertEqual(ids[0, 63, 63], 7)  # Input precision changed the geometry.
        data = inputs()
        data["scene_ids"] = np.array([0, 1, 2])
        data["world_velocities"][0] = [10.0, -10.0]
        data["root_headings"][0] = 0.0
        data["static_heights"][:] = 100.0
        raster, ids = self.compare(data)
        self.assertTrue(bool((ids == -1).all()))
        np.testing.assert_array_equal(raster[0, 0, 0].numpy(), [-15.0, -15.0, 15.0])

    def test_invalid_shapes_ids_devices_dtypes_and_overflow(self):
        data = {key: torch.from_numpy(value) for key, value in inputs().items()}
        invalid = [
            ("actor_ids", torch.tensor([1, 1, 2])),
            ("actor_ids", torch.tensor([True, False, True])),
            ("ego_rows", torch.tensor([0, 0])),
            ("ego_rows", torch.tensor([3])),
            ("ego_rows", torch.empty(0, dtype=torch.int64)),
            ("root_headings", torch.zeros(3, 1)),
            ("grid_heading", torch.tensor([torch.nan])),
            ("world_velocities", torch.full((3, 2), torch.inf)),
            ("root_headings", torch.zeros(3, dtype=torch.complex64)),
            ("static_heights", torch.zeros(1, 64, 63)),
            ("grid_heading", torch.empty(1, device="meta")),
            ("roots_xy", torch.empty(0, 2)),
            ("height_reference_world_z", torch.tensor([1e308], dtype=torch.float64)),
        ]
        data["static_heights"] = torch.full((1, 64, 64), -1e308, dtype=torch.float64)
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                render(**dict(data, **{key: value}), include_ego=False, device="cpu")
        with self.assertRaises(ValueError):
            render(**data, include_ego=1, device="cpu")
        with self.assertRaises(ValueError):
            render(**data, include_ego=False, device="cuda:0")
        with self.assertRaises(ValueError):
            render(**data, include_ego=False, device="cpu", ego_chunk_size=0)
        with self.assertRaises(ValueError):
            mirror(torch.zeros(1, 64, 64, 3), torch.full((1, 64, 64), -2), device="cpu")
        self.assertFalse(torch.cuda.is_initialized())

    def test_mixed_256_row_scene_layout_ego_order_and_chunk_invariance(self):
        roots = np.empty((256, 2))
        scenes = np.empty(256, np.int64)
        for group in range(64):
            rows = [2 * group, 2 * group + 1, 128 + 2 * group, 129 + 2 * group]
            scenes[rows] = group
            roots[rows] = np.array([[0.0, 0.0], [0.8, 0.0], [0.0, 0.8], [0.8, 0.8]]) + [
                group * 10.0,
                0.0,
            ]
        egos = np.array([129, 0, 3, 128, 255, 254])
        data = dict(
            roots_xy=roots,
            root_headings=np.zeros(256),
            world_velocities=np.zeros((256, 2)),
            actor_ids=np.arange(256) + 100,
            scene_ids=scenes,
            ego_rows=egos,
            grid_center_xy=roots[egos].copy(),
            grid_heading=np.zeros(6),
            static_heights=np.zeros((6, 64, 64)),
            height_reference_world_z=np.zeros(6),
        )
        raster, ids = self.compare(data, chunk=1)
        actual, owner = self.compare(data, chunk=16)
        self.assertTrue(torch.equal(actual, raster))
        self.assertTrue(torch.equal(owner, ids))
        order = np.arange(255, -1, -1)
        shuffled = dict(data, ego_rows=255 - egos)
        for key in (
            "roots_xy",
            "root_headings",
            "world_velocities",
            "actor_ids",
            "scene_ids",
        ):
            shuffled[key] = data[key][order].copy()
        actual, owner = self.compare(shuffled)
        self.assertTrue(torch.equal(actual, raster))
        self.assertTrue(torch.equal(owner, ids))

    def test_256_scenes_four_actors_complete_1024_egos(self):
        count = 256 * 4
        scenes = np.repeat(np.arange(256), 4)
        roots = np.tile(
            [[0.0, 0.0], [0.8, 0.0], [0.0, 0.8], [0.8, 0.8]], (256, 1)
        ) + np.stack((scenes * 10.0, scenes * 0.0), -1)
        data = dict(
            roots_xy=roots,
            root_headings=np.zeros(count),
            world_velocities=np.tile(
                [[0.2, 0.4], [1.2, -0.6], [0.7, 0.3], [-0.4, 0.9]], (256, 1)
            ),
            actor_ids=np.arange(count) * 3 + 100,
            scene_ids=scenes,
            ego_rows=np.arange(count),
            grid_center_xy=roots.copy(),
            grid_heading=np.zeros(count),
            static_heights=np.zeros((count, 64, 64)),
            height_reference_world_z=np.zeros(count),
        )
        raster, ids = self.compare(data)
        self.assertEqual(tuple(raster.shape), (1024, 64, 64, 3))
        self.assertEqual(tuple(ids.shape), (1024, 64, 64))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
