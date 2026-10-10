"""Fixed geometry checks, not a model or real-scene validation."""
import random
import unittest

import numpy as np

from social_raster import TASK_DIM, make_grid, mirror, render


def inputs():
    return dict(roots_xy=np.array([[0., 0.], [1., 0.], [-1., 0.]]),
                root_headings=np.array([np.pi/2, 0., 0.]),
                world_velocities=np.array([[.2, .4], [1.2, -.6], [2., 2.]]),
                actor_ids=np.array([20, 7, 8]), scene_ids=np.array([3, 3, 4]),
                ego_rows=np.array([0]), grid_center_xy=np.zeros((1, 2)), grid_heading=np.zeros(1),
                static_heights=np.full((1, 64, 64), 2.), height_reference_world_z=np.array([2.]))


class SocialRasterChecks(unittest.TestCase):
    def test_closed_form_grouping_self_and_velocity_frames(self):
        data = inputs()
        before = {key: value.copy() for key, value in data.items()}
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        grid = make_grid(data["grid_center_xy"], data["grid_heading"])
        self.assertEqual(TASK_DIM, 12308)
        np.testing.assert_array_equal(grid[0, 0, 0], [-2., -2.])
        np.testing.assert_array_equal(grid[0, -1, -1], [2., 2.])
        self.assertGreater(grid[0, 1, 0, 0], grid[0, 0, 0, 0])
        self.assertGreater(grid[0, 0, 1, 1], grid[0, 0, 0, 1])
        raster, ids = render(**data, include_ego=False)
        self.assertEqual(raster.dtype, np.float32)
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(ids[0, 48, 32], 7)  # Peer rectangle around world(1,0).
        self.assertEqual(ids[0, 16, 32], -1)  # Other scene's actor is excluded.
        self.assertEqual(ids[0, 32, 32], -1)  # Ego explicitly excluded.
        np.testing.assert_allclose(raster[0, 48, 32], [-8.5, -5., -5.], atol=1e-6)
        np.testing.assert_allclose(raster[0, 16, 32], [0., -2., 1.], atol=1e-6)
        with_ego, ids_with_ego = render(**data, include_ego=True)
        self.assertEqual(ids_with_ego[0, 32, 32], 20)
        np.testing.assert_allclose(with_ego[0, 32, 32], [-8.5, 0., 0.], atol=1e-6)
        self.assertEqual(random.getstate(), python_rng)
        after_rng = np.random.get_state()
        self.assertEqual(after_rng[0], numpy_rng[0])
        np.testing.assert_array_equal(after_rng[1], numpy_rng[1])
        self.assertEqual(after_rng[2:], numpy_rng[2:])
        for name, original in before.items():
            np.testing.assert_array_equal(data[name], original)

    def test_overlap_nearest_tie_permutation_and_height_once(self):
        data = inputs()
        data.update(roots_xy=np.array([[0., -1.5], [0., 0.], [0., 0.]]),
                    root_headings=np.zeros(3), scene_ids=np.zeros(3, dtype=np.int64),
                    actor_ids=np.array([100, 18, 7]), grid_center_xy=np.zeros((1, 2)),
                    static_heights=np.full((1, 64, 64), .5), height_reference_world_z=np.array([.5]))
        raster, ids = render(**data, include_ego=False)
        self.assertEqual(ids[0, 32, 32], 7)  # Coincident roots have exactly equal distance.
        self.assertEqual(raster[0, 32, 32, 0], -8.5)  # One1.7m addition despite overlap.
        data["roots_xy"][1:, 0] = [-.125, .125]
        raster, ids = render(**data, include_ego=False)
        self.assertEqual(ids[0, 30, 32], 18)  # Nearer root wins over smaller ID.
        order = np.array([2, 0, 1])
        permuted = dict(data)
        for key in ("roots_xy", "root_headings", "world_velocities", "actor_ids", "scene_ids"):
            permuted[key] = data[key][order]
        permuted["ego_rows"] = np.array([1])
        actual, actual_ids = render(**permuted, include_ego=False)
        np.testing.assert_array_equal(actual, raster)
        np.testing.assert_array_equal(actual_ids, ids)

    def test_rigid_world_covariance_and_reflection(self):
        data = inputs()
        data.update(root_headings=np.array([.6, -.3, .1]), grid_center_xy=np.array([[.1, -.2]]),
                    grid_heading=np.array([-.45]),
                    static_heights=(np.arange(64*64).reshape(1, 64, 64)/4096 + .1),
                    height_reference_world_z=np.array([.8]))
        raster, ids = render(**data, include_ego=True)
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shifted = dict(data)
        for name in ("roots_xy", "grid_center_xy"):
            shifted[name] = data[name] @ rotation.T + [7., -3.]
        shifted["world_velocities"] = data["world_velocities"] @ rotation.T
        shifted["root_headings"] = data["root_headings"] + angle
        shifted["grid_heading"] = data["grid_heading"] + angle
        actual, actual_ids = render(**shifted, include_ego=True)
        np.testing.assert_array_equal(actual_ids, ids)
        np.testing.assert_allclose(actual, raster, atol=1e-6, rtol=0)
        reflected = dict(data)
        for name in ("roots_xy", "grid_center_xy", "world_velocities"):
            reflected[name] = data[name] * [1., -1.]
        reflected["root_headings"] = -data["root_headings"]
        reflected["grid_heading"] = -data["grid_heading"]
        reflected["static_heights"] = data["static_heights"][:, :, ::-1].copy()
        actual, actual_ids = render(**reflected, include_ego=True)
        expected, expected_ids = mirror(raster, ids)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        twice, twice_ids = mirror(expected, expected_ids)
        np.testing.assert_array_equal(twice, raster)
        np.testing.assert_array_equal(twice_ids, ids)

    def test_boundaries_clipping_and_invalid_contracts(self):
        data = inputs()
        data["world_velocities"] = np.array([[10., -10.], [-10., 10.], [0., 0.]])
        data["static_heights"] = np.full((1, 64, 64), 100.)
        raster, _ = render(**data, include_ego=False)
        np.testing.assert_array_equal(raster[0, 16, 32], [-15., 15., 15.])
        with self.assertRaises(TypeError):
            render(**data)  # Explicit include_ego is mandatory.
        invalid = [("actor_ids", np.array([1, 1, 2])), ("actor_ids", np.array([True, False, True])),
                   ("ego_rows", np.array([0, 0])), ("ego_rows", np.array([3])),
                   ("root_headings", np.zeros((3, 1))), ("grid_heading", np.array([np.nan])),
                   ("world_velocities", np.full((3, 2), np.inf)),
                   ("static_heights", np.zeros((1, 64, 63))),
                   ("height_reference_world_z", np.array([np.nan]))]
        for name, value in invalid:
            with self.subTest(name=name):
                bad = dict(data, **{name: value})
                with self.assertRaises(ValueError):
                    render(**bad, include_ego=False)
        with self.assertRaises(ValueError):
            render(**data, include_ego=1)
        with self.assertRaises(ValueError):
            mirror(np.zeros((1, 64, 64, 3)), np.full((1, 64, 64), -2))


if __name__ == "__main__":
    unittest.main()
