"""Authored pure-Torch mocks; no native API, SDK, NN or real force proof."""
import importlib.util
import os
import sys
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Tests require explicitly hidden CUDA")

import torch

import net_normal_sampler as sampler


class View:
    def __init__(self, paths, payload):
        self.sensor_paths, self.sensor_count = list(paths), 24
        self.payload, self.calls = payload, []

    def get_net_contact_forces(self, dt):
        self.calls.append(dt)
        return self.payload

    def get_contact_force_matrix(self, dt):
        raise AssertionError("Pair matrix must never be used")


def fixture():
    sampler._source(sampler.BRIDGE_PATH, 6035, sampler.BRIDGE_SHA, readonly=True)
    spec = importlib.util.spec_from_file_location("_original_bridge_for_mock_sampler", sampler.BRIDGE_PATH)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    rows = bridge.mixed_group_rows(8, device="cpu")
    canonical = sampler._canonical_names()
    expected = torch.arange(8*24*3, dtype=torch.float32).reshape(8, 24, 3)
    views, layouts, names = [], [], []
    for actor in reversed(range(8)):
        order = tuple(reversed(range(24))) if actor == 3 else tuple(range(24))
        body_names = tuple(canonical[index] for index in order)
        paths = [f"/World/Actor_{actor}/{name}" for name in body_names]
        views.append(View(paths, expected[actor, list(order)].clone()))
        layouts.append({"actor": actor, "sources": paths})
        names.append(body_names)
    return bridge, rows, views, layouts, names, expected


def sample(bridge, rows, views, layouts, names, **kwargs):
    options = {"bridge": bridge, "device": "cpu", "physics_dt": 1/120,
               "physics_step": 32, "control_step": 8}
    options.update(kwargs)
    return sampler.sample_current_net_normal(views, layouts, names, rows, **options)


class CurrentNormalSamplerCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_complete_preflight_then_ordered_current_vector_bridge(self):
        bridge, rows, views, layouts, names, expected = fixture()
        rng, row_copy = torch.random.get_rng_state().clone(), rows.clone()
        force, metadata = sample(bridge, rows, views, layouts, names)
        self.assertTrue(torch.equal(force, expected))
        self.assertEqual(metadata["physics_dt_s"], 1/120)
        self.assertEqual(metadata["caller_physics_step"], 32)
        self.assertEqual(metadata["caller_control_step"], 8)
        self.assertFalse(metadata["clock_independently_verified"])
        self.assertTrue(all(view.calls == [1/120] for view in views))
        self.assertTrue(torch.equal(rows, row_copy))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for view in views:
            view.payload.zero_()
        self.assertTrue(torch.equal(force, expected))
        self.assertNotIn("warp", sys.modules)
        self.assertFalse(any(name in sys.modules for name in ("isaaclab", "isaaclab_physx", "isaacsim")))

    def test_all_layouts_checked_before_any_getter(self):
        bridge, rows, views, layouts, names, _ = fixture()
        views[-1].sensor_paths = list(reversed(views[-1].sensor_paths))
        with self.assertRaises(ValueError):
            sample(bridge, rows, views, layouts, names)
        self.assertTrue(all(view.calls == [] for view in views))
        views[-1].sensor_paths = list(layouts[-1]["sources"])
        names[-1] = ("Unknown", *names[-1][1:])
        with self.assertRaises(ValueError):
            sample(bridge, rows, views, layouts, names)
        self.assertTrue(all(view.calls == [] for view in views))

    def test_early_shape_clock_and_partition_rejections(self):
        bridge, rows, views, layouts, names, _ = fixture()
        bad_rows = rows.clone()
        bad_rows[0, 0] = bad_rows[0, 1]
        for row_value, kwargs in ((bad_rows, {}), (rows.double(), {}), (rows, {"physics_dt": 0.}),
                                  (rows, {"physics_dt": float("nan")}), (rows, {"physics_step": True}),
                                  (rows, {"control_step": -1}), (rows, {"device": "cuda"})):
            with self.assertRaises(ValueError):
                sample(bridge, row_value, views, layouts, names, **kwargs)
        layouts[-1]["actor"] = layouts[0]["actor"]
        with self.assertRaises(ValueError):
            sample(bridge, rows, views, layouts, names)
        self.assertTrue(all(view.calls == [] for view in views))

    def test_source_force_failures_are_not_transactional_or_sanitized(self):
        for bad in (torch.zeros((24, 3), dtype=torch.float64), torch.zeros((23, 3)),
                    torch.full((24, 3), float("nan"))):
            bridge, rows, views, layouts, names, _ = fixture()
            views[1].payload = bad
            with self.assertRaises(ValueError):
                sample(bridge, rows, views, layouts, names)
            self.assertEqual(views[0].calls, [1/120])
            self.assertEqual(views[1].calls, [1/120])
            self.assertTrue(all(view.calls == [] for view in views[2:]))


if __name__ == "__main__":
    unittest.main()
