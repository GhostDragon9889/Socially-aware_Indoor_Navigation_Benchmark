"""CPU bridge/derivation tests; synthetic planner calls are not SDK evidence."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch

from CrowdSim.humanoid.trajectory.social_force_runtime import SCENE, SocialForceSharedPlanner, fixture_grid
from scripts import evaluate_shared_social_navigation as baseline
from scripts.evaluate_shared_social_sfm import derive_driver


class SharedSFMIntegrationTests(unittest.TestCase):
    def test_grid_uses_ros_y_and_cell_intersections(self):
        scene = {"schema": "navisaaclab.reconstruction_scene.v1", "bounds_xy": [0, 0, 4, 4],
                 "cuboids": [{"center": [2.1, 3.1, 1], "size": [.01, .01, 2]}]}
        grid = fixture_grid(scene, source={"path": "synthetic", "sha256": "a" * 64}, resolution_m=1.)
        # A small physical cuboid between cell centers must still occupy a cell.
        self.assertEqual(grid.labels[0, 2], 0)
        with self.assertRaises(ValueError):
            grid.pixel_indices(np.array([[2.1, 3.1]], np.float64))
        self.assertGreater(grid.distance_m[3, 0], 0.)
        with self.assertRaises(ValueError):
            fixture_grid(scene, source={"path": "synthetic", "sha256": "a" * 64}, resolution_m=.3)

    def test_snapshot_preserves_inputs_rng_and_determinism(self):
        planner = SocialForceSharedPlanner.for_CPU_snapshot(scene_source=SCENE, seed=7)
        states = torch.tensor([[6., 6., 0, 0, 0], [6., 18., 0, 0, 0],
                               [18., 6., 0, 0, 0], [18., 18., 0, 0, 0]])
        goals = states[:, None, :2].repeat(1, 10, 1)
        goals[:, :, 0] += torch.tensor([3., 3., -3., -3.])[:, None]
        before = states.clone(), goals.clone(), torch.get_rng_state().clone()
        first = planner.predict_snapshot(states, goals)
        second = planner.predict_snapshot(states, goals)
        self.assertEqual(first.xy.shape, (4, 51, 2))
        self.assertTrue(torch.equal(first.xy, second.xy))
        self.assertTrue(torch.equal(states, before[0]) and torch.equal(goals, before[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(torch.equal(first.xy[:, 0], states[:, :2]))

    def test_subset_scene_or_nonfinite_snapshot_fails(self):
        planner = SocialForceSharedPlanner.for_CPU_snapshot(scene_source=SCENE, seed=7)
        with self.assertRaises(ValueError):
            planner.predict_snapshot(torch.zeros(2, 5), torch.zeros(2, 10, 2))
        state = torch.zeros(4, 5)
        state[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            planner.predict_snapshot(state, torch.zeros(4, 10, 2))

    def test_complete_driver_call_rejects_stale_neighbors_and_new_objectives(self):
        planner = SocialForceSharedPlanner.for_CPU_snapshot(scene_source=SCENE, seed=7)
        states = torch.tensor([[6., 6., 0, 0, 0], [6., 18., 0, 0, 0],
                               [18., 6., 0, 0, 0], [18., 18., 0, 0, 0]])
        history = states[:, None].repeat(1, 31, 1)
        neighbors = torch.stack([history[[peer for peer in range(4) if peer != actor]] for actor in range(4)])
        goals = states[:, None, :2].repeat(1, 10, 1)
        goals[:, :, 0] += torch.tensor([3., 3., -3., -3.])[:, None]
        inputs = dict(states=states, history=history, neighbors=neighbors, raster=torch.zeros(4, 2, 224, 224),
                      history_mask=torch.ones(4, 31, dtype=torch.bool), neighbor_mask=torch.ones(4, 3, 31, dtype=torch.bool),
                      waypoints=goals, waypoint_times=torch.arange(1, 11, dtype=torch.float32)/2, num_samples=1)
        options = dict(guidance_weight=20., guidance_mode="trace_clean", near_horizon_mass=0.,
                       scene_ids=torch.zeros(4, dtype=torch.int64), agent_collision_weight=5., agent_collision_radius=.5,
                       scene_collision_weight=10., scene_collision_radius=.5, trajectory_speed_weight=0.,
                       max_trajectory_speed_mps=1.5, trajectory_accel_weight=0., max_trajectory_accel_mps2=2.,
                       scene_bounds_xy=None, obstacle_centers_xy=None, obstacle_sizes_xy=None)
        prediction = planner.plan(**inputs, **options)
        self.assertTrue(torch.equal(prediction.xy, planner.predict_snapshot(states, goals).xy))
        with self.assertRaises(ValueError):
            planner.plan(**inputs, **options, undeclared_objective=1.)
        neighbors[0, 0, -1, 0] += 1
        with self.assertRaises(ValueError):
            planner.plan(**inputs, **options)

    def test_physical_derivation_preserves_shared_helpers_and_world_step(self):
        source, _ = baseline.derive_entry(baseline.PARENT.read_text())
        module = ModuleType("isolated_base_navigation_fixture")
        module.__file__ = str(baseline.PARENT)
        exec(compile(source, str(baseline.PARENT), "exec"), module.__dict__)
        driver = (Path(__file__).resolve().parents[1] / "scripts/train_paper_amp.py").read_text()
        original, _ = module._derive(driver)
        result, proof = derive_driver(driver, module._derive)
        self.assertTrue(proof["SFM_inverse_complete_bytes_equal"] and proof["shared_helper_AST_equal"])
        old, new = ast.parse(original), ast.parse(result)
        def calls(tree, method):
            return [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "world" and node.func.attr == method]
        for method in ("step", "reset", "state", "enable_pairwise_contacts"):
            self.assertEqual(calls(old, method), calls(new, method))
        self.assertNotIn("PaperDiffusionPlanner", result)
        self.assertIn('"diffusion_used_for_rollouts": False', result)
        # The Social entry passes an unused factory marker; its own
        # _social_admission owns the frozen policy. Gate explicit evaluation
        # arguments rather than depending on that legacy marker's boolean.
        gate = next(node for node in ast.walk(new) if isinstance(node, ast.If)
                    and any(isinstance(child, ast.Constant) and child.value ==
                            "SFM arm requires uncompiled pure shared-actor evaluation" for child in ast.walk(node)))
        context = {"curriculum": None, "pure_evaluation": False,
                   "args": SimpleNamespace(evaluate_steps=30, resume="actual_frozen_actor.pt", compile_diffusion_forward=False)}
        self.assertFalse(eval(compile(ast.Expression(gate.test), "source-bound-SFM-runtime-gate", "eval"), context))
        with self.assertRaises(ValueError):
            derive_driver(driver.replace("PaperDiffusionPlanner", "ChangedPlanner"), module._derive)


if __name__ == "__main__":
    unittest.main()
