"""Numerical boundaries that can invalidate a physical TRACE/PACER integration."""
# ruff: noqa: E402
# Repository imports intentionally follow standalone test path bootstrap.

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "CrowdSim").is_dir():
    ROOT = Path("/project/EXP/SangforAndSaut/ProtoMotions")
sys.path.insert(0, str(ROOT))
import torch
from CrowdSim.humanoid.diffusion.guidance import (
    PedestrianGuidance,
    PedestrianGuidanceConfig,
)
from CrowdSim.humanoid.runtime.trace_pace import (
    timed_amp_waypoints,
    trace_inputs_from_world,
    TracePaceClosedLoop,
)
from CrowdSim.humanoid.diffusion.trace_helpers import unicyle_forward_dynamics
from CrowdSim.humanoid.diffusion.dynamics import Unicycle
from CrowdSim.humanoid.analysis.trace_pace_capture import pack_candidate_samples
from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene


def guidance(states, *, image=None, scenes=None, targets=None, cfg=None):
    b = len(states)
    if image is None:
        image = torch.zeros(b, 2, 224, 224)
        image[:, 0] = 1
    return PedestrianGuidance(
        states,
        {"extent": torch.tensor([0.5, 0.5, 1.7]).repeat(b, 1), "image": image},
        cfg or PedestrianGuidanceConfig(),
        scene_ids=torch.zeros(b, dtype=torch.long) if scenes is None else scenes,
        local_targets=targets,
        target_indices=torch.tensor([1]) if targets is not None else None,
    )


class TracePaceTests(unittest.TestCase):
    def test_completed_control_frame_releases_retained_state_before_native_close(self):
        import sys
        import weakref
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        owner = SimpleNamespace(physics_frame=None, state_weak=None)
        def completed(owner):
            # External frame retention (observed in the actual SDK smoke) can
            # keep the control state alive after ordinary function return.
            owner.physics_frame = sys._getframe()
            state = torch.ones(3)
            owner.state_weak = weakref.ref(state)
        completed(owner)
        # An external observer may retain this materialised locals map. The
        # frame itself and this map must both release control-state objects.
        cached_locals = owner.physics_frame.f_locals
        self.assertIsNotNone(owner.state_weak())
        result = clear_completed_physics_frame(owner, completed)
        self.assertEqual(cached_locals, {})
        self.assertTrue(result["owned_completed_frame_cleared"])
        self.assertIsNone(owner.state_weak())
        self.assertIsNone(owner.physics_frame)

    def test_control_frame_release_rejects_active_or_unowned_frames(self):
        import sys
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        owner = SimpleNamespace(physics_frame=None)
        def active(owner):
            owner.physics_frame = sys._getframe()
            with self.assertRaises(RuntimeError):
                clear_completed_physics_frame(owner, active)
        active(owner)
        with self.assertRaises(ValueError):
            clear_completed_physics_frame(owner, lambda: None)
        clear_completed_physics_frame(owner, active)

    def test_thin_walls_are_occupied_even_when_grid_points_miss_the_box(self):
        scene = ReconstructionScene.load(ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_keep_right.yaml")
        planner = scene._get_global_planner()
        # Right/top20cm walls have no interior25cm grid samples. Cells must
        # nevertheless cover them; a route cannot hug x=21.75 through a wall.
        self.assertTrue(bool((planner.map_dialate[:, -2:] != 0).all()))
        self.assertTrue(bool((planner.map_dialate[-2:, :] != 0).all()))
        path = planner.get_astar_path((16, 64), (64, 16))
        self.assertIsNotNone(path)
        xy = torch.tensor(path[:, ::-1].copy(), dtype=torch.float32)*.25 + scene.bounds[:2]
        self.assertTrue(bool(scene.in_domain(xy).all()))
        self.assertFalse(bool(scene.occupied(xy).any()))

    def test_keep_right_separates_opposing_paths_without_moving_actual_roots(self):
        scene = ReconstructionScene.load(ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_keep_right.yaml")
        actual = torch.tensor([[6., 6.], [6., 18.]])
        original = actual.clone()
        paths = scene._planning_paths(actual, torch.tensor([[6., 18.], [6., 6.]]), timed=True)
        import numpy as np
        for path in paths:
            self.assertLessEqual(float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()), 18.)
            self.assertLessEqual(float(path[:, 0].max()), 8.)
            self.assertFalse(bool(scene.occupied(torch.tensor(path, dtype=torch.float32)).any()))
        north, south = [path[(path[:, 1] > 10) & (path[:, 1] < 14), 0] for path in paths]
        self.assertGreater(float(north.min() - south.max()), 1.4)
        torch.testing.assert_close(actual, original)
        self.assertEqual(scene.last_path_query_failures, [])
        legacy = ReconstructionScene.load(ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_wide.yaml")
        self.assertFalse(legacy.right_hand_traffic)

    def test_keep_right_rejects_occupied_diagonal_corner_and_nonboolean_flag(self):
        import numpy as np
        from CrowdSim.humanoid.runtime.keep_right_planning import KeepRightPathPlanner
        grid = np.asarray([[0, 1], [1, 0]])
        base = SimpleNamespace(map_dialate=grid, map_size=grid.shape, map_resolution=.25,
                               _build_directions=lambda _: [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])
        planner = KeepRightPathPlanner(base)
        self.assertIsNone(planner.get_astar_path((0, 0), (1, 1)))
        for v in (False, float("nan"), -1.):
            with self.assertRaises(ValueError):
                KeepRightPathPlanner(base, weight=v)
        desc = dict(ReconstructionScene.load(ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_wide.yaml").description)
        desc["right_hand_traffic"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            ReconstructionScene(desc)

    def test_reactive_replan_refreshes_waypoints_and_stationary_heading(self):
        actual = torch.tensor([[0., 0., 0., 0., 0.]])
        world = SimpleNamespace(
            dt=1/30, num_envs=1, device="cpu", shared_world=True,
            planner_inputs=lambda: {"states": actual, "history": actual[:, None].repeat(1, 31, 1),
                                   "neighbors": actual[:, None, None], "raster": None},
            state=lambda: actual,
        )
        received = []
        t = torch.arange(51)*.1
        def sample(**kw):
            received.append(kw)
            return SimpleNamespace(time=t, xy=torch.zeros(1, 51, 2),
                                   selected_sample=torch.tensor([0]), guidance_diagnostics={})
        runtime = TracePaceClosedLoop(SimpleNamespace(plan=sample),
                                     SimpleNamespace(act=lambda *args, **kw: torch.zeros(1, 69)),
                                     PedestrianGuidanceConfig(), stationary_route_heading=True)
        old = torch.tensor([[[1., 0.]]])
        fresh = torch.tensor([[[-1., 0.]]])
        runtime.action(world, 0, old, route_target_provider=lambda: fresh)
        self.assertAlmostEqual(abs(float(received[0]["states"][0, 3])), math.pi, places=6)
        torch.testing.assert_close(received[0]["waypoints"], fresh)
        runtime.action(world, 1, old, route_target_provider=lambda: self.fail("No scheduled or reactive replan"))
        runtime.allow_safety_braking = True
        runtime.mixed_safety_brakes = True
        runtime.braking_plan = lambda *args, **kw: SimpleNamespace(sample_states=torch.zeros(1, 1, 52, 6))
        runtime.actual_velocity_hazards = lambda *args: (torch.tensor([True]), {})
        runtime.action(world, 2, old, route_target_provider=lambda: fresh)
        self.assertEqual(len(received), 2)
        torch.testing.assert_close(received[1]["waypoints"], fresh)
        torch.testing.assert_close(actual, torch.zeros(1, 5))

    def test_waypoint_timing_matches_executed_author_pacer_sampler(self):
        import ast
        author = Path("/project/EXP/SangforAndSaut/TraceAndPace/pacer/pacer/env/tasks/humanoid_traj.py")
        module = ast.parse(author.read_text())
        method = next(node for node in ast.walk(module)
                      if isinstance(node, ast.FunctionDef) and node.name == "_fetch_traj_samples")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(author), "exec"), namespace)
        device = torch.device("cpu")
        original = SimpleNamespace(
            _num_traj_samples=10, _traj_sample_timestep=.5,
            progress_buf=torch.tensor([7]), dt=1/30, num_envs=1, device=device,
            _traj_gen=SimpleNamespace(calc_pos=lambda ids, t: torch.stack((t, 2*t, 0*t), -1)),
        )
        expected = namespace["_fetch_traj_samples"](original)[..., :2]
        t = torch.arange(51)*.1
        plan = SimpleNamespace(time=t, xy=torch.stack((t, 2*t), -1)[None])
        actual, held = timed_amp_waypoints(plan, 7/30)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(held, 0)
        legacy, _ = timed_amp_waypoints(plan, 7/30, first_sample_s=.5)
        self.assertGreater(float((legacy[:, 0]-expected[:, 0]).norm()), .5)


    def test_repeated_braking_does_not_move_amp_stop_anchor_forward(self):
        runtime = TracePaceClosedLoop(
            None, None, PedestrianGuidanceConfig(),
            allow_safety_braking=True, persistent_braking_target=True,
        )
        runtime.plan = SimpleNamespace(
            selected_sample=torch.tensor([-1, 2]),
            xy=torch.tensor([[[0., 0.], [.5, 0.]], [[2., 0.], [3., 0.]]]),
        )
        runtime.retain_braking_targets()
        command = runtime.apply_braking_targets(torch.ones(2, 10, 2))
        torch.testing.assert_close(command[0], torch.tensor([.5, 0.]).repeat(10, 1))
        torch.testing.assert_close(command[1], torch.ones(10, 2))
        runtime.plan.xy[0, -1] = torch.tensor([1., 0.])
        runtime.retain_braking_targets()
        torch.testing.assert_close(runtime.braking_anchor_xy[0], torch.tensor([.5, 0.]))
        runtime.plan.selected_sample[0] = 0
        runtime.retain_braking_targets()
        torch.testing.assert_close(runtime.apply_braking_targets(torch.ones_like(command) * 2)[0], torch.ones_like(command[0]) * 2)
        runtime.plan.selected_sample[0] = -1
        runtime.retain_braking_targets()
        torch.testing.assert_close(runtime.braking_anchor_xy[0], torch.tensor([1., 0.]))
        runtime.invalidate_after_reset()
        self.assertIsNone(runtime.braking_anchor_xy)


    def test_tracking_margin_brakes_before_delayed_controller_closing_pair(self):
        nav = torch.tensor([[0., 0., .2, 0., 0.], [2., 0., -.2, 0., math.pi]])
        packet = {"states": torch.tensor([[0., 0., .2, 0.], [2., 0., .2, math.pi]])}
        image = torch.zeros(2, 2, 224, 224)
        image[:, 0] = 1
        context = {"extent": torch.tensor([.5, .5, 1.7]).repeat(2, 1), "image": image}
        planner = SimpleNamespace(prepare_context=lambda **_: context)
        world = SimpleNamespace(planar=lambda: nav)
        baseline = TracePaceClosedLoop(planner, None, PedestrianGuidanceConfig())
        conservative = TracePaceClosedLoop(
            planner, None, PedestrianGuidanceConfig(),
            guard_horizon_s=2., guard_tracking_margin_m=.5,
        )
        ids = torch.zeros(2, dtype=torch.long)
        self.assertEqual(baseline.actual_velocity_hazards(world, packet, ids)[0].tolist(), [False, False])
        risk, details = conservative.actual_velocity_hazards(world, packet, ids)
        self.assertEqual(risk.tolist(), [True, True])
        self.assertEqual(details["physical_tracking_margin_m"], .5)
        torch.testing.assert_close(nav[:, :2], torch.tensor([[0., 0.], [2., 0.]]))


    def test_body_safe_recovery_covers_full_inflated_map_band(self):
        # Actual terminal root from wide-contact v1: physically free, but the
        # nearest padded grid node lies1.252m away, beyond old1.25m recovery.
        scene = ReconstructionScene.load(
            ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_wide.yaml"
        )
        position = torch.tensor([[7.6798129, 3.4998207]])
        original = position.clone()
        scene.start_recovery_radius = 1.25
        waypoints, _, _ = scene.route_waypoints(
            position, torch.tensor([2]), actor_ids=torch.tensor([2]), advance=False
        )
        self.assertEqual(scene.last_path_query_failures[0]["actor_id"], 2)
        self.assertEqual(
            scene.last_path_query_failures[0]["reason"],
            "no_reachable_body_safe_connector_within_recovery_radius",
        )
        torch.testing.assert_close(waypoints[:, 0], position)
        scene.start_recovery_radius = 2.9
        waypoints, _, _ = scene.route_waypoints(
            position, torch.tensor([2]), actor_ids=torch.tensor([2]), advance=False
        )
        self.assertEqual(scene.last_path_query_failures, [])
        self.assertGreater(float((waypoints[:, -1] - position).norm()), .5)
        self.assertGreater(scene.last_start_recovery_segments[0]["connector_length_m"], 1.25)
        torch.testing.assert_close(position, original)

    def test_recovery_connector_rejects_body_envelope_wall_intersection(self):
        scene = ReconstructionScene.load(
            ROOT / "configs/diffusion_amp/shared_obstacle_trace_pace_wide.yaml"
        )
        # Both roots are outside the cuboid itself, but the first body's.35m
        # envelope overlaps it. Recovery must not silently certify that path.
        import numpy as np
        self.assertFalse(scene._connector_avoids_cuboids(
            np.array([9.8, 12.]), np.array([7., 12.])
        ))
        self.assertTrue(scene._connector_avoids_cuboids(
            np.array([9.6, 12.]), np.array([7., 12.])
        ))


    def test_fixed_stop_allows_other_actor_to_keep_a_safe_diffusion_route(self):
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
        trajectory = torch.zeros(2, 2, 52, 6)
        trajectory[0, 0, :, 0] = 1
        trajectory[1, 0, :, 0] = -1
        selected, parts = guidance(state, targets=targets).select(
            trajectory, fixed_samples=torch.tensor([1, -1], dtype=torch.long)
        )
        self.assertEqual(selected.tolist(), [1, 0])
        self.assertEqual(
            float(parts["selected_joint_maximum_physical_overlap_m"].sum()), 0
        )

    def test_stationary_intent_can_face_reverse_route_without_body_pose_write(self):
        x = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0]])
        world = SimpleNamespace(
            planner_inputs=lambda: {
                "states": x,
                "history": x[:, None].repeat(1, 31, 1),
                "neighbors": x[:, None, None],
                "raster": None,
            }
        )
        targets = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]])
        packet = trace_inputs_from_world(world, stationary_route_targets=targets)
        self.assertAlmostEqual(abs(float(packet["states"][0, 3])), math.pi, places=6)
        self.assertAlmostEqual(float(packet["states"][1, 3]), 0, places=6)
        torch.testing.assert_close(packet["history"][:, -1], packet["states"])
        torch.testing.assert_close(packet["states"][:, :2], x[:, :2])
        torch.testing.assert_close(x[:, 4], torch.zeros(2))

    def test_comfort_tolerance_never_permits_centimetre_body_overlap(self):
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.70, 0.0, 0.0, 0.0]])
        cfg = PedestrianGuidanceConfig(agent_buffer_m=0, comfort_recovery_speed_m_s=0)
        with self.assertRaisesRegex(RuntimeError, "admission gate"):
            guidance(state, cfg=cfg).select(torch.zeros(2, 1, 52, 6))

    def test_actual_closing_pair_brakes_before_physical_separation_is_lost(self):
        # At t=0 the bodies are separated; their measured velocities converge
        # before the next scheduled DDPM update. An existing route is irrelevant.
        nav = torch.tensor([[0.0, 0.0, 0.6, 0.0, 0.0], [1.5, 0.0, -0.6, 0.0, math.pi]])
        image = torch.zeros(2, 2, 224, 224)
        image[:, 0] = 1
        context = {
            "extent": torch.tensor([[0.5, 0.5, 1.7]]).repeat(2, 1),
            "image": image,
        }
        runtime = TracePaceClosedLoop(
            SimpleNamespace(prepare_context=lambda **_: context),
            None,
            PedestrianGuidanceConfig(),
            allow_safety_braking=True,
        )
        packet = {
            "states": torch.tensor([[0.0, 0.0, 0.6, 0.0], [1.5, 0.0, 0.6, math.pi]])
        }
        world = SimpleNamespace(planar=lambda: nav)
        hazard, diagnostics = runtime.actual_velocity_hazards(
            world, packet, torch.tensor([0, 0])
        )
        self.assertEqual(hazard.tolist(), [True, True])
        self.assertEqual(diagnostics["closing_pair_at_risk"], [True, True])
        # Different physical scenes must never trigger a shared brake.
        isolated, _ = runtime.actual_velocity_hazards(
            world, packet, torch.tensor([0, 1])
        )
        self.assertFalse(bool(isolated.any()))

    def test_stationary_comfort_buffer_does_not_create_reactive_deadlock(self):
        nav = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0, 0.0]])
        image = torch.zeros(2, 2, 224, 224)
        image[:, 0] = 1
        context = {
            "extent": torch.tensor([[0.5, 0.5, 1.7]]).repeat(2, 1),
            "image": image,
        }
        runtime = TracePaceClosedLoop(
            SimpleNamespace(prepare_context=lambda **_: context),
            None,
            PedestrianGuidanceConfig(),
            allow_safety_braking=True,
        )
        packet = {"states": torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]])}
        hazard, _ = runtime.actual_velocity_hazards(
            SimpleNamespace(planar=lambda: nav), packet, torch.tensor([0, 0])
        )
        self.assertFalse(bool(hazard.any()))

    def test_optional_brake_preserves_actual_velocity_and_requires_safe_geometry(self):
        nav = torch.tensor([[0.0, 0.0, 1.2, 0.0, 0.0]])
        current = torch.tensor([[0.0, 0.0, 1.2, 0.0]])
        image = torch.zeros(1, 2, 224, 224)
        image[:, 0] = 1
        context = {"extent": torch.tensor([[0.5, 0.5, 1.7]]), "image": image}
        planner = SimpleNamespace(prepare_context=lambda **_: context)
        runtime = TracePaceClosedLoop(
            planner, None, PedestrianGuidanceConfig(max_acceleration_m_s2=2)
        )
        packet = {"states": current}
        world = SimpleNamespace(planar=lambda: nav)
        brake = runtime.braking_plan(world, packet, torch.tensor([0]))
        self.assertAlmostEqual(float(brake.xy[0, -1, 0]), 0.36, places=5)
        self.assertEqual(brake.selected_sample.tolist(), [-1])
        self.assertIn("NOT a diffusion sample", brake.reference)
        self.assertAlmostEqual(float(brake.vel[0, -1]), 0, places=6)
        image[:, 0, :, 62:] = 0
        image[:, 1, :, 62:] = 1
        with self.assertRaisesRegex(RuntimeError, "admission gate"):
            runtime.braking_plan(world, packet, torch.tensor([0]))

    def test_runtime_rejects_clock_jumps_before_reading_world(self):
        runtime = TracePaceClosedLoop(None, None, PedestrianGuidanceConfig())
        with self.assertRaisesRegex(ValueError, "Contiguous"):
            runtime.action(None, 2, None)

    def test_ragged_candidate_capture_is_lossless_without_padding_brakes(self):
        import numpy as np

        rows = [
            np.ones((4, 16, 52, 6), dtype=np.float32),
            np.ones((4, 1, 52, 6), dtype=np.float32) * 2,
        ]
        packed, offsets = pack_candidate_samples(rows)
        self.assertEqual(offsets.tolist(), [0, 16, 17])
        for index, row in enumerate(rows):
            np.testing.assert_array_equal(
                row, packed[offsets[index] : offsets[index + 1]].transpose(1, 0, 2, 3)
            )

    def test_social_buffer_recovers_gradually_without_admitting_body_overlap(self):
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]])
        trajectory = torch.zeros(2, 1, 52, 6)
        step = torch.arange(1, 53) * 0.01
        trajectory[0, 0, :, 0] = -step
        trajectory[1, 0, :, 0] = step
        selected, parts = guidance(state).select(trajectory)
        self.assertEqual(selected.tolist(), [0, 0])
        self.assertEqual(
            float(parts["selected_joint_maximum_physical_overlap_m"].sum()), 0
        )
        unsafe = state.clone()
        unsafe[1, 0] = 0.65
        with self.assertRaisesRegex(RuntimeError, "admission gate"):
            guidance(unsafe).select(trajectory)

    def test_faster_measured_start_is_braked_causally_and_yaw_is_bounded(self):
        dynamics = Unicycle("pedestrian", acce_bound=[-2, 2], vbound=[0, 1.4])
        initial = torch.tensor([[0.0, 0.0, 3.0, 0.0]])
        actions = torch.ones(1, 20, 2) * 100
        future = unicyle_forward_dynamics(
            dynamics,
            initial,
            actions,
            0.1,
            pedestrian_speed_ceiling=1.4,
            pedestrian_yaw_rate_ceiling=0.8,
        )
        previous = torch.cat((initial[:, None, 2:3], future[:, :-1, 2:3]), 1)
        self.assertAlmostEqual(float(future[0, 0, 2]), 2.8, places=5)
        self.assertLessEqual(
            float(((future[:, :, 2:3] - previous) / 0.1).abs().max()), 2.00001
        )
        self.assertAlmostEqual(float(future[0, -1, 2]), 1.4, places=5)
        self.assertLessEqual(
            float(torch.diff(future[:, :, 3], dim=1).abs().max() / 0.1), 0.80001
        )

    def test_map_gradient_escapes_obstacle_and_outside_domain(self):
        image = torch.zeros(1, 2, 224, 224)
        image[:, 0] = 1
        image[:, 0, :, 80:] = 0
        image[:, 1, :, 80:] = 1
        g = guidance(torch.zeros(1, 4), image=image)
        for x in [2.2, 30.0]:
            traj = torch.zeros(1, 1, 3, 6)
            traj[..., 0] = x
            traj.requires_grad_()
            loss = g.components(traj)["map"].sum()
            (grad,) = torch.autograd.grad(loss, traj)
            self.assertGreater(float(grad[..., 0].sum()), 0)
            self.assertTrue(torch.isfinite(grad).all())

    def test_joint_scene_selection_keeps_corresponding_samples(self):
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
        targets = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]])
        g = guidance(state, targets=targets)
        traj = torch.zeros(2, 2, 3, 6)
        traj[:, 0, :, 0] = 1
        traj[:, 1, :, 0] = 2
        chosen, _ = g.select(traj)
        self.assertEqual(chosen.tolist(), [0, 1])
        separate = guidance(state, scenes=torch.tensor([0, 1]), targets=targets)
        self.assertEqual(separate.select(traj)[0].tolist(), [0, 1])

    def test_mixed_sample_selection_checks_new_cross_actor_collisions(self):
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
        traj = torch.zeros(2, 2, 3, 6)
        traj[0, 0, :, 0] = 1
        traj[1, 0, :, 0] = -1
        selected, parts = guidance(state, targets=targets).select(traj)
        self.assertNotEqual(selected.tolist(), [0, 0])
        self.assertEqual(
            float(parts["selected_joint_maximum_agent_overlap_m"].sum()), 0
        )

    def test_peer_cost_uses_world_coordinates_and_scene_masks(self):
        traj = torch.zeros(2, 1, 3, 6)
        g = guidance(torch.zeros(2, 4))
        self.assertGreater(float(g.components(traj)["agent"].sum()), 0)
        separate = guidance(torch.zeros(2, 4), scenes=torch.tensor([0, 1]))
        self.assertEqual(float(separate.components(traj)["agent"].sum()), 0)
        moved = guidance(
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, math.pi / 2]])
        )
        self.assertEqual(float(moved.components(traj)["agent"].sum()), 0)

    def test_rejects_all_unsafe_candidates(self):
        g = guidance(torch.zeros(1, 4))
        traj = torch.zeros(1, 2, 3, 6)
        traj[..., 2] = 10
        with self.assertRaisesRegex(RuntimeError, "admission gate"):
            g.select(traj)
        with self.assertRaises(ValueError):
            guidance(torch.zeros(1, 4), image=torch.full((1, 2, 224, 224), 0.5))

    def test_realized_dynamics_do_not_treat_raw_controls_as_route_acceleration(self):
        traj = torch.zeros(1, 1, 3, 6)
        traj[..., 4:6] = 100
        parts = guidance(torch.zeros(1, 4)).components(traj)
        self.assertEqual(float(parts["maximum_acceleration_m_s2"].sum()), 0)
        self.assertEqual(float(parts["maximum_yaw_rate_rad_s"].sum()), 0)
        self.assertEqual(
            float(parts["raw_network_maximum_acceleration_m_s2"].sum()), 100
        )

    def test_guidance_gradient_is_independent_of_sampling_budget(self):
        derivatives = []
        for count in [1, 4]:
            traj = torch.zeros(1, count, 3, 6, requires_grad=True)
            targets = torch.tensor([[[2.0, 0.0]]])
            g = guidance(torch.zeros(1, 4), targets=targets)
            loss, _ = g.compute_guidance_loss(traj, {})
            (derivative,) = torch.autograd.grad(loss, traj)
            derivatives.append(derivative)
        torch.testing.assert_close(
            derivatives[0].expand_as(derivatives[1]), derivatives[1]
        )

    def test_goal_and_pair_cost_rotation_invariant(self):
        states = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
        traj = torch.zeros(2, 1, 3, 6)
        traj[0, ..., 0] = 1
        traj[1, ..., 1] = -1
        original = guidance(states).components(traj)["agent"]
        rotated = states.clone()
        rotated[:, :2] = torch.stack((-states[:, 1], states[:, 0]), -1)
        rotated[:, 3] = math.pi / 2
        self.assertTrue(
            torch.allclose(
                original, guidance(rotated).components(traj)["agent"], atol=1e-6
            )
        )

    def test_actual_clock_interpolates_and_reports_terminal_hold(self):
        time = torch.arange(51) * 0.1
        plan = SimpleNamespace(time=time, xy=torch.stack((time, time * 0), -1)[None])
        targets, held = timed_amp_waypoints(plan, 0.2, first_sample_s=.5)
        self.assertAlmostEqual(float(targets[0, 0, 0]), 0.7, places=6)
        self.assertAlmostEqual(float(targets[0, -1, 0]), 5, places=6)
        self.assertEqual(held, 1)
        with self.assertRaises(ValueError):
            timed_amp_waypoints(plan, -0.1)

    def test_nav_velocity_conversion_and_missing_history_preserved(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5]])
        mask = torch.tensor([[False, True]])
        world = SimpleNamespace(
            planner_inputs=lambda: {
                "states": x,
                "history": x[:, None].repeat(1, 2, 1),
                "neighbors": x[:, None, None],
                "history_mask": mask,
                "raster": None,
                "neighbor_mask": None,
            }
        )
        packet = trace_inputs_from_world(world)
        torch.testing.assert_close(
            packet["states"], torch.tensor([[1.0, 2.0, 5.0, math.atan2(4, 3)]])
        )
        self.assertIs(packet["history_mask"], mask)

    def test_velocity_heading_regenerates_obstacle_raster_from_geometry(self):
        x = torch.tensor([[2.0, 3.0, 0.0, 1.0, 0.0]])
        queried = []

        def raster(state):
            queried.append(state.clone())
            return torch.ones(1, 2, 224, 224)

        world = SimpleNamespace(
            planner_inputs=lambda: {
                "states": x,
                "history": x[:, None].repeat(1, 31, 1),
                "neighbors": x[:, None, None],
                "raster": torch.zeros(1, 2, 224, 224),
            },
            validation_scene=SimpleNamespace(raster=raster),
        )
        packet = trace_inputs_from_world(world)
        self.assertAlmostEqual(float(queried[0][0, 4]), math.pi / 2, places=6)
        torch.testing.assert_close(queried[0][:, :4], x[:, :4])
        self.assertEqual(float(x[0, 4]), 0)  # actual body's yaw was not changed
        torch.testing.assert_close(packet["raster"], torch.ones(1, 2, 224, 224))
        world.validation_scene = None
        with self.assertRaisesRegex(ValueError, "geometry provider"):
            trace_inputs_from_world(world)

    def test_near_stationary_heading_retains_measured_body_yaw(self):
        x = torch.tensor([[0.0, 0.0, 0.001, -0.001, 2.0]])
        world = SimpleNamespace(
            planner_inputs=lambda: {
                "states": x,
                "history": x[:, None],
                "neighbors": x[:, None, None],
                "raster": None,
            }
        )
        packet = trace_inputs_from_world(world)
        self.assertAlmostEqual(float(packet["states"][0, 3]), 2)


if __name__ == "__main__":
    unittest.main()
