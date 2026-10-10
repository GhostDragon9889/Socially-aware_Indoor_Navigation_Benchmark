"""Four bounded fake-interface checks of the unmodified frozen World methods.

The fixture deliberately bypasses constructor admission with __new__. The
counter/assets/reference states are synthetic, not SDK, physics, AMASS or FK
evidence. Only the World clock/control/reset and reused tensor math execute.
No policy/network, checkpoints, native SDK, GPU or earlier suite is exercised.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Explicitly hidden CUDA is required")

SOURCE = Path(__file__).with_name("shared_group_world.py")
SOURCE_SHA = "b608cae6cd8249e2c66d841813b4291b8bd229ca7beae45af625c33daf28fceb"
info, source = SOURCE.lstat(), SOURCE.read_bytes()
if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222
        or len(source) != 35097 or hashlib.sha256(source).hexdigest() != SOURCE_SHA):
    raise RuntimeError("Reviewed readonly World whole-byte identity changed")
sys.path.insert(0, str(SOURCE.resolve().parents[3]))
world_module = ModuleType("_frozen_shared_world_for_fake_CPU")
world_module.__file__ = str(SOURCE)
sys.modules[world_module.__name__] = world_module
exec(compile(source, str(SOURCE), "exec"), world_module.__dict__)
torch = world_module.torch
BASELINE, ACTORS = 137, 8


def robot_state(states, count):
    """Synthetic named COMMON state; repeat body offsets, no claimed FK."""
    positions = torch.zeros((count, 24, 3), dtype=torch.float32)
    positions[..., 0] = torch.arange(count, dtype=torch.float32)[:, None] * 5
    positions[..., 2] = 1 + torch.arange(24, dtype=torch.float32)[None] / 100
    quaternions = torch.zeros((count, 24, 4), dtype=torch.float32)
    quaternions[..., 3] = 1
    return states.RobotState(state_conversion=states.StateConversion.COMMON,
        rigid_body_pos=positions, rigid_body_rot=quaternions,
        rigid_body_vel=torch.zeros((count, 24, 3)), rigid_body_ang_vel=torch.zeros((count, 24, 3)),
        dof_pos=torch.zeros((count, 69)), dof_vel=torch.zeros((count, 69)))


class FakeSim:
    def __init__(self, events):
        self.counter, self.events = BASELINE, events

    def get_physics_step_count(self):
        return self.counter

    def step(self, *, render):
        if render is not False:
            raise AssertionError("Rendering is outside this fixture")
        self.counter += 1
        self.events.append(("step", self.counter))

    def forward(self):
        self.events.append(("forward", self.counter))


class FakeScene:
    def __init__(self, sim, events):
        self.sim, self.events = sim, events

    def write_data_to_sim(self):
        self.events.append(("write", self.sim.counter))

    def update(self, dt):
        self.events.append(("update", self.sim.counter, dt))

    def reset(self, *, env_ids):
        self.events.append(("scene_reset", self.sim.counter, env_ids.clone()))


class FakeAssetData:
    def __init__(self, asset):
        self.asset = asset

    @property
    def joint_vel(self):
        asset = self.asset
        asset.events.append(("velocity", asset.sim.counter, asset.slot))
        sign = torch.where(asset.actor_rows % 2 == 0, 1., -1.)
        return (sign[:, None] * (asset.sim.counter - BASELINE + 1)).expand(-1, 69).clone()


class FakeAsset:
    def __init__(self, slot, rows, sim, events):
        self.slot, self.actor_rows, self.sim, self.events = slot, rows[:, slot], sim, events
        self.native_order = torch.arange(69).roll(slot * 7).flip((0,))
        self.data = FakeAssetData(self)
        self.root_view = SimpleNamespace(get_dof_projected_joint_forces=self.effort)

    def effort(self):
        self.events.append(("effort", self.sim.counter, self.slot))
        common = ((self.actor_rows + 1).to(torch.float32)[:, None]
                  * torch.arange(1, 70, dtype=torch.float32)[None] * (self.sim.counter - BASELINE))
        return common[:, self.native_order].clone()


class FakeAdapter:
    """External snapshot/write boundary, not the state-adapter test suite."""
    def __init__(self, deps, assets, sim, events, states):
        self.deps, self.assets, self.sim, self.events, self.states = deps, assets, sim, events, states
        self.state = robot_state(states, ACTORS)
        self.mappings = tuple(SimpleNamespace(dof_convert_to_common=torch.argsort(asset.native_order)) for asset in assets)
        self.targets = []

    def snapshot(self):
        self.events.append(("snapshot", self.sim.counter))
        state = self.state.clone()
        root = self.states.RootOnlyState(state_conversion=self.states.StateConversion.COMMON,
            **{key: getattr(state, key).clone() for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel")})
        return self.deps.adapter.CommonSnapshot(state, root)

    def set_common_position_targets(self, target):
        self.events.append(("targets", self.sim.counter))
        self.targets.append(target.clone())

    def write_common_reset(self, reset, ids):
        self.events.append(("adapter_reset", self.sim.counter, ids.clone()))
        # Synthetic fresh link pose: translate existing body offsets only.
        offsets = self.state.rigid_body_pos[ids] - self.state.root_pos[ids, None]
        self.state.rigid_body_pos[ids] = reset.root_pos[ids, None] + offsets
        self.state.rigid_body_rot[ids] = reset.root_rot[ids, None]
        self.state.rigid_body_vel[ids] = reset.root_vel[ids, None]
        self.state.rigid_body_ang_vel[ids] = reset.root_ang_vel[ids, None]
        self.state.dof_pos[ids], self.state.dof_vel[ids] = reset.dof_pos[ids], reset.dof_vel[ids]


class FakeReference:
    """Deterministic contract data, not an AMASS/private curriculum substitute."""
    def __init__(self, states, sim, events):
        self.states, self.sim, self.events = states, sim, events
        self.plan_calls, self.reference_draws, self.plan_interrupt = 0, 0, None
        self.bound = None

    def build_plan(self, state, age, *, actor_ids, episode_ids):
        self.plan_calls += 1  # Observable side effect BEFORE the requested interrupt.
        self.events.append(("plan", self.sim.counter, actor_ids.clone(), episode_ids.clone()))
        if self.plan_interrupt is not None:
            raise self.plan_interrupt
        plan = state.root_pos[actor_ids, None, :2].expand(-1, 151, -1).clone()
        plan[..., 0] += torch.arange(151, dtype=torch.float32)[None] / 30
        return SimpleNamespace(native_xy=plan)

    def future_targets(self, age, plan_age, plans, *, episode_ids):
        return plans[:, torch.arange(1, 11) * 15].clone()

    def interpolate_reference(self, times, plans):
        index = (times[:, 0] * 30).round().to(torch.int64)
        return plans[torch.arange(len(plans)), index][:, None].clone()

    def completed_step(self, age, terminated, cap):
        self.events.append(("completed", self.sim.counter, age.clone(), terminated.clone(), cap.clone()))
        return torch.zeros(ACTORS, dtype=torch.bool)

    def sample_reference(self, ids, history_length):
        self.reference_draws += 1
        self.events.append(("reference", self.sim.counter, ids.clone()))
        self.sampled_history = robot_state(self.states, len(ids) * history_length)
        return (robot_state(self.states, len(ids)), self.sampled_history,
                None, 100 + ids, torch.full_like(ids, 9))

    def bind_reset(self, ids, motion_ids, frames, source_xy, reset_xy, episode_ids):
        self.events.append(("bind", self.sim.counter, ids.clone()))
        self.bound = SimpleNamespace(ids=ids.clone(), episode_ids=episode_ids.clone(), motion_ids=motion_ids.clone())


class FakeForceView:
    def __init__(self, actor, names, sim, events):
        self.actor, self.sim, self.events = actor, sim, events
        self.sensor_count = 24
        self.sensor_paths = [f"/Fake/Actor_{actor}/{name}" for name in names]
        self.force = torch.zeros((24, 3), dtype=torch.float32)

    def get_net_contact_forces(self, *, dt):
        self.events.append(("normal", self.sim.counter, self.actor, dt))
        return self.force


def fixture():
    """Bypass SDK constructor admission; keep all exercised World methods intact."""
    deps = world_module._dependencies()
    _, states, _ = deps.adapter._dependencies()
    world = world_module.SharedGroupWorld.__new__(world_module.SharedGroupWorld)
    world.deps, world.device, world.num_envs = deps, torch.device("cpu"), ACTORS
    world.rows = deps.bridge.mixed_group_rows(ACTORS, device=world.device)
    world.events = events = []
    world.sim = FakeSim(events)
    world.scene = FakeScene(world.sim, events)
    world.assets = tuple(FakeAsset(slot, world.rows, world.sim, events) for slot in range(4))
    world.adapter = FakeAdapter(deps, world.assets, world.sim, events, states)
    world.curriculum = FakeReference(states, world.sim, events)
    world.mapping = deps.observations.PacerSMPLMapping()
    world.actor_ids = torch.arange(ACTORS, dtype=torch.int64)
    world.scene_ids = torch.empty_like(world.actor_ids)
    world.scene_ids[world.rows] = torch.arange(len(world.rows))[:, None]
    world.spawn_world_xyz = world.adapter.state.root_pos.clone()
    world.lower, world.upper = -torch.ones(69), torch.ones(69)
    world.ground_height_at = lambda xy: torch.zeros(xy.shape[:-1], dtype=xy.dtype, device=xy.device)
    world.terrain_sampler = world.ground_height_at
    world.grid_geometry = lambda snapshot, yaw, defined: (snapshot.robot_state.root_pos[:, :2].clone(), yaw.clone(),
                                                        torch.zeros(ACTORS, dtype=torch.float32))
    world.grid_geometry_contract = world_module._geometry_contract(
        {"grid_center": "root", "grid_heading": "root", "height_reference": "static_center_ground_z"})
    world.include_ego, world.episode_steps = False, 100
    names = deps.observations.PACER_BODY_NAMES
    views = tuple(FakeForceView(actor, names, world.sim, events) for actor in range(ACTORS))
    world.bundle = (views, tuple({"actor": actor, "sources": view.sensor_paths} for actor, view in enumerate(views)),
                    (names,) * ACTORS)
    world.age = torch.full((ACTORS,), 2, dtype=torch.int64)
    world.episode_ids = 11 + world.scene_ids * 10
    world.reference_ids, world.plan_age = torch.zeros_like(world.actor_ids), torch.zeros_like(world.actor_ids)
    world.plan_valid = torch.zeros(ACTORS, dtype=torch.bool)
    world.plans = torch.zeros((ACTORS, 151, 2))
    world.amp_history = torch.full((ACTORS, 10, 195), 7., dtype=torch.float32)
    world.amp_history_episode_ids = world.episode_ids[:, None].expand(-1, 10).clone()
    world.planar_history = torch.full((ACTORS, 91, 5), 7., dtype=torch.float32)
    world.history_available = torch.ones((ACTORS, 91), dtype=torch.bool)
    world.heading_history_defined = world.history_available.clone()
    world.last_defined_heading = torch.zeros(ACTORS)
    world.heading_context_valid = torch.ones(ACTORS, dtype=torch.bool)
    world.used_task_heading = world.last_defined_heading.clone()
    world.task_heading_defined = world.heading_context_valid.clone()
    world.singular_heading_fallback = torch.zeros(ACTORS, dtype=torch.bool)
    world.control_step, world._physics_step, world._faulted = 30, BASELINE, False
    world.current = world.adapter.snapshot()
    world._prepared = world._pending = world.last_completion = None
    world.statistics = dict.fromkeys(("actor_controls", "physical_falls", "nonfoot_normal_contact_terminations",
        "original_terminal_actor_controls", "collateral_truncated_actor_controls", "source_end_actor_controls",
        "cap_actor_controls", "reset_actor_rows", "reset_groups", "initial_reset_actor_rows"), 0)
    events.clear()
    return world


class SharedGroupWorldCPU(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.rng = torch.random.get_rng_state().clone()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(torch.equal(self.rng, torch.random.get_rng_state()))
        forbidden = ("isaaclab", "isaaclab_physx", "isaacsim", "omni", "pxr", "warp", "mujoco", "dm_control",
                     "CrowdSim.humanoid.amp", "CrowdSim.humanoid.training")
        self.assertFalse(any(name == prefix or name.startswith(prefix + ".")
                             for name in sys.modules for prefix in forbidden))

    def test_four_substeps_matched_signed_power_and_end_normal(self):
        world = fixture()
        pre = world.prepare_control()
        actions = torch.zeros((ACTORS, 69), dtype=torch.float32)
        actions[:, world.deps.observations.PACER_DOF_NAMES.index("L_Hip_x")] = .25
        pending = world.step(actions)
        self.assertIs(pending.pre, pre)
        self.assertEqual((pending.physics_step_before, pending.physics_step_after), (137, 141))
        self.assertEqual((pending.control_step_before, pending.control_step_after), (30, 31))
        for index in range(4):
            events = [event for event in world.events if event[0] in {"targets", "write", "step", "update"}]
            self.assertEqual(events[index * 4:index * 4 + 4], [("targets", 137 + index), ("write", 137 + index),
                              ("step", 138 + index), ("update", 138 + index, world_module.DT)])
        reads = [event for event in world.events if event[0] in {"effort", "velocity"}]
        self.assertEqual(reads, [(kind, 138 + index, slot) for index in range(4) for slot in range(4)
                                for kind in ("effort", "velocity")])
        sign = torch.where(world.actor_ids % 2 == 0, 1., -1.)
        expected = sign * (world.actor_ids + 1).to(torch.float32) * 2415 * 10
        self.assertTrue(torch.equal(pending.signed_power_w, expected))
        self.assertFalse(torch.equal(pending.signed_power_w, expected * 2))  # Fourth-only mean is wrong.
        effort = (world.actor_ids + 1).to(torch.float32)[:, None] * torch.arange(1, 70)[None] * 4
        self.assertTrue(torch.equal(pending.snapshot.robot_state.dof_forces, effort))
        target = world.deps.action_to_pd(actions, world.lower, world.upper, dof_names=world.mapping.dof_names, clip=False)
        self.assertEqual(len(world.adapter.targets), 4)
        self.assertTrue(all(torch.equal(value, target) for value in world.adapter.targets))
        self.assertEqual([event for event in world.events if event[0] == "normal"],
                         [("normal", 141, actor, 1 / 120) for actor in range(ACTORS)])
        self.assertEqual(pending.normal_force_metadata["caller_physics_step"], 141)
        self.assertEqual(pending.normal_force_metadata["caller_control_step"], 31)
        self.assertFalse(bool(pending.reset_mask.any()))
        self.assertTrue(torch.equal(pending.post.amp_history[:, 1:], pre.amp_history[:, :-1]))
        self.assertEqual((pre.self_obs.shape, pre.task_obs.shape), ((ACTORS, 357), (ACTORS, 12308)))

    def test_terminal_archive_before_whole_group_history_reset(self):
        world = fixture()
        world.adapter.state.rigid_body_pos[0, :, 2] -= .7  # Synthetic true height fall.
        yaw = torch.tensor(.75, dtype=torch.float32)
        world.adapter.state.rigid_body_rot[1] = torch.stack((yaw * 0, yaw * 0, (yaw / 2).sin(), (yaw / 2).cos()))
        world.bundle[0][1].force[world.deps.observations.PACER_BODY_NAMES.index("Torso"), 0] = 51
        world.current = world.adapter.snapshot()
        world._select_task_heading(torch.zeros(ACTORS, dtype=torch.bool))
        old_episodes = world.episode_ids.clone()
        world.prepare_control()
        pending = world.step(torch.zeros((ACTORS, 69)))
        mask = torch.zeros(ACTORS, dtype=torch.bool)
        mask[[0, 1]] = True
        self.assertTrue(torch.equal(pending.terminated, mask))
        self.assertEqual(pending.physical_fall.nonzero().flatten().tolist(), [0])
        self.assertEqual(pending.nonfoot_normal_contact.contact_flag.nonzero().flatten().tolist(), [1])
        self.assertFalse(bool(pending.source_end_truncated.any() | pending.cap_truncated.any()))
        self.assertEqual(pending.collateral_truncated.nonzero().flatten().tolist(), [4, 5])
        self.assertTrue(torch.equal(pending.truncated, pending.collateral_truncated))
        values = torch.arange(ACTORS, dtype=torch.float32) + .25  # Terminal values need not be zero.

        def archive(actual, owned_values):
            world.events.append(("archive", world.sim.counter))
            self.assertIs(actual, pending)
            self.assertTrue(torch.equal(actual.post.episode_ids, old_episodes))
            self.assertTrue(torch.equal(world.episode_ids, old_episodes))
            self.assertTrue(torch.equal(owned_values, values))
            self.assertNotEqual(owned_values.data_ptr(), values.data_ptr())
            self.assertEqual(world.curriculum.reference_draws, 0)
            self.assertFalse(any(event[0] in {"adapter_reset", "scene_reset", "forward", "bind"} for event in world.events))

        ids = world.complete_transition(values, archive)
        self.assertEqual(ids.tolist(), [0, 1, 4, 5])
        self.assertEqual((world.sim.counter, world._physics_step, world.control_step), (141, 141, 31))
        tail = [event[0] for event in world.events if event[0] in {"archive", "reference", "adapter_reset", "scene_reset", "forward", "snapshot", "bind"}]
        self.assertEqual(tail[-7:], ["archive", "reference", "adapter_reset", "scene_reset", "forward", "snapshot", "bind"])
        reset_events = [event for event in world.events if event[0] in {"adapter_reset", "scene_reset"}]
        self.assertEqual(reset_events[0][2].tolist(), ids.tolist())
        self.assertEqual(reset_events[1][2].tolist(), [0])
        others = world.rows[1]
        self.assertTrue(torch.equal(world.episode_ids[ids], old_episodes[ids] + 1))
        self.assertTrue(torch.equal(world.episode_ids[others], old_episodes[others]))
        self.assertTrue(torch.equal(world.amp_history_episode_ids[ids], world.episode_ids[ids, None].expand(-1, 10)))
        self.assertTrue(torch.equal(world.amp_history[others], pending.post.amp_history[others]))
        self.assertTrue(torch.equal(world.amp_history[ids, 0], world.deps.observations.build_amp_observation(world.current.robot_state, mapping=world.mapping)[ids]))
        reference_amp = world.deps.observations.build_amp_observation(world.curriculum.sampled_history, mapping=world.mapping).reshape(len(ids), 10, 195)
        self.assertTrue(torch.equal(world.amp_history[ids, 1:], reference_amp[:, 1:]))
        self.assertTrue(bool((world.age[ids] == 0).all() & (world.plan_age[ids] == 0).all()))
        self.assertTrue(bool((world.age[others] == 3).all() & (world.plan_age[others] == 1).all()))
        self.assertFalse(bool(world.plan_valid[ids].any()))
        self.assertTrue(bool(world.plan_valid[others].all()))
        self.assertTrue(bool((world.plans[ids] == 0).all()))
        self.assertTrue(bool(world.heading_context_valid[ids].all() & world.task_heading_defined[ids].all()))
        self.assertFalse(bool(world.singular_heading_fallback[ids].any()))
        self.assertTrue(bool((world.used_task_heading[ids] == 0).all()))
        self.assertTrue(bool((world.history_available[ids].sum(1) == 1).all()))
        self.assertTrue(bool((world.heading_history_defined[ids].sum(1) == 1).all()))
        self.assertTrue(torch.equal(world.planar_history[ids], world._planar(world.current.robot_state)[ids, None].expand(-1, 91, -1)))
        self.assertEqual((world.curriculum.reference_draws, world.curriculum.plan_calls), (1, 1))
        self.assertTrue(torch.equal(world.curriculum.bound.episode_ids, world.episode_ids[ids]))
        self.assertEqual((world.statistics["physical_falls"], world.statistics["nonfoot_normal_contact_terminations"],
                          world.statistics["collateral_truncated_actor_controls"], world.statistics["reset_groups"]), (1, 1, 2, 1))
        self.assertIsNone(world._pending)

    def test_exact_singular_fallen_heading_continuation_only(self):
        world = fixture()
        angle = torch.tensor(.75, dtype=torch.float32)
        world.current.robot_state.rigid_body_rot[0, 0] = torch.stack((angle * 0, angle * 0, (angle / 2).sin(), (angle / 2).cos()))
        world._select_task_heading(torch.zeros(ACTORS, dtype=torch.bool))  # Real defined context in current episode.
        prior = world.last_defined_heading.clone()
        q = torch.tensor((.5, .5, -.5, .5), dtype=torch.float32)
        self.assertEqual(q.square().sum().item(), 1.)
        self.assertEqual((2*q[3].square()-1+2*q[0].square()).item(), 0.)
        self.assertEqual((2*q[3]*q[2]+2*q[0]*q[1]).item(), 0.)
        world.current.robot_state.rigid_body_rot[0, 0] = q
        original = world.current.robot_state.root_rot.clone()
        fallen = torch.zeros(ACTORS, dtype=torch.bool)
        fallen[0] = True
        world._select_task_heading(fallen)
        self.assertTrue(torch.equal(world.current.robot_state.root_rot, original))
        self.assertEqual(world.singular_heading_fallback.nonzero().flatten().tolist(), [0])
        self.assertFalse(world.task_heading_defined[0].item())
        self.assertTrue(torch.equal(world.used_task_heading[:1], prior[:1]))
        self.assertTrue(bool(world.task_heading_defined[1:].all()))
        with self.assertRaisesRegex(ValueError, "actual fall"):
            world._select_task_heading(torch.zeros(ACTORS, dtype=torch.bool))
        world.heading_context_valid[0] = False  # No established context for this episode.
        with self.assertRaisesRegex(ValueError, "prior defined heading"):
            world._select_task_heading(fallen)
        self.assertTrue(torch.equal(world.current.robot_state.root_rot, original))

    def test_BaseException_fault_and_no_retry_after_side_effects(self):
        with self.subTest(stage="prepare/replan KeyboardInterrupt"):
            world = fixture()
            signal = KeyboardInterrupt("synthetic reference interruption")
            world.curriculum.plan_interrupt = signal
            with self.assertRaises(KeyboardInterrupt) as caught:
                world.prepare_control()
            self.assertIs(caught.exception, signal)
            self.assertTrue(world._faulted)
            self.assertEqual((world.curriculum.plan_calls, world.sim.counter), (1, 137))
            with self.assertRaisesRegex(RuntimeError, "faulted"):
                world.prepare_control()
            self.assertEqual(world.curriculum.plan_calls, 1)
        with self.subTest(stage="complete/archive SystemExit"):
            world = fixture()
            world.adapter.state.rigid_body_pos[0, :, 2] -= .7
            world.current = world.adapter.snapshot()
            world.prepare_control()
            world.step(torch.zeros((ACTORS, 69)))
            episode = world.episode_ids.clone()
            signal = SystemExit("synthetic archive interruption")
            calls = []

            def interrupted_archive(pending, values):
                calls.append((pending.control_step_after, world.sim.counter))
                raise signal

            with self.assertRaises(SystemExit) as caught:
                world.complete_transition(torch.ones(ACTORS), interrupted_archive)
            self.assertIs(caught.exception, signal)
            self.assertTrue(world._faulted)
            self.assertEqual(calls, [(31, 141)])
            self.assertTrue(torch.equal(world.episode_ids, episode))
            self.assertIsNotNone(world._pending)
            self.assertEqual(world.curriculum.reference_draws, 0)
            self.assertFalse(any(event[0] in {"adapter_reset", "scene_reset", "forward", "bind"} for event in world.events))
            with self.assertRaisesRegex(RuntimeError, "faulted"):
                world.complete_transition(torch.ones(ACTORS), interrupted_archive)
            self.assertEqual(calls, [(31, 141)])


if __name__ == "__main__":
    unittest.main()
