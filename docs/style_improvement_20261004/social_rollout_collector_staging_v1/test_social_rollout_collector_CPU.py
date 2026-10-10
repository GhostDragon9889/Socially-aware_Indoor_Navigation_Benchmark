"""Four future fake-boundary tests of the intact frozen collector.collect.

UNEXECUTED at authoring. Constructor admission is deliberately bypassed with
__new__: FakeWorld clocks/reset and FakeTrainer act/AMP/value are synthetic.
No World/trainer, NN, SDK, dataset or checkpoint module is constructed/loaded.
The real collector, pinned reward helpers, planar AST and rollout validator run
only when root later authorizes these CPU tests. No value/physics quality claim.
"""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Future fake-interface execution requires explicitly hidden CUDA")

SOURCE = Path(__file__).with_name("social_rollout_collector.py")
SOURCE_SHA = "f7d2727051d92da5efc0fc4eae37338c88152aca3904971745d4daa7389556fd"
info, source = SOURCE.lstat(), SOURCE.read_bytes()
if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222
        or len(source) != 27252 or hashlib.sha256(source).hexdigest() != SOURCE_SHA):
    raise RuntimeError("Frozen collector whole-byte identity changed")
collector_module = ModuleType("_frozen_social_collector_for_fake_CPU")
collector_module.__file__ = str(SOURCE)
sys.modules[collector_module.__name__] = collector_module
exec(compile(source, str(SOURCE), "exec"), collector_module.__dict__)
torch = collector_module.torch
ACTORS, CONTROL, PHYSICS = 4, 10, 137
GEOMETRY = {"grid_center": "root", "grid_heading": "root", "height_reference": "root_z"}


class FakeConfig:
    def __init__(self):
        self.actor_count, self.include_ego = ACTORS, False
        self.discriminator_input, self.amp_normalization = "state_action", "per_frame"
        self.direction_mode = "backwards_penalty"
        self.amp_reward_weight, self.task_reward_weight = 2., .5
        self.grid_center, self.grid_heading, self.height_reference = "root", "root", "root_z"

    def validate(self):
        if self.actor_count != ACTORS or self.discriminator_input != "state_action":
            raise ValueError("Synthetic boundary dimensions changed")

    def to_document(self):
        return deepcopy(vars(self))


class FakeRMS:
    """Tensor holders only; no Normalizer/NN is instantiated or updated."""
    def __init__(self, dimension):
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.variance = torch.ones(dimension, dtype=torch.float64)
        self.count = torch.zeros((), dtype=torch.float64)
        self.clip = 5.

    def named_buffers(self):
        return iter((name, getattr(self, name)) for name in ("mean", "variance", "count"))


class FakeModel:
    """No nn.Module, layers, parameter sampling or forward function."""
    def __init__(self):
        self.training = False
        self.weight = torch.tensor([1., 2.], dtype=torch.float32)
        self.buffer = torch.tensor([3.], dtype=torch.float32)

    def named_parameters(self):
        return iter((("boundary_weight", self.weight),))

    def named_buffers(self):
        return iter((("boundary_buffer", self.buffer),))

    def named_modules(self):
        return iter((("", self),))


class FakeTrainer:
    def __init__(self, world):
        self.world, self.device, self.config = world, torch.device("cpu"), FakeConfig()
        self.model, self.update_count, self.physical_transition_count = FakeModel(), 0, 0
        self.self_normalizer, self.task_normalizer = FakeRMS(357), FakeRMS(12308)
        self.amp_normalizer, self.value_normalizer = FakeRMS(195), FakeRMS(1)
        # Same canonical neutral hand/toe triples as the sealed trainer config.
        self.action_mask = torch.ones(69, dtype=torch.float32)
        for start in (9, 21, 51, 66):
            self.action_mask[start:start + 3] = 0
        self.act_inputs, self.amp_inputs, self.value_inputs = [], [], []

    def act(self, self_obs, task_obs, *, deterministic):
        if deterministic is not False or self.world._prepared is None:
            raise AssertionError("Expected stochastic-policy interface before physics")
        self.world.events.append("act")
        self.act_inputs.append((self_obs.clone(), task_obs.clone()))
        raw = torch.linspace(-1.4, 1.4, 69).expand(ACTORS, -1).clone()
        return {"actions": raw, "applied_actions": raw.clamp(-1, 1) * self.action_mask,
                "mean_actions": raw * .5, "log_prob": torch.full((ACTORS,), -.7),
                "values": self.world.control_step + torch.arange(ACTORS, dtype=torch.float32)}

    def amp_reward(self, history, applied_actions):
        self.world.events.append("amp")
        if self.world._pending is not None:
            raise AssertionError("AMP must precede this control's fake physics")
        self.amp_inputs.append((history.clone(), applied_actions.clone()))
        return torch.arange(1, ACTORS + 1, dtype=torch.float32) * .1

    def value(self, self_obs, task_obs):
        self.world.events.append("value")
        pending = self.world._pending
        if (pending is None or not torch.equal(self.world.episode_ids, pending.post.episode_ids)
                or not torch.equal(self_obs, pending.post.self_obs) or not torch.equal(task_obs, pending.post.task_obs)):
            raise AssertionError("All nextV inputs must be terminal post state before reset")
        self.value_inputs.append((self_obs.clone(), task_obs.clone(), self.world.episode_ids.clone()))
        return 100 + self.world.control_step + torch.arange(ACTORS, dtype=torch.float32)


class FakeWorld:
    """Four synthetic actor rows, fake stamps/reset; no World algorithm or SDK."""
    def __init__(self, *, bad_clock_on=None):
        self.device, self.num_envs, self.include_ego = torch.device("cpu"), ACTORS, False
        self.grid_geometry_contract = dict(GEOMETRY)
        self.control_step, self._physics_step = CONTROL, PHYSICS
        self.age, self.plan_age = torch.full((ACTORS,), 2, dtype=torch.int64), torch.full((ACTORS,), 5, dtype=torch.int64)
        self.episode_ids = torch.full((ACTORS,), 11, dtype=torch.int64)
        self.amp_history = torch.full((ACTORS, 10, 195), 7., dtype=torch.float32)
        self.amp_history_episode_ids = self.episode_ids[:, None].expand(-1, 10).clone()
        self._faulted = False
        self._pending = self._prepared = None
        self.events, self.step_calls, self.reset_calls = [], 0, 0
        self.bad_clock_on = bad_clock_on

    def _counter(self):
        return self._physics_step

    def observation(self):
        return SimpleNamespace(self_obs=torch.full((ACTORS, 357), float(self.control_step)),
            task_obs=torch.full((ACTORS, 12308), float(self.control_step + 10)),
            amp_history=self.amp_history.clone(), history_episode_ids=self.amp_history_episode_ids.clone(),
            episode_ids=self.episode_ids.clone(), age=self.age.clone(), plan_age=self.plan_age.clone(),
            control_step=self.control_step, physics_step=self._physics_step)

    def prepare_control(self):
        self.events.append("prepare")
        self._prepared = self.observation()
        return self._prepared

    def step(self, applied_actions):
        self.events.append("step")
        self.step_calls += 1
        pre, before = self._prepared, self._physics_step
        self.control_step += 1
        self._physics_step += 3 if self.bad_clock_on == self.step_calls else 4
        self.age += 1
        self.plan_age += 1
        self.amp_history = torch.cat((torch.full((ACTORS, 1, 195), float(self.control_step)), self.amp_history[:, :-1]), 1)
        self.amp_history_episode_ids = self.episode_ids[:, None].expand(-1, 10).clone()
        post = self.observation()
        position = torch.tensor([.4, .8, 1.2]).expand(ACTORS, -1).clone()
        quat = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)]).expand(ACTORS, -1).clone()
        velocity = torch.tensor([-1., -2., 0.]).expand(ACTORS, -1).clone()
        state = SimpleNamespace(root_pos=position, root_rot=quat, root_vel=velocity,
            dof_forces=torch.full((ACTORS, 69), 100.), dof_vel=torch.full((ACTORS, 69), 10.),
            dof_pos=torch.zeros((ACTORS, 69)), rigid_body_pos=position[:, None].expand(-1, 24, -1).clone(),
            rigid_body_rot=quat[:, None].expand(-1, 24, -1).clone(),
            rigid_body_vel=velocity[:, None].expand(-1, 24, -1).clone(), rigid_body_ang_vel=torch.zeros((ACTORS, 24, 3)))
        native_root = SimpleNamespace(root_pos=position.clone(), root_rot=quat.clone(), root_vel=velocity.clone(),
                                      root_ang_vel=torch.zeros((ACTORS, 3)))
        terminal = torch.zeros(ACTORS, dtype=torch.bool)
        terminal[0] = self.step_calls == 1
        collateral = torch.full((ACTORS,), self.step_calls == 1, dtype=torch.bool) & ~terminal
        no_flags = torch.zeros(ACTORS, dtype=torch.bool)
        self._pending = SimpleNamespace(pre=pre, post=post, snapshot=SimpleNamespace(robot_state=state, root_state=native_root),
            applied_actions=applied_actions.clone(), target_xy=position[:, :2] + torch.tensor([.2, 0.]),
            signed_power_w=torch.tensor([4., 5., -2., 0.]), terminated=terminal, truncated=collateral,
            reset_mask=terminal | collateral, collateral_truncated=collateral, physical_fall=terminal.clone(),
            nonfoot_normal_contact=SimpleNamespace(contact_flag=no_flags.clone(), resultant_force_w=torch.zeros((ACTORS, 3)),
                                                   magnitude_n=torch.zeros(ACTORS)),
            source_end_truncated=no_flags.clone(), cap_truncated=no_flags.clone(),
            control_step_before=pre.control_step, control_step_after=self.control_step,
            physics_step_before=before, physics_step_after=self._physics_step,
            actual_root_quat_xyzw=quat.clone(), used_task_heading=torch.full((ACTORS,), .7),
            task_heading_defined=torch.ones(ACTORS, dtype=torch.bool), singular_heading_fallback=no_flags.clone())
        self._prepared = None
        return self._pending

    def complete_transition(self, next_values, archive):
        self.events.append("complete")
        pending = self._pending
        archive(pending, next_values.clone())
        if bool(pending.reset_mask.any()):
            self.events.append("reset")
            self.reset_calls += 1
            self.episode_ids += 1
            self.age.zero_()
            self.plan_age.zero_()
            self.amp_history.fill_(99.)
            self.amp_history_episode_ids = self.episode_ids[:, None].expand(-1, 10).clone()
        self._pending = None


def fixture(**kwargs):
    """Deliberately bypass real class/admission, keep entire collect untouched."""
    world = FakeWorld(**kwargs)
    trainer = FakeTrainer(world)
    collector = collector_module.SocialRolloutCollector.__new__(collector_module.SocialRolloutCollector)
    # This exact loader checks pinned source bytes but imports no World/trainer
    # or amp/training package; only rewards/contracts and planar AST execute.
    collector.reward_api, collector.contracts, collector.original_planar = collector_module._dependencies()
    collector.world, collector.trainer, collector.device, collector.actors = world, trainer, world.device, ACTORS
    collector._faulted = collector._collecting = False
    collector.failed_steps = None
    return collector, world, trainer


class CollectorFakeCPUChecks(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.rng = torch.random.get_rng_state().clone()
        self._forbidden_modules()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(torch.equal(self.rng, torch.random.get_rng_state()))
        self._forbidden_modules()

    def _forbidden_modules(self):
        forbidden = ("isaaclab", "isaaclab_physx", "isaacsim", "omni", "pxr", "warp", "mujoco", "dm_control",
                     "CrowdSim.humanoid.amp", "CrowdSim.humanoid.training")
        self.assertFalse(any(name == prefix or name.startswith(prefix + ".")
                             for name in sys.modules for prefix in forbidden))

    def test_intact_collect_terminal_order_raw_actions_rewards_and_stacks(self):
        collector, world, trainer = fixture()
        archives = []

        def archive(step):
            world.events.append("archive")
            self.assertTrue(torch.equal(step["diagnostics"]["post_episode_ids"], world.episode_ids))
            archives.append(step)

        result = collector.collect(2, external_archive=archive)
        rollout, diagnostic = result["rollout"], result["diagnostics"]
        self.assertEqual(world.events, ["prepare", "act", "amp", "step", "value", "complete", "archive", "reset",
                                        "prepare", "act", "amp", "step", "value", "complete", "archive"])
        self.assertEqual(tuple(rollout["self_obs"].shape), (2, ACTORS, 357))
        self.assertEqual(tuple(rollout["task_obs"].shape), (2, ACTORS, 12308))
        self.assertEqual(tuple(rollout["amp_history"].shape), (2, ACTORS, 10, 195))
        self.assertTrue(bool((rollout["actions"].abs() > 1).any()))
        self.assertTrue(torch.equal(rollout["log_prob"], torch.full((2, ACTORS), -.7)))
        self.assertTrue(torch.equal(rollout["applied_actions"], rollout["actions"].clamp(-1, 1) * trainer.action_mask))
        for index, (history, actions) in enumerate(trainer.amp_inputs):
            self.assertTrue(torch.equal(history, rollout["amp_history"][index]))
            self.assertTrue(torch.equal(actions, rollout["applied_actions"][index]))
            self.assertFalse(torch.equal(history, diagnostic["post_amp_history"][index]))
        self.assertTrue(torch.equal(rollout["next_values"][0], 111 + torch.arange(ACTORS, dtype=torch.float32)))
        self.assertTrue(torch.equal(rollout["episode_ids"][1], rollout["episode_ids"][0] + 1))
        self.assertTrue(torch.equal(rollout["terminated"][0], torch.tensor([True, False, False, False])))
        self.assertTrue(torch.equal(rollout["truncated"][0], torch.tensor([False, True, True, True])))
        self.assertTrue(torch.allclose(rollout["forward_velocity"], torch.full((2, ACTORS), -2.), atol=1e-6, rtol=0))
        expected_energy = -.0005 * torch.tensor([4., 5., -2., 0.])
        expected_task = math.exp(-2 * .2**2) + expected_energy - (1 - math.exp(-2 / 1.8))
        expected_reward = 2 * torch.arange(1, ACTORS + 1, dtype=torch.float32) * .1 + .5 * expected_task
        self.assertTrue(torch.allclose(rollout["energy_rewards"][0], expected_energy, atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(rollout["rewards"][0], expected_reward, atol=1e-6, rtol=0))
        advantages, _ = collector.contracts.generalized_advantages(rollout["rewards"], rollout["values"], rollout["next_values"],
            rollout["terminated"], rollout["truncated"])
        # Group boundaries cut carry; collateral truncations keep terminal-V bootstrap.
        first_delta = rollout["rewards"][0] + .99 * rollout["next_values"][0] * ~rollout["terminated"][0] - rollout["values"][0]
        self.assertTrue(torch.equal(advantages[0], first_delta))
        self.assertEqual(result["metadata"]["external_archive_returned_controls"], 2)
        self.assertFalse(result["metadata"]["archive_independently_verified"])
        archives[0]["rollout"]["actions"].fill_(9.)
        self.assertFalse(bool((rollout["actions"] == 9).any()))
        for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
            self.assertEqual(float(getattr(trainer, name).count), 0.)
        self.assertFalse(collector._faulted or world._faulted)

    def test_external_RMS_mutation_rejects_after_owned_partial_step(self):
        collector, world, trainer = fixture()

        def archive(step):
            step["rollout"]["actions"].fill_(9.)
            trainer.task_normalizer.mean[42] += 1

        with self.assertRaisesRegex(RuntimeError, "RMS buffers changed"):
            collector.collect(2, external_archive=archive)
        self.assertTrue(collector._faulted and world._faulted)
        self.assertFalse(collector._collecting)
        self.assertEqual(len(collector.failed_steps["rollout_steps"]), 1)
        self.assertFalse(bool((collector.failed_steps["rollout_steps"][0]["actions"] == 9).any()))
        self.assertEqual(world.reset_calls, 0)

    def test_interrupt_retains_partial_and_blocks_reuse(self):
        collector, world, _ = fixture()

        def archive(step):
            raise KeyboardInterrupt("synthetic archive interruption")

        with self.assertRaises(KeyboardInterrupt):
            collector.collect(2, external_archive=archive)
        self.assertTrue(collector._faulted and world._faulted)
        self.assertFalse(collector._collecting)
        self.assertEqual(len(collector.failed_steps["rollout_steps"]), 1)
        self.assertEqual(len(collector.failed_steps["diagnostic_steps"]), 1)
        self.assertEqual(world.reset_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "nonfaulted"):
            collector.collect(1)

    def test_bad_second_clock_rejects_no_partial_success(self):
        collector, world, _ = fixture(bad_clock_on=2)
        with self.assertRaisesRegex(RuntimeError, "four-substep identity"):
            collector.collect(2)
        self.assertTrue(collector._faulted and world._faulted)
        self.assertEqual(len(collector.failed_steps["rollout_steps"]), 1)
        self.assertEqual(len(collector.failed_steps["diagnostic_steps"]), 1)
        self.assertEqual(world.events.count("value"), 1)


if __name__ == "__main__":
    unittest.main()
