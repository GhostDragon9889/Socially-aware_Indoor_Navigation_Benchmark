"""Unexecuted fake-bank tests; __new__ bypasses SDK/corpus deserialization.

Real original AMASS class methods sample synthetic strided tensors. The fixed
manifest is metadata only; bank files/AMASS arrays/NN/SDK are never loaded.
"""
import ast
import __future__
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Explicit hidden CUDA is required")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_module(name, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


expert = load_module("_social_expert_fake_CPU", HERE / "expert_batch.py")
world_path = ROOT / expert.PINS["world"][0]
if hashlib.sha256(world_path.read_bytes()).hexdigest() != expert.PINS["world"][2]:
    raise RuntimeError("Frozen World changed")
world_module = load_module("_social_expert_world_fake_CPU", world_path)
torch = world_module.torch
corpus_path = ROOT / expert.PINS["corpus"][0]
corpus_module = ModuleType("_original_AMASS_class_for_fake_CPU")
corpus_module.__file__ = str(corpus_path)
sys.modules[corpus_module.__name__] = corpus_module
klass = next(node for node in ast.parse(corpus_path.read_bytes()).body
             if isinstance(node, ast.ClassDef) and node.name == "AMASSCorpus")
corpus_module.__dict__.update(torch=torch, SimpleNamespace=SimpleNamespace)
exec(compile(ast.Module(body=[klass], type_ignores=[]), str(corpus_path), "exec",
             flags=__future__.annotations.compiler_flag), corpus_module.__dict__)


def fixture(*, reverse_dofs=False):
    deps = world_module._dependencies()
    document = json.loads(expert.MANIFEST.read_bytes())
    entry = document["splits"]["train"]
    corpus = corpus_module.AMASSCorpus.__new__(corpus_module.AMASSCorpus)
    corpus.manifest_path, corpus.manifest, corpus.entry = expert.MANIFEST, document, entry
    corpus.split, corpus.device = "train", torch.device("cpu")
    corpus.ids = torch.tensor([row["motion_id"] for row in entry["clips"]], dtype=torch.int64)
    count = max(corpus.ids.tolist()) + 1
    frames = torch.zeros(count, dtype=torch.int64)
    frames[corpus.ids] = torch.tensor([row["frames"] for row in entry["clips"]])
    starts = frames.cumsum(0) - frames
    total = int(frames.sum())
    stamp = torch.arange(total, dtype=torch.float32) / 1000
    position = torch.stack((stamp, stamp * 0, stamp * 0 + 1), -1)[:, None].expand(-1, 24, -1)
    quaternion = torch.tensor((0., 0., 0., 1.)).expand(total, 24, 4)
    coordinates = torch.arange(69, dtype=torch.float32) / 10
    dof_order = torch.arange(69).flip((0,)) if reverse_dofs else torch.arange(69)
    corpus.data = {"motion_num_frames": frames, "length_starts": starts, "gts": position, "grs": quaternion,
        "gvs": stamp[:, None, None].expand(-1, 24, 3), "gavs": torch.zeros(3).expand(total, 24, 3),
        "dps": coordinates[dof_order].expand(total, 69), "dvs": torch.zeros(69).expand(total, 69)}
    corpus.generator = torch.Generator(device="cpu").manual_seed(7)  # Private synthetic fixture only.
    corpus._convert_dof_velocity = False
    corpus.dof_velocity_frame, corpus.stored_dof_velocity_frame, corpus.output_dof_velocity_frame = "legacy_stored", "child_joint", "child_joint"
    world = world_module.SharedGroupWorld.__new__(world_module.SharedGroupWorld)
    world.deps, world.device = deps, corpus.device
    world.curriculum = SimpleNamespace(corpus=corpus)
    names = tuple(deps.observations.PACER_DOF_NAMES[index] for index in dof_order.tolist())
    world.mapping = deps.observations.PacerSMPLMapping(dof_names=names)
    world.lower, world.upper = -torch.ones(69), torch.ones(69)
    world._faulted, world._physics_step = False, 137
    world._prepared = world._pending = None
    world.sim = SimpleNamespace(get_physics_step_count=lambda: 137)
    return world, corpus


class SocialAMASSExpertCPU(unittest.TestCase):
    def setUp(self):
        self.rng = torch.random.get_rng_state().clone()
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertTrue(torch.equal(self.rng, torch.random.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
        forbidden = ("isaaclab", "isaaclab_physx", "isaacsim", "omni", "pxr", "warp", "mujoco", "dm_control",
                     "CrowdSim.humanoid.amp", "CrowdSim.humanoid.training")
        self.assertFalse(any(name == prefix or name.startswith(prefix + ".")
                             for name in sys.modules for prefix in forbidden))

    def test_one_original_draw_and_newest_first_history(self):
        world, corpus = fixture()
        adapter = expert.SocialAMASSExpert(world, corpus)
        state = corpus.generator.get_state().clone()
        expected = corpus.sample_reference(3, 10)
        expected_rng = corpus.generator.get_state().clone()
        corpus.generator.set_state(state)
        result = adapter.sample(3)
        self.assertTrue(torch.equal(corpus.generator.get_state(), expected_rng))
        self.assertTrue(torch.equal(result.motion_ids, expected[3]))
        self.assertTrue(torch.equal(result.frame_indices, expected[4]))
        self.assertTrue(torch.equal(result.history.rigid_body_vel, expected[1].rigid_body_vel))
        absolute = corpus.data["length_starts"][result.motion_ids] + result.frame_indices
        velocities = result.history.root_vel.reshape(3, 10, 3)
        expected_stamp = (absolute[:, None] - torch.arange(10)) / 1000
        self.assertTrue(torch.equal(velocities[..., 0], expected_stamp))
        self.assertEqual((result.features.shape, result.actions.shape, result.batches, result.windows), ((3, 10, 195), (3, 69), 1, 3))

    def test_named_next_frame_action_proxy_and_frozen57(self):
        world, corpus = fixture(reverse_dofs=True)
        result = expert.SocialAMASSExpert(world, corpus).sample(2)
        scales = torch.full((69,), 1.2)
        for name in ("L_Knee_y", "R_Knee_y"):
            scales[world.deps.observations.PACER_DOF_NAMES.index(name)] = 5.
        expected = (torch.arange(69, dtype=torch.float32) / 10 / scales).clamp(-1, 1)
        expected[~result.effective_action_mask] = 0
        self.assertTrue(torch.equal(result.actions, expected[None].expand(2, -1)))
        self.assertEqual(int(result.effective_action_mask.sum()), 57)
        self.assertTrue(bool((result.actions[:, ~result.effective_action_mask] == 0).all()))
        self.assertTrue(bool((result.actions.abs() == 1).any()))

    def test_owned_outputs_no_clock_or_bank_mutation(self):
        world, corpus = fixture()
        adapter = expert.SocialAMASSExpert(world, corpus)
        result = adapter.sample(1)
        source_value = corpus.data["gvs"][0, 0, 0].clone()
        result.history.rigid_body_vel.fill_(99)
        result.history.root_vel.fill_(88)
        result.next_position_common.fill_(77)
        result.actions.fill_(66)
        self.assertTrue(torch.equal(corpus.data["gvs"][0, 0, 0], source_value))
        self.assertTrue(torch.equal(corpus.data["dps"][0], torch.arange(69, dtype=torch.float32) / 10))
        self.assertEqual(world._physics_step, 137)
        description = adapter.describe()
        self.assertFalse(description["author_clip_list"])
        self.assertFalse(description["bank"]["sha256_recomputed_here"])
        description["sampled_batches"] = 999
        self.assertEqual(adapter.describe()["sampled_batches"], 1)
        adapter = expert.SocialAMASSExpert(world, corpus)
        interrupt = KeyboardInterrupt("owned expert packet")
        calls = 0

        def interrupted_packet(**fields):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise interrupt
            return SimpleNamespace(**fields)

        with patch.object(expert, "SimpleNamespace", side_effect=interrupted_packet):
            with self.assertRaises(KeyboardInterrupt) as caught:
                adapter.sample(1)
        self.assertIs(caught.exception, interrupt)
        self.assertEqual(calls, 2)
        self.assertTrue(adapter._faulted)
        self.assertEqual((adapter.batches, adapter.windows), (0, 0))
        rng_after_interrupt = corpus.generator.get_state().clone()
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            adapter.sample(1)
        self.assertTrue(torch.equal(corpus.generator.get_state(), rng_after_interrupt))

    def test_train_and_explicit_sampler_admission_before_draw(self):
        world, corpus = fixture()
        rng = corpus.generator.get_state().clone()
        corpus.split = "validation"
        with self.assertRaisesRegex(ValueError, "train corpus"):
            expert.SocialAMASSExpert(world, corpus)
        corpus.split = "train"
        with self.assertRaisesRegex(ValueError, "explicit prepared sampler"):
            expert.SocialAMASSExpert(world, corpus, sampling_config={"mode": "stratified_windows"})
        with self.assertRaisesRegex(TypeError, "ExpertReferenceSampler"):
            expert.SocialAMASSExpert(world, corpus, expert_sampler=SimpleNamespace(), sampling_config={}, sampling_provenance={})
        adapter = expert.SocialAMASSExpert(world, corpus)
        with self.assertRaisesRegex(ValueError, "exact integer"):
            adapter.sample(True)
        world._prepared = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "idle World"):
            expert.SocialAMASSExpert(world, corpus)
        world._prepared, world._pending = None, SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "idle World"):
            adapter.sample(1)
        world._pending = None
        world.lower[0] -= .1
        with self.assertRaisesRegex(ValueError, "PD limits changed"):
            adapter.sample(1)
        world.lower = adapter._host.lower.clone()
        corpus.generator = torch.Generator(device="cpu")
        with self.assertRaisesRegex(ValueError, "generator/configuration"):
            adapter.sample(1)
        corpus.generator = adapter._generator
        corpus.manifest = dict(corpus.manifest, author_clip_list=True)
        with self.assertRaisesRegex(ValueError, "generator/configuration"):
            adapter.sample(1)
        corpus.manifest = adapter._document
        world.mapping = world.deps.observations.PacerSMPLMapping()
        with self.assertRaisesRegex(ValueError, "named mapping changed"):
            adapter.sample(1)
        self.assertTrue(torch.equal(corpus.generator.get_state(), rng))
        self.assertFalse(adapter._faulted)


if __name__ == "__main__":
    unittest.main()
