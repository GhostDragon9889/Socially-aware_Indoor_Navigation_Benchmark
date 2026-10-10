"""UNEXECUTED future CPU NN checks of owned actor/critic inference adapters.

Only one exact cached synthetic32-example fixture update is used. This module
imports stdlib only; running it must be separately authorized after formal
training exits and the required primary comparison finishes. No SDK/AMASS.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "trainer": ("docs/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py", 24078,
                "81a6319f2520b3be547f30454c88b088638d559d492bba52fe850d1c1d5ba96e"),
    "fixture": ("docs/style_improvement_20261004/social_trainer_staging_v1/test_social_trainer_future_CPU.py", 12606,
                "98de7ef9e8183b62b4b2019736086402faf27f8f25138c21510aae790207ec00"),
    "actor": ("docs/style_improvement_20261004/social_deployment_integration_v1/inference_adapter.py", 13766,
              "05261849db05c7c473c4a77ca9fd1e4c6930562a65bd46810849380484fbfc58"),
    "checkpoint": ("docs/style_improvement_20261004/social_checkpoint_inference_staging_v1/checkpoint_inference.py", 16176,
                   "cd227f5d2de81e21da560dba97a1d2f564e125beba9e73d739548c5c1800a352"),
    "value": ("docs/style_improvement_20261004/social_value_adapter_staging_v1/value_adapter.py", 12573,
              "9875c58b90e23ab6613ee7298bb4a199295435ad7f540f5b760e524a8f3b4f76"),
}
FORBIDDEN = ("isaaclab", "isaacsim", "omni", "pxr", "warp", "protomotions.simulator.isaaclab",
             "CrowdSim.humanoid.amp_world")
RMS_NAMES = ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")
ENTRY_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_binding():
    records = {}
    for role, (relative, size, sha) in PINS.items():
        path = ROOT / relative
        info, data = path.lstat(), path.read_bytes()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222
                or len(data) != size or hashlib.sha256(data).hexdigest() != sha):
            raise ValueError("Frozen future-check source differs: " + role)
        records[role] = {"path": str(path), "bytes": size, "sha256": sha}
    entry = Path(__file__)
    info, content = entry.lstat(), entry.read_bytes()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or hashlib.sha256(content).hexdigest() != ENTRY_SHA:
        raise ValueError("Future-check execution source changed after import")
    records["entry"] = {"path": str(entry), "bytes": len(content), "sha256": ENTRY_SHA}
    return records


def load_source(role, name):
    records = source_binding()
    path = Path(records[role]["path"])
    if name in sys.modules:
        raise ValueError("Use a fresh standalone CPU process: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def no_sdk(torch):
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA must remain uninitialized")
    for name in sys.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN):
            raise AssertionError("No native SDK import is permitted: " + name)


@contextmanager
def no_actor_sidepaths(torch, trainer, *, forbid_network_initializers):
    """Hooks apply only to the new actor path, not trainer reference policy()."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Actor-only path constructed/forwarded an excluded component")

    handles = [trainer.model.critic.register_forward_pre_hook(forbidden),
               trainer.model.discriminator.register_forward_pre_hook(forbidden)]
    try:
        with ExitStack() as stack:
            for owner in (torch.optim.Adam, trainer_module(trainer)._base.AMPPolicyReplay,
                          trainer_module(trainer).SocialAMPTrainer, torch.distributions.Normal):
                stack.enter_context(patch.object(owner, "__init__", forbidden))
            if forbid_network_initializers:
                for owner in (torch.nn.Linear, torch.nn.Conv2d, trainer_module(trainer)._base.RunningNormalizer):
                    stack.enter_context(patch.object(owner, "__init__", forbidden))
            yield
    finally:
        for handle in handles:
            handle.remove()


def trainer_module(trainer):
    module = sys.modules[type(trainer).__module__]
    if Path(module.__file__).resolve() != ROOT / PINS["trainer"][0]:
        raise AssertionError("Fixture used another trainer source")
    return module


class FutureSocialInferenceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Explicitly hide CUDA before any future NN import")
        cls.binding = source_binding()
        cls.stage = load_source("trainer", "social_trainer")
        cls.torch, cls.np = cls.stage.torch, cls.stage.np
        cls.torch.set_num_threads(4)
        no_sdk(cls.torch)
        cls.fixture = load_source("fixture", "_social_inference_exact_future_fixture")
        if cls.fixture.FutureSocialTrainerCPU._fixture is not None:
            raise AssertionError("The exact fixture must not have been run beforehand")
        # Call ONLY this cached source method, never its four-test runner or a
        # second update/fresh(). Its single synthetic update is intentional.
        cls.trainer, cls.rollout, cls.metrics, _ = cls.fixture.FutureSocialTrainerCPU.trained_fixture()
        if (cls.trainer.update_count, cls.trainer.physical_transition_count) != (1, 32):
            raise AssertionError("Expected exactly the pinned one-update fixture")
        if cls.trainer.provenance.get("physical_rollout") is not False:
            raise AssertionError("Synthetic fixture must not claim physical data")
        cls.actor_api = load_source("actor", "_social_inference_checked_actor")
        cls.checkpoint_api = load_source("checkpoint", "_social_inference_checked_checkpoint")
        cls.value_api = load_source("value", "_social_inference_checked_value")
        cls.directory = tempfile.TemporaryDirectory(prefix="social_inference_future_CPU_")
        cls.path = Path(cls.directory.name) / "synthetic_same_trained_fixture.ckpt"
        before = cls.torch.get_rng_state().clone()
        cls.trainer.save(cls.path)  # The same fixture, no trainer.load/second training.
        if not cls.torch.equal(before, cls.torch.get_rng_state()):
            raise AssertionError("Saving the fixture unexpectedly drew CPU RNG")
        cls.path.chmod(0o444)
        info = cls.path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
            raise AssertionError("Ordinary exclusive readonly fixture snapshot required")
        with cls.path.open("rb") as stream:
            cls.sha = hashlib.file_digest(stream, "sha256").hexdigest()
        cls.snapshot = None
        no_sdk(cls.torch)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.FutureSocialTrainerCPU._fixture = None
        cls.snapshot = None
        cls.directory.cleanup()
        no_sdk(cls.torch)
        if source_binding() != cls.binding:
            raise AssertionError("Reviewed sources changed during future CPU checks")

    def setUp(self):
        no_sdk(self.torch)
        self.cpu_rng = self.torch.get_rng_state().clone()
        self.python_rng = random.getstate()
        self.numpy_rng = self.np.random.get_state()
        self.moments = {name: {key: value.detach().clone() for key, value in getattr(self.trainer, name).state_dict().items()}
                        for name in RMS_NAMES}

    def tearDown(self):
        no_sdk(self.torch)
        self.assertTrue(self.torch.equal(self.cpu_rng, self.torch.get_rng_state()), "CPU global RNG changed")
        self.assertEqual(self.python_rng, random.getstate())
        current = self.np.random.get_state()
        self.assertEqual(self.numpy_rng[0], current[0])
        self.np.testing.assert_array_equal(self.numpy_rng[1], current[1])
        self.assertEqual(self.numpy_rng[2:], current[2:])
        for name, values in self.moments.items():
            for key, value in values.items():
                self.assertTrue(self.torch.equal(value, getattr(self.trainer, name).state_dict()[key]), name + "." + key)

    def inputs(self, batch):
        return self.rollout["self_obs"][0, :batch].clone(), self.rollout["task_obs"][0, :batch].clone()

    def equal_actor(self, actual, expected, batch):
        for name in ("actions", "mean_actions", "applied_actions"):
            self.assertEqual(tuple(actual[name].shape), (batch, 69))
            self.assertTrue(self.torch.equal(actual[name], expected[name]), name)
        self.assertEqual(int(self.trainer.action_mask.sum()), 57)
        self.assertTrue(self.torch.equal(actual["applied_actions"][..., self.trainer.action_mask == 0],
                                        self.torch.zeros(batch, 12)))

    def test_01_owned_actor_B1_B4_matches_same_trainer(self):
        expected = {batch: self.trainer.act(*self.inputs(batch), deterministic=True) for batch in (1, 4)}
        with no_actor_sidepaths(self.torch, self.trainer, forbid_network_initializers=True):
            snapshot = self.actor_api.FrozenSocialAMPInference.from_trainer(self.trainer)
            for batch in (1, 4):
                inputs = self.inputs(batch)
                before = tuple(value.clone() for value in inputs)
                self.equal_actor(snapshot.act(*inputs), expected[batch], batch)
                self.assertTrue(all(self.torch.equal(x, y) for x, y in zip(inputs, before, strict=True)))
        self.assertEqual(snapshot.describe()["grid_geometry_contract"]["height_reference"], "root_z")
        self.assertNotEqual(snapshot.describe()["grid_geometry_contract"],
                            {"grid_center": "root", "grid_heading": "root", "height_reference": "static_center_ground_z"})
        # No SDK/world is constructed: the actual fixture geometry deliberately
        # differs from the raw SDK evaluation producer's admitted ground frame.
        self.assertEqual(snapshot.describe()["training_actor_count"], 8)
        type(self).snapshot = snapshot

    def test_02_wrong_SHA_rejects_before_NN_or_payload_load(self):
        wrong = ("0" if self.sha[0] != "0" else "1") + self.sha[1:]
        with patch.object(self.checkpoint_api, "_owned_module", side_effect=AssertionError("Reached NN source admission")) as module_call:
            with patch.object(self.torch, "load", side_effect=AssertionError("Reached payload load")) as payload_call:
                with self.assertRaisesRegex(ValueError, "required selected SHA256"):
                    self.checkpoint_api.FrozenSocialCheckpointInference.from_checkpoint(self.path, expected_sha256=wrong)
                module_call.assert_not_called()
                payload_call.assert_not_called()

    def test_03_one_CPU_mmap_load_matches_owned_snapshot_without_optimizer_replay(self):
        self.assertIsNotNone(type(self).snapshot, "Run this four-case standalone program in its fixed order")
        module = self.checkpoint_api._owned_module("_social_checkpoint_source_trainer", "trainer")
        original_init = module.SocialAMPNetwork.__init__
        network_calls, hooks = [], []

        def forbidden(*args, **kwargs):
            raise AssertionError("Checkpoint actor path forwarded critic/D or constructed trainer/Adam/replay")

        def counted_network(network, *args, **kwargs):
            network_calls.append(1)
            original_init(network, *args, **kwargs)
            hooks.extend((network.critic.register_forward_pre_hook(forbidden),
                          network.discriminator.register_forward_pre_hook(forbidden)))

        try:
            with no_actor_sidepaths(self.torch, self.trainer, forbid_network_initializers=False):
                with patch.object(module.SocialAMPTrainer, "__init__", forbidden), patch.object(module.SocialAMPNetwork, "__init__", counted_network):
                    with patch.object(self.torch, "load", wraps=self.torch.load) as payload_call:
                        loaded = self.checkpoint_api.FrozenSocialCheckpointInference.from_checkpoint(
                            self.path, expected_sha256=self.sha, device="cpu")
                        self.assertEqual(payload_call.call_count, 1)
                        args, kwargs = payload_call.call_args
                        self.assertTrue(args[0].startswith("/proc/self/fd/"))
                        self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True, "mmap": True})
                    for batch in (1, 4):
                        raw = self.inputs(batch)
                        self.equal_actor(loaded.act(*raw), type(self).snapshot.act(*raw), batch)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(network_calls), 1, "One declared temporary full NN for strict tensor admission")
        metadata = loaded.describe()
        self.assertEqual(metadata["checkpoint"]["sha256"], self.sha)
        self.assertEqual(metadata["checkpoint"], metadata["checkpoint_identity_after_load"])
        self.assertEqual(metadata["grid_geometry_contract"], type(self).snapshot.describe()["grid_geometry_contract"])
        self.assertTrue(metadata["initializer_CPU_RNG_draws_performed"])
        self.assertTrue(metadata["caller_CPU_RNG_preserved"])
        self.assertFalse(metadata["restore_training_rng"])
        self.assertFalse(metadata["device_move_performed"])
        self.assertIn("checkpoint declaration", metadata["counter_source"])
        self.assertFalse(metadata["optimizer_replay_saved_RNG_runner_state_validated_or_restored"])
        self.assertEqual(metadata["retained_components"], ["task_encoder", "actor", "self_RMS357", "task_RMS12308", "action_mask69"])
        for name in ("_critic", "_discriminator", "model", "optimizer", "policy_replay", "artifact"):
            self.assertNotIn(name, vars(loaded))
        self.assertNotEqual(loaded._actor[0].weight.data_ptr(), self.trainer.model.actor[0].weight.data_ptr())
        self.assertNotEqual(loaded._self_rms.mean.data_ptr(), self.trainer.self_normalizer.mean.data_ptr())

    def test_04_candidate20_value_and_gradient_match_same_trainer_formula(self):
        torch = self.torch
        current_self, current_task = (value.requires_grad_() for value in self.inputs(4))
        fixed_self, fixed_raster = current_self.detach().clone(), current_task[:, 20:].detach().clone()
        candidate = torch.stack((current_task[:, :20].detach().reshape(4, 10, 2),
                                 current_task[:, :20].detach().reshape(4, 10, 2) + .02), 1).clone()
        std = torch.sqrt(self.trainer.task_normalizer.variance[0].float() + 1e-5)
        candidate[0, 1, 0, 0] = self.trainer.task_normalizer.mean[0].float() + 100 * std
        candidate.requires_grad_()
        independent = candidate.detach().clone().requires_grad_()
        # Do not apply actor-only critic hooks to this intentional value path.
        with ExitStack() as stack:
            for owner in (torch.nn.Linear, torch.nn.Conv2d, self.stage._base.RunningNormalizer,
                          torch.optim.Adam, self.stage._base.AMPPolicyReplay):
                stack.enter_context(patch.object(owner, "__init__", side_effect=AssertionError("Fresh value-path construction")))
            snapshot = self.value_api.SocialValueSnapshot(self.trainer)
        bound = snapshot.bind_current(current_self, current_task)
        with torch.no_grad():
            current_self.add_(10.)
            current_task[:, 20:].mul_(7.)
        self.assertTrue(torch.equal(bound._self, fixed_self))
        self.assertTrue(torch.equal(bound._raster, fixed_raster))
        value = bound.candidate_value(candidate)
        gradient, self_gradient, task_gradient = torch.autograd.grad(value.sum(), (candidate, current_self, current_task), allow_unused=True)
        self.assertIsNone(self_gradient)
        self.assertIsNone(task_gradient)
        raw_self = fixed_self[:, None].expand(4, 2, 357).reshape(8, 357)
        raw_task = torch.cat((independent.reshape(8, 20), fixed_raster[:, None].expand(4, 2, 12288).reshape(8, 12288)), -1)
        features = torch.cat((self.trainer.self_normalizer.normalize(raw_self),
                              self.trainer.model.task_encoder(self.trainer.task_normalizer.normalize(raw_task))), -1)
        normalized = self.trainer.model.critic(features).squeeze(-1)
        reference = self.trainer.value_normalizer.denormalize(normalized[:, None]).reshape(4, 2)
        reference_gradient, = torch.autograd.grad(reference.sum(), independent)
        self.assertTrue(torch.equal(value, reference))
        torch.testing.assert_close(gradient, reference_gradient, rtol=1e-6, atol=1e-7)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertEqual(float(gradient[0, 1, 0, 0]), 0., "Original clip5 blocks the deliberately saturated input gradient")
        self.assertGreater(int(torch.count_nonzero(gradient)), 0)
        self.assertTrue(all(not p.requires_grad for module in (snapshot._encoder, snapshot._critic) for p in module.parameters()))
        self.assertEqual(snapshot.metadata()["configuration_document"]["configuration"]["height_reference"], "root_z")


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES='' is required before future execution")
    unittest.main()
