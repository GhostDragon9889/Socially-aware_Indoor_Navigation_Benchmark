"""UNEXECUTED future CPU NN checks; synthetic features, never physics/AMASS.

Root must explicitly authorize one serial invocation after formal training and
required primary evaluation. Compilation of this file does not import Torch.
Only one cached32-example PPO/D update is used by the four cases collectively.
"""
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Future tests require explicitly hidden CUDA")

import numpy as np
import torch

import social_trainer as stage


def configuration():
    return stage.SocialTrainingConfig(
        grid_center="root", grid_heading="root", height_reference="root_z",
        include_ego=False, actor_count=8, update_epochs=1, minibatch_size=32,
        discriminator_batch_size=32, discriminator_replay_capacity=64,
        expert_action_semantics="user_supplied_normalized_actions",
    )


def fresh():
    config = configuration()
    trainer = stage.SocialAMPTrainer(config, device="cpu", provenance={
        "scope": "synthetic_CPU_algorithm_check", "physical_rollout": False,
        "expert_source": "synthetic finite features; not motion capture",
    })
    return config, trainer


class FutureSocialTrainerCPU(unittest.TestCase):
    _fixture = None

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU checks must not initialize CUDA")

    @classmethod
    def tearDownClass(cls):
        cls._fixture = None
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    @classmethod
    def trained_fixture(cls):
        if cls._fixture is None:
            _, trainer = fresh()
            generator = torch.Generator(device="cpu").manual_seed(20261004)
            self_obs = torch.randn((4, 8, 357), generator=generator) * .2
            task_obs = torch.randn((4, 8, 12308), generator=generator) * .2
            # Nonzero, deliberately unequal synthetic channel means. These are
            # not physical heights/velocities or a valid SMPL motion.
            task_obs[..., 20:] += .3
            history = torch.randn((4, 8, 10, 195), generator=generator) * .2
            behavior = trainer.act(self_obs.flatten(0, 1), task_obs.flatten(0, 1))
            next_values = trainer.value((self_obs + .01).flatten(0, 1), (task_obs + .02).flatten(0, 1))
            episode = torch.zeros((4, 8), dtype=torch.int64)
            rollout = {
                "self_obs": self_obs, "task_obs": task_obs, "amp_history": history,
                "actions": behavior["actions"].reshape(4, 8, 69),
                "applied_actions": behavior["applied_actions"].reshape(4, 8, 69),
                "log_prob": behavior["log_prob"].reshape(4, 8),
                "values": behavior["values"].reshape(4, 8), "next_values": next_values.reshape(4, 8),
                "rewards": torch.linspace(-.3, .7, 32).reshape(4, 8),
                "terminated": torch.zeros((4, 8), dtype=torch.bool),
                "truncated": torch.zeros((4, 8), dtype=torch.bool),
                "episode_ids": episode, "history_episode_ids": episode[..., None].expand(4, 8, 10).clone(),
            }
            expert_history = torch.randn((32, 10, 195), generator=generator) * .2
            expert_actions = (torch.randn((32, 69), generator=generator) * .1).clamp(-1, 1) * trainer.action_mask
            mirror, actions = stage._SOCIAL_MIRROR_FUNCTIONS
            exposed = []

            def observe_mirror(raw_self, raw_task):
                exposed.append((raw_self.detach().clone(), raw_task.detach().clone()))
                return mirror(raw_self, raw_task)

            # Observe the real raw coupled mirror; return its unchanged result.
            # No source file, mirror arithmetic or PPO formula is modified.
            with patch.dict(stage._original_update_with_social_mirror.__globals__,
                            {"_SOCIAL_MIRROR_FUNCTIONS": (observe_mirror, actions)}):
                metrics = trainer.update(rollout, expert_history, expert_actions)
            cls._fixture = trainer, rollout, metrics, exposed
        return cls._fixture

    def test_01_fresh_architecture_RMS_mask_and_detached_config(self):
        config, trainer = fresh()
        self.assertIsNot(config, trainer.config)
        self.assertIsNone(config.action_mask)
        self.assertEqual(sum(trainer.config.action_mask), 57)
        self.assertEqual(int(trainer.action_mask.sum()), 57)
        self.assertEqual(trainer.model.actor[0].in_features, 633)
        self.assertEqual(trainer.model.critic[0].in_features, 633)
        conv = [module for module in trainer.model.task_encoder.cnn if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual([(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
                          for module in conv], [(3, 16, (4, 4), (2, 2), (1, 1)),
                                                (16, 16, (4, 4), (2, 2), (1, 1)),
                                                (16, 16, (4, 4), (2, 2), (1, 1))])
        self.assertEqual(trainer.model.task_encoder.projection[0].in_features, 1024)
        self.assertEqual(trainer.model.task_encoder.projection[2].out_features, 256)
        self.assertEqual(trainer.model.discriminator[0].in_features, 2019)
        self.assertEqual(trainer.model.discriminator[-1].out_features, 1)
        self.assertTrue(torch.equal(trainer.model.log_std, torch.full((69,), -2.9)))
        for name, dimension in (("self_normalizer", 357), ("task_normalizer", 12308),
                                ("amp_normalizer", 195), ("value_normalizer", 1)):
            normalizer = getattr(trainer, name)
            self.assertEqual(normalizer.mean.shape, (dimension,))
            self.assertEqual(normalizer.mean.dtype, torch.float64)
            self.assertEqual(float(normalizer.count), 0.)
        policy_ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
        expected_ids = [id(p) for module in (trainer.model.task_encoder, trainer.model.actor, trainer.model.critic)
                        for p in module.parameters()]
        discriminator_ids = {id(p) for group in trainer.discriminator_optimizer.param_groups for p in group["params"]}
        self.assertEqual(policy_ids, expected_ids)
        self.assertEqual(len(policy_ids), len(set(policy_ids)))
        self.assertFalse(set(policy_ids) & discriminator_ids)
        self.assertEqual(discriminator_ids, {id(p) for p in trainer.model.discriminator.parameters()})

    def test_02_single_synthetic_update_full_raw_moments_and_mirror(self):
        trainer, rollout, metrics, exposed = self.trained_fixture()
        self.assertEqual(trainer.update_count, 1)
        self.assertEqual(trainer.physical_transition_count, 32)
        self.assertEqual(metrics["optimizer_minibatches"], 1)
        self.assertEqual(metrics["discriminator_minibatches"], 1)
        for name in ("self_normalizer", "task_normalizer", "value_normalizer"):
            self.assertEqual(float(getattr(trainer, name).count), 32.)
        self.assertEqual(float(trainer.amp_normalizer.count), 640.)  # (32policy+32expert)*10 frames.
        self.assertTrue(torch.equal(trainer.task_normalizer.mean, rollout["task_obs"].flatten(0, 1).double().mean(0)))
        self.assertEqual(len(exposed), 1)
        self.assertEqual(exposed[0][0].shape, (32, 357))
        self.assertEqual(exposed[0][1].shape, (32, 12308))
        self.assertTrue(torch.equal(exposed[0][1][:, 0].sort().values, rollout["task_obs"].flatten(0, 1)[:, 0].sort().values))
        self.assertGreater(metrics["motion_symmetry_loss"], 0.)
        self.assertEqual((trainer.policy_replay.size, trainer.policy_replay.cursor, trainer.policy_replay.accepted), (32, 32, 32))

    def test_03_complete_CPU_bundle_restore_then_same_inputs_and_RNG(self):
        trainer, rollout, _, _ = self.trained_fixture()
        with tempfile.TemporaryDirectory(prefix="social_trainer_future_CPU_") as directory:
            path = Path(directory) / "synthetic.pt"
            opaque = {"scope": "synthetic_nonphysical_caller_state", "marker": 7}
            trainer.save(path, runner_state=opaque)
            self_obs, task_obs = rollout["self_obs"][0], rollout["task_obs"][0]
            original = trainer.act(self_obs, task_obs)
            original_rng = random.random(), np.random.random(3), torch.rand(3)
            restored, supplied = stage.SocialAMPTrainer.load(path, device="cpu", restore_rng=False)
            self.assertEqual(supplied, opaque)  # Opaque roundtrip, not SDK restoration.
            self.assertIsNotNone(restored._pending_rng_state)
            # load(False) seeded a fresh constructor; do not claim global RNG
            # preservation. Restore the saved original RNG explicitly now.
            restored.restore_rng()
            self.assertIsNone(restored._pending_rng_state)
            repeated = restored.act(self_obs, task_obs)
            for name in original:
                self.assertTrue(torch.equal(original[name], repeated[name]), name)
            self.assertEqual(original_rng[0], random.random())
            np.testing.assert_array_equal(original_rng[1], np.random.random(3))
            self.assertTrue(torch.equal(original_rng[2], torch.rand(3)))
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(restored.physical_transition_count, 32)
            self.assertTrue(restored.optimizer.state)
            self.assertTrue(restored.discriminator_optimizer.state)
            for name, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)
            for name in ("optimizer", "discriminator_optimizer"):
                before, after = getattr(trainer, name).state_dict(), getattr(restored, name).state_dict()
                self.assertEqual(before["param_groups"], after["param_groups"])
                self.assertEqual(set(before["state"]), set(after["state"]))
                for parameter, moments in before["state"].items():
                    for field, value in moments.items():
                        self.assertTrue(torch.equal(value, after["state"][parameter][field]))
            for name in ("history", "actions"):
                self.assertTrue(torch.equal(trainer.policy_replay.state_dict()[name], restored.policy_replay.state_dict()[name]))
            for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
                for field, value in getattr(trainer, name).state_dict().items():
                    self.assertTrue(torch.equal(value, getattr(restored, name).state_dict()[field]))

    def test_04_specified_malformed_complete_bundles_are_rejected(self):
        trainer, _, _, _ = self.trained_fixture()
        with tempfile.TemporaryDirectory(prefix="social_trainer_future_CPU_negative_") as directory:
            path = Path(directory) / "base.pt"
            trainer.save(path)
            base = torch.load(path, map_location="cpu", weights_only=True)
            for risk in ("schema", "RMS_count", "model_dtype", "Adam_missing"):
                with self.subTest(risk=risk):
                    invalid = dict(base)
                    if risk == "schema":
                        invalid["format"] = "navisaaclab_method_reconstruction_training"
                    elif risk == "RMS_count":
                        invalid["normalizers"] = dict(base["normalizers"])
                        invalid["normalizers"]["task_normalizer"] = base["normalizers"]["task_normalizer"].copy()
                        invalid["normalizers"]["task_normalizer"]["count"] = base["normalizers"]["task_normalizer"]["count"] + 1
                    elif risk == "model_dtype":
                        invalid["model"] = base["model"].copy()
                        key = "task_encoder.cnn.0.weight"
                        invalid["model"][key] = base["model"][key].double()
                    else:
                        invalid["optimizer"] = dict(base["optimizer"])
                        invalid["optimizer"]["state"] = {}
                    rejected = Path(directory) / (risk + ".pt")
                    torch.save(invalid, rejected)
                    with self.assertRaises(ValueError):
                        stage.SocialAMPTrainer.load(rejected, device="cpu", restore_rng=False)


if __name__ == "__main__":
    unittest.main()
