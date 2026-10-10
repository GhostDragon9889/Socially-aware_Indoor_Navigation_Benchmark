"""Fail-closed serial release and transparent saved provenance, without SDK."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import ModuleType
import sys
import unittest

from CrowdSim.humanoid.application import episode_serial_release as release
from scripts import train_shared_social_episode_after_recovery as training
from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation
from scripts import own_shared_social_episode_pipeline as owner


class EpisodeAfterRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir="/tmp")
        cls.path = Path(cls.directory.name) / "release.json"
        cls.document = release.release_document()
        cls.path.write_text(json.dumps(cls.document))
        cls.path.chmod(0o444)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_actual_release_keeps_original_interruption_and_new_family(self):
        actual, _ = release.admit_release(self.path)
        self.assertEqual(actual, self.document)
        self.assertFalse(actual["original_certificate_created"])
        self.assertFalse(actual["original_S3_continuous_completion_claimed"])
        self.assertTrue(actual["candidate_fresh_initialization"])
        self.assertFalse(actual["weights_or_moments_transferred"])

    def test_modified_continuous_claim_or_checkpoint_rejected(self):
        for name, value in (("original_S3_continuous_completion_claimed", True),
                            ("weights_or_moments_transferred", True), ("completed_evaluation", {})):
            with self.subTest(name=name):
                path = Path(self.directory.name) / (name + ".json")
                bad = deepcopy(self.document)
                bad[name] = value
                path.write_text(json.dumps(bad))
                path.chmod(0o444)
                with self.assertRaisesRegex(ValueError, "explicit completed-S3-R serial release"):
                    release.admit_release(path)

    def test_fresh_profiles_and_training_sources_are_preserved(self):
        for name in ("smoke8", "smoke256", "pilot2000"):
            with self.subTest(stage=name):
                module, binding, selected = training.prepare_training(self.path, name)
                original, _ = training.original.prepare_entry()
                self.assertEqual(module._sources(), original._sources())
                self.assertEqual(module._profile(Path(selected["profile"]["path"])),
                                 original._profile(Path(selected["profile"]["path"])))
                self.assertEqual(binding["stage"], name)
                self.assertFalse(binding["model_optimizer_control_expert_reward_math_changed"])

    def test_unregistered_stage_rejected(self):
        with self.assertRaisesRegex(ValueError, "one exact registered candidate stage"):
            training.prepare_training(self.path, "future81")

    def test_actual_consumers_accept_only_the_explicit_episode_world(self):
        original, _ = training.original.prepare_entry()
        selected, binding, _ = training.prepare_training(self.path, "smoke8")
        rows = original._sources()
        consumers = release.consumers
        old_world_module = original._load(consumers.OLD_WORLD, "_consumer_test_periodic_world",
                                          release.record(consumers.OLD_WORLD))
        episode_module = original._load(training.original.WORLD, "_consumer_test_episode_world",
                                        release.record(training.original.WORLD))
        old_world = object.__new__(old_world_module.SharedGroupWorld)
        episode_world = object.__new__(episode_module.SharedGroupWorld)
        before = {key: release.record(pin[0]) for key, pin in consumers.PINS.items()}
        self.assertEqual(binding["consumer_World_admission"], consumers.describe())
        for key in ("expert", "collector"):
            with self.subTest(consumer=key):
                path = original.PINS[key][0]
                unchanged = original._load(path, "_consumer_test_original_" + key, rows[key])
                guard = (lambda value: unchanged._class_module(value, "world", "SharedGroupWorld")) if key == "expert" else (
                    lambda value: unchanged._class_origin(value, "SharedGroupWorld", "world"))
                guard(old_world)
                with self.assertRaises(TypeError):
                    guard(episode_world)
                adapted = selected._load(path, "_consumer_test_adapted_" + key, rows[key])
                admitted = (lambda value: adapted._class_module(value, "world", "SharedGroupWorld")) if key == "expert" else (
                    lambda value: adapted._class_origin(value, "SharedGroupWorld", "world"))
                admitted(episode_world)
                for wrong in (old_world, object()):
                    with self.subTest(wrong_type=type(wrong).__name__), self.assertRaises(TypeError):
                        admitted(wrong)
                with self.assertRaisesRegex(ValueError, "admission already changed"):
                    consumers.adapt(adapted, key)
                self.assertEqual({name: item for name, item in adapted.PINS.items() if name != "world"},
                                 {name: item for name, item in unchanged.PINS.items() if name != "world"})
                if key == "collector":
                    self.assertIsNone(adapted._DEPS)
                    self.assertEqual(adapted.WORLD_METHOD_AST_PINS, unchanged.WORLD_METHOD_AST_PINS)
                    self.assertEqual(len(adapted._dependencies()), 3)
                    self.assertIsNone(unchanged._DEPS)
        self.assertEqual(before, {key: release.record(pin[0]) for key, pin in consumers.PINS.items()})

    def test_failed_smoke_output_is_preserved_and_only_retry_output_changes(self):
        original = json.loads(release.PLAN.read_text())["stages"]
        actual = release.stages()
        self.assertEqual(actual[1:], original[1:])
        self.assertEqual({key: item for key, item in actual[0].items() if key != "output_directory"},
                         {key: item for key, item in original[0].items() if key != "output_directory"})
        self.assertNotEqual(actual[0]["output_directory"], original[0]["output_directory"])
        self.assertTrue((Path(original[0]["output_directory"]) / "result.json").is_file())

    def test_actual_probe_loads_declared_clock_derivations_without_changing_source(self):
        module, binding, _ = training.prepare_training(self.path, "smoke8")
        rows = module._sources()
        probe_name = "_clock_test_actual_private_probe"
        package_name = "_clock_test_actual_private_training"
        names = [probe_name, package_name]
        probe = module._load(module.PINS["probe_setup"][0], probe_name, rows["probe_setup"])
        package = ModuleType(package_name)
        package.__path__ = [str(release.ROOT / "CrowdSim/humanoid/training")]
        sys.modules[package_name] = package
        try:
            for key, filename in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
                item = binding["consumer_World_admission"]["selected_World"]
                self.assertEqual(item, release.record(training.original.WORLD))
                source = training.original.world_sources()[key]
                name = package_name + "." + filename
                names.append(name)
                probe._load(Path(source["path"]), name, source)
            for key, filename in (("primitive", "episode_trajectories"), ("curriculum", "episode_mixed_trajectories")):
                path = release.clock_grid.PINS[key][0]
                before = release.record(path)
                name = package_name + "." + filename
                names.append(name)
                loaded = probe._load(path, name, before)
                self.assertEqual(loaded.EPISODE_CLOCK_DERIVATION, release.clock_grid.describe()["derived_modules"][key])
                self.assertEqual(before, release.record(path))
                with self.assertRaisesRegex(ValueError, "unused actual pinned private"):
                    release.clock_grid.adapt(loaded, key)
            primitive = sys.modules[package_name + ".episode_trajectories"]
            import torch
            reference = primitive.EpisodeProceduralReference(2)
            reference.initialized[:] = True
            reference.episode_ids[:] = 1
            reference.paths[:] = torch.arange(reference.paths.shape[1])[:, None]
            ages, episodes = torch.full((2,), 300), torch.ones(2, dtype=torch.long)
            self.assertEqual(reference.query(ages, episodes, [5.])[0, 0, 0].item(), 150.)
            with self.assertRaisesRegex(ValueError, "within five seconds"):
                reference.query(ages, episodes, [5.000001])
        finally:
            for name in reversed(names):
                sys.modules.pop(name, None)

    def test_facade_returns_the_original_factory_result_and_preserves_inputs(self):
        loaded = ModuleType("original_trainer_fixture")
        calls, returned = [], object()

        class OriginalFactory:
            @staticmethod
            def prepare(config, *, device, provenance):
                calls.append((config, device, provenance))
                return returned

        loaded.SocialAMPTrainer = OriginalFactory
        loaded.SocialTrainingConfig = object()
        binding = {"actual_release": "fixed", "nested": {"fresh": True}}
        base, config = {"initialization": {"weights_or_moments_transferred": False}}, object()
        facade = training.provenance_facade(loaded, binding)
        self.assertIs(facade.SocialAMPTrainer.prepare(config, device="cpu", provenance=base), returned)
        self.assertIs(calls[0][0], config)
        self.assertEqual(calls[0][1], "cpu")
        self.assertEqual(calls[0][2], {**base, "episode_after_recovery_execution": binding})
        self.assertNotIn("episode_after_recovery_execution", base)
        self.assertIs(loaded.SocialAMPTrainer, OriginalFactory)
        self.assertIs(facade.SocialTrainingConfig, loaded.SocialTrainingConfig)
        calls[0][2]["episode_after_recovery_execution"]["nested"]["fresh"] = False
        self.assertTrue(binding["nested"]["fresh"])

    def test_training_owner_rejects_missing_failed_bool_and_reordered_exits(self):
        valid = {"schema": "shared_social.episode_after_recovery_training_completion.v1",
                 "status": "completed_fresh_training_and_CPU_audits", "steps": [
                     {"name": name, "status": "passed", "actual_exit_code": 0,
                      "exit_observation": "subprocess.Popen.wait"} for name in evaluation.STEPS]}
        evaluation.validate_training_owner(valid)
        for mode in ("missing", "failed", "bool", "inferred", "reordered"):
            with self.subTest(mode=mode):
                bad = deepcopy(valid)
                if mode == "missing":
                    bad["steps"].pop()
                elif mode == "reordered":
                    bad["steps"].reverse()
                elif mode == "inferred":
                    bad["steps"][0]["exit_observation"] = "PID absent"
                else:
                    bad["steps"][0]["actual_exit_code"] = False if mode == "bool" else -11
                with self.assertRaisesRegex(ValueError, "three real candidate training"):
                    evaluation.validate_training_owner(bad)

    def test_stage_budget_and_cleanup_have_to_match(self):
        _, binding, selected = training.prepare_training(self.path, "smoke8")
        result = {"status": "passed_local_training", "cleanup_succeeded": True, "cleanup_errors": [],
                  "training_loop_completed": True, "episode_after_recovery_execution": binding,
                  "profile": {"runtime": selected["runtime"]}, "completed_updates": 1,
                  "actual_training_transitions": 32}
        audit = {"status": "passed", "CUDA_initialized": False, "updates": 1, "physical_transitions": 32}
        owner.validate_stage(result, audit, selected, binding)
        for key, value in (("cleanup_succeeded", False), ("completed_updates", True),
                            ("actual_training_transitions", 31), ("episode_after_recovery_execution", {})):
            with self.subTest(key=key):
                bad = {**result, key: value}
                with self.assertRaisesRegex(ValueError, "stage result, provenance, budget"):
                    owner.validate_stage(bad, audit, selected, binding)


if __name__ == "__main__":
    unittest.main()
