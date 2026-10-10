"""Fail-closed serial release and transparent saved provenance, without SDK."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import ModuleType
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
