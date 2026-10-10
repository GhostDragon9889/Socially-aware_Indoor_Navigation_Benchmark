"""Candidate terminal metadata contracts; fixtures are not training evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from CrowdSim.humanoid.application import episode_evaluation_admission as admission
from scripts import run_shared_social_episode_evaluation as entry
from scripts import run_shared_social_post_training as parent


def metadata_fixture(registration):
    expected = registration["episode_reference_experiment"]
    reference = deepcopy(expected["expected_reference_configuration"])
    return {"episode_reference_experiment": {
        "binding": deepcopy(expected["training_binding"]), "actual_reference_configuration": deepcopy(reference),
        "caller_probe_namespace_World_selection_changed": True, "original_source_files_modified": False,
        "author_recipe_equivalence_claimed": False},
        "profile": json.loads(Path(registration["configuration"]["path"]).read_text()),
        "completed_updates": 2000, "actual_training_transitions": 16384000,
        "arguments": {"output": registration["run_directory"], "config": registration["configuration"]["path"]},
        "entry_sources": deepcopy(expected["entry_sources"]), "entry_sources_after": deepcopy(expected["entry_sources"]),
        "setup_sources": deepcopy(expected["setup_sources"]), "setup_sources_after": deepcopy(expected["setup_sources"]),
        "actual_import_origins": {"episode_reference_curriculum": {
            "module": "_shared_social_original_training.episode_reference_curriculum", **expected["training_binding"]["curriculum"]}},
        "final_world_description": {"num_actors": 256, "control_step": 64000,
            "curriculum": {"source": admission.CURRICULUM_SOURCE, "config": reference,
                "author_recipe_equivalence_claimed": False, "statistics": {
                    "control_steps": 64000, "amass_actor_transitions": 8192000,
                    "procedural_actor_transitions": 8192000, "procedural_plans": 128, "procedural_plan_views": 256}}}}


class EpisodeTerminalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration, _ = admission.admit_registration()

    def test_registered_candidate_metadata_fixture_matches(self):
        admission.validate_episode_result(metadata_fixture(self.registration), self.registration)

    def test_repeated_cold_admission_releases_its_owned_private_helper(self):
        import sys

        first, _ = admission.admit_registration()
        second, _ = admission.admit_registration()
        self.assertEqual(first, second)
        self.assertNotIn("_episode_evaluation_setup_sources", sys.modules)

    def test_old_periodic_reference_cannot_be_relabelled(self):
        result = metadata_fixture(self.registration)
        del result["episode_reference_experiment"]
        with self.assertRaisesRegex(ValueError, "actual registered episode"):
            admission.validate_episode_result(result, self.registration)

    def test_future81_cannot_enter_the_episode31_experiment(self):
        result = metadata_fixture(self.registration)
        result["episode_reference_experiment"]["actual_reference_configuration"]["real_reset_future_frames"] = 81
        with self.assertRaisesRegex(ValueError, "configuration/ownership"):
            admission.validate_episode_result(result, self.registration)

    def test_live_root_reference_cannot_enter_as_episode_reference(self):
        result = metadata_fixture(self.registration)
        result["final_world_description"]["curriculum"]["source"] = "engineering_fixed_actor_amass_procedural_mixture"
        with self.assertRaisesRegex(ValueError, "terminal curriculum"):
            admission.validate_episode_result(result, self.registration)

    def test_incomplete_exposure_or_foreign_output_rejected(self):
        for key, value in (("completed_updates", 1999), ("completed_updates", True),
                           ("actual_training_transitions", 16383999), ("actual_training_transitions", True),
                           ("output", "/tmp/foreign"), ("config", "/tmp/foreign.json")):
            with self.subTest(key=key, value=value):
                result = metadata_fixture(self.registration)
                if key in ("output", "config"):
                    result["arguments"][key] = value
                else:
                    result[key] = value
                with self.assertRaisesRegex(ValueError, "budget/output/profile"):
                    admission.validate_episode_result(result, self.registration)

    def test_changed_runtime_source_or_import_origin_rejected(self):
        for key in ("entry_sources", "entry_sources_after", "setup_sources", "setup_sources_after", "origin"):
            with self.subTest(key=key):
                result = metadata_fixture(self.registration)
                if key == "origin":
                    result["actual_import_origins"]["episode_reference_curriculum"]["module"] = "foreign.module"
                else:
                    result[key].clear()
                with self.assertRaisesRegex(ValueError, "source inventory|curriculum import"):
                    admission.validate_episode_result(result, self.registration)

    def test_actual_exit_and_audit_admission_cannot_be_skipped(self):
        base = Mock(side_effect=ValueError("actual exit missing"))
        with self.assertRaisesRegex(ValueError, "actual exit missing"):
            admission.admit_episode_training(Path("/tmp/fixture_receipt.json"), self.registration, base)
        base.assert_called_once_with(Path("/tmp/fixture_receipt.json"))

    def test_compatible_popart_family_is_insufficient_for_candidate_selection(self):
        base = Mock(return_value=(Path("/tmp/foreign/step_002000.pt"), "fixture", {"value_normalization": "popart"}))
        with self.assertRaisesRegex(ValueError, "preregistered episode terminal"):
            admission.admit_episode_training(Path("/tmp/fixture_receipt.json"), self.registration, base)

    def test_private_parent_keeps_original_child_paths_and_globals(self):
        original_sources = parent.SOURCE_PATHS.copy()
        original_admit = parent.admit_training
        module = entry.prepare_pipeline(self.registration)
        self.assertEqual(module.SOURCE_PATHS[:len(original_sources)], original_sources)
        self.assertEqual(len(module.SOURCE_PATHS), len(original_sources) + 9)
        self.assertEqual(parent.SOURCE_PATHS, original_sources)
        self.assertIs(parent.admit_training, original_admit)
        self.assertIsNot(module.admit_training, original_admit)
        self.assertEqual(module.main.__code__.co_code, parent.main.__code__.co_code)

    def test_second_namespace_is_fresh_and_source_list_is_not_shared(self):
        first, second = entry.prepare_pipeline(self.registration), entry.prepare_pipeline(self.registration)
        self.assertIsNot(first, second)
        self.assertIsNot(first.SOURCE_PATHS, second.SOURCE_PATHS)
        first.SOURCE_PATHS.append("fixture_only")
        self.assertNotIn("fixture_only", second.SOURCE_PATHS)

    def test_readonly_registration_drift_is_rejected(self):
        changed = deepcopy(self.registration)
        changed["preselected_checkpoint"] = "step_001500.pt"
        with patch.object(admission, "registration_document", return_value=changed):
            with self.assertRaisesRegex(ValueError, "registration or registered sources changed"):
                admission.admit_registration()


if __name__ == "__main__":
    unittest.main()
