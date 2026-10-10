"""Pure stdlib configuration contract cases; no NN or physical assertions."""
from dataclasses import asdict
import json
import sys
import unittest

from social_training_settings import PHYSICAL_ACTION_MASK, SocialTrainingConfig, TrainingConfig


def example(**changes):
    return SocialTrainingConfig(grid_center="root", grid_heading="root", height_reference="root_z",
                                include_ego=True, actor_count=256, **changes)


class SettingsTests(unittest.TestCase):
    def test_new_layout_and_shared_validation_do_not_mutate(self):
        value = example()
        before = asdict(value)
        value.validate()
        self.assertEqual(asdict(value), before)
        document = value.to_document()
        self.assertEqual(document["configuration"]["task_dim"], 12308)
        self.assertEqual(document["configuration"]["motion_symmetry_coefficient"], 50.)
        self.assertEqual(document["logical_layout"]["SDK_envs"], 64)
        with self.assertRaises(ValueError):
            TrainingConfig(task_dim=12308, motion_symmetry_coefficient=50.).validate()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("isaaclab", sys.modules)

    def test_json_roundtrip_and_reject_changed_derived_contract(self):
        original = example()
        document = json.loads(json.dumps(original.to_document()))
        copy = SocialTrainingConfig.from_document(document)
        self.assertEqual(asdict(copy), asdict(original))
        self.assertEqual(document, json.loads(json.dumps(copy.to_document())))
        document["normalization"]["whole_task_dimensions"] = 1044
        with self.assertRaises(ValueError):
            SocialTrainingConfig.from_document(document)

    def test_feature_and_initialization_incompatibility(self):
        for changes in ({"task_dim": 1044}, {"history_frames": 11}, {"log_std": -2.},
                        {"architecture_id": "old"}, {"task_terrain_statistics_mode": "preserve_declared_suffix"},
                        {"warm_start_discriminator": True}, {"warm_start_action_discriminator_from_state_prior": True},
                        {"normalization_clip": 4.}, {"normalize_values": False},
                        {"freeze_toes": False}, {"control_dt": 1 / 60}, {"mode": "strict_author_experiment"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                example(**changes).validate()
        value = example()
        for key, bad in (("actor_count", 255), ("actor_count", True), ("grid_center", "automatic"),
                         ("include_ego", 1), ("motion_symmetry_coefficient", float("nan")),
                         ("learning_rate", -1.), ("gamma", 2.), ("clip_ratio", 0.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                value.detached_copy(**{key: bad})
        self.assertEqual(value.actor_count, 256)
        with self.assertRaises(ValueError):
            value.detached_copy(action_mask=(0,) * 69)
        explicit = value.detached_copy(action_mask=PHYSICAL_ACTION_MASK)
        self.assertEqual(sum(explicit.action_mask), 57)

    def test_complete_fields_and_explicit_choices(self):
        document = example().to_document()
        document["configuration"].pop("grid_heading")
        with self.assertRaises(ValueError):
            SocialTrainingConfig.from_document(document)
        document = example().to_document()
        document["configuration"]["unknown_field"] = 1
        with self.assertRaises(ValueError):
            SocialTrainingConfig.from_document(document)
        # A different explicit sampling choice remains a different declaration.
        alternative = example().detached_copy(grid_center="head", grid_heading="head", height_reference="static_center_ground_z")
        self.assertEqual(alternative.to_document()["configuration"]["grid_center"], "head")


if __name__ == "__main__":
    unittest.main()
