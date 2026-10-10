"""Recovery guards and source inverse; fixtures do not establish SDK recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from CrowdSim.humanoid.runtime import social_training_recovery as recovery
from scripts import audit_shared_social_recovery as auditor
from scripts import train_shared_social as canonical


class RecoveryGuardsTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(recovery.ORIGINAL_PROFILE.read_text())
        self.profile = deepcopy(self.original)
        self.profile["runtime"].update(checkpoint_every_updates=25, archive_every_updates=125)

    def test_allowed_training_profile(self):
        recovery.validate_profile(self.profile, self.original, "training")

    def test_deny_changed_actor_rollout_seed_reward_and_epochs(self):
        for section, field, value in (("runtime", "actors", 8), ("runtime", "rollout_controls", 16),
                                     ("runtime", "seed", 19), ("training", "task_reward_weight", 1.),
                                     ("training", "update_epochs", 3)):
            changed = deepcopy(self.profile)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_profile(changed, self.original, "training")

    def test_smoke_is_one_real_update_from_boundary(self):
        self.profile["runtime"]["updates"] = 12001
        recovery.validate_profile(self.profile, self.original, "smoke")
        with self.assertRaises(ValueError):
            recovery.validate_profile(self.profile, self.original, "training")

    def test_constructor_mask_is_resolved_before_restore_comparison(self):
        from CrowdSim.humanoid.runtime import social_popart_training
        api = social_popart_training.load_trainer("_test_optimizer_recovery_mask")
        config = api.SocialTrainingConfig(actor_count=256, seed=7, grid_center="root", grid_heading="root",
            height_reference="static_center_ground_z", include_ego=False, **self.original["training"])
        self.assertIsNone(config.action_mask)
        resolved = recovery.resolve_requested_config(config, api)
        self.assertEqual(resolved.action_mask, api._settings.PHYSICAL_ACTION_MASK)
        self.assertEqual(sum(resolved.action_mask), 57)
        self.assertIsNone(config.action_mask)
        config.action_mask = (1,) * 69
        with self.assertRaises(ValueError):
            recovery.resolve_requested_config(config, api)

    def test_original_runner_preserved_with_explicit_segment_clock(self):
        derived, proof = recovery.derive_run(canonical.PARENT.read_text())
        self.assertTrue(proof["exact_inverse"])
        self.assertIn('(update - _recovery_base) * runtime["rollout_controls"]', derived)
        self.assertIn("trainer.restore_rng()", derived)
        self.assertIn('trainer.provenance["training_recovery"]["global_training_rng_restored_after_setup"] = True', derived)
        self.assertIn("probe._cleanup_owned_runtime", derived)

    def test_drifted_runner_rejected(self):
        changed = canonical.PARENT.read_text().replace('cumulative_expert_windows_drawn=expert.windows,', 'cumulative_expert_windows_drawn=0,')
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            recovery.derive_run(changed)

    def test_auditor_preserves_all_original_tensor_and_history_checks(self):
        derived, proof = auditor.derive(auditor.popart.PARENT.read_text())
        self.assertTrue(proof["exact_inverse"])
        self.assertIn('assert state["provenance"]["training_recovery"] == result["recovery"]', derived)
        self.assertIn('assert torch.equal(rollout["history_episode_ids"]', derived)
        self.assertIn('float(item["step"]) == updates * 12', derived)
        compile(derived, "recovery_audit_test", "exec")

    def test_unsaved_tail_is_excluded_without_changing_original_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [{"update_count": index, "physical_transitions": index * 256 * 32}
                    for index in range(1, 12003)]
            original = b"".join((json.dumps(row) + "\n").encode() for row in rows)
            path.write_bytes(original)
            metadata, prefix = recovery.read_retained_metrics(path)
            self.assertEqual(metadata["uncheckpointed_updates_not_inherited"], 2)
            self.assertEqual(len(prefix.splitlines()), 12000)
            self.assertEqual(path.read_bytes(), original)
            path.write_text(json.dumps(rows[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                recovery.read_retained_metrics(path)


if __name__ == "__main__":
    unittest.main()
