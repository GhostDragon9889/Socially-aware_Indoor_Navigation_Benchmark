"""Reject incorrect recovery provenance while preserving the scoring entries."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from CrowdSim.humanoid.application import recovery_evaluation_admission as admission
from scripts import evaluate_shared_social_recovery as evaluation


class RecoveryEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.identities, cls.numeric = admission.completed_documents()

    def validate(self, documents):
        admission.validate_documents(documents["completion"], documents["result"], documents["audit"],
                                     documents["external_verification"], self.identities, self.numeric)

    def test_actual_terminal_metadata_admitted_cold(self):
        self.validate(self.documents)
        self.assertNotIn("torch", sys.modules)

    def test_exit_is_actual_exact_integer_zero(self):
        for value in [False, True, None, "0", 1, -9]:
            with self.subTest(value=value):
                documents = deepcopy(self.documents)
                documents["completion"]["steps"][0]["actual_exit_code"] = value
                with self.assertRaisesRegex(ValueError, "actual recovery child exits"):
                    self.validate(documents)

    def test_progress_cannot_replace_OS_exit(self):
        documents = deepcopy(self.documents)
        documents["completion"]["steps"][0]["exit_observation"] = "metrics reached 12500"
        with self.assertRaisesRegex(ValueError, "OS wait"):
            self.validate(documents)

    def test_every_budget_counter_must_match(self):
        for key in ["actual_segment_updates", "effective_total_updates", "effective_training_transitions"]:
            with self.subTest(key=key):
                documents = deepcopy(self.documents)
                documents["completion"][key] -= 1
                with self.assertRaisesRegex(ValueError, "Incomplete recovery exposure"):
                    self.validate(documents)

    def test_failure_or_missing_cleanup_rejected(self):
        for key, value in [("status", "failed"), ("cleanup_succeeded", False), ("cleanup_errors", ["release"]),
                           ("source_profile_before_after_equal", False)]:
            with self.subTest(key=key):
                documents = deepcopy(self.documents)
                documents["result"][key] = value
                with self.assertRaisesRegex(ValueError, "cleanup disagree"):
                    self.validate(documents)

    def test_fresh_audit_is_not_recovery_audit(self):
        documents = deepcopy(self.documents)
        documents["audit"]["schema"] = "shared_social.completed_training_CPU_audit.v1"
        with self.assertRaisesRegex(ValueError, "recovery-specific CPU audit"):
            self.validate(documents)

    def test_physical_resume_claim_rejected(self):
        documents = deepcopy(self.documents)
        documents["result"]["recovery"]["physical_world_curriculum_history_restored"] = True
        with self.assertRaisesRegex(ValueError, "restarted physical state"):
            self.validate(documents)

    def test_checkpoint_substitution_rejected(self):
        documents = deepcopy(self.documents)
        documents["completion"]["checkpoint"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "preselected recovered terminal checkpoint"):
            self.validate(documents)

    def test_external_report_must_bind_actual_result(self):
        documents = deepcopy(self.documents)
        documents["external_verification"]["result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "actual completed recovery"):
            self.validate(documents)

    def test_readonly_receipt_and_original_fresh_gate(self):
        from scripts.evaluate_shared_social_tracking import admit_training as fresh_admit
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(admission.receipt_document()))
            with self.assertRaisesRegex(ValueError, "readonly recovery artifact"):
                admission.admit_training(path)
            path.chmod(0o444)
            checkpoint, _, accepted = admission.admit_training(path)
            self.assertEqual(checkpoint.name, "step_012500.pt")
            self.assertTrue(accepted["optimizer_recovery"]["physical_world_restarted"])
            with self.assertRaisesRegex(ValueError, "actual successful training-process exit"):
                fresh_admit(path)

    def test_fresh_S2_receipt_rejected_without_inference(self):
        path = admission.ROOT / "docs/style_improvement_20261005/shared_social_full_epochs_v1/pilot2000_execution.json"
        with self.assertRaisesRegex(ValueError, "explicit recovery execution receipt"):
            admission.admit_training(path)

    def test_all_stage_derivations_and_parent_pins(self):
        for stage in evaluation.PARENTS:
            with self.subTest(stage=stage):
                module, binding = evaluation.prepare(stage)
                self.assertTrue(binding["derivation"]["complete_source_inverse"])
                self.assertFalse(binding["derivation"]["tracking_style_navigation_metric_or_physics_math_changed"])
                self.assertIsNot(module.admit_training, admission.admit_training)
        self.assertNotIn("torch", sys.modules)

    def test_ambiguous_child_routing_rejected(self):
        source = (admission.ROOT / evaluation.PARENTS["tracking_style"][0]).read_text()
        broken = source.replace('[sys.executable, str(tracking), "--execution-receipt",',
                                '[sys.executable, "other-entry", "--execution-receipt",', 1)
        with self.assertRaisesRegex(ValueError, "Ambiguous recovery evaluation routing"):
            evaluation.derive(broken, "tracking_style")

    def test_parent_source_change_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.py"
            original, sha = evaluation.PARENTS["tracking"]
            path.write_text((admission.ROOT / original).read_text() + "\n# changed\n")
            with patch.dict(evaluation.PARENTS, tracking=(str(path), sha)):
                with self.assertRaisesRegex(ValueError, "Original recovery evaluation parent changed"):
                    evaluation.prepare("tracking")


if __name__ == "__main__":
    unittest.main()
