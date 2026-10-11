"""Actual prefix admission and fail-closed resumption exits, without SDK."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts import resume_shared_social_recovery_evaluation as resume


class ResumedRecoveryEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.here = resume.ROOT / "docs/style_improvement_20261008/shared_social_S3R_evaluation_v2"
        cls.completion = cls.here / "completion.json"
        cls.inventory = cls.here / "failed_output_inventory.json"
        cls.prefix = resume.admit_prefix(cls.completion, cls.inventory)

    def test_real_prefix_keeps_successes_and_the_failed_exit(self):
        self.assertEqual([row["name"] for row in self.prefix["successful_prefix_steps"]], resume.PREFIX_NAMES)
        self.assertEqual(len(self.prefix["retained_outputs"]), 54)
        self.assertEqual(self.prefix["original_pipeline_actual_exit_code"], 1)
        self.assertEqual(self.prefix["original_navigation_actual_exit_code"], -11)
        self.assertNotIn("navigation", self.prefix["measurements"])

    def test_completed_original_owner_is_not_this_resume_family(self):
        document = resume.original.read(self.completion)
        document["status"] = "completed_evaluation_quality_unclassified"
        with patch.object(resume.original, "read", return_value=document):
            with self.assertRaisesRegex(ValueError, "preserved failed owner"):
                resume.admit_prefix(self.completion, self.inventory)

    def test_owner_exit_must_be_actual_integer_one(self):
        for code in (0, True, "1", 1.0, -11, None):
            with self.subTest(code=code):
                document = resume.original.read(self.completion)
                document["steps"][0]["actual_exit_code"] = code
                with patch.object(resume.original, "read", return_value=document):
                    with self.assertRaisesRegex(ValueError, "actual failed post-training owner exit"):
                        resume.admit_prefix(self.completion, self.inventory)

    def test_steps_require_order_budget_and_real_exits(self):
        good = {"status": "passed_execution", "steps": [
            {"name": name, "status": "passed", "actual_exit_code": 0} for name in resume.RETRY_NAMES]}
        resume.successful_steps(good, resume.RETRY_NAMES, "passed_execution")
        for mutation in ("missing", "reversed", "duplicated", "bool_exit", "float_exit", "nonzero", "running"):
            with self.subTest(mutation=mutation):
                bad = deepcopy(good)
                if mutation == "missing":
                    bad["steps"].pop()
                elif mutation == "reversed":
                    bad["steps"].reverse()
                elif mutation == "duplicated":
                    bad["steps"].append(deepcopy(bad["steps"][0]))
                elif mutation in ("bool_exit", "float_exit", "nonzero"):
                    bad["steps"][0]["actual_exit_code"] = {"bool_exit": False, "float_exit": 0.0, "nonzero": -11}[mutation]
                else:
                    bad["status"] = "running"
                with self.assertRaisesRegex(ValueError, "ordered complete stages"):
                    resume.successful_steps(bad, resume.RETRY_NAMES, "passed_execution")

    def test_previous_failure_is_not_a_completed_resumption(self):
        with self.assertRaisesRegex(ValueError, "actual combined evaluation completion"):
            resume.admit_completion(self.completion)

    def test_resume_cold_admission_does_not_import_sdk(self):
        self.assertNotIn("torch", resume.sys.modules)
        self.assertNotIn("isaacsim", resume.sys.modules)


if __name__ == "__main__":
    unittest.main()
