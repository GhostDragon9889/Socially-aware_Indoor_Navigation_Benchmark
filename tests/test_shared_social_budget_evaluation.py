"""Budget contracts; these CPU fixtures do not certify a trained policy."""
from copy import deepcopy
import unittest

from scripts import run_shared_social_budget_evaluation as evaluation


class BudgetEvaluationTests(unittest.TestCase):
    def test_registered_exposure_matches_variable_budget(self):
        config = {"runtime": {"actors": 256, "rollout_controls": 32, "updates": 12500}}
        registered = {"schema": evaluation.SCHEMA, "planned_updates": 12500,
                      "planned_physical_transitions": 102400000, "preselected_checkpoint": "step_012500.pt"}
        self.assertEqual(evaluation.registered_budget(registered, config), (12500, 102400000))
        for key, wrong in (("planned_updates", 2000), ("planned_physical_transitions", 16384000),
                           ("preselected_checkpoint", "step_002000.pt"), ("planned_updates", True)):
            changed = dict(registered, **{key: wrong})
            with self.assertRaises(ValueError):
                evaluation.registered_budget(changed, config)

    def test_bool_empty_and_partial_group_exposure_is_rejected(self):
        config = {"runtime": {"actors": 8, "rollout_controls": 4, "updates": 1}}
        registered = {"schema": evaluation.SCHEMA, "planned_updates": 1,
                      "planned_physical_transitions": 32, "preselected_checkpoint": "step_000001.pt"}
        for key, wrong in (("actors", 7), ("updates", False), ("rollout_controls", 0)):
            changed = deepcopy(config)
            changed["runtime"][key] = wrong
            with self.assertRaisesRegex(ValueError, "exact integers"):
                evaluation.registered_budget(registered, changed)

    def test_derivation_restores_full_pipeline_and_does_not_change_stage_commands(self):
        original = evaluation.PARENT.read_text()
        derived, proof = evaluation.derive(original)
        self.assertTrue(proof["complete_source_inverse"])
        self.assertFalse(proof["tracking_style_navigation_and_child_exit_math_changed"])
        # Everything after input admission, including all child commands and
        # result/cleanup/source checks, must remain byte-for-byte identical.
        marker = '    if record(DIFFUSION)["sha256"] != DIFFUSION_SHA:'
        self.assertEqual(original[original.index(marker):], derived[derived.index(marker):])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            evaluation.derive(original.replace('result["completed_updates"] != 2000', 'result["completed_updates"] != 1999'))

    def test_entry_binds_itself_without_replacing_existing_child_paths(self):
        module, binding = evaluation.prepare()
        self.assertEqual(module.SOURCE_PATHS[1], "scripts/audit_shared_social_inference.py")
        self.assertEqual(module.SOURCE_PATHS[-1], "scripts/run_shared_social_budget_evaluation.py")
        self.assertEqual(binding["parent"]["sha256"], evaluation.PARENT_SHA)


if __name__ == "__main__":
    unittest.main()
