"""Synthetic CPU audit contracts; actual SDK evidence comes from separate exits."""
import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_shared_social_high_level as audit


def plans():
    return {"actor_ids": np.tile(np.arange(4, dtype=np.int64), 2),
            "control_step": np.repeat(np.array([0, 15], np.int64), 4),
            "raw_xy": np.zeros((8, 51, 2), np.float32), "reference_xy": np.zeros((8, 151, 2), np.float32),
            "reference_velocity_xy": np.zeros((8, 151, 2), np.float32)}


class HighLevelAuditTests(unittest.TestCase):
    def test_base_duration_replacements_restore_all_original_bytes(self):
        original = audit.AUDIT.read_text()
        for controls in (30, 6000):
            derived, proof = audit.derive_base_audit(original, controls)
            self.assertTrue(proof["whole_source_inverse"])
            self.assertFalse(proof["contact_episode_support_metric_math_changed"])
            tree = ast.parse(derived)
            call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and any(isinstance(child, ast.Constant) and child.value == "Require all registered controls at30Hz"
                                for child in ast.walk(node)))
            self.assertEqual(call.args[0].comparators[0].value, controls)
        with self.assertRaises(ValueError):
            audit.derive_base_audit(original, 30.0)
        with self.assertRaises(ValueError):
            audit.derive_base_audit(original.replace("steps == 6000", "steps == 5999"), 30)

    def test_duration_AST_only_changes_budget_nodes_and_full_duration_is_identical(self):
        module, _ = audit.load_source(audit.LOCOMOTION, audit.LOCOMOTION_SHA)
        helpers, _ = module.load_helpers()
        for function in (module.analyze, helpers[0].targets_from_saved_plans):
            _, proof = audit.duration_function(function, 6000)
            self.assertEqual(proof["numeric_literals_actually_changed"], 0)
            self.assertEqual(proof["original_AST_sha256"], proof["derived_AST_sha256"])
            changed, proof = audit.duration_function(function, 30)
            self.assertTrue(proof["exact_AST_inverse"])
            self.assertGreater(proof["numeric_literals_actually_changed"], 0)
            if function.__name__ == "targets_from_saved_plans":
                state = {key: np.zeros((30, 4, 2), np.float32) for key in
                         ("raw_target_xy", "target_xy", "reference_target_velocity_xy")}
                errors, phase = changed(state, plans())
                self.assertEqual(phase.shape, (30, 4))
                self.assertTrue((phase[0] == 1).all() and (phase[14] == 15).all() and (phase[15] == 1).all())
                self.assertTrue(all(value == 0 for value in errors.values()))
                state["target_xy"][0, 0, 0] = .01
                with self.assertRaisesRegex(ValueError, r"age\+1"):
                    changed(state, plans())
        source = inspect.getsource(helpers[0].targets_from_saved_plans).replace("np.arange(6000)", "np.arange(5999)", 1)
        with patch.object(audit.inspect, "getsource", return_value=source):
            with self.assertRaisesRegex(ValueError, "literal inventory"):
                audit.duration_function(helpers[0].targets_from_saved_plans, 30)

    def test_joint_plans_reject_partial_scene_rows_dtype_and_unordered_clocks(self):
        self.assertEqual(audit.joint_plan_contract(plans(), 30)["complete_joint_events"], 2)
        changed = plans()
        changed["actor_ids"][0] = 1
        with self.assertRaisesRegex(ValueError, "split"):
            audit.joint_plan_contract(changed, 30)
        changed = plans()
        changed["control_step"][4:] = 0
        with self.assertRaisesRegex(ValueError, "unordered"):
            audit.joint_plan_contract(changed, 30)
        changed = plans()
        changed["raw_xy"] = changed["raw_xy"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "FP32"):
            audit.joint_plan_contract(changed, 30)
        changed = plans()
        changed["actor_ids"] = changed["actor_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "complete"):
            audit.joint_plan_contract(changed, 30)

    def test_initial_link_binds_same_run_first_clock_episodes_and_all_anchors(self):
        # Mock the separately tested archive admission. This tests link phase
        # checks, not a real initial capture or genuine SDK exit.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root/"paired_initial_condition.json"
            document.write_text("{}\n")
            arrays = root/"initial_arrays.npz"
            np.savez(arrays, common_rigid_body_pos=np.zeros((4, 24, 3), np.float32),
                     common_rigid_body_vel=np.zeros((4, 24, 3), np.float32), world_episode_ids=np.ones(4, np.int64))
            ground = root/"ground_support.npz"
            np.savez(ground, physics_step_before=np.array([17]), episode_ids=np.ones((1, 4), np.int64))
            record = {"arrays": audit.record(arrays), "contract": {"capture_kind": "live_original_world",
                       "clock": {"physics_step_after": 17}}}
            result = {"paired_initial_condition": {"document": audit.record(document), "arrays": record["arrays"]}}
            with patch.object(audit, "read_snapshot", return_value=record):
                out = audit.initial_capture_link(document, result, ground, plans(), root)
                self.assertTrue(out["all_four_first_raw_reference_position_velocity_anchors_equal"])
                record["contract"]["capture_kind"] = "CPU_contract_fixture"
                with self.assertRaisesRegex(ValueError, "CPU fixture"):
                    audit.initial_capture_link(document, result, ground, plans(), root)
                record["contract"]["capture_kind"] = "live_original_world"
                altered = plans()
                altered["reference_velocity_xy"][0, 0, 0] = .01
                with self.assertRaisesRegex(ValueError, "anchor differs"):
                    audit.initial_capture_link(document, result, ground, altered, root)
                np.savez(ground, physics_step_before=np.array([18]), episode_ids=np.ones((1, 4), np.int64))
                with self.assertRaisesRegex(ValueError, "counter/episode"):
                    audit.initial_capture_link(document, result, ground, plans(), root)


if __name__ == "__main__":
    unittest.main()
