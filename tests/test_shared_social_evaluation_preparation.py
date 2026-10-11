"""Stdlib tests of evaluation admission; fixtures are never execution evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import TERMINAL, derive_single_asset_release, record
from CrowdSim.humanoid.runtime import social_observation_producer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_shared_tracking", ROOT / "scripts/evaluate_shared_social_tracking.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
style_spec = importlib.util.spec_from_file_location("_test_shared_style", ROOT / "scripts/evaluate_shared_social_style.py")
style = importlib.util.module_from_spec(style_spec)
style_spec.loader.exec_module(style)
pipeline_spec = importlib.util.spec_from_file_location("_test_shared_pipeline", ROOT / "scripts/run_shared_social_evaluation.py")
pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(pipeline)
navigation_spec = importlib.util.spec_from_file_location("_test_shared_navigation", ROOT / "scripts/evaluate_shared_social_navigation.py")
navigation = importlib.util.module_from_spec(navigation_spec)
navigation_spec.loader.exec_module(navigation)
diagnostic_spec = importlib.util.spec_from_file_location("_test_shared_diagnostic", ROOT / "scripts/diagnose_shared_social_training.py")
diagnostic = importlib.util.module_from_spec(diagnostic_spec)
diagnostic_spec.loader.exec_module(diagnostic)


class SharedEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self._modules_before = set(sys.modules)

    def assert_no_new_heavy_modules(self, prefix):
        self.assertFalse(any(name == prefix or name.startswith(prefix + '.')
            for name in set(sys.modules) - self._modules_before))

    def test_single_asset_guard_keeps_sdk_release_logic(self):
        source = TERMINAL.read_text()
        original = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_release_owned_asset_simulation_views")
        derived, proof = derive_single_asset_release(source)
        new = ast.parse(derived).body[0]
        old_nodes, new_nodes = list(ast.walk(original)), list(ast.walk(new))
        self.assertEqual(len(old_nodes), len(new_nodes))
        count_changes = text_changes = 0
        for old, current in zip(old_nodes, new_nodes, strict=True):
            if isinstance(old, ast.Constant) and old.value != current.value:
                if type(old.value) is int:
                    self.assertEqual((old.value, current.value), (4, 1))
                    count_changes += 1
                else:
                    self.assertIsInstance(old.value, str)
                    text_changes += 1
                old.value = current.value
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(new, include_attributes=False))
        self.assertEqual(count_changes, 5)
        self.assertEqual(text_changes, 4)
        self.assertTrue(proof["exact_inverse"])

    def test_ambiguous_owner_guard_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_single_asset_release(TERMINAL.read_text().replace("len(assets) != 4", "len(assets) != 3"))

    def test_full22_functions_load_without_sdk_or_torch(self):
        parent, functions, proof = entry.load_reference_functions()
        self.assertTrue(proof["derivation"]["whole_byte_reverse_exact"])
        self.assertIn("run_reference_clip", functions)
        self.assertEqual(parent.MANIFEST_SHA, "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca")
        self.assert_no_new_heavy_modules("torch")
        self.assert_no_new_heavy_modules("isaacsim")

    def test_navigation_derivation_keeps_physical_and_metric_functions(self):
        original = navigation.PARENT.read_text()
        derived, proof = navigation.derive_entry(original)
        before = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(original).body
                  if isinstance(node, ast.FunctionDef)}
        after = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(derived).body
                 if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(before), set(after))
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {"_sources", "main"})
        self.assertEqual(proof["replacement_count"], 10)
        self.assertTrue(proof["exact_inverse"])
        self.assertFalse(proof["physical_driver_changes_added"])
        self.assert_no_new_heavy_modules("torch")
        self.assert_no_new_heavy_modules("isaacsim")

    def test_navigation_rejects_changed_launch_ownership(self):
        altered = navigation.PARENT.read_text().replace('owned["app"] = launcher(options).app',
                                                       'owned["app"] = other_launcher(options).app')
        with self.assertRaisesRegex(ValueError, "Ambiguous navigation"):
            navigation.derive_entry(altered)

    def test_scripted_origin_migration_keeps_all_observation_geometry(self):
        original = social_observation_producer.PARENT.read_text()
        derived, proof = social_observation_producer.derive_origin_guard(original)
        old = ast.parse(original)
        new = ast.parse(derived)
        self.assertEqual(len(old.body), len(new.body))
        for before, after in zip(old.body, new.body, strict=True):
            if isinstance(before, ast.FunctionDef) and before.name == "_module":
                continue
            self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))
        self.assertTrue(proof["exact_inverse"])
        self.assertFalse(proof["observation_math_changed"])
        self.assert_no_new_heavy_modules("torch")

    def test_training_windows_use_cumulative_count_differences_and_exact_exposure(self):
        rows = [{"driver_reason_counts": {"fell_actor_controls": count}, "reward_mean": reward,
                 "return_mean": -1000., "optimizer_minibatches": 2, "policy_target_kl_stopped": True}
                for count, reward in ((3, .5), (5, 1.), (12, .75))]
        result = diagnostic.windows(rows, 8192, size=2)
        self.assertEqual(result[0]["reason_flags"], {"fell_actor_controls": 5})
        self.assertEqual(result[1]["reason_flags"], {"fell_actor_controls": 7})
        self.assertEqual([item["actor_controls"] for item in result], [16384, 8192])
        self.assertEqual([item["KL_stop_updates"] for item in result], [2, 1])
        self.assert_no_new_heavy_modules("torch")

    def fixture_receipt(self, directory):
        run = directory / "run"
        run.mkdir()
        checkpoint = run / "step_002000.pt"
        checkpoint.write_bytes(b"fixture checkpoint: not weights")
        checkpoint.chmod(0o444)
        result = run / "result.json"
        result.write_text(json.dumps({"status": "passed_local_training", "cleanup_succeeded": True,
            "cleanup_errors": [], "source_profile_before_after_equal": True,
            "completed_updates": 2000, "actual_training_transitions": 16384000,
            "profile": {"runtime": {"updates": 2000}}}))
        audit = directory / "audit.json"
        audit.write_text(json.dumps({"schema": "shared_social.completed_training_CPU_audit.v1", "status": "passed",
            "CUDA_initialized": False, "updates": 2000, "physical_transitions": 16384000,
            "checkpoint": record(checkpoint)}))
        receipt = directory / "receipt.json"
        receipt.write_text(json.dumps({"schema": "shared_social.actual_training_execution.v1", "terminal_exit_code": 0,
            "run_directory": str(run), "result_identity": record(result), "checkpoint_identity": record(checkpoint),
            "audit_identity": record(audit)}))
        return receipt

    def test_fixture_metadata_admitted_but_not_a_quality_claim(self):
        with tempfile.TemporaryDirectory() as path:
            receipt = self.fixture_receipt(Path(path))
            checkpoint, sha, admission = entry.admit_training(receipt)
            self.assertEqual(checkpoint.name, "step_002000.pt")
            self.assertEqual(sha, record(checkpoint)["sha256"])
            self.assertNotIn("quality_passed", admission)

    def test_popart_family_requires_matching_independent_numeric_audit(self):
        with tempfile.TemporaryDirectory() as path:
            receipt_path = self.fixture_receipt(Path(path))
            receipt = json.loads(receipt_path.read_text())
            result_path = Path(receipt["result_identity"]["path"])
            result = json.loads(result_path.read_text())
            binding = entry.social_popart_training.describe()
            result["canonical_entry"] = {"value_normalization": "popart", "numeric_migration": binding,
                "entry": record(ROOT / "scripts/train_shared_social.py"), "World_collector_expert_physics_changed": False}
            result_path.write_text(json.dumps(result))
            receipt["result_identity"] = record(result_path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "numeric source and CPU audit disagree"):
                entry.admit_training(receipt_path)
            audit_path = Path(receipt["audit_identity"]["path"])
            audit = json.loads(audit_path.read_text())
            audit["explicit_value_normalization"] = binding
            audit_path.write_text(json.dumps(audit))
            receipt["audit_identity"] = record(audit_path)
            receipt_path.write_text(json.dumps(receipt))
            admission = entry.admit_training(receipt_path)[2]
            self.assertEqual(admission["value_normalization"], "popart")
            self.assert_no_new_heavy_modules("torch")
            result["canonical_entry"]["value_normalization"] = "legacy"
            result_path.write_text(json.dumps(result))
            receipt["result_identity"] = record(result_path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                entry.admit_training(receipt_path)

    def test_nonzero_exit_and_changed_terminal_result_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            receipt_path = self.fixture_receipt(Path(path))
            receipt = json.loads(receipt_path.read_text())
            receipt["terminal_exit_code"] = 139
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "actual successful"):
                entry.admit_training(receipt_path)
            receipt["terminal_exit_code"] = 0
            receipt_path.write_text(json.dumps(receipt))
            result = Path(receipt["result_identity"]["path"])
            result.write_text("{}")
            with self.assertRaisesRegex(ValueError, "terminal result changed"):
                entry.admit_training(receipt_path)

    def test_social_style_requires_own_training_provenance_and_cleanup(self):
        with tempfile.TemporaryDirectory() as path:
            manifest = Path(path) / "manifest.json"
            manifest.write_text(json.dumps({"splits": {"train": {"sha256": "fixture bank identity",
                                                                  "clips": [{}] * 2848}}}))
            provenance = {"manifest": str(manifest), "manifest_sha256": record(manifest)["sha256"],
                          "source_sha256": "fixture bank identity", "clips": 2848, "split": "train"}
            runner = {"schema": "local.neutral_smpl.social_full22_tracking.v2", "cleanup_succeeded": True,
                "cleanup_errors": [], "training_execution_admission": {"terminal_exit_code": 0},
                "controller_provenance": {"AMASS": provenance}}
            self.assertEqual(style.training_manifest(runner)[0], manifest)
            runner["cleanup_succeeded"] = False
            with self.assertRaisesRegex(ValueError, "cleanup"):
                style.training_manifest(runner)
            runner["cleanup_succeeded"] = True
            runner["controller_provenance"] = {"dataset": provenance}
            with self.assertRaisesRegex(ValueError, "training provenance"):
                style.training_manifest(runner)

    def test_fixture_cuda_audit_and_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            receipt_path = self.fixture_receipt(Path(path))
            receipt = json.loads(receipt_path.read_text())
            audit_path = Path(receipt["audit_identity"]["path"])
            audit = json.loads(audit_path.read_text())
            audit["CUDA_initialized"] = True
            audit_path.write_text(json.dumps(audit))
            receipt["audit_identity"] = record(audit_path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "disagree"):
                entry.admit_training(receipt_path)
            audit["CUDA_initialized"] = False
            audit_path.write_text(json.dumps(audit))
            receipt["audit_identity"] = record(audit_path)
            receipt_path.write_text(json.dumps(receipt))
            checkpoint = Path(receipt["checkpoint_identity"]["path"])
            checkpoint.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "readonly terminal"):
                entry.admit_training(receipt_path)

    def test_pipeline_retains_failed_child_and_stops_next_steps(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            receipt = directory / "fixture_receipt.json"
            receipt.write_text("{}")
            output = directory / "evaluation"
            with patch("scripts.evaluate_shared_social_tracking.admit_training",
                       return_value=(None, None, {"value_normalization": "legacy"})), patch.object(
                    pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=139)) as child:
                code = pipeline.main(["--execution-receipt", str(receipt), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(child.call_count, 1)
            result = json.loads((output / "execution.json").read_text())
            self.assertEqual(result["steps"][0]["actual_exit_code"], 139)
            self.assertEqual(result["status"], "failed")
            self.assertTrue((output / "lifecycle_smoke.log").exists())
            self.assertFalse((output / "full22.log").exists())

    def test_pipeline_rejects_exit_zero_with_failed_cleanup(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            receipt = directory / "fixture_receipt.json"
            receipt.write_text("{}")
            output = directory / "evaluation"

            def child(command, **unused):
                target = Path(command[-1])
                target.mkdir()
                (target / "result.json").write_text(json.dumps({"status": "passed_lifecycle_smoke", "cleanup_succeeded": False}))
                return SimpleNamespace(returncode=0)

            with patch("scripts.evaluate_shared_social_tracking.admit_training",
                       return_value=(None, None, {"value_normalization": "legacy"})), patch.object(
                    pipeline.subprocess, "run", side_effect=child) as launched:
                code = pipeline.main(["--execution-receipt", str(receipt), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(launched.call_count, 1)
            result = json.loads((output / "execution.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("cleanup disagree", result["error"])
            self.assertFalse(result["paper_effect_achieved"])


if __name__ == "__main__":
    unittest.main()


def test_pure_preparation_import_and_operations_are_sdk_free_in_fresh_process():
    # Pytest collection imports Torch for other modules; only a fresh process
    # can prove that this preparation pipeline does not bootstrap the SDK.
    source = """
import importlib.util, sys, unittest
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), str(root)]
spec = importlib.util.spec_from_file_location('cold_preparation_tests', root / 'tests/test_shared_social_evaluation_preparation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = ['test_navigation_derivation_keeps_physical_and_metric_functions',
 'test_scripted_origin_migration_keeps_all_observation_geometry',
 'test_training_windows_use_cumulative_count_differences_and_exact_exposure',
 'test_popart_family_requires_matching_independent_numeric_audit']
result = unittest.TextTestRunner().run(unittest.TestSuite(module.SharedEvaluationPreparationTests(name) for name in names))
assert result.wasSuccessful()
assert not any(name.split('.')[0] in {'torch','isaaclab','isaacsim','omni','pxr','warp'} for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', source, str(ROOT)], check=True,
                   capture_output=True, text=True, timeout=30)
