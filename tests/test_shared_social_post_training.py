"""Post-training ownership gates; stdlib fixtures are not execution evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_shared_social_post_training as post


class PostTrainingOwnershipTests(unittest.TestCase):
    def fixture(self, directory, strategy="popart", updates=2000):
        config = directory / "config.json"
        document = {"fixture_only": True}
        config.write_text(json.dumps(document))
        run = directory / "run"
        run.mkdir()
        result = run / "result.json"
        result.write_text(json.dumps({"profile": document, "completed_updates": updates,
                                     "actual_training_transitions": 16384000}))
        receipt = directory / "receipt.json"
        receipt.write_text('{"fixture_only": true}')
        registration = directory / "registration.json"
        registration.write_text(json.dumps({"schema": "shared_social.popart_paired_experiment_registration.v1",
            "configuration": post.record(config), "run_directory": str(run), "preselected_checkpoint": "step_002000.pt"}))
        admission = (run / "step_002000.pt", "unused fixture SHA",
                     {"result": post.record(result), "value_normalization": strategy})
        arguments = ["--execution-receipt", str(receipt), "--registration", str(registration),
                     "--output", str(directory / "evaluation")]
        return arguments, admission

    def test_rejects_incomplete_budget_and_wrong_family_before_launch(self):
        for strategy, updates in (("legacy", 2000), ("popart", 1999)):
            with tempfile.TemporaryDirectory() as path:
                arguments, admission = self.fixture(Path(path), strategy, updates)
                with patch.object(post, "admit_training", return_value=admission), patch.object(post.subprocess, "run") as child:
                    with self.assertRaisesRegex(ValueError, "complete preregistered"):
                        post.main(arguments)
                    child.assert_not_called()
                self.assertFalse((Path(path) / "evaluation").exists())

    def test_nonzero_CPU_child_stops_all_GPU_work(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            arguments, admission = self.fixture(directory)
            original_record = post.record

            def identity(target):
                return {"sha256": post.DIFFUSION_SHA} if Path(target) == post.DIFFUSION else original_record(target)

            with patch.object(post, "admit_training", return_value=admission), patch.object(post, "record", side_effect=identity), patch.object(
                    post.subprocess, "run", return_value=SimpleNamespace(returncode=139)) as child:
                self.assertEqual(post.main(arguments), 1)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            result = json.loads((directory / "evaluation/execution.json").read_text())
            self.assertEqual(result["steps"][0]["actual_exit_code"], 139)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["paper_effect_achieved"])

    def test_zero_exit_with_missing_measurement_is_failed_and_preserved(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            arguments, admission = self.fixture(directory)
            original_record = post.record

            def identity(target):
                return {"sha256": post.DIFFUSION_SHA} if Path(target) == post.DIFFUSION else original_record(target)

            with patch.object(post, "admit_training", return_value=admission), patch.object(post, "record", side_effect=identity), patch.object(
                    post.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
                self.assertEqual(post.main(arguments), 1)
                self.assertEqual(child.call_count, 1)
            result = json.loads((directory / "evaluation/execution.json").read_text())
            self.assertEqual(result["steps"][0]["actual_exit_code"], 0)
            self.assertEqual(result["steps"][0]["status"], "failed")
            self.assertTrue((directory / "evaluation/actor_CPU_inference.log").exists())


if __name__ == "__main__":
    unittest.main()
