"""Cold tests for real native resources and explicitly synthetic exit guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

from CrowdSim.humanoid.application import episode_native_execution as bridge


class EpisodeNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.registration = Path(cls.temporary.name) / "registration.json"
        cls.document = bridge.registration_document()
        bridge.original._write_new(cls.registration, cls.document)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_real_cohorts_and_refusals_preserved(self):
        matrix, _ = bridge.original.current_matrix()
        candidate, _ = bridge.admit_registration(self.registration)
        self.assertEqual(len(candidate["cases"]), 156)
        self.assertEqual(candidate["resource_admitted_cohorts"], 144)
        self.assertEqual(candidate["resource_rejected_cohorts"], 12)
        self.assertEqual(candidate["CPU_resources"], matrix["CPU_resources"])
        self.assertEqual(candidate["inputs"], matrix["inputs"])
        self.assertEqual(candidate["scenes"], matrix["scenes"])
        self.assertFalse(candidate["author_eight_scene_tasks_or_seeds_recovered"])
        for old, new in zip(matrix["cases"], candidate["cases"], strict=True):
            if not old["arms"]:
                expected = deepcopy(old)
                expected["same_AMP_checkpoint_for_both_arms"] = candidate["preselected_episode_terminal_checkpoint"]
                self.assertEqual(new, expected)
        self.assertTrue(candidate["registration_derivation"]["complete_inherited_inverse"])

    def test_private_namespace_keeps_original_certificate_guard(self):
        original_callback = bridge.original.admit_S3_owner_completion
        namespace = bridge._namespace(self.registration)
        self.assertIs(bridge.original.admit_S3_owner_completion, original_callback)
        self.assertEqual(bridge.original.CHECKPOINT.name, "step_012500.pt")
        self.assertEqual(namespace.CHECKPOINT.name, "step_002000.pt")
        self.assertNotEqual(namespace.TRAINING_RECEIPT, bridge.original.TRAINING_RECEIPT)
        self.assertIsNot(namespace.admit_S3_owner_completion, original_callback)

    def test_native_command_changes_only_selected_receipt_and_outputs(self):
        old = bridge.original.prepare_case("Office_N03_seed7")
        new = bridge.prepare_case("Office_N03_seed7", registration_path=self.registration)
        self.assertEqual(new["request"], old["request"])
        self.assertEqual(new["raw_sources"], old["raw_sources"])
        self.assertEqual(new["style_sources"], old["style_sources"])
        self.assertEqual(len(new["stages"]), 4)
        for left, right in zip(old["stages"], new["stages"], strict=True):
            expected = [str(bridge.candidate.RECEIPT) if x == str(bridge.original.TRAINING_RECEIPT)
                        else x.replace(str(bridge.OLD_OUTPUT_ROOT), str(bridge.OUTPUT_ROOT)) for x in left["SDK_command"]]
            self.assertEqual(right["SDK_command"], expected)
            self.assertEqual(right["planner"], left["planner"])
            self.assertEqual(right["controls"], left["controls"])
        self.assertFalse(new["episode_native_binding"]["SDK_entry_raw_capture_style_and_metric_math_changed"])
        child = subprocess.run([sys.executable, "-c",
            "import sys;from CrowdSim.humanoid.application import episode_native_execution as e;"
            "e.prepare_case('Office_N03_seed7');assert 'torch' not in sys.modules"], cwd=bridge.ROOT,
            capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_rejected_resource_cannot_be_selected_or_replaced(self):
        with self.assertRaisesRegex(ValueError, "resource rejection"):
            bridge.prepare_case("Hospital_N15_seed7", registration_path=self.registration)

    def test_another_completion_path_cannot_release_training(self):
        with self.assertRaisesRegex(ValueError, "actual completed owner"):
            bridge.admit_episode_owner_completion(bridge.original.S3 / "invented_completion.json",
                                                  registration_path=self.registration)

    def test_real_S2_completion_cannot_release_episode_native(self):
        actual = json.loads(bridge.candidate.BASELINE.read_text())
        with self.assertRaisesRegex(ValueError, "all actual fresh episode"):
            bridge.candidate.validate_owner(actual)

    def test_failed_actual_episode_completion_stops_before_exit_publication(self):
        completed = json.loads(bridge.candidate.COMPLETION.read_text())
        self.assertEqual(completed["status"], "failed_preserved")
        self.assertNotEqual(completed["steps"][-1]["actual_exit_code"], 0)
        with self.assertRaisesRegex(ValueError, "all actual fresh episode"):
            bridge.publish_sdk_exit("Office_N03_seed7", planner="diffusion", controls=30,
                observation_path=Path(self.temporary.name) / "no_tool_observation.json",
                owner_completion=bridge.candidate.COMPLETION, registration_path=self.registration)
        self.assertFalse((bridge.OUTPUT_ROOT / "Office_N03_seed7/observations/diffusion_C30_SDK_exit.json").exists())

    def test_tampered_resource_rejection_cannot_enter_private_namespace(self):
        # Synthetic altered registration, never a resource observation.
        wrong = deepcopy(self.document)
        wrong["resource_rejected_cohorts"] = 0
        path = Path(self.temporary.name) / "synthetic_tampered_registration.json"
        bridge.original._write_new(path, wrong)
        with self.assertRaisesRegex(ValueError, "registration or declared sources changed"):
            bridge._namespace(path)

    def test_original_observer_keeps_failed_exits_and_rejects_inferred_success(self):
        plan = bridge.prepare_case("Office_N03_seed7", registration_path=self.registration)
        command = plan["stages"][0]["SDK_command"]
        fixture = {"schema": bridge.original.OBSERVATION_SCHEMA,
                   "cmd": bridge.original.tool_command(command, cuda="0"), "workdir": str(bridge.ROOT),
                   "actual_terminal_response_observed": True,
                   "response": {"exit_code": 0, "chunk_id": "synthetic_contract_fixture_not_an_actual_exit"}}
        self.assertEqual(bridge.original.validate_tool_observation(fixture, command=command, cuda="0")["actual_exit_code"], 0)
        for mode in ("bool", "missing", "inferred"):
            with self.subTest(mode=mode):
                wrong = deepcopy(fixture)
                if mode == "bool":
                    wrong["response"]["exit_code"] = False
                elif mode == "missing":
                    wrong["response"].pop("chunk_id")
                else:
                    wrong["actual_terminal_response_observed"] = False
                with self.assertRaisesRegex(ValueError, "actual terminal tools API"):
                    bridge.original.validate_tool_observation(wrong, command=command, cuda="0")
        failed = deepcopy(fixture)
        failed["response"]["exit_code"] = -11
        self.assertEqual(bridge.original.validate_tool_observation(failed, command=command, cuda="0")["actual_exit_code"], -11)


if __name__ == "__main__":
    unittest.main()
