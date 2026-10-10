"""Minimal metadata/closure checks; in-memory receipts are never actual runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("new_native_admission_test", HERE / "validate_native_admission.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
REGISTRATION = json.loads((HERE / "registration.json").read_text())
CHECKPOINT = {"path": REGISTRATION["future_checkpoint"]["path"], "bytes": 123, "sha256": "a" * 64}


def receipt_fixture():
    return {"schema": "pacer_reward.actual_native_launch.v1", "status": "closed", "actors": 4,
            "exit_code": 0, "child_exit_observed": True, "driver_completed": True, "prepare_only": False,
            "command": REGISTRATION["cases"]["4"]["command"], "amp_checkpoint": CHECKPOINT,
            "selected_source_before": REGISTRATION["current_source_binding"]["production_source"],
            "selected_source_after": REGISTRATION["current_source_binding"]["production_source"]}


def test_preparation_never_reads_future_checkpoint_capture_or_admission(monkeypatch):
    original = TOOL.record

    def current_only(path):
        assert str(path) != REGISTRATION["future_checkpoint"]["path"]
        assert str(path) != REGISTRATION["future_checkpoint"]["terminal_admission_path"]
        assert "pacer_reward_metrics_v1/navigation_raw" not in str(path)
        assert "native_warehouse_pacer_reward_clearance08_200s_v1" not in str(path)
        return original(path)

    monkeypatch.setattr(TOOL, "record", current_only)
    assert TOOL.read_registration() == REGISTRATION
    assert REGISTRATION["future_checkpoint"]["sha256"] is None
    assert REGISTRATION["future_checkpoint"]["bytes"] is None
    assert not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)


def test_exact_native_commands_and_physical_progression_only():
    assert REGISTRATION["style_score_is_launch_gate"] is False
    for actors in (4, 15):
        command = TOOL.native_command(actors)
        assert command[command.index("--actors") + 1] == str(actors)
        assert command[command.index("--duration-seconds") + 1] == "200"
        assert command[command.index("--seed") + 1] == "7"
        assert command[command.index("--arm-style-alpha") + 1] == "0"
        assert "--native-dynbench" in command and "--inference-only" in command
        assert command[command.index("--config") + 1] == str(TOOL.PROFILE)
        assert REGISTRATION["cases"][str(actors)]["actual_native_executed"] is False


def test_closed_owner_exact_source_and_observed_exit_required():
    receipt = receipt_fixture()
    assert TOOL.validate_closed_receipt(receipt, 4, CHECKPOINT) == receipt
    for key, value in [("child_exit_observed", False), ("exit_code", 0.0), ("prepare_only", True),
                       ("selected_source_after", {"source_count": 183})]:
        changed = deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            TOOL.validate_closed_receipt(changed, 4, CHECKPOINT)


def test_incomplete_or_nonzero_native_physical_reports_fail_before_future_owner_read():
    report = {"schema": "pacer_reward.native200s_independent_CPU_audit.v1",
              "status": "raw_sensor_source_contracts_passed_bounded_zero_fall_observed",
              "native_200s_artifacts_independently_audited": True,
              "counts": {"controls": 6000, "actors": 4, "duration_s": 200.0,
                         "falls": 0, "static_actor_controls": 0, "peer_pair_controls": 0}}
    for key, value in [("controls", 5999), ("falls", 1), ("static_actor_controls", 1), ("peer_pair_controls", 1)]:
        changed = deepcopy(report)
        changed["counts"][key] = value
        with pytest.raises(ValueError, match="zero-physical"):
            TOOL.validate_closed_native(changed, 4, CHECKPOINT)
