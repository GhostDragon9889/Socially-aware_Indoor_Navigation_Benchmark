"""CPU metadata gates, not fabricated checkpoint or physical validation.

Synthetic metadata tests are never saved as runtime evidence. Real registered
schema/source bytes and existing22 baselines supply the compatibility oracle.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/style_improvement_20261002"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


EVIDENCE = module("test_state_action_evidence", DOCS / "state_action_evaluation_evidence_v1.py")
TERMINAL = module("test_state_action_terminal", DOCS / "audit_state_action_terminal2000_v1.py")
EXECUTOR = module("test_frozen_state_action_executor", DOCS / "run_isolated_state_action_evaluation_v1.py")
REG = json.loads((DOCS / "state_action_evaluation_registration_v1/registration.json").read_text())


@pytest.fixture
def tracking():
    value = json.loads((ROOT / "output/diffusion_amp/motion_tracking_mixed_curriculum_step2000_full_validation_v1/result.json").read_text())
    value["arguments"] = deepcopy(REG["tracking"]["parsed"])
    value["arguments"]["checkpoint_sha256"] = "a" * 64
    return value


@pytest.fixture
def navigation():
    return {"status": "passed", "arguments": deepcopy(REG["navigation"]["parsed"]),
            "actual_control_steps": 6000, "physics": {"fall_definition": {
                "pelvis_ground_height_less_than_m": .45, "root_up_z_less_than": .1}}}


@pytest.fixture
def terminal():
    report = json.loads((DOCS / "state_action_isolated_checkpoint1000_review_v1/audit.json").read_text())
    row = report["checkpoints"][0]
    row.update(updates=2000, transitions=16384000)
    row["checkpoint"] = {"path": REG["future_checkpoint"]["path"], "sha256": "a" * 64, "bytes": 1}
    result = json.loads((ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_pilot_v2/result_step8.json").read_text())
    result.update(update_count=2000, physical_transition_count=16384000,
                  group_transition_exposure=deepcopy(row["group_exposure"]),
                  fall_resets=row["cumulative_training_falls_at_milestone"],
                  time_limit_resets=row["cumulative_training_time_limits_at_milestone"])
    result["arguments"].update(updates=2000, checkpoint_every=500, resume=None)
    return report, result


def test_terminal_addition_retains_executor_schema(terminal):
    report, result = terminal
    row = TERMINAL.validate_terminal(report, result)
    assert EXECUTOR.validate_admission(report, row["checkpoint"]) is row


@pytest.mark.parametrize("field,value", [("status", "failed"), ("update_count", 1999),
    ("physical_transition_count", 16383999), ("fall_resets", -1), ("time_limit_resets", -1)])
def test_terminal_rejects_nonterminal_or_different_actual_counters(terminal, field, value):
    report, result = terminal
    result[field] = value
    with pytest.raises(ValueError):
        TERMINAL.validate_terminal(report, result)


@pytest.mark.parametrize("key,value", [("resume", "old.pt"), ("num_envs", True),
    ("checkpoint_every", 200), ("physics_only", 1), ("trajectory_source", "diffusion")])
def test_terminal_rejects_wrong_fresh_recipe(terminal, key, value):
    report, result = terminal
    result["arguments"][key] = value
    with pytest.raises(ValueError):
        TERMINAL.validate_terminal(report, result)


def test_real1000_admission_cannot_launch_future2000():
    report = json.loads((DOCS / "state_action_isolated_checkpoint1000_review_v1/audit.json").read_text())
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(report, report["checkpoints"][0]["checkpoint"])


def test_failed_tracking_sequences_are_still_accepted_full_denominator(tracking):
    tracking["sequences"][0].update(tracking_failed=True, tracking_success=False, sequence_had_fall=True)
    EVIDENCE.validate_protocol(tracking, "tracking", REG)


@pytest.mark.parametrize("fault", ["partial", "test", "duplicate", "filtered", "float_count", "retuned", "extra_cli"])
def test_tracking_requires_all22_and_original_protocol(tracking, fault):
    if fault == "partial":
        tracking["sequences"].pop()
    elif fault == "test":
        tracking["arguments"]["split"] = "test"
    elif fault == "duplicate":
        tracking["sequences"][1]["motion_id"] = tracking["sequences"][0]["motion_id"]
    elif fault == "filtered":
        tracking["sequences"][0]["in_full_sequence_success_denominator"] = False
    elif fault == "float_count":
        tracking["aggregate"]["observed_metric_steps"] = 11982.
    elif fault == "retuned":
        tracking["arguments"]["failure_threshold_m"] = .6
    else:
        tracking["arguments"]["native_dynbench"] = True
    with pytest.raises(ValueError):
        EVIDENCE.validate_protocol(tracking, "tracking", REG)


def test_exact_registered_navigation_metadata(navigation):
    EVIDENCE.validate_protocol(navigation, "navigation", REG)


def test_navigation_cannot_lower_existing_physical_fall_threshold(navigation):
    navigation["physics"]["fall_definition"]["pelvis_ground_height_less_than_m"] = .35
    with pytest.raises(ValueError):
        EVIDENCE.validate_protocol(navigation, "navigation", REG)


@pytest.mark.parametrize("key,value", [("evaluate_steps", 900), ("num_envs", True),
    ("guidance_weight", 40.), ("near_horizon_waypoint_mass", .25), ("pairwise_contacts", False),
    ("measure_ground_support", False), ("power_source", "legacy_pd_proxy"), ("seed", 8)])
def test_navigation_rejects_other_windows_or_sources(navigation, key, value):
    navigation["arguments"][key] = value
    with pytest.raises(ValueError):
        EVIDENCE.validate_protocol(navigation, "navigation", REG)


@pytest.fixture
def ledger_fixture(navigation):
    # Unit-only hypothetical event metadata; never serialized as actual proof.
    declared = {row["path"]: row for row in json.loads((EVIDENCE.BUNDLE / "source_manifest_v3.json").read_text())["files"]}
    path = EVIDENCE.CODE / "scripts/train_paper_amp.py"
    expected = declared["code/scripts/train_paper_amp.py"]
    observed = {"module": "scripts.train_paper_amp.__main__", "path": str(path),
        "bytes": expected["bytes"], "sha256": expected["sha256"], "execution_observed": True}
    checkpoint = {"path": REG["future_checkpoint"]["path"], "bytes": 1, "sha256": "a" * 64}
    admission = {"path": "/tmp/unit-only-no-checkpoint-admission", "bytes": 1, "sha256": "b" * 64}
    _, argv = EXECUTOR.build_command("navigation", checkpoint["path"], checkpoint["sha256"],
                                     navigation["arguments"]["output"], REG)
    entry = {key: observed[key] for key in ("path", "bytes", "sha256")}
    events = [{"sequence": 1, "event": "start"}, {"sequence": 2, "event": "registered_evaluation_admitted",
        "protocol": "navigation", "adapter": EVIDENCE.record(DOCS / "run_isolated_state_action_evaluation_v1.py"),
        "registration": EVIDENCE.record(DOCS / "state_action_evaluation_registration_v1/registration.json"),
        "payload_admission": admission, "checkpoint": checkpoint, "entry": entry, "runner_arguments": argv},
        {"sequence": 3, "event": "registered_entry_selected", "path": str(path), "sha256": observed["sha256"], "argv": argv},
        {"sequence": 4, "event": "owned_execution", **observed}]
    ledger = {"schema": "paper_amp.actual_owned_import_ledger.v3", "source_manifest_sha256": EVIDENCE.MANIFEST_SHA,
        "launcher_sha256": EVIDENCE.LAUNCHER_SHA, "code": str(EVIDENCE.CODE), "all_observed_owned_modules_isolated": True,
        "event_count": 4, "owned_modules": [observed], "runtime_finalization_observed": False}
    return ledger, events, declared, admission, checkpoint, navigation["arguments"]


def test_ledger_keeps_actual_missing_finalization_explicit(ledger_fixture):
    ledger, events, declared, admission, checkpoint, arguments = ledger_fixture
    observed = EVIDENCE.validate_observed_ledger(ledger, events, declared, "navigation", REG, admission, checkpoint, arguments)
    assert observed["runtime_finalization_observed"] is False
    assert observed["unobserved_finalization_or_imports_inferred"] is False


@pytest.mark.parametrize("fault", ["training_ledger", "outside_copy", "wrong_sha", "lost_event", "changed_checkpoint", "wrong_argv"])
def test_actual_entry_source_ledger_cannot_be_substituted(ledger_fixture, fault):
    ledger, events, declared, admission, checkpoint, arguments = ledger_fixture
    if fault == "training_ledger":
        events[1]["event"] = "runner_entry_selected"
    elif fault == "outside_copy":
        ledger["owned_modules"][0]["path"] = str(ROOT / "scripts/train_paper_amp.py")
    elif fault == "wrong_sha":
        ledger["owned_modules"][0]["sha256"] = "0" * 64
    elif fault == "lost_event":
        events.pop()
    elif fault == "changed_checkpoint":
        events[1]["checkpoint"] = {**checkpoint, "sha256": "0" * 64}
    else:
        events[2]["argv"] = ["--evaluate-steps", "900"]
    with pytest.raises(ValueError):
        EVIDENCE.validate_observed_ledger(ledger, events, declared, "navigation", REG, admission, checkpoint, arguments)


def test_metadata_tool_imports_are_cold_without_torch_or_sdk():
    paths = [DOCS / name for name in ("state_action_evaluation_evidence_v1.py", "audit_state_action_terminal2000_v1.py",
        "compare_state_action_full22_style_v1.py", "audit_state_action_navigation200s_v1.py")]
    code = "import importlib.util,sys\n"
    for index, path in enumerate(paths):
        code += f"s=importlib.util.spec_from_file_location('cold{index}',{str(path)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
    code += "assert not any(k.split('.')[0] in {'torch','numpy','omni','isaacsim','isaaclab','pxr'} for k in sys.modules)\n"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
