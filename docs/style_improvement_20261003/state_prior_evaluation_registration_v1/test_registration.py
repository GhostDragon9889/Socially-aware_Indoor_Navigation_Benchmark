"""CPU/stdlib contracts only; fixtures are never model or payload evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXECUTOR = load("state_prior_executor_test", HERE / "run_registered_evaluation.py")
EVIDENCE = load("state_prior_evidence_test", HERE / "evaluation_evidence.py")
REGISTRATION = json.loads((HERE / "registration.json").read_text())
CHECKPOINT = {"path": REGISTRATION["future_checkpoint"]["path"], "bytes": 123, "sha256": "a" * 64}


def admitted_fixture():
    # Pure contract data, never written as an actual audit or checkpoint.
    return {
        "schema": "public_state_prior.actual_training_checkpoint_audit.v1", "status": "passed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
        "checks": [{"name": "fixture_only", "passed": True}], "check_count": 1,
        "copied_files_hashed": 934, "actual_observed_owned_imports_isolated": True,
        "protected_reference_policy_validated": True, "actual_terminal_fresh2000_confirmed": True,
        "phase": "terminal2000", "milestone_only_admission": False,
        "state_prior_initialization_validated": True, "public_D_or_AMP_RMS_transferred": True,
        "terminal_training_result": {"fixture_only": True}, "actual_closed_metrics": {"fixture_only": True},
        "source_manifest": deepcopy(REGISTRATION["sources"]["isolated_source_manifest"]),
        "auditor": deepcopy(REGISTRATION["sources"]["state_prior_auditor"]),
        "checkpoints": [{"updates": 2000, "transitions": 16384000, "D_input": 2019,
            "checkpoint": deepcopy(CHECKPOINT), "full_payload_resume_validated": True,
            "public_D_or_AMP_RMS_transferred": True, "state_prior_initialization_validated": True,
            "state_prior": {"explicit_flag": True, "strategy": "public_pacer_state_prior_zero_action_extension",
                "published_state_only_discriminator_transferred": True, "published_amp_history_moments_transferred": True,
                "public_AMP_count": 15482880001.0, "current_AMP_count": 24776000.0, "count_cap": 200000.0,
                "trained_active_columns_nonzero": 57, "frozen_action_columns_zero": 12,
                "initial_action_columns_zero_is_provenance_only": True},
            "RMS_counts": {"amp_normalizer": 24776000.0}, "protected_policy_independent_checks": True,
            "terrain_policy": {"schema": "navisaaclab.task_terrain_statistics_policy.v1", "mode": "preserve_declared_suffix",
                "bound": True, "task_slice": [20, 1044], "post_update_calls": 2000,
                "reference_count": 61931520001.0, "warm_start_task_count": 200000.0,
                "reference_statistics_sha256": "b" * 64, "global_count_and_prefix20_protected": False,
                "external_statistics_file_required": False}}]}


DEFECTS = [
    ("schema", "protected_terrain_state_action.actual_training_checkpoint_audit.v1"),
    ("phase", "milestone"), ("actual_terminal_fresh2000_confirmed", False), ("milestone_only_admission", True),
    ("copied_files_hashed", 931), ("copied_files_hashed", 934.0), ("check_count", 2),
    ("checks", []), ("checks", [{"passed": False}]), ("checks", [True]),
    ("public_D_or_AMP_RMS_transferred", False), ("state_prior_initialization_validated", False),
    ("actual_observed_owned_imports_isolated", False), ("terminal_training_result", None), ("actual_closed_metrics", None),
    ("source_manifest", {"sha256": "old931"}), ("auditor", {"sha256": "oldfalse_reader"}),
    ("checkpoints.0.updates", 2000.0), ("checkpoints.0.transitions", 16384000.0), ("checkpoints.0.D_input", 1950),
    ("checkpoints.0.checkpoint.sha256", "c" * 64), ("checkpoints.0.full_payload_resume_validated", False),
    ("checkpoints.0.public_D_or_AMP_RMS_transferred", False), ("checkpoints.0.state_prior_initialization_validated", False),
    ("checkpoints.0.state_prior.explicit_flag", False), ("checkpoints.0.state_prior.strategy", "native_state_action"),
    ("checkpoints.0.state_prior.published_state_only_discriminator_transferred", False),
    ("checkpoints.0.state_prior.published_amp_history_moments_transferred", False),
    ("checkpoints.0.state_prior.public_AMP_count", 200000.0), ("checkpoints.0.state_prior.current_AMP_count", 24576000.0),
    ("checkpoints.0.state_prior.count_cap", None), ("checkpoints.0.state_prior.trained_active_columns_nonzero", 0),
    ("checkpoints.0.state_prior.frozen_action_columns_zero", 11),
    ("checkpoints.0.state_prior.initial_action_columns_zero_is_provenance_only", False),
    ("checkpoints.0.RMS_counts.amp_normalizer", 16384000.0),
    ("checkpoints.0.protected_policy_independent_checks", False), ("checkpoints.0.terrain_policy.mode", "legacy"),
    ("checkpoints.0.terrain_policy.task_slice", [0, 1044]), ("checkpoints.0.terrain_policy.post_update_calls", 1999),
    ("checkpoints.0.terrain_policy.reference_count", 200000.0),
]


@pytest.mark.parametrize("path,value", DEFECTS)
def test_wrong_partial_or_old_admission_rejected(path, value):
    document = admitted_fixture()
    keys = path.split(".")
    node = document
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[keys[-1]] = value
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)


def test_complete_terminal_contract_and_duplicate_guard():
    document = admitted_fixture()
    assert EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)["D_input"] == 2019
    document["checkpoints"].append(deepcopy(document["checkpoints"][0]))
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)


@pytest.mark.parametrize("protocol", ["tracking", "navigation"])
def test_registered_complete_protocol(protocol):
    output = REGISTRATION[protocol]["parsed"]["output"]
    entry, argv = EXECUTOR.build_command(protocol, CHECKPOINT["path"], CHECKPOINT["sha256"], output, REGISTRATION)
    assert entry.is_relative_to(EXECUTOR.CODE)
    assert argv[argv.index("--output") + 1] == output
    if protocol == "tracking":
        assert not {"--max-clips", "--max-steps", "--motion-ids"}.intersection(argv)
        assert argv[argv.index("--split") + 1] == "validation"
        assert argv[-2:] == ["--checkpoint-sha256", CHECKPOINT["sha256"]]
    else:
        assert argv[argv.index("--evaluate-steps") + 1] == "6000"
        assert argv[argv.index("--guidance-weight") + 1] == "20.0"
        assert "--pairwise-contacts" in argv and "--measure-ground-support" in argv


@pytest.mark.parametrize("defect", ["wrong_path", "SHA_placeholder", "test34", "output_alias"])
def test_command_substitution_rejected(defect, tmp_path):
    protocol, path, digest = "tracking", CHECKPOINT["path"], CHECKPOINT["sha256"]
    output = REGISTRATION[protocol]["parsed"]["output"]
    if defect == "wrong_path":
        path = str(tmp_path / "last.pt")
    elif defect == "SHA_placeholder":
        digest = "ACTUAL_SEALED_STEP2000_SHA256"
    elif defect == "test34":
        protocol = "test34"
    else:
        output = str(tmp_path / "output")
    with pytest.raises(ValueError):
        EXECUTOR.build_command(protocol, path, digest, output, REGISTRATION)


def test_actual_frozen_reader_dependencies_and_tamper_rejected():
    EXECUTOR.validate_registered_dependencies(REGISTRATION)
    changed = deepcopy(REGISTRATION)
    changed["sources"]["state_prior_auditor_source_freeze"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        EXECUTOR.validate_registered_dependencies(changed)


def test_mutable_checkpoint_alias_is_rejected_before_file_read(monkeypatch, tmp_path):
    original = EXECUTOR.record
    alias = tmp_path / "last.pt"

    def no_mutable_read(path):
        assert Path(path).absolute() != alias, "Mutable checkpoint bytes were consumed"
        return original(path)

    monkeypatch.setattr(EXECUTOR, "record", no_mutable_read)
    with pytest.raises(ValueError, match="preregistered immutable step2000"):
        EXECUTOR.main(["--protocol", "tracking", "--checkpoint", str(alias),
            "--checkpoint-sha256", CHECKPOINT["sha256"], "--checkpoint-admission", 
            REGISTRATION["future_checkpoint"]["terminal_admission_path"],
            "--output", REGISTRATION["tracking"]["parsed"]["output"],
            "--import-report", str(EXECUTOR.BUNDLE / "state_prior_tracking2000_v1_GPU_imports.json")])


def test_real_pilot8_is_not_terminal_admission():
    path = EXECUTOR.ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1/actual_pilot8_v1/audit.json"
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(json.loads(path.read_text()), CHECKPOINT, REGISTRATION)


def test_registered_sha_sdk_free_imports_and_no_future_effect():
    assert EXECUTOR.record(HERE / "registration.json")["sha256"] == EXECUTOR.REGISTRATION_SHA
    assert REGISTRATION["future_effect"] is False
    assert all(value is False for value in REGISTRATION["pending"].values())
    assert REGISTRATION["acceptance"]["navigation_minimum_gate"] == {
        "complete_controls": 6000, "falls": 0, "minimum_route_vertices": 21,
        "inclusive_pose_velocity_mmd2_max": 0.07238060315284697}
    ledger = json.loads((HERE / "configuration_only_imports.json").read_text())
    assert ledger["runtime_finalization_observed"] is True and ledger["status"] == "finalized"
    assert ledger["phase"] == "passed"
    assert ledger["source_manifest_sha256"] == REGISTRATION["sources"]["isolated_source_manifest"]["sha256"]
    assert all(Path(row["path"]).is_relative_to(EXECUTOR.CODE) for row in ledger["owned_modules"])
    assert not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)


@pytest.mark.parametrize("defect", ["split", "max_clips", "motion_ids", "output"])
def test_actual_tracking_arguments_must_match_registration(defect):
    # Failure occurs at the CLI gate before any motion-capture fixture is needed.
    result = {"status": "passed", "error": None, "arguments": deepcopy(REGISTRATION["tracking"]["parsed"])}
    result["arguments"]["checkpoint_sha256"] = CHECKPOINT["sha256"]
    result["arguments"][defect] = {"split": "test", "max_clips": 1, "motion_ids": ["other"], "output": "/tmp/unregistered"}[defect]
    with pytest.raises(ValueError):
        EVIDENCE.validate_protocol(result, "tracking", REGISTRATION)
