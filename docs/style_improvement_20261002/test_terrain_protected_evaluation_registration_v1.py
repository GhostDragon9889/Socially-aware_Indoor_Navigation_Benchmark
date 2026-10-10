"""Fail-closed admission of pending evaluation; never load a model or SDK."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "tested_registered_state_action_executor",
    HERE / "run_isolated_terrain_protected_evaluation_v1.py",
)
EXECUTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTOR)
REGISTRATION = json.loads(
    (
        HERE / "terrain_protected_evaluation_registration_v1/registration.json"
    ).read_text()
)
CHECKPOINT = {
    "path": REGISTRATION["future_checkpoint"]["path"],
    "bytes": 123,
    "sha256": "a" * 64,
}


def admitted_fixture():
    # Mathematical admission fixture only; not a fake model/payload artifact.
    return {
        "schema": "protected_terrain_state_action.actual_training_checkpoint_audit.v1",
        "status": "passed",
        "CPU_only": True,
        "GPU_used": False,
        "SDK_started": False,
        "sealed34_outputs_read": False,
        "checks": [{"passed": True}],
        "actual_observed_owned_imports_isolated": True,
        "protected_reference_policy_validated": True,
        "actual_terminal_fresh2000_confirmed": True,
        "source_manifest": REGISTRATION["sources"]["isolated_source_manifest"],
        "checkpoints": [
            {
                "updates": 2000,
                "transitions": 16384000,
                "D_input": 2019,
                "checkpoint": deepcopy(CHECKPOINT),
                "full_payload_resume_validated": True,
                "public_D_or_AMP_RMS_transferred": False,
                "protected_policy_independent_checks": True,
                "terrain_policy": {
                    "schema": "navisaaclab.task_terrain_statistics_policy.v1",
                    "mode": "preserve_declared_suffix",
                    "bound": True,
                    "task_slice": [20, 1044],
                    "post_update_calls": 2000,
                    "reference_count": 61931520001.0,
                    "warm_start_task_count": 200000.0,
                    "reference_statistics_sha256": "b" * 64,
                    "global_count_and_prefix20_protected": False,
                    "external_statistics_file_required": False,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "defect",
    [
        "state_only",
        "float_updates",
        "float_transitions",
        "float_D",
        "wrong_SHA",
        "missing_source",
        "missing_resume",
        "public_RMS",
        "empty_checks",
        "failed_check",
        "unknown_schema",
        "duplicate",
    ],
)
def test_wrong_or_partial_admission_is_rejected(defect):
    document = admitted_fixture()
    row = document["checkpoints"][0]
    if defect == "state_only":
        row["D_input"] = 1950
    elif defect.startswith("float_"):
        key = {
            "float_updates": "updates",
            "float_transitions": "transitions",
            "float_D": "D_input",
        }[defect]
        row[key] = float(row[key])
    elif defect == "wrong_SHA":
        row["checkpoint"]["sha256"] = "b" * 64
    elif defect == "missing_source":
        document["actual_observed_owned_imports_isolated"] = False
    elif defect == "missing_resume":
        row["full_payload_resume_validated"] = False
    elif defect == "public_RMS":
        row["public_D_or_AMP_RMS_transferred"] = True
    elif defect == "empty_checks":
        document["checks"] = []
    elif defect == "failed_check":
        document["checks"] = [{"passed": False}]
    elif defect == "unknown_schema":
        document["schema"] = "unknown"
    elif defect == "duplicate":
        document["checkpoints"].append(deepcopy(row))
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(document, CHECKPOINT)


def test_complete_contract_accepts_without_importing_trainer():
    row = EXECUTOR.validate_admission(admitted_fixture(), CHECKPOINT)
    assert row["D_input"] == 2019


@pytest.mark.parametrize("protocol", ["tracking", "navigation"])
def test_fixed_template_keeps_complete_protocol_and_sealed_path(protocol, tmp_path):
    entry, argv = EXECUTOR.build_command(
        protocol, CHECKPOINT["path"], CHECKPOINT["sha256"], tmp_path, REGISTRATION
    )
    assert entry.is_relative_to(EXECUTOR.CODE)
    assert argv[argv.index("--output") + 1] == str(tmp_path.resolve())
    if protocol == "tracking":
        assert (
            "--max-clips" not in argv
            and "--max-steps" not in argv
            and "--motion-ids" not in argv
        )
        assert argv[argv.index("--split") + 1] == "validation"
        assert argv[argv.index("--checkpoint-sha256") + 1] == CHECKPOINT["sha256"]
    else:
        assert argv[argv.index("--evaluate-steps") + 1] == "6000"
        assert argv[argv.index("--guidance-weight") + 1] == "20.0"
        assert "--pairwise-contacts" in argv and "--measure-ground-support" in argv


@pytest.mark.parametrize("defect", ["wrong_path", "wrong_SHA", "unknown_protocol"])
def test_unregistered_entry_or_checkpoint_rejected(defect, tmp_path):
    protocol, path, digest = "tracking", CHECKPOINT["path"], CHECKPOINT["sha256"]
    if defect == "wrong_path":
        path = str(tmp_path / "step_500.pt")
    elif defect == "wrong_SHA":
        digest = "ACTUAL_SEALED_SHA256"
    else:
        protocol = "test34"
    with pytest.raises(ValueError):
        EXECUTOR.build_command(protocol, path, digest, tmp_path, REGISTRATION)


@pytest.mark.parametrize(
    "defect",
    [
        "legacy_schema",
        "legacy_policy",
        "wrong_slice",
        "counter",
        "reference",
        "warm_count",
        "missing_independent_checks",
        "top_policy",
        "nonterminal",
        "source_manifest",
    ],
)
def test_old_or_wrong_terrain_admission_cannot_launch_protected(defect):
    document = admitted_fixture()
    policy = document["checkpoints"][0]["terrain_policy"]
    if defect == "legacy_schema":
        document["schema"] = "state_action.actual_training_checkpoint_audit.v1"
    elif defect == "legacy_policy":
        policy["mode"] = "legacy"
    elif defect == "wrong_slice":
        policy["task_slice"] = [0, 1044]
    elif defect == "counter":
        policy["post_update_calls"] = 1999
    elif defect == "reference":
        policy["reference_count"] = 200000.0
    elif defect == "warm_count":
        policy["warm_start_task_count"] = 61931520001.0
    elif defect == "missing_independent_checks":
        document["checkpoints"][0]["protected_policy_independent_checks"] = False
    elif defect == "top_policy":
        document["protected_reference_policy_validated"] = False
    elif defect == "nonterminal":
        document["actual_terminal_fresh2000_confirmed"] = False
    else:
        document["source_manifest"] = {"sha256": "c" * 64}
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)


def test_registration_sha_and_fullmodel_old_adapter_are_explicit():
    import hashlib

    assert (
        hashlib.sha256(EXECUTOR.REGISTRATION.read_bytes()).hexdigest()
        == EXECUTOR.REGISTRATION_SHA
    )
    payload = (EXECUTOR.CODE / "CrowdSim/humanoid/runtime/amp_inference.py").read_text()
    assert "self.model = PaperAMPNetwork" in payload and "self.model.policy" in payload
    assert "FrozenAMPActor" not in payload
    assert (
        REGISTRATION["matched_protocol_and_source"]["full_model_adapter_both_variants"]
        is True
    )
    assert REGISTRATION["pending"]["future_full_payload_admitted"] is False
