"""Small CPU contract checks; fixtures never establish payload or model evidence."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "state_prior_evaluation_registration_v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXECUTOR = load("pacer_reward_evaluation_executor_test", HERE / "run_registered_evaluation.py")
EVIDENCE = load("pacer_reward_evaluation_evidence_test", HERE / "evaluation_evidence.py")
REGISTRATION_PATH = HERE / "registration.json"
if not REGISTRATION_PATH.exists():
    REGISTRATION_PATH = HERE / "registration_draft.json"
REGISTRATION = json.loads(REGISTRATION_PATH.read_text())
CHECKPOINT = {"path": REGISTRATION["future_checkpoint"]["path"], "bytes": 123, "sha256": "a" * 64}


def admitted_fixture():
    # Reuse the old complete provenance fixture, never invoke its tests/readers.
    tree = ast.parse((OLD / "test_registration.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "admitted_fixture")
    scope = {"REGISTRATION": REGISTRATION, "CHECKPOINT": CHECKPOINT, "deepcopy": deepcopy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_contract_fixture", "exec"), scope)
    document = scope["admitted_fixture"]()
    document["schema"] = "pacer_reward.actual_training_checkpoint_audit.v1"
    document["copied_files_hashed"] = 935
    document["reward_contract"] = deepcopy(REGISTRATION["future_checkpoint"]["required_reward_contract"])
    return document


def test_complete_new_contract_only_in_memory():
    row = EXECUTOR.validate_admission(admitted_fixture(), CHECKPOINT, REGISTRATION)
    assert row["updates"] == 2000 and row["transitions"] == 16384000


@pytest.mark.parametrize("key,value", [
    ("amp_reward_scale", 2), ("amp_reward_weight", 2.0),
    ("task_reward_weight", 1.0), ("direction_mode", "legacy"),
    ("amp_probability_epsilon", 0.001),
])
def test_reward_contract_drift_or_type_rejected(key, value):
    document = admitted_fixture()
    document["reward_contract"][key] = value
    with pytest.raises(ValueError, match="reward contract"):
        EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)


@pytest.mark.parametrize("key,value", [
    ("schema", "public_state_prior.actual_training_checkpoint_audit.v1"),
    ("copied_files_hashed", 934), ("copied_files_hashed", 935.0),
    ("phase", "milestone"), ("checks", []), ("check_count", True),
    ("source_manifest", {"sha256": "old934"}),
    ("auditor", {"sha256": "unfrozen_reader"}),
])
def test_old_partial_or_wrong_source_admission_rejected(key, value):
    document = admitted_fixture()
    document[key] = value
    with pytest.raises(ValueError):
        EXECUTOR.validate_admission(document, CHECKPOINT, REGISTRATION)


@pytest.mark.parametrize("protocol", ["tracking", "navigation"])
def test_real_new_argv_matches_original_math_and_complete_protocol(protocol):
    output = REGISTRATION[protocol]["parsed"]["output"]
    entry, argv = EXECUTOR.build_command(protocol, CHECKPOINT["path"], CHECKPOINT["sha256"], output, REGISTRATION)
    assert entry.is_relative_to(EXECUTOR.CODE)
    old = json.loads((OLD / "registration.json").read_text())
    allowed = {"checkpoint", "checkpoint_sha256", "output"} if protocol == "tracking" else {"config", "resume", "output", "scene_config"}
    assert {key: value for key, value in REGISTRATION[protocol]["parsed"].items() if key not in allowed} == {
        key: value for key, value in old[protocol]["parsed"].items() if key not in allowed}
    if protocol == "tracking":
        assert not {"--max-clips", "--max-steps", "--motion-ids"}.intersection(argv)
        assert REGISTRATION[protocol]["all_validation_clips"] == 22
        assert REGISTRATION[protocol]["expected_complete_metric_frames_from_same_fixture"] == 11982
    else:
        assert argv[argv.index("--evaluate-steps") + 1] == "6000"
        assert argv[argv.index("--num-envs") + 1] == "4"
        assert "--pairwise-contacts" in argv and "--measure-ground-support" in argv
        assert REGISTRATION[protocol]["duration_s"] == 200


def test_command_alias_and_SHA_placeholder_rejected():
    for checkpoint, digest in [("/tmp/last.pt", CHECKPOINT["sha256"]),
                               (CHECKPOINT["path"], "ACTUAL_SEALED_STEP2000_SHA256")]:
        with pytest.raises(ValueError):
            EXECUTOR.build_command("tracking", checkpoint, digest, REGISTRATION["tracking"]["parsed"]["output"], REGISTRATION)


def test_metric_math_and_command_builder_AST_unchanged():
    for filename, name in [("run_registered_evaluation.py", "build_command"),
                           ("evaluation_evidence.py", "validate_protocol")]:
        old = next(node for node in ast.parse((OLD / filename).read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in ast.parse((HERE / filename).read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == name)
        for node in ast.walk(old):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace("new934", "new935")
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    old_registration = json.loads((OLD / "registration.json").read_text())
    for key in ("metrics", "pose45", "acceptance"):
        assert REGISTRATION[key] == old_registration[key]
    assert REGISTRATION["sources"]["frozen_style_reference"]["sha256"].startswith("c274348f")
    assert REGISTRATION["sources"]["pose45_basis"]["sha256"].startswith("b25dffa6")


def test_actual_parser_ledger_is_new935_SDK_free():
    ledger = json.loads((HERE / "configuration_only_imports.json").read_text())
    assert ledger["status"] == "finalized" and ledger["phase"] == "passed"
    assert ledger["runtime_finalization_observed"] is True
    assert ledger["source_manifest_sha256"] == REGISTRATION["sources"]["isolated_source_manifest"]["sha256"]
    assert ledger["owned_modules"] and all(Path(row["path"]).is_relative_to(EXECUTOR.CODE) for row in ledger["owned_modules"])
    assert not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)


def test_actual_frozen_reader_binding_if_finalized():
    if REGISTRATION_PATH.name == "registration_draft.json":
        assert REGISTRATION["pending"]["independent_reader_source_freeze"] is True
        assert REGISTRATION["sources"]["state_prior_auditor_source_freeze"] is None
        assert EXECUTOR.REGISTRATION_SHA == "UNFROZEN_REGISTRATION_SHA256"
        return
    assert EXECUTOR.record(REGISTRATION_PATH)["sha256"] == EXECUTOR.REGISTRATION_SHA
    EXECUTOR.validate_registered_dependencies(REGISTRATION)
    changed = deepcopy(REGISTRATION)
    changed["sources"]["state_prior_auditor_source_freeze"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        EXECUTOR.validate_registered_dependencies(changed)
