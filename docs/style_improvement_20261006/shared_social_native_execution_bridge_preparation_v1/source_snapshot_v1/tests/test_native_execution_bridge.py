"""CPU command/exit contracts only; no SDK exits or completion proofs invented."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from CrowdSim.humanoid.application import native_execution_bridge as bridge


def declared_packet(command, cuda="0"):
    # In-memory shape test, never published as an actual tool observation.
    return {"schema": bridge.OBSERVATION_SCHEMA, "cmd": bridge.tool_command(command, cuda=cuda),
        "workdir": str(bridge.ROOT), "actual_terminal_response_observed": True,
        "response": {"exit_code": 0, "chunk_id": "IN_MEMORY_CPU_CONTRACT_ONLY"}}


@pytest.mark.parametrize("case_id,actors,seed", (("Office_N03_seed19", 3, 19), ("Warehouse_N15_seed42", 15, 42)))
def test_fixed_native_case_plan_uses_same_seed_and_candidate_without_creating_outputs(case_id, actors, seed):
    plan = bridge.prepare_case(case_id)
    assert plan["case"]["actors"] == actors and plan["case"]["seed"] == seed
    assert [(row["controls"], row["planner"]) for row in plan["stages"]] == \
        [(30, "diffusion"), (30, "sfm"), (6000, "diffusion"), (6000, "sfm")]
    for stage in plan["stages"]:
        args = stage["SDK_command"]
        assert args[args.index("--seed") + 1] == str(seed)
        assert args[args.index("--execution-receipt") + 1] == str(bridge.TRAINING_RECEIPT)
        assert not Path(stage["output"]).exists() and not Path(stage["SDK_receipt"]).exists()
        if stage["controls"] == 6000:
            original = next(row for row in plan["case"]["arms"] if row["planner"] == stage["planner"])
            assert stage["output"] == original["output"]
        assert not stage["CPU_or_SDK_exit_observed"]
    assert [row["controls"] for row in plan["initial_comparisons"]] == [30, 6000]
    assert plan["style_command"][plan["style_command"].index("--seed") + 1] == str(seed)
    assert plan["both_live_initial_pairings_required"] and plan["readonly_raw_array_seal_required_before_style"]
    assert not plan["actual_execution_admitted"] and not plan["GPU_or_SDK_started"]


@pytest.mark.parametrize("case_id", ("Hospital_N12_seed7", "Hospital_N13_seed19", "Hospital_N14_seed42", "Hospital_N15_seed7"))
def test_original_hospital_rejections_cannot_be_turned_into_commands(case_id):
    with pytest.raises(ValueError, match="Retain native resource rejection"):
        bridge.prepare_case(case_id)


@pytest.mark.parametrize("defect", ("command", "cwd", "not_observed", "missing_chunk", "pending", "bool", "float", "old_schema"))
def test_only_the_exact_command_and_actual_integer_terminal_exit_are_admitted(defect):
    command = ["CPU_VECTOR_ONLY", "--seed", "19"]
    packet = declared_packet(command)
    if defect == "command":
        packet["cmd"] = bridge.tool_command(["CPU_VECTOR_ONLY", "--seed", "42"], cuda="0")
    elif defect == "cwd":
        packet["workdir"] = "/WRONG_CPU_FIXTURE"
    elif defect == "not_observed":
        packet["actual_terminal_response_observed"] = False
    elif defect == "missing_chunk":
        packet["response"]["chunk_id"] = ""
    elif defect == "pending":
        packet["response"] = {"session_id": 12345, "exit_code": None}
    elif defect == "bool":
        packet["response"]["exit_code"] = False
    elif defect == "float":
        packet["response"]["exit_code"] = 0.
    else:
        packet["schema"] = "CPU_FIXTURE_OLD_SCHEMA"
    with pytest.raises(ValueError, match="actual terminal tools API"):
        bridge.validate_tool_observation(packet, command=command, cuda="0")


@pytest.mark.parametrize("code", (0, 1, -9))
def test_declared_integer_failures_are_preserved_and_never_rewritten_zero(code):
    packet = declared_packet(["CPU_VECTOR_ONLY"])
    packet["response"]["exit_code"] = code
    before = copy.deepcopy(packet)
    selected = bridge.validate_tool_observation(packet, command=["CPU_VECTOR_ONLY"], cuda="0")
    assert selected["actual_exit_code"] == code and packet == before


@pytest.mark.parametrize("packet", (None, [], {}, {"response": None}, {"response": []}))
def test_incomplete_tool_observation_has_an_explicit_type_boundary(packet):
    with pytest.raises(ValueError, match="mapping with a response"):
        bridge.validate_tool_observation(packet, command=["CPU_ONLY"], cuda="0")


def test_tool_command_quotes_spaces_apostrophes_newlines_and_shell_substitutions_literally():
    args = ["CPU_VECTOR_ONLY", "name with spaces", "one'two", "literal\nline", "$(MUST_NOT_EXECUTE)", "`MUST_NOT_EXECUTE`"]
    parsed = shlex.split(bridge.tool_command(args, cuda=""))
    assert parsed[-len(args):] == args and parsed[1] == "CUDA_VISIBLE_DEVICES="


def completion_fixture(tmp_path, *, stage="waiting_for_actual_training_exit", observed=False):
    source = tmp_path / "terminal_tool_owner.js"
    source.write_bytes((bridge.S3 / "terminal_tool_owner.js").read_bytes())
    source.chmod(0o444)
    state = tmp_path / "CPU_FIXTURE_owner_state.json"
    state.write_text(json.dumps({"stage": stage, "training_exit_observed": observed,
                               "training_terminal": {"exit_code": 0}}))
    state.chmod(0o444)
    certificate = {"schema": "shared_social.external_S3_owner_completion.v1", "completed_exec_cell_id": "581",
        "completion_observed_by_root": True, "owner_source": bridge.record(source),
        "terminal_owner_state": bridge.record(state)}
    return certificate


def test_declared_completion_cannot_override_an_incomplete_owner_or_create_receipts(monkeypatch, tmp_path):
    certificate = completion_fixture(tmp_path)
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_completion.json"
    path.write_text(json.dumps(certificate))
    path.chmod(0o444)
    monkeypatch.setattr(bridge, "S3", tmp_path)
    with pytest.raises(ValueError, match="serial post-evaluation owner are not complete"):
        bridge.publish_sdk_exit("Office_N03_seed19", planner="diffusion", controls=30,
                                observation_path=tmp_path / "MISSING", owner_completion=path)
    assert not (tmp_path / "observations").exists()


def test_an_actual_completed_S1_checkpoint_cannot_fill_the_S3_slot(monkeypatch, tmp_path):
    certificate = completion_fixture(tmp_path, stage="completed_measurements_quality_unclassified", observed=True)
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_completion.json"
    path.write_text(json.dumps(certificate))
    path.chmod(0o444)
    monkeypatch.setattr(bridge, "S3", tmp_path)
    monkeypatch.setattr(bridge, "TRAINING_RECEIPT", bridge.ROOT /
        "docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json")
    with pytest.raises(ValueError, match="preselected S3 terminal candidate"):
        bridge.admit_S3_owner_completion(path)


@pytest.mark.parametrize("defect", ("cell", "root_observed", "source", "schema"))
def test_unobserved_or_other_owner_completion_is_refused_before_any_training_resource(defect, monkeypatch, tmp_path):
    certificate = completion_fixture(tmp_path)
    if defect == "cell":
        certificate["completed_exec_cell_id"] = "OTHER"
    elif defect == "root_observed":
        certificate["completion_observed_by_root"] = False
    elif defect == "source":
        certificate["owner_source"]["sha256"] = "0" * 64
    else:
        certificate["schema"] = "CPU_OLD_SCHEMA"
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_completion.json"
    path.write_text(json.dumps(certificate))
    path.chmod(0o444)
    monkeypatch.setattr(bridge, "S3", tmp_path)
    with pytest.raises(ValueError, match="externally observed S3 owner completion"):
        bridge.admit_S3_owner_completion(path)


def test_source_preflight_CLI_is_fresh_CPU_and_creates_no_simulation_outputs():
    child = subprocess.run([sys.executable, str(bridge.ROOT / "scripts/prepare_shared_social_native_execution.py"),
        "--case-id", "Office_N03_seed19", "--mode", "source-preflight"], cwd=bridge.ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    plan = json.loads(child.stdout)
    assert plan["case"]["seed"] == 19 and not plan["GPU_or_SDK_started"] and not plan["actual_execution_admitted"]
