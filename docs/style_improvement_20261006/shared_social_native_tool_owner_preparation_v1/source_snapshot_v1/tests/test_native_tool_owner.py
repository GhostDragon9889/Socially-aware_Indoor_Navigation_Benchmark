"""CPU-only registration/sealing contracts. Fixtures are never native evidence."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_shared_social_native_owner as owner


def test_serial_plan_retains_fixed_stage_order_and_exact_commands(tmp_path):
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_registration.json"
    plan = owner.build_plan("Warehouse_N15_seed42", path)
    actions = plan["actions"]
    assert len(actions) == 21
    sdk = [row for row in actions if row["requires_host_GPU"]]
    assert [row["role"] for row in sdk] == ["diffusion_C30_SDK", "sfm_C30_SDK", "diffusion_C6000_SDK", "sfm_C6000_SDK"]
    assert actions[8]["role"] == "initial_pair_C30" and actions[17]["role"] == "initial_pair_C6000"
    assert [row["role"] for row in actions[-3:]] == ["seal_captures", "style_200s", "collect_measurements"]
    assert plan["actors"] == 15 and plan["seed"] == 42 and not path.exists()
    for row in actions:
        assert row["cmd"] == owner.bridge.tool_command(row["argv"], cuda=row["cuda"])
        if row["failure_publication"]:
            index = actions.index(row)
            assert row["failure_publication"] == actions[index + 1]
            assert row["observation"] in actions[index + 1]["argv"]


@pytest.mark.parametrize("case", ("Hospital_N12_seed7", "Hospital_N13_seed19", "Hospital_N14_seed42", "Hospital_N15_seed7"))
def test_rejected_hospital_cases_cannot_become_serial_plans(case, tmp_path):
    with pytest.raises(ValueError, match="resource rejection"):
        owner.build_plan(case, tmp_path / "CPU_FIXTURE_registration.json")


def test_registered_commands_are_cold_reproduced_and_tampering_rejected(tmp_path):
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_registration.json"
    plan = owner.build_plan("Office_N03_seed19", path)
    owner.bridge._write_new(path, plan)
    assert owner.admitted_plan(path)[0] == plan
    path.chmod(0o644)
    plan["actions"][0]["cmd"] = "CPU_FIXTURE_CHANGED_COMMAND"
    path.write_text(json.dumps(plan))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="Registered commands"):
        owner.admitted_plan(path)


def test_execution_preflight_refuses_missing_S3_completion_without_creating_native_outputs(tmp_path):
    path = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_registration.json"
    plan = owner.build_plan("Office_N03_seed19", path)
    owner.bridge._write_new(path, plan)
    assert not Path(plan["observation_directory"]).exists()
    with pytest.raises(FileNotFoundError):
        owner.main(["--mode", "execution-preflight", "--registration", str(path)])
    assert not Path(plan["observation_directory"]).exists()


def capture_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "CPU_FIXTURE_NOT_ACTUAL_SDK"
    directory.mkdir()
    names = ("result.json", "physical_states.npz", "ground_support.npz", "reference_plans.npz",
             "paired_initial_condition.json", "paired_initial_arrays.npz")
    for name in names:
        (directory / name).write_bytes(b"CPU_FIXTURE_NOT_ACTUAL_CAPTURE\n")
    rows = [owner.bridge.record(directory / name) for name in names]
    selected = SimpleNamespace(receipt={"result": rows[0]}, arrays=rows[1:4],
        result={"paired_initial_condition": {"document": rows[4]}}, initial={"arrays": rows[5]})
    monkeypatch.setattr(owner.bridge.raw, "admit_execution", lambda *args, **kwargs: selected)
    plan = {"actors": 3, "seed": 19, "scene_configuration": {"path": "CPU_FIXTURE", "sha256": "0" * 64},
        "case_plan": {"stages": [{"output": str(directory), "SDK_receipt": "CPU_FIXTURE_NOT_ACTUAL",
                                 "controls": 6000, "planner": "diffusion"}]}}
    return directory, rows, plan


def test_capture_seal_preserves_all_bytes_and_binds_initial_arrays(tmp_path, monkeypatch):
    directory, rows, plan = capture_fixture(tmp_path, monkeypatch)
    result = owner.seal_captures(plan)
    assert result["files"] == rows and len(result["files"]) == 6
    assert all(owner.bridge.record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222 for row in rows)
    assert all(path.read_bytes() == b"CPU_FIXTURE_NOT_ACTUAL_CAPTURE\n" for path in directory.iterdir())


@pytest.mark.parametrize("defect", ("changed", "symlink", "hardlink"))
def test_capture_seal_validates_entire_batch_before_any_chmod(defect, tmp_path, monkeypatch):
    directory, rows, plan = capture_fixture(tmp_path, monkeypatch)
    last = Path(rows[-1]["path"])
    if defect == "changed":
        last.write_bytes(b"CPU_FIXTURE_CHANGED")
    elif defect == "symlink":
        last.unlink()
        last.symlink_to(directory / "physical_states.npz")
    else:
        (directory / "CPU_FIXTURE_extra_hardlink").hardlink_to(last)
    with pytest.raises(ValueError):
        owner.seal_captures(plan)
    assert all(Path(row["path"]).stat().st_mode & 0o222 for row in rows[:-1])
