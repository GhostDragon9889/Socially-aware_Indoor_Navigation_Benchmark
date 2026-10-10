"""Thin registered samples10 identity adapter; reuse the frozen physical guards."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "docs/style_improvement_20261003/default_newDP_physical_reader_revision_v2"
BASE_DIR = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1"
sys.path.insert(0, str(PARENT))
import physical_evidence_v2 as V2  # noqa: E402

BASE = V2.BASE
read, record, require, load = BASE.read, BASE.record, BASE.require, BASE.load
CPU_ENV = BASE.CPU_ENV
# Make the parent's observed deterministic setting explicit, not a new setting.
GPU_ENV = {**BASE.GPU_ENV, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def registration():
    frozen = read(HERE / "source_freeze_v1.json")
    require(frozen["status"] == "frozen_prepared_only", "Frozen samples10 sources required")
    for item in frozen["files"]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222,
                "Samples10 own source changed")
    contract = read(HERE / "registration.json")
    for role, item in contract["frozen_inputs"].items():
        require(record(item["path"]) == item, "Samples10 parent/prepared input changed")
        mode = Path(item["path"]).stat().st_mode
        if role == "original_raw_math":
            require(mode & 0o777 == 0o664, "Preserve original raw source0664; exact bytes required")
        else:
            require(not mode & 0o222, "Frozen parent/prepared input must remain readonly")
    prepared = read(contract["frozen_inputs"]["samples10_prepared_freeze"]["path"])
    for item in prepared["files"]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222,
                "Prepared samples10 packet changed")
    parent = BASE.registration()
    parameters = read(contract["frozen_inputs"]["samples10_parameters"]["path"])
    preflight = read(contract["frozen_inputs"]["samples10_actual_preflight"]["path"])
    old_case = parent["cases"]["fixture_new"]
    expected = copy.deepcopy(old_case["preflight"])
    expected["driver_argv"] = parameters["forwarded_arguments"]
    expected["parsed_driver_arguments"] = parameters["parsed_driver_arguments"]
    expected["deployment_contract"]["independent_samples10_extension"] = preflight["deployment_contract"]["independent_samples10_extension"]
    require(preflight == expected, "Other matched four-person200s preflight fields changed")
    old_parsed, new_parsed = old_case["preflight"]["parsed_driver_arguments"], parameters["parsed_driver_arguments"]
    require({key for key in old_parsed if old_parsed[key] != new_parsed[key]} == {"output", "diffusion_samples"}
            and old_parsed["diffusion_samples"] == 1 and new_parsed["diffusion_samples"] == 10,
            "Only samples1 to10 plus output identity may differ")
    case = copy.deepcopy(old_case)
    case.update(label="new_samples10", output=parameters["output"],
                command=parameters["root_GPU_command_after_q_training_and_evaluation_only"], preflight=preflight,
                audit=contract["actual_paths"]["audit"], GPU_receipt=contract["actual_paths"]["GPU_receipt"],
                CPU_receipt=contract["actual_paths"]["CPU_receipt"], CPU_command=contract["CPU_command"],
                prior_physical_gate="Matched parent NEW complete200s closure bound; its failures remain included")
    plan = copy.deepcopy(parent)
    plan["cases"] = {"fixture_new": case}
    return plan


def source_binding(plan):
    require(plan == registration(), "Only the exact registered samples10 virtual plan is allowed")
    binding = BASE.source_binding(plan)  # Exact original185, model/data/source/DP-training checks.
    contract = read(HERE / "registration.json")
    parent = BASE.registration()
    old_case = parent["cases"]["fixture_new"]
    gpu = read(contract["frozen_inputs"]["matched_parent_GPU_owner"]["path"])
    cpu = read(contract["frozen_inputs"]["matched_parent_CPU_owner"]["path"])
    require(gpu["schema"] == "default_newDP.actual_physical_launch.v1" and gpu["phase"] == "fixture_new"
            and gpu["command"] == old_case["command"] and gpu["binding_before"] == gpu["binding_after"] == binding,
            "Matched parent NEW source/process identity differs")
    require(cpu["schema"] == "default_newDP.actual_CPU_audit_launch.v2"
            and cpu["audit_artifact"] == contract["frozen_inputs"]["matched_parent_raw_audit"]
            and cpu["binding_before"] == cpu["binding_after"] == binding,
            "Matched parent NEW actual CPU closure differs")
    for owner in (gpu, cpu):
        require(owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
                and owner["child_exit_observed"] is True and owner["driver_completed"] is True,
                "Actually closed complete parent process required; no quality filter")
    require(Path(plan["controller"]["path"]).stat().st_mode & 0o777 == 0o664,
            "Legacy default controller original0664 mode must be preserved")
    return {**binding, "samples10_extension": {
        "source_freeze": record(HERE / "source_freeze_v1.json"), "registration": record(HERE / "registration.json"),
        "prepared_source_freeze": contract["frozen_inputs"]["samples10_prepared_freeze"],
        "entry": contract["frozen_inputs"]["samples10_entry"], "parameters": contract["frozen_inputs"]["samples10_parameters"],
        "matched_parent_GPU_owner": contract["frozen_inputs"]["matched_parent_GPU_owner"],
        "matched_parent_CPU_owner": contract["frozen_inputs"]["matched_parent_CPU_owner"],
        "only_method_variable": {"diffusion_samples": {"before": 1, "after": 10}},
        "complete_runtime_import_ledger": False}}


def metadata_sources():
    """Pinned parent functions, modified only at declared metadata boundaries."""
    runtime = inspect.getsource(V2.validate_runtime_planner)
    before = 'runtime["diffusion_samples_per_actor"] == 1'
    after = 'runtime["diffusion_samples_per_actor"] == 10'
    require(runtime.count(before) == 1, "One original samples runtime guard required")
    runtime_revised = runtime.replace(before, after)
    admission = inspect.getsource(BASE.admit_capture)
    substitutions = (
        (V2.BEFORE, '    runtime_planner_identity = validate_runtime_planner(result, plan, phase)\n'),
        ('"default_newDP.actual_physical_launch.v1"', '"default_NEW.samples10.actual_physical_launch.v1"'),
        ('return {"schema": "default_newDP.actual_capture_source_admission.v1",',
         'return {"schema": "default_NEW.samples10.actual_capture_source_admission.v1", "runtime_planner_identity": runtime_planner_identity,'),
    )
    revised = admission
    for old, new in substitutions:
        require(revised.count(old) == 1, "One original admission metadata boundary required")
        revised = revised.replace(old, new)
    reverse = revised
    for old, new in reversed(substitutions):
        reverse = reverse.replace(new, old)
    require(reverse == admission and runtime_revised.replace(after, before) == runtime,
            "Only exact metadata substitutions may change parent functions")
    return runtime_revised, revised


def admit_capture(plan, phase):
    require(phase == "fixture_new", "Only registered four-person samples10 fixture")
    runtime, admission = metadata_sources()
    namespace = {**V2.__dict__, **BASE.__dict__, "HERE": HERE, "source_binding": source_binding}
    exec(compile(runtime, str(HERE / "samples10_evidence.py"), "exec"), namespace)
    exec(compile(admission, str(HERE / "samples10_evidence.py"), "exec"), namespace)
    result = namespace["admit_capture"](plan, phase)
    owner = read(plan["cases"][phase]["GPU_receipt"])
    require(owner["registration"] == record(HERE / "registration.json"), "Actual new owner registration differs")
    result["metadata_helper"] = record(__file__)
    return result
