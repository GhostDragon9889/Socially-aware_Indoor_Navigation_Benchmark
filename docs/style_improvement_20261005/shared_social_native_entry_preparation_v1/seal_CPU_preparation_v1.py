"""Bind observed CPU exits, archive exact sources, and seal preparation evidence."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SFM = ROOT / "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1"


def record(path):
    path = Path(path)
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError("Require ordinary single-link evidence files")
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    resource = json.loads((SFM / "resource_cohort_CPU.json").read_text())
    prediction = json.loads((SFM / "actual_CPU_snapshots_v1/CPU_preparation.json").read_text())
    child = json.loads((SFM / "CPU_prediction_execution.json").read_text())
    if (resource["status"] != "CPU_resource_case_review_complete" or len(resource["cases"]) != 52
            or resource["admitted_cases"] != 48 or resource["rejected_cases"] != 4 or resource["Astar_segments"] != 2058
            or prediction["status"] != "CPU_preparation_passed" or len(prediction["cases"]) != 48
            or child["actual_child_returncode"] != 0 or child["actual_process_exit_observed"] is not True
            or prediction["RNG_unchanged"] is not True or prediction["CUDA_initialized"] is not False):
        raise ValueError("Complete CPU resource/prediction admission is missing")
    for case in prediction["cases"]:
        if case["status"] != "CPU_joint_prediction_passed":
            raise ValueError("A registered CPU prediction case failed")
        for role in ("configuration", "input_archive", "archive"):
            if record(case[role]["path"]) != case[role]:
                raise ValueError("CPU snapshot binding changed")
    snapshots = HERE / "source_snapshot_v1"
    snapshots.mkdir(exist_ok=False)
    selected = (
        "CrowdSim/humanoid/trajectory/native_social_force_runtime.py",
        "CrowdSim/humanoid/runtime/native_social_observation_producer.py",
        "CrowdSim/humanoid/runtime/native_social_navigation_lifecycle.py",
        "scripts/native_shared_social_derivation.py", "scripts/prepare_shared_social_native_sfm.py",
        "scripts/evaluate_shared_social_native.py", "tests/test_native_social_force_runtime.py",
        "tests/test_native_shared_social_evaluation.py",
        "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/prepare_real_cohort_v1.py",
        "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/run_CPU_prediction_cases_v1.py",
        "CrowdSim/humanoid/runtime/social_navigation_lifecycle.py",
        "CrowdSim/humanoid/runtime/social_observation_producer.py",
        "CrowdSim/humanoid/trajectory/social_force_runtime.py",
        "CrowdSim/humanoid/trajectory/social_force.py")
    source_rows = []
    for name in (*selected, str(Path(__file__).relative_to(ROOT))):
        source, target = ROOT / name, snapshots / name
        original = record(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied = record(target)
        if copied["bytes"] != original["bytes"] or copied["sha256"] != original["sha256"] or record(source) != original:
            raise ValueError("Source archive copy changed")
        source_rows.append({"original": original, "archived": copied})
    old = HERE / "native_navigation_lifecycle_tested_v2.py"
    current = ROOT / "CrowdSim/humanoid/runtime/native_social_navigation_lifecycle.py"
    if ast.dump(ast.parse(old.read_text()), include_attributes=False) != ast.dump(ast.parse(current.read_text()), include_attributes=False):
        raise ValueError("Tested/current lifecycle AST differs")
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 owner source guard changed")
    observed = {"schema": "shared_social.native_CPU_preparation_root_observed_exits.v1",
        "CPU_only": True, "actual_native_SDK_exit_observed": False, "paper_effect_achieved": False,
        "GPU_training_owner_modified": False, "active_training_guard_count": len(guards), "training_guards_unchanged": True,
        "resource_review": {"launch_tool_chunk": "813b9b", "terminal_tool_chunk": "679e0a", "actual_exit_code": 0},
        "CPU_prediction": {"launch_tool_chunk": "961777", "terminal_tool_chunk": "8560e6", "actual_exit_code": 0,
                           "child_exit": record(SFM / "CPU_prediction_execution.json")},
        "test30": {"launch_tool_chunk": "7f632c", "terminal_tool_chunk": "55b038", "actual_exit_code": 0,
                   "log": record(HERE / "tests_final_v2.log")},
        "native_metadata_preflight": [
            {"role": "DP_N3", "terminal_tool_chunk": "f94322", "actual_exit_code": 0, "report": record(HERE / "DP_N3_preflight_v2.json")},
            {"role": "SFM_N15", "terminal_tool_chunk": "487e24", "actual_exit_code": 0, "report": record(HERE / "SFM_N15_preflight_v2.json")}],
        "final_source_checks": [{"role": "DP_N3", "terminal_tool_chunk": "7c6211", "actual_exit_code": 0},
                                {"role": "SFM_N15", "terminal_tool_chunk": "503b19", "actual_exit_code": 0}],
        "Ruff": {"terminal_tool_chunk": "232e6c", "actual_exit_code": 0, "log": record(HERE / "ruff_final_v3.log")},
        "source_snapshot": record(archive), "sources": source_rows, "tested_prior_lifecycle": record(old),
        "tested_prior_and_current_lifecycle_AST_equal": True, "seal_script": record(__file__),
        "phase": "CPU_preparation_complete_native_SDK_initial_pair_and_metrics_unverified"}
    write_new(HERE / "CPU_preparation_execution.json", observed)
    files = sorted(path for folder in (HERE, SFM) for path in folder.rglob("*") if path.is_file())
    before = [record(path) for path in files]
    for path in files:
        path.chmod(0o444)
    if [record(path) for path in files] != before:
        raise ValueError("Evidence bytes changed while sealing")
    index = HERE / "CPU_preparation_seal.json"
    write_new(index, {"schema": "shared_social.native_CPU_preparation_readonly_seal.v1",
        "files": before, "count": len(before), "bytes_unchanged_after_readonly": True,
        "native_SDK_execution_validated": False, "paper_effect_achieved": False})
    index.chmod(0o444)
    print(json.dumps({"status": "CPU_evidence_sealed_native_SDK_unverified", "sealed_files": len(before),
                      "receipt": record(HERE / "CPU_preparation_execution.json"), "seal": record(index)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
