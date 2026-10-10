"""Preserve registered raw-audit versions, retained failure and CPU exits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.with_name("shared_social_native_raw_audit_preparation_v1")


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identities(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield {key: value[key] for key in ("path", "bytes", "sha256")}
        else:
            for child in value.values():
                yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def snapshot(path, expected, target):
    actual = record(path)
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise ValueError("Registered source version changed: " + str(path))
    if target.exists():
        raise ValueError("Use a new source snapshot")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    copied = record(target)
    if copied["sha256"] != actual["sha256"] or copied["bytes"] != actual["bytes"]:
        raise ValueError("Source snapshot differs")
    return {"registered": expected, "snapshot": copied}


def main():
    registrations = [json.loads((folder / "CPU_registration.json").read_text()) for folder in (OLD, HERE)]
    final = json.loads((HERE / "CPU_preparation_execution.json").read_text())
    if (len(final["children"]) != 6 or any(row["actual_child_returncode"] != 0 for row in final["children"])
            or final["native_actual_capture_audited"] is not False):
        raise ValueError("Require observed final CPU children and explicit scope")
    for row in final["children"]:
        if record(row["log"]["path"]) != row["log"]:
            raise ValueError("Observed CPU log changed")
    if "62 passed" not in (HERE / "tests_final.log").read_text():
        raise ValueError("Final test log differs")
    source_rows = []
    snapshots = HERE / "source_snapshot_v1"
    for folder, registration in zip((OLD, HERE), registrations):
        for expected in registration["sources"]:
            path = Path(expected["path"])
            if folder == OLD and path.name == "native_navigation_audit.py":
                path = OLD / "native_raw_reader_before_mutation.py"
            elif folder == OLD and path.name == "test_native_navigation_audit.py":
                path = OLD / "test_native_navigation_audit_tested_v1.py"
            target = snapshots / folder.name / Path(expected["path"]).relative_to(ROOT)
            source_rows.append(snapshot(path, expected, target))
        for row in identities(registration["kernel_source_bindings"]):
            if row["path"] in {item["path"] for item in registration["sources"]}:
                continue
            target = snapshots / folder.name / Path(row["path"]).relative_to(ROOT)
            if not target.exists():
                source_rows.append(snapshot(row["path"], row, target))
        mutation = json.loads((folder / "mutation_review.json").read_text())
        if (mutation["mutation_detected"] is not True or mutation["exact_source_bytes_restored"] is not True
                or mutation["observed_child"]["actual_child_returncode"] == 0):
            raise ValueError("Actual mutation failure/restoration missing")
        for name in ("native_raw_reader_mutated_test_only.py",):
            path = folder / name
            source_rows.append(snapshot(path, record(path), snapshots / "TEST_ONLY_MUTATIONS" / folder.name / name))
    historical = json.loads((HERE / "historical_N4_metric_preservation.json").read_text())
    for case in historical["cases"]:
        for key in ("base_every_field_except_schema_equal", "all_body_pose_fields_equal",
                    "all_response_support_fields_equal", "joint_fields_equal_after_explicit_actor_label_rename"):
            if case[key] is not True:
                raise ValueError("Historical math changed")
        if case["formal_native_SDK_and_initial_admission_run"] is not False:
            raise ValueError("Do not relabel historical fixture data as native captures")
        for row in identities(case):
            if record(row["path"]) != row:
                raise ValueError("Historical evidence changed")
    # One real older SDK receipt must fail the formal native CLI before any output.
    rejection = [sys.executable, str(ROOT / "scripts/audit_shared_social_native.py"),
        "--actors", "3", "--control-steps", "6000", "--planner", "diffusion", "--execution-receipt",
        str(ROOT / "docs/style_improvement_20261005/shared_social_S1_paired_actual_v1/diffusion200s_execution.json"),
        "--scene-config", str(ROOT / "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/scenes/Warehouse_N03.json"),
        "--scene-sha256", "309abc3c5a99f71e1146f9ee0063ee38e9665330fd0d482d6af5869c906df200",
        "--output", str(HERE / "MUST_NOT_EXIST_old_fixture_as_native.json")]
    log = HERE / "actual_old_fixture_CLI_rejection.log"
    with log.open("xb") as stream:
        rejected = subprocess.run(rejection, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if (rejected.returncode == 0 or Path(rejection[-1]).exists()
            or "Require externally observed actual successful native SDK exit" not in log.read_text()):
        raise ValueError("Actual historical SDK fixture was not rejected before output")
    write_new(HERE / "actual_old_fixture_CLI_rejection.json", {"command": rejection,
        "actual_child_exit_code": rejected.returncode, "actual_child_exit_observed": True,
        "log": record(log), "old_fixture_receipt_rejected": True, "output_created": False,
        "SDK_started": False, "quality_measured": False})
    current_seal = record(__file__)
    source_rows.append(snapshot(__file__, current_seal, snapshots / "final_seal" / Path(__file__).name))
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 training guard changed")
    write_new(HERE / "root_observed_CPU_execution.json", {
        "schema": "shared_social.native_raw_root_observed_CPU_exits.v1",
        "root_observed_outer_processes": [
            {"version": 1, "launch_tool_chunk": "c91083", "terminal_tool_chunk": "0b4919", "actual_exit_code": 1,
             "failure": "Historical ABI assumption: KeyError native_dynbench", "failure_retained": True},
            {"version": 2, "launch_tool_chunk": "c37572", "terminal_tool_chunk": "2972c4", "actual_exit_code": 0}],
        "registration_records": [record(folder / "CPU_registration.json") for folder in (OLD, HERE)],
        "final_CPU_child_receipt": record(HERE / "CPU_preparation_execution.json"),
        "historical_N4_numeric_equality": record(HERE / "historical_N4_metric_preservation.json"),
        "actual_old_fixture_rejection": record(HERE / "actual_old_fixture_CLI_rejection.json"),
        "source_snapshots": source_rows, "source_archive": record(archive), "seal_source": current_seal,
        "active_training_guards": guards, "active_training_guard_count": len(guards),
        "training_parameters_or_sources_modified": False, "new_native_SDK_capture_audited": False,
        "paper_effect_achieved": False, "default_model_promoted": False})
    paths = sorted(path for folder in (OLD, HERE) for path in folder.rglob("*") if path.is_file())
    for path in paths:
        path.chmod(0o444)
    rows = [record(path) for path in paths]
    if any(record(row["path"]) != row for row in rows):
        raise ValueError("CPU evidence changed during sealing")
    seal = HERE / "CPU_evidence_seal.json"
    write_new(seal, {"schema": "shared_social.native_raw_CPU_readonly_seal.v1", "files": rows,
        "indexed_file_count": len(rows), "indexed_files_readonly": True,
        "external_seal_tool_exit_receipt_included_in_index": False, "new_native_SDK_capture_audited": False})
    seal.chmod(0o444)
    print(json.dumps({"status": "native_raw_CPU_evidence_sealed", "sealed_files": len(rows),
        "receipt": record(HERE / "root_observed_CPU_execution.json"), "seal": record(seal),
        "active_training_guards_unchanged": True, "new_native_SDK_capture_audited": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
