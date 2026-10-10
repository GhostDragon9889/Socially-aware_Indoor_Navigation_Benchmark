"""Seal the completed native tools bridge CPU preparation, without SDK claims."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_shared_social_native_owner as owner  # noqa: E402

bridge = owner.bridge

record = bridge.record


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


def main():
    registration = json.loads((HERE / "CPU_registration.json").read_text())
    execution = json.loads((HERE / "CPU_preparation_execution.json").read_text())
    if (len(execution["children"]) != 10
            or any(row["actual_child_exit_code"] != 0 or row["actual_child_exit_observed"] is not True
                   for row in execution["children"])
            or execution["actual_S3_readiness_admitted"] is not False
            or execution["actual_native_SDK_or_quality_measured"] is not False
            or "42 passed" not in (HERE / "tests_final.log").read_text()):
        raise ValueError("Actual CPU exits or explicit non-SDK scope missing")
    mutation = json.loads((HERE / "V8_mutation_review.json").read_text())
    if (mutation["CPU_contract_suite_refused_mutation"] is not True or mutation["exact_source_bytes_restored"] is not True
            or mutation["compilation_succeeded"] is not True
            or record(owner.DRIVER)["sha256"] != mutation["restored_sha256"]):
        raise ValueError("Mutation or exact source restore missing")
    v8 = json.loads((HERE / "V8_contract_execution.json").read_text())
    if v8["passed"] != 13 or v8["SDK_started"] is not False:
        raise ValueError("Actual V8 CPU contracts missing")
    graph = [registration]
    for row in execution["children"]:
        if record(row["log"]["path"]) != row["log"]:
            raise ValueError("Actual CPU child log changed")
        if row["role"].endswith("_envelope"):
            envelope = json.loads(Path(row["log"]["path"]).read_text())
            full, _ = owner.admitted_plan(Path(envelope["registration"]["path"]))
            if envelope["plan"] != owner.owner_view(full):
                raise ValueError("Runtime command view differs from registered plan")
            plan = full["case_plan"]
            if plan["GPU_or_SDK_started"] is not False or plan["actual_execution_admitted"] is not False:
                raise ValueError("CPU plan claims actual SDK execution")
            graph.append(plan)
    snapshots = HERE / "source_snapshot_v1"
    copied, seen = [], {}
    for row in [*identities(graph), record(__file__)]:
        path = Path(row["path"])
        if path.suffix not in (".py", ".js"):
            continue
        if row["path"] in seen:
            if seen[row["path"]] != row:
                raise ValueError("Conflicting source identities")
            continue
        seen[row["path"]] = row
        if record(path) != row:
            raise ValueError("CPU source changed after measurement")
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external") / row["sha256"] / path.name
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Preserve existing source snapshots")
        shutil.copyfile(path, target)
        if target.read_bytes() != path.read_bytes():
            raise ValueError("Source snapshot differs")
        copied.append({"source": row, "snapshot": record(target)})
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner_state = json.loads((bridge.S3 / "terminal_owner_state.json").read_text())
    guards = owner_state["bindings"]["guarded_sources"]
    if len(guards) != 23 or any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 source guard changed")
    write_new(HERE / "root_observed_CPU_execution.json", {
        "schema": "shared_social.native_tools_owner_root_CPU_exit.v1",
        "outer_launch_tool_chunk": "dc8d07", "outer_terminal_tool_chunk": "6256ea", "actual_outer_exit_code": 0,
        "actual_CPU_child_execution": record(HERE / "CPU_preparation_execution.json"), "sources": copied,
        "matrix": record(bridge.MATRIX), "source_archive": record(archive), "S3_guarded_sources": guards,
        "S3_guard_count": len(guards), "runtime_plan_identity_checked_before_metadata_or_child_launch": True,
        "actual_native_SDK_exit_or_readiness_tested": False, "training_or_data_modified": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False})
    paths = sorted(path for path in HERE.rglob("*") if path.is_file())
    for path in paths:
        path.chmod(0o444)
    seal = HERE / "CPU_evidence_seal.json"
    write_new(seal, {"schema": "shared_social.native_tools_owner_CPU_readonly_seal.v1",
        "files": [record(path) for path in paths], "indexed_file_count": len(paths), "indexed_files_readonly": True,
        "actual_seal_tool_exit_receipt_included_in_index": False, "actual_native_SDK_exit_or_readiness_tested": False})
    seal.chmod(0o444)
    print(json.dumps({"status": "native_tools_owner_CPU_preparation_sealed", "indexed_files": len(paths),
        "seal": record(seal), "root_observation": record(HERE / "root_observed_CPU_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
