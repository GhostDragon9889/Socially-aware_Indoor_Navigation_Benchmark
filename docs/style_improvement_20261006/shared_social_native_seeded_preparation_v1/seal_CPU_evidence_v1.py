"""Seal explicit-seed CPU sources/registration without claiming SDK execution."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_shared_social_native_seeded as entry  # noqa: E402

record = entry.record


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
    executed = json.loads((HERE / "CPU_preparation_execution.json").read_text())
    if (len(executed["children"]) != 17 or any(row["actual_child_exit_code"] != 0 for row in executed["children"])
            or executed["new_native_capture_or_quality_measured"] is not False):
        raise ValueError("Require all actual CPU children and explicit non-SDK scope")
    graph = []
    for row in executed["children"]:
        if record(row["log"]["path"]) != row["log"]:
            raise ValueError("Actual child log changed")
        if row["role"].endswith("_source") or row["role"].endswith("_preflight"):
            graph.append(json.loads(Path(row["log"]["path"]).read_text()))
    if "48 passed" not in (HERE / "tests_final.log").read_text():
        raise ValueError("Final CPU tests changed")
    mutation = json.loads((HERE / "mutation_review.json").read_text())
    if (mutation["mutation_detected"] is not True or mutation["exact_source_bytes_restored"] is not True
            or mutation["actual_child"]["actual_child_exit_code"] == 0):
        raise ValueError("Actual type-guard mutation/restoration missing")
    matrix = json.loads((HERE / "registered_S3_native_matrix.json").read_text())
    if ((matrix["registered_cohorts"], matrix["resource_admitted_cohorts"], matrix["resource_rejected_cohorts"],
         matrix["planned_full200s_arm_runs_after_all_gates"]) != (156, 144, 12, 288)
            or matrix["actual_training_exit_admitted"] is not False
            or matrix["native_seeded_raw_and_style_admission_still_required"] is not True):
        raise ValueError("Registered scope or pending actual gates changed")
    rows = [*registration["sources"], *identities(registration["bindings"]), *identities(graph)]
    rows.append(record(__file__))
    seen, snapshots, copied = set(), HERE / "source_snapshot_v1", []
    for row in rows:
        path = Path(row["path"])
        if path.suffix != ".py" or row["path"] in seen:
            continue
        seen.add(row["path"])
        if record(path) != row:
            raise ValueError("Source no longer matches actual execution")
        target = snapshots / path.relative_to(ROOT)
        if target.exists():
            raise ValueError("Use a new source archive")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        after = record(target)
        if (after["bytes"], after["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("Source archive differs")
        copied.append({"source": row, "snapshot": after})
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("S3 training guard changed")
    write_new(HERE / "root_observed_CPU_execution.json", {"schema": "shared_social.native_seeded_root_observed_CPU_exit.v1",
        "outer_launch_tool_chunk": "cd3d4b", "outer_terminal_tool_chunk": "6b72a5", "actual_outer_exit_code": 0,
        "actual_CPU_child_execution": record(HERE / "CPU_preparation_execution.json"),
        "registered_future_matrix": record(HERE / "registered_S3_native_matrix.json"),
        "sources": copied, "source_archive": record(archive), "S3_guarded_sources": guards,
        "S3_guard_count": len(guards), "training_or_data_modified": False,
        "new_seeded_raw_style_admission_complete": False, "new_native_capture_or_quality_measured": False,
        "paper_effect_achieved": False, "default_model_promoted": False})
    paths = sorted(path for path in HERE.rglob("*") if path.is_file())
    for path in paths:
        path.chmod(0o444)
    indexed = [record(path) for path in paths]
    if any(record(row["path"]) != row for row in indexed):
        raise ValueError("CPU evidence changed during sealing")
    seal = HERE / "CPU_evidence_seal.json"
    write_new(seal, {"schema": "shared_social.native_seeded_CPU_readonly_seal.v1", "files": indexed,
        "indexed_files_readonly": True, "indexed_file_count": len(indexed),
        "actual_seal_tool_exit_receipt_included_in_index": False, "new_native_capture_or_quality_measured": False})
    seal.chmod(0o444)
    print(json.dumps({"status": "native_seeded_CPU_evidence_sealed", "sealed_files": len(indexed),
        "receipt": record(HERE / "root_observed_CPU_execution.json"), "seal": record(seal),
        "S3_training_guards_unchanged": True, "new_native_capture_or_quality_measured": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
