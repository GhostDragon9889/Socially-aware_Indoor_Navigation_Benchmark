"""Seal completed seeded admission CPU evidence, preserving its non-SDK scope."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_seeded_style as style  # noqa: E402

record = style.metric.record


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
    if (len(execution["children"]) != 33 or any(row["actual_child_exit_code"] != 0 for row in execution["children"])
            or execution["new_native_capture_or_quality_measured"] is not False
            or "195 passed" not in (HERE / "tests_final.log").read_text()):
        raise ValueError("Actual completed CPU children or non-SDK scope missing")
    mutation = json.loads((HERE / "mutation_review.json").read_text())
    if (mutation["mutation_detected"] is not True or mutation["exact_source_bytes_restored"] is not True
            or mutation["actual_child"]["actual_child_exit_code"] == 0):
        raise ValueError("Actual seed condition mutation and exact restore missing")
    graph = [registration, json.loads((HERE / "historical_N4_regression.json").read_text())]
    for row in execution["children"]:
        if record(row["log"]["path"]) != row["log"]:
            raise ValueError("Actual CPU child log changed")
        if row["role"].startswith(("raw_", "style_")):
            graph.append(json.loads(Path(row["log"]["path"]).read_text()))
    for path in HERE.glob("shared_social_*/historical_N4_style_metric_preservation.json"):
        graph.append(json.loads(path.read_text()))
    originals = [*identities(graph), record(__file__)]
    copied, seen = [], {}
    snapshots = HERE / "source_snapshot_v1"
    for row in originals:
        path = Path(row["path"])
        if path.suffix != ".py":
            continue
        if row["path"] in seen:
            if seen[row["path"]] != row:
                raise ValueError("Conflicting source identities")
            continue
        seen[row["path"]] = row
        if record(path) != row:
            raise ValueError("CPU source no longer matches actual execution")
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external") / row["sha256"] / path.name
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Preserve source archives; use a new preparation version")
        shutil.copyfile(path, target)
        if target.read_bytes() != path.read_bytes():
            raise ValueError("Source archive differs")
        copied.append({"source": row, "snapshot": record(target)})
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if len(guards) != 23 or any(record(row["path"]) != row for row in guards):
        raise ValueError("S3 source guards changed")
    write_new(HERE / "root_observed_CPU_execution.json", {"schema": "shared_social.native_seeded_admission_root_CPU_exit.v1",
        "outer_launch_tool_chunk": "f60656", "outer_terminal_tool_chunk": "b25c7b", "actual_outer_exit_code": 0,
        "actual_CPU_child_execution": record(HERE / "CPU_preparation_execution.json"), "sources": copied,
        "source_archive": record(archive), "S3_guarded_sources": guards, "S3_guard_count": len(guards),
        "explicit_seed_raw_and_style_CPU_interfaces_prepared": True, "training_or_data_modified": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False, "default_model_promoted": False})
    paths = sorted(path for path in HERE.rglob("*") if path.is_file())
    for path in paths:
        path.chmod(0o444)
    indexed = [record(path) for path in paths]
    seal = HERE / "CPU_evidence_seal.json"
    write_new(seal, {"schema": "shared_social.native_seeded_admission_CPU_readonly_seal.v1", "files": indexed,
        "indexed_files_readonly": True, "indexed_file_count": len(indexed),
        "actual_seal_tool_exit_receipt_included_in_index": False, "new_native_capture_or_quality_measured": False})
    seal.chmod(0o444)
    print(json.dumps({"status": "seeded_native_admission_CPU_evidence_sealed", "sealed_files": len(indexed),
        "receipt": record(HERE / "root_observed_CPU_execution.json"), "seal": record(seal),
        "S3_sources_unchanged": True, "new_native_capture_or_quality_measured": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
