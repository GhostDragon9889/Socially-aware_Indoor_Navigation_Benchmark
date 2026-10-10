"""Use the original recorded comparison with the actual NEW CPU v2 producer ABI."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_DIR = ROOT / "docs/style_improvement_20261003/default_DP_pair_comparison_preparation_v1"
REV_DIR = ROOT / "docs/style_improvement_20261003/default_newDP_physical_reader_revision_v2"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL = load("default_DP_original_recorded_comparison", OLD_DIR / "compare_recorded_pairs.py")
record, read = ORIGINAL.record, ORIGINAL.read
if record(ORIGINAL.__file__)["sha256"] != "f9a9b5d7bfa72e949a13b92c1bbd1c6ccd60ff905d196456b2ee61ea4fc16e5f":
    raise ValueError("Original recorded comparison changed")
REV = load("default_DP_actual_NEW_revision_evidence", REV_DIR / "physical_evidence_v2.py")
if record(REV.__file__)["sha256"] != "96d9ccdfb9cd62b0c1f82b26b208728e6d70045bfe4f7251aab51f4c6487fa2a":
    raise ValueError("Actual NEW metadata revision helper changed")
ORIGINAL_CURRENT_AUDIT = ORIGINAL.current_audit


def current_audit(evidence, plan, phase, path):
    if phase == "fixture_old":
        return ORIGINAL_CURRENT_AUDIT(evidence, plan, phase, path)
    evidence.require(phase == "fixture_new", "Only the registered NEW producer identity is revised")
    contract = REV.revision()
    case = plan["cases"][phase]
    evidence.require(str(Path(path).absolute()) == case["audit"], "Exact registered actual NEW audit required")
    document = read(path)
    admission = REV.admit_capture(plan, phase)
    evidence.require(document["schema"] == "default_newDP.independent_physical_CPU_audit.v1"
                     and document["status"] == "passed" and document["phase"] == phase
                     and document["actual_capture_independently_audited"] is True
                     and document["actual_source_admission"] == admission,
                     "Complete actual NEW audit/source admission required; failures are retained")
    evidence.require(document["execution_reader"] == contract["original_raw_driver"]
                     and document["actual_execution_adapter"] == contract["driver"]
                     and document["metadata_revision_helper"] == contract["helper"]
                     and document["metadata_revision_registration"] == record(REV_DIR / "revision_registration.json")
                     and document["metadata_revision_source_freeze"] == record(REV_DIR / "source_freeze_v2.json")
                     and document["first_actual_CPU_failure"] == contract["first_actual_CPU_failure"]
                     and document["original_raw_audit_body_and_mathematics_unchanged"] is True,
                     "Actual revised entry and unchanged raw math identity required")
    owner = read(case["CPU_receipt"])
    evidence.require(owner["schema"] == "default_newDP.actual_CPU_audit_launch.v2" and owner["phase"] == phase
                     and owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
                     and owner["child_exit_observed"] is True and owner["driver_completed"] is True
                     and owner["CPU_only"] is True and owner["GPU_execution_requested"] is False
                     and owner["command"] == contract["CPU_command"] and owner["audit_artifact"] == record(path)
                     and owner["owner"] == contract["owner"] and owner["actual_capture_admission"] == admission
                     and owner["metadata_revision_registration"] == record(REV_DIR / "revision_registration.json")
                     and owner["metadata_revision_source_freeze"] == record(REV_DIR / "source_freeze_v2.json")
                     and owner["binding_before"] == owner["binding_after"] == evidence.source_binding(plan),
                     "Actual closed CPU v2 producer/report required")
    evidence.require(not Path(path).stat().st_mode & 0o222, "Actual audit must be sealed readonly")
    return document, read(admission["result"]["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-audit", type=Path, required=True)
    parser.add_argument("--new-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Exclusive paired comparison output required")
    ORIGINAL.current_audit = current_audit
    report = ORIGINAL.compare(args.old_audit, args.new_audit)
    report.update(schema="default_DP_pair.actual_recorded_comparison.v2", actual_execution_adapter=record(__file__),
                  actual_NEW_CPU_owner=record(REV.registration()["cases"]["fixture_new"]["CPU_receipt"]),
                  actual_NEW_metadata_revision_source_freeze=record(REV_DIR / "source_freeze_v2.json"),
                  original_compare_arrays_and_progress_mathematics_unchanged=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "OLD": report["OLD_current185"]["raw_audit_physical_counts"],
                      "NEW": report["current185_OLD_NEW"]["NEW_current185"]["raw_audit_physical_counts"],
                      "NEW_full_physical_zero": report["current185_OLD_NEW"]["NEW_full_physical_zero"],
                      "output": record(args.output)}))


if __name__ == "__main__":
    main()
