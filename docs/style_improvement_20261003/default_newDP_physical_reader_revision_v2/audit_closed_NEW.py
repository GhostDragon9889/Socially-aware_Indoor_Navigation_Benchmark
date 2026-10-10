"""Actual CPU retry using original frozen raw audit and corrected metadata admission."""
import argparse
import json
from pathlib import Path
import sys

from physical_evidence_v2 import BASE_DIR, PREP, admit_capture, record, registration, require, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixture_new",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = revision()
    require(str(args.output.absolute()) == contract["audit_output"] and not args.output.exists(), "Exclusive registered actual NEW retry output required")
    sys.path.insert(0, str(BASE_DIR))
    from physical_evidence_v2 import BASE
    core = BASE.load("default_DP_original_raw_audit_body", BASE_DIR / "audit_physical.py")
    require(record(core.__file__) == contract["original_raw_driver"], "Original frozen raw audit body changed")
    core.admit_capture = admit_capture  # The only replaced source admission callback; raw formulas stay original.
    try:
        report = core.audit(registration(), args.phase)
        report.update(actual_execution_adapter=record(__file__), metadata_revision_helper=contract["helper"],
                      metadata_revision_registration=record(PREP / "revision_registration.json"),
                      metadata_revision_source_freeze=record(PREP / "source_freeze_v2.json"),
                      original_raw_audit_body_and_mathematics_unchanged=True,
                      first_actual_CPU_failure=contract["first_actual_CPU_failure"])
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"schema": "default_newDP.independent_physical_CPU_audit.v1", "status": "failed", "phase": args.phase,
                       "failure": repr(error), "actual_capture_independently_audited": False,
                       "actual_execution_adapter": record(__file__)}, stream, indent=2)
            stream.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "physical_gate_counts": report["physical_gate_counts"],
                      "physical_zero_gate_passed": report["physical_zero_gate_passed"]}))


if __name__ == "__main__":
    main()
