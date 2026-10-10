"""One metadata callback around the exact original physical raw audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from samples10_evidence import BASE_DIR, HERE, admit_capture, load, read, record, registration, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixture_new",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read(HERE / "registration.json")
    require(str(args.output.absolute()) == contract["actual_paths"]["audit"] and not args.output.exists(),
            "Exclusive actual registered samples10 audit output required")
    plan = registration()
    sys.path.insert(0, str(BASE_DIR))
    core = load("samples10_original_physical_raw_body", BASE_DIR / "audit_physical.py")
    require(record(core.__file__) == contract["frozen_inputs"]["original_raw_body"], "Frozen raw body changed")
    core.admit_capture = admit_capture
    try:
        report = core.audit(plan, args.phase)
        report.update(schema="default_NEW.samples10.independent_physical_CPU_audit.v1",
                      actual_execution_adapter=record(__file__), metadata_helper=record(HERE / "samples10_evidence.py"),
                      samples10_actual_source_freeze=record(HERE / "source_freeze_v1.json"),
                      original_raw_math_unchanged=True, all_failed_samples_retained=True)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"schema": "default_NEW.samples10.independent_physical_CPU_audit.v1", "status": "failed",
                       "phase": args.phase, "failure": repr(error), "actual_capture_independently_audited": False,
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
