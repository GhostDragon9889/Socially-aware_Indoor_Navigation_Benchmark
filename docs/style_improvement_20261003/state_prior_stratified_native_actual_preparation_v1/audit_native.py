"""q metadata adapter around original d707 native4/15 full raw mathematics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import native_evidence as evidence


def registration(actors):
    core = evidence.raw_core()
    registered = core.registration(actors)
    # The immutable core's old schema is an internal ABI label, not q checkpoint provenance.
    registered["status"] = "prepared_actual_q_terminal_native_capture_pending"
    registered["registration_timing"] = "q terminal SHA/count/source are actual; only future native captures are pending."
    return {"schema": "stratified_prior.native200s_CPU_reader_registration.v1", "status": "prepared_native_capture_pending",
            "created_utc": datetime.now(timezone.utc).isoformat(), "actors": actors,
            "execution_entry": evidence.record(__file__), "metadata_helper": evidence.record(evidence.__file__),
            "original_core": evidence.record(evidence.CORE_DIR / "audit_state_prior_native.py"),
            "native_execution_audited": False, "core_registration": registered,
            "internal_core_schema_retained_only_for_frozen_reader_ABI": True,
            "raw_math_changed": False, "future_native_capture_read": False, "GPU_used": False, "SDK_started": False}


def audit(actors, registered):
    evidence.require(registered["schema"] == "stratified_prior.native200s_CPU_reader_registration.v1"
                     and registered["actors"] == actors and registered["native_execution_audited"] is False
                     and registered["execution_entry"] == evidence.record(__file__)
                     and registered["metadata_helper"] == evidence.record(evidence.__file__)
                     and registered["original_core"] == evidence.record(evidence.CORE_DIR / "audit_state_prior_native.py"),
                     "Exact q native reader metadata registration required")
    core = evidence.raw_core()
    report = core.audit(actors, registered["core_registration"])
    evidence.require(report["execution_reader"] == registered["original_core"], "Original complete raw executor changed")
    report["schema"] = "stratified_prior.native200s_independent_CPU_audit.v1"
    report["execution_entry"] = evidence.record(__file__)
    report["metadata_admission_helper"] = registered["metadata_helper"]
    report["q_metadata_registration"] = registered
    report["compiled_raw_math_changed"] = False
    report["evidence_limits"]["causality"] = "All failures/contacts/resets/routes retained. q versus19f changes expert sampling; matching public parameters do not imply bitwise planning noise or establish causal transfer/superiority. No MMD launch filter."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, choices=(4, 15), required=True)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence.require(not args.output.exists(), "Preserve earlier actual report")
    evidence.require(args.register_only or args.registration is not None, "Frozen q native registration required")
    report = registration(args.actors) if args.register_only else audit(args.actors, evidence.read(args.registration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "actual_native_audited": report.get("native_200s_artifacts_independently_audited", False),
                      "counts": report.get("counts"), "models_loaded_by_reader": False}))


if __name__ == "__main__":
    main()
