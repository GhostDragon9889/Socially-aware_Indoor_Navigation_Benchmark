"""Root-owned q native4 then15; original actual wait/cleanup body is retained."""
from __future__ import annotations

import inspect
import sys

import native_evidence as evidence


def owner_source():
    sys.path.insert(0, str(evidence.BASE_DIR))
    parent = evidence.load("q_native_original_actual_process_owner", evidence.BASE_DIR / "launch_registered.py")
    source = inspect.getsource(parent.main)
    old_ordering = '''    if not args.cpu_audit and args.phase == "fixture_new":
        # Complete OLD even if physically failed; its integrity/actual closure, not quality, orders NEW.
        old_case = plan["cases"]["fixture_old"]
        old_owner = read(old_case["GPU_receipt"])
        previous = admit_capture(plan, "fixture_old")
        require(old_owner["status"] == "closed", "Actual complete OLD fixture must precede NEW")
'''
    new_ordering = '''    if not args.cpu_audit and args.phase == "native15":
        previous = native4_gate(plan)
'''
    substitutions = (
        ('choices=("fixture_old", "fixture_new")', 'choices=("native4", "native15")'),
        (old_ordering, new_ordering),
        ('"default_newDP.actual_CPU_audit_launch.v1"', '"stratified_prior.actual_native_CPU_audit_launch.v1"'),
        ('"default_newDP.actual_physical_launch.v1"', '"stratified_prior.actual_native_physical_launch.v1"'),
        ('report["status"] == "passed"',
         'report["status"] in ("raw_sensor_source_contracts_passed_bounded_zero_fall_observed", "raw_sensor_source_contracts_passed_physical_stability_failed")'),
    )
    revised = source
    for old, new in substitutions:
        evidence.require(revised.count(old) == 1, "One original owner metadata boundary required")
        revised = revised.replace(old, new)
    reverse = revised
    for old, new in reversed(substitutions):
        reverse = reverse.replace(new, old)
    evidence.require(reverse == source, "Original process wait/cleanup/exception/finally bytes must remain exact")
    return parent, revised


def main():
    parent, revised = owner_source()
    namespace = {**parent.__dict__, "HERE": evidence.HERE, "ROOT": evidence.ROOT,
                 "CPU_ENV": evidence.CPU_ENV, "GPU_ENV": evidence.GPU_ENV,
                 "admit_capture": evidence.admit_capture, "native4_gate": evidence.native4_gate,
                 "read": evidence.read, "record": evidence.record, "registration": evidence.registration,
                 "require": evidence.require, "source_binding": evidence.source_binding,
                 "__file__": __file__, "__doc__": __doc__}
    exec(compile(revised, __file__, "exec"), namespace)
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
