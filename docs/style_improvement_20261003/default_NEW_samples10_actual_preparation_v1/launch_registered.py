"""Delegate registered samples10 processes to the frozen original cleanup body."""
from __future__ import annotations

import inspect
import sys

import samples10_evidence as evidence


def owner_source():
    sys.path.insert(0, str(evidence.BASE_DIR))
    parent = evidence.load("samples10_original_actual_process_owner", evidence.BASE_DIR / "launch_registered.py")
    source = inspect.getsource(parent.main)
    old_ordering = '''    if not args.cpu_audit and args.phase == "fixture_new":
        # Complete OLD even if physically failed; its integrity/actual closure, not quality, orders NEW.
        old_case = plan["cases"]["fixture_old"]
        old_owner = read(old_case["GPU_receipt"])
        previous = admit_capture(plan, "fixture_old")
        require(old_owner["status"] == "closed", "Actual complete OLD fixture must precede NEW")
'''
    substitutions = (
        ('choices=("fixture_old", "fixture_new")', 'choices=("fixture_new",)'),
        (old_ordering, ''),
        ('"default_newDP.actual_CPU_audit_launch.v1"', '"default_NEW.samples10.actual_CPU_audit_launch.v1"'),
        ('"default_newDP.actual_physical_launch.v1"', '"default_NEW.samples10.actual_physical_launch.v1"'),
    )
    revised = source
    for old, new in substitutions:
        evidence.require(revised.count(old) == 1, "Exact original owner metadata fragment required")
        revised = revised.replace(old, new)
    # Restore the removed ordering at its unique original boundary before comparison.
    reverse = revised.replace('    previous = None\n    admission =', '    previous = None\n' + old_ordering + '    admission =')
    for old, new in reversed(substitutions):
        if new:
            reverse = reverse.replace(new, old)
    evidence.require(reverse == source, "Original wait/cleanup/exception/finally bytes must remain exact")
    return parent, revised


def main():
    parent, revised = owner_source()
    namespace = {**parent.__dict__, "HERE": evidence.HERE, "ROOT": evidence.ROOT,
                 "CPU_ENV": evidence.CPU_ENV, "GPU_ENV": evidence.GPU_ENV,
                 "admit_capture": evidence.admit_capture, "read": evidence.read, "record": evidence.record,
                 "registration": evidence.registration, "require": evidence.require,
                 "source_binding": evidence.source_binding, "__file__": __file__, "__doc__": __doc__}
    exec(compile(revised, __file__, "exec"), namespace)
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
