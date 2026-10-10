"""Root-owned registered physical/CPU child; preparation never starts a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from physical_evidence_v2 import CPU_ENV, GPU_ENV, HERE, PREP, ROOT, admit_capture, read, record, registration, require, revision, source_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixture_new",), required=True)
    parser.add_argument("--cpu-audit", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(args.cpu_audit, "Revision is CPU audit only; no GPU launch path is authorized")
    contract = revision()
    plan = registration()
    binding = source_binding(plan)
    case = dict(plan["cases"][args.phase])
    case["CPU_command"] = contract["CPU_command"]
    command = case["CPU_command"] if args.cpu_audit else case["command"]
    settings = CPU_ENV if args.cpu_audit else GPU_ENV
    if args.prepare_only:
        print(json.dumps({"status": "prepared_source_bound_only", "phase": args.phase, "command": command,
                          "binding": binding, "future_physical_effect": False, "models_executed": False,
                          "GPU_used": False, "SDK_started": False, "pending_physical_gate": case["prior_physical_gate"]}))
        return 0
    previous = None
    if not args.cpu_audit and args.phase == "fixture_new":
        # Complete OLD even if physically failed; its integrity/actual closure, not quality, orders NEW.
        old_case = plan["cases"]["fixture_old"]
        old_owner = read(old_case["GPU_receipt"])
        previous = admit_capture(plan, "fixture_old")
        require(old_owner["status"] == "closed", "Actual complete OLD fixture must precede NEW")
    admission = admit_capture(plan, args.phase) if args.cpu_audit else None
    require(not Path(case["audit"] if args.cpu_audit else case["output"]).exists(), "Exclusive actual output; retain earlier failure")
    receipt_path = Path(case["CPU_receipt"] if args.cpu_audit else case["GPU_receipt"])
    log = receipt_path.with_name(receipt_path.stem + ("_CPU.log" if args.cpu_audit else "_GPU.log"))
    require(not receipt_path.exists() and not log.exists(), "Preserve previous actual receipt/log")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "default_newDP.actual_CPU_audit_launch.v2" if args.cpu_audit else "default_newDP.actual_physical_launch.v1",
               "status": "running", "phase": args.phase, "command": command, "cwd": str(ROOT),
               "declared_env": settings, "owner": record(__file__), "registration": record(HERE / "registration.json"),
               "binding_before": binding, "prior_physical_admission": previous, "actual_capture_admission": admission,
               "metadata_revision_registration": record(PREP / "revision_registration.json"),
               "metadata_revision_source_freeze": record(PREP / "source_freeze_v2.json"),
               "CPU_only": args.cpu_audit, "GPU_execution_requested": not args.cpu_audit,
               "sealed34_outputs_read": False, "start_utc": datetime.now(timezone.utc).isoformat(), "log": str(log),
               "artifacts": [], "child_exit_observed": False, "driver_completed": False, "exit_code": None}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    child, started = None, time.monotonic()
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **settings}, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["child_pid"] = child.pid
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = child.wait()
        receipt.update(status="closed", child_exit_observed=True, driver_completed=True)
        after = source_binding(plan)
        require(after == binding, "Registered source/model binding changed during execution")
        receipt["binding_after"] = after
        if receipt["exit_code"] == 0:
            if args.cpu_audit:
                artifact = Path(case["audit"])
                report = read(artifact)
                require(report["status"] == "passed" and report["actual_source_admission"] == admission,
                        "Actual independent CPU report/source admission differs")
                artifact.chmod(0o444)
                receipt["audit_artifact"] = record(artifact)
            else:
                result = read(Path(case["output"]) / "result.json")
                require(result["status"] == "passed" and result["actual_control_steps"] == 6000,
                        "Complete registered physical process result required; physical quality is audited separately")
                for artifact in sorted(Path(case["output"]).rglob("*")):
                    if artifact.is_file():
                        require(not artifact.is_symlink(), "Actual artifacts must be regular files")
                        artifact.chmod(0o444)
                        receipt["artifacts"].append(record(artifact))
    except BaseException as error:
        receipt.update(status="owner_failed", owner_error=repr(error), driver_completed=False)
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=10)
            except BaseException as cleanup_error:
                receipt["cleanup_error"] = repr(cleanup_error)
            finally:
                receipt.update(exit_code=child.returncode, child_exit_observed=child.returncode is not None)
        raise
    finally:
        receipt.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        if log.exists():
            log.chmod(0o444)
            receipt["actual_log"] = record(log)
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        receipt_path.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"], "receipt": record(receipt_path)}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
