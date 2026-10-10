"""Execute only a registered full22 or200s protocol from isolated frozen code.

The CPU preparation never calls this executor. Actual execution requires a
sealed2000 checkpoint and independent full-payload admission. Existing v3
guard/ledger supplies the same isolation boundary as fresh training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_v1"
CODE = BUNDLE / "code"
REGISTRATION = DOCS / "state_action_evaluation_registration_v1/registration.json"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_admission(document, checkpoint):
    """No checkpoint training counters are inferred from its filename."""
    if (document.get("schema") != "state_action.actual_training_checkpoint_audit.v1" or
            document.get("status") != "passed" or document.get("CPU_only") is not True or
            document.get("GPU_used") is not False or document.get("SDK_started") is not False or
            document.get("sealed34_outputs_read") is not False or
            not document.get("checks") or not all(row.get("passed") is True for row in document["checks"])):
        raise ValueError("Independent actual state-action payload admission required")
    rows = [row for row in document.get("checkpoints", []) if row.get("updates") == 2000]
    if (len(rows) != 1 or rows[0].get("checkpoint") != checkpoint or
            any(type(rows[0].get(key)) is not int for key in ("updates", "transitions", "D_input")) or
            rows[0].get("transitions") != 16384000 or rows[0].get("D_input") != 2019 or
            rows[0].get("full_payload_resume_validated") is not True or
            rows[0].get("public_D_or_AMP_RMS_transferred") is not False or
            document.get("actual_observed_owned_imports_isolated") is not True):
        raise ValueError("Sealed checkpoint/hash/count/D/resume/source admission differs")
    return rows[0]


def build_command(protocol, checkpoint_path, checkpoint_sha256, output, registration):
    if protocol not in {"tracking", "navigation"}:
        raise ValueError("Only registered tracking or navigation protocols")
    if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
        raise ValueError("Actual64-character lowercase sealed checkpoint SHA required")
    if Path(checkpoint_path).resolve() != Path(registration["future_checkpoint"]["path"]).resolve():
        raise ValueError("Only the preregistered immutable step2000 checkpoint is accepted")
    section = registration[protocol]
    argv = list(section["argv_without_future_SHA"] if protocol == "tracking" else section["argv"])
    argv[argv.index("--output") + 1] = str(Path(output).resolve())
    flag = "--checkpoint" if protocol == "tracking" else "--resume"
    argv[argv.index(flag) + 1] = str(Path(checkpoint_path).resolve())
    if protocol == "tracking":
        argv.extend(["--checkpoint-sha256", checkpoint_sha256])
    entry = CODE / "scripts" / ("benchmark_paper_motion_tracking.py" if protocol == "tracking" else "train_paper_amp.py")
    return entry, argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("tracking", "navigation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="Tracking headers only; navigation uses existing SDK-free resource preflight")
    args = parser.parse_args(argv)
    registration = json.loads(REGISTRATION.read_text())
    if registration["status"] != "protocol_and_isolated_parsers_passed_future_checkpoint_pending":
        raise ValueError("Unknown original protocol registration")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Fresh output only; no historical overwrite")
    launcher_path = BUNDLE / "launch_isolated_v3.py"
    if record(launcher_path) != registration["sources"]["isolated_launcher_v3"]:
        raise ValueError("Registered isolation launcher changed")
    if record(BUNDLE / "source_manifest_v3.json") != registration["sources"]["isolated_source_manifest_v3"]:
        raise ValueError("Registered isolation manifest changed")
    checkpoint = record(args.checkpoint)
    if checkpoint["sha256"] != args.checkpoint_sha256:
        raise ValueError("Actual sealed checkpoint SHA differs")
    admission = json.loads(args.checkpoint_admission.read_text())
    validate_admission(admission, checkpoint)
    entry, runner_args = build_command(args.protocol, args.checkpoint, args.checkpoint_sha256, args.output, registration)
    if record(entry) != registration["sources"]["entrypoint_bytes"][str(entry.relative_to(CODE))]:
        raise ValueError("Exact frozen isolated benchmark/driver bytes required")
    specification = importlib.util.spec_from_file_location("registered_evaluation_isolation_v3", launcher_path)
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    launcher.verify()
    ledger = launcher.ImportLedger(args.import_report)
    ledger.write("registered_evaluation_admitted", protocol=args.protocol, adapter=record(__file__),
                 registration=record(REGISTRATION), payload_admission=record(args.checkpoint_admission),
                 checkpoint=checkpoint, entry=record(entry), runner_arguments=runner_args)
    launcher.activate(ledger)
    status = "failed"
    try:
        if args.preflight_only and args.protocol == "navigation":
            from CrowdSim.humanoid.application.preflight import main as preflight
            status = "passed_preflight" if preflight(runner_args) == 0 else "failed_preflight"
            if status == "failed_preflight":
                raise RuntimeError("Registered navigation resource preflight failed")
        else:
            if args.preflight_only:
                runner_args.append("--preflight-only")
            ledger.phase = "registered_benchmark_entry"
            ledger.observe("scripts." + entry.stem + ".__main__", entry)
            ledger.write("registered_entry_selected", path=str(entry), sha256=record(entry)["sha256"],
                         argv=runner_args, execution_boundary="before_runpy_not_an_actual_physical_completion")
            sys.argv = [str(entry), *runner_args]
            runpy.run_path(str(entry), run_name="__main__")
            status = "passed"
    except SystemExit as error:
        status = "passed_system_exit" if error.code in (None, 0) else "failed_system_exit"
        raise
    finally:
        ledger.finish(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
