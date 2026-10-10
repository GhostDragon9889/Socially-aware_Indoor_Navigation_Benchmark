"""Execute only a registered full22 or200s protocol from isolated frozen code.

The CPU preparation never calls this executor. Actual execution requires a
sealed2000 checkpoint and independent full-payload admission. The new931 guarded launcher/ledger supplies the same isolation boundary as fresh training.
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
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_protected_v1"
CODE = BUNDLE / "code"
REGISTRATION = DOCS / "terrain_protected_evaluation_registration_v1/registration.json"


REGISTRATION_SHA = "498e353b8528ab0a741b9ac68fd56899d372e0648f1fb065df50123ff780868d"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate_admission(document, checkpoint, registration=None):
    """No checkpoint training counters are inferred from its filename."""
    if (
        document.get("schema")
        != "protected_terrain_state_action.actual_training_checkpoint_audit.v1"
        or document.get("status") != "passed"
        or document.get("CPU_only") is not True
        or document.get("GPU_used") is not False
        or document.get("SDK_started") is not False
        or document.get("sealed34_outputs_read") is not False
        or document.get("protected_reference_policy_validated") is not True
        or document.get("actual_terminal_fresh2000_confirmed") is not True
        or not document.get("checks")
        or not all(row.get("passed") is True for row in document["checks"])
    ):
        raise ValueError("Independent actual state-action payload admission required")
    rows = [
        row for row in document.get("checkpoints", []) if row.get("updates") == 2000
    ]
    if (
        len(rows) != 1
        or rows[0].get("checkpoint") != checkpoint
        or any(
            type(rows[0].get(key)) is not int
            for key in ("updates", "transitions", "D_input")
        )
        or rows[0].get("transitions") != 16384000
        or rows[0].get("D_input") != 2019
        or rows[0].get("full_payload_resume_validated") is not True
        or rows[0].get("public_D_or_AMP_RMS_transferred") is not False
        or document.get("actual_observed_owned_imports_isolated") is not True
    ):
        raise ValueError(
            "Sealed checkpoint/hash/count/D/resume/source admission differs"
        )
    policy = rows[0].get("terrain_policy", {})
    if (
        rows[0].get("protected_policy_independent_checks") is not True
        or policy.get("schema") != "navisaaclab.task_terrain_statistics_policy.v1"
        or policy.get("mode") != "preserve_declared_suffix"
        or policy.get("bound") is not True
        or policy.get("task_slice") != [20, 1044]
        or type(policy.get("post_update_calls")) is not int
        or policy["post_update_calls"] != 2000
        or policy.get("reference_count") != 61931520001.0
        or policy.get("warm_start_task_count") != 200000.0
        or policy.get("global_count_and_prefix20_protected") is not False
        or policy.get("external_statistics_file_required") is not False
    ):
        raise ValueError(
            "Independent protected suffix/reference/update-hook admission required"
        )
    digest = policy.get("reference_statistics_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(
            "Saved independent protected reference statistics SHA required"
        )
    if (
        registration is not None
        and document.get("source_manifest")
        != registration["sources"]["isolated_source_manifest"]
    ):
        raise ValueError(
            "Independent payload admission refers to a different isolated931 source manifest"
        )
    return rows[0]


def build_command(protocol, checkpoint_path, checkpoint_sha256, output, registration):
    if protocol not in {"tracking", "navigation"}:
        raise ValueError("Only registered tracking or navigation protocols")
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise ValueError("Actual64-character lowercase sealed checkpoint SHA required")
    if (
        Path(checkpoint_path).resolve()
        != Path(registration["future_checkpoint"]["path"]).resolve()
    ):
        raise ValueError(
            "Only the preregistered immutable step2000 checkpoint is accepted"
        )
    section = registration[protocol]
    argv = list(
        section["argv_without_future_SHA"]
        if protocol == "tracking"
        else section["argv"]
    )
    argv[argv.index("--output") + 1] = str(Path(output).resolve())
    flag = "--checkpoint" if protocol == "tracking" else "--resume"
    argv[argv.index(flag) + 1] = str(Path(checkpoint_path).resolve())
    if protocol == "tracking":
        argv.extend(["--checkpoint-sha256", checkpoint_sha256])
    entry = (
        CODE
        / "scripts"
        / (
            "benchmark_paper_motion_tracking.py"
            if protocol == "tracking"
            else "train_paper_amp.py"
        )
    )
    return entry, argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("tracking", "navigation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Tracking headers only; navigation uses existing SDK-free resource preflight",
    )
    args = parser.parse_args(argv)
    if record(REGISTRATION)["sha256"] != REGISTRATION_SHA:
        raise ValueError("Frozen preregistered protocol bytes changed")
    registration = json.loads(REGISTRATION.read_text())
    if (
        registration["status"]
        != "protocol_and_isolated_parsers_passed_future_checkpoint_pending"
    ):
        raise ValueError("Unknown original protocol registration")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise ValueError("Fresh output only; no historical overwrite")
    launcher_path = BUNDLE / "launch_isolated.py"
    if record(launcher_path) != registration["sources"]["isolated_launcher"]:
        raise ValueError("Registered isolation launcher changed")
    if (
        record(BUNDLE / "source_manifest.json")
        != registration["sources"]["isolated_source_manifest"]
    ):
        raise ValueError("Registered isolation manifest changed")
    checkpoint = record(args.checkpoint)
    if checkpoint["sha256"] != args.checkpoint_sha256:
        raise ValueError("Actual sealed checkpoint SHA differs")
    admission = json.loads(args.checkpoint_admission.read_text())
    validate_admission(admission, checkpoint, registration)
    entry, runner_args = build_command(
        args.protocol,
        args.checkpoint,
        args.checkpoint_sha256,
        args.output,
        registration,
    )
    if (
        record(entry)
        != registration["sources"]["entrypoint_bytes"][str(entry.relative_to(CODE))]
    ):
        raise ValueError("Exact frozen isolated benchmark/driver bytes required")
    specification = importlib.util.spec_from_file_location(
        "registered_protected_evaluation_isolation_v1", launcher_path
    )
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    launcher.verify()
    ledger = launcher.ImportLedger(args.import_report)
    ledger.write(
        "registered_evaluation_admitted",
        protocol=args.protocol,
        adapter=record(__file__),
        registration=record(REGISTRATION),
        payload_admission=record(args.checkpoint_admission),
        checkpoint=checkpoint,
        entry=record(entry),
        runner_arguments=runner_args,
    )
    launcher.activate(ledger)
    status = "failed"
    old_argv = sys.argv
    try:
        if args.preflight_only and args.protocol == "navigation":
            from CrowdSim.humanoid.application.preflight import main as preflight

            status = (
                "passed_preflight"
                if preflight(runner_args) == 0
                else "failed_preflight"
            )
            if status == "failed_preflight":
                raise RuntimeError("Registered navigation resource preflight failed")
        else:
            if args.preflight_only:
                runner_args.append("--preflight-only")
            ledger.phase = "registered_benchmark_entry"
            ledger.observe("scripts." + entry.stem + ".__main__", entry)
            ledger.write(
                "registered_entry_selected",
                path=str(entry),
                sha256=record(entry)["sha256"],
                argv=runner_args,
                execution_boundary="before_runpy_not_an_actual_physical_completion",
            )
            sys.argv = [str(entry), *runner_args]
            runpy.run_path(str(entry), run_name="__main__")
            status = "passed"
    except SystemExit as error:
        status = (
            "passed_system_exit" if error.code in (None, 0) else "failed_system_exit"
        )
        raise
    finally:
        sys.argv = old_argv
        ledger.finish(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
