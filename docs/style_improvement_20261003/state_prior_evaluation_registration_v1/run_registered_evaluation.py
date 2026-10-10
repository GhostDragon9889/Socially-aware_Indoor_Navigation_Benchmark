"""Execute only a registered full22 or200s protocol from isolated frozen code.

Actual execution requires the new934 immutable source, sealed2000 checkpoint
and the separate public-state-prior terminal full-payload admission. Preparing
or parsing a protocol never admits a future checkpoint or motion quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
DOCS = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
CODE = BUNDLE / "code"
REGISTRATION = DOCS / "registration.json"


REGISTRATION_SHA = "44cd49a57b58a3237406929e8ca4e2cda52ed0749b915307bd35f3edec39a9ca"


def record(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate_registered_dependencies(registration):
    """Bind the reader and its helper to the independent source freeze."""
    sources = registration["sources"]
    freeze_record = sources["state_prior_auditor_source_freeze"]
    if record(freeze_record["path"]) != freeze_record:
        raise ValueError("Independent auditor source freeze changed")
    freeze = json.loads(Path(freeze_record["path"]).read_text())
    files = freeze.get("source_files", [])
    if (
        freeze.get("schema") != "public_state_prior.independent_auditor_source_freeze.v1"
        or freeze.get("status") != "frozen"
        or freeze.get("source_manifest") != sources["isolated_source_manifest"]
        or not files
        or sources["state_prior_auditor"] not in files
        or len({row["path"] for row in files}) != len(files)
        or any(record(row["path"]) != row for row in files)
        or any(Path(row["path"]).stat().st_mode & 0o222 for row in files)
    ):
        raise ValueError("Independent frozen reader/dependencies/new934 source differ")


def validate_admission(document, checkpoint, registration=None):
    """No checkpoint training counters are inferred from its filename."""
    if (
        document.get("schema")
        != "public_state_prior.actual_training_checkpoint_audit.v1"
        or document.get("status") != "passed"
        or document.get("CPU_only") is not True
        or document.get("GPU_used") is not False
        or document.get("SDK_started") is not False
        or document.get("sealed34_outputs_read") is not False
        or document.get("protected_reference_policy_validated") is not True
        or document.get("actual_terminal_fresh2000_confirmed") is not True
        or document.get("phase") != "terminal2000"
        or document.get("milestone_only_admission") is not False
        or document.get("state_prior_initialization_validated") is not True
        or document.get("public_D_or_AMP_RMS_transferred") is not True
        or not isinstance(document.get("checks"), list)
        or not document["checks"]
        or not all(isinstance(row, dict) and row.get("passed") is True for row in document["checks"])
        or type(document.get("check_count")) is not int
        or document.get("check_count") != len(document["checks"])
        or type(document.get("copied_files_hashed")) is not int
        or document.get("copied_files_hashed") != 934
        or not isinstance(document.get("checkpoints"), list)
        or len(document["checkpoints"]) != 1
        or not isinstance(document["checkpoints"][0], dict)
        or not isinstance(document.get("terminal_training_result"), dict)
        or not isinstance(document.get("actual_closed_metrics"), dict)
    ):
        raise ValueError("Independent actual terminal state-prior payload admission required")
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
        or rows[0].get("public_D_or_AMP_RMS_transferred") is not True
        or rows[0].get("state_prior_initialization_validated") is not True
        or document.get("actual_observed_owned_imports_isolated") is not True
    ):
        raise ValueError(
            "Sealed checkpoint/hash/count/D/resume/source admission differs"
        )
    state_prior = rows[0].get("state_prior", {})
    expected_prior = {
        "explicit_flag": True,
        "strategy": "public_pacer_state_prior_zero_action_extension",
        "published_state_only_discriminator_transferred": True,
        "published_amp_history_moments_transferred": True,
        "public_AMP_count": 15482880001.0,
        "current_AMP_count": 24776000.0,
        "count_cap": 200000.0,
        "initial_action_columns_zero_is_provenance_only": True,
        "trained_active_columns_nonzero": 57,
        "frozen_action_columns_zero": 12,
    }
    if not isinstance(state_prior, dict) or any(type(state_prior.get(name)) is not type(value) or state_prior[name] != value for name, value in expected_prior.items()):
        raise ValueError("Independent coupled state-prior D/history-RMS/count provenance required")
    if rows[0].get("RMS_counts", {}).get("amp_normalizer") != 24776000.0:
        raise ValueError("Actual terminal AMP RMS count differs from coupled prior/update budget")
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
            "Independent payload admission refers to a different isolated934 source manifest"
        )
    if registration is not None and document.get("auditor") != registration["sources"]["state_prior_auditor"]:
        raise ValueError("Admission was not produced by the independently frozen state-prior reader")
    return rows[0]


def build_command(protocol, checkpoint_path, checkpoint_sha256, output, registration):
    if protocol not in {"tracking", "navigation"}:
        raise ValueError("Only registered tracking or navigation protocols")
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise ValueError("Actual64-character lowercase sealed checkpoint SHA required")
    if (
        Path(checkpoint_path).absolute()
        != Path(registration["future_checkpoint"]["path"]).absolute()
    ):
        raise ValueError(
            "Only the preregistered immutable step2000 checkpoint is accepted"
        )
    section = registration[protocol]
    if Path(output).absolute() != Path(section["parsed"]["output"]).absolute():
        raise ValueError("Only the fixed new registered output path is accepted")
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
        != "prepared_protocol_and_isolated_parsers_passed_future_terminal_pending"
    ):
        raise ValueError("Unknown original protocol registration")
    validate_registered_dependencies(registration)
    # Reject mutable or unregistered aliases before reading any checkpoint.
    entry, runner_args = build_command(
        args.protocol, args.checkpoint, args.checkpoint_sha256, args.output, registration
    )
    if str(args.checkpoint_admission.absolute()) != registration["future_checkpoint"]["terminal_admission_path"]:
        raise ValueError("Only the new registered terminal admission path is accepted")
    expected_report = BUNDLE / ("state_prior_" + args.protocol + "2000_v1_GPU_imports.json")
    if args.import_report.absolute() != expected_report or expected_report.exists() or expected_report.is_symlink():
        raise ValueError("Fresh registered evaluation import-report path required")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Fresh exclusive output only; no existing directory")
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
    record(args.checkpoint_admission)
    admission = json.loads(args.checkpoint_admission.read_text())
    validate_admission(admission, checkpoint, registration)
    if (
        record(entry)
        != registration["sources"]["entrypoint_bytes"][str(entry.relative_to(CODE))]
    ):
        raise ValueError("Exact frozen isolated benchmark/driver bytes required")
    specification = importlib.util.spec_from_file_location(
        "registered_state_prior_evaluation_isolation_v1", launcher_path
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
            # Guarded import observes module-definition execution. The runpy
            # selection below does not claim physical completion.
            imported_entry = importlib.import_module("scripts." + entry.stem)
            if Path(imported_entry.__file__).resolve() != entry.resolve():
                raise ImportError("Registered entry import escaped isolated code")
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
