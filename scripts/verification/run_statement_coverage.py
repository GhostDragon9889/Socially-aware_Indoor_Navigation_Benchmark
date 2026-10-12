"""Measure all owned statements through the configured SDK, with a strict gate.

The test owner keeps CPU and CUDA separate. Startup hooks live in the output
directory; this command never installs a hook into the SDK's site-packages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / ".social-navigation-root").is_file()
)
sys.path.insert(0, str(ROOT / "src"))

from social_navigation.coverage_audit import inventory, report  # noqa: E402 - initialize checkout imports first
from social_navigation.external_runtime import selected_python  # noqa: E402 - initialize checkout imports first


def _instrumentation(
    root: Path, output: Path, coverage, *, ignore_isaacgym: bool = False
) -> tuple[dict, Path, Path, dict]:
    baseline = inventory(root, ignore_isaacgym=ignore_isaacgym)
    (output / "source_before.json").write_text(json.dumps(baseline, indent=2) + "\n")
    startup, data_dir = output / "startup", output / "data"
    startup.mkdir()
    data_dir.mkdir()
    # Expose only coverage to host launcher processes, not the SDK's other
    # packages. The host and SDK must use the supported Python 3.12 ABI.
    (startup / "coverage").symlink_to(
        Path(coverage.__file__).parent, target_is_directory=True
    )
    (startup / "sitecustomize.py").write_text(
        'import os\nif os.environ.get("COVERAGE_PROCESS_START"):\n'
        "    import atexit, coverage\n    owner = coverage.process_startup() or coverage.Coverage.current()\n"
        "    if owner is not None:\n"
        "        def save_project_coverage():\n"
        "            owner.stop()\n            owner.save()\n"
        "        atexit.register(save_project_coverage)\n"
    )
    config = output / "coverage.ini"
    config.write_text(
        "[run]\nbranch = true\nparallel = true\n"
        "plugins = social_navigation.coverage_sources\n"
        "concurrency = multiprocessing,thread\nsigterm = true\npatch = _exit,execv\n"
        "data_file = "
        + str(data_dir / ".coverage")
        + "\ninclude =\n"
        # Trace the checkout with one prefix, rather than match
        # hundreds of filenames at every call. The Git inventory
        # in report() remains the authoritative denominator.
        + "    "
        + str(root)
        + "/*\nomit =\n"
        + "".join(
            "    " + str(root / name) + "/*\n"
            for name in (".git", ".runtime", "output", "tests")
        )
        + "    "
        + str(root)
        + "/*/tests/*\n"
        + "\n[report]\nexclude_lines =\npartial_branches =\n"
        + "\n[social_navigation.coverage_sources]\nroot = "
        + str(root)
        + "\nbaseline = "
        + str(output / "source_before.json")
        + "\nproof_dir = "
        + str(output / "virtual_frame_proofs")
        + "\n"
    )
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        COVERAGE_PROCESS_START=str(config),
        COVERAGE_RCFILE=str(config),
    )
    inherited = os.environ.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(startup) + (
        os.pathsep + inherited if inherited else ""
    )
    return baseline, config, data_dir, environment


def run(
    root: Path,
    output: Path,
    targets: list[str],
    *,
    cpu_only=False,
    ignore_isaacgym: bool = False,
) -> dict:
    import coverage

    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "social_navigation.coverage_execution.v1",
        "SDK_modified": False,
        "ignored_backends": ["isaacgym"] if ignore_isaacgym else [],
        "actual_test_returncode": None,
        "status": "running",
    }
    receipt = output / "result.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    try:
        baseline, config, data_dir, environment = _instrumentation(
            root, output, coverage, ignore_isaacgym=ignore_isaacgym
        )
        command = [
            str(selected_python(root)),
            str(root / "scripts/run_external_python.py"),
            "--sdk-child",
            str(root / "scripts/run_project_tests.py"),
            "--output-dir",
            str(output / "tests"),
        ]
        if cpu_only:
            command.append("--cpu-only")
        if ignore_isaacgym:
            command.extend(
                ["--", "--ignore=src/protomotions/tests/test_isaacgym_friction_mode.py"]
            )
        command.extend(targets)
        result["command"] = command
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        with (output / "test_owner.log").open("w") as stream:
            child = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        result["actual_test_returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(
                "Failed instrumented test execution; diagnostic outputs retained"
            )
        cov = coverage.Coverage(config_file=str(config))
        cov.combine(data_paths=[str(data_dir)], strict=True, keep=True)
        cov.save()
        measured = report(
            root,
            data_dir / ".coverage",
            baseline=baseline,
            ignore_isaacgym=ignore_isaacgym,
        )
        (output / "coverage.json").write_text(json.dumps(measured, indent=2) + "\n")
        result.update(
            statement_percent=measured["statement_percent"],
            source_files=measured["source_files"],
            missing_statements=measured["missing_statements"],
            gate_passed=measured["gate_passed"],
            source_stability_verified=measured["source_stability_verified"],
        )
        result["status"] = "passed" if measured["gate_passed"] else "coverage_gap"
    except BaseException as error:
        result.update(
            status="failed_preserved", error=type(error).__name__ + ": " + str(error)
        )
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--ignore-isaacgym",
        action="store_true",
        help="Explicitly waive the dedicated IsaacGym backend and its hardware tests",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Record missing statements without passing the 100%% gate",
    )
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args(argv)
    result = run(
        ROOT,
        args.output_dir,
        args.targets,
        cpu_only=args.cpu_only,
        ignore_isaacgym=args.ignore_isaacgym,
    )
    print(json.dumps(result, indent=2))
    return int(not args.report_only and not result["gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
