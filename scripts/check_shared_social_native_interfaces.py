#!/usr/bin/env python3
"""Check all registered N/seed/planner CLI interfaces without NN or an SDK.

These actual CPU child executions validate executable source derivations only.
They do not load the future episode checkpoint or certify native simulation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_native_execution as bridge  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as io  # noqa: E402
from social_navigation.external_runtime import selected_python  # noqa: E402

ENTRY = ROOT / "scripts/evaluate_shared_social_native_seeded.py"
PYTHON = selected_python(ROOT)


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(here, actors, seed, planner):
    stem = f"N{actors:02d}_seed{seed}_{planner}"
    command = [str(PYTHON), str(ENTRY), "--actors", str(actors), "--planner", planner,
               "--seed", str(seed), "--check-sources"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    started = now()
    stdout, stderr = here / (stem + ".stdout.json"), here / (stem + ".stderr")
    with stdout.open("xb") as output, stderr.open("xb") as errors:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=errors)
        code = child.wait()
    observed = {"actors": actors, "seed": seed, "planner": planner, "command": command,
        "pid": child.pid, "actual_exit_code": code, "exit_observation": "subprocess.Popen.wait",
        "started_at_utc": started, "finished_at_utc": now(),
        "stdout": bridge.record(stdout), "stderr": bridge.record(stderr), "status": "failed_preserved"}
    try:
        if type(code) is int and code == 0:
            report = json.loads(stdout.read_text())
            migration = report["migration"]
            lifecycle = migration["native_lifecycle"]
            seed_proof = migration["registered_native_seed"]
            bridge.require(report.get("status") == "native_source_checks_passed"
                and report.get("Torch_imported") is False and report.get("SDK_started") is False
                and migration["entry"] == bridge.record(ENTRY) and migration["planner"] == planner
                and type(lifecycle["navigation_actor_count"]) is int and lifecycle["navigation_actor_count"] == actors
                and lifecycle["expected_owned_contact_views"] == actors + 1
                and lifecycle["expected_SDK_sensor_objects"] == 24
                and type(seed_proof["seed"]) is int and seed_proof["seed"] == seed
                and seed_proof["extra_reseed_in_capture_hook"] is False
                and seed_proof["main_derivation"]["complete_inverse"] is True
                and seed_proof["outer_CLI_derivation"]["complete_inverse"] is True
                and report["physical_derivation"]["complete_inverse"] is True
                and report["physical_derivation"]["existing_control_or_metric_math_changed"] is False,
                "Actual CPU CLI report disagrees with its registered interface")
            observed.update(status="passed_source_interface", Torch_imported=False, SDK_started=False,
                            expected_owned_contact_views=actors + 1, expected_SDK_sensor_objects=24)
    except Exception as error:
        observed["report_admission_error"] = {"type": type(error).__name__, "message": str(error)}
    stdout.chmod(0o444)
    stderr.chmod(0o444)
    return observed


def identities(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    here = args.output.resolve()
    bridge.require(not here.exists(), "Preserve previous source-interface checks; use a fresh output")
    registered, registration = bridge.admit_registration()
    bridge.require(registered["seeds"] == [7, 19, 42], "Preserve all three registered seeds")
    source_bindings = bridge.original.raw.entry.source_bindings()
    sources = {row["path"]: row for row in identities([registered["sources"], source_bindings])}
    sources[str(Path(__file__).resolve())] = bridge.record(Path(__file__))
    sources = [sources[path] for path in sorted(sources)]
    here.mkdir(parents=True)
    report = {"schema": "shared_social.actual_native_source_interface_sweep.v1", "status": "running",
        "registration": registration, "source_bindings": source_bindings, "sources": sources,
        "started_at_utc": now(), "planned_interfaces": 78, "steps": [],
        "actors": list(range(3, 16)), "seeds": registered["seeds"], "planners": ["diffusion", "sfm"],
        "Torch_imported_by_owner": "torch" in sys.modules, "SDK_started": False,
        "episode_checkpoint_loaded": False, "native_physical_execution_verified": False,
        "author_eight_scene_protocol_recovered": False, "paper_effect_achieved": False}
    jobs = [(actors, seed, planner) for actors in range(3, 16) for seed in registered["seeds"] for planner in report["planners"]]
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [workers.submit(execute, here, *job) for job in jobs]
            for future in as_completed(pending):
                report["steps"].append(future.result())
                io.write_json(here / "progress.json", report)
        report["steps"].sort(key=lambda row: (row["actors"], row["seed"], row["planner"]))
        bridge.require(bridge.original.raw.entry.source_bindings() == source_bindings
            and all(bridge.record(row["path"]) == row for row in sources)
            and bridge.record(registration["path"]) == registration, "Registered sources changed during actual CPU checks")
        bridge.require([(row["actors"], row["seed"], row["planner"]) for row in report["steps"]] == jobs,
                       "Require every registered N/seed/planner interface exactly once")
        report.update(status="passed_all_source_interfaces" if all(row["status"] == "passed_source_interface" for row in report["steps"]) else "failed_preserved",
                      source_before_after_equal=True)
    except BaseException as error:
        report.update(status="failed_preserved", error={"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
    report.update(finished_at_utc=now(), actual_interfaces=len(report["steps"]))
    io.write_json(here / "result.json", report, readonly=True)
    print(json.dumps({"status": report["status"], "actual_interfaces": len(report["steps"]),
                      "result": bridge.record(here / "result.json"), "SDK_started": False}))
    return 0 if report["status"] == "passed_all_source_interfaces" else 1


if __name__ == "__main__":
    raise SystemExit(main())
