#!/usr/bin/env python3
"""Durable OS owner of recovery training and its terminal CPU audit.

The detached owner survives a chat/tool connection closing. Child exit codes
come from Popen.wait(), never from logs or PID disappearance. No S3 continuous
execution certificate or locomotion-quality claim is emitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def admit(plan_path):
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "shared_social.durable_recovery_owner_plan.v1" or plan_path.stat().st_mode & 0o222:
        raise ValueError("Require a readonly durable-owner plan")
    for row in plan["inputs"] + plan["sources"]:
        if record(row["path"]) != row:
            raise ValueError("Durable recovery source/input changed: " + row["path"])
    smoke = json.loads(Path(plan["smoke_execution"]).read_text())
    audit = json.loads(Path(plan["smoke_CPU_audit"]).read_text())
    if (smoke["actual_exit_code"] != 0 or type(smoke["actual_exit_code"]) is not int
            or smoke["actual_segment_updates"] != 1 or smoke["cleanup_succeeded"] is not True
            or audit["status"] != "passed" or audit["actual_recovery_segment_updates"] != 1
            or audit["CUDA_initialized"] is not False):
        raise ValueError("Require actual smoke exit0 and complete one-update CPU audit")
    return plan


def run_owner(plan_path):
    plan = admit(plan_path)
    here = plan_path.parent
    state_path = here / "owner_state.json"
    completion_path = here / "completion.json"
    if state_path.exists() or completion_path.exists():
        raise FileExistsError("Refuse duplicate recovery owner or completion overwrite")
    plan_identity = record(plan_path)
    state = {"schema": "shared_social.durable_recovery_owner_state.v1", "status": "running",
             "owner_pid": os.getpid(), "started_at_utc": now(), "plan": plan_identity,
             "steps": [], "original_continuous_S3_run_completed": False,
             "paper_effect_achieved": False, "default_model_promoted": False}
    write_json(state_path, state)

    def save():
        state["updated_at_utc"] = now()
        write_json(state_path, state)

    def child(name, command, cuda, output_log):
        admit(plan_path)
        if record(plan_path) != plan_identity:
            raise ValueError("Durable owner plan changed")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS="4",
                           MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
        step = {"name": name, "command": command, "started_at_utc": now(), "status": "running"}
        state["steps"].append(step)
        save()
        with output_log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            step["pid"] = process.pid
            save()
            exit_code = process.wait()
        step.update(actual_exit_code=exit_code, finished_at_utc=now(), log=record(output_log),
                    status="passed" if exit_code == 0 else "failed", exit_observation="subprocess.Popen.wait")
        save()
        if exit_code != 0:
            raise RuntimeError(f"{name} actual child exit {exit_code}; later stages not started")
        admit(plan_path)
        return step

    try:
        # The plan fixes empty new output, terminal selection and complete budget.
        registration = json.loads(Path(plan["registration"]).read_text())
        run = Path(registration["stages"]["training"]["output"])
        if run.exists():
            raise FileExistsError("Recovery training requires a new output directory")
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True)
        if gpu.stdout.strip():
            raise RuntimeError("Another GPU compute process is present: " + gpu.stdout.strip())
        state["GPU_compute_processes_before_launch"] = []
        child("training", [sys.executable, str(ROOT / "scripts/recover_shared_social_training.py"),
              "--registration", plan["registration"], "--stage", "training"], "0", here / "training_GPU.log")
        result = json.loads((run / "result.json").read_text())
        if (result["status"] != "passed_local_training" or result["completed_updates"] != 12500
                or result["training_update_returned_count"] != 500
                or result["actual_training_transitions"] != 102400000
                or result["cleanup_succeeded"] is not True or result["cleanup_errors"]):
            raise ValueError("Actual recovery training exit/result/budget/cleanup disagree")
        state["training_result"] = record(run / "result.json")
        save()
        audit_path = here / "training_CPU_audit.json"
        child("terminal_CPU_audit", [sys.executable, str(ROOT / "scripts/audit_shared_social_recovery.py"),
              str(run), "--output", str(audit_path)], "", here / "training_CPU_audit.log")
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "passed" or audit["actual_recovery_segment_updates"] != 500:
            raise ValueError("Incomplete terminal recovery CPU audit")
        state.update(status="completed_training_and_CPU_audit", audit=record(audit_path),
                     checkpoint=audit["checkpoint"], retained_updates=12000,
                     actual_segment_updates=500, effective_total_updates=12500,
                     actual_segment_transitions=4096000, effective_training_transitions=102400000,
                     physical_world_restarted=True, actual_child_exit_codes_observed=True)
        # Protect completed outputs without changing any original S3 artifact.
        for path in run.iterdir():
            if path.is_file():
                path.chmod(0o444)
    except BaseException as error:
        state.update(status="failed_preserved", error={"type": type(error).__name__,
                     "message": str(error), "traceback": traceback.format_exc()})
    state["finished_at_utc"] = now()
    save()
    write_json(completion_path, state)
    completion_path.chmod(0o444)
    print(json.dumps({"status": state["status"], "completion": record(completion_path)}), flush=True)
    return 0 if state["status"] == "completed_training_and_CPU_audit" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--detach", action="store_true")
    options.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    admit(plan_path)
    if args.check_plan:
        print(json.dumps({"status": "durable_owner_plan_admitted", "plan": record(plan_path), "SDK_started": False}))
        return 0
    if args.detach:
        if (plan_path.parent / "owner_state.json").exists() or (plan_path.parent / "owner.log").exists():
            raise FileExistsError("Durable owner already launched")
        with (plan_path.parent / "owner.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        print(json.dumps({"status": "durable_owner_launched", "owner_pid": process.pid,
                          "actual_training_exit_observed": False, "plan": record(plan_path)}))
        return 0
    return run_owner(plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
