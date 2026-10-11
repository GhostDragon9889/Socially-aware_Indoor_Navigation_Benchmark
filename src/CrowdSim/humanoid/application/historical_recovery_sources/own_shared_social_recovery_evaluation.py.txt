#!/usr/bin/env python3
"""Own the real recovered-model evaluation independently of the chat session.

The original post-training pipeline owns every measurement child's real exit
and cleanup. This outer OS owner records the pipeline exit via Popen.wait(),
keeps failures, and never infers quality from execution success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import run_shared_social_budget_evaluation as budget  # noqa: E402

SCHEMA = "shared_social.durable_recovery_evaluation_owner_plan.v1"
POLICY = {"minimum_free_memory_MiB": 4096, "preserve_existing_processes": True}


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
    admission.readonly_identity(plan_path)
    plan = json.loads(plan_path.read_text())
    admission.require(plan.get("schema") == SCHEMA and plan.get("GPU_resource_policy") == POLICY,
                      "Require an explicit readonly recovered-model evaluation plan")
    for item in plan["sources"] + plan["inputs"]:
        admission.require(record(item["path"]) == item, "Evaluation owner source/input changed: " + item["path"])
    checkpoint, _, accepted = admission.admit_training(Path(plan["training_receipt"]))
    registration = json.loads(Path(plan["registration"]).read_text())
    config = json.loads(Path(registration["configuration"]["path"]).read_text())
    budget.registered_budget(registration, config)
    admission.require(registration["run_directory"] == str(checkpoint.parent)
                      and registration["preselected_checkpoint"] == checkpoint.name
                      and registration.get("recovery_execution_admission") == accepted
                      and registration.get("evaluation_output") == plan["evaluation_output"],
                      "Evaluation plan does not select the complete recovered terminal model")
    preflight = json.loads(Path(plan["preflight_execution"]).read_text())
    admission.require(preflight.get("status") == "passed" and preflight.get("SDK_started") is False
                      and [step.get("stage") for step in preflight.get("steps", [])] == ["tracking", "navigation"]
                      and all(admission.exact_integer(step.get("actual_exit_code"), 0)
                              for step in preflight["steps"]),
                      "Require actual cold preflights for tracking and navigation")
    return plan


def run_owner(plan_path):
    plan = admit(plan_path)
    here = plan_path.parent
    output = Path(plan["evaluation_output"])
    state_path, completion = here / "owner_state.json", here / "completion.json"
    if output.exists() or state_path.exists() or completion.exists():
        raise FileExistsError("Refuse duplicate evaluation or overwrite of prior results")
    identity = record(plan_path)
    state = {"schema": "shared_social.durable_recovery_evaluation_owner_state.v1", "status": "running",
             "owner_pid": os.getpid(), "started_at_utc": now(), "plan": identity, "steps": [],
             "paper_effect_achieved": False, "default_model_promoted": False,
             "original_continuous_S3_run_completed": False}

    def save():
        state["updated_at_utc"] = now()
        write_json(state_path, state)

    save()
    try:
        while True:
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                  "--format=csv,noheader"], capture_output=True, text=True, check=True)
            memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
            free = int(memory.stdout.strip())
            state["GPU_capacity_observation"] = {"observed_at_utc": now(), "free_memory_MiB": free,
                "existing_compute_processes": gpu.stdout.strip().splitlines(), "policy": POLICY,
                "existing_processes_terminated": False}
            if free >= POLICY["minimum_free_memory_MiB"]:
                state["status"] = "running"
                save()
                break
            state["status"] = "waiting_for_GPU_capacity"
            save()
            time.sleep(30)
            admit(plan_path)
        command = [sys.executable, str(ROOT / "scripts/evaluate_shared_social_recovery.py"), "--stage", "post",
                   "--execution-receipt", plan["training_receipt"], "--registration", plan["registration"],
                   "--output", str(output)]
        step = {"name": "post_training_evaluation", "command": command,
                "started_at_utc": now(), "status": "running"}
        state["steps"].append(step)
        save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log = here / "evaluation.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            step["pid"] = process.pid
            save()
            code = process.wait()
        step.update(actual_exit_code=code, exit_observation="subprocess.Popen.wait", finished_at_utc=now(),
                    status="passed" if code == 0 else "failed", log=record(log))
        save()
        if code != 0:
            raise RuntimeError(f"Independent evaluation pipeline actual exit {code}; outputs retained")
        execution = output / "execution.json"
        report = json.loads(execution.read_text())
        admission.require(report.get("status") == "passed_execution"
                          and [row.get("name") for row in report.get("steps", [])] ==
                          ["actor_CPU_inference", "tracking_style", "navigation_smoke", "navigation200s",
                           "navigation_CPU_array_audit"]
                          and all(admission.exact_integer(row.get("actual_exit_code"), 0)
                                  and row.get("status") == "passed" for row in report["steps"]),
                          "Pipeline exit, complete stages and actual measurement exits disagree")
        for measurement in report["measurements"].values():
            admission.require(record(measurement["path"]) == measurement, "Completed measurement changed")
        admit(plan_path)
        admission.require(record(plan_path) == identity, "Evaluation plan changed during execution")
        for path in output.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)
        state.update(status="completed_evaluation_quality_unclassified", execution=record(execution),
                     measurements=report["measurements"], actual_pipeline_exit_observed=True)
    except BaseException as error:
        state.update(status="failed_preserved", error={"type": type(error).__name__,
                     "message": str(error), "traceback": traceback.format_exc()})
    state["finished_at_utc"] = now()
    save()
    write_json(completion, state)
    completion.chmod(0o444)
    print(json.dumps({"status": state["status"], "completion": record(completion)}), flush=True)
    return 0 if state["status"] == "completed_evaluation_quality_unclassified" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    admit(plan_path)
    if args.check_plan:
        print(json.dumps({"status": "evaluation_owner_plan_admitted", "plan": record(plan_path), "SDK_started": False}))
        return 0
    if args.detach:
        if any((plan_path.parent / name).exists() for name in ("owner_state.json", "owner.log", "completion.json")):
            raise FileExistsError("Evaluation owner already launched")
        with (plan_path.parent / "owner.log").open("xb") as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        print(json.dumps({"status": "evaluation_owner_launched", "owner_pid": process.pid,
                          "actual_pipeline_exit_observed": False, "plan": record(plan_path)}))
        return 0
    return run_owner(plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
