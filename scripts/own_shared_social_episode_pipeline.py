#!/usr/bin/env python3
"""Own serial episode-candidate smoke, full training and independent evaluation.

Detached OS ownership retains every failure and records actual child exits.
Source: https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen.wait
No inference of exit success from progress, file presence or PID disappearance.
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
from CrowdSim.humanoid.application import episode_serial_release as release  # noqa: E402
from scripts import train_shared_social_episode_after_recovery as training  # noqa: E402
from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as io  # noqa: E402

PLAN_SCHEMA = "shared_social.episode_after_recovery_pipeline_plan.v1"
POLICY = {"minimum_free_memory_MiB": 10240, "external_processes_preserved": True, "managed_GPU_jobs_concurrent_max": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def admit_plan(path, *, historical=False):
    from social_navigation.artifact_identity import artifact_path, same_artifact
    path = artifact_path(path)
    identity = release.recovery.readonly_identity(path)
    plan = json.loads(path.read_text())
    release.recovery.require(plan.get("schema") == PLAN_SCHEMA and plan.get("GPU_resource_policy") == POLICY,
                             "Require the readonly episode candidate serial pipeline plan")
    if historical:
        # This exact retained plan may be read to interpret old outcomes. It
        # cannot authorize execution of current sources through run_owner().
        expected = ROOT / "docs/style_improvement_20261008/shared_social_episode_after_recovery_v4/owner_plan.json"
        release.recovery.require(path == expected and identity["sha256"] ==
            "8e8314cb35d40eee4b095ac6a05fd7601e92d57027ba56a0ba20e155e3f79c4d", "Unknown historical episode plan")
        release_identity = release.recovery.readonly_identity(artifact_path(plan["serial_release"]["path"]))
        release.recovery.require(same_artifact(release_identity, plan["serial_release"]), "Historical serial release changed")
        inputs = plan["inputs"]
    else:
        from CrowdSim.humanoid.application import episode_pipeline_plan
        _, release_identity = release.admit_release(artifact_path(plan["serial_release"]["path"]))
        release.recovery.require(same_artifact(release_identity, plan["serial_release"]) and plan["stages"] == release.stages(),
                                 "Candidate stage/profile/output registration changed")
        inputs = plan["sources"] + plan["inputs"]
        episode_pipeline_plan.validate_inventory(plan)
    for row in inputs:
        release.recovery.require(same_artifact(release.record(artifact_path(row["path"])), row), "Pipeline source/input changed: " + row["path"])
    preflight = json.loads(artifact_path(plan["cold_preflight"]["path"]).read_text())
    release.recovery.require([row.get("stage") for row in preflight.get("steps", [])] ==
                             ["smoke8", "smoke256", "pilot2000", "evaluation"]
                             and all(release.recovery.exact_integer(row.get("actual_exit_code"), 0)
                                     and row.get("SDK_started") is False and row.get("Torch_imported") is False
                                     for row in preflight["steps"]), "Require all actual cold entry preflights")
    if not historical:
        episode_pipeline_plan.validate_preflight(plan, preflight)
    release.recovery.require(release.record(path) == identity, "Pipeline plan changed during admission")
    return plan


def validate_stage(result, audit, selected, binding):
    expected_updates = selected["runtime"]["updates"]
    exposure = expected_updates * selected["runtime"]["actors"] * selected["runtime"]["rollout_controls"]
    release.recovery.require(result.get("status") == "passed_local_training"
                             and result.get("cleanup_succeeded") is True and result.get("cleanup_errors") == []
                             and result.get("training_loop_completed") is True
                             and result.get("episode_after_recovery_execution") == binding
                             and result.get("profile", {}).get("runtime") == selected["runtime"]
                             and release.recovery.exact_integer(result.get("completed_updates"), expected_updates)
                             and release.recovery.exact_integer(result.get("actual_training_transitions"), exposure)
                             and audit.get("status") == "passed" and audit.get("CUDA_initialized") is False
                             and release.recovery.exact_integer(audit.get("updates"), expected_updates)
                             and release.recovery.exact_integer(audit.get("physical_transitions"), exposure),
                             "Candidate real stage result, provenance, budget or CPU audit disagree")


def run_owner(path):
    plan = admit_plan(path)
    here = path.parent
    if any((here / name).exists() for name in ("owner_state.json", "completion.json")):
        raise FileExistsError("Episode pipeline already started")
    outputs = [Path(row["output_directory"]) for row in plan["stages"]] + [evaluation.admission.CURRENT_OUTPUT]
    if any(target.exists() for target in outputs):
        raise FileExistsError("Preserve prior candidate outputs; never restart into an existing path")
    state = {"schema": "shared_social.episode_after_recovery_pipeline_owner_state.v1", "status": "running",
             "owner_pid": os.getpid(), "started_at_utc": now(), "plan": release.record(path), "steps": [],
             "paper_effect_achieved": False, "default_model_promoted": False, "original_S3_certificate_created": False}

    def save():
        state["updated_at_utc"] = now()
        io.write_json(here / "owner_state.json", state)

    def check():
        admit_plan(path)

    def capacity():
        while True:
            applications = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                          "--format=csv,noheader"], capture_output=True, text=True, check=True)
            memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
            free = int(memory.stdout.strip())
            state["GPU_capacity_observation"] = {"observed_at_utc": now(), "free_memory_MiB": free,
                "existing_compute_processes": applications.stdout.strip().splitlines(), "existing_processes_terminated": False}
            if free >= POLICY["minimum_free_memory_MiB"]:
                state["status"] = "running"
                save()
                return
            state["status"] = "waiting_for_GPU_capacity"
            save()
            time.sleep(30)
            check()

    def child(name, cuda, command, result_path, expected_status):
        check()
        if cuda:
            capacity()
        row = {"name": name, "status": "running", "command": command, "started_at_utc": now()}
        state["steps"].append(row)
        save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, PYTHONDONTWRITEBYTECODE="1",
                           PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log = here / (name + ".log")
        with log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            save()
            code = process.wait()
        row.update(actual_exit_code=code, exit_observation="subprocess.Popen.wait", finished_at_utc=now(), log=release.record(log))
        save()
        release.recovery.require(release.recovery.exact_integer(code, 0), f"Actual {name} exit {code}; later stages stopped")
        document = json.loads(result_path.read_text())
        release.recovery.require(document.get("status") == expected_status, "Child exit and result disagree: " + name)
        row.update(result=release.record(result_path), status="passed")
        save()
        check()
        return document, row

    save()
    try:
        # Completed S3-R owner records actual exits. Explicit host observation
        # also rejects any still-live recorded managed predecessor process.
        previous = json.loads(Path(plan["serial_release"]["path"]).read_text())
        prior_states = [json.loads(Path(previous[key]["path"]).read_text()) for key in
                        ("recovery_training_completion", "completed_evaluation")]
        previous_pids = {item["owner_pid"] for item in prior_states}
        previous_pids.update(row["pid"] for item in prior_states for row in item["steps"])
        observation = subprocess.run(["ps", "-p", ",".join(map(str, sorted(previous_pids))), "-o", "pid=,args="],
                                     capture_output=True, text=True, check=False)
        release.recovery.require(observation.returncode in (0, 1) and not observation.stdout.strip(),
                                 "A recorded managed predecessor PID is live; preserve it and stop")
        state["managed_predecessor_terminal_observation"] = {"pids": sorted(previous_pids), "stdout": observation.stdout,
                                                              "actual_ps_exit_code": observation.returncode}
        for selected in plan["stages"]:
            stage, output = selected["stage"], Path(selected["output_directory"])
            command = [sys.executable, str(Path(training.__file__)), "--serial-release", plan["serial_release"]["path"], "--stage", stage]
            result, training_step = child(stage + "_training", "0", command, output / "result.json", "passed_local_training")
            audit_path = here / (stage + "_CPU_audit.json")
            audit, audit_step = child(stage + "_CPU_audit", "", [sys.executable, str(ROOT / "scripts/audit_shared_social_popart.py"),
                str(output), "--output", str(audit_path)], audit_path, "passed")
            _, binding, _ = training.prepare_training(plan["serial_release"]["path"], stage)
            validate_stage(result, audit, selected, binding)
            provenance = json.loads((output / "resolved_configuration.json").read_text())["provenance"]
            release.recovery.require(provenance.get("episode_after_recovery_execution") == binding,
                                     "Actual saved checkpoint provenance lacks the execution release")
            for artifact in output.rglob("*"):
                if artifact.is_file() and not artifact.is_symlink():
                    artifact.chmod(0o444)
            audit_path.chmod(0o444)
            if stage == "pilot2000":
                training_completion = here / "training_completion.json"
                completed = {"schema": "shared_social.episode_after_recovery_training_completion.v1",
                    "status": "completed_fresh_training_and_CPU_audits", "recorded_at_utc": now(),
                    "plan": state["plan"], "steps": state["steps"], "result": training_step["result"],
                    "audit": audit_step["result"], "checkpoint": audit["checkpoint"],
                    "fresh_initialization": True, "paper_effect_achieved": False, "default_model_promoted": False}
                evaluation.validate_training_owner(completed)
                io.write_json(training_completion, completed, readonly=True)
                receipt = here / "training_execution.json"
                io.write_json(receipt, {"schema": "shared_social.actual_training_execution.v1", "terminal_exit_code": training_step["actual_exit_code"],
                    "run_directory": str(output), "result_identity": training_step["result"], "audit_identity": audit_step["result"],
                    "checkpoint_identity": audit["checkpoint"], "episode_training_owner_completion": release.record(training_completion)}, readonly=True)
                state["training_completion"] = release.record(training_completion)
                save()
        pipeline, evaluation_step = child("independent_evaluation", "0", [sys.executable,
            str(Path(evaluation.__file__)), "--serial-release", plan["serial_release"]["path"],
            "--execution-receipt", str(receipt)], evaluation.admission.CURRENT_OUTPUT / "execution.json", "passed_execution")
        io.successful_steps(pipeline, ["actor_CPU_inference", "tracking_style", "navigation_smoke", "navigation200s",
                                       "navigation_CPU_array_audit"], "passed_execution")
        state.update(status="completed_training_and_independent_evaluation_quality_unclassified",
                     evaluation_pipeline=evaluation_step["result"], measurements=pipeline["measurements"])
        for artifact in evaluation.admission.CURRENT_OUTPUT.rglob("*"):
            if artifact.is_file() and not artifact.is_symlink():
                artifact.chmod(0o444)
    except BaseException as error:
        if state["steps"] and state["steps"][-1]["status"] != "passed":
            state["steps"][-1]["status"] = "failed"
        state.update(status="failed_preserved", error={"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
    state["finished_at_utc"] = now()
    save()
    io.write_json(here / "completion.json", state, readonly=True)
    print(json.dumps({"status": state["status"], "completion": release.record(here / "completion.json")}), flush=True)
    return 0 if state["status"] == "completed_training_and_independent_evaluation_quality_unclassified" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    path = args.plan.resolve()
    admit_plan(path)
    if args.check_plan:
        print(json.dumps({"status": "episode_serial_pipeline_plan_admitted", "SDK_started": False}))
        return 0
    if args.detach:
        if any((path.parent / name).exists() for name in ("owner_state.json", "owner.log", "completion.json")):
            raise FileExistsError("Episode pipeline already launched")
        with (path.parent / "owner.log").open("xb") as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--plan", str(path)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        print(json.dumps({"status": "episode_serial_pipeline_launched", "owner_pid": process.pid, "actual_training_exit_observed": False}))
        return 0
    return run_owner(path)


if __name__ == "__main__":
    raise SystemExit(main())
