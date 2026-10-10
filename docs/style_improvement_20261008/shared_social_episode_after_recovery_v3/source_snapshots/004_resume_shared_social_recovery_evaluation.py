#!/usr/bin/env python3
"""Resume failed S3-R evaluation without rewriting its successful prefix.

The retained actor, full22/style and navigation smoke exits are admitted from
their actual failed pipeline. A new OS owner runs navigation200s and the original
CPU array audit once. The combined completion explicitly spans two executions;
it never claims the original post-training process exited successfully.
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
from scripts import compare_shared_social_evidence as original  # noqa: E402
from scripts import evaluate_shared_social_recovery as evaluation  # noqa: E402
from scripts import own_shared_social_recovery_evaluation as previous_owner  # noqa: E402
from scripts import run_shared_social_post_training as post  # noqa: E402

PLAN_SCHEMA = "shared_social.resumed_recovery_evaluation_plan.v1"
COMPLETION_SCHEMA = "shared_social.resumed_recovery_evaluation_completion.v1"
PREFIX_NAMES = ["actor_CPU_inference", "tracking_style", "navigation_smoke"]
RETRY_NAMES = ["navigation200s", "navigation_CPU_array_audit"]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document, *, readonly=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if readonly:
        path.chmod(0o444)


def successful_steps(document, names, status):
    admission.require(document.get("status") == status
                      and [row.get("name") for row in document.get("steps", [])] == names
                      and all(row.get("status") == "passed"
                              and admission.exact_integer(row.get("actual_exit_code"), 0)
                              for row in document["steps"]),
                      "Require ordered complete stages with actual integer-zero exits")


def admit_prefix(completion_path, inventory_path):
    """Admit this failed family's actual successful prefix, never its navigation."""
    completion_identity = admission.readonly_identity(completion_path)
    completed = original.read(completion_path)
    admission.require(completed.get("schema") == "shared_social.durable_recovery_evaluation_owner_state.v1"
                      and completed.get("status") == "failed_preserved", "Require the preserved failed owner")
    steps = completed.get("steps", [])
    admission.require(len(steps) == 1 and steps[0].get("name") == "post_training_evaluation"
                      and admission.exact_integer(steps[0].get("actual_exit_code"), 1)
                      and steps[0].get("exit_observation") == "subprocess.Popen.wait",
                      "Require the actual failed post-training owner exit")
    admission.require(record(steps[0]["log"]["path"]) == steps[0]["log"], "Failed owner log changed")
    plan_path = Path(completed["plan"]["path"])
    admission.require(admission.readonly_identity(plan_path) == completed["plan"], "Failed owner plan changed")
    plan = previous_owner.admit(plan_path)
    inventory_identity = admission.readonly_identity(inventory_path)
    inventory = original.read(inventory_path)
    admission.require(inventory.get("schema") == "shared_social.failed_recovery_evaluation_retained_outputs.v1"
                      and inventory.get("owner_completion") == completion_identity,
                      "Retained inventory belongs to another failed owner")
    pipeline_identity = inventory["pipeline_execution"]
    admission.require(Path(pipeline_identity["path"]) == Path(plan["evaluation_output"]) / "execution.json"
                      and admission.readonly_identity(pipeline_identity["path"]) == pipeline_identity,
                      "Failed pipeline identity changed")
    pipeline = original.bound(pipeline_identity)
    rows = pipeline.get("steps", [])
    admission.require(pipeline.get("status") == "failed"
                      and [row.get("name") for row in rows] == [*PREFIX_NAMES, "navigation200s"]
                      and rows[-1].get("status") == "failed"
                      and admission.exact_integer(rows[-1].get("actual_exit_code"), -11),
                      "Only the retained navigation SIGSEGV family can resume here")
    successful_steps({"status": "prefix", "steps": rows[:3]}, PREFIX_NAMES, "prefix")
    for item in pipeline["sources"] + pipeline["inputs"]:
        admission.require(record(item["path"]) == item, "Failed pipeline source/input changed")
    module, _ = evaluation.prepare("post")
    _, _, expected = module.admit_training(Path(plan["training_receipt"]))
    admission.require(pipeline["training_execution_admission"] == expected,
                      "Failed prefix admits a different training completion")
    actor, tracking, smoke = [original.bound(row["result"]) for row in rows[:3]]
    admission.require(actor.get("status") == "passed" and smoke.get("status") == "passed"
                      and smoke.get("actual_control_steps") == 30
                      and smoke.get("cleanup_succeeded") is True and smoke.get("cleanup_errors") == [],
                      "Actual actor or navigation smoke result/cleanup disagrees")
    successful_steps(tracking, ["lifecycle_smoke", "full22", "style"], "passed_execution")
    measurements = {"tracking": tracking["steps"][1]["result"], "style90": tracking["steps"][2]["result"]}
    style = original.bound(measurements["style90"])
    measurements["pose45"] = style["frozen_pose45_result"]
    tracked = original.bound(measurements["tracking"])
    admission.require(tracked.get("status") == "passed" and tracked.get("cleanup_succeeded") is True
                      and tracked.get("cleanup_errors") == [], "Full22 exit/result/cleanup disagrees")
    for stage, document in (("actor_cpu", actor), ("tracking_style", tracking), ("tracking", tracked)):
        module, _ = evaluation.prepare(stage)
        _, _, expected_stage = module.admit_training(Path(plan["training_receipt"]))
        admission.require(document.get("training_execution_admission") == expected_stage,
                          "Retained prefix stage admits a different checkpoint: " + stage)
    admission.require(tracking["training_receipt"] == expected["receipt"], "Retained tracking receipt changed")
    admission.require(style.get("status") == "passed" and original.bound(measurements["pose45"])["status"] == "passed",
                      "Retained style/pose result disagrees with successful exits")
    retained = []
    for item in inventory["output_inventory"]:
        path = Path(item["path"])
        admission.require(path.is_relative_to(Path(plan["evaluation_output"]))
                          and path.stat().st_size == item["bytes"] and item.get("readonly") is True,
                          "Invalid retained output inventory item")
        retained.append(admission.readonly_identity(path))
    admission.require({row["path"] for row in retained} ==
                      {str(path) for path in Path(plan["evaluation_output"]).rglob("*") if path.is_file()},
                      "Retained output inventory is incomplete")
    return {"failed_completion": completion_identity, "failed_pipeline": pipeline_identity,
            "inventory": inventory_identity, "retained_outputs": retained, "successful_prefix_steps": rows[:3],
            "measurements": measurements, "training_receipt": plan["training_receipt"],
            "training_execution_admission": expected, "original_pipeline_actual_exit_code": 1,
            "original_navigation_actual_exit_code": -11}


def admit_plan(path):
    admission.readonly_identity(path)
    plan = original.read(path)
    admission.require(plan.get("schema") == PLAN_SCHEMA and plan.get("GPU_resource_policy") == previous_owner.POLICY,
                      "Require an explicit readonly resumption plan")
    prefix = admit_prefix(Path(plan["prefix"]["failed_completion"]["path"]), Path(plan["prefix"]["inventory"]["path"]))
    admission.require(prefix == plan["prefix"], "Retained prefix changed after registration")
    for item in plan["sources"]:
        admission.require(record(item["path"]) == item, "Resumption source changed: " + item["path"])
    admission.require(record(post.DIFFUSION) == plan["diffusion"]
                      and plan["diffusion"]["sha256"] == post.DIFFUSION_SHA, "Formal Diffusion40k changed")
    return plan


def admit_completion(path):
    identity = admission.readonly_identity(path)
    completed = original.read(path)
    admission.require(completed.get("schema") == COMPLETION_SCHEMA
                      and completed.get("status") == "completed_resumed_evaluation_quality_unclassified"
                      and completed.get("original_post_pipeline_success_claimed") is False,
                      "Require the actual combined evaluation completion")
    plan = admit_plan(Path(completed["plan"]["path"]))
    admission.require(record(completed["plan"]["path"]) == completed["plan"], "Completed resumption plan changed")
    successful_steps(completed, RETRY_NAMES, "completed_resumed_evaluation_quality_unclassified")
    for row in completed["steps"]:
        admission.require(row.get("exit_observation") == "subprocess.Popen.wait", "Retry exit must come from OS wait")
        admission.require(record(row["log"]["path"]) == row["log"], "Retry child log changed")
    measurements = {**plan["prefix"]["measurements"],
                    "navigation": completed["steps"][0]["result"],
                    "navigation_CPU_audit": completed["steps"][1]["result"]}
    admission.require(measurements == completed["measurements"], "Combined measurement identities disagree")
    bundle = {key: original.bound(item) for key, item in measurements.items()}
    navigation, audit = bundle["navigation"], bundle["navigation_CPU_audit"]
    module, _ = evaluation.prepare("navigation")
    _, _, expected = module.admit_training(Path(plan["prefix"]["training_receipt"]))
    admission.require(navigation["shared_navigation_migration"]["training_execution_admission"] == expected
                      and navigation.get("status") == "passed" and navigation.get("cleanup_succeeded") is True
                      and navigation.get("cleanup_errors") == [] and navigation.get("actual_control_steps") == 6000,
                      "Retry navigation checkpoint/budget/result/cleanup disagrees")
    receipt = original.bound(completed["navigation_execution"])
    admission.require(receipt.get("schema") == "shared_social.actual_navigation200s_execution.v1"
                      and admission.exact_integer(receipt.get("actual_exit_code"), 0)
                      and receipt["result"] == measurements["navigation"] and receipt["log"] == completed["steps"][0]["log"]
                      and receipt["owner"] == plan["sources"][0], "Retry navigation receipt lineage disagrees")
    admission.require(audit.get("status") == "passed" and audit["result"] == measurements["navigation"]
                      and audit["execution_receipt"] == completed["navigation_execution"]
                      and audit.get("controls") == 6000 and audit.get("actor_controls") == 24000
                      and audit.get("Torch_imported") is False and audit.get("SDK_executed") is False,
                      "Retry original CPU audit budget/lineage disagrees")
    for item in audit["arrays"]:
        admission.require(record(item["path"]) == item, "Retry audited array changed")
    admission.require(record(path) == identity, "Resumption completion changed during admission")
    return bundle, identity


def run_owner(plan_path):
    plan = admit_plan(plan_path)
    here, output = plan_path.parent, Path(plan["evaluation_output"])
    if output.exists() or any((here / name).exists() for name in ("owner_state.json", "completion.json")):
        raise FileExistsError("Refuse duplicate resumption or overwrite")
    state = {"schema": COMPLETION_SCHEMA, "status": "running", "owner_pid": os.getpid(),
             "started_at_utc": now(), "plan": record(plan_path), "steps": [],
             "original_post_pipeline_success_claimed": False, "native_exit_root_cause_identified": False,
             "paper_effect_achieved": False, "default_model_promoted": False}

    def save():
        state["updated_at_utc"] = now()
        write_json(here / "owner_state.json", state)

    def child(name, cuda, command, result_path):
        admit_plan(plan_path)
        step = {"name": name, "status": "running", "command": command, "started_at_utc": now()}
        state["steps"].append(step)
        save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, PYTHONDONTWRITEBYTECODE="1",
                           PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log = output / (name + ".log")
        with log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            step["pid"] = process.pid
            save()
            code = process.wait()
        step.update(actual_exit_code=code, exit_observation="subprocess.Popen.wait", log=record(log), finished_at_utc=now())
        save()
        admission.require(admission.exact_integer(code, 0), f"Actual {name} exit {code}; failure retained")
        document = original.read(result_path)
        admission.require(document.get("status") == "passed", "Actual child result disagrees: " + name)
        if name == "navigation200s":
            admission.require(document.get("cleanup_succeeded") is True and document.get("cleanup_errors") == []
                              and document.get("actual_control_steps") == 6000, "Incomplete navigation cleanup/budget")
        step.update(status="passed", result=record(result_path))
        save()
        return document, step

    save()
    try:
        while True:
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                  "--format=csv,noheader"], capture_output=True, text=True, check=True)
            memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
            free = int(memory.stdout.strip())
            state["GPU_capacity_observation"] = {"observed_at_utc": now(), "free_memory_MiB": free,
                "existing_compute_processes": gpu.stdout.strip().splitlines(), "existing_processes_terminated": False}
            state["status"] = "running" if free >= previous_owner.POLICY["minimum_free_memory_MiB"] else "waiting_for_GPU_capacity"
            save()
            if state["status"] == "running":
                break
            time.sleep(30)
            admit_plan(plan_path)
        output.mkdir(parents=True)
        receipt = plan["prefix"]["training_receipt"]
        navigation_path = output / "navigation200s/result.json"
        command = [sys.executable, str(ROOT / "scripts/evaluate_shared_social_recovery.py"), "--stage", "navigation",
                   "--execution-receipt", receipt, "--diffusion-checkpoint", plan["diffusion"]["path"],
                   "--diffusion-sha256", post.DIFFUSION_SHA, "--control-steps", "6000",
                   "--output", str(navigation_path.parent)]
        document, step = child("navigation200s", "0", command, navigation_path)
        navigation_receipt = output / "navigation200s_execution.json"
        write_json(navigation_receipt, {"schema": "shared_social.actual_navigation200s_execution.v1",
            "status": "completed_measurement_quality_unclassified", "actual_exit_code": step["actual_exit_code"],
            "recorded_at_utc": now(), "owner": plan["sources"][0], "result": step["result"], "log": step["log"],
            "controls": 6000, "actor_controls": 24000, "fall_resets": document["fall_resets"],
            "route_vertices_reached": document["route_vertices_reached"],
            "paper_effect_achieved": False, "default_model_promoted": False}, readonly=True)
        audit_path = output / "navigation_CPU_array_audit.json"
        child("navigation_CPU_array_audit", "", [sys.executable, str(ROOT / "scripts/audit_shared_social_navigation.py"),
              "--execution-receipt", str(navigation_receipt), "--output", str(audit_path)], audit_path)
        admit_plan(plan_path)
        state.update(status="completed_resumed_evaluation_quality_unclassified",
                     navigation_execution=record(navigation_receipt),
                     measurements={**plan["prefix"]["measurements"], "navigation": record(navigation_path),
                                   "navigation_CPU_audit": record(audit_path)})
    except BaseException as error:
        if state["steps"] and state["steps"][-1]["status"] != "passed":
            state["steps"][-1]["status"] = "failed"
        state.update(status="failed_preserved", error={"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
    state["finished_at_utc"] = now()
    save()
    for path in output.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    write_json(here / "completion.json", state, readonly=True)
    print(json.dumps({"status": state["status"], "completion": record(here / "completion.json")}), flush=True)
    return 0 if state["status"] == "completed_resumed_evaluation_quality_unclassified" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--detach", action="store_true")
    modes.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    admit_plan(plan_path)
    if args.check_plan:
        print(json.dumps({"status": "resumption_plan_admitted", "SDK_started": False, "plan": record(plan_path)}))
        return 0
    if args.detach:
        if any((plan_path.parent / name).exists() for name in ("owner_state.json", "owner.log", "completion.json")):
            raise FileExistsError("Resumption already launched")
        with (plan_path.parent / "owner.log").open("xb") as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        print(json.dumps({"status": "resumption_launched", "owner_pid": process.pid, "actual_child_exit_observed": False}))
        return 0
    return run_owner(plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
