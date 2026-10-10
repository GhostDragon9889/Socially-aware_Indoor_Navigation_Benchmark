#!/usr/bin/env python3
"""Own serial S1 terminal inference, full22/style and complete navigation.

Starts only from a separately recorded actual training exit and CPU audit.
Every subprocess exit is saved. A failure stops the chain and retains data.
Successful execution does not certify gait quality or paper equivalence.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402

DIFFUSION = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt"
DIFFUSION_SHA = "4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff"
SOURCE_PATHS = ["scripts/" + name + ".py" for name in (
    "run_shared_social_post_training", "audit_shared_social_inference", "run_shared_social_evaluation",
    "evaluate_shared_social_tracking", "evaluate_shared_social_style", "evaluate_shared_social_navigation",
    "audit_shared_social_navigation")]
SOURCE_PATHS += ["CrowdSim/humanoid/runtime/" + name + ".py" for name in (
    "social_evaluation_lifecycle", "social_navigation_lifecycle", "social_observation_producer",
    "social_popart_inference", "social_popart_training")]
SOURCE_PATHS += ["CrowdSim/humanoid/training/value_normalization.py"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = args.execution_receipt.resolve()
    checkpoint, _, admission = admit_training(receipt)
    registration = json.loads(args.registration.read_text())
    result = json.loads(Path(admission["result"]["path"]).read_text())
    expected_config = json.loads(Path(registration["configuration"]["path"]).read_text())
    if (registration.get("schema") != "shared_social.popart_paired_experiment_registration.v1"
            or admission["value_normalization"] != "popart"
            or result["profile"] != expected_config
            or record(registration["configuration"]["path"]) != registration["configuration"]
            or str(checkpoint.parent) != registration["run_directory"]
            or checkpoint.name != registration["preselected_checkpoint"]
            or result["completed_updates"] != 2000 or result["actual_training_transitions"] != 16384000):
        raise ValueError("Require the complete preregistered fresh S1 terminal run")
    if record(DIFFUSION)["sha256"] != DIFFUSION_SHA:
        raise ValueError("Fixed formal Diffusion40k changed")
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory; retain prior and failed evaluations")
    sources = [record(ROOT / path) for path in SOURCE_PATHS]
    input_records = [record(receipt), record(args.registration)]
    output.mkdir(parents=True)
    report_path = output / "execution.json"
    report = {"schema": "shared_social.actual_post_training_pipeline.v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "inputs": input_records, "training_execution_admission": admission, "steps": [],
        "paper_effect_achieved": False, "default_model_promoted": False,
        "scope": "Actual subprocess exits and complete fixed-protocol measurements; no paper quality claim"}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def check_sources():
        if ([record(ROOT / path) for path in SOURCE_PATHS] != sources
                or [record(receipt), record(args.registration)] != input_records):
            raise RuntimeError("Post-training sources or admitted inputs changed")

    def child(name, cuda, command, result_path, expected_status, *, cleanup=False):
        check_sources()
        step = {"name": name, "status": "running", "command": command, "CUDA_VISIBLE_DEVICES": cuda,
                "started_at_utc": datetime.now(timezone.utc).isoformat()}
        report["steps"].append(step)
        save()
        print(json.dumps({"step": name, "status": "running"}), flush=True)
        started = time.monotonic()
        log = output / (name + ".log")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
        with log.open("xb") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, check=False)
        step.update(actual_exit_code=completed.returncode, elapsed_s=time.monotonic() - started, log=record(log))
        if completed.returncode != 0:
            step["status"] = "failed"
            raise RuntimeError(f"{name} exited {completed.returncode}; later stages not started")
        document = json.loads(result_path.read_text())
        step["result"] = record(result_path)
        if (document.get("status") != expected_status
                or cleanup and (document.get("cleanup_succeeded") is not True or document.get("cleanup_errors"))):
            step["status"] = "failed"
            raise RuntimeError("Actual child exit/result/cleanup disagree: " + name)
        check_sources()
        step["status"] = "passed"
        save()
        print(json.dumps({"step": name, "status": "passed", "actual_exit_code": completed.returncode}), flush=True)
        return document, step

    save()
    try:
        python = sys.executable
        child("actor_CPU_inference", "", [python, str(ROOT / SOURCE_PATHS[1]), "--execution-receipt", str(receipt),
            "--output", str(output / "actor_CPU_inference.json")], output / "actor_CPU_inference.json", "passed")
        evaluation = output / "tracking_style"
        child("tracking_style", "0", [python, str(ROOT / SOURCE_PATHS[2]), "--execution-receipt", str(receipt),
            "--output", str(evaluation)], evaluation / "execution.json", "passed_execution")
        navigation = [python, str(ROOT / "scripts/evaluate_shared_social_navigation.py"),
            "--execution-receipt", str(receipt), "--diffusion-checkpoint", str(DIFFUSION), "--diffusion-sha256", DIFFUSION_SHA]
        for name, controls in (("navigation_smoke", 30), ("navigation200s", 6000)):
            target = output / name
            document, step = child(name, "0", [*navigation, "--control-steps", str(controls), "--output", str(target)],
                                   target / "result.json", "passed", cleanup=True)
            if document.get("actual_control_steps") != controls:
                raise RuntimeError("Incomplete physical navigation budget")
            if name == "navigation200s":
                owned = {"schema": "shared_social.actual_navigation200s_execution.v1",
                    "status": "completed_measurement_quality_unclassified", "actual_exit_code": step["actual_exit_code"],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "owner": record(__file__),
                    "result": step["result"], "log": step["log"], "controls": controls, "actor_controls": controls * 4,
                    "fall_resets": document["fall_resets"], "route_vertices_reached": document["route_vertices_reached"],
                    "paper_effect_achieved": False, "default_model_promoted": False}
                navigation_receipt = output / "navigation200s_execution.json"
                with navigation_receipt.open("x") as stream:
                    json.dump(owned, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                navigation_receipt.chmod(0o444)
        child("navigation_CPU_array_audit", "", [python, str(ROOT / "scripts/audit_shared_social_navigation.py"),
            "--execution-receipt", str(navigation_receipt), "--output", str(output / "navigation_CPU_array_audit.json")],
            output / "navigation_CPU_array_audit.json", "passed")
        report["measurements"] = {"tracking": record(evaluation / "full22/result.json"),
            "style90": record(evaluation / "style/evaluation.json"), "pose45": record(evaluation / "style/pose45.json"),
            "navigation": record(output / "navigation200s/result.json"),
            "navigation_CPU_audit": record(output / "navigation_CPU_array_audit.json")}
        report["status"] = "passed_execution"
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        if report["steps"] and report["steps"][-1]["status"] != "passed":
            report["steps"][-1]["status"] = "failed"
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report["status"] == "passed_execution" else 1


if __name__ == "__main__":
    raise SystemExit(main())
