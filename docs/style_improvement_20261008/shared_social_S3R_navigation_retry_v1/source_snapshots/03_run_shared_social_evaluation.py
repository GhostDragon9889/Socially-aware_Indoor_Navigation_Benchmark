#!/usr/bin/env python3
"""Run lifecycle, full22 and frozen style checks sequentially in fresh processes.

Requires a terminal training execution receipt. Each child exit is recorded;
failed outputs stay intact and prevent subsequent steps. No model promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory; failed/previous evaluations are preserved")
    sys.path.insert(0, str(ROOT))
    from scripts.evaluate_shared_social_tracking import admit_training
    receipt = args.execution_receipt.resolve()
    _, _, admission = admit_training(receipt)
    sources = [ROOT / "scripts/evaluate_shared_social_tracking.py",
               ROOT / "scripts/evaluate_shared_social_style.py",
               ROOT / "CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py", Path(__file__)]
    sources.append(ROOT / "CrowdSim/humanoid/runtime/social_observation_producer.py")
    if admission["value_normalization"] == "popart":
        sources.extend(ROOT / path for path in (
            "CrowdSim/humanoid/runtime/social_popart_inference.py",
            "CrowdSim/humanoid/runtime/social_popart_training.py",
            "CrowdSim/humanoid/training/value_normalization.py"))
    source_records = [identity(path) for path in sources]
    output.mkdir(parents=True)
    report_path = output / "execution.json"
    report = {"schema": "shared_social.actual_evaluation_pipeline.v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_receipt": identity(receipt),
              "sources": source_records, "steps": [], "paper_effect_achieved": False,
              "training_execution_admission": admission,
              "claim_scope": "Actual child exits and independent full22/style measurements; no author equivalence"}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    tracking = sources[0]
    commands = [
        ("lifecycle_smoke", "0", [sys.executable, str(tracking), "--execution-receipt", str(receipt),
                                 "--lifecycle-smoke", "--output", str(output / "lifecycle_smoke")]),
        ("full22", "0", [sys.executable, str(tracking), "--execution-receipt", str(receipt),
                         "--output", str(output / "full22")]),
        ("style", "", [sys.executable, str(sources[1]), "--run", "shared2000=" + str(output / "full22"),
                        "--output", str(output / "style")]),
    ]
    save()
    try:
        for name, cuda, command in commands:
            if [identity(path) for path in sources] != source_records or identity(receipt) != report["training_receipt"]:
                raise RuntimeError("Evaluation sources or training receipt changed")
            log = output / (name + ".log")
            step = {"name": name, "command": command, "CUDA_VISIBLE_DEVICES": cuda,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
            report["steps"].append(step)
            save()
            print(json.dumps({"step": name, "status": "running"}), flush=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
            started = time.monotonic()
            with log.open("xb") as stream:
                child = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, check=False)
            step.update(status="exited_zero" if child.returncode == 0 else "failed",
                        actual_exit_code=child.returncode, elapsed_s=time.monotonic() - started, log=identity(log))
            if child.returncode != 0:
                raise RuntimeError(f"{name} failed with actual exit {child.returncode}; outputs retained")
            result = output / name / ("evaluation.json" if name == "style" else "result.json")
            document = json.loads(result.read_text())
            step["result"] = identity(result)
            expected_status = "passed_lifecycle_smoke" if name == "lifecycle_smoke" else "passed"
            if document.get("status") != expected_status or name != "style" and document.get("cleanup_succeeded") is not True:
                raise RuntimeError("Actual child exit/result/cleanup disagree: " + name)
            step["status"] = "passed"
            save()
            print(json.dumps({"step": name, "status": "passed", "actual_exit_code": child.returncode}), flush=True)
        if [identity(path) for path in sources] != source_records or identity(receipt) != report["training_receipt"]:
            raise RuntimeError("Evaluation sources or training receipt changed during execution")
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
