"""Two real trivial CPU child closures plus source-only preparation; no captures."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
ENTRY = HERE / "launch_CPU_metric.py"
specification = importlib.util.spec_from_file_location("prepared_CPU_metric_owner", ENTRY)
owner = importlib.util.module_from_spec(specification)
specification.loader.exec_module(owner)
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
start = time.monotonic()
command = [SDK, str(ENTRY), "--prepare-only"]
prepared = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
prepared_document = json.loads(prepared.stdout)
assert prepared_document["status"] == "prepared_not_executed" and prepared_document["future_capture_read"] is False
cases = []
with tempfile.TemporaryDirectory(prefix="pacer_reward_CPU_owner_proof_", dir="/tmp") as directory:
    for expected_exit in (0, 3):
        path = Path(directory) / f"closure_{expected_exit}.json"
        log = Path(directory) / f"closure_{expected_exit}.log"
        receipt = {"status": "running"}
        path.write_text(json.dumps(receipt))
        owner.run_child([SDK, "-c", f"raise SystemExit({expected_exit})"], env, log, receipt, path)
        assert receipt["status"] == "closed" and type(receipt["exit_code"]) is int
        assert receipt["exit_code"] == expected_exit and receipt["child_exit_observed"] is True
        assert receipt["driver_completed"] is True
        cases.append({"expected_exit": expected_exit, "observed_exit": receipt["exit_code"],
                      "child_exit_observed": True, "driver_completed": True,
                      "nonzero_is_success": False})
proof = {"schema": "pacer_reward.CPU_metric_owner_process_preparation_proof.v1", "status": "passed",
         "elapsed_seconds": time.monotonic() - start, "prepare_command": command, "prepare_exit_code": prepared.returncode,
         "prepare_stdout": prepared_document, "actual_trivial_child_cases": cases,
         "owner": owner.record(ENTRY), "proof_source": {"path": str(Path(__file__).absolute()),
         "bytes": Path(__file__).stat().st_size, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
         "CPU_only": True, "GPU_used": False, "SDK_started": False,
         "future_capture_read": False, "actual_metrics_computed": False, "sealed34_outputs_read": False}
with (HERE / "process_closure_CPU_proof.json").open("x") as stream:
    json.dump(proof, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": "passed", "trivial_children": len(cases), "prepare_exit_code": prepared.returncode}))
