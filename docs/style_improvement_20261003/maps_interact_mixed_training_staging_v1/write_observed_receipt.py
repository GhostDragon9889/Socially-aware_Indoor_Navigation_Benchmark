"""Record observed smoke closure; this does not launch a process."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
report = HERE / "actual_CPU_model_smoke.json"
document = json.loads(report.read_text())
if document["status"] != "passed" or document["torch_cuda_initialized"] is not False:
    raise ValueError("Unexpected actual smoke status")
receipt = {
    "schema": "local_orca.maps_interact_actual_CPU_closedreceipt.v1",
    "status": "closed", "actual_exit_code": 0, "child_exit_observed": True,
    "actual_tool_session_id": 93701, "actual_launch_chunk_id": "0258dc",
    "actual_completion_chunk_id": "793c9d",
    "command": ["/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python", str(HERE / "actual_CPU_model_smoke.py")],
    "cwd": str(ROOT),
    "env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
    "actual_internal_elapsed_seconds": document["elapsed_seconds"],
    "report": {"path": str(report), "bytes": report.stat().st_size,
               "sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
    "full_trainer_main_executed": False, "full_training_completed": False,
    "physics_or_DDPM_sampling_executed": False,
    "initial_opaque_hash_calls": 1, "initial_checkpoint_load_calls": 1,
    "scope": "Two real CPU B2 trainsteps in separate modes and their model/Adam/RNG restores; not paper effect evidence"
}
(HERE / "actual_CPU_closedreceipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"path": str(HERE / "actual_CPU_closedreceipt.json"),
                  "sha256": hashlib.sha256((HERE / "actual_CPU_closedreceipt.json").read_bytes()).hexdigest()}))

