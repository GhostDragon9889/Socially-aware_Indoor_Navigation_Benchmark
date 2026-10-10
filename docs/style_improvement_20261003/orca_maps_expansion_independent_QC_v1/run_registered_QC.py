"""Record one real CPU QC process, without regenerating the corpus."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    receipt_path, log_path = HERE / "actual_QC_process.json", HERE / "actual_QC_CPU.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("One new QC execution; preserve historical attempts")
    command = [str(ROOT / "data/tools/orca_generator/.venv/bin/python"), str(HERE / "audit_dataset.py")]
    settings = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    env = os.environ.copy()
    env.update(settings)
    receipt = {"status": "running", "start_utc": datetime.now(timezone.utc).isoformat(),
               "command": command, "cwd": str(ROOT), "declared_env": settings,
               "reader": record(HERE / "audit_dataset.py"), "driver": record(__file__),
               "log": str(log_path), "GPU_used": False, "SDK_started": False}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    tick = time.monotonic()
    with log_path.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        receipt["child_pid"] = child.pid
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        exit_code = child.wait()
    receipt.update(status="closed", exit_code=exit_code, child_exit_observed=True,
                   end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick)
    if exit_code == 0:
        receipt["actual_QC_report"] = record(HERE / "actual_QC.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_path.chmod(0o444)
    print(json.dumps(receipt))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
