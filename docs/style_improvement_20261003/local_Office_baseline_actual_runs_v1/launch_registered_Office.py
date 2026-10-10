"""Root-owned registered baseline Office3 process; --prepare-only uses CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path("/project/EXP/SangforAndSaut/ProtoMotions")
DOC = Path(__file__).resolve().parent
PREPARED = ROOT / "docs/style_improvement_20261003/local_Office_baseline_probe_preparation_v1"
COMMANDS = PREPARED / "commands.json"
COMMANDS_SHA = "df47d5c61b1e14a936df9a2e379e410fc90176784240a5e07b9de031572d298d"
FREEZE = PREPARED / "source_freeze_v1.json"
FREEZE_SHA = "998f4a9406b2a4223e3146e115835ed0e8bd1b82161d9b89d580507d62b5edcb"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular registered source/artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepared_sources():
    require(record(COMMANDS)["sha256"] == COMMANDS_SHA and record(FREEZE)["sha256"] == FREEZE_SHA,
            "Exact independently reviewed baseline Office commands/freeze required")
    commands = json.loads(COMMANDS.read_text())
    freeze = json.loads(FREEZE.read_text())
    require(commands["schema"] == "local_Office.baseline62c4_prepared_commands.v1"
            and freeze["status"] == "frozen_prepared_only" and freeze["independent_source_peer_passed"] is True,
            "Prepared baseline protocol source contract differs")
    for row in freeze["source_files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen baseline Office source changed")
    require(record(freeze["archive"]["path"]) == freeze["archive"], "Prepared baseline archive changed")
    adapter = Path(commands["adapter"]["path"])
    require(record(adapter) == commands["adapter"], "Registered baseline driver changed")
    spec = importlib.util.spec_from_file_location("Office_actual_owner_registered_adapter", adapter)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    plan, _ = driver.prepare_sources()
    require(plan["selected_checkpoint"] == commands["selected_checkpoint"]
            and plan["production_source"] == commands["production_source"]
            and record(plan["observer"]["path"]) == commands["observer"],
            "Actual current baseline185/controller/observer differs")
    return commands, plan, driver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    commands, plan, driver = prepared_sources()
    if args.prepare_only:
        print(json.dumps({"status": "prepared_source_bound_not_executed", "GPU_used": False,
                          "SDK_started": False, "actual_Office_launched": False,
                          "checkpoint": plan["selected_checkpoint"], "owner": record(__file__),
                          "production_source": plan["production_source"]}))
        return 0
    command = commands["future_root_owned_command"]
    require(command == [commands["CPU_prepare_command"][0], str(PREPARED / "driver_adapter.py")]
            and "--prepare-only" not in command, "Exact registered actual Office command required")
    output = Path(plan["actual_Office_capture"]["path"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Office actual output must be new")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         check=True, text=True, capture_output=True)
    require(not gpu.stdout.strip(), "Another CUDA compute process is active")
    receipt_path = Path(commands["actual_launch_record"])
    require(receipt_path == DOC / "Office3_actual_launch.json", "Registered root Office receipt path differs")
    log = DOC / "Office3_GPU.log"
    if receipt_path.exists() or log.exists():
        raise FileExistsError("Preserve actual Office launch records")
    environment = os.environ.copy()
    environment.update(commands["future_GPU_environment"])
    receipt = {
        "schema": commands["actual_launch_schema"], "status": "running", "actors": 3,
        "start_utc": datetime.now(timezone.utc).isoformat(), "command": command, "cwd": str(ROOT),
        "declared_env": commands["future_GPU_environment"], "prepare_only": False,
        "registration": record(COMMANDS), "prepared_source_freeze": record(FREEZE), "owner": record(__file__),
        "adapter": commands["adapter"], "observer": commands["observer"], "plan": commands["plan"],
        "selected_checkpoint": plan["selected_checkpoint"], "selected_source_before": plan["production_source"],
        "log": str(log), "no_parallel_GPU_at_launch": True, "atomic_GPU_lock_claimed": False,
        "exit_code": None, "child_exit_observed": False, "driver_completed": False,
        "actual_Office_physical_quality_claimed": False, "complete_runtime_import_ledger": False,
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["child_pid"] = child.pid
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = child.wait()
        receipt.update(status="closed", child_exit_observed=True, driver_completed=True)
    except BaseException as error:
        receipt.update(status="owner_failed", owner_error=repr(error), driver_completed=False)
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=10)
            except BaseException as cleanup_error:
                receipt["cleanup_error"] = repr(cleanup_error)
            finally:
                receipt.update(exit_code=child.returncode, child_exit_observed=child.returncode is not None)
        raise
    finally:
        receipt["end_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            after, _ = driver.prepare_sources()
            receipt["selected_source_after"] = after["production_source"]
            receipt["selected_checkpoint_after"] = after["selected_checkpoint"]
        except BaseException as error:
            receipt.update(status="source_verification_failed", source_verification_error=repr(error))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt_path.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"],
                      "receipt": str(receipt_path)}), flush=True)
    return 1 if "source_verification_error" in receipt else receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
