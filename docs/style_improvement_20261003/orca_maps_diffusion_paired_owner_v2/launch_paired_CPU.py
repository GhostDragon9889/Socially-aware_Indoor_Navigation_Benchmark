"""Actual CPU owner for the original frozen100-window paired command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_paired_evaluation_v1"
TRAIN = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_actual_runs_v1"
RUNS = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_paired_actual_runs_v1"
FREEZE_SHA = "db039644f92f2c6d14dc47986d7240f3035e5cf5563ffbb7c49eed451f98b7e6"
MODEL_SHA = "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90"
ROOT_SEAL_SHA = "83ce0b532f66cd421066545ba84a8b293420a8ccba32ac1b6460486ed897b3a5"
TRAIN_RECEIPT_SHA = "876dc805bdbf11b349252bc25f6fac7130bcfae22d5351f48d6ba5cb436a13c5"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare():
    freeze_path = PREP / "source_freeze_v1.json"
    require(record(freeze_path)["sha256"] == FREEZE_SHA, "Original paired preparation changed")
    freeze = json.loads(freeze_path.read_text())
    for item in freeze["source_files"] + [freeze["archive"]]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222,
                "Frozen paired source changed")
    commands = json.loads((PREP / "commands.json").read_text())
    registration = json.loads((PREP / "registration.json").read_text())
    owner_path, seal_path = TRAIN / "training_actual_launch.json", TRAIN / "terminal5000_root_seal.json"
    require(record(owner_path)["sha256"] == TRAIN_RECEIPT_SHA and record(seal_path)["sha256"] == ROOT_SEAL_SHA,
            "Actual closed training and root seal changed")
    owner, seal = json.loads(owner_path.read_text()), json.loads(seal_path.read_text())
    training = json.loads(Path(registration["sources"]["training_registration"]["path"]).read_text())
    require(owner["schema"] == "local_orca_maps.actual_diffusion_training_launch.v1" and owner["status"] == "closed"
            and type(owner["exit_code"]) is int and owner["exit_code"] == 0
            and owner["child_exit_observed"] is True and owner["driver_completed"] is True
            and owner["command"] == training["command"] and owner["registration"] == registration["sources"]["training_registration"]
            and owner["selected_source_before"] == owner["selected_source_after"]
            and owner["selected_source_after"]["manifest"] == registration["sources"]["isolated_source_manifest"]
            and owner["selected_source_after"]["regular_files"] == 935
            and owner["selected_source_after"]["all_declared_files_verified"] is True
            and seal["schema"] == "local_orca_maps.actual_diffusion_terminal_root_seal.v1"
            and seal["actual_root_session_exit0"] is True and seal["completed_optimizer_updates"] == 5000
            and seal["all5000_metric_rows_retained"] is True and seal["selected_source_before_after_equal"] is True,
            "Actual completed source-bound training required")
    original = record(registration["future_checkpoint"]["path"])
    copy_path = seal["exclusive_terminal_copy"]["path"]
    independent = record(copy_path)
    require(original["sha256"] == independent["sha256"] == MODEL_SHA
            and original["bytes"] == independent["bytes"] == 279052920
            and independent == {key: seal["exclusive_terminal_copy"][key] for key in ("path", "bytes", "sha256")}
            and Path(copy_path).stat().st_nlink == 1
            and not Path(copy_path).stat().st_mode & 0o222
            and not Path(original["path"]).stat().st_mode & 0o222,
            "Original readonly checkpoint and independent exact sealed copy required")
    command = [MODEL_SHA if value == "ACTUAL_READONLY_EXPANDED_FINAL5000_SHA256" else value
               for value in commands["actual_command_after_training_closes_and_checkpoint_is_sealed"]]
    binding = {"prepared_source_freeze": record(freeze_path), "commands": record(PREP / "commands.json"),
               "registration": record(PREP / "registration.json"), "actual_closed_training": record(owner_path),
               "actual_terminal_root_seal": record(seal_path), "registered_original_checkpoint": original,
               "independent_sealed_checkpoint_copy": independent, "original_checkpoint_nlink": Path(original["path"]).stat().st_nlink,
               "complete_training_runtime_import_ledger_claimed": False}
    return command, commands["CPU_environment"], Path(registration["output"]), binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    command, settings, output, binding = prepare()
    if args.prepare_only:
        print(json.dumps({"status": "prepared_actual_source_bound", "command": command, "binding": binding,
                          "models_executed": False, "GPU_used": False, "SDK_started": False}))
        return 0
    require(not output.exists(), "Actual paired output is exclusive; retain first failure")
    RUNS.mkdir(parents=True, exist_ok=True)
    path, log = RUNS / "comparison_actual_launch.json", RUNS / "comparison_CPU.log"
    require(not path.exists() and not log.exists(), "Actual paired owner records are exclusive")
    receipt = {"schema": "local_orca_maps.actual_paired_CPU_launch.v1", "status": "running", "command": command,
               "cwd": str(ROOT), "declared_env": settings, "binding_before": binding, "owner": record(__file__),
               "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
               "start_utc": datetime.now(timezone.utc).isoformat(), "log": str(log), "artifacts": [],
               "child_exit_observed": False, "driver_completed": False, "exit_code": None}
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    child, started = None, time.monotonic()
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **settings}, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["child_pid"] = child.pid
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = child.wait()
        receipt.update(status="closed", child_exit_observed=True, driver_completed=True)
        after = prepare()[3]
        require(after == binding, "Paired source/training bindings changed during evaluation")
        receipt["binding_after"] = after
        if receipt["exit_code"] == 0:
            report = json.loads((output / "comparison.json").read_text())
            require(report["status"] == "passed" and report["windows"] == report["unique_scenes"] == 100
                    and report["shared_loss_timesteps_noise"] is True and report["shared_plan_seed_per_batch"] is True,
                    "Actual paired comparison report differs")
            for artifact in sorted(output.rglob("*")):
                if artifact.is_file():
                    artifact.chmod(0o444)
                    receipt["artifacts"].append(record(artifact))
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
        receipt.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        if log.exists():
            log.chmod(0o444)
            receipt["actual_log"] = record(log)
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        path.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"], "receipt": record(path)}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
