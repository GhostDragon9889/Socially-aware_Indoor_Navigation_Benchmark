"""Observe and seal actual CPU processes for the unchanged five metric commands."""
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
PREP = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_CPU_metrics_preparation_v1"
CATALOG = PREP / "commands_CPU_metrics_v1.json"
FREEZE = PREP / "source_freeze_v1.json"
CATALOG_SHA = "eadad9fe03169c4d64a4938c335af22ae0831a55ee77494fd19c0e0ce1f5f37d"
FREEZE_SHA = "787401ece9a596cb03c56363e13e9f7d2ec6ae5010450330bd9ff8262e6cabd2"
EVALUATION = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_v1/commands.json"
EVALUATION_SHA = "898ca372d5c6c6296f9f9724993af5a72a0338812b0d13eb0f9dc80732d11152"
RUNS = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_CPU_metrics_actual_runs_v1"
GPU_RUNS = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_actual_runs_v1"
CHECKPOINT_SHA = "09e802847caba0b8c1ef836da2299d3468166be65aad127e2249f455326e0066"
TERMINAL_SHA = "c0e93cf94222a6f46a1d3bde8f0202915d0f1ca73dc56539c8941d2d754184c0"
REPORTS = {"tracking90": "evaluation.json", "tracking_decomposition": "comparison.json",
           "navigation_raw": "audit.json", "navigation90": "evaluation.json", "navigation45": "evaluation.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular local artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def preparation():
    require(record(CATALOG)["sha256"] == CATALOG_SHA and record(FREEZE)["sha256"] == FREEZE_SHA,
            "Frozen CPU metric preparation changed")
    frozen = json.loads(FREEZE.read_text())
    for item in frozen["source_files"] + [frozen["archive"]]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222,
                "Frozen CPU metric source changed: " + item["path"])
    require(record(EVALUATION)["sha256"] == EVALUATION_SHA, "Registered GPU commands changed")
    catalog = json.loads(CATALOG.read_text())
    for metric, row in catalog["metrics"].items():
        require(metric in REPORTS and row["command"] == [
            "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python", catalog["runner"], "--metric", metric],
            "Original registered metric command required")
    require(catalog["CPU_environment"]["CUDA_VISIBLE_DEVICES"] == "", "CPU environment must disable CUDA")
    return catalog


def closed_evaluation(metric, catalog):
    protocol = catalog["metrics"][metric]["protocol"]
    path = GPU_RUNS / (protocol + "_actual_launch.json")
    receipt = json.loads(path.read_text())
    plan = json.loads(EVALUATION.read_text())
    command = [CHECKPOINT_SHA if value == "ACTUAL_SEALED_STEP2000_SHA256" else value
               for value in plan[protocol + "_GPU_root_only"]]
    require(receipt.get("schema") == "pacer_reward.actual_evaluation_launch.v1"
            and receipt.get("protocol") == protocol and receipt.get("status") == "closed"
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and receipt.get("child_exit_observed") is True and receipt.get("driver_completed") is True
            and receipt.get("command") == command and receipt.get("cwd") == str(ROOT)
            and receipt.get("commands") == record(EVALUATION)
            and receipt.get("amp_checkpoint", {}).get("sha256") == CHECKPOINT_SHA
            and receipt.get("terminal_admission", {}).get("sha256") == TERMINAL_SHA,
            "Actual closed registered GPU evaluation required before CPU metrics")
    require(not path.stat().st_mode & 0o222, "Actual GPU owner receipt must be sealed readonly")
    return record(path)


def run_child(command, environment, log, receipt, path):
    """Keep first failure and observe cleanup; never infer a successful closure."""
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["child_pid"] = child.pid
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = child.wait()
        receipt.update(status="closed", child_exit_observed=True, driver_completed=True)
    except BaseException as error:
        receipt.update(status="owner_failed", owner_error=repr(error), driver_completed=False)
        if child is not None:
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
            receipt.update(exit_code=child.returncode, child_exit_observed=child.returncode is not None)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=tuple(REPORTS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    catalog = preparation()
    if args.prepare_only:
        require(args.metric is None, "Preparation does not execute or admit a metric")
        print(json.dumps({"status": "prepared_not_executed", "future_effect": False,
                          "GPU_used": False, "SDK_started": False, "future_capture_read": False,
                          "catalog": record(CATALOG), "metric_preparation_freeze": record(FREEZE)}))
        return 0
    require(args.metric is not None, "Actual metric name required")
    row = catalog["metrics"][args.metric]
    gpu_closure = closed_evaluation(args.metric, catalog)
    output = Path(row["output"])
    require(not output.exists(), "Actual metric output is exclusive; retain any first failure")
    RUNS.mkdir(parents=True, exist_ok=True)
    path, log = RUNS / (args.metric + "_actual_launch.json"), RUNS / (args.metric + "_CPU.log")
    require(not path.exists() and not log.exists(), "Actual CPU closure records are exclusive")
    env = {**os.environ, **catalog["CPU_environment"]}
    receipt = {"schema": "pacer_reward.actual_CPU_metric_launch.v1", "metric": args.metric,
               "status": "running", "command": row["command"], "cwd": str(ROOT),
               "declared_env": catalog["CPU_environment"], "CPU_only": True, "GPU_used": False,
               "SDK_started": False, "sealed34_outputs_read": False, "prepare_only": False,
               "start_utc": datetime.now(timezone.utc).isoformat(), "owner": record(__file__),
               "catalog": record(CATALOG), "runner": record(catalog["runner"]),
               "metric_preparation_freeze": record(FREEZE), "actual_GPU_owner_closure": gpu_closure,
               "log": str(log), "artifacts": {}, "metric_success_is_quality_success": False}
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    start = time.monotonic()
    try:
        run_child(row["command"], env, log, receipt, path)
        require(closed_evaluation(args.metric, catalog) == gpu_closure, "Closed GPU evidence changed during CPU run")
        require(record(CATALOG)["sha256"] == CATALOG_SHA, "Metric command catalog changed during CPU run")
        if receipt["exit_code"] == 0:
            report_path, sidecar_path = output / REPORTS[args.metric], output / "actual_source_admission.json"
            sidecar = json.loads(sidecar_path.read_text())
            require(sidecar.get("status") == "passed" and sidecar.get("metric") == args.metric
                    and sidecar.get("runner") == receipt["runner"] and sidecar.get("catalog") == receipt["catalog"],
                    "Actual metric source admission differs")
            artifacts = {"raw_report" if args.metric == "navigation_raw" else "metric_report": record(report_path),
                         "source_sidecar": record(sidecar_path)}
            for artifact in sorted(output.rglob("*")):
                if artifact.is_file():
                    artifact.chmod(0o444)
            imports = Path(row["CPU_import_report"])
            for artifact in (imports, Path(str(imports) + ".events.jsonl")):
                artifact.chmod(0o444)
            receipt["artifacts"] = artifacts
            receipt["sealed_output_artifacts"] = [record(p) for p in sorted(output.rglob("*")) if p.is_file()]
            receipt["actual_CPU_import_ledger"] = record(imports)
            receipt["actual_CPU_import_events"] = record(Path(str(imports) + ".events.jsonl"))
    except BaseException as error:
        receipt.update(status="owner_failed", owner_error=repr(error), driver_completed=False)
        raise
    finally:
        receipt.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start)
        if log.exists():
            log.chmod(0o444)
            receipt["actual_log"] = record(log)
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        path.chmod(0o444)
    print(json.dumps({"metric": args.metric, "status": receipt["status"],
                      "exit_code": receipt["exit_code"], "receipt": record(path)}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
