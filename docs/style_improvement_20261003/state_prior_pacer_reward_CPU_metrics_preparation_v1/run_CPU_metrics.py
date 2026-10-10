"""Run frozen CPU metrics only after actual new935 capture/source admission."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CATALOG = HERE / "commands_CPU_metrics_v1.json"
CATALOG_SHA = "eadad9fe03169c4d64a4938c335af22ae0831a55ee77494fd19c0e0ce1f5f37d"
METRICS = ("tracking90", "navigation90", "navigation45", "navigation_raw", "tracking_decomposition")


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metric_argv(metric, catalog):
    row = catalog["metrics"][metric]
    if metric in {"tracking90", "navigation90"}:
        return ["--run", row["run_name"] + "=" + row["input"], "--reference-artifact",
                catalog["sources"]["style90_reference"]["path"], "--cpu-threads", "4",
                "--minimum-bin-samples", "30", "--maximum-motion-samples", "256",
                "--output", row["output"]]
    if metric == "navigation45":
        return ["--run", row["run_name"] + "=" + row["input"], "--output", row["output"]]
    return []


def validate_preparation(catalog):
    require(catalog["schema"] == "pacer_reward.prepared_CPU_metric_commands.v1"
            and catalog["status"] == "prepared_not_executed" and catalog["future_effect"] is False,
            "Unknown prepared metric catalog")
    freeze_record = catalog["sources"]["evaluation_source_freeze"]
    require(record(freeze_record["path"]) == freeze_record, "Frozen evaluation bridge metadata changed")
    freeze = json.loads(Path(freeze_record["path"]).read_text())
    require(freeze["schema"] == "pacer_reward.prepared_evaluation_source_freeze.v1"
            and freeze["status"] == "frozen_prepared_only", "Frozen new935 evaluation bridge required")
    for row in freeze["source_files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen evaluation dependency changed: " + row["path"])
    require(record(freeze["archive"]["path"]) == freeze["archive"], "Frozen evaluation archive changed")
    for row in catalog["sources"].values():
        require(record(row["path"]) == row, "Frozen metric source/reference changed: " + row["path"])
    executor = load("metric_prepared_evaluation_executor", catalog["sources"]["evaluation_executor"]["path"])
    registration = json.loads(Path(catalog["sources"]["evaluation_registration"]["path"]).read_text())
    require(record(executor.REGISTRATION)["sha256"] == executor.REGISTRATION_SHA,
            "Preregistered protocol bytes changed")
    executor.validate_registered_dependencies(registration)
    for metric, row in catalog["metrics"].items():
        require(metric in METRICS and row["input"] == registration[row["protocol"]]["parsed"]["output"],
                "Metric input must be the actual fixed registered capture")
        require(row["checkpoint_admission"] == registration["future_checkpoint"]["terminal_admission_path"],
                "Only registered new terminal admission is accepted")
        require(Path(row["output"]).parent == ROOT / "docs/style_improvement_20261003/pacer_reward_metrics_v1",
                "Independent fixed metric output required")
        require(row["metric_argv"] == metric_argv(metric, catalog), "Fixed metric parameters differ")
    source = json.loads(Path(catalog["sources"]["isolated_source_manifest"]["path"]).read_text())
    bundle = Path(catalog["bundle"])
    code = bundle / "code"
    require(len(source["files"]) == 935, "New935 immutable source required")
    for row in source["files"]:
        path = bundle / row["path"]
        item = record(path)
        require(path.is_relative_to(code) and item["bytes"] == row["bytes"] and item["sha256"] == row["sha256"]
                and not path.stat().st_mode & 0o222, "Immutable metric code changed: " + row["path"])
    return executor, registration


def copy_external_source(catalog, key, directory):
    original = catalog["sources"][key]
    path = directory / "legacy" / "docs" / Path(original["path"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(original["path"]).read_bytes())
    path.chmod(0o444)
    actual = record(path)
    require(actual["bytes"] == original["bytes"] and actual["sha256"] == original["sha256"],
            "External frozen tool staging differs")
    return path, {"original": original, "staged_execution": actual,
                  "purpose": "Exact bytes outside checkout so the original strict owned-import guard remains active"}


def reuse_saved_basis(tool, basis_path, original_evaluation, evidence_record):
    import numpy as np
    with np.load(basis_path, allow_pickle=False) as data:
        require(set(data.files) == {"centers_50", "centers_100"}, "Fixed 45D basis keys differ")
        centers = {key: data[f"centers_{key}"].copy() for key in (50, 100)}
    for key, value in centers.items():
        require(value.dtype == np.float64 and value.shape == (key, 45) and np.isfinite(value).all(),
                "Fixed finite float64 45D centers required")

    def reuse(reference, **kwargs):
        require(len(reference) == original_evaluation["method"]["fit"]["available_reference_poses"],
                "Frozen AMASS reference population differs")
        fit = deepcopy(original_evaluation["method"]["fit"])
        fit["refit_performed_for_this_new_run"] = False
        fit["saved_basis_artifact"] = evidence_record(basis_path)
        return centers, fit

    tool.fit_reference = reuse
    return centers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU metrics require CUDA_VISIBLE_DEVICES=''")
    require(record(CATALOG)["sha256"] == CATALOG_SHA, "Frozen CPU metric catalog changed")
    catalog = json.loads(CATALOG.read_text())
    executor, registration = validate_preparation(catalog)
    if args.prepare_only:
        require(args.metric is None, "Preparation cannot select or evaluate an actual capture")
        require(not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
                "Preparation unexpectedly imported Torch or SDK")
        print(json.dumps({"status": "prepared_sources_passed_no_actual_capture_read", "future_effect": False,
              "CPU_only": True, "GPU_used": False, "copied_files_hashed": 935, "metrics": list(METRICS),
              "actual_checkpoint_read": False, "actual_metrics_computed": False}))
        return 0
    require(args.metric is not None, "Select a registered actual metric")
    row = catalog["metrics"][args.metric]
    output = Path(row["output"])
    require(not output.exists() and not output.is_symlink(), "Fresh exclusive registered metric output required")
    evidence = load("actual_new935_metric_source_bridge", catalog["sources"]["evaluation_evidence"]["path"])
    result_path = Path(row["input"]) / "result.json"
    result = json.loads(result_path.read_text())
    # No physical archive, reference population, basis, Torch or measurement
    # function is processed until complete real capture/source admission passes.
    source = evidence.admit_actual_evaluation(result, row["protocol"], row["checkpoint_admission"], row["GPU_import_report"])
    ledger_bytes, event_bytes = source.pop("ledger_bytes"), source.pop("event_prefix_bytes")
    launcher = load("new935_actual_CPU_metric_isolation", catalog["sources"]["isolated_launcher"]["path"])
    launcher.verify()
    ledger = launcher.ImportLedger(output.parent / (args.metric + "_CPU_imports.json"))
    stages = {}
    stage_dir = Path(tempfile.mkdtemp(prefix="state_prior_frozen_metric_tools_", dir="/tmp"))
    staged = {}
    keys = {"navigation45": ["pose45_evaluator"], "navigation_raw": ["raw_ground_tool", "idle_tool"],
            "tracking_decomposition": ["tracking_decomposition_tool", "tracking_comparison_tool"]}.get(args.metric, [])
    for key in keys:
        staged[key], stages[key] = copy_external_source(catalog, key, stage_dir)
    for path in sorted(stage_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    stage_dir.chmod(0o555)
    launcher.activate(ledger)
    status = "failed"
    try:
        require(not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
                "CPU metrics unexpectedly imported SDK")
        if args.metric in {"tracking90", "navigation90"}:
            tool = importlib.import_module("scripts.evaluate_pedestrian_style")
            require(record(tool.__file__) == catalog["sources"]["style90_entry"], "Actual style90 entry escaped newCODE")
            tool.main(row["metric_argv"])
            report = json.loads((output / "evaluation.json").read_text())
            require(report["sources"] == {"implementation": catalog["sources"]["style90_entry"],
                                          "module": catalog["sources"]["style90_math"]}, "Actual style90 source identities differ")
            require(record(output / "heldout_reference.json")["sha256"] == catalog["sources"]["style90_reference"]["sha256"],
                    "Actual style90 frozen reference changed")
        elif args.metric == "navigation45":
            tool = load("unchanged_pose45_saved_reference", staged["pose45_evaluator"])
            tool.ROOT = launcher.CODE
            tool.REF = Path(catalog["sources"]["style90_reference"]["path"])
            baseline = json.loads(Path(catalog["sources"]["pose45_reference_evaluation"]["path"]).read_text())
            centers = reuse_saved_basis(tool, catalog["sources"]["pose45_basis"]["path"], baseline, record)
            tool.main(row["metric_argv"])
            import numpy as np
            with np.load(output / "reference_basis.npz", allow_pickle=False) as saved:
                require(all(np.array_equal(saved[f"centers_{key}"], value) for key, value in centers.items()),
                        "Actual output centers differ from saved AMASS-only basis")
            report = json.loads((output / "evaluation.json").read_text())
            require(report["method"]["fit"]["refit_performed_for_this_new_run"] is False,
                    "Saved-basis provenance requires no refit")
        elif args.metric == "navigation_raw":
            raw = load("unchanged_ground_support_metrics", staged["raw_ground_tool"])
            idle = load("unchanged_navigation_idle_metrics", staged["idle_tool"])
            audit = raw.Audit()
            raw.verify_sources = lambda *_args, **_kwargs: source
            report = raw.audit_capture(row["input"], audit)
            report.update(schema="pacer_reward.actual_navigation_ground_audit.v1", status="passed",
                          checks=audit.checks, check_count=len(audit.checks), SDK_started=False, GPU_used=False,
                          sealed34_outputs_read=False, execution_pass_is_quality_pass=False)
            stationary = idle.stagnation(row["input"], report)
            output.mkdir(parents=True)
            write(output / "audit.json", report)
            write(output / "stagnation.json", stationary)
        else:
            tool = load("unchanged_tracking_decomposition", staged["tracking_decomposition_tool"])
            tool.ROOT = launcher.CODE
            report = tool.decompose(Path(row["input"]))
            require(report["primary_aggregate_unchanged"] == result["aggregate"], "Tracking primary aggregate changed")
            style_path = Path(catalog["metrics"]["tracking90"]["output"]) / "evaluation.json"
            style = json.loads(style_path.read_text())
            item = style["runs"][row["run_name"]]
            require(item["sources"]["result"] == record(result_path), "Tracking style result source differs")
            output.mkdir(parents=True)
            write(output / "decomposition.json", report)
            comparison = load("unchanged_tracking_style_summary", staged["tracking_comparison_tool"])
            summary = comparison.summarize(item, result, report, output / "decomposition.json")
            write(output / "comparison.json", {"schema": "pacer_reward.actual_tracking_style_summary.v1", "status": "passed",
                  "CPU_only": True, "GPU_used": False, "run": row["run_name"], "summary": summary,
                  "historical_reports_recomputed": False, "sealed34_outputs_read": False})
        import torch
        require(not torch.cuda.is_initialized(), "CPU metrics initialized CUDA")
        require(not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
                "CPU metrics started SDK")
        write(output / "actual_source_admission.json", {"schema": "pacer_reward.actual_CPU_metric_source_admission.v1",
              "status": "passed", "metric": args.metric, "CPU_only": True, "GPU_used": False, "Kit_started": False,
              "sealed34_outputs_read": False, "runner": record(__file__), "catalog": record(CATALOG),
              "actual_source_admission": source, "fixed_metric_argv": row["metric_argv"], "staged_exact_external_tools": stages,
              "pose45_reference_refit_performed": False if args.metric == "navigation45" else None,
              "metric_pass_is_quality_pass": False})
        (output / "GPU_import_ledger_as_observed.json").write_bytes(ledger_bytes)
        (output / "GPU_import_events_as_observed.jsonl").write_bytes(event_bytes)
        status = "passed"
    finally:
        ledger.finish(status)
    print(json.dumps({"status": "passed", "metric": args.metric, "output": str(output), "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
