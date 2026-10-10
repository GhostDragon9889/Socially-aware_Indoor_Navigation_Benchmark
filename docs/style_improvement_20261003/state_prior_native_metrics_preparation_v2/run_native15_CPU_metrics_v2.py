"""Bind the native15 JSON-proof revision and reuse frozen v1 mathematics."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "docs/style_improvement_20261003/state_prior_native_metrics_preparation_v1"
sys.path.insert(0, str(BASE))
import native_metric_sources as core  # noqa: E402

CATALOG = HERE / "commands_native15_CPU_metrics_v2.json"
CATALOG_SHA = "5870b6638b14668976c09d2f30a13003b328e4732e540a3c5107737c50dcb000"


def modules(catalog):
    runner = core.load("immutable_native_metric_math_v1", catalog["sources"]["base_runner"]["path"])
    comparison = core.load("immutable_native_metric_comparison_v1", catalog["sources"]["base_comparison"]["path"])
    return runner, comparison


def readonly_freeze(row):
    core.exact(row)
    value = json.loads(Path(row["path"]).read_text())
    for item in value["files"]:
        core.exact(item)
        core.require(not Path(item["path"]).stat().st_mode & 0o222, "Frozen revision source must be readonly")
    core.exact(value["archive"])


def validate_revision_report(raw, catalog):
    source = catalog["sources"]
    outer = json.loads(Path(source["raw15_revision_registration"]["path"]).read_text())
    core.require(outer["schema"] == "public_state_prior.native15_JSON_proof_registration.v2"
                 and outer["source_count"] == 183 and outer["native_execution_audited"] is False,
                 "Exact JSON representation revision registration required")
    core.require(raw.get("representation_adapter") == source["raw15_revision_driver"]
                 and raw.get("representation_revision_registration") == outer
                 and raw.get("immutable_math_core_freeze") == source["native_reader_freeze"]
                 and raw.get("first_actual_CPU_failure") == outer["first_actual_failure"]
                 and raw.get("compiled_v1_v2_routes_math_changed") is False
                 and raw.get("proof_tuple_list_difference_only") is True,
                 "Native15 must bind the approved representation-only adapter; no blind v1 acceptance")
    core.require(outer["representation_adapter"] == source["raw15_revision_driver"]
                 and outer["immutable_core_freeze"] == source["native_reader_freeze"]
                 and outer["immutable_core_registration"] == source["native15_core_registration"]
                 and raw["adapter_proof"] == outer["actual_original_proof"] == raw["registration"]["adapter_proof"],
                 "Canonical JSON proof values and immutable mathematics must be identical")


def validate(catalog):
    core.require(catalog.get("schema") == "public_state_prior.prepared_native15_metric_commands.v2"
                 and catalog.get("status") == "prepared_not_executed" and catalog.get("future_effect") is False,
                 "Exact new native15 prepared catalog required")
    readonly_freeze(catalog["sources"]["base_metrics_freeze"])
    readonly_freeze(catalog["sources"]["raw15_revision_freeze"])
    allowed = core.validate_engine(catalog)
    runner, _ = modules(catalog)
    core.require(set(catalog["metrics"]) == {"90", "45"}, "Only new native15 metrics in this revision")
    for dimension, row in catalog["metrics"].items():
        core.require(row["kind"] == "new" and row["actors"] == 15 and row["source_count"] == 183
                     and row["dimension"] == int(dimension)
                     and row["input"] == str(ROOT / "output/diffusion_amp/native_warehouse_state_prior_15_clearance08_200s_v1")
                     and row["output"] == str(ROOT / f"docs/style_improvement_20261003/state_prior_native_metrics_v2/new15_{dimension}")
                     and row["registration"] == catalog["sources"]["native15_core_registration"]["path"]
                     and row["raw_audit"] == catalog["sources"]["actual_native15_raw_audit"]["path"],
                     "Exact native15 output/source/registration aliases required")
        core.require(row["metric_argv"] == runner.argv_for(row, catalog), "Fixed native15 metric parameters changed")
    raw = json.loads(Path(catalog["sources"]["actual_native15_raw_audit"]["path"]).read_text())
    validate_revision_report(raw, catalog)
    return allowed


def admit(catalog):
    row = catalog["metrics"]["90"]
    raw_path = Path(row["raw_audit"])
    core.exact(catalog["sources"]["actual_native15_raw_audit"])
    core.require(not raw_path.stat().st_mode & 0o222, "Actual raw15 report must be readonly")
    raw = json.loads(raw_path.read_text())
    validate_revision_report(raw, catalog)
    source = core.admit_run(catalog, row)
    source["JSON_representation_revision"] = {
        "freeze": catalog["sources"]["raw15_revision_freeze"],
        "driver": catalog["sources"]["raw15_revision_driver"],
        "outer_registration": catalog["sources"]["raw15_revision_registration"],
        "compiled_math_changed": False, "proof_tuple_list_difference_only": True,
        "first_actual_failure": raw["first_actual_CPU_failure"]}
    return source


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metric(dimension, catalog, allowed):
    row = catalog["metrics"][str(dimension)]
    output = Path(row["output"])
    core.require(not output.exists() and not output.is_symlink(), "Fresh native15 metric output required")
    source = admit(catalog)
    core.require("torch" not in sys.modules, "Native source bridge must not deserialize models")
    ledger = core.CPULedger(output.parent / f"new15_{dimension}_CPU_imports.json", allowed, core.record(CATALOG))
    runner, _ = modules(catalog)
    ledger.activate()
    status = "failed"
    try:
        if dimension == 90:
            tool = importlib.import_module("scripts.evaluate_pedestrian_style")
            tool.main(row["metric_argv"])
            report = json.loads((output / "evaluation.json").read_text())
            core.require(report["sources"] == {"implementation": catalog["sources"]["style90_entry"],
                                               "module": catalog["sources"]["style90_math"]}, "Original90 source identity changed")
        else:
            runner.pose45(row, catalog)
        core.require(core.record(output / "heldout_reference.json")["sha256"] == catalog["sources"]["style_reference"]["sha256"],
                     "Exact c274 reference required")
        import torch
        core.require(not torch.cuda.is_initialized(), "CPU native15 metrics initialized CUDA")
        core.require(not any(n == "isaacsim" or n.startswith(("omni.", "isaaclab.")) for n in sys.modules), "CPU metrics imported SDK")
        validate(catalog)
        write(output / "actual_native_source_admission.json", {
            "schema": "public_state_prior.native15_style_source_admission.v2", "status": "passed",
            "metric": f"new15_{dimension}", "CPU_only": True, "GPU_used": False, "Kit_started": False,
            "sealed34_outputs_read": False, "runner": core.record(__file__), "catalog": core.record(CATALOG),
            "immutable_v1_math_runner": catalog["sources"]["base_runner"],
            "actual_native_source_admission": source, "fixed_metric_argv": row["metric_argv"],
            "reference_refit_performed": False, "style_pass_is_quality_pass": False, "default_promotion": False})
        status = "passed"
    finally:
        ledger.finish(status)
    for item in output.iterdir():
        if item.is_file():
            item.chmod(0o444)
    ledger.path.chmod(0o444)
    output.chmod(0o555)


def comparison(catalog):
    base = json.loads(Path(catalog["sources"]["base_catalog"]["path"]).read_text())
    _, helper = modules(catalog)
    current_new = admit(catalog)
    current_old = core.admit_run(base, base["metrics"]["old15_90"])
    values, records = {}, []
    for kind in ("old", "new"):
        values[kind] = {}
        for dimension in (90, 45):
            row = base["metrics"][f"old15_{dimension}"] if kind == "old" else catalog["metrics"][str(dimension)]
            directory = Path(row["output"])
            expected = current_old if kind == "old" else current_new
            admission_path = directory / "actual_native_source_admission.json"
            admission = json.loads(admission_path.read_text())
            expected_catalog = catalog["sources"]["base_catalog"] if kind == "old" else core.record(CATALOG)
            core.require(admission["status"] == "passed" and admission["catalog"] == expected_catalog
                         and admission["actual_native_source_admission"] == expected, "Saved metric source epoch/revision differs")
            ledger_path = directory.parent / f"{kind}15_{dimension}_CPU_imports.json"
            ledger = json.loads(ledger_path.read_text())
            core.require(ledger["status"] == "passed" and ledger["runtime_finalization_observed"] is True
                         and ledger["catalog"] == expected_catalog, "Finalized CPU source ledger required")
            report_path = directory / "evaluation.json"
            report = json.loads(report_path.read_text())
            item = report["runs"][row["run_name"]]
            core.require(item["sources"]["states"] == expected["capture_artifacts"]["physical_states.npz"]
                         and item["sources"]["result"] == expected["capture_artifacts"]["result.json"], "Saved metric capture identity differs")
            for path in (admission_path, ledger_path, report_path):
                core.require(not path.stat().st_mode & 0o222, "Sealed evidence only")
                records.append(core.record(path))
            values[kind][dimension] = item
    old = values["old"][90]["inclusive_speed_conditioned"]
    new = values["new"][90]["inclusive_speed_conditioned"]
    oc, nc, common = helper.common_supported_bins(old, new)
    output = Path(catalog["comparison"]["output"])
    core.require(not output.exists() and not output.is_symlink(), "Fresh native15 comparison required")
    output.mkdir(parents=True, exist_ok=False)
    raw_old = json.loads(Path(base["metrics"]["old15_90"]["raw_audit"]).read_text())
    raw_new = json.loads(Path(catalog["metrics"]["90"]["raw_audit"]).read_text())
    write(output / "comparison.json", {
        "schema": "public_state_prior.actual_native15_same_scene_comparison.v2", "status": "passed",
        "CPU_only": True, "GPU_used": False, "actors": 15,
        "runtime_sources": {"old": 174, "new": 183}, "counts": {"old": raw_old["counts"], "new": raw_new["counts"]},
        "primary90": {"old": old, "new": new, "old_supported_bins": oc, "new_supported_bins": nc,
                      "primary_aggregates_replaced": False}, "secondary_common_speed_bins": common,
        "pose45": {k: values[k][45]["poses"] for k in ("old", "new")}, "bound_metric_artifacts": records,
        "actual_native15_raw_revision": current_new["JSON_representation_revision"], "catalog": core.record(CATALOG),
        "all_failed_contact_idle_samples_retained": True, "same_RNG_or_bitwise_noise_claimed": False,
        "causal_single_factor_gain_claimed": False, "overall_style_gain_claimed": False, "default_promotion": False})
    (output / "comparison.json").chmod(0o444)
    output.chmod(0o555)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metric", type=int, choices=(90, 45))
    group.add_argument("--compare", action="store_true")
    group.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be disabled")
    core.require(core.record(CATALOG)["sha256"] == CATALOG_SHA, "Frozen native15 revision catalog changed")
    catalog = json.loads(CATALOG.read_text())
    allowed = validate(catalog)
    if args.prepare_only:
        core.require("torch" not in sys.modules, "Preparation must not load models")
        print(json.dumps({"status": "prepared_revision_sources_passed", "future_effect": False,
                          "actual_raw_audit_metadata_bound": True, "actual_saved_arrays_read": False,
                          "model_deserialized": False, "actual_metrics_computed": False, "GPU_used": False}))
    elif args.compare:
        comparison(catalog)
    else:
        metric(args.metric, catalog, allowed)
    print(json.dumps({"status": "passed", "mode": "compare" if args.compare else args.metric, "GPU_used": False}))


if __name__ == "__main__":
    main()
