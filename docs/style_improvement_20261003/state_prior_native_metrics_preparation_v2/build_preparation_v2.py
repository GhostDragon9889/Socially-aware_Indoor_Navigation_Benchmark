"""Build a separate native15 catalog; keep all v1 preparation/results intact."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "docs/style_improvement_20261003/state_prior_native_metrics_preparation_v1"
sys.path.insert(0, str(BASE))
from native_metric_sources import record, require  # noqa: E402


def main():
    require(not (HERE / "source_freeze_v2.json").exists(), "Keep every preparation revision")
    old = json.loads((BASE / "commands_native_CPU_metrics_v1.json").read_text())
    revision = ROOT / "docs/style_improvement_20261003/state_prior_native_reader_json_revision_v2"
    sources = deepcopy(old["sources"])
    for key, path in {
        "base_metrics_freeze": BASE / "source_freeze_v1.json", "base_catalog": BASE / "commands_native_CPU_metrics_v1.json",
        "base_runner": BASE / "run_native_CPU_metrics.py", "base_comparison": BASE / "compare_native_metrics.py",
        "raw15_revision_freeze": revision / "source_freeze_v2.json",
        "raw15_revision_driver": revision / "audit_native15_json_revision.py",
        "raw15_revision_registration": revision / "native15_revision_registration.json",
        "native15_core_registration": ROOT / "docs/style_improvement_20261003/state_prior_native_reader_preparation_v1/native15_registration.json",
        "actual_native15_raw_audit": ROOT / "docs/style_improvement_20261003/state_prior_native_actual_review_v1/native15_audit.json"}.items():
        sources[key] = record(path)
    require(sources["raw15_revision_freeze"]["sha256"] == "837fe71a8f7574e10b8c68b707363c05bb6dfae9d6b608528c955984c8162866"
            and sources["actual_native15_raw_audit"]["sha256"] == "66f16e6dd9d36e517b95197aae83829d1db1b19ae378128016f40c4ac0d47176", "Exact actual JSON-revision evidence required")
    catalog = {"schema": "public_state_prior.prepared_native15_metric_commands.v2", "status": "prepared_not_executed",
               "future_effect": False, "CPU_only": True, "GPU_used": False, "Kit_started": False,
               "actual_raw_audit_metadata_bound": True, "actual_saved_arrays_read": False,
               "actual_metrics_computed": False, "model_deserialized": False, "default_promotion": False,
               "sources": sources, "engine_extra_sources": old["engine_extra_sources"],
               "metric_rules": old["metric_rules"], "CPU_environment": old["CPU_environment"], "metrics": {}}
    runner = HERE / "run_native15_CPU_metrics_v2.py"
    for dimension in (90, 45):
        row = deepcopy(old["metrics"][f"new15_{dimension}"])
        row["output"] = str(ROOT / f"docs/style_improvement_20261003/state_prior_native_metrics_v2/new15_{dimension}")
        row["metric_argv"][-1] = row["output"]
        row["command"] = [old["python"], str(runner), "--metric", str(dimension)]
        catalog["metrics"][str(dimension)] = row
    catalog["prepare_only_command"] = [old["python"], str(runner), "--prepare-only"]
    catalog["comparison"] = {"command": [old["python"], str(runner), "--compare"],
                             "output": str(ROOT / "docs/style_improvement_20261003/state_prior_native_metrics_v2/comparison15"),
                             "old15_outputs_are_v1_and_unchanged": True}
    path = HERE / "commands_native15_CPU_metrics_v2.json"
    require(not path.exists(), "Never silently rebind a catalog")
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    text = runner.read_text()
    require(text.count('CATALOG_SHA = "UNFROZEN"') == 1, "Runner already bound")
    runner.write_text(text.replace('CATALOG_SHA = "UNFROZEN"', f'CATALOG_SHA = "{record(path)["sha256"]}"'))
    print(json.dumps(record(path)))


if __name__ == "__main__":
    main()
