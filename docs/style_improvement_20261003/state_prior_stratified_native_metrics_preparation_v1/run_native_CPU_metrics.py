"""q98 native metadata hooks around the unchanged native 90D/45D runner."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "docs/style_improvement_20261003/state_prior_native_metrics_preparation_v1"
sys.path.insert(0, str(BASE))
import native_metric_sources as core  # noqa: E402

CATALOG = HERE / "commands_native_CPU_metrics_v1.json"
CATALOG_SHA = "b3cc11fa2484fe3c03868241c40fbc1b9afcaf9ae2488917563403caa92cd18a"


def revised(function, substitutions):
    """Only existing identity/metadata functions are recompiled; inverse is exact."""
    original = inspect.getsource(function)
    source = original
    for before, after in substitutions:
        core.require(source.count(before) == 1, "One exact metadata boundary required")
        source = source.replace(before, after)
    reverse = source
    for before, after in reversed(substitutions):
        reverse = reverse.replace(after, before)
    core.require(reverse == original, "Metadata inverse differs from the frozen function")
    namespace = dict(function.__globals__)
    exec(compile(textwrap.dedent(source), __file__, "exec"), namespace)
    return namespace[function.__name__]


def admit_run(catalog, row):
    """Require real q native GPU and raw CPU closure before any style arrays."""
    actors = row["actors"]
    bridge = core.load("q_native_actual_closed_metadata", catalog["sources"]["native_evidence"]["path"])
    plan = bridge.registration()
    case = plan["cases"][f"native{actors}"]
    registered = json.loads(Path(row["registration"]).read_text())
    core.require(registered["execution_entry"] == catalog["sources"]["q_native_reader"]
                 and registered["metadata_helper"] == catalog["sources"]["native_evidence"]
                 and registered["original_core"] == catalog["sources"]["native_reader"],
                 "Exact q native outer reader identity required")
    source = bridge.admit_closed_native(actors, registered["core_registration"])
    audit_path = Path(row["raw_audit"])
    core.require(not audit_path.stat().st_mode & 0o222, "Actual raw report must be readonly")
    audit = json.loads(audit_path.read_text())
    core.validate_raw_contract(audit, actors, 185, "stratified_prior.native200s_independent_CPU_audit.v1")
    core.require(audit.get("actual_source_admission") == source
                 and audit.get("registration") == registered["core_registration"]
                 and audit.get("q_metadata_registration") == registered
                 and audit.get("execution_reader") == catalog["sources"]["native_reader"]
                 and audit.get("execution_entry") == catalog["sources"]["q_native_reader"]
                 and audit.get("metadata_admission_helper") == catalog["sources"]["native_evidence"]
                 and audit.get("compiled_raw_math_changed") is False,
                 "Actual q raw report must bind original mathematics and actual metadata entry")
    cpu_path = Path(row["raw_CPU_receipt"])
    core.require(str(cpu_path) == case["CPU_receipt"] and str(audit_path) == case["audit"]
                 and str(row["input"]) == case["output"]
                 and not cpu_path.stat().st_mode & 0o222,
                 "Exact readonly q raw CPU owner and case required")
    cpu = json.loads(cpu_path.read_text())
    bridge.closed_owner(cpu, plan, f"native{actors}", cpu=True)
    core.require(cpu.get("audit_artifact") == core.record(audit_path)
                 and cpu.get("actual_capture_admission") == source,
                 "Real closed raw CPU command must bind this actual audit and source admission")
    core.require(audit["source_epoch"]["manifest"] == source["production_source"]["manifest"]
                 and audit["source_epoch"]["archive"] == source["production_source"]["archive"]
                 and Path(audit["run"]).resolve() == Path(row["input"]).resolve(),
                 "Actual q native production185/run identity differs")
    core.require(set(audit["artifacts"]) == {"result.json", "physical_states.npz", "ground_support.npz",
                                            "reference_plans.npz", "deployment_entry.json"},
                 "Original five complete native artifact records required")
    for name, artifact in audit["artifacts"].items():
        core.require(Path(artifact["path"]).absolute() == Path(row["input"]) / name
                     and not Path(artifact["path"]).stat().st_mode & 0o222, "Actual native artifact alias/unsealed")
        core.exact(artifact)
    return {"raw_audit": core.record(audit_path), "raw_status": audit["status"],
            "capture_artifacts": audit["artifacts"], "native_source_admission": source,
            "runtime_source_count": 185, "actual_raw_CPU_closure": core.record(cpu_path),
            "actual_q_metadata_entry": catalog["sources"]["q_native_reader"],
            "metric_metadata_adapter": core.record(__file__),
            "immutable_metric_math_runner": catalog["sources"]["base_runner"],
            "style_success_is_quality_success": False, "default_promotion": False,
            "MMD_native_launch_filter": False}


def wire():
    runner = core.load("immutable_native_style_math_runner", BASE / "run_native_CPU_metrics.py")
    runner.HERE, runner.CATALOG, runner.CATALOG_SHA = HERE, CATALOG, CATALOG_SHA
    runner.METRICS = tuple(f"new{actors}_{dimension}" for actors in (4, 15) for dimension in (90, 45))
    runner.validate_catalog = revised(runner.validate_catalog, (
        ('"public_state_prior.prepared_native_CPU_metric_commands.v1"',
         '"stratified_prior.prepared_native_CPU_metric_commands.v1"'),
        ('"native_warehouse_state_prior"', '"native_warehouse_stratified_prior"'),
        ('state_prior_native_metrics_v1/{key}', 'state_prior_stratified_native_metrics_v1/{key}'),
        ('183 if row["kind"] == "new"', '185 if row["kind"] == "new"'),
        ('state_prior_native_actual_review_v1/native{actors}_audit.json',
         'state_prior_stratified_native_actual_runs_v1/native{actors}_raw/audit.json'),
        ('state_prior_native_reader_preparation_v1/native{actors}_registration.json',
         'state_prior_stratified_native_actual_preparation_v1/native{actors}_reader_registration.json'),
    ))
    # Recompiled metadata function globals retain original runner objects. Bind only changed identities.
    runner.validate_catalog.__globals__.update(HERE=HERE, METRICS=runner.METRICS)
    runner.validate_engine = revised(core.validate_engine, (
        ('"production_V11"', '"production_V12"'),
        ('== 183, "Exact production V11 required"', '== 185, "Exact production V12 required"'),
        ('    exact(freeze["archive"])',
         '    if "archive" in freeze:\n        exact(freeze["archive"])'),
    ))
    core.CPULedger.finish = revised(core.CPULedger.finish, (
        ('"public_state_prior.native_metric_CPU_owned_sources.v1"',
         '"stratified_prior.native_metric_CPU_owned_sources.v1"'),
        ('"math_source_epoch": 183', '"math_source_epoch": 185'),
    ))
    runner.admit_run = admit_run
    return runner


if __name__ == "__main__":
    raise SystemExit(wire().main())
