"""Fixed validation-only style90/pose45 after real same-AMP EMA40000 closure.

Metadata adapters prevent the original CLI's all-split tensor validator from
loading Test34. Feature/bin/calibration and saved-center pose mathematics stay
bound to their frozen sources. This file has not been executed in preparation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CATALOG = HERE / "commands_CPU_metrics.json"
METRICS = ("style90", "pose45")
PINNED_SHA = {
    "style90_entry": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "style90_math": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
    "style_reference": "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564",
    "pose45_basis": "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb",
    "pose45_reference_evaluation": "9478afe6b16540f7624867e1d736f30f78399c84177c12d6c9b616bc496c3780",
    "pose45_tool": "208ff775d765b7d046e7e7e8c331dbb2b8d3bfdc4c1567c1df499a584af30bfa",
    "base_metric_runner": "35fc2b77d4a6f495d62ae37853d8fc15af3a27e95f7d28a758dd8eca384aedaf",
    "base_metadata": "72d8d0c821e4cac69dec036a401ce71b3328e6fd53d276e48d5e6058749929a5",
    "base_metric_catalog": "f3bed1b88466ad009fa44fa7a08ce2352115b328781753e14b5203fafd00c067",
    "raw_reader": "e003be389c98d24b2a10a5ee9a73349e087f6fc1cd450b99aa60cdd694aa1b55",
    "fixture_plan": "8124d8b033549fb4b81bbc325bca23e237939ab6220ae182814cf807a0da6b9a",
    "raw_audit": "f792375fc55f3c6fd8d50349a98760611bda204d4c6ae93da3d2ef8a37c483d7",
    "CPU_owner": "bcf5b37f5c9e1f27aeeaff5bef3b4882ded75a30f5bb5e2ae413fe4688c783e6",
    "GPU_owner": "e1ee8bc14f41a9da36514ed080e1dad2c7f4a95a5c01f7132c72c768d3ef83bc",
    "terminal_audit": "fbe1880fae2b4687165d7caeffd9bf6fd8c3cdedf57a44b26d0030185e5c1e44",
    "training_manifest_v7": "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca",
}
STYLE90_EDITS = (
    ('    from CrowdSim.humanoid.training.manifests import validate_training_manifest\n',
     '    # Full split tensor validation is replaced by the explicit subset metadata contract.\n'),
    ('    gate = validate_training_manifest(manifest, verify_package_hashes=args.verify_package_hashes)\n',
     '    gate = _fixed_validation_resource_gate(manifest, selection, artifact)\n'),
    ('        training_gate = validate_training_manifest(training_manifest, verify_package_hashes=False)\n',
     '        training_gate = _fixed_training_manifest_metadata_gate(training_manifest)\n'),
)
POSE45_EDITS = (
    ('"schema": "public_state_prior.native_fixed_pose45_metrics.v1"',
     '"schema": "formal_EMA40000.fixed_validation_pose45_metrics.v1"'),
    ('Original 45D pose metric functions and fixed b25 centers; actor count adapted to N=4/15.',
     'Original 45D pose metric functions and fixed b25 centers; complete same-AMP four-actor fixture.'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path, *, readonly=False):
    path = Path(path).absolute()
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "Regular non-symlink artifact required: " + str(path))
    require(not readonly or not before.st_mode & 0o222, "Readonly artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        require(os.fstat(stream.fileno()) == before, "Artifact changed before open: " + str(path))
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
    require(all(getattr(before, key) == getattr(after, key) == getattr(path.lstat(), key) for key in fields),
            "Artifact changed during read: " + str(path))
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


_EXECUTION_RECORD = record(__file__)


def exact(row, *, readonly=False):
    require(record(row["path"], readonly=readonly) == row, "Fixed artifact identity changed: " + row["path"])


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def function_delta(source, name, edits):
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(functions) == 1, "One original function required: " + name)
    original = ast.get_source_segment(source, functions[0])
    changed = original
    for before, after in edits:
        require(changed.count(before) == 1, "One exact function boundary required")
        changed = changed.replace(before, after, 1)
    reverse = changed
    for before, after in reversed(edits):
        require(reverse.count(after) == 1, "One exact inverse boundary required")
        reverse = reverse.replace(after, before, 1)
    require(reverse == original and ast.dump(ast.parse(reverse), include_attributes=False)
            == ast.dump(ast.parse(original), include_attributes=False), "Function whole-byte/AST inverse differs")
    compile(changed, __file__ + ":derived_" + name, "exec")
    return changed, {"function": name, "exact_replacements": len(edits), "whole_byte_AST_reverse_exact": True,
                     "original_function_sha256": hashlib.sha256(original.encode()).hexdigest(),
                     "derived_function_sha256": hashlib.sha256(changed.encode()).hexdigest()}


def sources():
    catalog_record = record(CATALOG, readonly=True)
    catalog = json.loads(CATALOG.read_text())
    require(catalog["schema"] == "formal_EMA40000.prepared_fixed_validation_style_commands.v1"
            and catalog["status"] == "prepared_not_executed" and catalog["future_effect"] is False,
            "Reviewed fixed style catalog required")
    require(catalog["entry"] == _EXECUTION_RECORD == record(__file__, readonly=True), "Imported adapter source changed")
    require(set(catalog["sources"]) == set(PINNED_SHA) and set(catalog["metrics"]) == set(METRICS),
            "Fixed source/metric inventory differs")
    for role, row in catalog["sources"].items():
        require(row["sha256"] == PINNED_SHA[role], "Pinned source/reference/capture identity differs: " + role)
        exact(row)
    require(catalog["engine_extra_sources"] == json.loads(Path(catalog["sources"]["base_metric_catalog"]["path"]).read_text())["engine_extra_sources"],
            "Original exact CPU engine extras must remain unchanged")
    for name in METRICS:
        row = catalog["metrics"][name]
        require(row["actors"] == 4 and row["dimension"] == (90 if name == "style90" else 45)
                and row["run_name"] == "formal_EMA40000_same_AMP_fixture200s"
                and row["input"] == catalog["actual_capture"]
                and row["output"] == str(ROOT / "docs/style_improvement_20261003/formal_EMA40000_same_AMP_fixture_metrics_v1" / name)
                and row["command"] == [catalog["python"], str(HERE / "run_CPU_metrics.py"), "--metric", name],
                "Exact registered capture/output/command required")
    return catalog, catalog_record


def metadata_module(name, source, filename):
    module = ModuleType(name)
    module.__file__ = str(filename)
    exec(compile(source, str(filename), "exec"), module.__dict__)
    return module


def admit(catalog):
    """Original complete terminal/source/GPU admission plus actual raw CPU closure."""
    rows = catalog["sources"]
    helper = metadata_module("_formal_style_original_metadata", Path(rows["raw_reader"]["path"]).read_text(),
                             rows["raw_reader"]["path"])
    require(record(helper.PLAN, readonly=True) == rows["fixture_plan"]
            and helper.PLAN_SHA == rows["fixture_plan"]["sha256"], "Original actual fixture plan differs")
    admission = helper.admit_capture(json.loads(helper.PLAN.read_text()), rows["terminal_audit"]["path"],
        rows["terminal_audit"]["sha256"], rows["GPU_owner"]["path"], rows["GPU_owner"]["sha256"])
    for role in ("CPU_owner", "raw_audit", "raw_reader", "fixture_plan"):
        exact(rows[role], readonly=True)
    cpu = json.loads(Path(rows["CPU_owner"]["path"]).read_text())
    raw = json.loads(Path(rows["raw_audit"]["path"]).read_text())
    require(cpu["schema"] == "formal_EMA40000.actual_fixture_CPU_root_owner.v1" and cpu["status"] == "closed"
            and type(cpu["actual_exit_code"]) is int and cpu["actual_exit_code"] == 0
            and cpu["child_exit_observed"] is True and type(cpu["actual_root_tool_session_id"]) is int
            and bool(cpu["actual_root_launch_chunk"]) and bool(cpu["actual_root_completion_chunk"])
            and cpu["command"] == catalog["actual_raw_CPU_command"] and cpu["env"] == catalog["CPU_environment"]
            and cpu["cwd"] == str(ROOT) and cpu["audit"] == rows["raw_audit"]
            and cpu["reader"] == rows["raw_reader"] and cpu["source_plan"] == rows["fixture_plan"]
            and cpu["GPU_fixture_owner"] == rows["GPU_owner"] and cpu["terminal_audit"] == rows["terminal_audit"],
            "Actual closed raw CPU root-tool evidence differs")
    require(raw["schema"] == "formal_EMA40000.independent_physical_CPU_audit.v1" and raw["status"] == "passed"
            and raw["execution_reader"] == rows["raw_reader"] and raw["source_plan"] == rows["fixture_plan"]
            and raw["actual_source_admission"] == admission == raw["sources"]
            and raw["actual_capture_independently_audited"] is True
            and raw["all_failed_contact_idle_reset_samples_retained"] is True
            and raw["mathematics_adapter_proof"]["arrays_force_COM_contacts_falls_counters_math_changed"] is False
            and bool(raw["checks"]) and all(item["passed"] is True for item in raw["checks"])
            and raw["counts"]["controls"] == 6000 and raw["counts"]["actors"] == 4
            and raw["counts"]["duration_s"] == 200. and raw["counts"]["control_actor_frames"] == 24000,
            "Complete original raw source/count audit required; quality is not an admission gate")
    require(cpu["physical_gate_counts"] == raw["physical_gate_counts"]
            and cpu["physical_zero_gate_passed"] == raw["physical_zero_gate_passed"]
            and cpu["failed_samples_retained"] is True and cpu["default_promoted"] is False,
            "Observed physical outcomes must remain unchanged")
    return admission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS)
    parser.add_argument("--check-source-only", action="store_true")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDAhidden CPU is required")
    catalog, catalog_record = sources()
    rows = catalog["sources"]
    style_source = Path(rows["style90_entry"]["path"]).read_text()
    base_source = Path(rows["base_metric_runner"]["path"]).read_text()
    style_main, style_derivation = function_delta(style_source, "main", STYLE90_EDITS)
    pose_function, pose_derivation = function_delta(base_source, "pose45", POSE45_EDITS)
    if args.check_source_only:
        require(args.metric is None, "Source inspection cannot select an actual measurement")
        print(json.dumps({"status": "source_only_ready", "catalog": catalog_record,
                          "style90_derivation": style_derivation, "pose45_derivation": pose_derivation,
                          "capture_reference_basis_arrays_or_model_deserialized": False,
                          "fixed_pose45_basis_bytes_hashed": True, "Test34_arrays_or_outputs_read": False,
                          "NN_SDK_GPU_executed": False}))
        return 0
    require(args.metric is not None, "Choose a fixed registered metric")
    require(all(os.environ.get(key) == value for key, value in catalog["CPU_environment"].items()),
            "Actual CPU environment must equal the fixed four-thread CUDAhidden recipe")
    row = catalog["metrics"][args.metric]
    output = Path(row["output"])
    require(not output.exists() and not output.is_symlink(), "Fresh exclusive metric output required")
    admission = admit(catalog)  # Before any array/reference/Torch/measurement imports.
    core_source = Path(rows["base_metadata"]["path"]).read_text()
    old, new = '"math_source_epoch": 183,', '"math_source_epoch": 185,'
    require(core_source.count(old) == 1, "One exact CPU ledger epoch metadata replacement required")
    revised_core = core_source.replace(old, new, 1)
    require(revised_core.replace(new, old, 1) == core_source, "CPU ledger metadata inverse differs")
    core = metadata_module("native_metric_sources", revised_core, __file__)
    sys.modules["native_metric_sources"] = core
    base = metadata_module("_formal_style_original_pose_runner", base_source, rows["base_metric_runner"]["path"])
    require(row["metric_argv"] == base.argv_for(row, catalog), "Original fixed metric argv differ")
    allowed = {item["path"]: item for item in admission["production185_records"] + catalog["engine_extra_sources"]}
    for item in allowed.values():
        exact(item)
    ledger = core.CPULedger(output.parent / (args.metric + "_CPU_imports.json"), allowed, catalog_record)
    ledger.activate()
    reference = json.loads(Path(rows["style_reference"]["path"]).read_text())
    reference_checks = []

    def reference_gate(manifest, selection, artifact):
        require(artifact == reference and selection == reference["selection"]
                and selection["split"] == "validation" and len(selection["windows"]) == 256
                and selection["window_frames"] == 61 and record(manifest) == reference["manifest"],
                "Complete frozen c274 validation256x61 selection/manifest required")
        document = json.loads(Path(manifest).read_text())
        clips = {clip["motion_id"]: clip for clip in document["splits"]["validation"]["clips"]}
        require(len(clips) == 22, "Only actual validation22 reference subset is permitted")
        for window in selection["windows"]:
            clip = clips[window["motion_id"]]
            require(clip["source_path"] == window["source_path"]
                    and 0 <= window["start_frame"] < window["stop_frame_exclusive"] <= clip["frames"]
                    and window["stop_frame_exclusive"] - window["start_frame"] == 61,
                    "Frozen reference window metadata differs")
        package = document["splits"]["validation"]
        path = Path(package["path"])
        path = path if path.is_absolute() else Path(manifest).parent / path
        package_row = {"path": str(path.resolve()), "bytes": package["bytes"], "sha256": package["sha256"]}
        exact(package_row)  # Only the selected validation bank; never train/test tensor validation.
        value = {"mode": "fixed_validation22_resource_and_selected_window_identity", "manifest": record(manifest),
                 "selection_artifact": rows["style_reference"], "validation_package": package_row,
                 "selected_windows": 256, "window_frames": 61, "full_all_split_tensor_validator_executed": False,
                 "Test34_bank_or_pose_or_output_read": False, "actual_selected_array_SHA_checked_by_original_loader": True}
        if reference_checks:
            require(value == reference_checks[0], "Frozen reference resource changed during metric")
        reference_checks.append(value)
        return value

    def training_gate(manifest):
        require(record(manifest) == rows["training_manifest_v7"], "Actual trainv7 manifest68d41 identity changed")
        result = json.loads(Path(admission["result"]["path"]).read_text())
        require(result["arguments"]["manifest"] == str(Path(manifest)), "Actual capture declared another manifest")
        document = json.loads(Path(manifest).read_text())
        require(document["schema"] == "navisaaclab.amass_reconstruction.v1"
                and len(document["splits"]["train"]["clips"]) == 2848,
                "Exact actual training manifest metadata required")
        return {"mode": "exact_trainv7_manifest_metadata_bound_to_actual_capture", "manifest": rows["training_manifest_v7"],
                "train_clips": 2848, "actual_capture": admission["result"],
                "full_all_split_tensor_validator_executed": False, "Test34_arrays_read": False,
                "tensor_bank_validation_by_this_metadata_gate": False}

    status = "failed"
    try:
        require(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
                "CPU metrics cannot start SDK")
        training_gate(Path(rows["training_manifest_v7"]["path"]))
        if args.metric == "style90":
            tool = importlib.import_module("scripts.evaluate_pedestrian_style")
            require(record(tool.__file__) == rows["style90_entry"], "Actual style CLI source differs")
            namespace = dict(tool.__dict__)
            namespace.update(__file__=str(Path(__file__).resolve()), _fixed_validation_resource_gate=reference_gate,
                             _fixed_training_manifest_metadata_gate=training_gate)
            exec(compile(style_main, __file__ + ":style90_metadata_delta", "exec"), namespace)
            namespace["main"](row["metric_argv"])
        else:
            reference_gate(Path(reference["manifest"]["path"]), reference["selection"], reference)
            namespace = dict(base.__dict__)
            namespace["__file__"] = str(Path(__file__).resolve())
            exec(compile(pose_function, __file__ + ":pose45_metadata_delta", "exec"), namespace)
            namespace["pose45"](row, catalog)
        reference_gate(Path(reference["manifest"]["path"]), reference["selection"], reference)
        require(admit(catalog) == admission, "Actual source/closure evidence changed during metric")
        require(sources()[1] == catalog_record, "Metric source catalog changed during execution")
        exact(rows["style_reference"])
        require(record(output / "heldout_reference.json")["sha256"] == rows["style_reference"]["sha256"],
                "Output fixed c274 reference differs")
        report = json.loads((output / "evaluation.json").read_text())
        if args.metric == "style90":
            require(report["sources"] == {"implementation": _EXECUTION_RECORD, "module": rows["style90_math"]},
                    "Actual derived entry and unchanged style math identities differ")
        else:
            require(report["source"]["runner"] == _EXECUTION_RECORD
                    and report["method"]["fit"]["refit_performed_for_this_new_run"] is False,
                    "Actual pose adapter identity or frozen-basis provenance differs")
        import torch
        require(not torch.cuda.is_initialized(), "CPU metrics initialized CUDA")
        require(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
                "CPU metrics imported SDK")
        write(output / "actual_source_admission.json", {
            "schema": "formal_EMA40000.actual_fixed_validation_style_source_admission.v1", "status": "passed",
            "metric": args.metric, "CPU_only": True, "GPU_used": False, "SDK_started": False,
            "execution_adapter": _EXECUTION_RECORD, "catalog": catalog_record, "actual_fixture_source_admission": admission,
            "raw_audit": rows["raw_audit"], "raw_CPU_owner": rows["CPU_owner"], "fixed_metric_argv": row["metric_argv"],
            "style90_derivation": style_derivation if args.metric == "style90" else None,
            "pose45_derivation": pose_derivation if args.metric == "pose45" else None,
            "CPU_ledger_metadata_only_epoch_delta": {"original_source": rows["base_metadata"], "183_to_185": True},
            "validation_resource_admission": reference_checks[0], "all_failure_idle_reset_samples_retained": True,
            "full_manifest_tensor_validator_executed": False, "Test34_bank_or_pose_or_output_read": False,
            "fixed_b25_centers_refit_performed": False, "metric_pass_is_physical_quality_pass": False,
            "default_promoted": False})
        status = "passed"
    finally:
        ledger.finish(status)
    print(json.dumps({"status": status, "metric": args.metric, "output": str(output), "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
