"""CPU-only registration/audit for the exact source168 Warehouse4 200s run.

The frozen v1/v2/route functions are invoked unchanged. Registration is never
a completed-physics report. No SDK, policy, checkpoint loader or GPU is used.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
RUN = ROOT / "output/diffusion_amp/native_warehouse_clearance08_200s_v1"
SOURCE = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v4.json"
SOURCE_SHA = "07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f"
ARCHIVE_SHA = "310b87c79851ac44ffe02efc282cb412bb31f27236340242e39f21724cadc9f0"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readers():
    declared = json.loads((DOC / "frozen_reader_inputs.json").read_text())
    for row in declared.values():
        if (record(row["original"]["path"]) != row["original"]
                or record(row["copy"])["sha256"] != row["original"]["sha256"]):
            raise ValueError("Frozen upstream or copied CPU reader changed")
        original = ast.dump(ast.parse(Path(row["original"]["path"]).read_text()), include_attributes=False)
        copied = ast.dump(ast.parse(Path(row["copy"]).read_text()), include_attributes=False)
        if copied != original:
            raise ValueError("Copied reader formulas or gates changed")
    base = _load("native200_frozen_v1", DOC / "frozen_v1.py")
    route = _load("native200_frozen_route", DOC / "frozen_routes.py")
    native = _load("native200_frozen_v2", DOC / "frozen_v2.py")
    # v2 originally imports these same CPU-only helpers; route all actual calls
    # to the exact byte copies so the observed execution sources are explicit.
    native.BASE, native.ROUTE = base, route
    return native


def source_binding():
    if record(SOURCE)["sha256"] != SOURCE_SHA:
        raise ValueError("Exact source168 manifest changed")
    source = json.loads(SOURCE.read_text())
    archive_record = record(ROOT / source["archive"]["path"])
    if archive_record["sha256"] != ARCHIVE_SHA or archive_record["bytes"] != source["archive"]["bytes"]:
        raise ValueError("Exact source168 prelaunch archive changed")
    if source["source_count"] != 168 or len(source["files"]) != 168:
        raise ValueError("Expected exactly168 registered source members")
    names = [row["path"] for row in source["files"]]
    if len(names) != len(set(names)) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Source archive paths must be unique, relative and nontraversing")
    current = []
    with tarfile.open(archive_record["path"], "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != 168 or {m.name for m in members} != set(names) or not all(m.isfile() for m in members):
            raise ValueError("Source archive contains unregistered, duplicate or nonregular members")
        for row in source["files"]:
            payload = archive.extractfile(row["path"]).read()
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("Captured source member changed: " + row["path"])
            observed = record(ROOT / row["path"])
            if observed["bytes"] != row["bytes"] or observed["sha256"] != row["sha256"]:
                raise ValueError("Current source168 differs from the registered pre/post scope: " + row["path"])
            current.append(observed)
    return {"manifest": record(SOURCE), "archive": archive_record, "source_count": 168,
            "all_archive_members_exact": True, "all_current_source168_bytes_exact": True,
            "current_source_records": current, "full_runtime_import_mapping_claim": False}


def registration():
    readers()
    source = source_binding()
    entry_path = RUN / "deployment_entry.json"
    entry = json.loads(entry_path.read_text())
    args = entry["parsed_driver_arguments"]
    if (entry["status"] != "SDK_free_preflight_passed" or entry["physical_execution_verified"] is not False
            or entry["control_steps"] != 6000 or entry["physical_transitions_planned"] != 24000
            or entry["actual_duration_seconds"] != 200 or args["evaluate_steps"] != 6000 or args["num_envs"] != 4
            or Path(args["output"]).resolve() != RUN or args["physics_only"] is not True
            or args["measure_ground_support"] is not True or args["pairwise_contacts"] is not True
            or args["deterministic_evaluation"] is not True or args["ground_support_threshold_n"] != 1.
            or args["contact_threshold_n"] != 1. or args["power_source"] != "projected_joint_effort_substep_mean"):
        raise ValueError("Expected the exact registered Warehouse4 6000-control measured deployment")
    for key in ("configuration_file", "preflight", "native_map", "probe_script_snapshot"):
        bound = entry["native_scene_request"][key]
        if record(bound["path"]) != bound:
            raise ValueError("Registered scene resource bytes changed: " + key)
    return {"schema": "native_Warehouse4_200s_CPU_audit_registration.v1",
            "status": "prepared_actual_terminal_artifacts_pending", "created_utc": datetime.now(timezone.utc).isoformat(),
            "CPU_only": True, "SDK_started": False, "GPU_used": False, "native_execution_audited": False,
            "physical_capture_admission": None, "measured_counts": None, "navigation_quality_claim": None,
            "run": str(RUN), "source_binding": source, "deployment_entry": record(entry_path),
            "parsed_driver_arguments": args, "native_scene_request": entry["native_scene_request"],
            "expected_counts": {"controls": 6000, "actors": 4, "duration_s": 200,
                "actor_controls": 24000, "shared_physics_substeps": 24000, "actor_physics_substeps": 96000,
                "feet_observations": 384000, "live_floor_rays": 132},
            "frozen_reader_inputs": json.loads((DOC / "frozen_reader_inputs.json").read_text()),
            "reader": record(__file__)}


def audit(registered):
    if (registered.get("schema") != "native_Warehouse4_200s_CPU_audit_registration.v1"
            or registered["native_execution_audited"] is not False or registered["run"] != str(RUN)
            or record(__file__) != registered["reader"]):
        raise ValueError("Exact pending registration/reader required")
    source = source_binding()
    if source != registered["source_binding"] or record(RUN / "deployment_entry.json") != registered["deployment_entry"]:
        raise ValueError("Registered source or deployment preflight changed before terminal read")
    result = json.loads((RUN / "result.json").read_text())
    if result["arguments"] != registered["parsed_driver_arguments"]:
        raise ValueError("Terminal executed arguments differ from the exact preflight")
    native = readers()
    report = native.audit(RUN, SOURCE)
    counts, expected = report["counts"], registered["expected_counts"]
    for key in ("controls", "actors", "duration_s", "actor_controls", "shared_physics_substeps", "actor_physics_substeps"):
        if counts[key] != expected[key]:
            raise ValueError("Terminal raw count differs from fixed protocol: " + key)
    if (counts["first_counter"] != 0 or counts["last_counter"] != 24000
            or report["evidence_limits"]["per_query_counter_rows_retained"] != 132):
        raise ValueError("Full6000/24000 counters or live132 observations are missing")
    physical, ground = native.BASE.arrays(RUN / "physical_states.npz"), native.BASE.arrays(RUN / "ground_support.npz")
    capture = result["ground_support_measurement"]["capture"]
    if (capture["completed_controls"] != 6000 or capture["completed_physics_substeps"] != 24000
            or capture["actors"] != 4 or capture["all_controls_valid"] is not True
            or ground["foot_floor_normal_force_w"].shape != (6000, 4, 4, 4, 1, 3)):
        raise ValueError("Complete raw four-foot/single-floor/substep archive required")
    for key, shape in capture["array_shapes"].items():
        if list(ground[key].shape) != shape:
            raise ValueError("Ground capture shape metadata differs from raw: " + key)
    report["counts"]["feet_observations"] = 384000
    report["counts"]["all_raw_failed_actor_controls_retained"] = int(physical["terminated"].sum())
    report["counts"]["all_raw_time_limit_actor_controls_retained"] = int(physical["truncated"].sum())
    report["registration"] = registered
    report["wrapper_source_binding_pre_post"] = {"before": registered["source_binding"], "after": source,
                                                  "all168_current_bytes_equal_before_after": True}
    report["execution_reader"] = record(__file__)
    report["evidence_limits"]["scope"] = (
        "Actual saved live context/source/floor inventory and132 pre-reset all-hit support queries; "
        "all6000 postcontrol frames/four native substeps/true COM/floor/static/peer channels checked with unchanged v1/v2 formulas. "
        "No stage/query/physics was reopened by the CPU reader; submitted counters are not observed native engine timestamps.")
    report["evidence_limits"]["causality"] = (
        "Complete200s observations are retained including every terminal/reset frame. "
        "Falls, contacts, arrivals and within-episode cycles do not establish a unique cause, author scene equivalence or all-scene success.")
    report["evidence_limits"]["thirty_second_run_is_200second_quality_proof"] = False
    report["schema"] = "native_Warehouse4_200s_independent_CPU_audit.v1"
    report["native_200s_artifacts_independently_audited"] = True
    if any(name in sys.modules for name in ("omni", "isaacsim", "isaaclab", "torch")):
        raise RuntimeError("CPU reader imported a prohibited SDK/Torch execution provider")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path, default=DOC / "registration.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve prior output; destination must be new")
    if args.register_only:
        report = registration()
    else:
        registered = json.loads(args.registration.read_text())
        report = audit(registered)
        report["registration_record"] = record(args.registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "native_200s_artifacts_independently_audited":
                      report.get("native_200s_artifacts_independently_audited", False), "counts": report.get("counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
