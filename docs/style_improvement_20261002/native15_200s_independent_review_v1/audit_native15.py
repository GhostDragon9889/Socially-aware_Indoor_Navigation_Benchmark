"""Register and audit source174 Warehouse15 200s on CPU using frozen formulas."""
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
import types

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
RUN = ROOT / "output/diffusion_amp/native_warehouse_15_clearance08_200s_v1"
SOURCE = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v7/native_pedestrian_source_v7.json"
SOURCE_SHA = "f9c007ec503236790766e768967a8b745b0dcde1a4a96dfd0b58986763ce403f"
PARENT_REGISTRATION = ROOT / "docs/style_improvement_20261002/native15_200s_registration_v1/registration.json"
LIVE_ROWS = 15 + 128


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
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


def adapt_origin_checks(tree, *, reverse=False):
    """Only four actor-count constants change; substep/foot formulas stay intact."""
    old_actor, new_actor = (15, 4) if reverse else (4, 15)
    old_rows, new_rows = (143, 132) if reverse else (132, 143)
    allowed = {"all reset origins plus128 deterministic free samples retained",
               "exact FP32 runtime reset origin queries", "exact ray roles"}
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "check" and node.args:
            label = node.args[0]
            if isinstance(label, ast.Constant) and label.value in allowed:
                for value in ast.walk(node.args[1]):
                    if isinstance(value, ast.Constant) and type(value.value) is int:
                        if value.value == old_actor:
                            value.value = new_actor
                            changes.append((label.value, old_actor, new_actor))
                        elif value.value == old_rows:
                            value.value = new_rows
                            changes.append((label.value, old_rows, new_rows))
    require(len(changes) == 4, "Exactly four declared origin-count/slice/role constants must change")
    return tree, changes


def readers():
    identity = json.loads((DOC / "reader_input_identity.json").read_text())
    for name, declared in identity.items():
        require(all(record(DOC/name)[k] == declared[k] for k in ("bytes", "sha256")), "Frozen reader copy changed: " + name)
        require(all(record(ROOT/declared["original"])[k] == declared[k] for k in ("bytes", "sha256")), "Historical reader source changed: " + name)
    base = load("native15_frozen_base", DOC / "frozen_v1.py")
    routes = load("native15_frozen_routes", DOC / "frozen_routes.py")
    source = (DOC / "frozen_v2.py").read_text()
    original = ast.dump(ast.parse(source), include_attributes=False)
    tree, changes = adapt_origin_checks(ast.parse(source))
    restored, _ = adapt_origin_checks(ast.parse(ast.unparse(tree)), reverse=True)
    require(ast.dump(restored, include_attributes=False) == original, "No old formula or gate changed beyond declared shape constants")
    module = types.ModuleType("native15_frozen_v2_shape_adapter")
    module.__file__ = str(DOC / "frozen_v2.py")
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    module.BASE, module.ROUTE = base, routes
    return module, {"four_shape_constant_changes": changes, "restored_full_AST_equals_original": True,
                    "frozen_base_and_route_bytes_unchanged": True}


def source_binding():
    require(record(SOURCE)["sha256"] == SOURCE_SHA, "Exact source174 manifest SHA required")
    manifest = json.loads(SOURCE.read_text())
    require(manifest["source_count"] == len(manifest["files"]) == 174, "Expected exact174 source paths")
    archive_path = ROOT / manifest["archive"]["path"]
    require(all(record(archive_path)[k] == manifest["archive"][k] for k in ("bytes", "sha256")), "Frozen174 source archive changed")
    names = [v["path"] for v in manifest["files"]]
    require(len(set(names)) == 174 and all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names), "Bounded174 unique relative paths")
    current, mismatches = [], []
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == 174 and {m.name for m in members} == set(names) and all(m.isfile() for m in members), "Exact174 regular archive members")
        for row in manifest["files"]:
            raw = tar.extractfile(row["path"]).read()
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "Frozen source member changed: " + row["path"])
            observed = record(ROOT/row["path"])
            current.append(observed)
            if any(observed[k] != row[k] for k in ("bytes", "sha256")):
                mismatches.append(row["path"])
    return {"manifest": record(SOURCE), "archive": record(archive_path), "source_count": 174,
            "all_archive_members_exact": True, "current_source_observation": current,
            "current_mismatched_paths": mismatches, "not_full_runtime_import_ledger": True}


def validate_entry(entry):
    args = entry["parsed_driver_arguments"]
    require(entry["status"] == "SDK_free_preflight_passed" and entry["physical_execution_verified"] is False,
            "Entry must be actual SDK-free preflight, not a terminal physics claim")
    require(entry["control_steps"] == args["evaluate_steps"] == 6000 and args["num_envs"] == 15
            and entry["physical_transitions_planned"] == 90000 and entry["actual_duration_seconds"] == 200,
            "Exact15 actors/6000 controls/90000 planned actor controls required")
    require(Path(args["output"]).resolve() == RUN and args["physics_only"] is True
            and args["measure_ground_support"] is True and args["pairwise_contacts"] is True
            and args["deterministic_evaluation"] is True and args["diffusion_samples"] == 1
            and args["ground_support_threshold_n"] == args["contact_threshold_n"] == 1
            and args["power_source"] == "projected_joint_effort_substep_mean", "Registered measured deterministic deployment contract")
    return args


def registration():
    _, proof = readers()
    binding = source_binding()
    require(not binding["current_mismatched_paths"], "Source174 must match at registration")
    parent = json.loads(PARENT_REGISTRATION.read_text())
    require(parent["status"] == "registered_actual_execution_pending" and parent["source"]["sha256"] == SOURCE_SHA
            and parent["actors"] == 15 and parent["control_steps"] == 6000, "Exact parent15 registration required")
    entry_path = RUN / "deployment_entry.json"
    entry = json.loads(entry_path.read_text())
    args = validate_entry(entry)
    for key in ("configuration_file", "preflight", "native_map", "probe_script_snapshot"):
        declared = entry["native_scene_request"][key]
        require(record(declared["path"]) == declared, "Registered native scene resource changed: " + key)
    return {"schema": "native_Warehouse15_200s_independent_CPU_registration.v1",
            "status": "prepared_actual_terminal_artifacts_pending", "created_utc": datetime.now(timezone.utc).isoformat(),
            "CPU_only": True, "SDK_or_GPU_started": False, "native_execution_audited": False,
            "actual_quality": None, "actual_counts": None, "run": str(RUN), "source_binding": binding,
            "parent_registration": record(PARENT_REGISTRATION), "deployment_entry": record(entry_path),
            "parsed_driver_arguments": args, "reader": record(__file__), "shape_only_reader_proof": proof,
            "expected_counts": {"controls": 6000, "actors": 15, "duration_s": 200, "actor_controls": 90000,
                "shared_physics_substeps": 24000, "actor_physics_substeps": 360000, "feet_observations": 1440000,
                "static_denominator_actor_controls": 90000, "peer_denominator_unique_pair_controls": 630000,
                "live_floor_rays": LIVE_ROWS, "live_floor_reset_origins": 15, "live_free_samples": 128}}


def audit(registered):
    require(registered["schema"] == "native_Warehouse15_200s_independent_CPU_registration.v1"
            and registered["native_execution_audited"] is False and record(__file__) == registered["reader"], "Exact pending registration/reader bytes required")
    require(record(RUN/"deployment_entry.json") == registered["deployment_entry"]
            and record(PARENT_REGISTRATION) == registered["parent_registration"], "Registered preflight unchanged")
    result = json.loads((RUN/"result.json").read_text())
    require(result["arguments"] == registered["parsed_driver_arguments"], "Actual terminal argv equals registered15 argv")
    module, proof = readers()
    report = module.audit(RUN, SOURCE)
    expected = registered["expected_counts"]
    for key in ("controls", "actors", "duration_s", "actor_controls", "shared_physics_substeps", "actor_physics_substeps",
                "static_denominator_actor_controls", "peer_denominator_unique_pair_controls"):
        require(report["counts"][key] == expected[key], "Actual complete raw denominator: " + key)
    ground = module.BASE.arrays(RUN/"ground_support.npz")
    require(ground["foot_floor_normal_force_w"].shape == (6000, 4, 15, 4, 1, 3), "1440000 exact per-foot/substep/floor observations")
    require(report["counts"]["first_counter"] == 0 and report["counts"]["last_counter"] == 24000, "All24000 submitted physics counters retained")
    capture = result["ground_support_measurement"]["capture"]
    require(capture["completed_controls"] == 6000 and capture["completed_physics_substeps"] == 24000
            and capture["actors"] == 15 and capture["all_controls_valid"] is True, "Full actual15 capture contract")
    for name, shape in capture["array_shapes"].items():
        require(list(ground[name].shape) == shape, "Actual capture shape: " + name)
    report["counts"]["feet_observations"] = 1440000
    report["registration"] = registered
    report["source_post_observation"] = source_binding()
    report["shape_only_reader_proof"] = proof
    report["execution_reader"] = record(__file__)
    report["evidence_limits"]["per_query_counter_rows_retained"] = LIVE_ROWS
    report["evidence_limits"]["scope"] = "All6000 controls/15 actors/four native substeps/true COM/floor/static/peer channels and143 pre-reset live support rays. Old v1/v2/route mathematics unchanged; only four origin-count/slice/role constants changed from4 to15 and132 to143. CPU does not reopen Kit."
    report["evidence_limits"]["causality"] = "Complete actual falls, contacts, resets and within-episode route cycles retained; integrity and effect quality reported separately."
    report["schema"] = "native_Warehouse15_200s_independent_CPU_audit.v1"
    report["native_200s_artifacts_independently_audited"] = True
    require(not any(n in sys.modules for n in ("omni", "isaacsim", "isaaclab", "torch")), "Reader must not import prohibited SDK/Torch")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path, default=DOC/"registration.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing report")
    report = registration() if args.register_only else audit(json.loads(args.registration.read_text()))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "native_execution_audited": report.get("native_200s_artifacts_independently_audited", False),
                      "counts": report.get("counts"), "expected_counts": report.get("expected_counts")}))


if __name__ == "__main__":
    main()
