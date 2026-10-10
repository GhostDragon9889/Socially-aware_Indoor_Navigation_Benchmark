"""Register/audit complete native4/15 captures on CPU with frozen mathematics."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import types

from native_evidence import (HERE, SOURCE, admit_closed_native, expected_counts, prepare_evidence,
                             read, record, require, run_path, source_binding)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_index_adapter(tree, reverse=False):
    """Only map absolute manifest row paths to relative USTAR member names."""
    before = ast.parse('members[row["path"]]', mode="eval").body
    after = ast.parse('members[str(Path(row["path"]).relative_to(ROOT))]', mode="eval").body
    if reverse:
        before, after = after, before
    expected = ast.dump(before, include_attributes=False)

    class Replace(ast.NodeTransformer):
        count = 0

        def visit_Subscript(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                self.count += 1
                return after
            return self.generic_visit(node)

    replacement = Replace()
    tree = replacement.visit(tree)
    require(replacement.count == 1, "Exactly one source archive index adaptation required")
    return ast.fix_missing_locations(tree)


def origin_shape_adapter(tree, reverse=False):
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
    require(len(changes) == 4, "Exactly four origin-count/slice/role constants must change")
    return tree, changes


def readers(actors):
    identity = read(HERE / "reader_input_identity.json")
    for name, row in identity.items():
        require(record(row["original"])["bytes"] == record(HERE / name)["bytes"] == row["bytes"] and record(row["original"])["sha256"] == record(HERE / name)["sha256"] == row["sha256"], "Frozen original/copy reader changed")
    base_source = (HERE / "frozen_v1.py").read_text()
    base_tree = source_index_adapter(ast.parse(base_source))
    restored = source_index_adapter(ast.parse(ast.unparse(base_tree)), reverse=True)
    require(ast.dump(restored, include_attributes=False) == ast.dump(ast.parse(base_source), include_attributes=False), "Source index adapter changed another base formula/gate")
    base = types.ModuleType("state_prior_native_frozen_v1_absolute_path_adapter")
    base.__file__ = str(HERE / "frozen_v1.py")
    exec(compile(base_tree, base.__file__, "exec"), base.__dict__)
    routes = load("state_prior_native_frozen_routes", HERE / "frozen_routes.py")
    native_source = (HERE / "frozen_v2.py").read_text()
    native_tree = ast.parse(native_source)
    changes = []
    if actors == 15:
        native_tree, changes = origin_shape_adapter(native_tree)
        restored, _ = origin_shape_adapter(ast.parse(ast.unparse(native_tree)), reverse=True)
        require(ast.dump(restored, include_attributes=False) == ast.dump(ast.parse(native_source), include_attributes=False), "Origin adapter changed another v2 formula/gate")
    native = types.ModuleType("state_prior_native_frozen_v2_shape_adapter")
    native.__file__ = str(HERE / "frozen_v2.py")
    exec(compile(native_tree, native.__file__, "exec"), native.__dict__)
    native.BASE, native.ROUTE = base, routes
    proof = {"base_source_archive_index_changes": 1, "base_reverse_restored_full_AST_equals_original": True,
                    "origin_shape_constant_changes": changes, "v2_reverse_restored_full_AST_equals_original": True,
                    "route_reader_bytes_and_AST_unchanged": True, "COM_force_fall_route_math_unchanged": True,
                    "reader_inputs": identity}
    return native, json.loads(json.dumps(proof, allow_nan=False))


def reader_records():
    return [record(HERE / name) for name in ("audit_state_prior_native.py", "native_evidence.py", "reader_input_identity.json", "frozen_v1.py", "frozen_v2.py", "frozen_routes.py")]


def registration(actors):
    _, proof = readers(actors)
    evidence = prepare_evidence(actors)
    return {"schema": "pacer_reward.native200s_CPU_reader_registration.v1", "status": "prepared_actual_terminal_artifacts_pending",
            "created_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False, "SDK_started": False,
            "native_execution_audited": False, "actual_quality": None, "actors": actors, "run": str(run_path(actors)),
            "evidence": evidence, "reader_files": reader_records(), "adapter_proof": proof,
            "registration_timing": "Prepared before actual new4/15 captures; checkpoint SHA and actual terminal/native evidence remain pending. Actual closed owner records are required for execution.",
            "expected_counts": expected_counts(actors),
            "falls_zero_required_for_integrity_admission": False, "failed_contact_idle_or_reset_actor_frames_filtered": False}


def audit(actors, registered):
    require(registered["schema"] == "pacer_reward.native200s_CPU_reader_registration.v1" and reader_records() == registered["reader_files"], "Exact pending registration and frozen reader bytes required")
    admission = admit_closed_native(actors, registered)
    native, proof = readers(actors)
    require(proof == registered["adapter_proof"], "Frozen mathematics adapter proof changed")
    report = native.audit(run_path(actors), SOURCE)
    counts, expected = report["counts"], registered["expected_counts"]
    for key in ("controls", "actors", "duration_s", "actor_controls", "shared_physics_substeps", "actor_physics_substeps", "static_denominator_actor_controls", "peer_denominator_unique_pair_controls"):
        require(counts[key] == expected[key], "Complete raw denominator differs: " + key)
    physical = native.BASE.arrays(run_path(actors) / "physical_states.npz")
    ground = native.BASE.arrays(run_path(actors) / "ground_support.npz")
    result = read(run_path(actors) / "result.json")
    capture = result["ground_support_measurement"]["capture"]
    require(counts["first_counter"] == 0 and counts["last_counter"] == 24000 and ground["foot_floor_normal_force_w"].shape == (6000, 4, actors, 4, 1, 3) and capture["completed_controls"] == 6000 and capture["completed_physics_substeps"] == 24000 and capture["actors"] == actors and capture["all_controls_valid"] is True, "All6000/24000/foot observations required")
    for name, shape in capture["array_shapes"].items():
        require(list(ground[name].shape) == shape, "Actual ground capture metadata shape differs")
    counts["feet_observations"] = expected["feet_observations"]
    counts["all_raw_failed_actor_controls_retained"] = int(physical["terminated"].sum())
    counts["all_raw_time_limit_actor_controls_retained"] = int(physical["truncated"].sum())
    require(counts["all_raw_failed_actor_controls_retained"] == counts["falls"], "Raw failures omitted")
    report["schema"] = "pacer_reward.native200s_independent_CPU_audit.v1"
    report["native_200s_artifacts_independently_audited"] = True
    report["actual_source_admission"] = admission
    report["registration"] = registered
    report["adapter_proof"] = proof
    report["source_post_observation"] = source_binding()
    require(report["source_post_observation"] == registered["evidence"]["production_source"] and report["artifacts"]["result.json"] == admission["result"], "Actual source/result changed during complete raw read")
    report["execution_reader"] = record(__file__)
    report["evidence_limits"]["per_query_counter_rows_retained"] = expected["live_floor_rays"]
    report["evidence_limits"]["scope"] = "All6000 controls and every actor/four substep/trueCOM/floor/static/peer channel retained, with actual pre-reset all-hit floor rows. Exact old v1/v2/routes mathematics; one absolute-source/archive-index adapter and for15 four origin-count/slice/role constants. CPU reader does not reopen physics."
    report["evidence_limits"]["causality"] = "Actual falls, contacts, resets and route cycles retained independently of integrity status. Same public parameters do not imply bitwise matched planning noise: old1950/new2019 network initialization consumes RNG differently. No causal style gain or default promotion."
    report["evidence_limits"]["complete_runtime_import_ledger"] = False
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "CPU reader imported SDK/model")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, choices=(4, 15), required=True)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous report")
    require(args.register_only or args.registration is not None, "Actual stage requires frozen registration")
    report = registration(args.actors) if args.register_only else audit(args.actors, read(args.registration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "actual_native_audited": report.get("native_200s_artifacts_independently_audited", False), "counts": report.get("counts"), "expected_counts": report.get("expected_counts")}))


if __name__ == "__main__":
    main()
