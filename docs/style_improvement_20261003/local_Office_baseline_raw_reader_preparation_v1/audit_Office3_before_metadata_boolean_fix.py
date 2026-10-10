"""CPU-only Office3/300 reader: original raw math, registered metadata adapters."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from office_evidence import (admit_actual, load, read, record, require, run_path, source_evidence)  # noqa: E402


def replace_exact(tree, before, after, expected=1):
    """Replace only a preregistered metadata AST expression, never raw math."""
    signature = ast.dump(ast.parse(before, mode="eval").body, include_attributes=False)
    replacement = ast.parse(after, mode="eval").body

    class Replace(ast.NodeTransformer):
        count = 0

        def generic_visit(self, node):
            if ast.dump(node, include_attributes=False) == signature:
                self.count += 1
                return replacement
            return super().generic_visit(node)

    transform = Replace()
    result = transform.visit(tree)
    require(transform.count == expected, "Exact metadata adapter site count differs: " + before)
    return ast.fix_missing_locations(result)


def readers():
    identity = read(HERE / "reader_input_identity.json")
    for name, row in identity.items():
        for path in (Path(row["original"]), HERE / name):
            actual = record(path)
            require(actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"],
                    "Original immutable raw reader bytes differ")
    changes_base = [
        ('members[row["path"]]', 'members[str(Path(row["path"]).relative_to(ROOT))]'),
        ('len(static) == provider["static_collider_count"] == 745', 'len(static) == provider["static_collider_count"] == 3642'),
    ]
    changes_native = [
        ('live["reset_origins_tested"] == len(origins) == 4', 'live["reset_origins_tested"] == len(origins) == 3'),
        ('len(rows) == 132', 'len(rows) == 131'),
        ('rows[:4]', 'rows[:3]'),
        ('["reset_origin"] * 4 + ["deterministic_admitted_free_sample"] * 128',
         '["reset_origin"] * 3 + ["deterministic_admitted_free_sample"] * 128'),
        ('bool(live["authored_source_physics_scenes_inactive"])', 'live["authored_source_physics_scenes_inactive"] == []'),
        ('ROUTE.analyze_arrays(physical, result)', 'ROUTE.analyze_arrays(physical, result, vertices_per_cycle=3)'),
    ]
    modules = []
    proof = []
    for name, changes in (("frozen_v1.py", changes_base), ("frozen_v2.py", changes_native)):
        original = ast.parse((HERE / name).read_text())
        transformed = ast.parse((HERE / name).read_text())
        for before, after in changes:
            transformed = replace_exact(transformed, before, after)
        restored = ast.parse(ast.unparse(transformed))
        for before, after in reversed(changes):
            restored = replace_exact(restored, after, before)
        require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
                "Metadata adapters changed another original raw formula or check")
        module = types.ModuleType("Office_metadata_only_" + name.replace(".", "_"))
        module.__file__ = str(HERE / name)
        exec(compile(transformed, module.__file__, "exec"), module.__dict__)
        modules.append(module)
        proof.append({"file": name, "exact_expression_changes": changes,
                      "reverse_restores_original_complete_AST": True})
    base, native = modules
    routes = load(HERE / "frozen_routes.py", "Office_original_route_math")
    native.BASE, native.ROUTE = base, routes
    return native, {"metadata_adapters": proof, "original_math_files": identity,
                    "foot_contact_COM_fall_route_math_bytes_unchanged": True,
                    "four_substeps_four_feet_reshape_constants_unchanged": True}


def own_records():
    return [record(HERE / name) for name in ("audit_Office3.py", "office_evidence.py", "reader_input_identity.json",
                                            "frozen_v1.py", "frozen_v2.py", "frozen_routes.py")]


def registration():
    plan, _, evidence = source_evidence()
    _, proof = readers()
    return {
        "schema": "local_Office.baseline62c4_short_reader_registration.v1", "status": "prepared_not_audited",
        "source_evidence": evidence, "run": str(run_path()), "reader_files": own_records(), "adapter_proof": proof,
        "expected_counts": plan["expected_counts"], "route_vertices_per_cycle": 3,
        "falls_or_contacts_are_integrity_launch_filters": False,
        "failed_controls_or_negative_heights_filtered": False, "actual_Office_gravity_or_stability_observed": False,
        "SDK_started": False, "GPU_used": False, "future_Office_capture_read": False,
    }


def audit(registered):
    require(registered["schema"] == "local_Office.baseline62c4_short_reader_registration.v1"
            and registered["reader_files"] == own_records() and registered["expected_counts"]["controls"] == 300,
            "Exact registered short reader sources/counts required")
    admission = admit_actual(registered)
    native, proof = readers()
    report = native.audit(run_path(), registered["source_evidence"]["production_source"]["manifest"]["path"])
    counts = report["counts"]
    expected = {"controls": 300, "actors": 3, "duration_s": 10.0, "actor_controls": 900,
                "shared_physics_substeps": 1200, "actor_physics_substeps": 3600,
                "static_denominator_actor_controls": 900, "peer_denominator_unique_pair_controls": 900}
    require(all(type(counts[key]) is type(value) and counts[key] == value for key, value in expected.items()),
            "Actual short-case denominators differ")
    ground = native.BASE.arrays(run_path() / "ground_support.npz")
    require(ground["foot_floor_normal_force_w"].shape == (300, 4, 3, 4, 1, 3)
            and counts["first_counter"] == 0 and counts["last_counter"] == 1200,
            "Complete300controls/four120Hzsubsteps/all3actors foot data required")
    result = read(run_path() / "result.json")
    gravity = result["native_scene_live_observation"]["local_configured_gravity_observation"]
    require(gravity["schema"] == "local_Office.configured_gravity_observation.v1"
            and gravity["status"] == "observed_consistent_singleton_config_and_active_USD_gravity"
            and gravity["configured_gravity_m_s2"] == [0.0, 0.0, -9.81]
            and gravity["submitted_step_counter_before"] == gravity["submitted_step_counter_after"] == 0
            and gravity["physics_steps_advanced_by_observer"] == 0
            and gravity["native_engine_gravity_measured"] is False,
            "Actual saved configured-gravity observation missing or incompatible")
    report["schema"] = "local_Office.baseline62c4_short_independent_CPU_audit.v1"
    report["execution_reader"] = record(__file__)
    report["metadata_admission_helper"] = record(HERE / "office_evidence.py")
    report["actual_source_admission"] = admission
    report["adapter_proof"] = proof
    report["counts"]["feet_observations"] = 14400
    report["counts"]["all_failed_actor_controls_retained"] = counts["falls"]
    report["configured_gravity_observation"] = gravity
    report["Office_short_capture_independently_audited"] = True
    report["Office_200s_stability_verified"] = False
    report["author_eight_scene_equivalence"] = False
    report["evidence_limits"]["causality"] = "Local Office short resource/support diagnostic; falls and contacts are retained and do not by themselves establish a model, gravity, scene, or planner cause."
    report["evidence_limits"]["scope"] = "All300controls/3actors/four120Hzsubsteps/trueCOM/floor/static/peer channels retained. Immutable original raw math; only registered source/archive index and Office counts/inventories/route vertex metadata adapt. CPU reader never opens physics or deserializes the model."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.register_only and args.registration is None:
        parser.error("Actual audit requires the prepared registration")
    document = registration() if args.register_only else audit(read(args.registration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": document["status"], "report": record(args.output),
                      "GPU_used": False, "SDK_started": False}))


if __name__ == "__main__":
    main()
