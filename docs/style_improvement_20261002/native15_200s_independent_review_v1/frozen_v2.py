"""CPU audit of retained native captures including measured live-stage rows.

The frozen v1 reader remains unchanged. Its full source/raw/COM/force checks
are reused, then the new saved live observation and actual route events are
checked. This reader never creates a stage, physics query, or GPU context.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("native_actual_audit_v1", ROOT / "docs/style_improvement_20261002/native_scene_sdk_review_v1/audit_actual_capture.py")
ROUTE = _load("native_recorded_route_audit", ROOT / "docs/amp_improvement_20261001/analyze_route_completion.py")


def _check_ray(row, *, expected_paths, floor_paths, robot_roots, height, counter):
    xy = np.asarray(row["world_xy_m"], dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Native saved ray requires finite world XY")
    if row["counter_before"] != counter or row["counter_after"] != counter:
        raise ValueError("Saved ray advanced the submitted-step counter")
    support = False
    for hit in row["all_hits"]:
        position = np.asarray(hit["position"], dtype=np.float64)
        normal = np.asarray(hit["normal"], dtype=np.float64)
        distance = float(hit["distance"])
        if (position.shape != (3,) or normal.shape != (3,)
                or not np.isfinite(np.r_[position, normal, distance]).all()
                or distance < 0 or not np.isclose(np.linalg.norm(normal), 1., atol=1e-3, rtol=0)):
            raise ValueError("Saved native ray hit has invalid geometry")
        path = hit["collision"]
        near = normal[2] >= .9 and abs(position[2] - height) <= .08
        dynamic = any(path == root or path.startswith(root + "/") for root in robot_roots)
        if dynamic:
            expected = "excluded_exact_dynamic_robot"
        elif path not in expected_paths:
            raise ValueError("Saved native ray contains foreign exact collision path")
        elif path in floor_paths:
            expected = "selected_floor_support" if near else "selected_floor_other_surface"
        elif near:
            raise ValueError("Saved native ray has undeclared coplanar support")
        else:
            expected = "known_static_nonfloor_surface"
        if hit["classification"] != expected:
            raise ValueError("Saved native ray classification differs from geometry/path")
        support |= expected == "selected_floor_support"
    if not support:
        raise ValueError("Saved native all-hit ray lacks selected floor support")


def audit(run, source_manifest):
    report = BASE.audit(run, source_manifest)
    run = Path(run).resolve()
    result = json.loads((run / "result.json").read_text())
    physical = BASE.arrays(run / "physical_states.npz")
    provider, live = result["native_scene_provider"], result["native_scene_live_observation"]
    additional = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        additional.append(name)

    check("completed driver status", result["status"] == "passed")
    check("actual live observation schema/status", live["schema"] == "dynbench_live_runtime_observation.v1"
          and live["status"] == "passed_live_stage_and_sampled_native_floor_ownership")
    check("observed same singleton/current stage", live["observed_current_singleton_context_identity"] is True
          and live["observed_active_stage_matches_context"] is True)
    check("observed nonempty live stage identifiers", bool(live["stage_root_identifier"]) and bool(live["stage_session_identifier"]))
    check("no new App or Context created by verifier", live["new_App_or_Context_created_by_verifier"] is False)
    check("historical constructor counts honestly unmeasured", live["App_creation_count_observed"] is None
          and live["SimulationContext_creation_count_observed"] is None)
    check("exact source composed once", live["source_session_occurrences"] == 1)
    check("observed source identity matches provider", live["source_USD"] == provider["source_USD"])
    check("full source USD bytes rehashed", BASE.record(live["source_USD"]["path"]) == live["source_USD"])
    check("source layer before/after unchanged recorded", live["source_layers_rehashed_before_after_unchanged"] is True)
    preflight = json.loads(Path(provider["preflight"]["path"]).read_text())
    check("exact all recorded source layers", live["source_layer_records"] == preflight["layers"])
    for row in live["source_layer_records"]:
        check("live source layer full bytes " + row["path"], BASE.record(row["path"]) == row)
    source_roots = live["source_root_paths"]
    check("observed source roots retained uniquely as absolute paths", bool(source_roots)
          and len(source_roots) == len(set(source_roots)) and all(path.startswith("/") for path in source_roots))
    check("exact sole active live physics scene", len(live["active_physics_scene_paths"]) == 1
          and not set(live["active_physics_scene_paths"]).intersection(live["authored_source_physics_scenes_inactive"]))
    check("recorded authored source PhysicsScenes inactive at source paths", bool(live["authored_source_physics_scenes_inactive"])
          and all(any(path == root or path.startswith(root + "/") for root in source_roots)
                  for path in live["authored_source_physics_scenes_inactive"]))
    expected = set(result["physical_contact_measurement"]["static_collider_paths"]) | set(provider["floor_paths"])
    check("exact enabled collision inventory matches observed contact filters", set(live["enabled_static_and_floor_collision_paths"]) == expected
          and len(live["enabled_static_and_floor_collision_paths"]) == len(expected))
    check("sole selected floor identity observed", live["floor_paths"] == provider["floor_paths"] == ["/__PaperMapProbeReferenceGround"])
    check("all original Plane/decals disabled", live["disabled_original_flat_ground_paths"] == provider["disabled_original_flat_ground_paths"])
    check("no foreign/generated terrain reported", live["generated_or_foreign_terrain_collision_paths"] == [])
    check("actual finite floor geometry/transform matches recipe", live["observed_floor_geometry_matches_recipe"] is True)
    check("actual floor material matches preflight recipe", live["observed_floor_material"] == preflight["reference_ground"]["material"])
    check("actual floor material matches runtime report", all(result["physics"]["terrain_material"][key] == value
          for key, value in live["observed_floor_material"].items()))
    check("exact dynamic robots excluded", live["exact_dynamic_robot_roots"] == provider["exact_dynamic_robot_roots"])
    before, after = live["submitted_step_counter_before"], live["submitted_step_counter_after"]
    check("live queries happened before first submitted physics step", before == after == 0 and live["query_physics_advance"] == 0)
    check("native engine timestamp not invented", live["native_engine_time_observed"] is False
          and live["native_engine_time_before_after_delta"] is None)
    rows = live["native_floor_ray_rows"]
    origins = result["evaluation_scene"]["origins_xy"]
    check("all reset origins plus128 deterministic free samples retained", live["reset_origins_tested"] == len(origins) == 4
          and live["optional_admitted_free_samples"] == 128 and len(rows) == 132)
    check("exact FP32 runtime reset origin queries", np.array_equal(
        np.asarray([row["world_xy_m"] for row in rows[:4]], dtype=np.float32), np.asarray(origins, dtype=np.float32)))
    check("exact ray roles", [row["role"] for row in rows] == ["reset_origin"] * 4 + ["deterministic_admitted_free_sample"] * 128)
    for index, row in enumerate(rows):
        _check_ray(row, expected_paths=expected, floor_paths=provider["floor_paths"],
                   robot_roots=live["exact_dynamic_robot_roots"], height=provider["ground_height_m"], counter=before)
        check("actual all-hit ownership geometry/counter row " + str(index), True)
        check("exactly one observed selected-floor support hit row " + str(index),
              sum(hit["classification"] == "selected_floor_support" for hit in row["all_hits"]) == 1)

    # Independently count ordered within-episode cycles; total arrivals >=4
    # across actors or reset episodes are never treated as a completed route.
    routes = ROUTE.analyze_arrays(physical, result)
    check("all route events/index/episode continuity valid", routes["status"] == "checked" and routes["reported_count_match"] is True)
    route_xy = np.asarray(result["evaluation_scene"]["routes_xy"], dtype=physical["route_vertex_xy"].dtype)
    arrival_radius = result["physics"]["scene_geometry"]["route_arrival_radius_m"]
    distances = []
    for frame, actor in np.argwhere(physical["route_vertex_reached"]):
        vertex = (physical["route_indices"][frame, actor] - 1) % route_xy.shape[1]
        target = route_xy[actor, vertex]
        check(f"actual reached vertex identity {frame}/{actor}", np.array_equal(target, physical["route_vertex_xy"][frame, actor]))
        distance = float(np.linalg.norm(physical["root_pos"][frame, actor, :2] - target))
        check(f"actual reached vertex physical distance {frame}/{actor}", distance <= arrival_radius + 1e-6)
        distances.append({"frame0": int(frame), "actor0": int(actor), "vertex0": int(vertex), "distance_m": distance})
    threshold = result["physical_contact_measurement"]["threshold_n"]
    check("strict native static threshold raw force", np.array_equal(physical["static_contact"], physical["static_contact_max_force_n"] > threshold))
    check("strict native peer threshold raw force", np.array_equal(physical["peer_contact"], physical["peer_contact_max_force_n"] > threshold))
    check("peer exact symmetric force/mask", np.array_equal(physical["peer_contact"], physical["peer_contact"].swapaxes(-1, -2))
          and np.array_equal(physical["peer_contact_max_force_n"], physical["peer_contact_max_force_n"].swapaxes(-1, -2)))
    check("read did not mutate result", BASE.record(run / "result.json") == report["artifacts"]["result.json"])
    report["schema"] = "native_pedestrian_world_independent_audit.v2"
    report["checks"].extend(additional)
    report["checks_passed"] = len(report["checks"])
    report["reader_sources"] = {"reader": BASE.record(__file__), "frozen_base_reader": BASE.record(BASE.__file__), "route_reader": BASE.record(ROUTE.__file__)}
    report["actual_live_observation"] = live
    report["route_analysis"] = routes
    report["arrival_geometry"] = {"radius_m": arrival_radius, "events": distances}
    report["counts"]["complete_cycles"] = routes["complete_cycles_per_actor"]
    report["counts"]["complete_cycles_total"] = routes["complete_cycles_total"]
    report["physical"]["max_peer_force_n"] = float(physical["peer_contact_max_force_n"].max())
    report["evidence_limits"] = {
        "fresh_runtime_floor_inventory_and_all_hit_ownership_observed": True,
        "measured_single_App_or_context_construction_count": None,
        "current_context_stage_identity_observed": True,
        "per_query_counter_rows_retained": 132,
        "native_engine_time_observed": False,
        "native_engine_time_before_after_delta": None,
        "scope": "Saved live stage/floor inventory and132 all-hit support queries were measured before initial reset, independently checked here. All900 controls/four native substeps/true COM/floor and static/peer force channels are retained. This reader does not reopen or remeasure the old live stage.",
        "causality": "This bounded30-second zero-fall run does not identify causes of historical failures or isolate clearance from changed spawn/route geometry.",
        "PhysX_Warp_bitwise_guarantee": False,
        "thirty_second_run_is_200second_quality_proof": False,
        "declared_result_status_passed_means": "driver completed requested steps; actual falls/contacts/arrivals/cycles are reported separately"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Preserve existing independent audit: " + str(destination))
    report = audit(args.run, args.source_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"], "counts": report["counts"],
                      "COM_max_error_m": report["support"]["COM_recomposition_max_error_m"]}))


if __name__ == "__main__":
    main()
