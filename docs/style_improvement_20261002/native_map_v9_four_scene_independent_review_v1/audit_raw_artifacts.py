"""Independently recheck actual V9 artifacts on CPU without provider admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[3]
PRODUCER = "e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82"
CHECKS = []


def require(value, name):
    if not value:
        raise ValueError(name)
    CHECKS.append(name)


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def check_file(declared):
    actual = record(declared["path"])
    require(actual == declared, "file SHA/size: " + actual["path"])
    return actual


def arrays(declared):
    check_file(declared)
    with np.load(declared["path"], allow_pickle=False) as a:
        require(all(a[k].dtype.kind != "O" for k in a.files), "no object array")
        return {k: a[k] for k in a.files}


def equal(a, b, name):
    require(np.array_equal(a, b), name)


def indices(points, low, dims):
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), "finite world positions")
    cell = np.rint((points[:, :2] - low[:2]) / .1 - .5).astype(np.int64)
    require(np.all(cell >= 0) and np.all(cell < dims[:2]), "world positions inside native XY")
    require(np.allclose(points[:, :2], low[:2] + (cell + .5) * .1, atol=2e-4, rtol=0), "official fixed world-center roundtrip")
    result = cell[:, 1] * dims[0] + dims[0] - 1 - cell[:, 0]
    require(len(np.unique(result)) == len(result), "no duplicate world centers")
    return result


def csr(a, prefix, count):
    offsets, paths = a[prefix + "offsets"], a[prefix + "collision_paths"]
    require(offsets.dtype == np.int64 and offsets.shape == (count + 1,) and offsets[0] == 0
            and np.all(np.diff(offsets) >= 0) and offsets[-1] == len(paths), "CSR complete offsets")
    require(paths.ndim == 1 and paths.dtype.kind in "US"
            and a[prefix + "rigid_body_paths"].shape == a[prefix + "classification"].shape == paths.shape, "CSR full exact path/class rows")
    return np.repeat(np.arange(count), np.diff(offsets)), paths


def fresh_layer(declared):
    check_file(declared)
    layer = Sdf.Layer.CreateAnonymous("independent_v9_export.usda")
    require(layer.ImportFromString(Path(declared["path"]).read_text()), "parse current export bytes in fresh Sdf object")
    return layer


def audit_source(data, preflight):
    layers = [check_file(v) for v in preflight["layers"]]
    require(data["source"] == preflight["source"] and check_file(data["source"]) in layers, "original source/file-layer identity")
    source = Sdf.Layer.FindOrOpen(data["source"]["path"])
    original = Usd.Stage.Open(source)
    ram = hashlib.sha256(source.ExportToString().encode()).hexdigest()
    composition, time = data["runtime_stage_composition"], data["runtime_time_range_override"]
    require(all(v == ram for v in [composition["original_source_root_in_memory_sha256"],
            composition["original_source_root_in_memory_sha256_at_query"], time["original_source_root_in_memory_sha256"],
            time["original_source_root_in_memory_sha256_at_query"]]), "original source Sdf RAM independently matches all actual phases")
    root = fresh_layer(data["exports"]["runtime_root.usda"])
    session = fresh_layer(data["exports"]["runtime_session.usda"])
    require(not root.rootPrims and list(root.subLayerPaths) == [data["source"]["path"]], "owned runtime root metadata only, one absolute original sublayer")
    require(data["exports"]["runtime_root.usda"]["sha256"] == time["source_root_in_memory_sha256_at_query"], "actual query root RAM/export SHA exact")
    tcps = original.GetTimeCodesPerSecond()
    end = max(original.GetEndTimeCode(), tcps)
    require(root.startTimeCode == session.startTimeCode == 0 and root.endTimeCode == session.endTimeCode == end
            and root.timeCodesPerSecond == tcps and root.framesPerSecond == original.GetFramesPerSecond()
            and root.pseudoRoot.GetInfo("upAxis") == "Z" and root.pseudoRoot.GetInfo("metersPerUnit") == 1., "owned root/session source FPS/units and engineering range")
    default = original.GetDefaultPrim()
    require(root.defaultPrim == (default.GetName() if default else ""), "source defaultPrim retained")
    require(composition["owned_runtime_root_anonymous"] is True and time["session_edit_target_at_query"] is True
            and composition["original_source_layers_saved"] is False and time["source_layers_saved"] is False,
            "recorded anonymous ownership and no source saves")
    require(time["observed_timeline_start_at_query_s"] == 0
            and abs(time["observed_timeline_end_at_query_s"] - end/tcps) < 1e-12
            and time["composed_stage_start_at_query_time_code"] == 0
            and time["composed_stage_end_at_query_time_code"] == end
            and time["composed_stage_time_codes_per_second_at_query"] == tcps,
            "actual timeline query range matches separately owned USD range")
    attachment = data["runtime_stage_attachment"]
    require(attachment["expected_stage_cache_id"] == attachment["observed_context_stage_cache_id"]
            and type(attachment["expected_stage_cache_id"]) is int and attachment["expected_stage_cache_id"] >= 0
            and attachment["active_context_root_identifier"] == composition["owned_runtime_root_identifier"]
            and attachment["active_context_session_identifier"] == composition["runtime_session_identifier"]
            and attachment["submitted_step_counter_before"] == attachment["submitted_step_counter_after"] == 0
            and attachment["explicit_physics_steps_issued_by_wait"] == 0
            and attachment["extra_App_or_Context_created"] is False and attachment["manual_context_or_PhysX_attach_called"] is False,
            "actual SDK context/cache/root/session bind and zero-step attach")
    ground = data["reference_ground"]
    disabled = set(ground["disabled_flat_ground_collision_paths"])
    generated = (ground["path"], "/__PaperMapProbePhysicsScene")
    deinstanced = set(ground["deinstanced_floor_roots_in_session"])
    def inspect(prim):
        path = str(prim.path)
        if any(path == p or path.startswith(p + "/") for p in generated):
            return
        old = original.GetPrimAtPath(path)
        require(bool(old) and prim.specifier == Sdf.SpecifierOver and not prim.typeName, "session source prim is existing over: " + path)
        keys = {"specifier"}
        if path in disabled:
            keys.add("apiSchemas")
        if path in deinstanced:
            keys.add("instanceable")
            require(prim.GetInfo("instanceable") is False, "declared source deinstance false")
        if old.GetTypeName() == "PhysicsScene":
            keys.add("active")
            require(prim.GetInfo("active") is False, "authored PhysicsScene inactive")
        require(set(prim.ListInfoKeys()) <= keys, "no source spatial/composition metadata override")
        for prop in prim.properties:
            require(path in disabled and prop.name == "physics:collisionEnabled" and isinstance(prop, Sdf.AttributeSpec)
                    and prop.default is False and not session.ListTimeSamplesForPath(prop.path), "source properties only declared collision disable")
        for child in prim.nameChildren:
            inspect(child)
    for prim in session.rootPrims:
        inspect(prim)
    stage = Usd.Stage.Open(root, session)
    floor = stage.GetPrimAtPath(ground["path"])
    equal(np.asarray(UsdGeom.Mesh(floor).GetPointsAttr().Get()), np.asarray(ground["points"], dtype=np.float32), "actual session finite-floor points match recipe")
    equal(np.asarray(UsdGeom.Mesh(floor).GetFaceVertexIndicesAttr().Get()), ground["face_indices"], "actual session finite-floor triangle index recipe")
    require(all(stage.GetPrimAtPath(p).GetAttribute("physics:collisionEnabled").Get() is False for p in disabled), "all authored flat floors disabled in actual session")
    require(hashlib.sha256(source.ExportToString().encode()).hexdigest() == ram, "original source RAM unchanged by CPU parse/composition")
    return {"source_layer_files": len(layers), "original_source_RAM_sha256": ram,
            "source_start_end_TCPS": [original.GetStartTimeCode(), original.GetEndTimeCode(), tcps],
            "source_FPS": original.GetFramesPerSecond(), "source_disabled_floor_count": len(disabled),
            "owned_runtime_root_query_sha256": data["exports"]["runtime_root.usda"]["sha256"],
            "actual_stage_cache_id": attachment["expected_stage_cache_id"]}


def audit_scene(directory, frozen_files):
    data = json.loads((directory / "result.json").read_text())
    check_file(data["preflight"])
    p = json.loads(Path(data["preflight"]["path"]).read_text())
    require(data["status"] == "scene_resource_gate_passed" and data["native_started"] is True and data["fallback_used"] is False,
            "actual native execution and resource gate status")
    require(data["script"]["sha256"] == PRODUCER == p["script"]["sha256"] == frozen_files["scripts/probe_paper_scene_maps.py"]["sha256"], "exact e47 producer/frozen174 identity")
    source = audit_source(data, p)
    r = arrays(data["native_omap"])
    world = arrays(data["coordinate_validation"]["canonical_world_positions_artifact"])
    ray = arrays(data["floor_support"]["artifact"])
    band = arrays(data["full_height_band_queries"]["artifact"])
    dims, lo, hi, buf = r["dimensions"], r["min_bound"], r["max_bound"], r["buffer"]
    require(dims.dtype == np.int64 and dims.shape == (3,) and buf.shape == (int(dims[0]*dims[1]),)
            and np.isin(buf, [4, 5, 6]).all() and np.allclose(hi[:2]-lo[:2], dims[:2]*.1, rtol=0, atol=1e-5), "native dimensions/labels/world extent")
    occ, api = indices(r["occupied_positions"], lo, dims), indices(r["free_positions"], lo, dims)
    equal(np.sort(occ), np.flatnonzero(buf == 4), "every native4 has exact API occupied center")
    require(np.isin(buf[api], [5, 6]).all(), "API free list never overlaps native4")
    missing = np.setdiff1d(np.flatnonzero(buf == 5), api)
    rows, cols = np.divmod(missing, dims[0])
    generated = np.column_stack([lo[0]+(dims[0]-cols-.5)*.1, lo[1]+(rows+.5)*.1, np.full(len(missing), (lo[2]+hi[2])/2)])
    api5rows = np.flatnonzero(buf[api] == 5)
    canonical = np.concatenate([r["free_positions"][api5rows], generated])
    count = len(canonical)
    canonical_indices = indices(canonical, lo, dims)
    equal(np.sort(canonical_indices), np.flatnonzero(buf == 5), "canonical full raw5 only, no unknown upgrade")
    for key, expected in [("canonical_free_world_positions", canonical), ("canonical_free_API_row", np.r_[api5rows, np.full(len(missing), -1)]),
            ("missing_known5_buffer_indices", missing), ("missing_known5_world_positions", generated), ("API_free_buffer_indices", api),
            ("canonical_known5_native_buffer_indices", canonical_indices), ("API_free_buffer_labels", buf[api]),
            ("API_free_unknown_world_positions", r["free_positions"][buf[api] == 6])]:
        equal(world[key], expected, "canonical artifact recompute: " + key)
    unknown_missing = np.setdiff1d(np.flatnonzero(buf == 6), api)
    equal(world["unreported_unknown_buffer_indices"], unknown_missing, "missing API unknown indices preserved")
    equal(world["unreported_unknown_buffer_labels"], buf[unknown_missing], "missing API unknown labels remain6")
    equal(world["canonical_free_position_source"], np.r_[np.full(len(api5rows), "native_API_row"), np.full(len(missing), "official_buffer_center")], "each native/API or engineering center source explicit")
    require(data["known5_API_complete"] is (len(missing) == 0)
            and data["coordinate_validation"]["missing_known5_API_count"] == len(missing), "Office missing API rows explicitly disclosed")
    floor_paths = data["floor_collider_paths"]
    owner = data["floor_ownership"]
    static = set(owner["known_static_nonfloor_collision_paths"])
    certificates = {v["path"]: v for v in p["flat_ground_collider_inventory"]}
    inventory = owner["actual_enabled_flat_ground_inventory"]
    require(len(floor_paths) == 1 and owner["actual_enabled_flat_ground_paths"] == floor_paths
            and {v["path"] for v in inventory if v["near_ground_flat_surface"]} == set(floor_paths), "actual inventory exactly one selected flat floor")
    require(all(v == certificates[v["path"]] for v in inventory if v["path"] in static), "nonfloor geometry certificates source-bound exact")
    for a in [ray, band]:
        equal(a["native_free_index"], np.arange(count), "every raw5 query index/order")
        equal(a["free_world_positions"], canonical, "every raw5 exact query world coordinate")
    rr, paths = csr(ray, "all_hit_", count)
    near = (ray["all_hit_normals"][:, 2] >= .9) & (np.abs(ray["all_hit_positions"][:, 2]-data["selected_floor_height_m"]) <= .08)
    isfloor = np.isin(paths, floor_paths)
    isstatic = np.isin(paths, list(static))
    require((isfloor | isstatic).all() and np.isfinite(ray["all_hit_positions"]).all()
            and np.isfinite(ray["all_hit_distances"]).all() and np.all(ray["all_hit_distances"] >= 0)
            and np.allclose(np.linalg.norm(ray["all_hit_normals"], axis=-1), 1, atol=1e-3, rtol=0), "actual floor CSR finite exact paths/normals")
    equal(np.diff(ray["all_hit_offsets"]), ray["all_hit_count"], "all saved ray counts agree with CSR")
    selected = isfloor & near
    equal(np.bincount(rr[selected], minlength=count), np.ones(count, dtype=np.int64), "all raw5 exactly one native selected-floor support")
    equal(ray["selected_floor_support_hit_count"], np.ones(count, dtype=np.int64), "stored support counts agree with CSR")
    equal(ray["hit_position"], ray["all_hit_positions"][selected].astype(ray["hit_position"].dtype), "selected floor geometry unchanged")
    equal(ray["hit_normal"], ray["all_hit_normals"][selected].astype(ray["hit_normal"].dtype), "selected floor normals match actual CSR")
    equal(ray["collision"], paths[selected], "selected floor exact path matches actual CSR")
    low_obstacle = np.zeros(count, dtype=bool)
    low_obstacle[rr[near & ~isfloor & isstatic]] = True
    equal(ray["known_static_nonfloor_near_ground_obstacle_hit"], low_obstacle, "near-ground certified obstacle mask from raw CSR")
    expected_class = np.where(isfloor, np.where(near, "selected_floor_support", "selected_floor_other_surface"),
                              np.where(near & isstatic, "known_static_nonfloor_near_ground_obstacle", "known_nonfloor_surface"))
    equal(ray["all_hit_classification"], expected_class, "floor CSR classes recomputed")
    equal(ray["supported"], np.ones(count, dtype=bool), "all raw5 native floor support, no synthetic fill")
    require(not ray["foreign_hit"].any() and not ray["undeclared_near_floor_hit"].any(), "no undeclared/foreign floor hits")
    br, bpaths = csr(band, "hit_", count)
    bf, bs = np.isin(bpaths, floor_paths), np.isin(bpaths, list(static))
    require((bf | bs).all(), "fullheight exact collider inventory")
    equal(band["hit_classification"], np.where(bf, "selected_floor_excluded", "authored_obstacle"), "fullheight classes recomputed")
    obstacle = np.zeros(count, dtype=bool)
    obstacle[br[bs]] = True
    equal(band["authored_obstacle_hit"], obstacle, "fullheight obstacle mask from CSR")
    equal(band["full_height_clear"], ~obstacle, "fullheight clear mask from CSR")
    clock = data["full_height_band_queries"]
    check_file(clock["clock_artifact"])
    initial = data["actual_query_clock_initial"]
    check_file(initial["artifact"])
    require(clock["clock_before"] == clock["clock_after"] == {"submitted_physics_step": 2, "timeline_s": 0.}
            and initial["observation"]["submitted_physics_step"] == 2 and initial["observation"]["timeline_s"] == 0
            and data["physics"]["timeline_after_queries_s"] == 0 and data["physics"]["counter_before_snapshot"] == 0,
            "actual querycounter 2/0timeline unchanged after all queries")
    require(json.loads(Path(clock["clock_artifact"]["path"]).read_text()) == {k:v for k,v in clock.items() if k not in {"artifact", "clock_artifact", "queried_cells"}}, "fullheight clock artifact raw complete identity")
    equal(json.loads(Path(initial["artifact"]["path"]).read_text()), initial["observation"], "raw clock JSON identity")
    native = np.choose((buf.reshape(int(dims[1]), int(dims[0]))[::-1, ::-1]-4).astype(int), [0, 254, 205]).astype(np.uint8)
    grid = np.rint((canonical[:, :2]-lo[:2])/.1-.5).astype(np.int64)
    pix = np.column_stack([dims[1]-1-grid[:, 1], grid[:, 0]])
    refined = native.copy()
    refined[pix[obstacle | low_obstacle, 0], pix[obstacle | low_obstacle, 1]] = 0
    bounds, edims = np.array(p["projection_metadata"]["bounds"]), np.array(p["projection_metadata"]["dims"])
    pad = np.rint((bounds[0]-lo[:2])/.1).astype(np.int64)
    require(np.isin(pad, [0, 1]).all() and np.allclose(lo[:2]+pad*.1, bounds[0], rtol=0, atol=1e-5)
            and np.allclose(hi[:2], bounds[1], rtol=0, atol=1e-5) and np.array_equal(dims[:2]-pad, edims), "native padding crop preserves exact declared world intersection")
    crop = refined[:int(dims[1]-pad[1]), int(pad[0]):].copy()
    projection_record = p["input_projection"]["occupancy.png"]
    check_file(projection_record)
    projection = np.asarray(Image.open(projection_record["path"]))
    crop[(projection == 205) & (crop != 0)] = 205
    for name, expected in [("native_reconstructed.png", native), ("height_band_refined.png", refined), ("occupancy.png", crop)]:
        check_file(data["exports"][name])
        equal(np.asarray(Image.open(data["exports"][name]["path"])), expected, "all exported pixels raw-derived: " + name)
    origin = data["origin_verification"]
    origin_selected = [hit for hit in origin["floor_ray_all_hits"] if hit["collision"] in floor_paths
                       and hit["normal"][2] >= .9 and abs(hit["position"][2]-data["selected_floor_height_m"]) <= .08]
    require(len(origin_selected) == 1 and origin["floor_ownership"]["selected_floor_support_hit_count"] == 1
            and origin["floor_ownership"]["foreign_ray_collision_paths"] == [], "origin has exactly one actual selected floor")
    require(data["full_height_band_queries"]["queried_cells"] == data["floor_support"]["tested"] == count
            and data["height_band_refinement"]["height_blocked_known5_cells"] == int(obstacle.sum())
            and data["height_band_refinement"]["known_static_nonfloor_near_ground_blocked_known5_cells"] == int(low_obstacle.sum()), "fullcell coverage and derived block counts exact")
    require(data["navigation_ready"] is False and data["human_navigation_validated"] is False
            and data["height_band_refinement"]["pure_omap2D_full_height_claim"] is False, "engineering scene-resource only, no navigation/author effect promotion")
    return {"scene": data["scene"], "result": record(directory / "result.json"), "source": source,
            "native_raw_label_counts_4_5_6": [int(np.sum(buf == v)) for v in [4, 5, 6]],
            "full_floor_and_height_cells": count, "floor_CSR_hits": len(paths), "band_CSR_hits": len(bpaths),
            "missing_known5_API_cells": len(missing), "API_unknown_rows_preserved": int(np.sum(buf[api] == 6)),
            "near_ground_nonfloor_blocked_cells": int(low_obstacle.sum()), "fullheight_blocked_cells": int(obstacle.sum()),
            "final_export_value_counts": {str(v): int(np.sum(crop == v)) for v in [0, 254, 205]},
            "native_padding_cells_xy": pad.tolist(), "actual_querycounter": 2, "timeline_s": 0,
            "all_raw_files_untouched": True, "resource_artifacts_verified": True, "new_navigation_quality_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only environment")
    require(not args.output.exists(), "new report only")
    manifest = json.loads(args.source_manifest.read_text())
    files = {v["path"]: v for v in manifest["files"]}
    require(len(files) == manifest["source_count"] == 174, "exact 174 source manifest")
    archive = manifest["archive"]
    require(record(archive["path"])["sha256"] == archive["sha256"], "frozen174 archive SHA")
    with tarfile.open(archive["path"], "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(files) and len({m.name for m in members}) == len(members), "unique exact174 archive members")
        for member in members:
            require(member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts, "bounded regular relative archive member")
            payload = tar.extractfile(member).read()
            declared = files[member.name]
            require(len(payload) == declared["bytes"] and hashlib.sha256(payload).hexdigest() == declared["sha256"], "frozen archive member SHA: " + member.name)
    current = {name: record(ROOT/name) for name in files}
    current_matches = {name: current[name]["bytes"] == row["bytes"] and current[name]["sha256"] == row["sha256"] for name,row in files.items()}
    scenes = [audit_scene(args.run / name, files) for name in ["Warehouse", "Office", "Hospital", "Full_Warehouse"]]
    require(all(record(ROOT/name) == observed for name,observed in current.items()), "current source observation unchanged during independent CPU read")
    report = {"schema": "native_v9_four_scene_independent_raw_artifact_review.v1", "status": "independent_artifact_checks_passed",
              "CPU_only": True, "SDK_or_GPU_started": False, "production_and_artifacts_modified": False,
              "provider_admission_or_config_generation_repeated": False, "source_manifest": record(args.source_manifest),
              "source_archive": record(archive["path"]), "checks_passed": len(CHECKS), "checks": CHECKS, "scenes": scenes,
              "current_source_observation_not_import_ledger": {"all_174_match": all(current_matches.values()), "mismatched_paths": [name for name,value in current_matches.items() if not value], "records": current},
              "full_native_known5_cells": sum(v["full_floor_and_height_cells"] for v in scenes),
              "limits": ["Actual submitted API clocks and sensor artifacts are verified; standalone CPU cannot observe live Kit again.",
                         "Full-height one-cell static maps with generated reference floor are an engineering resource gate, not full-body navigation or author eight-scene validation."]}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"], "full_native_known5_cells": report["full_native_known5_cells"]}))


if __name__ == "__main__":
    main()
