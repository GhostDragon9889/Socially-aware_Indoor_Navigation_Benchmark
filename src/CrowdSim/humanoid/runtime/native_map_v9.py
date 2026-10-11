"""Versioned v9 static-session clock admission, retaining v7 geometry contracts.

Only the exact v9 producer SHA admits this branch. Source ranges remain recorded;
session0..max(end,TCPS) and actual query clocks are checked independently.
No SDK/tensor import happens at module import.
"""
from __future__ import annotations

from .dynbench_scene import same_file_record

from copy import deepcopy
import json
from social_navigation.resource_bindings import read_json as read_resource_json
import hashlib
import math
from pathlib import Path
import re

from .native_map_v7 import LAYOUT, _archive, _csr, _floor_evidence, _indices, _require

PROBE_SHA256 = "e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82"


def is_v9(data):
    return data.get("script", {}).get("sha256") == PROBE_SHA256



def _owned_layer_content(provider, data, original_ram):
    """Keep standalone USD parsing out of CPU-only metric processes."""
    import os
    import subprocess
    import sys
    if "pxr" in sys.modules:
        return _parse_owned_layer_content(provider, data, original_ram)
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.dumps({"provider": provider._data, "data": data, "original_ram": original_ram},
                         allow_nan=False, sort_keys=True)
    completed = subprocess.run(
        [sys.executable, "-m", "CrowdSim.humanoid.runtime.native_map_v9"],
        input=payload, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    if completed.returncode != 0:
        raise ValueError("Standalone USD content verification failed: " + completed.stderr[-4000:])
    expected = {"verified": True, "source_sha256": digest,
                "input_sha256": hashlib.sha256(payload.encode()).hexdigest()}
    try:
        actual = json.loads(completed.stdout)
    except (TypeError, ValueError) as error:
        raise ValueError("Standalone USD verifier returned invalid evidence") from error
    if actual != expected or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("Standalone USD verifier source or input identity changed")


def _parse_owned_layer_content(provider, data, original_ram):
    """Parse saved USD using standalone OpenUSD; never open an App/context."""
    from pxr import Sdf, Usd, Tf
    source = Sdf.Layer.FindOrOpen(provider._data["source"]["path"])
    if source is None or hashlib.sha256(source.ExportToString().encode()).hexdigest() != original_ram:
        raise ValueError("Original source RAM artifact differs from source bytes")
    root_record = data["exports"]["runtime_root.usda"]
    def fresh_export(record):
        # FindOrOpen(path) can return a previously cached layer after its file
        # changed. Bind the current bytes and parse them in a fresh own layer.
        payload = Path(record["path"]).read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Owned layer export bytes changed during verification")
        layer = Sdf.Layer.CreateAnonymous("native_v9_verified_export.usda")
        try:
            if not layer.ImportFromString(payload.decode()):
                raise ValueError("Owned layer export cannot be parsed")
        except (UnicodeError, Tf.ErrorException) as error:
            raise ValueError("Owned layer export cannot be parsed") from error
        return layer
    root = fresh_export(root_record)
    session = fresh_export(data["exports"]["runtime_session.usda"])
    if root is None or session is None or root.rootPrims:
        raise ValueError("Owned root may contain only metadata and original sublayer")
    if root_record["sha256"] != data["runtime_time_range_override"]["source_root_in_memory_sha256_at_query"]:
        raise ValueError("Owned root export differs from actually observed query RAM")
    preview = Usd.Stage.Open(source)
    ground = provider._data["reference_ground"]
    disabled = set(ground["disabled_flat_ground_collision_paths"])
    deinstanced = set(ground.get("deinstanced_floor_roots_in_session", []))
    generated = (ground["path"], "/__PaperMapProbePhysicsScene")
    def inspect(prim):
        path = str(prim.path)
        if any(path == root_path or path.startswith(root_path + "/") for root_path in generated):
            return
        original = preview.GetPrimAtPath(path)
        if not original or prim.specifier != Sdf.SpecifierOver or prim.typeName:
            raise ValueError("Session adds an undeclared source/spatial prim: " + path)
        allowed = {"specifier"}
        if path in disabled:
            allowed.add("apiSchemas")
            if prim.HasInfo("apiSchemas"):
                api = prim.GetInfo("apiSchemas")
                if list(api.prependedItems) != ["PhysicsCollisionAPI"] or api.explicitItems or api.appendedItems or api.deletedItems:
                    raise ValueError("Source session API override differs from collision-disable recipe")
        if path in deinstanced:
            allowed.add("instanceable")
            if prim.HasInfo("instanceable") and prim.GetInfo("instanceable") is not False:
                raise ValueError("Source session deinstance override must be false")
        if original.GetTypeName() == "PhysicsScene":
            allowed.add("active")
            if prim.HasInfo("active") and prim.GetInfo("active") is not False:
                raise ValueError("Authored source physics must be inactive")
        if set(prim.ListInfoKeys()) - allowed:
            raise ValueError("Session changes source metadata/spatial composition: " + path)
        for prop in prim.properties:
            if path not in disabled or prop.name != "physics:collisionEnabled" or not isinstance(prop, Sdf.AttributeSpec) or prop.default is not False or session.ListTimeSamplesForPath(prop.path):
                raise ValueError("Session changes source geometry or properties: " + str(prop.path))
        for child in prim.nameChildren:
            inspect(child)
    for prim in session.rootPrims:
        inspect(prim)


def _session_range_evidence(provider, data):
    """Validate owned runtime root and unchanged original file-backed source."""
    from .dynbench_scene import _check_record
    try:
        original, observed = data["source_time_range"], data["runtime_time_range_override"]
        composition, attached = data["runtime_stage_composition"], data["runtime_stage_attachment"]
        start, end, tcps = (original[key] for key in ("start_time_code", "end_time_code", "time_codes_per_second"))
        if (not is_v9(data) or any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end, tcps)) or tcps <= 0
                or original["USD_time_range_modified"] is not False
                or observed["schema"] != "dynbench.native_frozen_timeline_session.v9"
                or composition["schema"] != "dynbench.native_anonymous_runtime_root.v9"
                or composition["original_source_time_range"] != original
                or observed["original_source_composed_range"] != {"start_time_code": start, "end_time_code": end, "time_codes_per_second": tcps}):
            raise ValueError("Original source or exact v9 owned-stage contract disagrees")
        runtime_end = max(float(end),float(tcps))
        expected = {"requested_runtime_start_time_code":0., "requested_runtime_end_time_code":runtime_end,
            "requested_runtime_start_s":0., "requested_runtime_end_s":runtime_end/tcps,
            "composed_stage_start_time_code":0., "composed_stage_end_time_code":runtime_end,
            "composed_stage_start_at_query_time_code":0., "composed_stage_end_at_query_time_code":runtime_end,
            "composed_stage_time_codes_per_second_at_query":tcps, "observed_timeline_start_at_query_s":0.}
        if any(type(observed[key]) not in (float,int) or not math.isfinite(observed[key]) or observed[key] != value for key,value in expected.items()):
            raise ValueError("Actual query/composed session range disagrees with explicit formula")
        actual_end=observed["observed_timeline_end_at_query_s"]
        if type(actual_end) not in (float,int) or not math.isfinite(actual_end) or not math.isclose(actual_end,runtime_end/tcps,rel_tol=0.,abs_tol=1e-12):
            raise ValueError("Actual query timeline end disagrees")
        source = provider._data["source"]["path"]
        runtime_id=composition["owned_runtime_root_identifier"]
        session_id=composition["runtime_session_identifier"]
        if (type(runtime_id) is not str or not runtime_id.startswith("anon:") or type(session_id) is not str or not session_id.startswith("anon:")
                or composition["absolute_original_source_sublayers"] != [source]
                or observed["owned_runtime_root_sublayers_at_query"] != [source]
                or composition["original_source_layer_identifier"] != source or observed["original_source_layer_identifier"] != source
                or observed["owned_runtime_root_identifier"] != runtime_id or observed["source_root_layer_identifier"] != runtime_id
                or observed["session_layer_identifier"] != session_id):
            raise ValueError("Owned anonymous root/session or absolute original source identity disagrees")
        for record, keys in ((composition,("owned_runtime_root_anonymous","original_source_root_in_memory_unchanged_after_composition","original_source_root_in_memory_unchanged_at_query")),
                (observed,("owned_runtime_root_anonymous","owned_runtime_root_anonymous_at_query","original_source_root_in_memory_unchanged_after_setup","original_source_root_in_memory_unchanged_at_query","session_edit_target_at_query","runtime_session_time_range_modified"))):
            if any(record[key] is not True for key in keys):
                raise ValueError("Original RAM or runtime session ownership failed")
        if (any(composition[key] is not False for key in ("original_source_layers_saved","extra_App_or_Context_created","source_prim_paths_relocated"))
                or any(observed[key] is not False for key in ("source_layers_saved","time_codes_per_second_modified","actual_query_clock_pass_claim"))):
            raise ValueError("Source saved/relocated or extra context/invented clock declarations failed")
        original_ram=composition["original_source_root_in_memory_sha256"]
        if type(original_ram) is not str or not re.fullmatch("[0-9a-f]{64}",original_ram) or any(value != original_ram for value in (
                composition["original_source_root_in_memory_sha256_at_query"],observed["original_source_root_in_memory_sha256"],observed["original_source_root_in_memory_sha256_at_query"])):
            raise ValueError("Original source RAM changed between composition/setup/query")
        for key in ("source_root_in_memory_sha256","source_root_in_memory_sha256_at_query"):
            if type(observed[key]) is not str or not re.fullmatch("[0-9a-f]{64}",observed[key]):
                raise ValueError("Owned root RAM observations must be actual SHA256 values")
        if (attached["schema"] != "dynbench.native_runtime_stage_attachment.v9"
                or attached["active_context_root_identifier"] != runtime_id or attached["active_context_session_identifier"] != session_id
                or type(attached["expected_stage_cache_id"]) is not int or attached["expected_stage_cache_id"] < 0
                or attached["observed_context_stage_cache_id"] != attached["expected_stage_cache_id"]
                or type(attached["submitted_step_counter_before"]) is not int or attached["submitted_step_counter_before"] < 0
                or attached["submitted_step_counter_after"] != attached["submitted_step_counter_before"]
                or type(attached["app_updates_waiting_for_async_attachment"]) is not int or attached["app_updates_waiting_for_async_attachment"] < 0
                or attached["observed_context_owns_exact_runtime_stage"] is not True
                or attached["explicit_physics_steps_issued_by_wait"] != 0 or attached["native_engine_time_observed"] is not None
                or attached["extra_App_or_Context_created"] is not False or attached["manual_context_or_PhysX_attach_called"] is not False):
            raise ValueError("Actual context/cache stage ownership or attachment wait counter failed")
        metadata=composition["source_stage_metadata"]
        if metadata["up_axis"] != "Z" or metadata["meters_per_unit"] != 1. or type(metadata["frames_per_second"]) not in (float,int) or not math.isfinite(metadata["frames_per_second"]) or metadata["frames_per_second"] <= 0:
            raise ValueError("Original source units/FPS missing or incompatible")
        for filename, values in (("runtime_session.usda",{"startTimeCode":0.,"endTimeCode":runtime_end}),
                ("runtime_root.usda",{"startTimeCode":0.,"endTimeCode":runtime_end,"timeCodesPerSecond":tcps,"framesPerSecond":metadata["frames_per_second"],"metersPerUnit":1.})):
            artifact=data["exports"][filename]
            _check_record(artifact)
            text=Path(artifact["path"]).read_text()
            if not text.startswith("#usda 1.0\n(") or "\n)\n" not in text:
                raise ValueError("Owned runtime export lacks global metadata")
            header=text.partition("\n)\n")[0]
            for key,value in values.items():
                matches=re.findall(r"(?m)^\s*"+key+r"\s*=\s*([-+0-9.eE]+)\s*$",header)
                if len(matches)!=1 or float(matches[0])!=value:
                    raise ValueError("Owned runtime export contradicts "+key)
            if filename=="runtime_root.usda":
                block=re.search(r"(?s)\bsubLayers\s*=\s*\[(.*?)\]",header)
                if block is None or re.findall(r"@([^@]+)@",block.group(1)) != [source] or not re.search(r'(?m)^\s*upAxis\s*=\s*"Z"\s*$',header):
                    raise ValueError("Owned root absolute source composition or upAxis disagrees")
                default=metadata["default_prim_path"]
                names=re.findall(r'(?m)^\s*defaultPrim\s*=\s*"([^"]+)"\s*$',header)
                if (default is None and names) or (default is not None and names != [str(default).removeprefix("/")]):
                    raise ValueError("Owned root defaultPrim differs from original source")
        _owned_layer_content(provider, data, original_ram)
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError("Native v9 complete original/owned stage evidence is missing") from error


def verify_v9(provider, data, *, admitted=False):
    """Recompute the complete v7 raw/CSR/export contract; return grid metadata."""
    import numpy as np
    from PIL import Image
    from .dynbench_scene import _check_record, HEIGHT_BAND

    _session_range_evidence(provider, data)
    try:
        _require(is_v9(data) and data["native_direct_cell_validation"] is True, "requires exact frozen probe identity and direct every-cell validation")
        _require(data["status"] == "scene_resource_gate_passed" and data["scene_resource_gate_passed"] is True
                 and data["native_started"] is True and data["native_probe_completed"] is True
                 and data["native_mapping_verified"] is True and data["fallback_used"] is False
                 and data["original_source_layers_unchanged"] is True and data["projection_unknown_cells_preserved"] is True
                 and data["foreign_native_hit_paths"] == [] and data["sampled_free_cells_with_overlap"] == 0
                 and data["scene"] == provider.scene_name and data["source"] == provider._data["source"]
                 and same_file_record(data["preflight"], provider._preflight_record) and data["script"] == provider._data["script"]
                 and data["floor_collider_paths"] == list(provider.floor_paths) and data["selected_floor_height_m"] == provider.ground_height_m
                 and data["dynamic_activity"]["static_snapshot_gate"] is True
                 and data["physics"]["scene_query_support"] is True and data["physics"]["no_query_time_advance"] is True,
                 "requires completed matching native resource admission, never failed/projection fallback")
        if admitted:
            provider.require_navigation_map()
            provider.verify_sources()
            _check_record(provider._native["record"])
            _require(read_resource_json(provider._native["record"]["path"]) == data, "admitted result differs from bound bytes")
        raw = _archive(data["native_omap"])
        dims, low, high, buffer = raw["dimensions"], raw["min_bound"], raw["max_bound"], raw["buffer"]
        _require(dims.shape == (3,) and dims.dtype.kind in "iu" and np.all(dims[:2] > 0) and dims[2] in (0, 1)
                 and low.shape == high.shape == (3,) and np.isfinite(low).all() and np.isfinite(high).all()
                 and buffer.shape == (int(dims[0] * dims[1]),) and np.isin(buffer, [4, 5, 6]).all()
                 and np.allclose(high[:2] - low[:2], dims[:2] * .1, atol=1e-5, rtol=0)
                 and data["actual_dimensions"] == dims.tolist()
                 and np.array_equal(np.asarray(data["actual_bounds_world_m"]), np.stack((low, high))), "native dimensions/bounds/labels disagree")
        coords = data["coordinate_validation"]
        source = data["known5_position_source"]
        _require(source in ("api_verified", "official_buffer_centers") and coords["known5_position_source"] == source
                 and coords["fixed_native_layout"] == LAYOUT and coords["layout_fitted_dynamically"] is False
                 and coords["unknown_promoted_to_free"] is False and coords["positions_roundtrip_verified"] is True,
                 "requires explicit official position source and fixed coordinates")
        occupied = _indices(raw["occupied_positions"], low, dims)
        api_indices = _indices(raw["free_positions"], low, dims)
        _require(np.array_equal(np.sort(occupied), np.flatnonzero(buffer == 4))
                 and not np.intersect1d(occupied, api_indices).size and np.isin(buffer[api_indices], [5, 6]).all(), "API positions contradict raw4/5/6 labels")
        api_rows = np.flatnonzero(buffer[api_indices] == 5)
        missing = np.setdiff1d(np.flatnonzero(buffer == 5), api_indices)
        _require(not len(missing) or source == "official_buffer_centers", "missing API raw5 requires explicit official buffer centers")
        row, column = np.divmod(missing, dims[0])
        derived = np.column_stack((low[0] + (dims[0] - column - .5) * .1, low[1] + (row + .5) * .1,
                                   np.full(len(missing), (low[2] + high[2]) / 2)))
        canonical = np.concatenate((raw["free_positions"][api_rows], derived)) if len(missing) else raw["free_positions"][api_rows]
        canonical_indices = _indices(canonical, low, dims)
        count = len(canonical)
        _require(count > 0 and np.array_equal(np.sort(canonical_indices), np.flatnonzero(buffer == 5)), "canonical rows must exactly cover raw5, never unknown6")
        world = _archive(coords["canonical_world_positions_artifact"])
        expected_rows = np.r_[api_rows, np.full(len(missing), -1, dtype=np.int64)]
        unknown_missing = np.setdiff1d(np.flatnonzero(buffer == 6), api_indices)
        for key, expected in (("occupied_world_positions", raw["occupied_positions"]), ("canonical_free_world_positions", canonical),
                ("canonical_free_API_row", expected_rows), ("canonical_free_position_source", np.where(expected_rows >= 0, "native_API_row", "official_buffer_center")),
                ("canonical_known5_native_buffer_indices", canonical_indices), ("missing_known5_buffer_indices", missing),
                ("missing_known5_world_positions", derived), ("API_free_buffer_indices", api_indices),
                ("API_free_buffer_labels", buffer[api_indices]), ("API_free_unknown_world_positions", raw["free_positions"][buffer[api_indices] == 6]),
                ("unreported_unknown_buffer_indices", unknown_missing), ("unreported_unknown_buffer_labels", buffer[unknown_missing])):
            _require(np.array_equal(world[key], expected), "canonical source field contradicts raw API/buffer: " + key)
        for key in ("canonical_free_API_row", "canonical_known5_native_buffer_indices", "missing_known5_buffer_indices",
                    "API_free_buffer_indices", "unreported_unknown_buffer_indices"):
            _require(world[key].dtype == np.int64, "canonical index fields require exact int64: " + key)
        _require(world["canonical_free_API_row"].dtype.kind == "i"
                 and data["known5_API_complete"] is (len(missing) == 0) and coords["known5_API_complete"] is (len(missing) == 0)
                 and coords["missing_known5_API_count"] == len(missing)
                 and coords["engineering_buffer_center_reconstruction"] is bool(len(missing))
                 and coords["native_direct_cell_validation_required"] is bool(len(missing)), "missing-API provenance must remain explicit and honest")
        # Raw/full-height images remain native-sized. Export alone is cropped
        # by a fixed, zero-or-one lower-cell offset in real world coordinates.
        bounds = np.asarray(provider.bounds_xy)
        export_dims = np.asarray(provider._data["projection_metadata"]["dims"])
        padding = np.rint((bounds[0] - low[:2]) / .1).astype(np.int64)
        _require(np.isin(padding, [0, 1]).all() and np.allclose(low[:2] + padding * .1, bounds[0], atol=1e-5, rtol=0)
                 and np.allclose(high[:2], bounds[1], atol=1e-5, rtol=0) and np.array_equal(dims[:2] - padding, export_dims)
                 and np.allclose(bounds[1] - bounds[0], export_dims * .1, atol=1e-5, rtol=0), "only verified zero/one lower padding may be cropped")
        crop = data["derived_export_grid"]
        _require(crop == {"schema": "dynbench.native_projection_world_crop.v1", "lower_padding_cells_xy": padding.tolist(),
            "native_window_rows": [0, int(dims[1] - padding[1])], "native_window_columns": [int(padding[0]), int(dims[0])],
            "export_bounds_xy_m": bounds.tolist(), "export_dimensions_xy": export_dims.tolist(),
            "engineering_native_padding_crop": bool(padding.any()), "world_grid_alignment_tolerance_m": 1e-5,
            "author_projection_files_modified": False, "native_raw_grid_preserved": True}, "crop metadata contradicts fixed source-world window")
        rays = _archive(data["floor_support"]["artifact"])
        band = data["full_height_band_queries"]
        hits = _archive(band["artifact"])
        for values in (rays, hits):
            _require(values["native_free_index"].dtype.kind in "iu" and np.array_equal(values["native_free_index"], np.arange(count))
                     and np.array_equal(values["free_world_positions"], canonical), "every-cell queries must match canonical source order")
        support, refinement = data["floor_support"], data["height_band_refinement"]
        _require(rays["supported"].dtype == np.bool_ and rays["supported"].shape == (count,) and rays["supported"].all()
                 and support["full_grid"] is True and support["native_free_cells"] == support["tested"] == support["supported"] == count
                 and support["unsupported"] == 0 and band["queried_cells"] == count
                 and refinement["known5_source"] == refinement["known5_tested"] == count
                 and all(refinement[key] is True for key in ("all_native_known5_cells_tested", "native_unknown_never_promoted", "engineering_refinement"))
                 and refinement["pure_omap2D_full_height_claim"] is False
                 and refinement["untested_known5_exported_unknown"] == refinement["unsupported_tested_cells"] == refinement["unclassified_cells"] == 0
                 and refinement["cell_xy_size_m"] == .1 and refinement["band_above_selected_floor_m"] == list(HEIGHT_BAND)
                 and np.allclose(data["requested_height_band_world_m"], np.asarray(HEIGHT_BAND) + provider.ground_height_m, atol=1e-8, rtol=0),
                 "requires full native5 floor and height coverage")
        low_obstacle = _floor_evidence(provider, data, rays, count)
        rows, paths = _csr(hits, "hit_", count)
        floor, static = np.isin(paths, provider.floor_paths), np.isin(paths, provider.static_paths)
        _require((floor | static).all() and np.array_equal(hits["hit_classification"], np.where(floor, "selected_floor_excluded", "authored_obstacle")),
                 "height overlap contains foreign/misclassified exact paths")
        obstacle = np.zeros(count, bool)
        obstacle[rows[static]] = True
        for key, expected in (("authored_obstacle_hit", obstacle), ("full_height_clear", ~obstacle), ("all_hit_paths_classified", np.ones(count, bool))):
            _require(hits[key].dtype == np.bool_ and np.array_equal(hits[key], expected), "height masks contradict CSR: " + key)
        _require(hits["selected_floor_hit_count"].dtype.kind in "iu" and np.array_equal(hits["selected_floor_hit_count"], np.bincount(rows[floor], minlength=count))
                 and refinement["height_blocked_known5_cells"] == int(obstacle.sum())
                 and refinement["height_clear_known5_cells"] == int((~obstacle).sum())
                 and refinement["known_static_nonfloor_near_ground_blocked_known5_cells"] == int(low_obstacle.sum()), "height/low-obstacle counts disagree")
        _check_record(band["clock_artifact"])
        clock = read_resource_json(band["clock_artifact"]["path"])
        before = band["clock_before"]
        _require(before == band["clock_after"] and band["query_time_unchanged"] is True and band["foreign_hit_paths"] == []
                 and clock == {key: value for key, value in band.items() if key not in {"artifact", "clock_artifact", "queried_cells"}}
                 and type(before["submitted_physics_step"]) is int and before["submitted_physics_step"] >= 0
                 and type(before["timeline_s"]) in (float, int) and before["timeline_s"] == 0., "queries require actual unchanged nonnegative counter and zero frozen timeline")
        initial = data["actual_query_clock_initial"]
        _check_record(initial["artifact"])
        observed = initial["observation"]
        _require(read_resource_json(initial["artifact"]["path"]) == observed
                 and type(observed["submitted_physics_step"]) is int
                 and type(observed["timeline_s"]) in (float, int)
                 and observed["submitted_physics_step"] == before["submitted_physics_step"] == data["physics"]["counter_at_query"]
                 and observed["timeline_s"] == observed["requested_frozen_timeline_s"] == data["physics"]["timeline_at_query_s"] == 0.
                 and observed["counter_python_type"] == "int", "initial query clock artifact must match actual query clock")
        image = np.where(buffer.reshape(int(dims[1]), int(dims[0]))[::-1, ::-1] == 4, 0,
                         np.where(buffer.reshape(int(dims[1]), int(dims[0]))[::-1, ::-1] == 5, 254, 205)).astype(np.uint8)
        cells = np.rint((canonical[:, :2] - low[:2]) / .1 - .5).astype(np.int64)
        pixels = np.column_stack((dims[1] - 1 - cells[:, 1], cells[:, 0]))
        refined = image.copy()
        blocked = obstacle | low_obstacle
        refined[pixels[blocked, 0], pixels[blocked, 1]] = 0
        for name, expected in (("native_reconstructed.png", image), ("height_band_refined.png", refined)):
            _check_record(data["exports"][name])
            _require(np.array_equal(np.asarray(Image.open(data["exports"][name]["path"])), expected), "full native image contradicts raw/CSR: " + name)
        exported = refined[:int(dims[1] - padding[1]), int(padding[0]):]
        projection = np.asarray(Image.open(provider._data["input_projection"]["occupancy.png"]["path"]))
        _require(exported.shape == projection.shape == tuple(export_dims[::-1]), "export/projection dimensions disagree")
        exported = exported.copy()
        exported[(projection == 205) & (exported != 0)] = 205
        _check_record(data["exports"]["occupancy.png"])
        _require(np.array_equal(np.asarray(Image.open(data["exports"]["occupancy.png"]["path"])), exported)
                 and (not admitted or np.array_equal(provider._map, exported)), "export must retain unknown and block height/low obstacles exactly")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Native v9 complete source/canonical/crop/CSR evidence is missing") from error
    return {"native_evidence": deepcopy(provider._native["record"]) if admitted else None, "canonical_known5_cells": count,
            "height_blocked_known5_cells": int(obstacle.sum()), "known_static_nonfloor_near_ground_blocked_known5_cells": int(low_obstacle.sum()),
            "source_bound_artifacts_checked": True, "fixed_native_layout": LAYOUT, "query_clock": deepcopy(before),
            "full_height_band_above_floor_m": list(HEIGHT_BAND), "new_physical_execution_verified": False,
            "admission_protocol": "v9_exact_probe_SHA", "probe_sha256": PROBE_SHA256,
            "known5_API_complete": not len(missing), "missing_known5_API_count": len(missing),
            "known5_position_source": source, "derived_export_grid": deepcopy(crop),
            "runtime_time_range_override": deepcopy(data["runtime_time_range_override"]),
            "runtime_stage_composition": deepcopy(data["runtime_stage_composition"]),
            "runtime_stage_attachment": deepcopy(data["runtime_stage_attachment"])}


if __name__ == "__main__":
    import sys
    from types import SimpleNamespace
    payload = sys.stdin.read()
    request = json.loads(payload)
    _parse_owned_layer_content(SimpleNamespace(_data=request["provider"]),
                               request["data"], request["original_ram"])
    if any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "warp"} for name in sys.modules):
        raise RuntimeError("Standalone USD verifier imported a simulator or tensor runtime")
    print(json.dumps({"verified": True,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "input_sha256": hashlib.sha256(payload.encode()).hexdigest()}))
