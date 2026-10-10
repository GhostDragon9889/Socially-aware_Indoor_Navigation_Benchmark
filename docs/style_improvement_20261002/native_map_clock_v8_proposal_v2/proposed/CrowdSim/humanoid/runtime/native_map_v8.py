"""Versioned v8 static-session clock admission, retaining v7 geometry contracts.

Only the exact v8 producer SHA admits this branch. Source ranges remain recorded;
session0..max(end,TCPS) and actual query clocks are checked independently.
No SDK/tensor import happens at module import.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .native_map_v7 import LAYOUT, _archive, _csr, _floor_evidence, _indices, _require

PROBE_SHA256 = "20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a"


def is_v8(data):
    return data.get("script", {}).get("sha256") == PROBE_SHA256


def _session_range_evidence(provider, data):
    from .dynbench_scene import _check_record
    try:
        original, observed = data["source_time_range"], data["runtime_time_range_override"]
        start, end, tcps = (original[key] for key in ("start_time_code", "end_time_code", "time_codes_per_second"))
        if (not is_v8(data) or any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end, tcps)) or tcps <= 0
                or original["USD_time_range_modified"] is not False
                or observed["schema"] != "dynbench.native_frozen_timeline_session.v8"
                or observed["original_source_composed_range"] != {"start_time_code": start, "end_time_code": end, "time_codes_per_second": tcps}):
            raise ValueError("Original source range or exact v8 session contract disagrees")
        runtime_end = max(float(end), float(tcps))
        expected = {"requested_runtime_start_time_code": 0., "requested_runtime_end_time_code": runtime_end,
                    "requested_runtime_start_s": 0., "requested_runtime_end_s": runtime_end / tcps,
                    "composed_stage_start_time_code": 0., "composed_stage_end_time_code": runtime_end,
                    "composed_stage_start_at_query_time_code": 0., "composed_stage_end_at_query_time_code": runtime_end,
                    "composed_stage_time_codes_per_second_at_query": tcps,
                    "observed_timeline_start_at_query_s": 0.}
        if any(type(observed[key]) not in (int, float) or not math.isfinite(observed[key]) or observed[key] != value for key, value in expected.items()):
            raise ValueError("Requested/composed/actual-query session range is not the exact zero-start formula")
        actual_end = observed["observed_timeline_end_at_query_s"]
        if (type(actual_end) not in (int, float) or not math.isfinite(actual_end)
                or not math.isclose(actual_end, runtime_end / tcps, rel_tol=0., abs_tol=1e-12)):
            raise ValueError("Actual query timeline end disagrees with the composed session end")
        if (any(observed[key] is not True for key in ("source_root_in_memory_unchanged", "runtime_session_time_range_modified", "session_edit_target_at_query", "source_root_in_memory_unchanged_at_query"))
                or any(observed[key] is not False for key in ("source_layers_saved", "time_codes_per_second_modified", "actual_query_clock_pass_claim"))
                or type(observed["session_layer_identifier"]) is not str or not observed["session_layer_identifier"].startswith("anon:")
                or observed["source_root_layer_identifier"] != provider._data["source"]["path"]):
            raise ValueError("Session ownership, original root identity or unchanged-source declarations failed")
        ram = observed["source_root_in_memory_sha256"]
        if type(ram) is not str or not re.fullmatch("[0-9a-f]{64}", ram) or observed["source_root_in_memory_sha256_at_query"] != ram:
            raise ValueError("Original source root RAM changed between setup and actual queries")
        session_record = data["exports"]["runtime_session.usda"]
        _check_record(session_record)
        text = Path(session_record["path"]).read_text()
        if not text.startswith("#usda 1.0\n(") or "\n)\n" not in text:
            raise ValueError("Actual session export lacks global time metadata")
        header = text.partition("\n)\n")[0]
        for key, expected_value in (("startTimeCode", 0.), ("endTimeCode", runtime_end)):
            values = re.findall(r"(?m)^\s*" + key + r"\s*=\s*([-+0-9.eE]+)\s*$", header)
            if len(values) != 1 or float(values[0]) != expected_value:
                raise ValueError("Actual saved session time metadata contradicts " + key)
        rates = re.findall(r"(?m)^\s*timeCodesPerSecond\s*=\s*([-+0-9.eE]+)\s*$", header)
        if rates and (len(rates) != 1 or float(rates[0]) != tcps):
            raise ValueError("Saved session changed original timeCodesPerSecond")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Native v8 original/session range evidence is incomplete") from error


def verify_v8(provider, data, *, admitted=False):
    """Recompute the complete v7 raw/CSR/export contract; return grid metadata."""
    import numpy as np
    from PIL import Image
    from .dynbench_scene import _check_record, HEIGHT_BAND

    _session_range_evidence(provider, data)
    try:
        _require(is_v8(data) and data["native_direct_cell_validation"] is True, "requires exact frozen probe identity and direct every-cell validation")
        _require(data["status"] == "scene_resource_gate_passed" and data["scene_resource_gate_passed"] is True
                 and data["native_started"] is True and data["native_probe_completed"] is True
                 and data["native_mapping_verified"] is True and data["fallback_used"] is False
                 and data["original_source_layers_unchanged"] is True and data["projection_unknown_cells_preserved"] is True
                 and data["foreign_native_hit_paths"] == [] and data["sampled_free_cells_with_overlap"] == 0
                 and data["scene"] == provider.scene_name and data["source"] == provider._data["source"]
                 and data["preflight"] == provider._preflight_record and data["script"] == provider._data["script"]
                 and data["floor_collider_paths"] == list(provider.floor_paths) and data["selected_floor_height_m"] == provider.ground_height_m
                 and data["dynamic_activity"]["static_snapshot_gate"] is True
                 and data["physics"]["scene_query_support"] is True and data["physics"]["no_query_time_advance"] is True,
                 "requires completed matching native resource admission, never failed/projection fallback")
        if admitted:
            provider.require_navigation_map()
            provider.verify_sources()
            _check_record(provider._native["record"])
            _require(json.loads(Path(provider._native["record"]["path"]).read_text()) == data, "admitted result differs from bound bytes")
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
        clock = json.loads(Path(band["clock_artifact"]["path"]).read_text())
        before = band["clock_before"]
        _require(before == band["clock_after"] and band["query_time_unchanged"] is True and band["foreign_hit_paths"] == []
                 and clock == {key: value for key, value in band.items() if key not in {"artifact", "clock_artifact", "queried_cells"}}
                 and type(before["submitted_physics_step"]) is int and before["submitted_physics_step"] >= 0
                 and type(before["timeline_s"]) in (float, int) and before["timeline_s"] == 0., "queries require actual unchanged nonnegative counter and zero frozen timeline")
        initial = data["actual_query_clock_initial"]
        _check_record(initial["artifact"])
        observed = initial["observation"]
        _require(json.loads(Path(initial["artifact"]["path"]).read_text()) == observed
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
        raise ValueError("Native v8 complete source/canonical/crop/CSR evidence is missing") from error
    return {"native_evidence": deepcopy(provider._native["record"]) if admitted else None, "canonical_known5_cells": count,
            "height_blocked_known5_cells": int(obstacle.sum()), "known_static_nonfloor_near_ground_blocked_known5_cells": int(low_obstacle.sum()),
            "source_bound_artifacts_checked": True, "fixed_native_layout": LAYOUT, "query_clock": deepcopy(before),
            "full_height_band_above_floor_m": list(HEIGHT_BAND), "new_physical_execution_verified": False,
            "admission_protocol": "v8_exact_probe_SHA", "probe_sha256": PROBE_SHA256,
            "known5_API_complete": not len(missing), "missing_known5_API_count": len(missing),
            "known5_position_source": source, "derived_export_grid": deepcopy(crop),
            "runtime_time_range_override": deepcopy(data["runtime_time_range_override"])}
