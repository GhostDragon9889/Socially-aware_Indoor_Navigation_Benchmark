"""Explicit v7 native-map admission; no SDK or tensor import at module load.

The exact frozen probe SHA selects this protocol. It retains the raw native
grid, directly tests every raw5 cell, crops only verified lower padding, and
blocks certified low static obstacles. Synthetic fixtures are not native proof.
"""
from __future__ import annotations

from copy import deepcopy
import json
from social_navigation.resource_bindings import read_json as read_resource_json, resolve_path
import math
from pathlib import Path

PROBE_SHA256 = "a538bf92ffb64b536a7005329f38c080bd1b8818114df58d9880d5506f0665d5"
LAYOUT = "row-major x-fastest; row0=minY, column0=maxX"
NONFLOOR_REASONS = ("Floor mesh is not thin and near selected floor height",
                    "Plane is not horizontal at the selected ground height")


def is_v7(data):
    return data.get("script", {}).get("sha256") == PROBE_SHA256


def _require(condition, message):
    if not bool(condition):
        raise ValueError("Native v7 " + message)


def _archive(record):
    import numpy as np
    from .dynbench_scene import _check_record
    _check_record(record)
    with np.load(record["path"], allow_pickle=False) as archive:
        _require(all(archive[key].dtype.kind != "O" for key in archive.files), "forbids object buffers")
        return {key: archive[key].copy() for key in archive.files}


def _indices(points, low, dims):
    import numpy as np
    _require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), "cell centers require finite[N,3]")
    grid = np.rint((points[:, :2] - low[:2]) / .1 - .5).astype(np.int64)
    _require(not np.any(grid < 0) and not np.any(grid >= dims[:2])
             and np.allclose(points[:, :2], low[:2] + (grid + .5) * .1, atol=2e-4, rtol=0)
             and len(np.unique(grid, axis=0)) == len(grid), "cell centers contradict fixed native layout")
    return grid[:, 1] * dims[0] + dims[0] - 1 - grid[:, 0]


def _csr(arrays, prefix, count):
    import numpy as np
    offsets, paths = arrays[prefix + "offsets"], arrays[prefix + "collision_paths"]
    _require(offsets.shape == (count + 1,) and offsets.dtype.kind == "i" and offsets[0] == 0
             and np.all(np.diff(offsets) >= 0) and np.all(offsets >= 0) and np.all(offsets <= len(paths))
             and offsets[-1] == len(paths) and paths.ndim == 1 and paths.dtype.kind in "US"
             and arrays[prefix + "rigid_body_paths"].shape == paths.shape
             and arrays[prefix + "classification"].shape == paths.shape,
             "all-hit CSR paths/offsets/classes are inconsistent")
    return np.repeat(np.arange(count), np.diff(offsets)), paths


def _floor_evidence(provider, data, rays, count):
    import numpy as np
    owner = data["floor_ownership"]
    floor_paths, static_paths = provider.floor_paths, provider.static_paths
    ground = provider.ground_height_m
    inventory = owner["actual_enabled_flat_ground_inventory"]
    _require(owner["schema"] == "dynbench.native_floor_owner.v1"
             and owner["declared_floor_paths"] == list(floor_paths)
             and owner["actual_enabled_flat_ground_paths"] == list(floor_paths)
             and all(owner[key] is True for key in ("all_support_hit_paths_declared", "single_declared_floor_owner_verified",
                 "floor_ownership_verified", "full_grid_floor_rays", "all_ray_hits_saved",
                 "single_selected_floor_support_verified", "near_ground_nonfloor_hit_cells_exported_occupied"))
             and owner["undeclared_near_floor_collision_paths"] == owner["foreign_ray_collision_paths"] == []
             and owner["undeclared_near_floor_hit_rows"] == owner["foreign_hit_rows"] == 0
             and owner["height_tolerance_m"] == .08 and owner["normal_z_min"] == .9
             and owner["artifact"] == data["floor_support"]["artifact"]
             and len({row["path"] for row in inventory}) == len(inventory)
             and {row["path"] for row in inventory} == set((*floor_paths, *static_paths))
             and all(row["enabled"] is True and row["static"] is True for row in inventory)
             and {row["path"] for row in inventory if row["near_ground_flat_surface"]} == set(floor_paths),
             "requires exactly one declared floor and complete all-hit ownership")
    source_rows = {row["path"]: row for row in provider._data["flat_ground_collider_inventory"]}
    certified = set()
    for row in inventory:
        if (row.get("enabled") is True and row.get("static") is True
                and row.get("near_ground_flat_surface") is False and row.get("type_name") in ("Mesh", "Plane")
                and row.get("reason") in NONFLOOR_REASONS and row["path"] in static_paths):
            _require(source_rows.get(row["path"]) == row, "static nonfloor certificate differs from source-bound geometry inventory")
            certified.add(row["path"])
    _require(owner["known_static_nonfloor_collision_paths"] == sorted(certified), "static nonfloor registry is not exact certified source paths")
    rows, paths = _csr(rays, "all_hit_", count)
    offsets = rays["all_hit_offsets"]
    _require(np.array_equal(np.diff(offsets), rays["all_hit_count"]), "floor hit counts disagree with CSR")
    positions, normals, distances = rays["all_hit_positions"], rays["all_hit_normals"], rays["all_hit_distances"]
    _require(positions.shape == normals.shape == (len(paths), 3) and distances.shape == paths.shape
             and np.isfinite(positions).all() and np.isfinite(normals).all() and np.isfinite(distances).all()
             and np.all(distances >= 0) and np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=1e-3, rtol=0),
             "floor all-hit geometry is invalid")
    floor = np.isin(paths, floor_paths)
    near = (normals[:, 2] >= .9) & (np.abs(positions[:, 2] - ground) <= .08)
    known_low = near & ~floor & np.isin(paths, sorted(certified))
    _require(np.isin(paths, (*floor_paths, *static_paths)).all() and not np.any(near & ~floor & ~known_low),
             "rejects foreign or uncertified near-floor support")
    classes = np.where(floor, np.where(near, "selected_floor_support", "selected_floor_other_surface"),
                       np.where(known_low, "known_static_nonfloor_near_ground_obstacle", "known_nonfloor_surface"))
    _require(np.array_equal(rays["all_hit_classification"], classes), "floor classes contradict exact paths/geometry")
    selected_counts = np.bincount(rows[floor & near], minlength=count)
    _require(rays["selected_floor_support_hit_count"].dtype.kind in "iu"
             and np.array_equal(rays["selected_floor_support_hit_count"], selected_counts)
             and np.all(selected_counts == 1), "every raw5 cell must have exactly one selected-floor support hit")
    selected_hits = floor & near
    _require(np.allclose(positions[selected_hits, :2], rays["free_world_positions"][:, :2], atol=2e-4, rtol=0)
             and np.array_equal(rays["hit_position"], positions[selected_hits].astype(rays["hit_position"].dtype))
             and np.array_equal(rays["hit_normal"], normals[selected_hits].astype(rays["hit_normal"].dtype))
             and np.array_equal(rays["collision"], paths[selected_hits]),
             "selected support fields must match the actual per-cell all-hit geometry")
    low = np.zeros(count, bool)
    low[rows[known_low]] = True
    _require(rays["known_static_nonfloor_near_ground_obstacle_hit"].dtype == np.bool_
             and np.array_equal(rays["known_static_nonfloor_near_ground_obstacle_hit"], low)
             and owner["known_static_nonfloor_near_ground_hit_rows"] == int(low.sum())
             and owner["known_static_nonfloor_near_ground_paths"] == sorted(set(paths[known_low])),
             "low-obstacle masks/counts contradict all-hit CSR")
    for key in ("undeclared_near_floor_hit", "foreign_hit"):
        _require(rays[key].shape == (count,) and rays[key].dtype == np.bool_ and not rays[key].any(), "floor row masks are invalid/failed")
    origin = data["origin_verification"]
    diagnostic = origin["floor_ownership"]
    origin_classes, origin_support = [], 0
    for hit in origin["floor_ray_all_hits"]:
        position, normal = np.asarray(hit["position"]), np.asarray(hit["normal"])
        distance = hit["distance"]
        _require(position.shape == normal.shape == (3,) and np.isfinite(position).all() and np.isfinite(normal).all()
                 and np.isclose(np.linalg.norm(normal), 1., atol=1e-3, rtol=0) and math.isfinite(distance) and distance >= 0,
                 "origin all-hit geometry is invalid")
        path = hit["collision"]
        near_origin = normal[2] >= .9 and abs(position[2] - ground) <= .08
        _require(path in (*floor_paths, *static_paths) and not (near_origin and path not in floor_paths),
                 "origin must be clear of foreign/low obstacles")
        category = "selected_floor_support" if path in floor_paths and near_origin else "selected_floor_other_surface" if path in floor_paths else "known_nonfloor_surface"
        origin_classes.append(category)
        origin_support += category == "selected_floor_support"
    _require(origin_support == 1 and diagnostic["selected_floor_support_hit_count"] == 1
             and diagnostic["single_selected_floor_support_verified"] is True
             and diagnostic["classification"] == origin_classes and diagnostic["all_support_hit_paths_declared"] is True
             and diagnostic["undeclared_near_floor_collision_paths"] == diagnostic["foreign_ray_collision_paths"] == []
             and diagnostic["known_static_nonfloor_near_ground_paths"] == [], "origin ownership is not exactly one clear selected floor")
    return low


def verify_v7(provider, data, *, admitted=False):
    """Recompute the complete v7 raw/CSR/export contract; return grid metadata."""
    import numpy as np
    from PIL import Image
    from .dynbench_scene import _check_record, HEIGHT_BAND

    try:
        _require(is_v7(data) and data["native_direct_cell_validation"] is True, "requires exact frozen probe identity and direct every-cell validation")
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
        raise ValueError("Native v7 complete source/canonical/crop/CSR evidence is missing") from error
    return {"native_evidence": deepcopy(provider._native["record"]) if admitted else None, "canonical_known5_cells": count,
            "height_blocked_known5_cells": int(obstacle.sum()), "known_static_nonfloor_near_ground_blocked_known5_cells": int(low_obstacle.sum()),
            "source_bound_artifacts_checked": True, "fixed_native_layout": LAYOUT, "query_clock": deepcopy(before),
            "full_height_band_above_floor_m": list(HEIGHT_BAND), "new_physical_execution_verified": False,
            "admission_protocol": "v7_exact_probe_SHA", "probe_sha256": PROBE_SHA256,
            "known5_API_complete": not len(missing), "missing_known5_API_count": len(missing),
            "known5_position_source": source, "derived_export_grid": deepcopy(crop)}
