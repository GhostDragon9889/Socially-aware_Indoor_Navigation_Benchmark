#!/usr/bin/env python3
"""Inspect local DynBench scenes, then optionally run actual PhysX/omap queries.

Default is SDK-free preflight. --execute-physx starts one Kit App for all selected
scenes. Original source layers are never saved. Optional finite reference ground
and full-height refinement are explicit engineering choices. No projection
fallback or human-navigation quality claim is made. Limited queries cannot
admit a full scene-resource map.
Standalone USD preflight runs in child processes to keep its pxr provider out of
Kit's process. All source layers are hashed before and after actual execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")
VALUES = {"occupied": 4, "free": 5, "unknown": 6}
RESOLUTION = .1
BAND = (.15, 1.8)
DT = 1 / 120


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def finite_array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def validate_projection_metadata(scene, data):
    """Admit explicit units, floor and the existing projection's transform."""
    if scene not in SCENES or data.get("scene") != scene:
        raise ValueError("Unknown or mismatched scene")
    if data.get("composition_complete") is not True or data.get("composition_errors"):
        raise ValueError("Projection has incomplete composition")
    stage = data["stage"]
    if stage.get("stage_up_axis") != "Z" or stage.get("stage_meters_per_unit") != 1.0:
        raise ValueError("Only original Z-up, metersPerUnit=1 scenes are admitted")
    if data["metadata"]["resolution"] != RESOLUTION:
        raise ValueError("This probe requires the declared .1m projection")
    if list(data["pedestrian_height_band_above_floor_m"]) != list(BAND):
        raise ValueError("Height band must be .15..1.8m above selected floor")
    bounds = finite_array(data["bounds_xy_m"], (2, 2), "bounds")
    dims_raw = data["image_dimensions_xy"]
    if len(dims_raw) != 2 or any(type(v) is not int or v <= 0 for v in dims_raw):
        raise ValueError("Projection dimensions must be positive integers")
    dims = np.asarray(dims_raw)
    if not np.allclose(bounds[1] - bounds[0], dims * RESOLUTION, atol=1e-8, rtol=0):
        raise ValueError("Projection dimensions/bounds disagree")
    ground = data["floor"]["height_m"]
    if type(ground) not in (int, float) or not math.isfinite(ground):
        raise ValueError("Explicit finite floor height required")
    floors = data["floor"]["selected_floor_prims"]
    if not floors or len(set(floors)) != len(floors) or any(not p.startswith("/") for p in floors):
        raise ValueError("Exact authored floor paths required")
    origin_data = data.get("free_origin_candidate")
    if not isinstance(origin_data, dict):
        raise ValueError("No existing free origin candidate; do not invent one")
    origin = finite_array(origin_data["omap_origin_world_map_frame_m"], (3,), "origin")
    low = finite_array(origin_data["omap_relative_min_bound_map_m"], (3,), "relative min")
    high = finite_array(origin_data["omap_relative_max_bound_map_m"], (3,), "relative max")
    if np.any(low >= high) or not np.allclose((origin + low)[:2], bounds[0], atol=1e-8):
        raise ValueError("Origin/relative min/bounds disagree")
    if not np.allclose((origin + high)[:2], bounds[1], atol=1e-8):
        raise ValueError("Origin/relative max/bounds disagree")
    if not np.allclose([origin[2] + low[2], origin[2] + high[2]], np.asarray(BAND) + ground, atol=1e-8):
        raise ValueError("Origin/relative bounds do not express declared height band")
    if np.any(origin[:2] <= bounds[0]) or np.any(origin[:2] >= bounds[1]):
        raise ValueError("Origin candidate outside map")
    return {"bounds": bounds, "dims": dims, "ground": float(ground), "origin": origin,
            "relative_min": low, "relative_max": high, "floor_prims": floors}


def positions_to_pixels(points, minimum, maximum, dims_xy, resolution=RESOLUTION):
    """Native returned world cell centers -> ROS row0 north; strict round trip."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Native positions must be [N,3]")
    minimum = finite_array(minimum, (3,), "native minimum")
    maximum = finite_array(maximum, (3,), "native maximum")
    dims = np.asarray(dims_xy)
    if dims.shape != (2,) or dims.dtype.kind not in "iu" or np.any(dims <= 0):
        raise ValueError("Native XY dimensions must be positive integers")
    if not np.isfinite(points).all() or resolution <= 0 or not math.isfinite(resolution):
        raise ValueError("Nonfinite positions or invalid resolution")
    tolerance = 2e-4  # native Float3 bounds/positions; at most .2mm, not a whole cell
    if not np.allclose(maximum[:2] - minimum[:2], dims * resolution, atol=tolerance, rtol=0):
        raise ValueError("Native bounds do not match dimensions times resolution")
    fractional = (points[:, :2] - minimum[:2]) / resolution - .5
    indices = np.rint(fractional).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= dims):
        raise ValueError("Returned cell center outside native bounds")
    reconstructed = minimum[:2] + (indices + .5) * resolution
    if not np.allclose(reconstructed, points[:, :2], atol=tolerance, rtol=0):
        raise ValueError("Native positions are not centers on the declared grid")
    pixels = np.column_stack((dims[1] - 1 - indices[:, 1], indices[:, 0]))
    if len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Duplicate native XY cell centers")
    return pixels


def native_position_buffer_indices(points, minimum, maximum, dimensions):
    """Fixed installed compute_coordinates layout: row0 south, column0 east."""
    dimensions = np.asarray(dimensions)
    pixels = positions_to_pixels(points, minimum, maximum, dimensions[:2])
    width, height = map(int, dimensions[:2])
    return (height - 1 - pixels[:, 0]) * width + (width - 1 - pixels[:, 1])


def reconstruct_native_map(buffer, occupied, free, minimum, maximum, dimensions):
    """Validate the fixed official native layout against every world position.

    get_free_positions can include label6 unknown cells. Only raw label5,
    cross-checked against its world position, is exported as known free.
    """
    dimensions = np.asarray(dimensions)
    if dimensions.shape != (3,) or dimensions.dtype.kind not in "iu" or dimensions[2] not in (0, 1):
        raise ValueError("Expected 2D native Int3 dimensions")
    w, h = map(int, dimensions[:2])
    if w <= 0 or h <= 0:
        raise ValueError("Empty native map")
    raw = np.asarray(buffer, dtype=np.float64)
    if raw.shape != (w * h,) or not np.isfinite(raw).all() or not np.isin(raw, list(VALUES.values())).all():
        raise ValueError("Native buffer size or labels invalid")
    occupied = np.asarray(occupied, dtype=np.float64).reshape(-1, 3)
    free = np.asarray(free, dtype=np.float64).reshape(-1, 3)
    oi = native_position_buffer_indices(occupied, minimum, maximum, dimensions)
    fi = native_position_buffer_indices(free, minimum, maximum, dimensions)
    if not np.array_equal(np.sort(oi), np.flatnonzero(raw == VALUES["occupied"])):
        raise ValueError("Official native layout: occupied world positions must exactly equal buffer4 cells")
    if np.intersect1d(oi, fi).size or np.any(raw[fi] == VALUES["occupied"]):
        raise ValueError("Official native layout: free-list positions intersect occupied buffer4 cells")
    if not np.isin(np.flatnonzero(raw == VALUES["free"]), fi).all():
        raise ValueError("Official native layout: every known buffer5 cell needs a free-list world position")
    missing = np.setdiff1d(np.arange(w*h), np.r_[oi, fi])
    if np.any(raw[missing] != VALUES["unknown"]):
        raise ValueError("Official native layout: unreported world positions must be buffer6 unknown cells")
    linear = np.arange(w*h)
    native_row, native_col = np.divmod(linear, w)
    ros_row, ros_col = h - 1 - native_row, w - 1 - native_col
    image = np.empty((h, w), dtype=np.uint8)
    labels = np.where(raw == VALUES["occupied"], 0, np.where(raw == VALUES["free"], 254, 205)).astype(np.uint8)
    image[ros_row, ros_col] = labels
    if np.count_nonzero(image == 205) != np.count_nonzero(raw == VALUES["unknown"]):
        raise ValueError("Unknown count cannot be reconciled")
    known = raw[fi] == VALUES["free"]
    return image, {"raw_value_counts": {k: int(np.count_nonzero(raw == v)) for k, v in VALUES.items()},
        "positions_roundtrip_verified": True, "all_occupied_world_labels_equal4": True,
        "all_buffer5_have_free_list_world_position": True, "free_list_buffer4_count": 0,
        "API_free_list_count": len(free), "canonical_known_free_count": int(known.sum()),
        "API_free_list_unknown_count": int((~known).sum()), "unreported_unknown_count": len(missing),
        "unreported_unknown_native_rows": np.unique(missing//w).tolist(), "duplicate_xy_positions": 0,
        "unknown_promoted_to_free": False, "unverified_buffer_spatial_layout_assumed": False,
        "fixed_native_layout": "row-major x-fastest; row0=minY, column0=maxX",
        "layout_source": "installed isaacsim.asset.gen.omap/utils/utils.py compute_coordinates; MapGenerator.h getBuffer",
        "layout_fitted_dynamically": False, "row0": "max world Y", "column0": "min world X"}


def supported_floor_hit(hits, floor_paths, ground, height_tolerance=.08):
    """Only exact selected-floor collider hits with an upward, near-level normal."""
    valid = []
    for hit in hits:
        position = finite_array(hit["position"], (3,), "ray hit position")
        normal = finite_array(hit["normal"], (3,), "ray hit normal")
        distance = hit["distance"]
        if not math.isfinite(distance) or distance < 0 or not np.isclose(np.linalg.norm(normal), 1., atol=1e-3):
            raise ValueError("Invalid native hit distance")
        if hit["collision"] in floor_paths and normal[2] >= .9 and abs(position[2] - ground) <= height_tolerance:
            valid.append(hit)
    return min(valid, key=lambda hit: hit["distance"]) if valid else None


def classify_floor_ray_hits(hits, floor_paths, enabled_paths, ground, height_tolerance=.08):
    """Retain every hit; an undeclared level support invalidates floor ownership."""
    supported_floor_hit(hits, floor_paths, ground, height_tolerance)
    categories, undeclared, foreign = [], set(), set()
    for hit in hits:
        path = hit["collision"]
        near = hit["normal"][2] >= .9 and abs(hit["position"][2] - ground) <= height_tolerance
        if path not in enabled_paths:
            category = "foreign_collision_path"
            foreign.add(path)
        elif path in floor_paths:
            category = "selected_floor_support" if near else "selected_floor_other_surface"
        elif near:
            category = "undeclared_near_floor_support"
            undeclared.add(path)
        else:
            category = "known_nonfloor_surface"
        categories.append(category)
    return {"classification": categories, "undeclared_near_floor_collision_paths": sorted(undeclared),
            "foreign_ray_collision_paths": sorted(foreign), "all_support_hit_paths_declared": not undeclared and not foreign}


def query_floor_support(query, positions, selected, ground, floor_paths, enabled_paths, *, scene=""):
    """Publish complete all-hit CSR rows, rather than only the nearest owner hit."""
    selected = np.asarray(selected)
    supported = np.zeros(len(selected), dtype=bool)
    hit_position = np.full((len(selected), 3), np.nan, dtype=np.float32)
    hit_normal = np.full((len(selected), 3), np.nan, dtype=np.float32)
    collision, rows, categories, offsets = [], [], [], [0]
    undeclared, foreign = set(), set()
    undeclared_mask, foreign_mask = np.zeros(len(selected), dtype=bool), np.zeros(len(selected), dtype=bool)
    for n, index in enumerate(selected):
        hits = rays(query, positions[index, :2], ground)
        classification = classify_floor_ray_hits(hits, floor_paths, enabled_paths, ground)
        undeclared.update(classification["undeclared_near_floor_collision_paths"])
        foreign.update(classification["foreign_ray_collision_paths"])
        undeclared_mask[n] = bool(classification["undeclared_near_floor_collision_paths"])
        foreign_mask[n] = bool(classification["foreign_ray_collision_paths"])
        categories.extend(classification["classification"])
        rows.extend(hits)
        offsets.append(len(rows))
        hit = supported_floor_hit(hits, floor_paths, ground)
        collision.append("" if hit is None else hit["collision"])
        if hit is not None:
            supported[n] = True
            hit_position[n], hit_normal[n] = hit["position"], hit["normal"]
        if n and n % 20000 == 0:
            print(f"{scene}: floor rays {n}/{len(selected)}", flush=True)
    arrays = {"native_free_index": selected, "free_world_positions": positions[selected], "supported": supported,
        "hit_position": hit_position, "hit_normal": hit_normal, "collision": np.asarray(collision),
        "all_hit_count": np.diff(offsets), "all_hit_offsets": np.asarray(offsets, dtype=np.int64),
        "all_hit_collision_paths": np.asarray([r["collision"] for r in rows], dtype=str),
        "all_hit_rigid_body_paths": np.asarray([r["rigid_body"] for r in rows], dtype=str),
        "all_hit_positions": np.asarray([r["position"] for r in rows], dtype=np.float64).reshape(-1, 3),
        "all_hit_normals": np.asarray([r["normal"] for r in rows], dtype=np.float64).reshape(-1, 3),
        "all_hit_distances": np.asarray([r["distance"] for r in rows], dtype=np.float64),
        "all_hit_classification": np.asarray(categories, dtype=str), "undeclared_near_floor_hit": undeclared_mask,
        "foreign_hit": foreign_mask}
    return {"arrays": arrays, "undeclared_near_floor_collision_paths": sorted(undeclared),
            "foreign_ray_collision_paths": sorted(foreign), "all_support_hit_paths_declared": not undeclared and not foreign}


def fixed_probe_cells(shape, count=256, seed=7):
    total = math.prod(shape)
    ids = np.random.default_rng(seed).choice(total, min(count, total), replace=False)
    return np.column_stack(np.unravel_index(ids, shape))


def floor_probe_indices(free_count, limit):
    if type(limit) is not int or limit < 0 or free_count <= 0:
        raise ValueError("Floor probe limit must be a nonnegative integer; native free cells required")
    return np.arange(free_count) if limit == 0 else np.unique(np.linspace(0, free_count - 1, min(limit, free_count), dtype=int))


def navigation_admission(*, origin_clear, origin_supported, mapping_valid, full_floor_probe,
                         support_mask, static_unchanged, sampled_free_overlap_count, source_unchanged,
                         full_height_probe, height_clear_mask, all_band_hit_paths_classified, floor_ownership_verified):
    support_mask = np.asarray(support_mask)
    height_clear_mask = np.asarray(height_clear_mask)
    if support_mask.ndim != 1 or support_mask.dtype != np.bool_:
        raise ValueError("Floor support mask must be one-dimensional bool")
    if height_clear_mask.shape != support_mask.shape or height_clear_mask.dtype != np.bool_:
        raise ValueError("Height clear mask must be aligned one-dimensional bool")
    return bool(origin_clear and origin_supported and mapping_valid and full_floor_probe and full_height_probe
                and len(support_mask) and support_mask.all() and (support_mask & height_clear_mask).any()
                and all_band_hit_paths_classified and floor_ownership_verified and static_unchanged
                and sampled_free_overlap_count == 0 and source_unchanged)


def height_band_box(xy, ground, resolution=RESOLUTION, band=BAND):
    """A cell volume, explicitly .1m XY and .15..1.8m above declared ground."""
    xy = finite_array(xy, (2,), "cell world XY")
    band = finite_array(band, (2,), "height band")
    if (isinstance(ground, (bool, np.bool_)) or not math.isfinite(ground)
            or isinstance(resolution, (bool, np.bool_)) or not math.isfinite(resolution)
            or resolution != RESOLUTION or not np.array_equal(band, BAND)):
        raise ValueError("Full-height query requires declared .1m cell and .15..1.8m band, finite ground")
    center = (float(xy[0]), float(xy[1]), float(ground + band.mean()))
    half_extent = (resolution/2, resolution/2, float((band[1]-band[0])/2))
    return center, half_extent


def observed_query_clock(counter_getter, time_getter):
    counter, timestamp = counter_getter(), time_getter()
    if (type(counter) is not int or counter < 0 or isinstance(timestamp, (bool, np.bool_))
            or not math.isfinite(timestamp) or timestamp < 0):
        raise ValueError("Query clock needs actual nonnegative submitted counter/finite timeline")
    return {"submitted_physics_step": counter, "timeline_s": float(timestamp)}


def require_query_clock_unchanged(before, after):
    if before != after:
        raise RuntimeError("Physics/timeline advanced during read-only full-height-band queries")


def query_full_height_band(query, positions, selected, ground, floor_paths, enabled_paths,
                           *, counter_getter, time_getter, scene=""):
    """Record every native overlap hit; only exact authored paths are classified."""
    positions = np.asarray(positions)
    selected = np.asarray(selected)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all()
            or selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
            or np.any(selected >= len(positions)) or np.any(selected < 0) or np.any(np.diff(selected) <= 0)):
        raise ValueError("Full-height query needs finite world points and unique sorted valid indices")
    floor_paths, enabled_paths = set(floor_paths), set(enabled_paths)
    if (not floor_paths or not floor_paths <= enabled_paths
            or not all(isinstance(path, str) and path.startswith("/") for path in enabled_paths)):
        raise ValueError("Band classification requires exact enabled paths and selected floor subset")
    before = observed_query_clock(counter_getter, time_getter)
    offsets, collisions, rigid_bodies, classes = [0], [], [], []
    obstacle = np.zeros(len(selected), dtype=bool)
    classified = np.ones(len(selected), dtype=bool)
    floor_hits = np.zeros(len(selected), dtype=np.int32)
    foreign = set()
    for row, index in enumerate(selected):
        center, extent = height_band_box(positions[index, :2], ground)
        hits = overlaps(query, center, extent)
        for hit in hits:
            collision = hit["collision"]
            if collision in floor_paths:
                category = "selected_floor_excluded"
                floor_hits[row] += 1
            elif collision in enabled_paths:
                category = "authored_obstacle"
                obstacle[row] = True
            else:
                category = "foreign_unclassified"
                classified[row] = False
                foreign.add(collision)
            collisions.append(collision)
            rigid_bodies.append(hit["rigid_body"])
            classes.append(category)
        offsets.append(len(collisions))
        if row and row % 20000 == 0:
            print(f"{scene}: full-height cells {row}/{len(selected)}", flush=True)
    after = observed_query_clock(counter_getter, time_getter)
    return {"arrays": {"native_free_index": selected, "free_world_positions": positions[selected],
        "full_height_clear": ~obstacle & classified, "authored_obstacle_hit": obstacle,
        "all_hit_paths_classified": classified, "selected_floor_hit_count": floor_hits,
        "hit_offsets": np.asarray(offsets, dtype=np.int64), "hit_collision_paths": np.asarray(collisions, dtype=str),
        "hit_rigid_body_paths": np.asarray(rigid_bodies, dtype=str), "hit_classification": np.asarray(classes, dtype=str)},
        "clock_before": before, "clock_after": after, "query_time_unchanged": before == after,
        "foreign_hit_paths": sorted(foreign)}


def refine_known_free_map(native_image, free_positions, selected, support_mask, band_arrays,
                          minimum, maximum, dimensions):
    """Untested/unsupported candidates become unknown; raw6 is never free."""
    native_image = np.asarray(native_image)
    selected = np.asarray(selected)
    support_mask = np.asarray(support_mask)
    if (native_image.shape != tuple(np.asarray(dimensions)[:2][::-1]) or native_image.dtype != np.uint8
            or not np.isin(native_image, (0, 205, 254)).all()
            or selected.ndim != 1 or selected.dtype.kind not in "iu"
            or support_mask.shape != selected.shape or support_mask.dtype != np.bool_):
        raise ValueError("Refinement requires exact native labels and aligned support masks")
    pixels = positions_to_pixels(free_positions, minimum, maximum, np.asarray(dimensions)[:2])
    if np.any(selected < 0) or np.any(selected >= len(pixels)) or len(np.unique(selected)) != len(selected):
        raise ValueError("Refinement selected indices are invalid")
    if not (native_image[pixels[:, 0], pixels[:, 1]] == 254).all():
        raise ValueError("Only canonical native known5 world points can enter refinement")
    for key in ("full_height_clear", "authored_obstacle_hit", "all_hit_paths_classified"):
        value = np.asarray(band_arrays[key])
        if value.shape != selected.shape or value.dtype != np.bool_:
            raise ValueError("Refinement band masks must be aligned genuine booleans")
    if not np.array_equal(band_arrays["native_free_index"], selected):
        raise ValueError("Band/support queries must refer to identical canonical rows")
    clear = band_arrays["full_height_clear"]
    obstacle = band_arrays["authored_obstacle_hit"]
    classified = band_arrays["all_hit_paths_classified"]
    if not np.array_equal(clear, ~obstacle & classified):
        raise ValueError("Clear mask contradicts recorded obstacle/classification masks")
    derived = native_image.copy()
    derived[pixels[:, 0], pixels[:, 1]] = 205
    tested = pixels[selected]
    derived[tested[obstacle & classified, 0], tested[obstacle & classified, 1]] = 0
    valid_free = clear & support_mask
    derived[tested[valid_free, 0], tested[valid_free, 1]] = 254
    if not np.array_equal(derived[native_image == 205], native_image[native_image == 205]):
        raise ValueError("Native raw6 unknown labels changed")
    return derived


def floor_mesh_geometry_certificate(points, face_counts, face_indices, matrix, ground, orientation="rightHanded"):
    """Verify real finite static-floor faces; no AABB substitute for mesh topology."""
    points = np.asarray(points, dtype=np.float64)
    counts, indices = np.asarray(face_counts), np.asarray(face_indices)
    matrix = finite_array(matrix, (4, 4), "floor transform")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Floor requires actual finite mesh points")
    if counts.ndim != 1 or indices.ndim != 1 or counts.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise ValueError("Floor face counts/indices must be integer vectors")
    if not len(counts) or np.any(counts < 3) or counts.sum() != len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("Floor mesh face topology invalid")
    if orientation not in ("rightHanded", "leftHanded") or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("Unsupported floor orientation or degenerate transform")
    # Gf Matrix4 transforms row vectors: translation is in row3.
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix
    if not np.allclose(homogeneous[:, 3], 1., atol=1e-8):
        raise ValueError("Non-affine floor transform")
    world = homogeneous[:, :3]
    low, high = world.min(0), world.max(0)
    if high[2] - low[2] > .25 or abs(high[2] - ground) > .1:
        raise ValueError("Floor mesh is not thin and near selected floor height")
    top_area, top_triangles, cursor = 0., [], 0
    for count in counts:
        face = world[indices[cursor:cursor + count]]
        cursor += count
        for i in range(1, int(count) - 1):
            tri = face[[0, i, i + 1]]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if orientation == "leftHanded":
                normal *= -1
            norm = np.linalg.norm(normal)
            if norm > 1e-12 and normal[2] / norm >= .9 and np.max(np.abs(tri[:, 2] - ground)) <= .1:
                top_area += norm / 2
                top_triangles.append(tri.tolist())
    if not top_triangles or top_area < .01:
        raise ValueError("Floor mesh has no upward near-floor surface")
    geometry_hash = hashlib.sha256()
    for name, array in (("points", points), ("counts", counts.astype(np.int64)),
                        ("indices", indices.astype(np.int64)), ("matrix", matrix)):
        geometry_hash.update(name.encode())
        geometry_hash.update(np.ascontiguousarray(array).tobytes())
    return {"point_count": len(points), "face_count": len(counts), "index_count": len(indices),
            "world_bounds_m": [low.tolist(), high.tolist()], "upward_top_fan_triangles": len(top_triangles), "top_triangles_world_m": top_triangles,
            "upward_top_fan_area_m2": float(top_area), "geometry_sha256": geometry_hash.hexdigest(),
            "orientation": orientation, "area_is_geometric_diagnostic_not_native_support": True}


def floor_plane_geometry_certificate(axis, matrix, ground, height_tolerance=.08):
    """A transformed USD Plane can cook as infinite; its tiny visual extent is irrelevant."""
    matrix = finite_array(matrix, (4, 4), "plane transform")
    if axis not in ("X", "Y", "Z") or not math.isfinite(ground) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("Plane requires a supported axis and invertible transform")
    if not np.allclose(matrix[:, 3], [0., 0., 0., 1.], atol=1e-8):
        raise ValueError("Non-affine plane transform")
    normal = np.eye(3)[("X", "Y", "Z").index(axis)] @ np.linalg.inv(matrix[:3, :3]).T
    normal /= np.linalg.norm(normal)
    if abs(normal[2]) < .9 or abs(matrix[3, 2] - ground) > height_tolerance:
        raise ValueError("Plane is not horizontal at the selected ground height")
    return {"axis": axis, "world_origin_m": matrix[3, :3].tolist(), "world_unit_normal": normal.tolist(),
            "visual_extent_not_used_for_PhysX_plane_coverage": True, "native_cooking_verified": False}


def flat_ground_collider_inventory(stage, ground):
    """Inspect all enabled collider types; recognize actual Plane and Mesh surfaces."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        record = {"path": str(prim.GetPath()), "type_name": str(prim.GetTypeName()), "enabled": True,
                  "near_ground_flat_surface": False, "static": True}
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get():
                record["static"] = False
            if any(a.GetNumTimeSamples() for a in ancestor.GetAttributes() if a.GetName().startswith("xformOp:")):
                record["static"] = False
            ancestor = ancestor.GetParent()
        try:
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            if str(prim.GetTypeName()) == "Plane":
                record["geometry"] = floor_plane_geometry_certificate(str(prim.GetAttribute("axis").Get() or "Z"), matrix, ground)
            elif prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                attrs = (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr())
                if any(a.GetNumTimeSamples() for a in attrs):
                    record["static"] = False
                record["geometry"] = floor_mesh_geometry_certificate(*(a.Get() for a in attrs), matrix, ground,
                                                                      str(mesh.GetOrientationAttr().Get()))
            else:
                record["reason"] = "Type inventoried; no flat-surface certificate. Native all-hit floor gate still required."
                records.append(record)
                continue
            record["near_ground_flat_surface"] = True
        except (ValueError, TypeError) as error:
            record["reason"] = str(error)
        records.append(record)
    return records


def triangle_xy_intersection_area(first, second):
    """Convex triangle intersection for coplanar duplicate-surface screening."""
    subject, clip = np.asarray(first, dtype=float)[:, :2], np.asarray(second, dtype=float)[:, :2]
    def cross(a, b):
        return float(a[0] * b[1] - a[1] * b[0])
    if cross(clip[1] - clip[0], clip[2] - clip[0]) < 0:
        clip = clip[::-1]
    polygon = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        output = []
        if not polygon:
            return 0.
        prev = polygon[-1]
        prev_side = cross(b - a, prev - a)
        for point in polygon:
            side = cross(b - a, point - a)
            if (side >= -1e-10) != (prev_side >= -1e-10):
                fraction = prev_side / (prev_side - side)
                output.append(prev + fraction * (point - prev))
            if side >= -1e-10:
                output.append(point)
            prev, prev_side = point, side
        polygon = output
    if len(polygon) < 3:
        return 0.
    polygon = np.asarray(polygon)
    return abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                      - polygon[:, 1] * np.roll(polygon[:, 0], -1))) / 2


def choose_unique_floor_surfaces(records):
    """Exclude named decorative layers and exact duplicate surfaces explicitly."""
    structural, excluded, unresolved = [], [], []
    for record in sorted(records, key=lambda item: item["path"]):
        if any(token in record["path"].lower() for token in ("decal", "carpet")):
            excluded.append({"path": record["path"], "reason": "explicit decorative decal/carpet name exclusion"})
            continue
        duplicate = None
        for other in structural:
            a, b = np.asarray(record["world_bounds_m"]), np.asarray(other["world_bounds_m"])
            if np.any(np.minimum(a[1, :2], b[1, :2]) - np.maximum(a[0, :2], b[0, :2]) <= 1e-5):
                continue
            overlap = sum(triangle_xy_intersection_area(t, u) for t in record["top_triangles_world_m"]
                          for u in other["top_triangles_world_m"])
            if overlap <= 1e-4:
                continue
            # Both surfaces near the chosen floor; do not allow stacked contacts.
            area, other_area = record["upward_top_fan_area_m2"], other["upward_top_fan_area_m2"]
            if abs(overlap - area) <= 1e-4 and abs(overlap - other_area) <= 1e-4:
                duplicate = other["path"]
                break
            unresolved.append({"first": other["path"], "second": record["path"], "overlap_area_m2": overlap})
        if duplicate is None:
            structural.append(record)
        else:
            excluded.append({"path": record["path"], "reason": "same whole XY floor surface as retained prim",
                             "retained": duplicate})
    return structural, excluded, unresolved


def make_session_paths_editable(stage, paths):
    """Materialize instance roots only in session to edit exact collider paths."""
    stage.SetEditTarget(stage.GetSessionLayer())
    changed = []
    for path in paths:
        for _ in range(len(path.split("/")) + 1):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Missing collision edit path: {path}")
            if not prim.IsInstanceProxy():
                break
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.IsInstance() and not ancestor.IsInstanceProxy():
                    root = str(ancestor.GetPath())
                    ancestor.SetInstanceable(False)
                    changed.append(root)
                    break
                ancestor = ancestor.GetParent()
            else:
                raise ValueError(f"Cannot resolve editable instance root for {path}")
        else:
            raise ValueError(f"Unresolved nested instance path: {path}")
    return sorted(set(changed))


def author_floor_overlay(stage, paths, disabled_paths=()):
    """Explicit workspace/session recipe; never save or overwrite original USD."""
    from pxr import Sdf, UsdPhysics
    stage.SetEditTarget(stage.GetSessionLayer())
    deinstanced = make_session_paths_editable(stage, list(paths) + list(disabled_paths))
    for path in disabled_paths:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        # This exact schema/attribute spelling is validated by Kit at execution.
        # Standalone USD can author it but cannot prove native PhysX cooking.
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.002)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if (not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none"
                or "PhysxCollisionAPI" not in prim.GetMetadata("apiSchemas").GetAppliedItems()
                or not np.isclose(prim.GetAttribute("physxCollision:contactOffset").Get(), .002)
                or prim.GetAttribute("physxCollision:restOffset").Get() != 0):
            raise ValueError(f"Floor overlay schema authoring failed: {path}")
    return {"paths": paths, "deinstanced_roots_in_session": deinstanced, "disabled_collision_paths": list(disabled_paths), "approximation": "none", "static_triangle_mesh": True,
            "contact_offset_m": .002, "rest_offset_m": 0., "native_cooking_verified": False}


def reference_ground_geometry(bounds, ground, margin=.2):
    """One finite plane from declared map bounds; no hole/stair inference."""
    bounds = finite_array(bounds, (2, 2), "ground bounds")
    if np.any(bounds[0] >= bounds[1]) or not math.isfinite(ground) or not math.isfinite(margin) or not .1 <= margin <= 1.:
        raise ValueError("Reference ground needs valid bounds/height and explicit .1..1m margin")
    low, high = bounds[0] - margin, bounds[1] + margin
    points = np.array([[low[0], low[1], ground], [high[0], low[1], ground],
                       [high[0], high[1], ground], [low[0], high[1], ground]])
    return points, np.array([3, 3]), np.array([0, 1, 2, 0, 2, 3])


def author_reference_ground(stage, bounds, ground, disabled_paths, margin=.2):
    """Explicit generated reference-plane owner, analogous to reference Terrain."""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    path = "/__PaperMapProbeReferenceGround"
    if stage.GetPrimAtPath(path):
        raise ValueError("Reference ground reserved path already exists; no stacked ground")
    stage.SetEditTarget(stage.GetSessionLayer())
    deinstanced = make_session_paths_editable(stage, disabled_paths)
    for disabled in disabled_paths:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(disabled)).CreateCollisionEnabledAttr(False)
    points, counts, indices = reference_ground_geometry(bounds, ground, margin)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr(counts.tolist())
    mesh.CreateFaceVertexIndicesAttr(indices.tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr("rightHanded")
    schema = author_floor_overlay(stage, [path])
    material = UsdShade.Material.Define(stage, path + "/PhysicsMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.)
    physics.CreateDynamicFrictionAttr(1.)
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose="physics")
    material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    material.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    return {"path": path, "floor_roots": [path], "finite_world_bounds_xy_m":
        [(np.asarray(bounds)[0] - margin).tolist(), (np.asarray(bounds)[1] + margin).tolist()],
        "height_m": ground, "margin_m": margin, "points": points.tolist(),
        "face_counts": counts.tolist(), "face_indices": indices.tolist(), "schema": schema,
        "disabled_visual_floor_collision_paths": list(disabled_paths),
        "disabled_flat_ground_collision_paths": list(disabled_paths), "deinstanced_floor_roots_in_session": deinstanced,
        "material": {"static_friction": 1., "dynamic_friction": 1., "restitution": 0., "combine_mode": "average"},
        "floor_owner": "reference_flat_terrain", "engineering_reference_ground_mode": True,
        "original_USD_floor_collider": False, "paper_private_scene_configuration": False,
        "holes_steps_and_other_floors_inferred": False, "native_cooking_verified": False}


def retain_projection_unknown(native_image, projection):
    """Unknown projection support must not become free through added plane."""
    native_image, projection = np.asarray(native_image), np.asarray(projection)
    if native_image.shape != projection.shape or native_image.ndim != 2:
        raise ValueError("Native/projection grids differ")
    if not np.isin(native_image, [0, 205, 254]).all() or not np.isin(projection, [0, 205, 254]).all():
        raise ValueError("Invalid map label")
    result = native_image.copy()
    result[(projection == 205) & (native_image != 0)] = 205
    return result


def preflight_scene(scene, projection_dir, output_dir, max_cells, floor_overlay="none", floor_owner="none", ground_margin=.2):
    """This function is only executed in a standalone-USD subprocess."""
    sys.path.insert(0, str(ROOT))
    from scripts.generate_pedestrian_maps import audit_composition
    from pxr import Usd, UsdGeom, UsdPhysics

    folder = Path(projection_dir) / scene
    data = json.loads((folder / "provenance.json").read_text())
    validated = validate_projection_metadata(scene, data)
    if math.prod(validated["dims"]) > max_cells:
        raise ValueError("Map exceeds explicit max-cells cap")
    source = Path(data["source_usd"]).resolve()
    if digest(source) != data["source_sha256"]:
        raise ValueError("Original scene source SHA disagrees with projection")
    input_files = {"provenance.json": file_record(folder / "provenance.json")}
    for name, expected in data["files"].items():
        if Path(name).name != name:
            raise ValueError("Projection file name contains a path")
        record = file_record(folder / name)
        if record["bytes"] != expected["bytes"] or record["sha256"] != expected["sha256"]:
            raise ValueError(f"Projection artifact hash mismatch: {name}")
        input_files[name] = record
    old_composition = json.loads((folder / "composition.json").read_text())
    if old_composition.get("composition_complete") is not True:
        raise ValueError("Previously recorded composition is incomplete")
    composition = audit_composition(source)
    if not composition["composition_complete"]:
        raise ValueError("Current scene composition incomplete; no PhysX fallback")
    layers = [file_record(item["path"]) for item in composition["layers"]]
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None or stage.GetCompositionErrors():
        raise ValueError("Current fully-loaded stage has composition errors")
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Actual USD stage is not Z-up/meters=1")
    colliders, bodies, animated = [], [], []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({"path": path, "type_name": str(prim.GetTypeName()),
                              "enabled": bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            bodies.append({"path": path, "enabled": bool(api.GetRigidBodyEnabledAttr().Get()),
                           "kinematic": bool(api.GetKinematicEnabledAttr().Get()),
                           "velocity": list(api.GetVelocityAttr().Get() or (0., 0., 0.)),
                           "angular_velocity": list(api.GetAngularVelocityAttr().Get() or (0., 0., 0.))})
        if any(a.GetNumTimeSamples() for a in prim.GetAttributes() if a.GetName().startswith("xformOp:")):
            animated.append(path)
    enabled = {r["path"] for r in colliders if r["enabled"]}
    ground_inventory = flat_ground_collider_inventory(stage, validated["ground"])
    floor_paths = set()
    for path in validated["floor_prims"]:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"Missing selected authored floor: {path}")
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if str(ancestor.GetPath()) in enabled:
                floor_paths.add(str(ancestor.GetPath()))
            ancestor = ancestor.GetParent()
        floor_paths.update(p for p in enabled if p.startswith(path + "/"))
    if not enabled:
        raise ValueError("No enabled authored colliders; no fallback")
    floor_geometry, blocked = [], []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in validated["floor_prims"]:
        prim = stage.GetPrimAtPath(path)
        try:
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError("Selected visual floor is not a Mesh")
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get():
                    raise ValueError(f"Floor has enabled rigid-body ancestor: {ancestor.GetPath()}")
                if any(a.GetNumTimeSamples() for a in ancestor.GetAttributes() if a.GetName().startswith("xformOp:")):
                    raise ValueError(f"Animated floor transform: {ancestor.GetPath()}")
                ancestor = ancestor.GetParent()
            mesh = UsdGeom.Mesh(prim)
            attributes = (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr())
            if any(a.GetNumTimeSamples() for a in attributes):
                raise ValueError("Animated floor mesh topology/points")
            cert = floor_mesh_geometry_certificate(*(a.Get() for a in attributes),
                np.asarray(xform_cache.GetLocalToWorldTransform(prim)), validated["ground"], str(mesh.GetOrientationAttr().Get()))
            floor_geometry.append({"path": path, "original_enabled_collision": path in enabled, **cert})
        except ValueError as error:
            blocked.append({"path": path, "reason": str(error)})
    unique, excluded, overlaps_unresolved = choose_unique_floor_surfaces(floor_geometry)
    # Explicit overlay owns all retained structural floors, including prior native
    # ones; named decorative/duplicate collision layers are disabled in session.
    recipe_paths = [r["path"] for r in unique]
    blocked_structural = [r for r in blocked if not any(t in r["path"].lower() for t in ("decal", "carpet"))]
    for item in blocked:
        if item not in blocked_structural:
            excluded.append({"path": item["path"], "reason": "decorative layer excluded; geometry not admitted: " + item["reason"]})
    disabled_paths = [r["path"] for r in excluded]
    original_floor_paths = sorted(floor_paths)
    recipe = {"schema": "dynbench_confirmed_visual_floor_collision_overlay.v1", "scene": scene,
              "source": file_record(source), "floor_roots": recipe_paths, "geometry": floor_geometry,
              "blocked_visual_floor_prims": blocked, "blocked_structural_floor_prims": blocked_structural,
              "original_floor_collision_paths": original_floor_paths,
              "excluded_floor_layers": excluded, "disabled_collision_paths": disabled_paths,
              "unresolved_surface_overlaps": overlaps_unresolved, "additional_Terrain_or_ground_created": False,
              "floor_owner": "scene_usd_mesh_overlay_engineering_alternative",
              "original_reference_floor_owner": "generated Terrain mesh; this is not the author default",
              "original_author_floor_physics_complete": all(p in enabled for p in validated["floor_prims"]),
              "approximation": "none", "contact_offset_m": .002, "rest_offset_m": 0.,
              "declared_engineering_completion": True, "native_physx_verified": False}
    write_json(Path(output_dir) / "floor_overlay_recipe.json", recipe)
    overlay_authored = None
    overlay_blocked = False
    if floor_overlay == "confirmed_visual_meshes":
        if blocked_structural or overlaps_unresolved or not recipe_paths:
            overlay_blocked = True
        else:
            overlay_authored = author_floor_overlay(stage, recipe_paths, disabled_paths)
            stage.GetSessionLayer().Export(str(Path(output_dir) / "floor_collision_overlay.usda"))
            floor_paths = set(recipe_paths)
    reference_ground = None
    if floor_owner == "reference_flat_terrain":
        if any(r["near_ground_flat_surface"] and not r["static"] for r in ground_inventory):
            raise ValueError("Animated/dynamic authored flat ground cannot be replaced by a static reference floor")
        disabled_flat = sorted(set(original_floor_paths) | {r["path"] for r in ground_inventory if r["near_ground_flat_surface"]})
        reference_ground = author_reference_ground(stage, validated["bounds"], validated["ground"], disabled_flat, ground_margin)
        reference_ground["original_flat_ground_collider_inventory"] = ground_inventory
        stage.GetSessionLayer().Export(str(Path(output_dir) / "reference_ground_overlay.usda"))
        reference_ground["overlay_source"] = file_record(Path(output_dir) / "reference_ground_overlay.usda")
        write_json(Path(output_dir) / "reference_ground_recipe.json", reference_ground)
        floor_paths = set(reference_ground["floor_roots"])
    if not source_unchanged({"layers": layers}):
        raise ValueError("Source layers changed during SDK-free floor inspection")
    projection = np.asarray(Image.open(folder / "occupancy.png"))
    if projection.shape != tuple(validated["dims"][::-1]) or projection.ndim != 2 or not np.isin(projection, [0, 205, 254]).all():
        raise ValueError("Projection image shape/labels invalid")
    result = {"schema": "dynbench_physx_map_preflight.v1", "scene": scene, "status": "failed" if overlay_blocked else "passed",
              "floor_overlay_blocked": overlay_blocked, "SDK_started": False, "script": file_record(__file__), "source": file_record(source), "layers": layers,
              "input_projection": input_files, "up_axis": "Z", "meters_per_unit": 1.0,
              "authored_colliders": colliders, "exact_floor_collider_paths": sorted(floor_paths),
              "flat_ground_collider_inventory": ground_inventory,
              "authored_rigid_bodies": bodies, "time_sampled_xform_paths": animated,
              "floor_owner": floor_owner, "reference_ground": reference_ground,
              "reference_ground_recipe": file_record(Path(output_dir) / "reference_ground_recipe.json") if reference_ground else None,
              "floor_overlay_mode": floor_overlay, "floor_overlay_recipe": file_record(Path(output_dir) / "floor_overlay_recipe.json"),
              "floor_overlay_schema_gate": overlay_authored, "original_enabled_floor_collision_paths": original_floor_paths,
              "original_author_floor_physics_complete": recipe["original_author_floor_physics_complete"],
              "projection_metadata": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in validated.items()},
              "navigation_ready": False, "original_composition_unchanged": True}
    write_json(Path(output_dir) / "preflight.json", result)
    return result


def native_hit_record(hit):
    return {"collision": str(hit.collision), "rigid_body": str(hit.rigid_body),
            "position": list(hit.position), "normal": list(hit.normal), "distance": float(hit.distance)}


def rays(query, xy, ground):
    hits = []
    def collect(hit):
        hits.append(native_hit_record(hit))
        return True
    any_hit = query.raycast_all((float(xy[0]), float(xy[1]), ground + .3), (0., 0., -1.), .6, collect, False)
    if bool(any_hit) != bool(hits):
        raise ValueError("Native ray boolean disagrees with callbacks")
    return hits


def overlaps(query, center, extent):
    hits = []
    def collect(hit):
        hits.append({"collision": str(hit.collision), "rigid_body": str(hit.rigid_body)})
        return True
    count = query.overlap_box(tuple(map(float, extent)), tuple(map(float, center)), (0., 0., 0., 1.), collect, False)
    if int(count) != len(hits):
        raise ValueError("Native overlap hit count does not match callbacks")
    return hits


def source_unchanged(preflight):
    return all(Path(r["path"]).stat().st_size == r["bytes"] and digest(r["path"]) == r["sha256"] for r in preflight["layers"])


def run_native_scene(app, scene, output_dir, preflight, args):
    # Imports are intentionally after AppLauncher; no standalone pxr in this process.
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils.stage import open_stage
    from isaaclab_physx.physics import PhysxCfg
    from isaacsim.asset.gen.omap.bindings import _omap
    from pxr import Usd, UsdGeom, UsdPhysics

    started = time.monotonic()
    output_dir = Path(output_dir)
    metadata = preflight["projection_metadata"]
    result = {"schema": "dynbench_physx_map_probe.v1", "scene": scene, "status": "failed",
              "navigation_ready": False, "author_eight_scene_map": False, "human_navigation_validated": False,
              "source": preflight["source"], "preflight": file_record(output_dir / "preflight.json"),
              "script": file_record(__file__), "native_started": True, "fallback_used": False,
              "limits": ["Selected flat floor and one static snapshot only.", "No robot or human clearance/route acceptance.",
                         "Four local scenes are not the paper's eight-scene benchmark."]}
    sim = None
    try:
        if not source_unchanged(preflight) or file_record(__file__) != preflight["script"]:
            raise ValueError("Admitted source/script changed before native probe")
        for record in preflight["input_projection"].values():
            if file_record(record["path"]) != record:
                raise ValueError("Admitted projection input changed before native probe")
        stage = open_stage(preflight["source"]["path"])
        if stage.GetCompositionErrors() or str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise ValueError("Kit composition/units differ from admitted standalone stage")
        stage.SetEditTarget(stage.GetSessionLayer())
        recipe = json.loads(Path(preflight["floor_overlay_recipe"]["path"]).read_text())
        if file_record(preflight["floor_overlay_recipe"]["path"]) != preflight["floor_overlay_recipe"]:
            raise ValueError("Admitted floor overlay recipe changed")
        if preflight["floor_overlay_mode"] == "confirmed_visual_meshes":
            from pxr import PhysxSchema
            author_floor_overlay(stage, recipe["floor_roots"], recipe["disabled_collision_paths"])
            for path in recipe["floor_roots"]:
                api = PhysxSchema.PhysxCollisionAPI(stage.GetPrimAtPath(path))
                if not stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxCollisionAPI) or not api or not np.isclose(api.GetContactOffsetAttr().Get(), .002) or api.GetRestOffsetAttr().Get() != 0:
                    raise ValueError("Actual Kit PhysxCollisionAPI rejects overlay schema/offsets")
        ground_recipe = preflight["reference_ground"]
        if ground_recipe is not None:
            if file_record(preflight["reference_ground_recipe"]["path"]) != preflight["reference_ground_recipe"]:
                raise ValueError("Admitted reference ground recipe changed")
            authored = author_reference_ground(stage, metadata["bounds"], metadata["ground"],
                ground_recipe["disabled_visual_floor_collision_paths"], ground_recipe["margin_m"])
            from pxr import PhysxSchema
            prim = stage.GetPrimAtPath(authored["path"])
            api = PhysxSchema.PhysxCollisionAPI(prim)
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI) or not np.isclose(api.GetContactOffsetAttr().Get(), .002):
                raise ValueError("Kit PhysxCollisionAPI rejects finite reference ground")
            if authored["deinstanced_floor_roots_in_session"] != ground_recipe["deinstanced_floor_roots_in_session"]:
                raise ValueError("Actual session instance edit paths differ from reference ground recipe")
            result["reference_ground"] = {**authored, "recipe": preflight["reference_ground_recipe"],
                                          "overlay_source": ground_recipe["overlay_source"]}
        result["floor_collision_overlay"] = {"mode": preflight["floor_overlay_mode"],
            "recipe": preflight["floor_overlay_recipe"], "floor_roots": recipe["floor_roots"],
            "original_author_floor_physics_complete": preflight["original_author_floor_physics_complete"],
            "source_USD_modified": False, "native_cooking_verified": False,
            "excluded_floor_layers": recipe["excluded_floor_layers"], "additional_Terrain_or_ground_created": ground_recipe is not None,
            "floor_owner": preflight["floor_owner"] if ground_recipe is not None else (recipe["floor_owner"] if preflight["floor_overlay_mode"] != "none" else "scene_usd_only"),
            "original_reference_floor_owner": recipe["original_reference_floor_owner"]}
        # Inactive session overrides hide authored PhysicsScenes without removing
        # their weaker source-layer specs; Lab RemovePrim alone cannot erase those.
        original_physics_scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        original_physics_paths = [str(p.GetPath()) for p in original_physics_scenes]
        for original_scene in original_physics_scenes:
            original_scene.SetActive(False)
        result["authored_physics_scenes_disabled_in_session"] = original_physics_paths
        physics_path = "/__PaperMapProbePhysicsScene"
        if stage.GetPrimAtPath(physics_path):
            raise ValueError("Reserved runtime physics path already exists")
        sim = SimulationContext(SimulationCfg(device="cpu", dt=DT, gravity=(0., 0., 0.),
            physics_prim_path=physics_path, use_fabric=False, enable_scene_query_support=True,
            physics=PhysxCfg(), visualizer_cfgs=[]))
        if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
            raise ValueError("Runtime edits escaped session layer")
        scene_api = UsdPhysics.Scene(stage.GetPrimAtPath(physics_path))
        scene_prim = scene_api.GetPrim()
        if scene_api.GetGravityMagnitudeAttr().Get() != 0 or scene_prim.GetAttribute("physxScene:enableGPUDynamics").Get() is not False:
            raise ValueError("CPU PhysX / zero gravity configuration not actually applied")
        if scene_prim.GetAttribute("physxScene:enableSceneQuerySupport").Get() is not True:
            raise ValueError("Actual PhysX scene queries are not enabled")
        if scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get() != 120:
            raise ValueError("Actual PhysX frequency is not 120Hz")
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        sim.reset()
        timeline.set_auto_update(False)
        sim.set_setting("/app/player/playSimulations", False)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            if time.monotonic() - started > args.load_timeout_s:
                raise TimeoutError("Stage loading deadline exceeded")
            app.update()
        enabled = {r["path"] for r in preflight["authored_colliders"] if r["enabled"]}
        if preflight["floor_overlay_mode"] == "confirmed_visual_meshes":
            enabled.update(recipe["floor_roots"])
            enabled.difference_update(recipe["disabled_collision_paths"])
        if ground_recipe is not None:
            enabled.difference_update(ground_recipe["disabled_visual_floor_collision_paths"])
            enabled.add(ground_recipe["path"])
        actual = {str(p.GetPath()) for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
        if actual != enabled:
            raise ValueError("Kit authored collision paths differ from preflight")
        paths = [r["path"] for r in preflight["authored_rigid_bodies"] if r["enabled"]]
        def body_transforms():
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            return {p: np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)), dtype=float) for p in paths}
        before = body_transforms()
        counter_before = sim.get_physics_step_count()
        for _ in range(args.snapshot_steps):
            sim.step(render=False)
        after = body_transforms()
        deltas = {p: float(np.max(np.abs(after[p] - before[p]))) for p in paths}
        query = omni.physx.get_physx_scene_query_interface()
        ground = metadata["ground"]
        origin = np.asarray(metadata["origin"])
        origin_hits = overlaps(query, origin, (.01, .01, .01))
        floor_paths = set(preflight["exact_floor_collider_paths"])
        actual_ground_inventory = flat_ground_collider_inventory(stage, ground)
        actual_flat_paths = {r["path"] for r in actual_ground_inventory if r["near_ground_flat_surface"]}
        origin_rays = rays(query, origin[:2], ground)
        origin_support = supported_floor_hit(origin_rays, floor_paths, ground)
        origin_ownership = classify_floor_ray_hits(origin_rays, floor_paths, enabled, ground)
        result["origin_verification"] = {"world_origin_m": origin.tolist(), "overlap_hits": origin_hits,
            "floor_ray_all_hits": origin_rays, "floor_support": origin_support, "floor_ownership": origin_ownership}
        if origin_hits:
            raise ValueError("Existing projection origin is occupied; no alternative origin/fallback")
        generator = _omap.Generator(omni.physx.get_physx_interface(), omni.usd.get_context().get_stage_id())
        generator.update_settings(RESOLUTION, 4, 5, 6)
        generator.set_transform(tuple(origin), tuple(metadata["relative_min"]), tuple(metadata["relative_max"]))
        query_counter = sim.get_physics_step_count()
        query_time = timeline.get_current_time()
        generator.generate2d()
        raw = np.asarray(generator.get_buffer(), dtype=np.float32)
        occupied = np.asarray(generator.get_occupied_positions(), dtype=np.float32).reshape(-1, 3)
        free = np.asarray(generator.get_free_positions(), dtype=np.float32).reshape(-1, 3)
        dims = np.asarray(generator.get_dimensions(), dtype=np.int64)
        low = np.asarray(generator.get_min_bound(), dtype=np.float64)
        high = np.asarray(generator.get_max_bound(), dtype=np.float64)
        # Preserve native output before any admission check, including failed mapping.
        np.savez_compressed(output_dir / "native_omap.npz", buffer=raw, occupied_positions=occupied,
                            free_positions=free, dimensions=dims, min_bound=low, max_bound=high)
        result["native_omap"] = file_record(output_dir / "native_omap.npz")
        image, mapping = reconstruct_native_map(raw, occupied, free, low, high, dims)
        # Retain every raw API position; canonical free uses only verified label5.
        api_free = free
        api_free_indices = native_position_buffer_indices(api_free, low, high, dims)
        api_free_labels = raw[api_free_indices]
        canonical_rows = np.flatnonzero(api_free_labels == VALUES["free"])
        free = api_free[canonical_rows]
        occupied_indices = native_position_buffer_indices(occupied, low, high, dims)
        unreported_indices = np.setdiff1d(np.arange(len(raw)), np.r_[occupied_indices, api_free_indices])
        np.savez_compressed(output_dir / "canonical_omap_world_positions.npz",
            occupied_world_positions=occupied, canonical_free_world_positions=free,
            canonical_free_API_row=canonical_rows, API_free_buffer_indices=api_free_indices,
            API_free_buffer_labels=api_free_labels,
            API_free_unknown_world_positions=api_free[api_free_labels == VALUES["unknown"]],
            unreported_unknown_buffer_indices=unreported_indices,
            unreported_unknown_buffer_labels=raw[unreported_indices])
        mapping["canonical_world_positions_artifact"] = file_record(output_dir / "canonical_omap_world_positions.npz")
        result["coordinate_validation"] = mapping
        origin_column_row = np.floor((origin[:2] - low[:2]) / RESOLUTION).astype(int)
        if image[image.shape[0] - 1 - origin_column_row[1], origin_column_row[0]] != 254:
            raise ValueError("Omap did not mark the verified origin cell free")
        expected_bounds = np.asarray(metadata["bounds"])
        if not np.allclose(np.stack((low[:2], high[:2])), expected_bounds, atol=2e-4, rtol=0):
            raise ValueError("Native actual map bounds differ from requested projection bounds")
        if not np.array_equal(dims[:2], metadata["dims"]):
            raise ValueError("Native actual dimensions differ from declared projection")
        selected = floor_probe_indices(len(free), args.floor_probe_limit)
        floor_queries = query_floor_support(query, free, selected, ground, floor_paths, enabled, scene=scene)
        support_mask = floor_queries["arrays"]["supported"]
        np.savez_compressed(output_dir / "floor_support_rays.npz", **floor_queries["arrays"])
        foreign_hit_paths = set(floor_queries["foreign_ray_collision_paths"]) | set(origin_ownership["foreign_ray_collision_paths"])
        undeclared_floor_paths = sorted(set(floor_queries["undeclared_near_floor_collision_paths"])
                                        | set(origin_ownership["undeclared_near_floor_collision_paths"]))
        floor_ownership_verified = (actual_flat_paths == floor_paths
            and origin_ownership["all_support_hit_paths_declared"] and floor_queries["all_support_hit_paths_declared"])
        result["floor_ownership"] = {"schema": "dynbench.native_floor_owner.v1",
            "actual_enabled_flat_ground_inventory": actual_ground_inventory,
            "actual_enabled_flat_ground_paths": sorted(actual_flat_paths), "declared_floor_paths": sorted(floor_paths),
            "undeclared_near_floor_collision_paths": undeclared_floor_paths,
            "foreign_ray_collision_paths": sorted(foreign_hit_paths),
            "all_support_hit_paths_declared": origin_ownership["all_support_hit_paths_declared"] and floor_queries["all_support_hit_paths_declared"],
            "single_declared_floor_owner_verified": ground_recipe is not None and len(floor_paths) == 1 and floor_ownership_verified,
            "floor_ownership_verified": floor_ownership_verified, "full_grid_floor_rays": args.floor_probe_limit == 0,
            "all_ray_hits_saved": True, "undeclared_near_floor_hit_rows": int(floor_queries["arrays"]["undeclared_near_floor_hit"].sum()),
            "foreign_hit_rows": int(floor_queries["arrays"]["foreign_hit"].sum()),
            "height_tolerance_m": .08, "normal_z_min": .9, "artifact": file_record(output_dir / "floor_support_rays.npz")}
        band_queries = query_full_height_band(query, free, selected, ground, floor_paths, enabled,
            counter_getter=sim.get_physics_step_count, time_getter=timeline.get_current_time, scene=scene)
        np.savez_compressed(output_dir / "full_height_band_queries.npz", **band_queries["arrays"])
        band_clock_path = output_dir / "full_height_band_query_clock.json"
        write_json(band_clock_path, {key: value for key, value in band_queries.items() if key != "arrays"})
        result["full_height_band_queries"] = {"artifact": file_record(output_dir / "full_height_band_queries.npz"),
            "clock_artifact": file_record(band_clock_path), "queried_cells": len(selected),
            "clock_before": band_queries["clock_before"], "clock_after": band_queries["clock_after"],
            "query_time_unchanged": band_queries["query_time_unchanged"],
            "foreign_hit_paths": band_queries["foreign_hit_paths"]}
        require_query_clock_unchanged(band_queries["clock_before"], band_queries["clock_after"])
        foreign_hit_paths.update(band_queries["foreign_hit_paths"])
        refined = refine_known_free_map(image, free, selected, support_mask, band_queries["arrays"], low, high, dims)
        Image.fromarray(image).save(output_dir / "native_reconstructed.png")
        Image.fromarray(refined).save(output_dir / "height_band_refined.png")
        projection_folder = Path(args.projection_dir) / scene
        projection = np.asarray(Image.open(projection_folder / "occupancy.png"))
        probes = fixed_probe_cells(image.shape, args.comparison_points, 7)
        comparisons = []
        sampled_free_overlap = 0
        for row, column in probes:
            xy = low[:2] + np.array([column + .5, image.shape[0] - row - .5]) * RESOLUTION
            center = (xy[0], xy[1], ground + sum(BAND) / 2)
            hits = overlaps(query, center, (RESOLUTION / 2, RESOLUTION / 2, (BAND[1] - BAND[0]) / 2))
            foreign_hit_paths.update(h["collision"] for h in hits if h["collision"] not in enabled)
            value = int(image[row, column])
            refined_value = int(refined[row, column])
            included_hits = [hit for hit in hits if hit["collision"] not in floor_paths]
            sampled_free_overlap += int(refined_value == 254 and bool(included_hits))
            comparisons.append({"pixel_row_column": [int(row), int(column)], "world_xy_m": xy.tolist(),
                "CPU_projection_label": int(projection[row, column]), "native_omap_label": value,
                "height_band_refined_label": refined_value, "native_band_box_overlap_hits": hits,
                "projection_native_discrepancy": int(projection[row, column]) != value})
        write_json(output_dir / "fixed_world_point_comparison.json", {"seed": 7, "points": comparisons,
            "native_probe": "full .1m cell XY box over .15..1.8m band; not a pedestrian footprint"})
        if sim.get_physics_step_count() != query_counter or timeline.get_current_time() != query_time:
            raise ValueError("Physics/timeline advanced while map and support queries were sampled")
        unchanged = source_unchanged(preflight)
        static = (not preflight["time_sampled_xform_paths"]
                  and not any(r["enabled"] and not r["kinematic"] for r in preflight["authored_rigid_bodies"])
                  and all(d <= 1e-5 for d in deltas.values()))
        full = args.floor_probe_limit == 0 and len(selected) == len(free)
        origin_row = image.shape[0] - 1 - origin_column_row[1]
        origin_band_clear = refined[origin_row, origin_column_row[0]] == 254
        ready = navigation_admission(origin_clear=not origin_hits and origin_band_clear,
            origin_supported=origin_support is not None, mapping_valid=True, full_floor_probe=full,
            full_height_probe=full, height_clear_mask=band_queries["arrays"]["full_height_clear"],
            all_band_hit_paths_classified=band_queries["arrays"]["all_hit_paths_classified"].all(),
            floor_ownership_verified=floor_ownership_verified,
            support_mask=support_mask, static_unchanged=static,
            sampled_free_overlap_count=sampled_free_overlap + len(foreign_hit_paths), source_unchanged=unchanged)
        projection = np.asarray(Image.open(Path(args.projection_dir) / scene / "occupancy.png"))
        exported = retain_projection_unknown(refined, projection) if ground_recipe is not None else refined
        Image.fromarray(exported).save(output_dir / "occupancy.png")
        (output_dir / "occupancy.yaml").write_text(f"image: occupancy.png\nresolution: 0.1\norigin: [{low[0]}, {low[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
        stage.GetSessionLayer().Export(str(output_dir / "runtime_session.usda"))
        result.update(status="scene_resource_gate_passed" if ready else "probe_complete_not_navigation_ready",
            scene_resource_gate_passed=ready, navigation_ready=False, navigation_quality_validated=False,
            projection_unknown_cells_preserved=ground_recipe is not None, native_probe_completed=True,
            native_mapping_verified=True, requested_height_band_world_m=[ground + BAND[0], ground + BAND[1]],
            actual_bounds_world_m=[low.tolist(), high.tolist()], actual_dimensions=dims.tolist(),
            selected_floor_height_m=ground, floor_collider_paths=sorted(floor_paths),
            height_band_refinement={"method": "native2D omap plus explicit native full-height cell queries and floor support",
                "engineering_refinement": True, "pure_omap2D_full_height_claim": False, "author_map": False,
                "cell_xy_size_m": RESOLUTION, "band_above_selected_floor_m": list(BAND),
                "pedestrian_footprint_or_body_clearance": False, "all_native_known5_cells_tested": full,
                "known5_tested": len(selected), "known5_source": len(free),
                "height_blocked_known5_cells": int(band_queries["arrays"]["authored_obstacle_hit"].sum()),
                "height_clear_known5_cells": int(band_queries["arrays"]["full_height_clear"].sum()),
                "unclassified_cells": int((~band_queries["arrays"]["all_hit_paths_classified"]).sum()),
                "unsupported_tested_cells": int((~support_mask).sum()), "native_unknown_never_promoted": True,
                "untested_known5_exported_unknown": len(free)-len(selected),
                "refined_value_counts": {"occupied": int((refined == 0).sum()), "free": int((refined == 254).sum()), "unknown": int((refined == 205).sum())}},
            floor_support={"full_grid": full, "native_free_cells": len(free), "tested": len(selected),
                           "supported": int(support_mask.sum()), "unsupported": int((~support_mask).sum()),
                           "ray_start_above_floor_m": .3, "ray_length_m": .6, "normal_z_min": .9,
                           "height_tolerance_m": .08, "artifact": file_record(output_dir / "floor_support_rays.npz")},
            physics={"device": sim.device, "dt_s": DT, "gravity": [0., 0., 0.], "scene_query_support": True,
                     "counter_before_snapshot": counter_before, "counter_at_query": query_counter,
                     "snapshot_steps": args.snapshot_steps, "timeline_at_query_s": query_time,
                     "timeline_after_queries_s": timeline.get_current_time(), "no_query_time_advance": True,
                     "snapshot_physics_time_increment_s": (query_counter - counter_before) * DT,
                     "automatic_timeline_update": False, "app_player_playSimulations": False},
            dynamic_activity={"authored_rigid_bodies": preflight["authored_rigid_bodies"],
                              "USD_body_transform_max_abs_delta": deltas, "static_snapshot_gate": static,
                              "time_sampled_xform_paths": preflight["time_sampled_xform_paths"],
                              "limits": "USD transform diagnostic only; enabled nonkinematic actors/animated transforms reject the static navigation gate"},
            sampled_free_cells_with_overlap=sampled_free_overlap, foreign_native_hit_paths=sorted(foreign_hit_paths),
            projection_discrepancy_count=sum(r["projection_native_discrepancy"] for r in comparisons),
            original_source_layers_unchanged=unchanged, runtime_USD_version=list(Usd.GetVersion()),
            exports={name: file_record(output_dir / name) for name in ("occupancy.png", "native_reconstructed.png", "height_band_refined.png", "occupancy.yaml", "runtime_session.usda", "fixed_world_point_comparison.json")})
        if not unchanged:
            raise ValueError("Original source layer contents changed")
        if not floor_ownership_verified:
            raise ValueError("Native all-hit rays or flat collider inventory reject declared floor ownership")
    except Exception as error:
        result.update(status="failed", navigation_ready=False, error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write_json(output_dir / "result.json", result)
        if sim is not None:
            sim.stop()
            SimulationContext.clear_instance()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection-dir", default=str(ROOT / "data/diffusion_amp/maps_resource_complete"))
    p.add_argument("--output", required=True, help="Fresh directory; existing output is refused")
    p.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    p.add_argument("--execute-physx", action="store_true", help="Starts Kit; default only standalone USD preflight")
    p.add_argument("--preflight-worker", choices=SCENES, help=argparse.SUPPRESS)
    p.add_argument("--floor-owner", choices=("none", "reference_flat_terrain"), default="none",
                   help="Explicit finite generated reference-ground mode; no original-USD or private-paper claim")
    p.add_argument("--reference-ground-margin-m", type=float, default=.2)
    p.add_argument("--floor-collision-overlay", choices=("none", "confirmed_visual_meshes"), default="none",
                   help="Explicit engineering completion of confirmed original visual floor meshes; default preserves missing/disabled floors")
    p.add_argument("--floor-probe-limit", type=int, default=256, help="0=full-height and floor queries for all canonical known5 cells; limited queries never ready")
    p.add_argument("--comparison-points", type=int, default=256)
    p.add_argument("--snapshot-steps", type=int, default=2)
    p.add_argument("--load-timeout-s", type=float, default=300)
    p.add_argument("--max-cells", type=int, default=2_000_000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.floor_probe_limit < 0 or args.comparison_points < 1 or args.snapshot_steps < 1 or args.max_cells < 1:
        raise ValueError("Probe limits/counts must be positive (floor limit may be zero)")
    if not math.isfinite(args.load_timeout_s) or args.load_timeout_s <= 0:
        raise ValueError("Load timeout must be finite and positive")
    if args.floor_owner != "none" and args.floor_collision_overlay != "none":
        raise ValueError("Only one explicit ground owner allowed; no stacked floor modes")
    if not math.isfinite(args.reference_ground_margin_m) or not .1 <= args.reference_ground_margin_m <= 1.:
        raise ValueError("Reference ground margin must be finite within .1..1m")
    output = Path(args.output).resolve()
    if args.preflight_worker:
        checked = preflight_scene(args.preflight_worker, args.projection_dir, output, args.max_cells, args.floor_collision_overlay, args.floor_owner, args.reference_ground_margin_m)
        return 0 if checked["status"] == "passed" else 1
    if len(set(args.scenes)) != len(args.scenes):
        raise ValueError("Duplicate scene arguments refused")
    output.mkdir(parents=True, exist_ok=False)
    reports, failed = [], []
    for scene in args.scenes:
        folder = output / scene
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--preflight-worker", scene,
                   "--projection-dir", str(Path(args.projection_dir).resolve()), "--output", str(folder),
                   "--max-cells", str(args.max_cells), "--floor-collision-overlay", args.floor_collision_overlay,
                   "--floor-owner", args.floor_owner, "--reference-ground-margin-m", str(args.reference_ground_margin_m)]
        try:
            with (folder / "preflight.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.load_timeout_s)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            record = {"scene": scene, "error": str(error), "log": file_record(folder / "preflight.log")}
            if (folder / "preflight.json").exists():
                record["preflight"] = json.loads((folder / "preflight.json").read_text())
            failed.append(record)
            continue
        reports.append(json.loads((folder / "preflight.json").read_text()))
    write_json(output / "manifest.json", {"schema": "dynbench_physx_maps.v1", "status": "preflight_failed" if failed else "preflight_passed",
        "Kit_started": False, "navigation_ready": False, "failed_scenes": failed,
        "scene_preflights": reports, "script": file_record(__file__)})
    if failed:
        return 1
    if not args.execute_physx:
        return 0
    # This parent never imports standalone pxr. One actual Kit App, all scenes.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device="cpu", enable_cameras=False)
    app = launcher.app
    try:
        import torch
        torch.set_num_threads(4)
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.set_extension_enabled_immediate("isaacsim.asset.gen.omap", True):
            raise RuntimeError("Actual omap extension did not enable")
        actual = [run_native_scene(app, r["scene"], output / r["scene"], r, args) for r in reports]
        write_json(output / "manifest.json", {"schema": "dynbench_physx_maps.v1", "Kit_started": True,
            "status": "passed" if all(r.get("scene_resource_gate_passed", False) for r in actual) else "not_navigation_ready",
            "navigation_ready": False, "scene_resource_gate_passed": all(r.get("scene_resource_gate_passed", False) for r in actual), "scenes": actual,
            "script": file_record(__file__), "result_JSON_authoritative": True})
        return 0 if all(r.get("scene_resource_gate_passed", False) for r in actual) else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
