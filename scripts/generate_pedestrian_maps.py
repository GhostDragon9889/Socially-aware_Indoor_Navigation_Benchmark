#!/usr/bin/env python3
"""Generate explicit CPU USD-projection prototypes, never PhysX-verified maps.

Each scene runs in a bounded subprocess. Remote composition dependencies are
rejected before opening a stage. Missing local geometry may be inspected with
--allow-incomplete, but its export has no free cells: unoccupied candidates are
unknown until that geometry is supplied. Original scene files remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "NewMetro": "",
    "SecondFloor": "",
    "ThirdFloor": "",
    "Full_Warehouse": "Full_Warehouse/full_warehouse.usd",
    "Warehouse": "Warehouse/warehouse.usd",
    "Office": "Office/office_partnav.usd",
    "Hospital": "Hospital/hospital_partnav.usd",
    "YOHO": str(ROOT / "data/diffusion_amp/scenes/yoho_static.usda"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def audit_composition(usd_path):
    """Inspect local layer arcs without resolving any remote asset URI."""
    from pxr import Sdf

    pending = [Path(usd_path).resolve()]
    layers, missing, remote, invalid = {}, [], [], []
    arcs = 0
    while pending:
        path = pending.pop().resolve()
        if str(path) in layers:
            continue
        record = {"path": str(path), "exists": path.is_file()}
        layers[str(path)] = record
        if not path.is_file():
            missing.append(str(path))
            continue
        record.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                invalid.append({"path": str(path), "reason": "unmaterialized_git_lfs_pointer"})
                continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            invalid.append({"path": str(path), "reason": "cannot_open_local_layer"})
            continue
        record["composition_dependencies"] = list(layer.GetCompositionAssetDependencies())
        for reference in record["composition_dependencies"]:
            # Empty asset paths identify internal references in flattened USD.
            # They resolve in the current layer, not a filesystem directory.
            if reference == "":
                continue
            arcs += 1
            if "://" in reference:
                remote.append({"layer": str(path), "asset": reference})
                continue
            target = Path(reference)
            target = target if target.is_absolute() else path.parent / target
            if target.suffix.lower() in {".usd", ".usda", ".usdc"}:
                pending.append(target)
            else:
                invalid.append({"path": str(target), "reason": "unsupported_composition_asset_type"})
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    metadata = {key: layer.pseudoRoot.GetInfo(key) for key in ("metersPerUnit", "upAxis", "defaultPrim")
                if layer.pseudoRoot.HasInfo(key)}
    return {"root": str(Path(usd_path).resolve()), "root_sha256": sha256(usd_path),
            "metadata": metadata, "layers": list(layers.values()), "arc_count": arcs,
            "missing_local_layers": missing, "remote_composition_arcs": remote,
            "invalid_layers": invalid, "composition_complete": not (missing or remote or invalid),
            "texture_or_shader_resolution_checked": False,
            "dependency_content_hashes_recomputed": False}


def stage_to_map_matrix(up_axis, meters_per_unit):
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("Stage metersPerUnit must be finite and positive")
    if up_axis == "Z":
        rotation = np.eye(3)
    elif up_axis == "Y":
        # Right handed +90deg about X: map(X,Y,Z) = stage(X,-Z,Y).
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    else:
        raise ValueError(f"Unsupported USD up axis: {up_axis}")
    return rotation * meters_per_unit


def transformed_bounds(low, high, matrix):
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    mapped = corners @ matrix.T
    return mapped.min(0), mapped.max(0)


def collect_geometry(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    matrix = stage_to_map_matrix(up_axis, units)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"],
                            False, True)
    types = Counter()
    colliders = []
    geometry = []
    skipped = Counter()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        types[prim.GetTypeName()] += 1
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if collision:
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            colliders.append({"prim": str(prim.GetPath()), "type": prim.GetTypeName(),
                              "collision_enabled": bool(enabled)})
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        collider_ancestors = []
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                collider_ancestors.append(str(ancestor.GetPath()))
            ancestor = ancestor.GetParent()
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible and not collider_ancestors:
            skipped["invisible_without_collision_schema"] += 1
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned.IsEmpty():
            skipped["empty_bound"] += 1
            continue
        low, high = transformed_bounds(np.asarray(aligned.GetMin()), np.asarray(aligned.GetMax()), matrix)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"Nonfinite world bounds: {prim.GetPath()}")
        geometry.append({"prim": str(prim.GetPath()), "type": prim.GetTypeName(),
                         "min_map_m": low.tolist(), "max_map_m": high.tolist(),
                         "collision_schema_self": collision,
                         "collision_schema_ancestors": collider_ancestors,
                         "time_sampled_transform": any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()
                                                       if attr.GetName().startswith("xformOp:"))})
    if not geometry:
        raise ValueError("Stage has no finite composed geometry; no empty map is exported")
    return geometry, {"usd_version": list(Usd.GetVersion()), "stage_up_axis": up_axis,
                      "stage_meters_per_unit": units, "stage_units_to_map_meters": matrix.tolist(),
                      "gprim_count": len(geometry), "prim_types": dict(types),
                      "authored_collision_schema_count": len(colliders),
                      "authored_colliders": colliders, "skipped_geometry": dict(skipped)}


def choose_floor(geometry, ground_height=None):
    candidates = []
    for record in geometry:
        low, high = np.array(record["min_map_m"]), np.array(record["max_map_m"])
        area = float(np.prod((high - low)[:2]))
        path = record["prim"].lower()
        if any(token in path for token in ("floor", "ground", "plane")) and area >= 1 and high[2] - low[2] <= .25:
            candidates.append({**record, "area_xy_m2": area, "surface_height_m": float(high[2])})
    if ground_height is None:
        if not candidates:
            raise ValueError("Cannot infer a floor. Supply --ground-height explicitly; no silent z=0 assumption")
        area_by_height = defaultdict(float)
        for record in candidates:
            area_by_height[round(record["surface_height_m"] / .1) * .1] += record["area_xy_m2"]
        ground_height = min(area_by_height, key=lambda height: (-area_by_height[height], height))
        inference = "dominant .1m floor-height cluster weighted by XY AABB area; ties choose lower"
    else:
        inference = "explicit user supplied ground height in map meters"
    selected = [record for record in candidates if abs(record["surface_height_m"] - ground_height) <= .1]
    if not selected:
        raise ValueError("Chosen floor height has no authored horizontal support candidate")
    return float(ground_height), selected, {"height_m": float(ground_height), "inference": inference,
                                          "candidate_count": len(candidates), "selected_floor_count": len(selected),
                                          "selected_floor_prims": [record["prim"] for record in selected],
                                          "support_geometry": "union of horizontal authored floor AABBs, a prototype approximation"}


def aligned_grid_bounds(records, resolution, margin, bounds_quantum=None):
    lower = np.min([record["min_map_m"][:2] for record in records], axis=0)
    upper = np.max([record["max_map_m"][:2] for record in records], axis=0)
    origin = np.floor((lower - margin) / resolution) * resolution
    maximum = np.ceil((upper + margin) / resolution) * resolution
    if bounds_quantum is not None:
        origin = np.floor(origin / bounds_quantum) * bounds_quantum
        maximum = np.ceil(maximum / bounds_quantum) * bounds_quantum
    size = np.rint((maximum - origin) / resolution).astype(int)
    if np.any(size <= 0):
        raise ValueError("Map requires nonzero world XY extents")
    return origin, maximum, size


def pixel_rectangle(low, high, origin, resolution, shape, *, cell_centers=False):
    height, width = shape
    low_grid = (np.asarray(low)[:2] - origin) / resolution
    high_grid = (np.asarray(high)[:2] - origin) / resolution
    if cell_centers:
        # Free support is only assigned when a pixel center is inside an
        # authored floor footprint, never an extra border cell at its edge.
        minimum = np.ceil(low_grid - .5 - 1e-9).astype(int)
        maximum = np.floor(high_grid - .5 + 1e-9).astype(int)
    else:
        # Obstacles conservatively include cells touched by their boundaries.
        minimum = np.floor(low_grid - 1e-9).astype(int)
        maximum = np.floor(high_grid + 1e-9).astype(int)
    if maximum[0] < 0 or maximum[1] < 0 or minimum[0] >= width or minimum[1] >= height:
        return None
    minimum = np.maximum(minimum, 0)
    maximum = np.minimum(maximum, (width - 1, height - 1))
    return (slice(height - 1 - maximum[1], height - minimum[1]),
            slice(minimum[0], maximum[0] + 1))


def world_to_pixel(xy, origin, resolution, shape):
    cell = np.floor((np.asarray(xy) - origin) / resolution).astype(int)
    return np.stack((shape[0] - 1 - cell[..., 1], cell[..., 0]), axis=-1)


def pixel_to_world_center(row_column, origin, resolution, shape):
    pixels = np.asarray(row_column)
    return origin + np.stack((pixels[..., 1] + .5, shape[0] - pixels[..., 0] - .5), axis=-1) * resolution


def project_geometry(geometry, floor_records, *, origin, size, resolution, ground_height,
                     height_min=.15, height_max=1.8, incomplete=False):
    shape = (int(size[1]), int(size[0]))
    support = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    for record in floor_records:
        rectangle = pixel_rectangle(record["min_map_m"], record["max_map_m"], origin, resolution, shape,
                                    cell_centers=True)
        if rectangle:
            support[rectangle] = True
    projected = []
    skipped = Counter()
    for record in geometry:
        lower, upper = record["min_map_m"], record["max_map_m"]
        if upper[2] < ground_height + height_min:
            skipped["below_pedestrian_height_band"] += 1
            continue
        if lower[2] > ground_height + height_max:
            skipped["above_pedestrian_height_band"] += 1
            continue
        rectangle = pixel_rectangle(lower, upper, origin, resolution, shape)
        if rectangle is None:
            skipped["outside_selected_world_bounds"] += 1
            continue
        occupied[rectangle] = True
        projected.append(record)
    if not occupied.any():
        raise ValueError("Projection has zero occupied cells; reject a blank scene map")
    preview = np.full(shape, 205, dtype=np.uint8)
    preview[support] = 254
    preview[occupied] = 0
    exported = preview.copy()
    if incomplete:
        # Missing geometry has no trustworthy extent. It must not disappear
        # into an apparently free scene; retain known obstacles, no free cells.
        exported[exported == 254] = 205
    return exported, preview, {"projected_obstacle_gprims": len(projected), "skipped_obstacles": dict(skipped),
                               "occupied_cells": int((exported == 0).sum()),
                               "free_cells": int((exported == 254).sum()),
                               "unknown_cells": int((exported == 205).sum()),
                               "preview_free_candidates": int((preview == 254).sum()),
                               "projected_obstacle_bounds": projected}


def triangle_projection(stage, geometry, *, origin, size, resolution, ground_height, height_min, height_max):
    """Explicit conservative projection of actual clipped triangles, not merged AABBs.

    Native PhysX full-height validation remains mandatory. One-cell dilation
    covers raster edge rounding; no unknown floor support becomes known free.
    """
    from pxr import Usd, UsdGeom
    from PIL import ImageDraw
    from scipy.ndimage import binary_dilation
    canvas = Image.new("L", tuple(map(int,size)), 0);draw=ImageDraw.Draw(canvas)
    cache=UsdGeom.XformCache(Usd.TimeCode.Default());triangles=0
    lower,upper=ground_height+height_min,ground_height+height_max
    def clip(poly,z,above):
        output=[]
        if not poly:return output
        a=poly[-1];ain=(a[2]>=z) if above else (a[2]<=z)
        for b in poly:
            bin=(b[2]>=z) if above else (b[2]<=z)
            if ain!=bin:output.append(a+(b-a)*((z-a[2])/(b[2]-a[2])))
            if bin:output.append(b)
            a,ain=b,bin
        return output
    for rec in geometry:
        if rec['max_map_m'][2]<lower or rec['min_map_m'][2]>upper:continue
        prim=stage.GetPrimAtPath(rec['prim'])
        if not prim.IsA(UsdGeom.Mesh):
            rect=pixel_rectangle(rec['min_map_m'],rec['max_map_m'],origin,resolution,(int(size[1]),int(size[0])))
            if rect:draw.rectangle((rect[1].start,rect[0].start,rect[1].stop-1,rect[0].stop-1),fill=255)
            continue
        mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        world=np.column_stack((points,np.ones(len(points))))@np.asarray(cache.GetLocalToWorldTransform(prim))
        cursor=0
        for count in counts:
            face=indices[cursor:cursor+count];cursor+=count
            for j in range(1,len(face)-1):
                poly=clip(clip([world[k,:3] for k in [face[0],face[j],face[j+1]]],lower,True),upper,False)
                if not poly:continue
                xy=(np.asarray(poly)[:,:2]-origin)/resolution
                pixels=[(float(x-.5),float(size[1]-.5-y)) for x,y in xy]
                draw.polygon(pixels,fill=255);triangles+=1
    return binary_dilation(np.asarray(canvas)>0,iterations=1),{'clipped_triangle_count':triangles,'conservative_raster_dilation_cells':1,'native_verified':False}


def generate_scene(args):
    from pxr import Usd
    from scipy.ndimage import distance_transform_edt

    started = time.monotonic()
    output = Path(args.output).resolve() / args.worker
    output.mkdir(parents=True, exist_ok=True)
    usd = Path(args.source_usd).resolve() if getattr(args, "source_usd", None) else Path(args.assets).resolve() / SCENES[args.worker]
    composition = audit_composition(usd)
    write_json(output / "composition.json", composition)
    if composition["remote_composition_arcs"] or composition["invalid_layers"]:
        raise ValueError("Unresolved remote/invalid composition dependency; no stage/map fallback")
    if composition["missing_local_layers"] and not args.allow_incomplete:
        raise ValueError("Missing local geometry; --allow-incomplete exports known occupied plus unknown, never free")
    if args.inventory_only:
        return {"scene": args.worker, "status": "inventory_complete" if composition["composition_complete"] else "inventory_incomplete",
                "composition_complete": composition["composition_complete"], "composition": str(output / "composition.json")}
    stage = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError("Could not open composed stage")
    if stage.GetCompositionErrors() and composition["composition_complete"]:
        raise ValueError("Unexpected composition errors; refuse a misleading map")
    geometry, stage_info = collect_geometry(stage)
    ground, floors, floor_info = choose_floor(geometry, args.ground_height)
    extent_records = geometry if args.extent_mode == "world" else floors
    origin, maximum, size = aligned_grid_bounds(extent_records, args.resolution, args.margin, args.native_bounds_quantum_m)
    if int(np.prod(size)) > args.max_cells:
        raise ValueError(f"Grid {size.tolist()} exceeds explicit max-cells {args.max_cells}; change resolution or extent-mode")
    incomplete = not composition["composition_complete"]
    occupancy, preview, statistics = project_geometry(geometry, floors, origin=origin, size=size,
        resolution=args.resolution, ground_height=ground, height_min=args.height_min,
        height_max=args.height_max, incomplete=incomplete)
    if args.triangle_projection:
        blocked,triangle_stats=triangle_projection(stage,geometry,origin=origin,size=size,resolution=args.resolution,ground_height=ground,height_min=args.height_min,height_max=args.height_max)
        # Preserve the separately declared floor support/unknown mask.
        support=np.zeros(occupancy.shape,dtype=bool)
        for floor in floors:
            rect=pixel_rectangle(floor['min_map_m'],floor['max_map_m'],origin,args.resolution,occupancy.shape,cell_centers=True)
            if rect:support[rect]=True
        preview=np.full(occupancy.shape,205,dtype=np.uint8);preview[support]=254;preview[blocked]=0
        occupancy=preview.copy()
        if incomplete:occupancy[occupancy==254]=205
        statistics.update(triangle_stats,projection_geometry='actual world triangles clipped to pedestrian height band',occupied_cells=int((occupancy==0).sum()),free_cells=int((occupancy==254).sum()),unknown_cells=int((occupancy==205).sum()),preview_free_candidates=int((preview==254).sum()))
    Image.fromarray(occupancy).save(output / "occupancy.png")
    Image.fromarray(preview).save(output / "projection_preview.png")
    metadata = {"image": "occupancy.png", "resolution": args.resolution,
                "origin": [float(origin[0]), float(origin[1]), 0.], "negate": 0,
                "occupied_thresh": .65, "free_thresh": .196}
    (output / "occupancy.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    sample = None
    if (occupancy == 254).any():
        clearance = distance_transform_edt(occupancy == 254) * args.resolution
        row_column = np.array(np.unravel_index(clearance.argmax(), occupancy.shape))
        xy = pixel_to_world_center(row_column, origin, args.resolution, occupancy.shape)
        if args.native_bounds_quantum_m is not None:
            # Choose a binary-exact origin from prepared free cells; no native
            # labels or returned bounds are fitted or changed after probing.
            q = args.native_bounds_quantum_m
            candidates = np.array([[x,y] for x in np.arange(origin[0]+q,maximum[0],q)
                                   for y in np.arange(origin[1]+q,maximum[1],q)])
            pixels = world_to_pixel(candidates, origin, args.resolution, occupancy.shape)
            valid = clearance[pixels[:,0],pixels[:,1]] >= .4
            candidates,pixels = candidates[valid],pixels[valid]
            if not len(candidates):raise ValueError('No prepared clear quantized omap origin')
            chosen = np.linalg.norm(candidates-xy,axis=-1).argmin()
            xy,row_column = candidates[chosen],pixels[chosen]
        sample = {"pixel_row_column": row_column.tolist(), "world_xy_m": xy.tolist(),
                  "projection_clearance_m": float(clearance[tuple(row_column)]),
                  "omap_origin_world_map_frame_m": [float(xy[0]), float(xy[1]), ground + .75],
                  "omap_relative_min_bound_map_m": [float(origin[0] - xy[0]), float(origin[1] - xy[1]), -.6],
                  "omap_relative_max_bound_map_m": [float(maximum[0] - xy[0]), float(maximum[1] - xy[1]), 1.05],
                  "physx_origin_not_yet_verified_free": True}
    pixels = np.array([[0, 0], [size[1] - 1, size[0] - 1], [size[1] // 2, size[0] // 2]])
    centers = pixel_to_world_center(pixels, origin, args.resolution, occupancy.shape)
    roundtrip = world_to_pixel(centers, origin, args.resolution, occupancy.shape)
    if not np.array_equal(roundtrip, pixels):
        raise AssertionError("World/grid center roundtrip failed")
    report = {"schema": "navisaaclab.usd_projection_map.v1", "scene": args.worker,
              "status": "prototype_incomplete_no_free_cells" if incomplete else "prototype_ready_for_physx_review",
              "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_usd": str(usd), "source_sha256": composition["root_sha256"],
              "generator": str(Path(__file__).resolve()), "generator_sha256": sha256(__file__),
              "method": "conservative world XY AABB projection of geometry intersecting one height band",
              "physx_collider_verified": False, "author_eight_scene_map": False,
              "navigation_ready_claim": False, "composition_complete": composition["composition_complete"],
              "missing_geometry": composition["missing_local_layers"], "stage": stage_info,
              "composition_errors": [str(error) for error in stage.GetCompositionErrors()],
              "global_geometry_world_bbox_m": [np.min([r["min_map_m"] for r in geometry], axis=0).tolist(),
                                               np.max([r["max_map_m"] for r in geometry], axis=0).tolist()],
              "extent_mode": args.extent_mode, "floor": floor_info,
              "pedestrian_height_band_above_floor_m": [args.height_min, args.height_max],
              "bounds_xy_m": [origin.tolist(), maximum.tolist()], "image_dimensions_xy": size.tolist(),
              "metadata": metadata, "statistics": statistics, "free_origin_candidate": sample,
              "coordinate_check": {"pixel_centers_world_grid_roundtrip": True,
                                   "image_row0": "max world map Y", "image_column0": "min world map X",
                                   "ros_origin": "bottom-left cell lower corner, map meters, yaw0",
                                   "examples": {"pixels_row_column": pixels.tolist(), "world_centers_xy_m": centers.tolist()}},
              "files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                        for name in ("occupancy.png", "occupancy.yaml", "projection_preview.png", "composition.json")},
              "limitations": ["Authored collision schemas are inventoried, not executed or verified by PhysX.",
                              "Axis-aligned geometry/floor projections can overestimate footprints and floor support; doorways and rotated geometry require collider review.",
                              "Default-time snapshot; animated geometry and different floors require separate maps.",
                              "Missing local geometry leaves all otherwise-free cells unknown; preview is not a navigation map.",
                              "No original Assets or scene configuration is modified; workspace exports are not automatically referenced.",
                              "Standalone USD26.8 composition is not a substitute for Kit USD25.11 runtime validation."],
              "elapsed_s": time.monotonic() - started}
    write_json(output / "provenance.json", report)
    return {"scene": args.worker, "status": report["status"], "composition_complete": not incomplete,
            "dimensions_xy": size.tolist(), "origin": metadata["origin"],
            "occupied_cells": statistics["occupied_cells"], "free_cells": statistics["free_cells"],
            "unknown_cells": statistics["unknown_cells"], "provenance": str(output / "provenance.json")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", help="Explicit composed source override for one selected scene")
    parser.add_argument("--assets", default=str(ROOT.parent / "Assets/DynBench/isaacsim_scene"))
    parser.add_argument("--output", default=str(ROOT / "data/diffusion_amp/maps"))
    parser.add_argument("--scene", action="append", choices=list(SCENES))
    parser.add_argument("--resolution", type=float, default=.1)
    parser.add_argument("--margin", type=float, default=.5)
    parser.add_argument("--native-bounds-quantum-m",type=float, help="Opt-in .5m binary-exact bounds/origin; extra border cells remain unknown")
    parser.add_argument("--height-min", type=float, default=.15)
    parser.add_argument("--height-max", type=float, default=1.8)
    parser.add_argument("--ground-height", type=float)
    parser.add_argument("--extent-mode", choices=("floor-support", "world"), default="floor-support")
    parser.add_argument("--max-cells", type=int, default=16_000_000)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--triangle-projection", action="store_true", help="Conservatively project actual clipped mesh triangles for merged-geometry scenes")
    parser.add_argument("--worker", choices=list(SCENES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.source_usd and not args.worker and (not args.scene or len(args.scene)!=1):
        parser.error("Source override requires exactly one selected scene")
    if args.native_bounds_quantum_m is not None and args.native_bounds_quantum_m != .5:
        parser.error('Pinned native rounding workaround currently admits only explicit .5m quantum')
    numeric = [args.resolution, args.margin, args.height_min, args.height_max, args.timeout]
    if not all(math.isfinite(value) for value in numeric) or args.resolution <= 0 or args.margin < 0:
        parser.error("Resolution must be positive; margin nonnegative; parameters finite")
    if args.height_min < 0 or args.height_max <= args.height_min or args.timeout <= 0 or args.max_cells < 1:
        parser.error("Invalid height band, timeout or cell bound")
    if args.ground_height is not None and not math.isfinite(args.ground_height):
        parser.error("Ground height must be finite")
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT) and not output.is_relative_to(Path(tempfile.gettempdir())):
        parser.error("This offline preparation only writes within the workspace or the system temporary directory")
    return args


def main():
    args = parse_args()
    if args.worker:
        try:
            result = generate_scene(args)
        except Exception as error:
            result = {"scene": args.worker, "status": "blocked", "error": f"{type(error).__name__}: {error}"}
            write_json(Path(args.output) / args.worker / "generation_failure.json", result)
        print(json.dumps(result), flush=True)
        return int(result["status"] == "blocked")
    results = []
    for scene in args.scene or SCENES:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", scene]
        env = dict(os.environ, PXR_WORK_THREAD_LIMIT="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=env)
            log = Path(args.output) / scene / "generation.stderr.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(completed.stderr)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            result = {"scene": scene, "status": "blocked_timeout", "timeout_s": args.timeout,
                      "map_was_not_generated_or_validated": True}
        except (ValueError, IndexError) as error:
            result = {"scene": scene, "status": "blocked", "error": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(Path(args.output) / "generation_manifest.json", {
        "schema": "navisaaclab.usd_projection_batch.v1", "arguments": vars(args), "results": results,
        "physx_verified": False, "author_protocol_reproduced": False})
    return int(any(result["status"].startswith("blocked") for result in results))


if __name__ == "__main__":
    sys.exit(main())
