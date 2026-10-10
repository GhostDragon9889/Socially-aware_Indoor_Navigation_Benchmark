"""Deterministic CPU preparation of explicit, source-bound native routes.

This module authors a local navigation protocol; it never creates SDK objects,
alters native map labels, or claims pedestrian physics/navigation success.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import hashlib
import json
import math
from pathlib import Path

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
from .native_deployment import SCHEMA, preflight_native_scene


def route_shifts(actors, shifts=None):
    """Different anchor offsets; return-to-origin is legal only at the end."""
    if type(actors) is not int or not 3 <= actors <= 15:
        raise ValueError("Maximin scene preparation supports3..15 actors; fewer require a custom nondegenerate route")
    if shifts is None:
        shifts = [1, 2, 3] if actors == 3 else [1, actors // 4 + 1, actors // 2 + 1, 3 * actors // 4 + 1]
    if (type(shifts) not in (list, tuple) or not 2 <= len(shifts) <= actors
            or any(type(x) is not int or not 0 <= x <= actors for x in shifts)):
        raise ValueError("Route shifts require2..N integer offsets in0..N")
    offsets = [x % actors for x in shifts]
    if len(set(offsets)) != len(offsets) or 0 in offsets[:-1]:
        raise ValueError("Route shifts must be unique moduloN; zero is permitted only at the final vertex")
    return list(shifts)


def _clearance(value, label, minimum):
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be finite and at least{minimum}m")
    return float(value)


def select_origins(labels, origin_xy, *, actors, clearance_m=.8, spawn_clearance_m=3.,
                   minimum_spawn_separation_m=1.6, resolution_m=.1):
    """Largest four-neighbour component, deepest first, then first maximin."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label

    route_shifts(actors)
    clearance_m = _clearance(clearance_m, "Route clearance", .8)
    spawn_clearance_m = _clearance(spawn_clearance_m, "Spawn clearance", max(3., clearance_m))
    minimum_spawn_separation_m = _clearance(minimum_spawn_separation_m, "Root spawn separation", 1.6)
    labels, origin = np.asarray(labels), np.asarray(origin_xy)
    if (labels.ndim != 2 or not labels.size or labels.dtype != np.uint8
            or not np.isin(labels, [0, 205, 254]).all() or origin.shape != (2,)
            or origin.dtype.kind not in "fiu" or not np.isfinite(origin).all() or resolution_m != .1):
        raise ValueError("Expected actual uint8 native labels0/205/254, finite world origin, and.1m cells")
    known_free = labels == 254
    # The padded exterior is blocked. Subtract both full-cell half diagonals,
    # matching NativeGridScene rather than claiming clearance at cell centers.
    distance = distance_transform_edt(np.pad(known_free, 1, constant_values=False))[1:-1, 1:-1] * resolution_m
    conservative = distance - math.sqrt(2) * resolution_m
    walkable = known_free & (conservative >= clearance_m)
    components, count = label(walkable, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    if count == 0:
        raise ValueError("No conservative known-free navigation component")
    sizes = np.bincount(components.ravel())[1:]
    component_id = int(np.argmax(sizes)) + 1  # First ROS-row-major component wins a size tie.
    candidates = np.argwhere((components == component_id) & (conservative >= spawn_clearance_m))
    if len(candidates) < actors:
        raise ValueError(f"Largest component has only{len(candidates)} eligible spawn cells for{actors} actors")
    grid_xy = np.stack((candidates[:, 1], labels.shape[0] - 1 - candidates[:, 0]), -1)
    world = origin.astype(np.float64) + (grid_xy + .5) * resolution_m
    selected = [int(np.argmax(conservative[candidates[:, 0], candidates[:, 1]]))]
    nearest_squared = np.sum((world - world[selected[0]]) ** 2, axis=-1)
    nearest_squared[selected[0]] = -1.
    while len(selected) < actors:
        index = int(np.argmax(nearest_squared))  # Candidates use deterministic ROS row-major order.
        selected.append(index)
        nearest_squared = np.minimum(nearest_squared, np.sum((world - world[index]) ** 2, axis=-1))
        nearest_squared[selected] = -1.
    cells, origins = candidates[selected], world[selected]
    pairwise = np.linalg.norm(origins[:, None] - origins[None, :], axis=-1)
    minimum_separation = float(pairwise[np.triu_indices(actors, 1)].min())
    if minimum_separation < minimum_spawn_separation_m:
        raise ValueError(f"Maximin root spawn separation{minimum_separation:.6g}m is below the declared"
                         f" minimum{minimum_spawn_separation_m:.6g}m; no fallback/resampling")
    return {"origins_xy": origins, "pixels": cells, "walkable": walkable,
            "component_mask": components == component_id,
            "selection": {"algorithm": "largest4-neighbour; deepest first; first row-major maximin argmax",
                          "component_label": component_id, "component_cells": int(sizes[component_id - 1]),
                          "component_count": count, "eligible_spawn_cells": len(candidates),
                          "route_clearance_m": clearance_m, "minimum_required_spawn_clearance_m": spawn_clearance_m,
                          "minimum_required_spawn_separation_m": minimum_spawn_separation_m,
                          "selected_pixels_ROS_row_column": cells.tolist(),
                          "selected_spawn_clearance_m": conservative[cells[:, 0], cells[:, 1]].tolist(),
                          "minimum_spawn_separation_m": minimum_separation,
                          "root_separation_is_not_body_collision_proof": True,
                          "unknown_and_exterior_blocked": True,
                          "tie_breaking": "first NumPy argmax in ROS row-major candidates; no RNG"}}


def verify_route_paths(grid, origins, routes):
    """Require actual A* for each initial segment and all cycle/wrap edges."""
    import numpy as np
    origins, routes = np.asarray(origins), np.asarray(routes)
    paths = []
    for actor, (origin, route) in enumerate(zip(origins, routes)):
        if not grid.is_walkable(np.concatenate((origin[None], route))).all():
            raise ValueError("Origins/routes must remain conservative known-free")
        for segment, (start, end) in enumerate(zip([origin, *route], [*route, route[0]])):
            path = grid.astar_path(start, end)
            if path is None or len(path) < 2 or not grid.is_walkable(path).all():
                raise ValueError(f"Actor{actor} has a disconnected/degenerate route segment{segment}")
            paths.append({"actor": actor, "segment": segment,
                          "kind": "initial" if segment == 0 else "wrap" if segment == len(route) else "route",
                          "from_xy": start.tolist(), "to_xy": end.tolist(), "vertices": len(path),
                          "length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()),
                          "float64_C_order_sha256": hashlib.sha256(path.tobytes(order="C")).hexdigest()})
    return paths


def prepare_configuration(preflight_path, native_map_path, *, actors, probe_script_snapshot=None,
                          clearance_m=.8, spawn_clearance_m=3., minimum_spawn_separation_m=1.6, shifts=None):
    """Admit real resource arrays before using them; return a cold config/proof."""
    import numpy as np
    from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene

    offsets = route_shifts(actors, shifts)
    preflight, native = file_record(preflight_path), file_record(native_map_path)
    provider = DynBenchSceneProvider.from_preflight(preflight["path"], expected_sha256=preflight["sha256"],
        probe_script_snapshot=probe_script_snapshot)
    provider.admit_native_map(native["path"], expected_sha256=native["sha256"])
    if provider._native["data"].get("CPU_contract_fixture_only") is True:
        raise ValueError("Synthetic contract fixtures cannot be exported as actual native scene resources")
    selection = select_origins(provider._map, provider._map_origin_xy, actors=actors,
                               clearance_m=clearance_m, spawn_clearance_m=spawn_clearance_m,
                               minimum_spawn_separation_m=minimum_spawn_separation_m)
    origins = selection["origins_xy"]
    routes = origins[(np.arange(actors)[:, None] + np.asarray(offsets)[None]) % actors]
    grid = NativeGridScene.from_provider(provider, origins_xy=origins, routes_xy=routes,
                                        clearance_m=clearance_m, device="cpu")
    if not np.array_equal(grid._walkable, selection["walkable"]):
        raise ValueError("Preparation clearance differs from admitted native runtime clearance")
    paths = verify_route_paths(grid, origins, routes)
    def bound(record):
        return {key: record[key] for key in ("path", "sha256")}
    configuration = {"schema": SCHEMA, "scene": provider.scene_name,
                     "preflight": bound(preflight), "native_map": bound(native),
                     "clearance_m": float(clearance_m), "origins_xy": origins.tolist(), "routes_xy": routes.tolist()}
    if probe_script_snapshot is not None:
        configuration["probe_script_snapshot"] = bound(file_record(probe_script_snapshot))
    proof = {"schema": "native_pedestrian_scene_preparation_CPU_proof.v1", "CPU_only": True,
             "SDK_started": False, "GPU_used": False, "scene": provider.scene_name,
             "preflight": preflight, "native_map": native,
             "probe_script_snapshot": None if probe_script_snapshot is None else file_record(probe_script_snapshot),
             "admitted_native_grid": grid.describe(), "actors": actors, "route_shifts": offsets,
             "route_vertex_protocol": "three-anchor local protocol" if len(offsets) == 3 else f"{len(offsets)}-anchor local protocol",
             "selection": selection["selection"], "origins_xy": origins.tolist(), "routes_xy": routes.tolist(),
             "all_initial_route_and_wrap_Astar_passed": True, "route_paths": paths,
             "physical_navigation_validated": False, "paper_eight_scene_protocol_claim": False,
             "source_layers_unchanged": True, "source_layers": provider._data["layers"],
             "preparer": file_record(__file__)}
    return configuration, proof


def publish_configuration(config, proof, output_config, output_proof):
    """Exclusive new files; validate the exact cold schema before proof output."""
    output_config, output_proof = Path(output_config).resolve(), Path(output_proof).resolve()
    if output_config == output_proof or output_config.exists() or output_proof.exists():
        raise ValueError("Configuration/proof destinations must be different new paths; no overwrite")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_proof.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    request = preflight_native_scene(output_config, actors=len(config["origins_xy"]))
    proof = {**proof, "configuration": file_record(output_config), "cold_schema_preflight": request,
             "entrypoint": file_record(repository_root(__file__) / "scripts/prepare_native_pedestrian_scene.py")}
    with output_proof.open("x") as stream:
        stream.write(json.dumps(proof, indent=2) + "\n")
    return proof
