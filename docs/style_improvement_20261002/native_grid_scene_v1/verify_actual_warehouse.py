"""Read actual v5 native artifacts on CPU; no new SDK cooking/navigation run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    readiness_path = ROOT / "docs/style_improvement_20261002/paper_scene_maps_probe_readiness_v5.json"
    readiness = json.loads(readiness_path.read_text())
    archive = readiness["archive"]
    assert file_record(archive["path"]) == archive
    script = next(record for record in readiness["source_files"] if record["path"].endswith("scripts/probe_paper_scene_maps.py"))
    with tarfile.open(archive["path"]) as source:
        members = [member for member in source.getmembers() if member.isfile() and member.name.endswith("scripts/probe_paper_scene_maps.py")]
        assert len(members) == 1
        content = source.extractfile(members[0]).read()
    assert len(content) == script["bytes"] and hashlib.sha256(content).hexdigest() == script["sha256"]
    snapshot = OUT / "probe_source_v5.py"
    if snapshot.exists():
        assert snapshot.read_bytes() == content
    else:
        snapshot.write_bytes(content)
    result_path = ROOT / "output/scene_maps/native_reference_warehouse_floor_owner_v5/Warehouse/result.json"
    result_record = file_record(result_path)
    result = json.loads(result_path.read_text())
    assert result["script"] == script and result["floor_ownership"]["single_declared_floor_owner_verified"] is True
    provider = DynBenchSceneProvider.from_preflight(result["preflight"]["path"],
        expected_sha256=result["preflight"]["sha256"], probe_script_snapshot=snapshot)
    provider.admit_native_map(result_path, expected_sha256=result_record["sha256"])
    assert provider.floor_roots == ("/__PaperMapProbeReferenceGround",)
    assert "/Root/GroundPlane/CollisionPlane" in provider.describe()["disabled_original_flat_ground_paths"]
    assert "/Root/GroundPlane/CollisionPlane" not in provider.static_collision_paths
    free = provider._map == 254
    radius = .35
    distance = distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1] * .1
    walkable = free & (distance - np.sqrt(2) * .1 >= radius)
    components, count = label(walkable)
    sizes = np.bincount(components.reshape(-1))[1:]
    assert count and sizes.max() > 1
    region = np.argmax(sizes) + 1
    pixels = np.argwhere(components == region)
    origin = np.asarray(provider._map_origin_xy)
    xy = origin + np.column_stack((pixels[:, 1] + .5, provider._map.shape[0] - pixels[:, 0] - .5)) * .1
    start = xy[0]
    end = xy[np.argmax(np.linalg.norm(xy - start, axis=-1))]
    # Declared smoke-test points within the largest four-connected conservative
    # free region; these are not physically spawned actors or paper scene routes.
    navigation = NativeGridScene.from_provider(provider, origins_xy=[start.tolist()],
        routes_xy=[[start.tolist(), end.tolist()]], clearance_m=radius, arrival_radius_m=.75)
    path = navigation.astar_path(start, end)
    assert path is not None and navigation.is_walkable(path).all()
    dense = np.concatenate([a[None] + np.linspace(0, 1, 11)[:, None] * (b - a)[None] for a, b in zip(path[:-1], path[1:])])
    assert navigation.is_walkable(dense).all()
    assert np.array_equal(navigation.world_to_pixel(navigation.pixel_to_world(pixels)), pixels)
    assert not any(name.split(".")[0] in {"torch", "omni", "isaaclab", "isaacsim", "pxr"} for name in sys.modules)
    import torch
    assert not torch.cuda.is_initialized()
    points, _, reached = navigation.route_waypoints([start], [1], advance=False)
    assert points.shape == (1, 10, 2) and torch.isfinite(points).all() and not reached.any()
    # Boundary sample is deterministic and actual-grid-derived, not an oracle
    # future trajectory or an actual physical contact/clearance measurement.
    adjacency = free[:, :-1] & ~free[:, 1:]
    pair = np.argwhere(adjacency)[0]
    centers = navigation.pixel_to_world(np.array([pair, pair + [0, 1]]))
    test_xy = torch.tensor([[[(.25 * centers[0] + .75 * centers[1]).tolist()]]], dtype=torch.float64, requires_grad=True)
    cost = navigation.collision_cost(test_xy, radius=.5)
    gradient, = torch.autograd.grad(cost.sum(), test_xy)
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"omni", "isaaclab", "isaacsim", "pxr"} for name in sys.modules)
    report = {"schema": "dynbench_actual_native_grid_CPU_admission.v1", "status": "passed_CPU_artifact_admission_and_planning",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False, "SDK_started": False,
        "native_probe_source_epoch": script, "source_archive": archive, "verified_script_copy": file_record(snapshot),
        "current_probe_script": file_record(ROOT / "scripts/probe_paper_scene_maps.py"),
        "native_result": result_record, "preflight": result["preflight"], "source_USD": result["source"],
        "native_omap": result["native_omap"], "full_height_artifact": result["full_height_band_queries"]["artifact"],
        "floor_ray_all_hit_artifact": result["floor_support"]["artifact"],
        "floor_owner": result["floor_ownership"], "companion": navigation.describe(),
        "native_map_labels": {str(value): int((provider._map == value).sum()) for value in (0, 205, 254)},
        "conservative_walkable_cells": int(walkable.sum()), "four_connected_components": int(count),
        "CPU_smoke_route": {"selection": "lexicographic first and farthest cell in largest four-connected conservative free region; not paper route",
            "start_xy_m": start.tolist(), "goal_xy_m": end.tolist(), "vertices": len(path),
            "length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()),
            "all_vertices_and_dense_segment_samples_walkable": True, "world_pixel_roundtrip_all_region_cells": True,
            "physical_actor_spawned_or_moved": False},
        "grid_SDF_CPU_gradient": {"world_xy_m": test_xy.detach().reshape(2).tolist(), "radius_m": .5,
            "cost_m": float(cost.detach().item()), "gradient": gradient.reshape(2).tolist(),
            "finite_nonzero": True, "physical_contact_or_body_clearance_claim": False},
        "new_provider_physical_execution_verified": False, "runtime_hook_integrated": False,
        "human_navigation_quality_validated": False, "weights_or_datasets_changed": False,
        "limitations": ["Original v5 GPU probe establishes resource snapshot; this reader does not start new physics.",
            "No full-body swept-volume, new provider cooking, controller stability, or DynBench navigation outcome measured.",
            "Grid SDF is a differentiable guidance proxy; height_at is a clipped policy feature, not true obstacle top.",
            "Source-grid routes are CPU smoke-test choices and do not reproduce unpublished author routes."]}
    (OUT / "actual_warehouse_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "native_known5_cells": report["companion"]["admission"]["canonical_known5_cells"],
        "conservative_walkable_cells": int(walkable.sum()), "CPU_path_vertices": len(path),
        "grid_SDF_gradient_finite_nonzero": True, "new_physics_or_human_navigation_verified": False}))


if __name__ == "__main__":
    main()
