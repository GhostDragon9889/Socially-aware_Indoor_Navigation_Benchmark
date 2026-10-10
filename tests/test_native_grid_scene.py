"""Mathematical grid fixtures test contracts; no native PhysX execution here."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, FLOOR_PATH, file_record, reference_ground_geometry
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value))
    return file_record(path)


def fixture_provider(folder, labels=None, *, height_blocked_pixels=(), projection_unknown_pixels=()):
    """Persist raw/CSR/clock/export fixture files, not a real native resource."""
    folder.mkdir(exist_ok=True)
    labels = np.full((80, 90), 254, dtype=np.uint8) if labels is None else np.asarray(labels, dtype=np.uint8)
    height, width = labels.shape
    low = np.array([-5., -4., .15])
    high = low + np.array([width * .1, height * .1, 1.65])
    ground = 0.

    def centers(pixels):
        pixels = np.asarray(pixels).reshape(-1, 2)
        return np.column_stack((low[0] + (pixels[:, 1] + .5) * .1,
                                low[1] + (height - pixels[:, 0] - .5) * .1, np.full(len(pixels), .75)))

    source, script, overlay = (folder / name for name in ("source.usda", "probe.py", "overlay.usda"))
    source.write_text("#usda 1.0\n")
    script.write_text("# contract fixture only; no native execution\n")
    overlay.write_text("#usda 1.0\n")
    projection = labels.copy()
    for pixel in projection_unknown_pixels:
        projection[pixel] = 205
    Image.fromarray(projection).save(folder / "projection.png")
    inventory = [{"path": "/Root/Floor", "type_name": "Mesh", "enabled": True, "static": True,
                  "near_ground_flat_surface": True, "geometry": {"contract_fixture_only": True}},
                 {"path": "/Root/Wall", "type_name": "Mesh", "enabled": True, "static": True,
                  "near_ground_flat_surface": False}]
    points, counts, indices = reference_ground_geometry([low[:2].tolist(), high[:2].tolist()], ground)
    recipe = {"path": FLOOR_PATH, "floor_roots": [FLOOR_PATH], "points": points,
        "face_counts": counts, "face_indices": indices, "height_m": ground, "margin_m": .2,
        "floor_owner": "reference_flat_terrain", "holes_steps_and_other_floors_inferred": False,
        "disabled_visual_floor_collision_paths": ["/Root/Floor"],
        "disabled_flat_ground_collision_paths": ["/Root/Floor"], "original_flat_ground_collider_inventory": inventory,
        "material": {"static_friction": 1., "dynamic_friction": 1., "restitution": 0., "combine_mode": "average"},
        "overlay_source": file_record(overlay)}
    recipe_record = write_json(folder / "recipe.json", recipe)
    preflight = {"schema": "dynbench_physx_map_preflight.v1", "scene": "Office", "status": "passed",
        "SDK_started": False, "floor_owner": "reference_flat_terrain", "floor_overlay_mode": "none",
        "up_axis": "Z", "meters_per_unit": 1., "original_composition_unchanged": True,
        "time_sampled_xform_paths": [], "authored_rigid_bodies": [], "source": file_record(source),
        "layers": [file_record(source)], "input_projection": {"occupancy.png": file_record(folder / "projection.png")},
        "reference_ground_recipe": recipe_record, "reference_ground": recipe, "script": file_record(script),
        "projection_metadata": {"bounds": [low[:2].tolist(), high[:2].tolist()], "ground": 0., "dims": [width, height]},
        "exact_floor_collider_paths": [FLOOR_PATH], "original_enabled_floor_collision_paths": ["/Root/Floor"],
        "flat_ground_collider_inventory": inventory,
        "authored_colliders": [{"path": "/Root/Floor", "enabled": True}, {"path": "/Root/Wall", "enabled": True}]}
    preflight_record = write_json(folder / "preflight.json", preflight)
    known_pixels = np.argwhere(labels == 254)
    canonical = centers(known_pixels)
    count = len(canonical)
    support_path = folder / "floor.npz"
    hit_position = canonical.copy()
    hit_position[:, 2] = 0.
    np.savez(support_path, native_free_index=np.arange(count), free_world_positions=canonical,
        supported=np.ones(count, bool), hit_position=hit_position,
        hit_normal=np.tile([0., 0., 1.], (count, 1)), collision=np.full(count, FLOOR_PATH),
        all_hit_count=np.ones(count, dtype=np.int32), all_hit_offsets=np.arange(count + 1),
        all_hit_collision_paths=np.full(count, FLOOR_PATH), all_hit_rigid_body_paths=np.full(count, ""),
        all_hit_positions=hit_position, all_hit_normals=np.tile([0., 0., 1.], (count, 1)),
        all_hit_distances=np.full(count, .3), all_hit_classification=np.full(count, "selected_floor_support"),
        undeclared_near_floor_hit=np.zeros(count, bool), foreign_hit=np.zeros(count, bool))
    raw_buffer = np.where(labels[::-1, ::-1] == 0, 4, np.where(labels[::-1, ::-1] == 254, 5, 6)).astype(np.float32).reshape(-1)
    raw_path = folder / "native_omap.npz"
    np.savez(raw_path, buffer=raw_buffer, occupied_positions=centers(np.argwhere(labels == 0)),
        free_positions=canonical, dimensions=np.array([width, height, 0]), min_bound=low, max_bound=high)
    world_path = folder / "canonical.npz"
    np.savez(world_path, canonical_free_world_positions=canonical)
    blocked = np.array([tuple(pixel) in height_blocked_pixels for pixel in known_pixels], dtype=bool)
    band_path = folder / "band.npz"
    np.savez(band_path, native_free_index=np.arange(count), free_world_positions=canonical,
        full_height_clear=~blocked, authored_obstacle_hit=blocked, all_hit_paths_classified=np.ones(count, bool),
        selected_floor_hit_count=np.zeros(count, dtype=np.int32), hit_offsets=np.r_[0, blocked.cumsum()],
        hit_collision_paths=np.full(int(blocked.sum()), "/Root/Wall"),
        hit_rigid_body_paths=np.full(int(blocked.sum()), ""), hit_classification=np.full(int(blocked.sum()), "authored_obstacle"))
    clock = {"clock_before": {"submitted_physics_step": 1, "timeline_s": 1.},
        "clock_after": {"submitted_physics_step": 1, "timeline_s": 1.},
        "query_time_unchanged": True, "foreign_hit_paths": []}
    clock_record = write_json(folder / "clock.json", clock)
    refined = labels.copy()
    refined[known_pixels[blocked, 0], known_pixels[blocked, 1]] = 0
    exported = refined.copy()
    exported[(projection == 205) & (exported != 0)] = 205
    exports = {}
    for name, values in (("occupancy.png", exported), ("native_reconstructed.png", labels), ("height_band_refined.png", refined)):
        Image.fromarray(values).save(folder / name)
        exports[name] = file_record(folder / name)
    yaml_path = folder / "occupancy.yaml"
    yaml_path.write_text(f"image: occupancy.png\nresolution: 0.1\norigin: [{low[0]},{low[1]},0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    exports["occupancy.yaml"] = file_record(yaml_path)
    native = {"schema": "dynbench_physx_map_probe.v1", "scene": "Office", "source": preflight["source"],
        "preflight": preflight_record, "script": preflight["script"], "status": "scene_resource_gate_passed",
        "scene_resource_gate_passed": True, "native_started": True, "native_probe_completed": True,
        "native_mapping_verified": True, "fallback_used": False, "original_source_layers_unchanged": True,
        "projection_unknown_cells_preserved": True, "floor_collider_paths": [FLOOR_PATH],
        "selected_floor_height_m": 0., "sampled_free_cells_with_overlap": 0, "foreign_native_hit_paths": [],
        "origin_verification": {"overlap_hits": [], "floor_support": {"collision": FLOOR_PATH},
            "floor_ray_all_hits": [{"collision": FLOOR_PATH, "position": [0., 0., 0.], "normal": [0., 0., 1.], "distance": .3}],
            "floor_ownership": {"classification": ["selected_floor_support"], "undeclared_near_floor_collision_paths": [],
                "foreign_ray_collision_paths": [], "all_support_hit_paths_declared": True}},
        "floor_support": {"full_grid": True, "native_free_cells": count, "tested": count, "supported": count,
            "unsupported": 0, "artifact": file_record(support_path)},
        "physics": {"scene_query_support": True, "no_query_time_advance": True,
            "timeline_at_query_s": 1., "timeline_after_queries_s": 1.},
        "dynamic_activity": {"static_snapshot_gate": True}, "exports": exports, "native_omap": file_record(raw_path),
        "actual_bounds_world_m": [low.tolist(), high.tolist()],
        "requested_height_band_world_m": [.15, 1.8],
        "coordinate_validation": {"fixed_native_layout": "row-major x-fastest; row0=minY, column0=maxX",
            "layout_fitted_dynamically": False, "unknown_promoted_to_free": False,
            "positions_roundtrip_verified": True, "canonical_world_positions_artifact": file_record(world_path)},
        "height_band_refinement": {"all_native_known5_cells_tested": True, "native_unknown_never_promoted": True,
            "untested_known5_exported_unknown": 0, "unsupported_tested_cells": 0, "unclassified_cells": 0,
            "cell_xy_size_m": .1, "band_above_selected_floor_m": [.15, 1.8],
            "engineering_refinement": True, "pure_omap2D_full_height_claim": False,
            "known5_source": count, "known5_tested": count, "height_blocked_known5_cells": int(blocked.sum()),
            "height_clear_known5_cells": int((~blocked).sum())},
        "full_height_band_queries": {**clock, "artifact": file_record(band_path),
            "clock_artifact": clock_record, "queried_cells": count},
        "floor_ownership": {"schema": "dynbench.native_floor_owner.v1", "declared_floor_paths": [FLOOR_PATH],
            "actual_enabled_flat_ground_paths": [FLOOR_PATH], "all_support_hit_paths_declared": True,
            "single_declared_floor_owner_verified": True, "floor_ownership_verified": True, "full_grid_floor_rays": True,
            "all_ray_hits_saved": True, "undeclared_near_floor_collision_paths": [], "foreign_ray_collision_paths": [],
            "undeclared_near_floor_hit_rows": 0, "foreign_hit_rows": 0, "height_tolerance_m": .08, "normal_z_min": .9,
            "artifact": file_record(support_path), "actual_enabled_flat_ground_inventory": [
                {"path": FLOOR_PATH, "near_ground_flat_surface": True}, {"path": "/Root/Wall", "near_ground_flat_surface": False}]}}
    native_record = write_json(folder / "native.json", native)
    provider = DynBenchSceneProvider.from_preflight(folder / "preflight.json", expected_sha256=preflight_record["sha256"])
    provider.admit_native_map(folder / "native.json", expected_sha256=native_record["sha256"])
    return provider


def scene(provider, *, clearance=0., origins=None, routes=None, lookahead=5.):
    origins = [[-3.45, -.05]] if origins is None else origins
    routes = [[[-3.45, -.05], [2.55, -.05]]] if routes is None else routes
    return NativeGridScene.from_provider(provider, origins_xy=origins, routes_xy=routes,
        clearance_m=clearance, arrival_radius_m=.15, goal_lookahead_m=lookahead)


def mutate_native(provider, change):
    path = Path(provider._native["record"]["path"])
    data = deepcopy(provider._native["data"])
    change(data)
    record = write_json(path, data)
    provider._native = {"record": record, "data": data}


def test_import_is_lazy_without_torch_or_SDK():
    code = """import sys
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
assert not any(name.split('.')[0] in {'torch','pxr','omni','isaaclab','isaacsim','scipy','numpy'} for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_preflight_without_native_or_height_band_cannot_navigate(tmp_path):
    provider = fixture_provider(tmp_path)
    provider._native = None
    with pytest.raises(RuntimeError, match="native full-grid"):
        scene(provider)
    provider = fixture_provider(tmp_path)
    mutate_native(provider, lambda data: data.pop("full_height_band_queries"))
    with pytest.raises(ValueError, match="incomplete"):
        scene(provider)


@pytest.mark.parametrize("change", ["partial", "layout_fit", "layout_flip", "unknown_promoted", "clock", "CSR", "foreign", "mask"])
def test_admission_rejects_partial_or_contradictory_raw_evidence(tmp_path, change):
    provider = fixture_provider(tmp_path)

    def edit(data):
        if change == "partial":
            data["height_band_refinement"]["all_native_known5_cells_tested"] = False
        elif change == "layout_fit":
            data["coordinate_validation"]["layout_fitted_dynamically"] = True
        elif change == "layout_flip":
            data["coordinate_validation"]["fixed_native_layout"] = "row0=north"
        elif change == "unknown_promoted":
            data["height_band_refinement"]["native_unknown_never_promoted"] = False
        elif change == "clock":
            data["full_height_band_queries"]["clock_after"]["submitted_physics_step"] += 1
        else:
            path = Path(data["full_height_band_queries"]["artifact"]["path"])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            if change == "CSR":
                arrays["hit_offsets"][-1] = 1
            elif change == "foreign":
                arrays["hit_offsets"][1:] = 1
                arrays["hit_collision_paths"] = np.array(["/Root/Foreign"])
                arrays["hit_rigid_body_paths"] = np.array([""])
                arrays["hit_classification"] = np.array(["foreign_unclassified"])
            else:
                arrays["full_height_clear"][0] = False
            np.savez(path, **arrays)
            data["full_height_band_queries"]["artifact"] = file_record(path)
    mutate_native(provider, edit)
    with pytest.raises(ValueError):
        scene(provider)


def test_raw_height_obstacle_becomes_blocked_without_fake_cuboid(tmp_path):
    navigation = scene(fixture_provider(tmp_path, height_blocked_pixels=((40, 40),)))
    point = navigation.pixel_to_world(np.array([40, 40]))
    assert not navigation.is_walkable(point)
    assert navigation.describe()["admission"]["height_blocked_known5_cells"] == 1
    assert not hasattr(navigation, "centers") and not hasattr(navigation, "sizes")
    assert not navigation.describe()["grid_as_PhysX_cuboids"]


def test_projection_unknown_remains_blocked_but_native_obstacle_is_not_erased(tmp_path):
    navigation = scene(fixture_provider(tmp_path, height_blocked_pixels=((40, 40),),
        projection_unknown_pixels=((40, 40), (40, 41))))
    assert navigation._labels[40, 40] == 0 and navigation._labels[40, 41] == 205
    assert not navigation.is_walkable(navigation.pixel_to_world(np.array([[40, 40], [40, 41]]))).any()


def test_negative_world_coordinates_and_fixed_ros_centers_no_clamp(tmp_path):
    navigation = scene(fixture_provider(tmp_path))
    pixels = np.array([[0, 0], [79, 89], [40, 15]])
    points = navigation.pixel_to_world(pixels)
    np.testing.assert_allclose(points, [[-4.95, 3.95], [3.95, -3.95], [-3.45, -.05]], atol=1e-14)
    np.testing.assert_array_equal(navigation.world_to_pixel(points), pixels)
    assert not navigation.is_walkable([4., 0.]) and not navigation.in_domain([4., 0.])
    assert navigation.world_to_pixel([-5.0001, 0.])[1] == -1
    with pytest.raises(ValueError, match="outside"):
        navigation.pixel_to_world(np.array([0, -1]))
    with pytest.raises(ValueError, match="safe indexing"):
        navigation.world_to_pixel([1e200, 0.])


@pytest.mark.parametrize("resolution", ["nan", "inf", "0", "0.2"])
def test_wrong_or_nonfinite_native_resolution_cannot_rebind_loaded_snapshot(tmp_path, resolution):
    provider = fixture_provider(tmp_path)

    def edit(data):
        path = Path(data["exports"]["occupancy.yaml"]["path"])
        path.write_text(path.read_text().replace("resolution: 0.1", f"resolution: {resolution}"))
        data["exports"]["occupancy.yaml"] = file_record(path)
    mutate_native(provider, edit)
    with pytest.raises(ValueError):
        scene(provider)


def test_astar_winds_around_two_alternating_walls_and_all_segments_are_free(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[:56, 40] = 0
    labels[25:, 60] = 0
    navigation = scene(fixture_provider(tmp_path, labels))
    start, goal = np.array([-3.45, -.05]), np.array([2.55, -.05])
    path = navigation.astar_path(start, goal)
    assert path is not None and np.array_equal(path[0], start) and np.array_equal(path[-1], goal)
    pixels = navigation.world_to_pixel(path)
    assert pixels[:, 0].max() >= 57 and pixels[:, 0].min() <= 23
    assert np.linalg.norm(np.diff(path, axis=0), axis=-1).sum() > np.linalg.norm(goal - start) * 1.5
    assert navigation.is_walkable(path).all()
    # Dense subcell samples test continuous segments, not only A* vertices.
    dense = np.concatenate([a[None] + np.linspace(0, 1, 11)[:, None] * (b - a)[None] for a, b in zip(path[:-1], path[1:])])
    assert navigation.is_walkable(dense).all()


def test_unknown_barrier_blocks_astar_and_failure_holds_actual_position(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 205
    navigation = scene(fixture_provider(tmp_path, labels))
    position = torch.tensor([[-3.45, -.05]])
    points, indices, reached = navigation.route_waypoints(position, torch.tensor([1]), advance=False)
    assert torch.equal(points, position[:, None].repeat(1, 10, 1))
    assert indices.tolist() == [1] and reached.tolist() == [False]
    assert navigation.global_path_failures == 1
    assert navigation.astar_path([-3.45, -.05], [2.55, -.05]) is None


def test_route_vertex_across_thin_wall_is_not_a_ghost_arrival(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 0
    navigation = NativeGridScene.from_provider(fixture_provider(tmp_path, labels),
        origins_xy=[[-1.25, -.05]], routes_xy=[[[-.65, -.05], [-1.25, -.05]]], clearance_m=0., arrival_radius_m=.75)
    updated, reached = navigation.advance_route([[-1.25, -.05]], [0])
    assert updated.tolist() == [0] and reached.tolist() == [False]


def test_clearance_blocks_narrow_passage_and_boundary_cells(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[30:50, :41] = 0
    labels[30:50, 48:] = 0  # Only seven .1m cells between physical wall cells.
    provider = fixture_provider(tmp_path, labels)
    origins = [[-3.45, 2.55]]
    routes = [[[-3.45, 2.55], [2.55, -2.55]]]
    loose = scene(provider, clearance=.1, origins=origins, routes=routes)
    strict = scene(provider, clearance=.35, origins=origins, routes=routes)
    middle = strict.pixel_to_world(np.array([40, 44]))
    assert loose.is_walkable(middle) and not strict.is_walkable(middle)
    assert not strict.is_walkable(strict.pixel_to_world(np.array([0, 20])))
    assert strict.astar_path(origins[0], routes[0][1]) is None


def test_nearest_spawn_is_explicit_bounded_and_deterministic(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[35:45, 35:45] = 205
    navigation = scene(fixture_provider(tmp_path, labels))
    hint = navigation.pixel_to_world(np.array([40, 40]))
    chosen = navigation.nearest_spawn(hint, max_distance_m=2.)
    assert navigation.is_walkable(chosen) and np.linalg.norm(chosen - hint) <= 2.
    assert np.array_equal(chosen, navigation.nearest_spawn(hint, max_distance_m=2.))
    with pytest.raises(ValueError, match="within explicit"):
        navigation.nearest_spawn(hint, max_distance_m=.01)


def test_route_subset_progress_endpoint_and_timed_path_consistency(tmp_path):
    origins = [[-3.45, -.05], [2.55, 2.55]]
    routes = [[[-3.45, -.05], [2.55, -.05]], [[2.55, 2.55], [-3.45, 2.55]]]
    navigation = scene(fixture_provider(tmp_path), origins=origins, routes=routes)
    pos = torch.tensor([[2.55, 2.55]])
    advanced, reached = navigation.advance_route(pos, torch.tensor([0]), actor_ids=torch.tensor([1]))
    assert advanced.tolist() == [1] and reached.tolist() == [True]
    points, indices, found = navigation.route_waypoints(pos, advanced, actor_ids=torch.tensor([1]), advance=False)
    endpoint, endpoint_indices, endpoint_found = navigation.route_goal(pos, advanced, actor_ids=torch.tensor([1]), advance=False)
    assert torch.equal(endpoint, points[:, -1])
    assert torch.equal(indices, endpoint_indices) and torch.equal(found, endpoint_found)
    assert not found.any() and not torch.equal(points[:, 0], points[:, -1])
    empty, _, _ = navigation.route_waypoints(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert empty.shape == (0, 10, 2)
    for bad in ([True], [0., 0.], [1., .5], [6.], [float("nan")]):
        with pytest.raises(ValueError):
            navigation.route_waypoints(pos, advanced, actor_ids=[1], times=bad)


def test_grid_SDF_true_gradient_reduces_soft_cost_and_exterior_points_are_not_clipped(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 0
    navigation = scene(fixture_provider(tmp_path, labels))
    xy = torch.tensor([[[[-.99, -.05]]]], dtype=torch.float64, requires_grad=True)
    cost = navigation.collision_cost(xy, radius=.35)
    gradient, = torch.autograd.grad(cost.sum(), xy)
    assert gradient[..., 0].item() > .1 and torch.isfinite(gradient).all()
    with torch.no_grad():
        improved = navigation.collision_cost(xy - .01 * gradient, radius=.35)
        assert improved.item() < cost.item()
        epsilon = 1e-5
        delta = xy.new_tensor([epsilon, 0.])
        numerical = (navigation.collision_cost(xy + delta, .35) - navigation.collision_cost(xy - delta, .35)) / (2 * epsilon)
        assert numerical.item() == pytest.approx(gradient[..., 0].item(), rel=1e-6)
    exterior = torch.tensor([[[[4.2, 0.]]]], requires_grad=True)
    snapshot = exterior.detach().clone()
    outside_cost = navigation.collision_cost(exterior, .35)
    outside_gradient, = torch.autograd.grad(outside_cost.sum(), exterior)
    assert outside_gradient[..., 0].item() == pytest.approx(1.)
    assert torch.equal(exterior.detach(), snapshot)
    assert outside_cost.item() == pytest.approx(.55)
    with pytest.raises(ValueError, match="finite FP32"):
        navigation.signed_distance(torch.tensor([[float("nan"), 0.]]))


def test_grid_guidance_mean_has_batch_invariance_and_unknown_is_penalized(tmp_path):
    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 205
    navigation = scene(fixture_provider(tmp_path, labels))
    one = torch.tensor([[[[-.99, -.05], [-.99, .15]]]], requires_grad=True)
    many = one.detach().repeat(3, 2, 1, 1).requires_grad_()
    first = navigation.collision_cost(one, .35)
    repeated = navigation.collision_cost(many, .35)
    assert torch.equal(repeated, first.repeat(3, 2))
    grad_one, = torch.autograd.grad(first.sum(), one)
    grad_many, = torch.autograd.grad(repeated.sum(), many)
    assert torch.equal(grad_many, grad_one.repeat(3, 2, 1, 1))
    assert navigation.signed_distance(one.detach()).min() < 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_readonly_four_corner_SDF_matches_grid_sample_values_and_world_Jacobian(tmp_path, dtype, monkeypatch):
    """Independent old operator oracle; the production path cannot call it."""
    from torch.nn import functional

    labels = np.full((80, 90), 254, np.uint8)
    labels[:, 40] = 0
    labels[20:25, 10:20] = 205
    navigation = scene(fixture_provider(tmp_path, labels))
    generator = torch.Generator().manual_seed(19)
    points = torch.rand((2, 3, 43, 2), generator=generator, dtype=dtype)
    bounds = torch.tensor(navigation._bounds, dtype=dtype)
    points = points * (bounds[2:] - bounds[:2] + .4) + bounds[:2] - .2
    # Exact source cell centers/edges plus repeated outside coordinates.
    points[0, 0, :4] = torch.tensor(np.asarray([
        navigation.pixel_to_world(np.array([0, 0])),
        navigation.pixel_to_world(np.array([79, 89])),
        [bounds[0] - .2, bounds[1] - .2],
        [bounds[2] + .2, bounds[3] + .2],
    ]), dtype=dtype)
    points.requires_grad_()
    origin = points.new_tensor(navigation._origin)
    cell = (points - origin) / .1 - .5
    height, width = labels.shape
    grid = torch.stack((2 * (cell[..., 0] + .5) / width - 1,
                        1 - 2 * (cell[..., 1] + .5) / height), -1)
    table = torch.tensor(navigation._signed_grid, dtype=dtype)[None, None]
    expected = functional.grid_sample(table, grid.reshape(1, 1, -1, 2),
        mode="bilinear", padding_mode="border", align_corners=False).reshape(points.shape[:-1])
    q = (points - (bounds[:2] + bounds[2:]) / 2).abs() - (bounds[2:] - bounds[:2]) / 2
    domain = -(torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0))
    expected = torch.minimum(expected, domain)
    expected_gradient, = torch.autograd.grad(expected.sum(), points)

    def forbidden_operator(*args, **kwargs):
        raise AssertionError("Production SDF must not use strict-CUDA-unsupported grid_sample backward")

    monkeypatch.setattr(functional, "grid_sample", forbidden_operator)
    actual = navigation.signed_distance(points)
    actual_gradient, = torch.autograd.grad(actual.sum(), points)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
    assert all(not table.requires_grad for table in navigation._sdf_tensors.values())


def test_raster_delegates_measured_labels_and_never_returns_fake_height(tmp_path):
    navigation = scene(fixture_provider(tmp_path))
    states = torch.tensor([[-3.45, -.05, 0., 0., 0.]])
    assert torch.equal(navigation.raster(states), navigation.provider.raster(states.numpy()))
    assert navigation.raster(states).shape == (1, 2, 224, 224)
    with pytest.raises(RuntimeError, match="explicit actual native"):
        navigation.height_at([[-3.45, -.05]])
    assert navigation.describe()["status"] == "prepared_not_runtime_integrated"
