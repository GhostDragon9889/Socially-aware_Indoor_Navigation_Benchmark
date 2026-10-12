"""Prepared provider contracts; synthetic native JSON is never a PhysX proof."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from CrowdSim.humanoid.runtime.dynbench_scene import (
    DynBenchSceneProvider, FLOOR_PATH, _active_spawn_stage, file_record, reference_ground_geometry,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value))
    return file_record(path)


@pytest.fixture
def evidence(tmp_path):
    """Tiny mathematical contract fixture, not a generated native scene map."""
    source = tmp_path / "source.usda"
    source.write_text('#usda 1.0\ndef Xform "Root" {}\n')
    script = tmp_path / "probe.py"
    script.write_text("# contract fixture only\n")
    overlay = tmp_path / "overlay.usda"
    overlay.write_text("#usda 1.0\n")
    projection = tmp_path / "projection.png"
    image = np.array([[254, 205, 0], [254, 0, 0], [0, 0, 0]], dtype=np.uint8)
    Image.fromarray(image).save(projection)
    recipe_file = tmp_path / "recipe.json"
    bounds = [[-.2, -.3], [.1, 0.]]
    points, counts, indices = reference_ground_geometry(bounds, 0.)
    inventory = [{"path": "/Root/Floor", "type_name": "Mesh", "enabled": True, "static": True,
                  "near_ground_flat_surface": True, "geometry": {"contract_fixture_only": True}},
                 {"path": "/Root/Wall", "type_name": "Mesh", "enabled": True, "static": True,
                  "near_ground_flat_surface": False}]
    recipe = {"path": FLOOR_PATH, "floor_roots": [FLOOR_PATH], "points": points,
        "face_counts": counts, "face_indices": indices, "height_m": 0., "margin_m": .2,
        "floor_owner": "reference_flat_terrain", "holes_steps_and_other_floors_inferred": False,
        "disabled_visual_floor_collision_paths": ["/Root/Floor"],
        "disabled_flat_ground_collision_paths": ["/Root/Floor"], "original_flat_ground_collider_inventory": inventory,
        "material": {"static_friction": 1., "dynamic_friction": 1., "restitution": 0., "combine_mode": "average"},
        "overlay_source": file_record(overlay)}
    recipe_record = write_json(recipe_file, recipe)
    preflight = {"schema": "dynbench_physx_map_preflight.v1", "scene": "Office", "status": "passed",
        "SDK_started": False, "floor_owner": "reference_flat_terrain", "floor_overlay_mode": "none",
        "up_axis": "Z", "meters_per_unit": 1., "original_composition_unchanged": True,
        "time_sampled_xform_paths": [], "authored_rigid_bodies": [], "source": file_record(source),
        "layers": [file_record(source)], "input_projection": {"occupancy.png": file_record(projection)},
        "reference_ground_recipe": recipe_record, "reference_ground": recipe, "script": file_record(script),
        "projection_metadata": {"bounds": bounds, "ground": 0., "dims": [3, 3]},
        "exact_floor_collider_paths": [FLOOR_PATH], "original_enabled_floor_collision_paths": ["/Root/Floor"],
        "flat_ground_collider_inventory": inventory,
        "authored_colliders": [{"path": "/Root/Floor", "enabled": True}, {"path": "/Root/Wall", "enabled": True}]}
    preflight_path = tmp_path / "preflight.json"
    preflight_record = write_json(preflight_path, preflight)
    native_png = tmp_path / "occupancy.png"
    Image.fromarray(image).save(native_png)
    native_yaml = tmp_path / "occupancy.yaml"
    native_yaml.write_text("image: occupancy.png\nresolution: 0.1\norigin: [-0.2,-0.3,0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    rays = tmp_path / "rays.npz"
    np.savez(rays, native_free_index=np.arange(2), supported=np.ones(2, dtype=bool),
        hit_position=np.array([[-.15, -.05, 0], [-.15, -.15, 0]]), hit_normal=np.array([[0., 0., 1.]] * 2),
        free_world_positions=np.array([[-.15, -.05, .75], [-.15, -.15, .75]]),
        all_hit_count=np.ones(2, dtype=np.int32), collision=np.array([FLOOR_PATH] * 2),
        all_hit_offsets=np.arange(3), all_hit_collision_paths=np.array([FLOOR_PATH] * 2), all_hit_rigid_body_paths=np.array([""] * 2),
        all_hit_positions=np.array([[-.15, -.05, 0], [-.15, -.15, 0]]), all_hit_normals=np.array([[0., 0., 1.]] * 2),
        all_hit_distances=np.array([.3, .3]), all_hit_classification=np.array(["selected_floor_support"] * 2),
        undeclared_near_floor_hit=np.zeros(2, bool), foreign_hit=np.zeros(2, bool))
    omap = tmp_path / "omap.npz"
    np.savez(omap, contract_fixture_only=np.array([True]))
    band = tmp_path / "band.npz"
    np.savez(band, native_free_index=np.arange(2), free_world_positions=np.array([[-.15, -.05, .75], [-.15, -.15, .75]]),
        full_height_clear=np.ones(2, bool), authored_obstacle_hit=np.zeros(2, bool), all_hit_paths_classified=np.ones(2, bool),
        selected_floor_hit_count=np.zeros(2, dtype=np.int64), hit_offsets=np.zeros(3, dtype=np.int64),
        hit_collision_paths=np.array([], dtype=str), hit_rigid_body_paths=np.array([], dtype=str), hit_classification=np.array([], dtype=str))
    clock = {"clock_before": {"submitted_physics_step": 1, "timeline_s": 1.},
             "clock_after": {"submitted_physics_step": 1, "timeline_s": 1.}, "query_time_unchanged": True, "foreign_hit_paths": []}
    clock_record = write_json(tmp_path / "clock.json", clock)
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
        "floor_support": {"full_grid": True, "native_free_cells": 2, "tested": 2, "supported": 2, "unsupported": 0,
                          "artifact": file_record(rays)},
        "physics": {"scene_query_support": True, "no_query_time_advance": True,
                    "timeline_at_query_s": 1., "timeline_after_queries_s": 1.},
        "dynamic_activity": {"static_snapshot_gate": True},
        "exports": {"occupancy.yaml": file_record(native_yaml), "occupancy.png": file_record(native_png)},
        "native_omap": file_record(omap), "actual_bounds_world_m": [[-.2, -.3, .15], [.1, 0., 1.8]],
        "height_band_refinement": {"all_native_known5_cells_tested": True, "known5_source": 2, "known5_tested": 2,
            "cell_xy_size_m": .1, "band_above_selected_floor_m": [.15, 1.8], "untested_known5_exported_unknown": 0,
            "unclassified_cells": 0, "unsupported_tested_cells": 0, "native_unknown_never_promoted": True,
            "height_blocked_known5_cells": 0, "height_clear_known5_cells": 2},
        "full_height_band_queries": {**clock, "artifact": file_record(band), "clock_artifact": clock_record, "queried_cells": 2},
        "floor_ownership": {"schema": "dynbench.native_floor_owner.v1", "declared_floor_paths": [FLOOR_PATH],
            "actual_enabled_flat_ground_paths": [FLOOR_PATH], "all_support_hit_paths_declared": True,
            "single_declared_floor_owner_verified": True, "floor_ownership_verified": True, "full_grid_floor_rays": True,
            "all_ray_hits_saved": True, "undeclared_near_floor_collision_paths": [], "foreign_ray_collision_paths": [],
            "undeclared_near_floor_hit_rows": 0, "foreign_hit_rows": 0, "height_tolerance_m": .08, "normal_z_min": .9,
            "artifact": file_record(rays), "actual_enabled_flat_ground_inventory": [
                {"path": FLOOR_PATH, "near_ground_flat_surface": True}, {"path": "/Root/Wall", "near_ground_flat_surface": False}]}}
    native_path = tmp_path / "native_fixture.json"
    native_record = write_json(native_path, native)
    return SimpleNamespace(preflight=preflight, preflight_path=preflight_path, preflight_record=preflight_record,
        native=native, native_path=native_path, native_record=native_record, image=image, tmp=tmp_path)


def provider(fixture):
    return DynBenchSceneProvider.from_preflight(fixture.preflight_path,
                                              expected_sha256=fixture.preflight_record["sha256"])


def admitted_fixture(fixture):
    result = provider(fixture)
    result.admit_native_map(fixture.native_path, expected_sha256=fixture.native_record["sha256"])
    return result


def test_import_and_geometry_do_not_bootstrap_tensor_or_sdk():
    code = """import sys
from CrowdSim.humanoid.runtime.dynbench_scene import reference_ground_geometry
assert len(reference_ground_geometry([[-2,-3],[2,3]],0)[0]) == 4
assert not any(name.split('.')[0] in {'torch','numpy','pxr','omni','isaaclab','isaacsim'} for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_reference_floor_is_finite_positive_winding_with_negative_world_coordinates():
    points, counts, indices = reference_ground_geometry([[-4., -3.], [1., 2.]], -.25)
    assert points == [[-4.2, -3.2, -.25], [1.2, -3.2, -.25], [1.2, 2.2, -.25], [-4.2, 2.2, -.25]]
    assert counts == [3, 3] and indices == [0, 1, 2, 0, 2, 3]
    p = np.asarray(points)
    assert np.cross(p[1] - p[0], p[2] - p[0])[2] > 0
    assert np.cross(p[2] - p[0], p[3] - p[0])[2] > 0


@pytest.mark.parametrize("bounds,ground,margin", [([[0, 0], [0, 1]], 0, .2),
    ([[0, 0], [1, 1]], float("nan"), .2), ([[0, 0], [1, 1]], 0, 0), ([[0, 0], [1, 1]], 0, True)])
def test_invalid_floor_geometry_fails(bounds, ground, margin):
    with pytest.raises(ValueError):
        reference_ground_geometry(bounds, ground, margin)


def test_preflight_is_prepared_not_navigation_admitted_and_sources_are_bound(evidence):
    scene = provider(evidence)
    description = scene.describe()
    assert description["status"] == "prepared_not_runtime_integrated"
    assert description["floor_paths"] == [FLOOR_PATH]
    assert scene.static_paths == ("/Root/Wall",)
    assert set(scene.floor_paths).isdisjoint(scene.static_paths)
    assert not description["native_resource_map_admitted"] and not description["new_provider_PhysX_cooking_verified"]
    for action in (scene.require_navigation_map, lambda: scene.raster(np.zeros((1, 5))),
                   lambda: scene.ground_height_at([[-.1, -.1]])):
        with pytest.raises(RuntimeError, match="native full-grid"):
            action()
    Path(evidence.preflight["source"]["path"]).write_text("changed bytes")
    with pytest.raises(ValueError, match="changed"):
        scene.verify_sources()


@pytest.mark.parametrize("change", ["stacked_floor", "dynamic", "animated", "different_owner", "wildcard", "bad_dimensions"])
def test_preflight_rejects_inconsistent_physics_contracts(evidence, change):
    data = deepcopy(evidence.preflight)
    if change == "stacked_floor":
        data["reference_ground"]["disabled_visual_floor_collision_paths"] = []
    elif change == "dynamic":
        data["authored_rigid_bodies"] = [{"enabled": True, "kinematic": False}]
    elif change == "animated":
        data["time_sampled_xform_paths"] = ["/Root/Wall"]
    elif change == "different_owner":
        data["floor_owner"] = "scene_usd"
    elif change == "wildcard":
        data["authored_colliders"][1]["path"] = "/Root/Wall*"
    else:
        data["projection_metadata"]["dims"] = [4, 3]
    with pytest.raises(ValueError):
        DynBenchSceneProvider(data, evidence.preflight_record)


def test_query_enable_hook_copies_config_and_requires_known_precreate_field(evidence):
    scene = provider(evidence)
    original = SimpleNamespace(enable_scene_query_support=False)
    prepared = scene.prepare_simulation_cfg(original)
    assert prepared.enable_scene_query_support and not original.enable_scene_query_support
    assert scene.describe()["pre_create_hook_applied"]
    with pytest.raises(ValueError):
        scene.prepare_simulation_cfg(SimpleNamespace())


def test_lazy_scene_cfg_removes_only_generated_floor_and_keeps_one_global_source(evidence, monkeypatch):
    scene = provider(evidence)
    # This branch specifically represents a cold SDK import boundary.
    monkeypatch.delitem(sys.modules, "isaaclab.sim", raising=False)
    with pytest.raises(RuntimeError, match="Launch Kit"):
        scene.prepare_scene_cfg(SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.sim", SimpleNamespace(UsdFileCfg=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "isaaclab.assets", SimpleNamespace(AssetBaseCfg=lambda **kwargs: SimpleNamespace(**kwargs)))
    # A real CPU USD body-path lookup exercises the current contact filters.
    from pxr import Usd, UsdGeom, UsdPhysics
    from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES
    robot_usd = evidence.tmp / "robot.usda"
    stage = Usd.Stage.CreateNew(str(robot_usd))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    for body in PACER_BODY_NAMES:
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Robot/" + body).GetPrim())
    stage.GetRootLayer().Save()
    old = SimpleNamespace(terrain="existing terrain", filter_collisions=False, replicate_physics=True,
        robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(robot_usd))),
        contact_sensor_L_Toe=SimpleNamespace(filter_prim_paths_expr=["/World/ground/terrain/mesh", "/Root/Wall"]))
    with pytest.raises(ValueError, match="pre-create"):
        scene.prepare_scene_cfg(old)
    scene.prepare_simulation_cfg(SimpleNamespace(enable_scene_query_support=False))
    prepared = scene.prepare_scene_cfg(old)
    assert prepared.terrain is None and not prepared.replicate_physics
    assert prepared.robot == old.robot and old.terrain == "existing terrain" and not hasattr(old, "dynbench_source")
    assert prepared.contact_sensor_L_Toe.filter_prim_paths_expr == [FLOOR_PATH, "/Root/Wall"] + [
        "/World/envs/env_.*/Robot/" + body for body in PACER_BODY_NAMES]
    assert old.contact_sensor_L_Toe.filter_prim_paths_expr == ["/World/ground/terrain/mesh", "/Root/Wall"]
    assert prepared.dynbench_source.spawn.usd_path == evidence.preflight["source"]["path"]
    assert prepared.dynbench_source.spawn.func.__self__ is scene


@pytest.mark.parametrize("change", ["preflight_only", "sampled_support", "fallback", "floor_wall_mix", "changed_clock",
    "projection_promoted", "foreign_paths", "wrong_source", "missing_raw_floor_support"])
def test_native_admission_rejects_incomplete_or_mislabeled_contract_fixture(evidence, change):
    scene = provider(evidence)
    data = deepcopy(evidence.native)
    if change == "preflight_only":
        data = deepcopy(evidence.preflight)
    elif change == "sampled_support":
        data["floor_support"]["full_grid"] = False
    elif change == "fallback":
        data["fallback_used"] = True
    elif change == "floor_wall_mix":
        data["floor_collider_paths"] = [FLOOR_PATH, "/Root/Wall"]
    elif change == "changed_clock":
        data["physics"]["timeline_after_queries_s"] = 2.
    elif change == "projection_promoted":
        image = evidence.image.copy()
        image[0, 1] = 254
        Image.fromarray(image).save(evidence.tmp / "occupancy.png")
        data["exports"]["occupancy.png"] = file_record(evidence.tmp / "occupancy.png")
    elif change == "foreign_paths":
        data["foreign_native_hit_paths"] = ["/Unknown"]
    elif change == "wrong_source":
        data["source"]["sha256"] = "0" * 64
    else:
        p = evidence.tmp / "rays.npz"
        np.savez(p, native_free_index=np.arange(2), supported=np.array([True, False]),
            hit_position=np.zeros((2, 3)), hit_normal=np.ones((2, 3)), collision=np.array([FLOOR_PATH] * 2))
        data["floor_support"]["artifact"] = file_record(p)
    record = write_json(evidence.native_path, data)
    with pytest.raises(ValueError):
        scene.admit_native_map(evidence.native_path, expected_sha256=record["sha256"])
    assert not scene.describe()["native_resource_map_admitted"]


def test_admitted_contract_fixture_preserves_unknown_and_outside_raster(evidence):
    scene = admitted_fixture(evidence)
    raster = scene.raster(np.array([[-.15, -.05, 0., 0., 0.]]))
    assert tuple(raster.shape) == (1, 2, 224, 224)
    assert float(raster[0, 0, 112, 56]) == 1.
    assert float(raster[0, 0, 112, 57]) == .5
    assert float(raster[0, 0, 0, 0]) == .5
    assert np.allclose(raster.sum(1).numpy(), 1.)
    with pytest.raises(ValueError, match="indexing range"):
        scene.raster(np.array([[1e200, 0., 0., 0., 0.]]))


def test_ground_static_and_robot_native_hits_are_exactly_separate(evidence):
    scene = admitted_fixture(evidence)
    hits = [SimpleNamespace(collision=FLOOR_PATH, position=(0., 0., 0.), normal=(0., 0., 1.)),
            SimpleNamespace(collision="/Root/Wall", position=(0., 0., 1.2), normal=(0., 0., 1.)),
            SimpleNamespace(collision="/World/envs/env_0/Robot/Head", position=(0., 0., 1.7), normal=(0., 0., 1.))]

    def raycast(origin, direction, length, callback, any_hit):
        assert direction == (0., 0., -1.) and length == 2.1 and any_hit is False
        for hit in hits:
            callback(hit)
        return True

    scene.bind_native_query(SimpleNamespace(raycast_all=raycast), physics_step_count=lambda: 12,
                            dynamic_robot_roots=("/World/envs/env_0/Robot",))
    assert scene.ground_height_at([[0., 0.]]).tolist() == [0.]
    assert scene.native_surface_height_at([[0., 0.]]).tolist() == [1.2]
    assert scene.height_at([[0., 0.]]).tolist() == [1.8]  # Outside the tiny map is conservatively blocked.
    import torch
    tensor_input = torch.tensor([[0., 0.]], dtype=torch.float32)
    tensor_output = scene.native_surface_height_at(tensor_input)
    assert tensor_output.dtype == tensor_input.dtype and tensor_output.device == tensor_input.device
    assert tensor_output.shape == (1,)
    hits[:] = hits[1:]
    with pytest.raises(ValueError, match="exact-floor support"):
        scene.native_surface_height_at([[0., 0.]])


def test_native_height_rejects_counter_advance_and_out_of_finite_floor(evidence):
    scene = admitted_fixture(evidence)
    counter = iter((0, 1))

    def raycast(origin, direction, length, callback, any_hit):
        callback(SimpleNamespace(collision=FLOOR_PATH, position=(0., 0., 0.), normal=(0., 0., 1.)))
        return True

    scene.bind_native_query(SimpleNamespace(raycast_all=raycast), physics_step_count=lambda: next(counter))
    with pytest.raises(RuntimeError, match="advanced"):
        scene.ground_height_at([[0., 0.]])
    with pytest.raises(ValueError, match="no clamping"):
        scene.ground_height_at([[-100., -100.]])


def test_all_original_flat_Plane_and_Mesh_colliders_are_disabled_and_excluded_from_static(evidence):
    data = deepcopy(evidence.preflight)
    plane = {"path": "/Root/GroundPlane/CollisionPlane", "type_name": "Plane", "enabled": True,
             "static": True, "near_ground_flat_surface": True, "geometry": {"contract_fixture_only": True}}
    data["flat_ground_collider_inventory"].append(plane)
    data["authored_colliders"].append({"path": plane["path"], "enabled": True})
    data["reference_ground"]["original_flat_ground_collider_inventory"] = deepcopy(data["flat_ground_collider_inventory"])
    data["reference_ground_recipe"] = write_json(evidence.tmp / "recipe.json", data["reference_ground"])
    record = write_json(evidence.preflight_path, data)
    with pytest.raises(ValueError, match="flat Plane/Mesh"):
        DynBenchSceneProvider(data, record)
    for key in ("disabled_flat_ground_collision_paths", "disabled_visual_floor_collision_paths"):
        data["reference_ground"][key].append(plane["path"])
    data["reference_ground_recipe"] = write_json(evidence.tmp / "recipe.json", data["reference_ground"])
    record = write_json(evidence.preflight_path, data)
    scene = DynBenchSceneProvider(data, record)
    assert scene.static_collision_paths == ("/Root/Wall",) and scene.floor_roots == (FLOOR_PATH,)
    assert set(scene.describe()["disabled_original_flat_ground_paths"]) == {"/Root/Floor", plane["path"]}


@pytest.mark.parametrize("change", ["v4_missing_owner", "coplanar_all_hit", "coplanar_origin", "missing_all_hit_CSR", "missing_full_height"])
def test_nearest_declared_floor_hit_does_not_admit_extra_coplanar_support(evidence, change):
    scene = provider(evidence)
    data = deepcopy(evidence.native)
    if change == "v4_missing_owner":
        data.pop("floor_ownership")
    elif change == "coplanar_origin":
        data["origin_verification"]["floor_ray_all_hits"].append({"collision": "/Root/Wall", "position": [0., 0., 0.],
            "normal": [0., 0., 1.], "distance": .3})
    elif change == "missing_full_height":
        data.pop("full_height_band_queries")
    else:
        path = evidence.tmp / "rays.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        if change == "missing_all_hit_CSR":
            arrays.pop("all_hit_offsets")
        else:
            # Nearest selected hits remain intact; an additional coplanar static
            # collider hidden in the all-hit row must independently reject.
            arrays["all_hit_offsets"] = np.array([0, 2, 3])
            arrays["all_hit_count"] = np.array([2, 1])
            arrays["all_hit_collision_paths"] = np.array([FLOOR_PATH, "/Root/Wall", FLOOR_PATH])
            arrays["all_hit_rigid_body_paths"] = np.array(["", "", ""])
            arrays["all_hit_positions"] = np.array([[-.15, -.05, 0.], [-.15, -.05, 0.], [-.15, -.15, 0.]])
            arrays["all_hit_normals"] = np.tile([0., 0., 1.], (3, 1))
            arrays["all_hit_distances"] = np.full(3, .3)
            arrays["all_hit_classification"] = np.array(["selected_floor_support", "known_nonfloor_surface", "selected_floor_support"])
        np.savez(path, **arrays)
        data["floor_support"]["artifact"] = file_record(path)
        data["floor_ownership"]["artifact"] = file_record(path)
    record = write_json(evidence.native_path, data)
    with pytest.raises(ValueError):
        scene.admit_native_map(evidence.native_path, expected_sha256=record["sha256"])
    assert not scene.describe()["native_resource_map_admitted"]


def test_historical_probe_source_requires_exact_immutable_snapshot_not_current_code(evidence):
    snapshot = evidence.tmp / "historical_probe.py"
    snapshot.write_bytes(Path(evidence.preflight["script"]["path"]).read_bytes())
    Path(evidence.preflight["script"]["path"]).write_text("# later unrelated probe version\n")
    with pytest.raises(ValueError, match="changed"):
        provider(evidence)
    scene = DynBenchSceneProvider.from_preflight(evidence.preflight_path,
        expected_sha256=evidence.preflight_record["sha256"], probe_script_snapshot=snapshot)
    assert scene.verify_sources()
    assert scene.describe()["probe_script_verified_historical_snapshot"] == file_record(snapshot)
    snapshot.write_text("# wrong epoch\n")
    with pytest.raises(ValueError, match="changed"):
        scene.verify_sources()
    with pytest.raises(ValueError, match="recorded size and full SHA"):
        DynBenchSceneProvider.from_preflight(evidence.preflight_path,
            expected_sha256=evidence.preflight_record["sha256"], probe_script_snapshot=snapshot)


def test_policy_height_uses_full_band_blocked_cap_no_tall_top_or_per_pixel_callback_claim(evidence):
    scene = admitted_fixture(evidence)

    def no_per_pixel_callback(*args):
        raise AssertionError("Static admitted policy feature must not issue 32x32 native rays")

    scene.bind_native_query(SimpleNamespace(raycast_all=no_per_pixel_callback), physics_step_count=lambda: 9)
    points = np.array([[-.15, -.05], [-.05, -.05], [.05, -.05], [.15, .1]])
    np.testing.assert_array_equal(scene.height_at(points), [0., 1.8, 1.8, 1.8])
    assert not scene.describe()["policy_height_true_obstacle_top_claim"]
    assert scene.describe()["policy_height_per_pixel_native_callbacks"] == 0
    assert scene.height_at([[1., 1.]]).tolist() == [1.8]  # Crop feature may extend beyond actual support.
    with pytest.raises(ValueError, match="finite reference floor"):
        scene.ground_height_at([[1., 1.]])  # Actual root support cannot be invented or clipped.
    steps = iter((4, 5))
    scene.bind_native_query(SimpleNamespace(raycast_all=no_per_pixel_callback), physics_step_count=lambda: next(steps))
    with pytest.raises(RuntimeError, match="advanced"):
        scene.height_at(points)


def test_dynamic_robot_exclusion_has_exact_path_boundary_and_cannot_hide_source_floor(evidence):
    scene = admitted_fixture(evidence)
    collision = ["/World/envs/env_0/Robot/Head"]

    def raycast(origin, direction, length, callback, any_hit):
        callback(SimpleNamespace(collision=collision[0], position=(0., 0., .1), normal=(0., 0., 1.)))
        callback(SimpleNamespace(collision=FLOOR_PATH, position=(0., 0., 0.), normal=(0., 0., 1.)))
        return True

    scene.bind_native_query(SimpleNamespace(raycast_all=raycast), physics_step_count=lambda: 2,
        dynamic_robot_roots=("/World/envs/env_0/Robot",))
    assert scene.ground_height_at([[0., 0.]]).tolist() == [0.]
    collision[0] = "/World/envs/env_0/Robot2/Head"
    with pytest.raises(ValueError, match="undeclared collider"):
        scene.ground_height_at([[0., 0.]])
    with pytest.raises(ValueError, match="declared floor/static"):
        scene.bind_native_query(SimpleNamespace(raycast_all=raycast), physics_step_count=lambda: 2,
            dynamic_robot_roots=("/Root",))


def test_spawner_uses_pinned_thread_local_stage_getter_no_global_context_fallback(monkeypatch):
    stage = object()
    sdk = SimpleNamespace(get_current_stage=lambda: stage)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(sim=sdk))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sdk)
    assert _active_spawn_stage() is stage
    sdk.get_current_stage = lambda: None
    with pytest.raises(RuntimeError, match="No active thread-local"):
        _active_spawn_stage()
    sdk.get_current_stage = None
    with pytest.raises(RuntimeError, match="public get_current_stage"):
        _active_spawn_stage()


def test_fixed_SDK_xyzw_spawn_cannot_silently_treat_180degree_X_as_wxyz_identity(evidence, monkeypatch):
    from pxr import Usd
    # The public spawner now owns a real EditContext before calling the
    # argument validator. Supply that context and verify rejection is atomic.
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(sys.modules[DynBenchSceneProvider.__module__], "_active_spawn_stage", lambda: stage)
    before = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    target = stage.GetEditTarget()
    scene = provider(evidence)
    scene.prepare_simulation_cfg(SimpleNamespace(enable_scene_query_support=False))
    with pytest.raises(ValueError, match="one untransformed global"):
        scene._spawn("/__PaperDynBenchSceneProvider", None, orientation=(1., 0., 0., 0.))
    assert (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()) == before
    assert stage.GetEditTarget() == target
