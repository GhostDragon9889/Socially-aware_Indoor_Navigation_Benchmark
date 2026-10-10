"""Real offline USD geometry tests; no Kit, PhysX or GPU claim."""
from argparse import Namespace
import json

import numpy as np
from PIL import Image
import pytest
import yaml

pytest.importorskip("pxr.Usd")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from scripts.generate_pedestrian_maps import (
    aligned_grid_bounds, audit_composition, choose_floor, collect_geometry,
    generate_scene, pixel_to_world_center, project_geometry,
    stage_to_map_matrix, world_to_pixel,
)


def cube(stage, path, scale, translation):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.GetSizeAttr().Set(2.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*translation))
    shape.AddScaleOp().Set(Gf.Vec3f(*scale))
    return shape


def centimeter_scene(path=None):
    stage = Usd.Stage.CreateNew(str(path)) if path else Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    cube(stage, "/World/Floor", (500, 500, 5), (-300, 200, -5))
    obstacle = cube(stage, "/World/Wall", (10, 40, 80), (-600, 350, 80))
    UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
    cube(stage, "/World/Roof", (200, 200, 10), (-300, 200, 400))
    if path:
        stage.GetRootLayer().Save()
    return stage


def projected(stage, incomplete=False):
    geometry, info = collect_geometry(stage)
    ground, floors, floor_info = choose_floor(geometry)
    origin, maximum, size = aligned_grid_bounds(floors, .1, .1)
    image, preview, statistics = project_geometry(geometry, floors, origin=origin,
        size=size, resolution=.1, ground_height=ground, incomplete=incomplete)
    return image, preview, statistics, origin, maximum, info, floor_info


def test_real_centimeter_negative_world_geometry_and_height_filter():
    image, _, stats, origin, _, info, floor = projected(centimeter_scene())
    assert info["stage_meters_per_unit"] == .01
    assert floor["height_m"] == 0
    np.testing.assert_allclose(origin, [-8.1, -3.1])
    obstacle = world_to_pixel([-6., 3.5], origin, .1, image.shape)
    free = world_to_pixel([-3., 2.], origin, .1, image.shape)
    assert image[tuple(obstacle)] == 0
    assert image[tuple(free)] == 254  # Overhead roof was excluded.
    assert image[0, 0] == 205  # Padding is not authored floor support.
    assert stats["projected_obstacle_gprims"] == 1
    assert stats["skipped_obstacles"]["above_pedestrian_height_band"] == 1
    assert info["authored_collision_schema_count"] == 1


def test_actual_y_up_geometry_preserves_right_handed_meter_coordinates():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    cube(stage, "/World/Floor", (5, .05, 5), (-2, -.05, -3))
    cube(stage, "/World/Wall", (.2, .8, .3), (-4, .8, -5))
    image, _, _, origin, _, info, _ = projected(stage)
    np.testing.assert_allclose(info["stage_units_to_map_meters"], [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.linalg.det(stage_to_map_matrix("Y", 1.)) == 1
    pixel = world_to_pixel([-4., 5.], origin, .1, image.shape)
    assert image[tuple(pixel)] == 0
    assert info["stage_up_axis"] == "Y"


def test_world_grid_centers_y_flip_and_negative_origin():
    shape = (13, 17)
    origin = np.array([-11.3, -7.2])
    pixels = np.array([[0, 0], [12, 16], [7, 9]])
    centers = pixel_to_world_center(pixels, origin, .05, shape)
    assert centers[0, 1] > centers[1, 1]
    np.testing.assert_array_equal(world_to_pixel(centers, origin, .05, shape), pixels)
    np.testing.assert_allclose(centers[-1], [-10.825, -6.925])


def test_missing_geometry_never_becomes_free_and_preview_is_separate():
    image, preview, statistics, *_ = projected(centimeter_scene(), incomplete=True)
    assert not np.any(image == 254)
    assert np.any(preview == 254)
    assert np.any(image == 0)
    np.testing.assert_array_equal(image == 0, preview == 0)
    assert statistics["free_cells"] == 0


def test_preflight_records_remote_and_missing_layers_without_stage_resolution(tmp_path):
    path = tmp_path / "offline.usda"
    layer = Sdf.Layer.CreateNew(str(path))
    prim = Sdf.CreatePrimInLayer(layer, "/World")
    prim.specifier = Sdf.SpecifierDef
    prim.referenceList.prependedItems = [Sdf.Reference("omniverse://unavailable.example/scene.usd"),
                                        Sdf.Reference("missing.usd")]
    layer.Save()
    before = path.read_bytes()
    audit = audit_composition(path)
    assert not audit["composition_complete"]
    assert audit["missing_local_layers"] == [str(tmp_path / "missing.usd")]
    assert audit["remote_composition_arcs"][0]["asset"].startswith("omniverse://")
    assert path.read_bytes() == before


def test_real_png_yaml_export_has_meter_origin_and_provenance(tmp_path):
    assets = tmp_path / "assets/Office"
    assets.mkdir(parents=True)
    usd = assets / "office_partnav.usd"
    centimeter_scene(usd)
    before = usd.read_bytes()
    output = tmp_path / "maps"
    args = Namespace(output=output, assets=assets.parent, worker="Office", allow_incomplete=False,
                     inventory_only=False, resolution=.1, margin=.1, height_min=.15, height_max=1.8,
                     ground_height=None, extent_mode="floor-support", max_cells=100_000)
    result = generate_scene(args)
    assert result["status"] == "prototype_ready_for_physx_review"
    metadata = yaml.safe_load((output / "Office/occupancy.yaml").read_text())
    np.testing.assert_allclose(metadata["origin"], [-8.1, -3.1, 0])
    assert metadata["resolution"] == .1
    image = np.array(Image.open(output / "Office/occupancy.png"))
    wall_pixel = world_to_pixel([-6, 3.5], np.array(metadata["origin"][:2]), .1, image.shape)
    assert image[tuple(wall_pixel)] == 0
    report = json.loads((output / "Office/provenance.json").read_text())
    assert report["coordinate_check"]["pixel_centers_world_grid_roundtrip"]
    assert not report["physx_collider_verified"]
    assert not report["author_eight_scene_map"]
    assert usd.read_bytes() == before
