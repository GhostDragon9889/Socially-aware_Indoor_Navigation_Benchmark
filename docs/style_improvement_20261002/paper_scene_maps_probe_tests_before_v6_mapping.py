"""SDK-free contracts for native map proof; these tests do not execute PhysX."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_paper_scene_maps as probe


def metadata():
    return {"scene": "Office", "composition_complete": True, "composition_errors": [],
            "stage": {"stage_up_axis": "Z", "stage_meters_per_unit": 1.0},
            "metadata": {"resolution": .1}, "pedestrian_height_band_above_floor_m": [.15, 1.8],
            "bounds_xy_m": [[-1., -2.], [1., 2.]], "image_dimensions_xy": [20, 40],
            "floor": {"height_m": 0., "selected_floor_prims": ["/Root/floor/mesh"]},
            "free_origin_candidate": {"omap_origin_world_map_frame_m": [-.2, .6, .75],
                "omap_relative_min_bound_map_m": [-.8, -2.6, -.6],
                "omap_relative_max_bound_map_m": [1.2, 1.4, 1.05]}}


def native_fixture():
    # Hand specified rectangular grid. North row=Y .15, south row=Y .05.
    occupied = np.array([[.05, .15, .75], [.35, .05, .75]])
    free = np.array([[.15, .15, .75], [.25, .05, .75]])
    # Fixed documented native layout: x descending within y ascending rows.
    raw = np.array([4, 5, 6, 6, 6, 6, 5, 4], dtype=np.float32)
    return raw, occupied, free, np.array([0., 0., .15]), np.array([.4, .2, 1.8]), np.array([4, 2, 1])


def test_fixed_official_native_layout_world_positions_verify_north_up():
    image, proof = probe.reconstruct_native_map(*native_fixture())
    np.testing.assert_array_equal(image, [[0, 254, 205, 205], [205, 205, 254, 0]])
    assert proof["raw_value_counts"] == {"occupied": 2, "free": 2, "unknown": 4}
    assert proof["unverified_buffer_spatial_layout_assumed"] is False
    assert proof["layout_fitted_dynamically"] is False
    scrambled = native_fixture()
    with pytest.raises(ValueError, match="occupied world positions"):
        probe.reconstruct_native_map(np.roll(scrambled[0], 1), *scrambled[1:])


@pytest.mark.parametrize("kind", ["counts", "duplicate", "both", "off_center", "outside", "bounds", "label", "nonfinite", "3d"])
def test_invalid_native_mapping_rejected(kind):
    args = list(deepcopy(native_fixture()))
    if kind == "counts":
        args[0][0] = 5
    elif kind == "duplicate":
        args[1][1] = args[1][0]
    elif kind == "both":
        args[2][0] = args[1][0]
    elif kind == "off_center":
        args[1][0, 0] += .025
    elif kind == "outside":
        args[1][0, 0] = -.05
    elif kind == "bounds":
        args[4][0] += .1
    elif kind == "label":
        args[0][0] = 99
    elif kind == "nonfinite":
        args[2][0, 1] = np.nan
    elif kind == "3d":
        args[5][2] = 2
    with pytest.raises(ValueError):
        probe.reconstruct_native_map(*args)


def test_real_float32_rounding_is_accepted_but_half_cell_shift_is_not():
    positions = np.array([[-24.55, 60.55, .75], [6.45, -27.55, .75]], dtype=np.float32)
    pixels = probe.positions_to_pixels(positions, [-24.6, -27.6, .15], [6.5, 60.6, 1.8], np.array([311, 882]))
    np.testing.assert_array_equal(pixels, [[0, 0], [881, 310]])
    positions[0, 0] += .05
    with pytest.raises(ValueError, match="centers"):
        probe.positions_to_pixels(positions, [-24.6, -27.6, .15], [6.5, 60.6, 1.8], np.array([311, 882]))


def test_only_exact_selected_floor_upward_near_height_support_is_admitted():
    def hit(collision, z, normal, distance):
        return {"collision": collision, "position": [0., 0., z], "normal": normal, "distance": distance}
    wall = hit("/Root/floor/mesh/wall", 0., [0., 0., 1.], .1)
    ceiling = hit("/Root/floor/mesh", .2, [0., 0., 1.], .1)
    underside = hit("/Root/floor/mesh", 0., [0., 0., -1.], .2)
    actual_floor = hit("/Root/floor/mesh", .01, [0., 0., 1.], .29)
    assert probe.supported_floor_hit([wall, ceiling, underside], {"/Root/floor/mesh"}, 0.) is None
    assert probe.supported_floor_hit([ceiling, actual_floor], {"/Root/floor/mesh"}, 0.) == actual_floor
    bad_normal = hit("/Root/floor/mesh", .01, [0., 0., 2.], .29)
    with pytest.raises(ValueError):
        probe.supported_floor_hit([bad_normal], {"/Root/floor/mesh"}, 0.)


def test_raycast_all_can_find_floor_behind_nonfloor_closest_hit():
    class Query:
        def raycast_all(self, origin, direction, distance, callback, both_sides):
            assert origin == (2., 3., .3) and direction == (0., 0., -1.) and distance == .6
            assert both_sides is False
            for collision, z in (("/table", .2), ("/floor", 0.)):
                assert callback(SimpleNamespace(collision=collision, rigid_body="", position=(2., 3., z),
                    normal=(0., 0., 1.), distance=.3 - z)) is True
            return True
    hits = probe.rays(Query(), [2., 3.], 0.)
    assert len(hits) == 2
    assert probe.supported_floor_hit(hits, {"/floor"}, 0.)["collision"] == "/floor"


def test_floor_owner_rejects_extra_coplanar_known_plane_but_not_a_high_roof():
    hits = [{"collision": path, "rigid_body": path, "position": [0., 0., z],
             "normal": [0., 0., 1.], "distance": .3-z}
            for path, z in (("/floor", 0.), ("/Plane", 8.94e-8), ("/roof", .2))]
    result = probe.classify_floor_ray_hits(hits, {"/floor"}, {"/floor", "/Plane", "/roof"}, 0.)
    assert result["classification"] == ["selected_floor_support", "undeclared_near_floor_support", "known_nonfloor_surface"]
    assert result["undeclared_near_floor_collision_paths"] == ["/Plane"]
    assert result["all_support_hit_paths_declared"] is False
    without_plane = probe.classify_floor_ray_hits([hits[0], hits[2]], {"/floor"}, {"/floor", "/roof"}, 0.)
    assert without_plane["all_support_hit_paths_declared"] is True
    foreign = probe.classify_floor_ray_hits([hits[2]], {"/floor"}, {"/floor"}, 0.)
    assert foreign["foreign_ray_collision_paths"] == ["/roof"]
    assert foreign["all_support_hit_paths_declared"] is False


def test_complete_floor_ray_CSR_preserves_extra_and_foreign_hits_for_each_cell():
    class Query:
        def raycast_all(self, origin, direction, distance, callback, both_sides):
            paths = ["/floor", "/Plane"] if origin[0] == 0. else ["/floor", "/foreign"]
            for path in paths:
                callback(SimpleNamespace(collision=path, rigid_body=path, position=(*origin[:2], 0.),
                    normal=(0., 0., 1.), distance=.3))
            return len(paths)
    result = probe.query_floor_support(Query(), np.array([[0., 0., .75], [1., 0., .75]]),
        np.array([0, 1]), 0., {"/floor"}, {"/floor", "/Plane"})
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["all_hit_offsets"], [0, 2, 4])
    np.testing.assert_array_equal(arrays["all_hit_count"], [2, 2])
    np.testing.assert_array_equal(arrays["supported"], [True, True])
    np.testing.assert_array_equal(arrays["undeclared_near_floor_hit"], [True, False])
    np.testing.assert_array_equal(arrays["foreign_hit"], [False, True])
    np.testing.assert_array_equal(arrays["all_hit_collision_paths"], ["/floor", "/Plane", "/floor", "/foreign"])
    np.testing.assert_array_equal(arrays["all_hit_positions"], [[0., 0., 0.]]*2+[[1., 0., 0.]]*2)
    assert result["all_support_hit_paths_declared"] is False


def test_plane_certificate_uses_actual_world_axis_and_height_not_name_or_visual_extent():
    transform = np.diag([.01, .01, .01, 1.])
    certificate = probe.floor_plane_geometry_certificate("Z", transform, 0.)
    np.testing.assert_allclose(certificate["world_unit_normal"], [0., 0., 1.])
    assert certificate["visual_extent_not_used_for_PhysX_plane_coverage"] is True
    transform[3, 2] = 2.
    with pytest.raises(ValueError, match="height"):
        probe.floor_plane_geometry_certificate("Z", transform, 0.)
    transform[3, 2] = 0.
    with pytest.raises(ValueError, match="horizontal"):
        probe.floor_plane_geometry_certificate("X", transform, 0.)


def test_flat_ground_inventory_includes_plane_types_and_session_disables_only_ground():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    plane = UsdGeom.Plane.Define(stage, "/unhelpfully_named_surface")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
    roof = UsdGeom.Plane.Define(stage, "/GroundPlane_but_actually_roof")
    roof.CreateAxisAttr("Z")
    roof.AddTranslateOp().Set((0., 0., 2.))
    UsdPhysics.CollisionAPI.Apply(roof.GetPrim()).CreateCollisionEnabledAttr(True)
    wall = UsdGeom.Plane.Define(stage, "/wall")
    wall.CreateAxisAttr("X")
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim()).CreateCollisionEnabledAttr(True)
    original = stage.GetRootLayer().ExportToString()
    inventory = probe.flat_ground_collider_inventory(stage, 0.)
    assert len(inventory) == 3 and all(r["type_name"] == "Plane" for r in inventory)
    selected = [r["path"] for r in inventory if r["near_ground_flat_surface"]]
    assert selected == ["/unhelpfully_named_surface"]
    recipe = probe.author_reference_ground(stage, [[-1., -1.], [1., 1.]], 0., selected)
    assert recipe["disabled_flat_ground_collision_paths"] == selected
    assert stage.GetRootLayer().ExportToString() == original
    after = probe.flat_ground_collider_inventory(stage, 0.)
    assert [r["path"] for r in after if r["near_ground_flat_surface"]] == [recipe["path"]]
    assert UsdPhysics.CollisionAPI(roof.GetPrim()).GetCollisionEnabledAttr().Get() is True
    assert UsdPhysics.CollisionAPI(wall.GetPrim()).GetCollisionEnabledAttr().Get() is True


def test_flat_ground_inventory_reports_enabled_dynamic_plane_instead_of_static_certificate():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    plane = UsdGeom.Plane.Define(stage, "/moving_plane")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(plane.GetPrim()).CreateRigidBodyEnabledAttr(True)
    record = probe.flat_ground_collider_inventory(stage, 0.)[0]
    assert record["near_ground_flat_surface"] is True and record["static"] is False


def test_actual_v4_warehouse_claimed_pass_is_rejected_by_undeclared_plane_gate():
    import hashlib
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "output/scene_maps/native_reference_warehouse_fullheight_v1/Warehouse/result.json"
    if not path.exists():
        pytest.skip("Requires the explicitly preserved native Warehouse v4 result")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "fafd094f2661886044093ead2c69438cfd0d289064b4b44c0f3947bbc07a9732"
    result = json.loads(path.read_text())
    hits = result["origin_verification"]["floor_ray_all_hits"]
    floor = set(result["floor_collider_paths"])
    verdict = probe.classify_floor_ray_hits(hits, floor, floor | {"/Root/GroundPlane/CollisionPlane"}, 0.)
    assert result["scene_resource_gate_passed"] is True  # The historical claim stays untouched.
    assert verdict["all_support_hit_paths_declared"] is False
    assert verdict["undeclared_near_floor_collision_paths"] == ["/Root/GroundPlane/CollisionPlane"]


def test_overlap_uses_actual_xyzw_identity_and_strict_callback_count():
    class Query:
        def overlap_box(self, extent, position, quaternion, callback, any_hit):
            assert extent == (.01, .01, .01) and position == (1., 2., .75)
            assert quaternion == (0., 0., 0., 1.) and any_hit is False
            callback(SimpleNamespace(collision="/obstacle", rigid_body="/body"))
            return 1
    assert probe.overlaps(Query(), [1., 2., .75], [.01] * 3) == [{"collision": "/obstacle", "rigid_body": "/body"}]


@pytest.mark.parametrize("kind", ["incomplete", "axis", "units", "resolution", "band", "dims", "origin", "floor"])
def test_preflight_metadata_rejects_unsupported_or_inconsistent_scene(kind):
    value = metadata()
    if kind == "incomplete":
        value["composition_complete"] = False
    elif kind == "axis":
        value["stage"]["stage_up_axis"] = "Y"
    elif kind == "units":
        value["stage"]["stage_meters_per_unit"] = .01
    elif kind == "resolution":
        value["metadata"]["resolution"] = .05
    elif kind == "band":
        value["pedestrian_height_band_above_floor_m"][0] = 0
    elif kind == "dims":
        value["image_dimensions_xy"][0] = 21
    elif kind == "origin":
        value["free_origin_candidate"]["omap_relative_min_bound_map_m"][0] = -10
    elif kind == "floor":
        value["floor"]["selected_floor_prims"] = []
    with pytest.raises(ValueError):
        probe.validate_projection_metadata("Office", value)


def test_admitted_metadata_has_world_height_band_not_origin_relative_confusion():
    checked = probe.validate_projection_metadata("Office", metadata())
    np.testing.assert_allclose(checked["origin"] + checked["relative_min"], [-1., -2., .15])
    np.testing.assert_allclose(checked["origin"] + checked["relative_max"], [1., 2., 1.8])


def test_limited_probe_never_navigation_ready_even_when_every_sample_supported():
    kwargs = dict(origin_clear=True, origin_supported=True, mapping_valid=True, full_floor_probe=False,
                  full_height_probe=True, height_clear_mask=np.ones(8, dtype=bool), all_band_hit_paths_classified=True,
                  floor_ownership_verified=True,
                  support_mask=np.ones(8, dtype=bool), static_unchanged=True, sampled_free_overlap_count=0, source_unchanged=True)
    assert probe.navigation_admission(**kwargs) is False
    kwargs["full_floor_probe"] = True
    assert probe.navigation_admission(**kwargs) is True
    for key in ("origin_clear", "origin_supported", "mapping_valid", "static_unchanged", "source_unchanged"):
        modified = dict(kwargs, **{key: False})
        assert probe.navigation_admission(**modified) is False
    assert probe.navigation_admission(**dict(kwargs, sampled_free_overlap_count=1)) is False
    assert probe.navigation_admission(**dict(kwargs, support_mask=np.array([True, False]), height_clear_mask=np.array([True, True]))) is False
    assert probe.navigation_admission(**dict(kwargs, full_height_probe=False)) is False
    assert probe.navigation_admission(**dict(kwargs, all_band_hit_paths_classified=False)) is False
    assert probe.navigation_admission(**dict(kwargs, floor_ownership_verified=False)) is False
    assert probe.navigation_admission(**dict(kwargs, height_clear_mask=np.zeros(8, dtype=bool))) is False
    with pytest.raises(ValueError):
        probe.navigation_admission(**dict(kwargs, support_mask=np.array([1., np.nan])))


def test_sampling_is_reproducible_complete_without_random_fallback():
    np.testing.assert_array_equal(probe.floor_probe_indices(7, 0), np.arange(7))
    np.testing.assert_array_equal(probe.floor_probe_indices(7, 3), [0, 3, 6])
    points = probe.fixed_probe_cells((2, 4), 6, 7)
    np.testing.assert_array_equal(points, probe.fixed_probe_cells((2, 4), 6, 7))
    assert len(np.unique(points, axis=0)) == 6
    assert probe.parser().parse_args(["--output", "/tmp/fresh"]).execute_physx is False


def structural_floor(path, x=0.):
    points = np.array([[x, 0., 0.], [x + 1, 0., 0.], [x + 1, 1., 0.], [x, 1., 0.]])
    return {"path": path, **probe.floor_mesh_geometry_certificate(points, np.array([4]), np.arange(4), np.eye(4), 0.)}


def test_floor_certificate_uses_real_face_winding_topology_and_row_vector_transform():
    transform = np.eye(4)
    transform[3, :3] = [3., 4., 0.]
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    cert = probe.floor_mesh_geometry_certificate(points, np.array([4]), np.arange(4), transform, 0.)
    np.testing.assert_allclose(cert["world_bounds_m"], [[3., 4., 0.], [4., 5., 0.]])
    assert cert["upward_top_fan_area_m2"] == 1.
    with pytest.raises(ValueError, match="upward"):
        probe.floor_mesh_geometry_certificate(points, np.array([4]), np.arange(4)[::-1], transform, 0.)
    with pytest.raises(ValueError, match="topology"):
        probe.floor_mesh_geometry_certificate(points, np.array([4]), np.array([0, 1, 2, 8]), transform, 0.)


def test_unique_floor_recipe_excludes_decor_and_whole_duplicate_but_keeps_seams():
    records = [structural_floor("/base"), structural_floor("/same"),
               structural_floor("/next", 1.), structural_floor("/FloorDecal", .5)]
    retained, excluded, unresolved = probe.choose_unique_floor_surfaces(records)
    assert [r["path"] for r in retained] == ["/base", "/next"]
    assert {r["path"] for r in excluded} == {"/same", "/FloorDecal"}
    assert unresolved == []
    assert next(r for r in excluded if r["path"] == "/same")["retained"] == "/base"


def test_partial_coplanar_overlap_blocks_unique_floor_recipe():
    _, _, unresolved = probe.choose_unique_floor_surfaces([structural_floor("/base"), structural_floor("/overlap", .5)])
    assert len(unresolved) == 1 and np.isclose(unresolved[0]["overlap_area_m2"], .5)


def test_session_overlay_authors_static_triangle_offsets_without_mutating_source_layer():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/floor")
    mesh.CreatePointsAttr([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    decal = UsdGeom.Mesh.Define(stage, "/decal")
    UsdPhysics.CollisionAPI.Apply(decal.GetPrim()).CreateCollisionEnabledAttr(True)
    original = stage.GetRootLayer().ExportToString()
    record = probe.author_floor_overlay(stage, ["/floor"], ["/decal"])
    assert stage.GetRootLayer().ExportToString() == original
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get() is True
    assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == "none"
    assert UsdPhysics.CollisionAPI(decal.GetPrim()).GetCollisionEnabledAttr().Get() is False
    assert record["native_cooking_verified"] is False


def test_reference_ground_is_finite_upward_triangle_mesh_at_declared_height():
    points, counts, indices = probe.reference_ground_geometry([[-1., -2.], [3., 4.]], .125, .2)
    np.testing.assert_allclose(points, [[-1.2, -2.2, .125], [3.2, -2.2, .125],
                                      [3.2, 4.2, .125], [-1.2, 4.2, .125]])
    np.testing.assert_array_equal(counts, [3, 3])
    cert = probe.floor_mesh_geometry_certificate(points, counts, indices, np.eye(4), .125)
    assert np.isclose(cert['upward_top_fan_area_m2'], 4.4 * 6.4)
    with pytest.raises(ValueError):
        probe.reference_ground_geometry([[-1., -2.], [3., 4.]], 0., np.inf)


def test_reference_ground_session_has_one_owner_and_disables_original_visual_floor():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateInMemory()
    visual = UsdGeom.Mesh.Define(stage, '/floor_visual')
    UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True)
    original = stage.GetRootLayer().ExportToString()
    record = probe.author_reference_ground(stage, [[-1., -2.], [3., 4.]], 0., ['/floor_visual'])
    assert stage.GetRootLayer().ExportToString() == original
    assert UsdPhysics.CollisionAPI(visual.GetPrim()).GetCollisionEnabledAttr().Get() is False
    ground = stage.GetPrimAtPath(record['path'])
    assert UsdPhysics.CollisionAPI(ground).GetCollisionEnabledAttr().Get() is True
    assert not ground.HasAPI(UsdPhysics.RigidBodyAPI)
    mat = UsdShade.Material(stage.GetPrimAtPath(record['path'] + '/PhysicsMaterial'))
    assert UsdPhysics.MaterialAPI(mat.GetPrim()).GetStaticFrictionAttr().Get() == 1.
    assert UsdPhysics.MaterialAPI(mat.GetPrim()).GetRestitutionAttr().Get() == 0.
    assert UsdShade.MaterialBindingAPI(ground).ComputeBoundMaterial('physics')[0].GetPath() == mat.GetPath()
    assert record['original_USD_floor_collider'] is False
    assert record['paper_private_scene_configuration'] is False
    with pytest.raises(ValueError, match='stacked'):
        probe.author_reference_ground(stage, [[-1., -2.], [3., 4.]], 0., ['/floor_visual'])


def test_added_reference_ground_cannot_promote_projection_unknown_to_free():
    native = np.array([[254, 0, 254], [205, 254, 0]], dtype=np.uint8)
    original_projection = np.array([[205, 205, 254], [205, 0, 254]], dtype=np.uint8)
    np.testing.assert_array_equal(probe.retain_projection_unknown(native, original_projection),
                                  [[205, 0, 254], [205, 254, 0]])
    np.testing.assert_array_equal(native, [[254, 0, 254], [205, 254, 0]])


def test_reference_ground_disables_instance_proxy_collider_only_in_session():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    template = UsdGeom.Xform.Define(stage, '/Template')
    mesh = UsdGeom.Mesh.Define(stage, '/Template/floor')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    instance = UsdGeom.Xform.Define(stage, '/Instance').GetPrim()
    instance.GetReferences().AddInternalReference(template.GetPath())
    instance.SetInstanceable(True)
    assert stage.GetPrimAtPath('/Instance/floor').IsInstanceProxy()
    root_layer_original = stage.GetRootLayer().ExportToString()
    record = probe.author_reference_ground(stage, [[-1., -2.], [3., 4.]], 0., ['/Instance/floor'])
    assert record['deinstanced_floor_roots_in_session'] == ['/Instance']
    assert not stage.GetPrimAtPath('/Instance/floor').IsInstanceProxy()
    assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/Instance/floor')).GetCollisionEnabledAttr().Get() is False
    assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/Template/floor')).GetCollisionEnabledAttr().Get() is True
    assert stage.GetRootLayer().ExportToString() == root_layer_original


@pytest.mark.parametrize("ground_recipe, expected", [(None, False), ({"path": "/__PaperMapProbeReferenceGround"}, True)])
def test_actual_floor_owner_metadata_marks_only_created_reference_ground(ground_recipe, expected):
    """Evaluate the published declaration from the real entry without Kit."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(probe.run_native_scene))
    declarations = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                    and any(isinstance(key, ast.Constant) and key.value == "additional_Terrain_or_ground_created"
                            for key in node.keys)]
    assert len(declarations) == 1
    node = declarations[0]
    value = next(value for key, value in zip(node.keys, node.values, strict=True)
                 if isinstance(key, ast.Constant) and key.value == "additional_Terrain_or_ground_created")
    expression = ast.Expression(body=value)
    actual = eval(compile(ast.fix_missing_locations(expression), "actual-floor-owner-metadata", "eval"),
                  {"__builtins__": {}}, {"ground_recipe": ground_recipe})
    assert actual is expected


def test_API_free_unknown_positions_remain_unknown_without_promoting_free():
    args = list(native_fixture())
    args[2] = np.concatenate((args[2], [[.15, .05, .75]]))
    image, proof = probe.reconstruct_native_map(*args)
    assert image[1, 1] == 205
    assert proof["API_free_list_count"] == 3
    assert proof["canonical_known_free_count"] == 2
    assert proof["API_free_list_unknown_count"] == 1
    assert proof["unreported_unknown_count"] == 3
    assert proof["unknown_promoted_to_free"] is False


def test_actual_warehouse_native_raw_requires_exact_occupied_knownfree_and_unknown_preservation():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "output/scene_maps/native_reference_warehouse_pilot_v1/Warehouse/native_omap.npz"
    if not path.exists():
        pytest.skip("Requires the explicitly recorded Warehouse native PhysX raw fixture")
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "d3931cd7f6557dd046ef809a62ba8764e473351c3fd4aaa069c7779e357a9f29"
    with np.load(path, allow_pickle=False) as archive:
        image, proof = probe.reconstruct_native_map(archive["buffer"], archive["occupied_positions"],
            archive["free_positions"], archive["min_bound"], archive["max_bound"], archive["dimensions"])
        index = probe.native_position_buffer_indices(archive["free_positions"], archive["min_bound"],
            archive["max_bound"], archive["dimensions"])
        assert (archive["buffer"][index] == 4).sum() == 0
    assert proof["raw_value_counts"] == {"occupied": 3343, "free": 56239, "unknown": 33288}
    assert proof["API_free_list_count"] == 89025
    assert proof["API_free_list_unknown_count"] == 32786
    assert proof["unreported_unknown_count"] == 502
    assert proof["unreported_unknown_native_rows"] == [1, 3]
    assert (image == 205).sum() == 33288
    assert (image == 254).sum() == 56239


def test_missing_known_free_world_point_and_false_free_occupied_point_are_rejected():
    args = list(native_fixture())
    args[2] = args[2][1:]
    with pytest.raises(ValueError, match="every known buffer5"):
        probe.reconstruct_native_map(*args)
    args = list(native_fixture())
    args[2] = np.concatenate((args[2], args[1][:1]))
    with pytest.raises(ValueError, match="intersect occupied buffer4"):
        probe.reconstruct_native_map(*args)


@pytest.mark.parametrize("ground", [0., 2., -3.])
def test_full_height_cell_box_uses_declared_world_height_and_cell_extents(ground):
    center, extent = probe.height_band_box([1.2, -2.3], ground)
    np.testing.assert_allclose(np.asarray(center)-extent, [1.15, -2.35, ground+.15])
    np.testing.assert_allclose(np.asarray(center)+extent, [1.25, -2.25, ground+1.8])


@pytest.mark.parametrize("args", [([1., np.nan], 0.), ([1., 2.], np.nan), ([1., 2.], True)])
def test_full_height_query_rejects_nonfinite_or_boolean_coordinates(args):
    with pytest.raises(ValueError):
        probe.height_band_box(*args)


@pytest.mark.parametrize("kwargs", [{"resolution": .2}, {"resolution": True}, {"band": (.7, .8)},
                                    {"band": (1.8, .15)}, {"band": (0., 2.)}])
def test_full_height_query_rejects_changed_resolution_or_short_height_band(kwargs):
    with pytest.raises(ValueError):
        probe.height_band_box([1., 2.], 0., **kwargs)


class HeightQuery:
    def __init__(self):
        self.volumes = []

    def overlap_box(self, extent, position, quaternion, callback, any_hit):
        self.volumes.append((extent, position))
        assert quaternion == (0., 0., 0., 1.) and any_hit is False
        np.testing.assert_allclose(extent, [.05, .05, .825])
        np.testing.assert_allclose(position[2], .975)
        collision = "/floor" if position[0] < .2 else ("/authored/box" if position[0] < .3 else "/foreign")
        assert callback(SimpleNamespace(collision=collision, rigid_body="/static")) is True
        return 1


def test_full_height_records_exact_paths_excludes_floor_and_rejects_foreign_classification():
    query = HeightQuery()
    output = probe.query_full_height_band(query, np.array([[.15, .05, .75], [.25, .05, .75], [.35, .05, .75]]),
        np.array([0, 1, 2]), 0., {"/floor"}, {"/floor", "/authored/box"},
        counter_getter=lambda: 2, time_getter=lambda: .0)
    arrays = output["arrays"]
    assert len(query.volumes) == 3
    np.testing.assert_array_equal(arrays["full_height_clear"], [True, False, False])
    np.testing.assert_array_equal(arrays["authored_obstacle_hit"], [False, True, False])
    np.testing.assert_array_equal(arrays["all_hit_paths_classified"], [True, True, False])
    np.testing.assert_array_equal(arrays["hit_offsets"], [0, 1, 2, 3])
    assert arrays["hit_collision_paths"].tolist() == ["/floor", "/authored/box", "/foreign"]
    assert arrays["hit_classification"].tolist() == ["selected_floor_excluded", "authored_obstacle", "foreign_unclassified"]
    assert output["foreign_hit_paths"] == ["/foreign"]
    probe.require_query_clock_unchanged(output["clock_before"], output["clock_after"])


@pytest.mark.parametrize("dimension", ["physics", "timeline"])
def test_any_query_clock_advance_fails_after_retaining_actual_query_evidence(dimension):
    counts = iter([2, 3] if dimension == "physics" else [2, 2])
    times = iter([0., .000001] if dimension == "timeline" else [0., 0.])
    output = probe.query_full_height_band(HeightQuery(), np.array([[.15, .05, .75]]), np.array([0]),
        0., {"/floor"}, {"/floor", "/authored/box"}, counter_getter=lambda: next(counts), time_getter=lambda: next(times))
    assert output["query_time_unchanged"] is False
    assert output["arrays"]["hit_collision_paths"].tolist() == ["/floor"]
    with pytest.raises(RuntimeError, match="Physics/timeline advanced"):
        probe.require_query_clock_unchanged(output["clock_before"], output["clock_after"])


@pytest.mark.parametrize("counter, timestamp", [(True, 0.), (-1, 0.), (0, np.nan), (0, True)])
def test_invalid_query_clock_is_rejected(counter, timestamp):
    with pytest.raises(ValueError):
        probe.observed_query_clock(lambda: counter, lambda: timestamp)


def band_masks(selected, clear, obstacle, classified):
    return {"native_free_index": np.asarray(selected), "full_height_clear": np.asarray(clear, dtype=bool),
        "authored_obstacle_hit": np.asarray(obstacle, dtype=bool), "all_hit_paths_classified": np.asarray(classified, dtype=bool)}


def test_refinement_downgrades_high_obstacle_keeps_rawunknown_and_requires_support():
    native = np.array([[0, 254, 205, 254]], dtype=np.uint8)
    points = np.array([[.15, .05, .75], [.35, .05, .75]])
    result = probe.refine_known_free_map(native, points, np.array([0, 1]), np.array([True, True]),
        band_masks([0, 1], [False, True], [True, False], [True, True]),
        [0., 0., .7], [.4, .1, .8], np.array([4, 1, 1]))
    np.testing.assert_array_equal(result, [[0, 0, 205, 254]])
    limited = probe.refine_known_free_map(native, points, np.array([1]), np.array([False]),
        band_masks([1], [True], [False], [True]), [0., 0., .7], [.4, .1, .8], np.array([4, 1, 1]))
    np.testing.assert_array_equal(limited, [[0, 205, 205, 205]])
    unclassified = probe.refine_known_free_map(native, points, np.array([1]), np.array([True]),
        band_masks([1], [False], [False], [False]), [0., 0., .7], [.4, .1, .8], np.array([4, 1, 1]))
    np.testing.assert_array_equal(unclassified, [[0, 205, 205, 205]])
    np.testing.assert_array_equal(native, [[0, 254, 205, 254]])


def test_raw6_API_points_cannot_enter_free_refinement_and_false_clear_is_rejected():
    native = np.array([[205, 254]], dtype=np.uint8)
    with pytest.raises(ValueError, match="canonical native known5"):
        probe.refine_known_free_map(native, [[.05, .05, .75]], np.array([0]), np.array([True]),
            band_masks([0], [True], [False], [True]), [0., 0., .7], [.2, .1, .8], np.array([2, 1, 1]))
    with pytest.raises(ValueError, match="contradicts"):
        probe.refine_known_free_map(native, [[.15, .05, .75]], np.array([0]), np.array([True]),
            band_masks([0], [True], [True], [True]), [0., 0., .7], [.2, .1, .8], np.array([2, 1, 1]))


def test_actual_warehouse_raw_refinement_preserves_all33288_unknown_without_native_claim():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "output/scene_maps/native_reference_warehouse_pilot_v1/Warehouse/native_omap.npz"
    if not path.exists():
        pytest.skip("Requires explicitly recorded native raw Warehouse fixture")
    with np.load(path, allow_pickle=False) as z:
        image, _ = probe.reconstruct_native_map(z["buffer"], z["occupied_positions"], z["free_positions"], z["min_bound"], z["max_bound"], z["dimensions"])
        indices = probe.native_position_buffer_indices(z["free_positions"], z["min_bound"], z["max_bound"], z["dimensions"])
        points = z["free_positions"][z["buffer"][indices] == 5]
        selected = np.arange(len(points))
        blocked = np.zeros(len(points), dtype=bool)
        blocked[0] = True  # synthetic classification only; no new native query is claimed
        classified = np.ones_like(blocked)
        masks = band_masks(selected, ~blocked, blocked, classified)
        result = probe.refine_known_free_map(image, points, selected, np.ones_like(blocked), masks,
            z["min_bound"], z["max_bound"], z["dimensions"])
    assert (result == 205).sum() == 33288
    assert (result == 254).sum() == 56238
    assert (result == 0).sum() == 3344
