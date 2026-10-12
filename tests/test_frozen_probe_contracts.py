"""Execute retained probe bodies; source hashes alone are not functionality tests.

These contracts test numerical admission and real OpenUSD layer operations.
They do not launch Kit or claim fresh PhysX scene acceptance.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

import test_paper_scene_maps_probe as contracts

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "resources/runtime_sources/style_improvement_20261002"
SOURCE_PATHS = {
    "v6": BASE / "paper_scene_maps_probe_source_before_v6_mapping.py",
    "v7": BASE
    / "native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py",
    "v8": BASE
    / "native_map_clock_v8_proposal_v2/proposed/scripts/probe_paper_scene_maps.py",
    "v9": BASE
    / "native_map_clock_v9_proposal_v2/proposed/scripts/probe_paper_scene_maps.py",
    "collected": ROOT
    / "resources/releases/pedestrians_20261010/evidence/probe_collected_cell_queries_v1.py",
}


def load_sources():
    modules = {}
    # Load before test collection brings in the simulator. The collected
    # release explicitly requires stock CPython's path-stat before Kit import.
    for name, path in SOURCE_PATHS.items():
        alias = "frozen_probe_contract_" + name
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


SOURCES = load_sources()
SOURCES["current"] = contracts.probe


@pytest.fixture(params=list(SOURCES))
def original_probe(request, monkeypatch):
    module = SOURCES[request.param]
    path = Path(module.__file__)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(contracts, "probe", module)
    yield module
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


COMMON_CONTRACTS = [
    "test_fixed_official_native_layout_world_positions_verify_north_up",
    "test_real_float32_rounding_is_accepted_but_half_cell_shift_is_not",
    "test_only_exact_selected_floor_upward_near_height_support_is_admitted",
    "test_raycast_all_can_find_floor_behind_nonfloor_closest_hit",
    "test_floor_owner_rejects_extra_coplanar_known_plane_but_not_a_high_roof",
    "test_plane_certificate_uses_actual_world_axis_and_height_not_name_or_visual_extent",
    "test_flat_ground_inventory_includes_plane_types_and_session_disables_only_ground",
    "test_flat_ground_inventory_reports_enabled_dynamic_plane_instead_of_static_certificate",
    "test_overlap_uses_actual_xyzw_identity_and_strict_callback_count",
    "test_admitted_metadata_has_world_height_band_not_origin_relative_confusion",
    "test_limited_probe_never_navigation_ready_even_when_every_sample_supported",
    "test_sampling_is_reproducible_complete_without_random_fallback",
    "test_floor_certificate_uses_real_face_winding_topology_and_row_vector_transform",
    "test_unique_floor_recipe_excludes_decor_and_whole_duplicate_but_keeps_seams",
    "test_partial_coplanar_overlap_blocks_unique_floor_recipe",
    "test_session_overlay_authors_static_triangle_offsets_without_mutating_source_layer",
    "test_reference_ground_is_finite_upward_triangle_mesh_at_declared_height",
    "test_reference_ground_session_has_one_owner_and_disables_original_visual_floor",
    "test_added_reference_ground_cannot_promote_projection_unknown_to_free",
    "test_reference_ground_disables_instance_proxy_collider_only_in_session",
    "test_API_free_unknown_positions_remain_unknown_without_promoting_free",
    "test_missing_known_free_world_point_and_false_free_occupied_point_are_rejected",
    "test_refinement_downgrades_high_obstacle_keeps_rawunknown_and_requires_support",
    "test_raw6_API_points_cannot_enter_free_refinement_and_false_clear_is_rejected",
    "test_complete_floor_ray_CSR_preserves_extra_and_foreign_hits_for_each_cell",
    "test_full_height_records_exact_paths_excludes_floor_and_rejects_foreign_classification",
]


@pytest.mark.parametrize("contract", COMMON_CONTRACTS)
def test_retained_versions_obey_shared_admission_contracts(original_probe, contract):
    getattr(contracts, contract)()


@pytest.mark.parametrize(
    "kind",
    [
        "counts",
        "duplicate",
        "both",
        "off_center",
        "outside",
        "bounds",
        "label",
        "nonfinite",
        "3d",
    ],
)
def test_retained_versions_reject_invalid_native_grid(original_probe, kind):
    contracts.test_invalid_native_mapping_rejected(kind)


@pytest.mark.parametrize(
    "kind",
    ["incomplete", "axis", "units", "resolution", "band", "dims", "origin", "floor"],
)
def test_retained_versions_reject_inconsistent_projection_metadata(
    original_probe, kind
):
    contracts.test_preflight_metadata_rejects_unsupported_or_inconsistent_scene(kind)


def test_probe_file_receipts_and_finite_json_are_exact(original_probe, tmp_path):
    source = tmp_path / "fixture.bin"
    source.write_bytes(b"fixture data")
    record = original_probe.file_record(source)
    assert record == {
        "path": str(source),
        "bytes": 12,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    output = tmp_path / "receipt.json"
    original_probe.write_json(output, record)
    import json

    assert json.loads(output.read_text()) == record
    with pytest.raises(ValueError):
        original_probe.write_json(output, {"nonfinite": float("nan")})


@pytest.mark.parametrize("dimension", ["physics", "timeline"])
def test_retained_probe_keeps_evidence_when_query_clock_advances(
    original_probe, dimension
):
    contracts.test_any_query_clock_advance_fails_after_retaining_actual_query_evidence(
        dimension
    )


@pytest.mark.parametrize(
    "counter,timestamp", [(True, 0.0), (-1, 0.0), (0, float("nan")), (0, True)]
)
def test_retained_probe_rejects_invalid_query_clock(original_probe, counter, timestamp):
    contracts.test_invalid_query_clock_is_rejected(counter, timestamp)


def test_duplicate_floor_hits_preserve_versioned_certificate_contract(original_probe):
    if original_probe is not SOURCES["v6"]:
        contracts.test_duplicate_selected_floor_hits_remain_saved_and_fail_single_floor_certificate()
        return

    import numpy as np
    from types import SimpleNamespace

    class Query:
        def raycast_all(self, origin, direction, distance, callback, both_sides):
            for _ in range(2):
                callback(
                    SimpleNamespace(
                        collision="/floor",
                        rigid_body="",
                        position=(*origin[:2], 0.0),
                        normal=(0.0, 0.0, 1.0),
                        distance=0.3,
                    )
                )
            return 2

    result = original_probe.query_floor_support(
        Query(),
        np.array([[0.0, 0.0, 0.75]]),
        np.array([0]),
        0.0,
        {"/floor"},
        {"/floor"},
    )
    np.testing.assert_array_equal(result["arrays"]["all_hit_count"], [2])
    # The sealed v6 receipt predates the single-owner certificate. Preserve
    # its evidence bytes and expose the limitation; it cannot certify a scene.
    assert "single_selected_floor_support_verified" not in result
    assert "selected_floor_support_hit_count" not in result["arrays"]


@pytest.fixture(params=["v7", "v8", "v9", "collected", "current"])
def modern_probe(request, monkeypatch):
    # These versions introduced the explicit buffer-center and padding APIs.
    module = SOURCES[request.param]
    original = Path(module.__file__).read_bytes()
    monkeypatch.setattr(contracts, "probe", module)
    yield module
    assert Path(module.__file__).read_bytes() == original


@pytest.mark.parametrize(
    "contract",
    [
        "test_explicit_official_buffer_centers_mark_absent_API5_without_promoting_API6",
        "test_official_centers_still_require_complete_APIoccupied_and_unique_valid_free_rows",
        "test_lower_onecell_padding_crop_matches_every_worldcenter_with_north_up_rows",
        "test_new_buffer_center_mode_is_explicit_and_default_stays_APIverified",
        "test_known_source_static_nonfloor_upface_blocks_cell_without_becoming_another_floor",
    ],
)
def test_newer_sealed_versions_keep_explicit_native_position_evidence(
    modern_probe, contract
):
    getattr(contracts, contract)()


@pytest.mark.parametrize("source", [None, "best_fit", "guess", True])
def test_newer_sealed_versions_reject_unrecognized_position_source(
    modern_probe, source
):
    contracts.test_unknown_known5_position_sources_reject(source)


@pytest.mark.parametrize(
    "kind", ["two_lower", "fractional", "upper", "smaller", "dimensions"]
)
def test_newer_sealed_versions_reject_unsupported_crop_shift(modern_probe, kind):
    contracts.test_padding_crop_rejects_any_other_resize_shift_or_halfcell_alignment(
        kind
    )


@pytest.mark.parametrize("ground", [0.0, 2.0, -3.0])
def test_every_probe_uses_declared_world_height_for_query_box(original_probe, ground):
    contracts.test_full_height_cell_box_uses_declared_world_height_and_cell_extents(
        ground
    )


@pytest.mark.parametrize(
    "args", [([1.0, float("nan")], 0.0), ([1.0, 2.0], float("nan")), ([1.0, 2.0], True)]
)
def test_every_probe_rejects_invalid_query_box_coordinates(original_probe, args):
    contracts.test_full_height_query_rejects_nonfinite_or_boolean_coordinates(args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": 0.2},
        {"resolution": True},
        {"band": (0.7, 0.8)},
        {"band": (1.8, 0.15)},
        {"band": (0.0, 2.0)},
    ],
)
def test_every_probe_rejects_altered_height_band_or_resolution(original_probe, kwargs):
    contracts.test_full_height_query_rejects_changed_resolution_or_short_height_band(
        kwargs
    )
