"""Versioned mathematical map fixtures; no native v7 execution is claimed."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from CrowdSim.humanoid.runtime.dynbench_scene import (
    DynBenchSceneProvider,
    FLOOR_PATH,
    file_record,
)
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
from CrowdSim.humanoid.runtime.native_map_v7 import PROBE_SHA256
from scripts import probe_paper_scene_maps as official
import test_native_grid_scene as legacy

ROOT = Path(__file__).resolve().parents[1]


def fixture_v7(folder, *, padding=(1, 1), missing=3):
    """Use frozen official CPU formulas, synthetic rays/overlaps and clocks."""
    labels = np.full((20, 24), 254, np.uint8)
    labels[2, 3], labels[10, 12] = 205, 0
    old = legacy.fixture_provider(
        folder, labels=labels, projection_unknown_pixels=((3, 4),)
    )
    data, preflight = deepcopy(old._native["data"]), deepcopy(old._data)
    script = folder / "probe.py"
    shutil.copyfile(ROOT / "docs/style_improvement_20261002/native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py", script)
    assert file_record(script)["sha256"] == PROBE_SHA256
    preflight["script"] = file_record(script)
    preflight["flat_ground_collider_inventory"][1]["reason"] = (
        "Floor mesh is not thin and near selected floor height"
    )
    preflight["reference_ground"]["original_flat_ground_collider_inventory"] = (
        preflight["flat_ground_collider_inventory"]
    )
    preflight["reference_ground_recipe"] = legacy.write_json(
        folder / "recipe.json", preflight["reference_ground"]
    )
    preflight_record = legacy.write_json(folder / "preflight.json", preflight)
    data.update(
        script=preflight["script"],
        preflight=preflight_record,
        CPU_contract_fixture_only=True,
    )
    low = np.array([-5.0 - padding[0] * 0.1, -4.0 - padding[1] * 0.1, 0.15])
    dims = np.array([24 + padding[0], 20 + padding[1], 0], np.int64)
    high = low + np.array([dims[0] * 0.1, dims[1] * 0.1, 1.65])
    native_image = np.pad(
        labels, ((0, padding[1]), (padding[0], 0)), constant_values=254
    )
    buffer = (
        np.where(
            native_image[::-1, ::-1] == 0,
            4,
            np.where(native_image[::-1, ::-1] == 254, 5, 6),
        )
        .astype(np.float32)
        .ravel()
    )

    def centers(pixels):
        pixels = np.asarray(pixels).reshape(-1, 2)
        return np.column_stack(
            (
                low[0] + (pixels[:, 1] + 0.5) * 0.1,
                low[1] + (dims[1] - pixels[:, 0] - 0.5) * 0.1,
                np.full(len(pixels), (low[2] + high[2]) / 2),
            )
        )

    occupied = centers(np.argwhere(native_image == 0))
    all_known = centers(np.argwhere(native_image == 254))
    known_api = all_known[:-missing] if missing else all_known
    api = np.concatenate((known_api, centers(np.argwhere(native_image == 205))))
    reconstructed, coordinates = official.reconstruct_native_map(
        buffer,
        occupied,
        api,
        low,
        high,
        dims,
        known5_position_source="official_buffer_centers",
    )
    canonical = official.canonical_known5_positions(
        buffer, api, low, high, dims, source="official_buffer_centers"
    )
    free = canonical["positions"]
    count = len(free)
    api_indices = official.native_position_buffer_indices(api, low, high, dims)
    occupied_indices = official.native_position_buffer_indices(
        occupied, low, high, dims
    )
    unknown_missing = np.setdiff1d(np.flatnonzero(buffer == 6), api_indices)
    np.savez(
        folder / "native_omap.npz",
        buffer=buffer,
        occupied_positions=occupied,
        free_positions=api,
        dimensions=dims,
        min_bound=low,
        max_bound=high,
    )
    data["native_omap"] = file_record(folder / "native_omap.npz")
    np.savez(
        folder / "canonical.npz",
        occupied_world_positions=occupied,
        canonical_free_world_positions=free,
        canonical_free_API_row=canonical["API_rows"],
        canonical_free_position_source=canonical["position_source"],
        canonical_known5_native_buffer_indices=canonical["native_buffer_indices"],
        missing_known5_buffer_indices=canonical["missing_known5_buffer_indices"],
        missing_known5_world_positions=canonical["missing_known5_world_positions"],
        API_free_buffer_indices=api_indices,
        API_free_buffer_labels=buffer[api_indices],
        API_free_unknown_world_positions=api[buffer[api_indices] == 6],
        unreported_unknown_buffer_indices=unknown_missing,
        unreported_unknown_buffer_labels=buffer[unknown_missing],
    )
    assert len(np.unique(np.r_[api_indices, occupied_indices])) == len(
        api_indices
    ) + len(occupied_indices)
    coordinates["canonical_world_positions_artifact"] = file_record(
        folder / "canonical.npz"
    )
    data["coordinate_validation"] = coordinates
    data["derived_export_grid"] = official.projection_grid_window(
        low, high, dims, preflight["projection_metadata"]["bounds"], np.array([24, 20])
    )
    data.update(
        actual_bounds_world_m=[low.tolist(), high.tolist()],
        actual_dimensions=dims.tolist(),
        native_direct_cell_validation=True,
        known5_API_complete=coordinates["known5_API_complete"],
        known5_position_source="official_buffer_centers",
    )
    # One low static upface below .15m must block its cell even when the
    # full-height band is clear. Every cell still has exactly one floor hit.
    low_row = 11
    positions, normals, paths, classes, offsets = [], [], [], [], [0]
    for index, point in enumerate(free):
        positions.append([*point[:2], 0.0])
        normals.append([0.0, 0.0, 1.0])
        paths.append(FLOOR_PATH)
        classes.append("selected_floor_support")
        if index == low_row:
            positions.append([*point[:2], 0.05])
            normals.append([0.0, 0.0, 1.0])
            paths.append("/Root/Wall")
            classes.append("known_static_nonfloor_near_ground_obstacle")
        offsets.append(len(paths))
    low_mask = np.arange(count) == low_row
    hit_position = free.copy()
    hit_position[:, 2] = 0.0
    np.savez(
        folder / "floor.npz",
        native_free_index=np.arange(count),
        free_world_positions=free,
        supported=np.ones(count, bool),
        hit_position=hit_position,
        hit_normal=np.tile([0.0, 0.0, 1.0], (count, 1)),
        collision=np.full(count, FLOOR_PATH),
        all_hit_count=np.diff(offsets),
        all_hit_offsets=np.asarray(offsets),
        all_hit_collision_paths=np.asarray(paths),
        all_hit_rigid_body_paths=np.asarray(paths),
        all_hit_positions=np.asarray(positions),
        all_hit_normals=np.asarray(normals),
        all_hit_distances=np.full(len(paths), 0.3),
        all_hit_classification=np.asarray(classes),
        undeclared_near_floor_hit=np.zeros(count, bool),
        foreign_hit=np.zeros(count, bool),
        selected_floor_support_hit_count=np.ones(count, np.int64),
        known_static_nonfloor_near_ground_obstacle_hit=low_mask,
    )
    data["floor_support"].update(
        native_free_cells=count,
        tested=count,
        supported=count,
        artifact=file_record(folder / "floor.npz"),
    )
    owner = data["floor_ownership"]
    owner.update(
        artifact=data["floor_support"]["artifact"],
        single_selected_floor_support_verified=True,
        known_static_nonfloor_collision_paths=["/Root/Wall"],
        known_static_nonfloor_near_ground_paths=["/Root/Wall"],
        known_static_nonfloor_near_ground_hit_rows=1,
        near_ground_nonfloor_hit_cells_exported_occupied=True,
        actual_enabled_flat_ground_inventory=[
            {"path": FLOOR_PATH, "near_ground_flat_surface": True, "enabled": True, "static": True},
            deepcopy(preflight["flat_ground_collider_inventory"][1]),
        ],
    )
    data["origin_verification"]["floor_ownership"].update(
        selected_floor_support_hit_count=1,
        single_selected_floor_support_verified=True,
        known_static_nonfloor_near_ground_paths=[],
    )
    blocked = np.arange(count) == 20
    np.savez(
        folder / "band.npz",
        native_free_index=np.arange(count),
        free_world_positions=free,
        full_height_clear=~blocked,
        authored_obstacle_hit=blocked,
        all_hit_paths_classified=np.ones(count, bool),
        selected_floor_hit_count=np.zeros(count, np.int64),
        hit_offsets=np.r_[0, blocked.cumsum()],
        hit_collision_paths=np.full(1, "/Root/Wall"),
        hit_rigid_body_paths=np.full(1, "/Root/Wall"),
        hit_classification=np.full(1, "authored_obstacle"),
    )
    clock = {
        "clock_before": {"submitted_physics_step": 2, "timeline_s": 0.0},
        "clock_after": {"submitted_physics_step": 2, "timeline_s": 0.0},
        "query_time_unchanged": True,
        "foreign_hit_paths": [],
    }
    data["full_height_band_queries"] = {
        **clock,
        "artifact": file_record(folder / "band.npz"),
        "clock_artifact": legacy.write_json(folder / "clock.json", clock),
        "queried_cells": count,
    }
    observation = {
        "submitted_physics_step": 2,
        "counter_python_type": "int",
        "timeline_s": 0.0,
        "timeline_python_type": "float",
        "requested_frozen_timeline_s": 0.0,
    }
    data["actual_query_clock_initial"] = {
        "observation": observation,
        "artifact": legacy.write_json(folder / "initial_clock.json", observation),
    }
    data["physics"].update(
        counter_at_query=2, timeline_at_query_s=0.0, timeline_after_queries_s=0.0
    )
    data["height_band_refinement"].update(
        known5_source=count,
        known5_tested=count,
        height_blocked_known5_cells=1,
        height_clear_known5_cells=count - 1,
        known_static_nonfloor_near_ground_blocked_known5_cells=1,
    )
    cells = np.rint((free[:, :2] - low[:2]) / 0.1 - 0.5).astype(int)
    refined = reconstructed.copy()
    mask = blocked | low_mask
    refined[dims[1] - 1 - cells[mask, 1], cells[mask, 0]] = 0
    export = official.crop_to_projection_grid(refined, data["derived_export_grid"])
    projection = np.asarray(
        Image.open(preflight["input_projection"]["occupancy.png"]["path"])
    )
    export = official.retain_projection_unknown(export, projection)
    for name, image in (
        ("native_reconstructed.png", reconstructed),
        ("height_band_refined.png", refined),
        ("occupancy.png", export),
    ):
        Image.fromarray(image).save(folder / name)
        data["exports"][name] = file_record(folder / name)
    return readmit(old, data), low_row


def readmit(provider, data):
    path = Path(provider._native["record"]["path"])
    native_record = legacy.write_json(path, data)
    fresh = DynBenchSceneProvider.from_preflight(
        provider._preflight_record["path"],
        expected_sha256=file_record(provider._preflight_record["path"])["sha256"],
    )
    fresh.admit_native_map(path, expected_sha256=native_record["sha256"])
    return fresh


def edit_archive(data, group, change):
    record = (
        data[group]["artifact"]
        if group != "canonical"
        else data["coordinate_validation"]["canonical_world_positions_artifact"]
    )
    with np.load(record["path"], allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    change(arrays)
    np.savez(record["path"], **arrays)
    updated = file_record(record["path"])
    if group == "canonical":
        data["coordinate_validation"]["canonical_world_positions_artifact"] = updated
    else:
        data[group]["artifact"] = updated
    if group == "floor_support":
        data["floor_ownership"]["artifact"] = updated


@pytest.mark.parametrize(
    "padding,missing", [((0, 0), 0), ((1, 0), 3), ((0, 1), 3), ((1, 1), 3)]
)
def test_v7_synthetic_native_contract_with_official_crop_and_missing_API(
    tmp_path, padding, missing
):
    provider, _ = fixture_v7(tmp_path, padding=padding, missing=missing)
    grid = NativeGridScene.from_provider(
        provider,
        origins_xy=[[-4.45, -3.45]],
        routes_xy=[[[-4.45, -3.45], [-3.45, -2.45]]],
        clearance_m=0.0,
    )
    assert grid._admission["admission_protocol"] == "v7_exact_probe_SHA"
    assert grid._admission["missing_known5_API_count"] == missing
    assert grid._admission["known5_API_complete"] is (missing == 0)
    assert (
        grid._admission["known_static_nonfloor_near_ground_blocked_known5_cells"] == 1
    )
    assert grid._labels.shape == (20, 24) and not grid.is_walkable([[-1e3, -1e3]])[0]
    assert len(grid.astar_path([-4.45, -3.45], [-3.45, -2.45])) > 1


@pytest.mark.parametrize(
    "corruption",
    [
        "failed_status",
        "direct_validation",
        "crop",
        "API_claim",
        "missing_source",
        "canonical_API_row",
        "canonical_unknown",
        "registry",
        "low_mask",
        "floor_count",
        "foreign_hit",
        "nonfinite_hit",
        "origin_duplicate",
        "clock",
        "low_export_free",
    ],
)
def test_v7_rejects_source_crop_floor_unknown_or_CSR_corruption(tmp_path, corruption):
    provider, _ = fixture_v7(tmp_path)
    data = deepcopy(provider._native["data"])
    if corruption == "failed_status":
        data["status"] = "failed"
    elif corruption == "direct_validation":
        data["native_direct_cell_validation"] = False
    elif corruption == "crop":
        data["derived_export_grid"]["native_window_columns"][0] = 0
    elif corruption == "API_claim":
        data["known5_API_complete"] = True
    elif corruption == "missing_source":
        data["known5_position_source"] = "api_verified"
    elif corruption == "canonical_API_row":
        edit_archive(
            data, "canonical", lambda a: a["canonical_free_API_row"].__setitem__(-1, 0)
        )
    elif corruption == "canonical_unknown":
        edit_archive(
            data,
            "canonical",
            lambda a: a["canonical_known5_native_buffer_indices"].__setitem__(-1, 0),
        )
    elif corruption == "registry":
        data["floor_ownership"]["known_static_nonfloor_collision_paths"] = []
    elif corruption == "low_mask":
        edit_archive(
            data,
            "floor_support",
            lambda a: a["known_static_nonfloor_near_ground_obstacle_hit"].fill(False),
        )
    elif corruption == "floor_count":
        edit_archive(
            data,
            "floor_support",
            lambda a: a["selected_floor_support_hit_count"].__setitem__(0, 2),
        )
    elif corruption == "foreign_hit":
        edit_archive(
            data,
            "floor_support",
            lambda a: a["all_hit_collision_paths"].__setitem__(0, "/Foreign/Floor"),
        )
    elif corruption == "nonfinite_hit":
        edit_archive(
            data,
            "floor_support",
            lambda a: a["all_hit_positions"].__setitem__((0, 0), np.nan),
        )
    elif corruption == "origin_duplicate":
        data["origin_verification"]["floor_ownership"][
            "selected_floor_support_hit_count"
        ] = 2
    elif corruption == "clock":
        data["full_height_band_queries"]["clock_after"]["submitted_physics_step"] = 3
    else:
        path = Path(data["exports"]["height_band_refined.png"]["path"])
        image = np.asarray(Image.open(path)).copy()
        image[image == 0] = 254
        Image.fromarray(image).save(path)
        data["exports"]["height_band_refined.png"] = file_record(path)
    with pytest.raises(ValueError):
        readmit(provider, data)


def _before_module(name, filename):
    path = (
        ROOT
        / "docs/style_improvement_20261002/native_map_admission_v7/before/CrowdSim/humanoid/runtime"
        / filename
    )
    spec = importlib.util.spec_from_file_location(
        "CrowdSim.humanoid.runtime." + name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_description_grid_astar_and_SDF_value_Jacobian_remain_bytewise(tmp_path):
    provider = legacy.fixture_provider(tmp_path)
    before_provider = _before_module("before_v7_provider", "dynbench_scene.py")
    old = before_provider.DynBenchSceneProvider.from_preflight(
        provider._preflight_record["path"],
        expected_sha256=provider._preflight_record["sha256"],
    )
    old_description = old.admit_native_map(
        provider._native["record"]["path"],
        expected_sha256=provider._native["record"]["sha256"],
    )
    current_description = provider.describe()
    # The maintained provider adds explicit feature/portable/contact metadata;
    # numerical map, path and Jacobian contracts remain byte-exact below.
    added = {"policy_height_mode", "policy_height_padding_claims_walkable_support",
             "portable_USD_location_derivation", "static_target_contact_report_paths",
             "static_target_contact_report_threshold_n"}
    assert set(current_description) - set(old_description) == added
    assert current_description["policy_height_mode"] == "native_obstacle_cap"
    assert current_description["policy_height_padding_claims_walkable_support"] is False
    assert current_description["portable_USD_location_derivation"] is None
    assert current_description["static_target_contact_report_paths"] == []
    assert current_description["static_target_contact_report_threshold_n"] == 0.
    for description in (old_description, current_description):
        assert "floor+1.8m cap" in description["policy_height_feature"]
        assert "no exterior support claim" in description["policy_height_feature"]
    assert {key: value for key, value in old_description.items() if key != "policy_height_feature"} == {
        key: value for key, value in current_description.items()
        if key not in added and key != "policy_height_feature"}
    before_grid = _before_module("before_v7_grid", "native_grid_scene.py")
    kwargs = dict(
        origins_xy=[[-3.45, -0.05]],
        routes_xy=[[[-3.45, -0.05], [2.55, -0.05]]],
        clearance_m=0.35,
    )
    current, original = (
        NativeGridScene.from_provider(provider, **kwargs),
        before_grid.NativeGridScene.from_provider(provider, **kwargs),
    )
    before_description, maintained_description = original.describe(), current.describe()
    added_grid_defaults = {"cyclic_route_lookahead": False, "start_connector_radius_m": 0.,
        "start_connector_clearance_m": .4, "route_clearance_weight": 0.,
        "route_comfort_clearance_m": .6}
    assert set(maintained_description) - set(before_description) == set(added_grid_defaults)
    assert {key: maintained_description[key] for key in added_grid_defaults} == added_grid_defaults
    assert "no diagonal corner cutting" in maintained_description["astar"]
    assert {key: value for key, value in before_description.items() if key != "astar"} == {
        key: value for key, value in maintained_description.items()
        if key not in added_grid_defaults and key != "astar"}
    assert np.array_equal(
        current.astar_path([-3.45, -0.05], [2.55, -0.05]),
        original.astar_path([-3.45, -0.05], [2.55, -0.05]),
    )
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor(
            [[-3.32, -0.03], [-5.01, -4.02], [5.0, 5.0]],
            dtype=dtype,
            requires_grad=True,
        )
        y = x.detach().clone().requires_grad_(True)
        values, old_values = current.signed_distance(x), original.signed_distance(y)
        assert torch.equal(values, old_values)
        assert torch.equal(
            torch.autograd.grad(values.sum(), x)[0],
            torch.autograd.grad(old_values.sum(), y)[0],
        )


def test_v7_fields_do_not_select_new_protocol_without_exact_probe_bytes(tmp_path):
    provider, _ = fixture_v7(tmp_path)
    data = deepcopy(provider._native["data"])
    preflight = deepcopy(provider._data)
    script = Path(preflight["script"]["path"])
    script.write_text("# unsupported probe revision; mathematical fixture only\n")
    preflight["script"] = file_record(script)
    data["script"] = preflight["script"]
    data["preflight"] = legacy.write_json(Path(provider._preflight_record["path"]), preflight)
    with pytest.raises(ValueError, match="coplanar|single floor|low static obstacle"):
        readmit(provider, data)


def test_v7_selected_support_cannot_move_to_a_different_grid_cell(tmp_path):
    provider, _ = fixture_v7(tmp_path)
    data = deepcopy(provider._native["data"])

    def relocate(arrays):
        arrays["all_hit_positions"][0, 0] += .1
        arrays["hit_position"][0, 0] += .1

    edit_archive(data, "floor_support", relocate)
    with pytest.raises(ValueError, match="per-cell"):
        readmit(provider, data)


def test_v7_grid_rechecks_completed_status_after_private_snapshot_tampering(tmp_path):
    provider, _ = fixture_v7(tmp_path)
    legacy.mutate_native(provider, lambda data: data.__setitem__("status", "failed"))
    with pytest.raises(ValueError, match="completed matching"):
        NativeGridScene.from_provider(provider, origins_xy=[[-4.45, -3.45]], routes_xy=[[[-4.45, -3.45], [-3.45, -2.45]]], clearance_m=0.)
