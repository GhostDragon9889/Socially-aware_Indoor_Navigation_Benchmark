"""Explicit synthetic CPU contracts, never evidence of native SDK capture."""
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.analysis import native_paired_initial as initial
from scripts import evaluate_shared_social_native as base_entry
from scripts import evaluate_shared_social_native_paired as entry


def fixture(actors):
    core, proof = initial.count_backend(actors)
    state = SimpleNamespace(state_conversion=SimpleNamespace(value="common"))
    for name, shape in core.COMMON_FIELDS.items():
        value = np.zeros(shape, np.float32)
        if name == "rigid_body_rot":
            value[..., 3] = 1
        setattr(state, name, value)
    provider = object()
    origins = np.tile([-.3, 1.6], (actors, 1))
    routes = np.tile([[[-.3, 1.6], [.3, 2.0]]], (actors, 1, 1))
    scene = SimpleNamespace(provider=provider, routes=routes.astype(np.float32), _routes=routes,
        _origins=origins, _labels=np.zeros((6, 8), np.uint8), _origin=np.array([-.4, 1.5], np.float64),
        _bounds=np.array([-.4, 1.5, .4, 2.1], np.float64), _signed_grid=np.ones((6, 8), np.float32))
    world = SimpleNamespace(num_envs=actors, shared_world=True, scene_provider=provider,
                            validation_scene=scene, state=lambda: state)
    for name, shape in core.WORLD_FIELDS.items():
        dtype = np.bool_ if name == "history_available" else (
            np.int64 if name in {"age", "episode_ids", "reference_ids", "amp_history_episode_ids"} else np.float32)
        setattr(world, name, np.zeros(shape, dtype))
    world.episode_ids.fill(1)
    world.history_available[:, -1] = True
    world.origins[:] = origins.astype(np.float32)
    native = {key: np.zeros(shape, np.float32) for key, shape in core.NATIVE_FIELDS.items()}
    for common, key in (("rigid_body_pos", "body_link_pos_w"), ("rigid_body_rot", "body_link_quat_w"),
                        ("rigid_body_vel", "body_link_lin_vel_w"), ("rigid_body_ang_vel", "body_link_ang_vel_w")):
        native[key] = getattr(state, common).copy()
    ground = SimpleNamespace(world=world, counter=lambda: 17, _state_snapshot=lambda: native,
                             native_names=["Pelvis"] + [str(i) for i in range(23)],
                             common_names=["Pelvis"] + [str(i) for i in range(23)])
    return core, proof, world, ground, scene


def complete_snapshot(actors):
    core, proof, world, ground, scene = fixture(actors)
    arrays, clock = core.collect_snapshot(world, ground)
    rng, metadata = core.rng_snapshot(torch.Generator().manual_seed(7))
    arrays.update(rng)
    arrays.update(initial.map_arrays(scene))
    contract = {"capture_kind": "CPU_contract_fixture", "clock": clock, "rng": metadata,
                "boundary": "synthetic; not live SDK or actual-exit evidence",
                "native_protocol": initial.native_protocol(actors, scene), "native_count_derivation": proof}
    return arrays, contract, scene


@pytest.mark.parametrize("actors", [3, 15])
def test_count_core_and_map_copy_preserve_clock_RNG_and_detached_payload(actors):
    core, proof, world, ground, scene = fixture(actors)
    generator = torch.Generator().manual_seed(7)
    before, metadata = core.rng_snapshot(generator)
    arrays, clock = core.collect_snapshot(world, ground)
    maps = initial.map_arrays(scene)
    after, other = core.rng_snapshot(generator)
    assert initial.original._equal_arrays(before, after) and metadata == other
    assert not metadata["CUDA_initialized"] and clock["physics_step_before"] == clock["physics_step_after"] == 17
    assert arrays["world_amp_history"].shape == (actors, 10, 195)
    world.state().rigid_body_pos[:] = 5
    scene._labels[:] = 254
    assert not arrays["common_rigid_body_pos"].any() and not maps["native_scene_labels"].any()
    assert proof["literal_seams"] == 9 and proof["literal_occurrences"] == 13
    assert proof["exact_inverse"] and proof["unchanged_helpers_AST_equal"]
    assert not proof["copy_RNG_archive_match_math_changed"]


def test_count_modules_do_not_mutate_the_fixed_fixture_core_or_each_other():
    three, _ = initial.count_backend(3)
    fifteen, _ = initial.count_backend(15)
    three.COMMON_FIELDS["rigid_body_pos"] = (99, 24, 3)
    assert fifteen.COMMON_FIELDS["rigid_body_pos"] == (15, 24, 3)
    assert initial.original.COMMON_FIELDS["rigid_body_pos"] == (4, 24, 3)
    assert initial.original.SCHEMA != initial.SCHEMA
    with pytest.raises(ValueError, match="exact fixed N4"):
        initial.derive_count_core(initial.PARENT.read_text() + "\n")


@pytest.mark.parametrize("actors", [2, 16, True, 3.0])
def test_actor_count_is_explicit_and_not_coerced(actors):
    with pytest.raises(ValueError, match="explicit N3..15"):
        initial.count_backend(actors)


@pytest.mark.parametrize("defect,pattern", [
    ("clock", "advanced physics"), ("native", "disagree"), ("age", "age0"),
    ("foreign_provider", "complete shared"), ("foreign_capture", "complete shared"),
    ("nonfinite", "nonfinite"),
])
def test_live_collector_core_rejects_changed_initial_state_and_owners(defect, pattern):
    core, _, world, ground, _ = fixture(15)
    if defect == "clock":
        ground.counter = Mock(side_effect=[17, 18])
    elif defect == "native":
        ground._state_snapshot()["body_link_pos_w"][0, 0, 0] = .01
    elif defect == "age":
        world.age[0] = 1
    elif defect == "foreign_provider":
        world.scene_provider = object()
    elif defect == "foreign_capture":
        ground.world = object()
    else:
        world.state().dof_vel[0, 0] = np.nan
    with pytest.raises(ValueError, match=pattern):
        core.collect_snapshot(world, ground)


@pytest.mark.parametrize("defect,pattern", [
    ("partial", "partial"), ("actor_dtype", "integers"), ("map_dtype", "dtype/bytes"),
    ("map_bytes", "dtype/bytes"), ("bad_labels", "labels/frame"), ("frame", "labels/frame"),
    ("layout", "explicit native"), ("resolution", "explicit native"),
])
def test_native_channels_reject_missing_wrong_dtype_map_or_coordinates(defect, pattern):
    arrays, contract, _ = complete_snapshot(3)
    if defect == "partial":
        arrays.pop("world_reference_ids")
    elif defect == "actor_dtype":
        arrays["world_episode_ids"] = arrays["world_episode_ids"].astype(np.float32)
    elif defect == "map_dtype":
        arrays["native_scene_labels"] = arrays["native_scene_labels"].astype(np.int64)
    elif defect == "map_bytes":
        arrays["native_scene_labels"][0, 0] = 254
    elif defect == "bad_labels":
        arrays["native_scene_labels"][0, 0] = 1
    elif defect == "frame":
        arrays["native_scene_bounds_xy"][2] += .1
    elif defect == "layout":
        contract["native_protocol"]["layout"] = "flipped row order"
    else:
        contract["native_protocol"]["resolution_m"] = 1
    if defect in {"bad_labels", "frame"}:
        key = "native_scene_labels" if defect == "bad_labels" else "native_scene_bounds_xy"
        contract["native_protocol"]["map_channels"][key] = initial.original._description(arrays[key])
    with pytest.raises(ValueError, match=pattern):
        initial.validate_channels(arrays, contract)


@pytest.mark.parametrize("actors", [3, 15])
def test_complete_CPU_archive_requires_explicit_fixture_opt_in_and_keeps_exclusive_files(actors, tmp_path):
    arrays, contract, _ = complete_snapshot(actors)
    left, right = tmp_path/"left", tmp_path/"right"
    left.mkdir()
    right.mkdir()
    initial.publish_snapshot(left, arrays, contract)
    initial.publish_snapshot(right, arrays, contract)
    paths = left/"paired_initial_condition.json", right/"paired_initial_condition.json"
    with pytest.raises(ValueError, match="CPU fixtures"):
        initial.compare_initial_conditions(*paths)
    report = initial.compare_initial_conditions(*paths, require_live=False)
    assert report["actors"] == actors and report["channel_count"] == 35
    assert report["same_initial_body_map_and_RNG_bytes"]
    assert report["capture_kind"] == "CPU_contract_fixture"
    assert not report["actual_process_exit_verified"] and not report["navigation_quality_established"]
    with pytest.raises(FileExistsError):
        initial.publish_snapshot(left, arrays, contract)
    changed = tmp_path/"changed"
    changed.mkdir()
    arrays["rng_corpus_CPU"][0] ^= 1
    initial.publish_snapshot(changed, arrays, contract)
    with pytest.raises(ValueError, match="do not share exact"):
        initial.compare_initial_conditions(paths[0], changed/"paired_initial_condition.json", require_live=False)


def test_archive_tampering_or_old_schema_is_rejected_before_match(tmp_path):
    arrays, contract, _ = complete_snapshot(3)
    initial.publish_snapshot(tmp_path, arrays, contract)
    path = tmp_path/"paired_initial_condition.json"
    value = json.loads(path.read_text())
    value["arrays"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity differs"):
        initial.read_snapshot(path)
    value["schema"] = initial.original.SCHEMA
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="native paired initial schema"):
        initial.read_snapshot(path)


def live_validation_fixture():
    """In-memory declared contract only; never publish as a live archive."""
    arrays, contract, scene = complete_snapshot(3)
    sources = initial.source_bindings()
    fixture_record = sources["core"]
    contract.update(capture_kind=initial.LIVE_KIND, native_sources=sources,
        initial_capture_source=sources["native_reader"], native_capture_source=sources["foot_capture"],
        native_resources={"actors": 3, "configuration_file": fixture_record},
        arguments_excluding_output={}, actor={}, physics={}, reference={}, runtime_numerics={},
        rng_preparation={}, scene={}, resources={"scene": fixture_record, "manifest": fixture_record,
            "profile": fixture_record, "paired_diffusion_binding": fixture_record})
    return arrays, contract, scene


@pytest.mark.parametrize("defect,pattern", [
    ("source", "source bindings"), ("capture_source", "source bindings"),
    ("scene_source", "count/scene"), ("count", "count/scene"), ("proof", "count/scene"),
    ("origins", "origins/routes"), ("routes", "origins/routes"), ("grid", "actual admitted grid"),
])
def test_live_contract_validator_requires_actual_selected_source_frame_and_routes(defect, pattern):
    arrays, contract, scene = live_validation_fixture()
    initial.validate_channels(arrays, contract, expected_scene=scene)
    if defect == "source":
        contract["native_sources"]["core"]["sha256"] = "0" * 64
    elif defect == "capture_source":
        contract["initial_capture_source"] = contract["native_sources"]["core"]
    elif defect == "scene_source":
        contract["native_resources"]["configuration_file"] = {"path": "/wrong", "sha256": "0" * 64}
    elif defect == "count":
        contract["native_resources"]["actors"] = 15
    elif defect == "proof":
        contract["native_count_derivation"]["exact_inverse"] = False
    elif defect == "origins":
        arrays["world_origins"][0, 0] += .1
    elif defect == "routes":
        arrays["scene_routes_xy"][0, 0, 0] += .1
    else:
        scene._signed_grid[0, 0] += .1
    with pytest.raises(ValueError, match=pattern):
        initial.validate_channels(arrays, contract, expected_scene=scene)


@pytest.mark.parametrize("actors,planner", [(3, "diffusion"), (15, "diffusion"), (3, "sfm"), (15, "sfm")])
def test_native_hooks_precede_each_planner_and_preserve_existing_control(actors, planner):
    base, _ = base_entry.prepare_entry(actors=actors, planner=planner)
    parent, binding = entry.prepare_entry(actors=actors, planner=planner)
    original = parent.DRIVER.read_text()
    before, _ = base._derive(original)
    source, proof = parent._derive(original)
    assert proof["complete_inverse"] and proof["existing_helpers_AST_equal"]
    assert not proof["existing_control_or_metric_math_changed"]
    assert binding["native_initial_capture"]["actors"] == actors
    assert not binding["native_initial_capture"]["actual_native_SDK_capture_verified"]
    hook = '_paired_initial(world, ground_capture, output, report, args, provenance)'
    assert source.count(hook) == 1
    assert source.index(hook) < source.index("        planner = ")
    for call in ("world.step(", "world.reset(", "world.planner_inputs("):
        assert source.count(call) == before.count(call)
    planner_class = "NativeSocialForcePlanner" if planner == "sfm" else "PaperDiffusionPlanner"
    assert source.count("        planner = " + planner_class) == (1 if planner == "sfm" else 0)
    assert planner_class in source
    marker = ("        planner = (PaperDiffusionPlanner" if planner == "diffusion" else
              "        if curriculum is not None or not args.evaluate_steps")
    with pytest.raises(ValueError, match="Ambiguous"):
        entry.derive_driver(original, lambda _: (before.replace(marker, "        changed = "), {}), planner=planner)


@pytest.mark.parametrize("actors,planner", [(3, "diffusion"), (15, "sfm")])
def test_fresh_native_paired_source_CLI_does_not_import_torch_or_start_SDK(actors, planner):
    result = subprocess.run([sys.executable, str(entry.ROOT/"scripts/evaluate_shared_social_native_paired.py"),
        "--actors", str(actors), "--planner", planner, "--check-sources"], cwd=entry.ROOT,
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["Torch_imported"] is False and report["SDK_started"] is False
    assert report["physical_derivation"]["complete_inverse"]
    assert report["migration"]["native_initial_capture"]["actors"] == actors


def test_formal_comparison_CLI_rejects_CPU_fixture_and_creates_no_report(tmp_path):
    arrays, contract, _ = complete_snapshot(3)
    initial.publish_snapshot(tmp_path, arrays, contract)
    path = tmp_path/"paired_initial_condition.json"
    target = tmp_path/"uncreated"/"comparison.json"
    result = subprocess.run([sys.executable, str(entry.ROOT/"scripts/compare_shared_social_native_initial_conditions.py"),
        "--left", str(path), "--right", str(path), "--output", str(target)], cwd=entry.ROOT,
        text=True, capture_output=True)
    assert result.returncode != 0 and "CPU fixtures" in result.stderr
    assert not target.parent.exists()


def test_current_native_source_change_is_rejected(monkeypatch):
    monkeypatch.setattr(initial, "PARENT_SHA", "0" * 64)
    with pytest.raises(ValueError, match="core changed"):
        initial.count_backend(3)
