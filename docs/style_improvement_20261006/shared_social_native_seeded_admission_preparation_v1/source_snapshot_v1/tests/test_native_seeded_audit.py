"""CPU contracts only; no native SDK exit, capture or quality is fabricated."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.analysis import native_navigation_seeded_audit as raw
from scripts import audit_shared_social_native_seeded as cli
from test_native_navigation_audit import declaration_fixture, raw_fixture


def seeded_declaration(*, actors=3, controls=30, planner="diffusion", seed=19):
    receipt, result, request, snapshot = declaration_fixture(actors, controls, planner)
    receipt.update(schema=raw.EXECUTION_SCHEMA, seed=seed)
    result["arguments"]["seed"] = seed
    snapshot["contract"]["arguments_excluding_output"]["seed"] = seed
    migration = result["social_entry"]["native_current_migration"]
    _, expected = raw.entry.prepare_entry(actors=actors, planner=planner, seed=seed)
    migration["entry"] = raw.original.record(raw.ENTRY)
    migration["seed7_paired_entry"] = expected["seed7_paired_entry"]
    migration["registered_native_seed"] = dict(expected["registered_native_seed"],
                                             outer_CLI_derivation=raw.entry.outer_CLI_source()[1])
    return receipt, result, request, snapshot


@pytest.mark.parametrize("actors,controls,planner", ((3, 30, "diffusion"), (15, 6000, "sfm")))
@pytest.mark.parametrize("seed", (7, 19, 42))
def test_seeded_declarations_preserve_old_checks_and_do_not_claim_exit(actors, controls, planner, seed):
    values = seeded_declaration(actors=actors, controls=controls, planner=planner, seed=seed)
    before = copy.deepcopy(values)
    admitted = raw.validate_result_contract(*values, actors=actors, controls=controls, planner=planner, seed=seed)
    assert admitted["registered_seed"] == seed and admitted["receipt_driver_initial_and_entry_seed_equal"]
    assert not admitted["actual_process_exit_independently_observed_by_reader"]
    assert not admitted["paper_effect_achieved"] and values == before


@pytest.mark.parametrize("role", ("receipt", "driver", "initial", "registration"))
@pytest.mark.parametrize("value", (None, True, 19., 7, 42))
def test_every_seed_boundary_rejects_missing_coerced_or_other_seed(role, value):
    receipt, result, request, snapshot = seeded_declaration()
    selected = {"receipt": receipt, "driver": result["arguments"],
        "initial": snapshot["contract"]["arguments_excluding_output"],
        "registration": result["social_entry"]["native_current_migration"]["registered_native_seed"]}[role]
    selected["seed"] = value
    with pytest.raises(ValueError, match="Registered native seed"):
        raw.validate_result_contract(receipt, result, request, snapshot,
                                    actors=3, controls=30, planner="diffusion", seed=19)


@pytest.mark.parametrize("defect", ("entry_source", "inner_proof", "outer_proof", "extra_reseed", "old_entry", "old_parent"))
def test_seed_proofs_cannot_relax_source_or_reseed_boundaries(defect):
    values = seeded_declaration()
    migration = values[1]["social_entry"]["native_current_migration"]
    registered = migration["registered_native_seed"]
    if defect == "entry_source":
        registered["sources"]["seeded_entry"]["sha256"] = "0" * 64
    elif defect in ("inner_proof", "outer_proof"):
        registered["main_derivation" if defect == "inner_proof" else "outer_CLI_derivation"]["complete_inverse"] = False
    elif defect == "extra_reseed":
        registered["extra_reseed_in_capture_hook"] = True
    elif defect == "old_entry":
        migration["entry"] = raw.original.record(raw.frozen.ENTRY)
    else:
        migration["seed7_paired_entry"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sources or interface proofs|source bindings"):
        raw.validate_result_contract(*values, actors=3, controls=30, planner="diffusion", seed=19)


@pytest.mark.parametrize("defect", ("views", "stage", "body", "controls", "exit"))
def test_new_seed_layer_keeps_existing_native_refusals(defect):
    values = seeded_declaration()
    receipt, result, _, snapshot = values
    if defect == "views":
        result["navigation_owned_contact_views"]["live_before_app_shutdown"] = 1
    elif defect == "stage":
        result["native_scene_live_observation"]["query_physics_advance"] = 1
    elif defect == "body":
        snapshot["contract"]["clock"]["common_body_names"] = ["wrong"] * 24
    elif defect == "controls":
        result["actual_control_steps"] = 6000
    else:
        receipt["actual_exit_code"] = False
    with pytest.raises(ValueError):
        raw.validate_result_contract(*values, actors=3, controls=30, planner="diffusion", seed=19)


@pytest.mark.parametrize("receipt", (
    {"schema": raw.frozen.EXECUTION_SCHEMA, "actual_exit_code": 0, "terminal_tool_chunk": "CPU_ONLY", "seed": 19},
    {"schema": raw.EXECUTION_SCHEMA, "actual_exit_code": False, "terminal_tool_chunk": "CPU_ONLY", "seed": 19},
    {"schema": raw.EXECUTION_SCHEMA, "actual_exit_code": 0, "terminal_tool_chunk": "CPU_ONLY", "seed": 7},
    {"schema": raw.EXECUTION_SCHEMA, "actual_exit_code": 0, "terminal_tool_chunk": "CPU_ONLY", "seed": 19.},
))
def test_receipt_is_refused_before_any_missing_native_resource(receipt, tmp_path):
    path = tmp_path / "CPU_FIXTURE_ONLY.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="successful seeded native SDK exit|Registered native seed"):
        raw.admit_execution(path, actors=3, controls=30, planner="diffusion", seed=19,
                            scene_config=tmp_path / "MISSING", scene_sha256="0" * 64)


def test_frozen_seed7_reader_does_not_silently_accept_the_new_entry():
    values = seeded_declaration()
    values[0]["schema"] = raw.frozen.EXECUTION_SCHEMA
    with pytest.raises(ValueError, match="source bindings"):
        raw.frozen.validate_result_contract(*values, actors=3, controls=30, planner="diffusion")


@pytest.mark.parametrize("actors,controls", ((3, 30), (15, 6000)))
@pytest.mark.parametrize("seed", (7, 19, 42))
def test_all_six_measurement_kernels_and_their_derivation_proofs_remain_exact(actors, controls, seed):
    before = raw.frozen.source_bindings()
    old = raw.frozen.prepare_kernels(actors=actors, controls=controls)
    new = raw.prepare_kernels(actors=actors, controls=controls, seed=seed)
    assert new.derivation["kernels"] == old.derivation["kernels"]
    for name in ("joint", "body", "initial_link", "analyze"):
        assert getattr(new, name).__code__.co_code == getattr(old, name).__code__.co_code
    assert new.base.main.__code__.co_code == old.base.main.__code__.co_code
    assert new.base._native_execution_schema == raw.EXECUTION_SCHEMA
    assert old.base._native_execution_schema == raw.frozen.EXECUTION_SCHEMA
    assert new.derivation["registered_seed"] == seed
    assert raw.frozen.source_bindings() == before and raw.frozen.ENTRY != raw.ENTRY


@pytest.mark.parametrize("actors", (3, 15))
def test_actual_CPU_kernel_body_plan_response_values_equal_seed7(actors, tmp_path):
    state, _, plans, result, _ = raw_fixture(actors, 30, tmp_path)
    old = raw.frozen.prepare_kernels(actors=actors, controls=30)
    new = raw.prepare_kernels(actors=actors, controls=30, seed=42)
    paths = [raw.original.record(tmp_path / name) for name in
             ("physical_states.npz", "ground_support.npz", "reference_plans.npz")]
    base = {"fall_termination_actor_controls": 1, "route_vertices_reached": result["route_vertices_reached"],
            "same_episode_position_intervals": 29 * actors - 1}
    assert old.body(tmp_path / "physical_states.npz", tmp_path / "ground_support.npz", 30, result) == \
        new.body(tmp_path / "physical_states.npz", tmp_path / "ground_support.npz", 30, result)
    assert old.joint(plans, 30) == new.joint(plans, 30)
    assert old.analyze(result, base, paths, old.helpers) == new.analyze(result, base, paths, new.helpers)
    for key in ("summary",):
        assert getattr(new.helpers[0], key) is getattr(old.helpers[0], key) or \
            getattr(new.helpers[0], key).__code__.co_code == getattr(old.helpers[0], key).__code__.co_code
    old_targets = old.helpers[0].targets_from_saved_plans(state, plans)
    new_targets = new.helpers[0].targets_from_saved_plans(state, plans)
    assert old_targets[0] == new_targets[0]
    np.testing.assert_array_equal(old_targets[1], new_targets[1])


@pytest.mark.parametrize("actors,controls,planner,seed", ((3, 30, "diffusion", 19), (15, 6000, "sfm", 42)))
def test_fresh_seeded_raw_CLI_forwards_seed_without_Torch_or_SDK(actors, controls, planner, seed):
    child = subprocess.run([sys.executable, str(cli.__file__), "--actors", str(actors),
        "--control-steps", str(controls), "--planner", planner, "--seed", str(seed), "--check-sources"],
        cwd=raw.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    assert report["seed"] == seed and report["derivation"]["registered_seed"] == seed
    assert not report["Torch_imported"] and not report["SDK_started"] and not report["actual_native_capture_audited"]


def test_new_raw_CLI_requires_explicit_seed_and_exact_source_seams():
    _, proof = cli.prepared_CLI()
    assert proof["complete_source_inverse"] and proof["literal_seams"] == 4
    child = subprocess.run([sys.executable, str(cli.__file__), "--actors", "3", "--control-steps", "30",
        "--planner", "diffusion", "--check-sources"], cwd=raw.ROOT, capture_output=True, text=True)
    assert child.returncode == 2 and "--seed" in child.stderr
