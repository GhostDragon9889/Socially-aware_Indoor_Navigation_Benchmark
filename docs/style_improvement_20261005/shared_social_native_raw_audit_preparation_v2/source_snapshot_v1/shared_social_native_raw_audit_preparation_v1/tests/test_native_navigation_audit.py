"""Synthetic CPU contracts only; no SDK captures, process exits or quality proof."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.analysis import native_navigation_audit as native


def raw_fixture(actors, controls, folder):
    """Known linear motion, one fall/reset, one static and one directed peer flag."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(1, controls + 1, dtype=np.float64) / 30
    root = np.zeros((controls, actors, 3), np.float32)
    root[..., 0], root[..., 1], root[..., 2] = time[:, None], np.arange(actors)[None] * .5, 1
    quaternion = np.zeros((controls, actors, 4), np.float32)
    quaternion[..., 3] = 1
    state = {"root_pos": root, "root_rot": quaternion,
        "body_pos": np.repeat(root[:, :, None], 24, axis=2),
        **{key: root[..., :2].copy() for key in ("target_xy", "raw_target_xy")},
        "reference_target_velocity_xy": np.tile(np.array([1, 0], np.float32), (controls, actors, 1)),
        "reference_deviation_m": np.zeros((controls, actors), np.float32),
        "episode_ids": np.ones((controls, actors), np.int64),
        **{key: np.zeros((controls, actors), bool) for key in
           ("reset_mask", "terminated", "truncated", "static_contact", "route_vertex_reached")},
        "peer_contact": np.zeros((controls, actors, actors), bool),
        "static_contact_max_force_n": np.zeros((controls, actors), np.float32),
        "peer_contact_max_force_n": np.zeros((controls, actors, actors), np.float32),
        **{key: np.zeros((controls, actors, 69), np.float32) for key in ("dof_pos", "dof_vel", "dof_forces")}}
    state["reset_mask"][0] = True
    state["terminated"][6, -1] = True
    state["reset_mask"][7, -1] = True
    state["episode_ids"][7:, -1] = 2
    state["static_contact"][10, -1] = True
    state["static_contact_max_force_n"][10, -1] = 2
    state["peer_contact"][12, 0, -1] = True
    state["peer_contact_max_force_n"][12, 0, -1] = 2
    state["route_vertex_reached"][-1, 0] = True
    names = ["Pelvis"] + [f"CPU_fixture_body_{index}" for index in range(1, 24)]
    body_quaternion = np.zeros((controls, 4, actors, 24, 4), np.float32)
    body_quaternion[..., 3] = 1
    velocity = np.zeros((controls, 4, actors, 24, 3), np.float32)
    velocity[..., 0] = 1
    force = np.zeros((controls, 4, actors, 4, 1, 3), np.float32)
    force[..., 2] = 2
    ground = {"common_body_names": np.array(names),
        "body_link_pos_w": np.repeat(state["body_pos"][:, None], 4, axis=1),
        "body_link_quat_w": body_quaternion, "body_link_lin_vel_w": velocity,
        "control_step": np.arange(controls, dtype=np.int64), "control_step_after": np.arange(1, controls + 1),
        "physics_step_before": 17 + np.arange(controls) * 4,
        "physics_step_end": 17 + np.arange(1, controls + 1) * 4,
        "physics_step": 17 + np.arange(1, controls * 4 + 1).reshape(controls, 4),
        "substep_valid_count": np.full(controls, 4, np.int64),
        "actor_substep_valid_count": np.full((controls, actors), 4, np.int64), "valid": np.ones(controls, bool),
        **{key: state[key].copy() for key in ("episode_ids", "reset_mask", "terminated", "truncated")},
        "foot_names": np.array(["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]),
        "foot_floor_normal_force_w": force, "foot_contact_substeps": np.ones((controls, 4, actors, 4), bool),
        "side_contact_substeps": np.ones((controls, 4, actors, 2), bool),
        "body_origin_tangential_speed_proxy_mps": np.ones((controls, 4, actors, 4), np.float32),
        "no_support_fraction": np.zeros((controls, actors), np.float32),
        "double_support_fraction": np.ones((controls, actors), np.float32)}
    events = np.arange(0, controls, 15, dtype=np.int64)
    actor_ids = np.tile(np.arange(actors, dtype=np.int64), len(events))
    event_rows = np.repeat(events, actors)
    plans = {"actor_ids": actor_ids, "control_step": event_rows}
    for key, knots, hz in (("raw_xy", 51, 10), ("reference_xy", 151, 30)):
        value = np.empty((len(actor_ids), knots, 2), np.float32)
        value[..., 0] = event_rows[:, None] / 30 + np.arange(knots)[None] / hz
        value[..., 1] = actor_ids[:, None] * .5
        plans[key] = value
    plans["reference_velocity_xy"] = np.tile(np.array([1, 0], np.float32), (len(actor_ids), 151, 1))
    for name, values in (("physical_states.npz", state), ("ground_support.npz", ground), ("reference_plans.npz", plans)):
        np.savez_compressed(folder / name, **values)
    result = {"status": "passed", "cleanup_succeeded": True, "cleanup_errors": [], "CPU_contract_fixture": True,
        "actual_control_steps": controls, "fall_resets": 1, "time_limit_resets": 0,
        "route_vertices_reached": [1] + [0] * (actors - 1), "completed_episode_lengths": [7],
        "physics": {"control_hz": 30, "physics_hz": 120},
        "state_capture": {"path": str(folder / "physical_states.npz"), "frequency_hz": 30, "body_names": names},
        "physical_contact_measurement": {"threshold_n": 1., "physics_substeps_per_control": 4,
            "physics_dt_s": 1 / 120, "control_dt_s": 1 / 30}}
    path = folder / "synthetic_result.json"
    path.write_text(json.dumps(result))
    receipt = {"schema": native.EXECUTION_SCHEMA, "actual_exit_code": 0, "CPU_contract_fixture": True,
        "not_an_observed_SDK_exit": True, "result": native.original.record(path), "fall_resets": 1,
        "route_vertices_reached": result["route_vertices_reached"]}
    receipt_path = folder / "synthetic_kernel_metadata.json"
    receipt_path.write_text(json.dumps(receipt))
    return state, ground, plans, result, receipt_path


@pytest.mark.parametrize("actors,controls", [(3, 30), (15, 30), (3, 6000)])
def test_known_fixture_counts_body_axes_and_native_substeps_are_distinct(actors, controls, tmp_path):
    state, ground, plans, result, receipt_path = raw_fixture(actors, controls, tmp_path)
    kernels = native.prepare_kernels(actors=actors, controls=controls)
    # Run the original CPU-only base in its own interpreter so pytest's other
    # tests may import Torch without weakening the production CPU gate.
    command = """from pathlib import Path
import sys
from CrowdSim.humanoid.analysis.native_navigation_audit import prepare_kernels
n, c, receipt, target = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
k = prepare_kernels(actors=n, controls=c)
sys.argv = ['CPU_fixture_kernel_only', '--execution-receipt', receipt, '--output', target]
raise SystemExit(k.base.main())
"""
    output = tmp_path / "synthetic_base_kernel.json"
    child = subprocess.run([sys.executable, "-c", command, str(actors), str(controls), str(receipt_path), str(output)],
                           cwd=native.ROOT, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    base = json.loads(output.read_text())
    assert base["actor_controls"] == controls * actors
    assert base["physics_substeps"] == controls * 4
    assert base["fall_termination_actor_controls"] == 1 and base["static_contact_actor_controls"] == 1
    assert base["unordered_peer_pair_controls"] == 1 and base["same_episode_position_intervals"] == (controls - 1) * actors - 1
    assert base["double_support_substep_fraction_mean"] == 1
    assert not base["quality_or_author_equivalence_established"] and not base["Torch_imported"]
    pose = kernels.body(tmp_path/"physical_states.npz", tmp_path/"ground_support.npz", controls, result)
    assert pose["body_position_samples"] == controls * actors * 24
    assert pose["native_body_quaternion_samples"] == controls * 4 * actors * 24
    response = kernels.analyze(result, base, base["arrays"], kernels.helpers)
    assert response["all_actor_controls"] == controls * actors
    assert response["fall_actor_controls"] == 1 and response["all_failed_contact_idle_reset_frames_retained"]
    assert response["same_episode_previous_position_intervals"] == (controls - 1) * actors - 1
    assert response["all_control_metrics"]["native_speed_mps"]["mean"] == 1
    assert max(response["all_target_phase_max_abs_errors"].values()) <= 1e-5
    assert sum(response["plan_age_control_counts"].values()) == controls * actors
    assert kernels.joint(plans, controls)["complete_joint_events"] == controls // 15
    assert state["root_rot"].shape[-1] == ground["body_link_quat_w"].shape[-1] == 4


@pytest.mark.parametrize("actors,controls", [(3, 30), (15, 6000), (4, 30)])
def test_derivations_are_reversible_and_leave_original_sources_and_globals_unchanged(actors, controls):
    before = native.source_bindings()
    kernels = native.prepare_kernels(actors=actors, controls=controls)
    assert len(kernels.derivation["kernels"]) == 6
    assert all(row["complete_source_inverse"] and not row["metric_formulae_changed"] for row in kernels.derivation["kernels"])
    assert sum(row["literal_occurrences"] for row in kernels.derivation["kernels"]) == 43
    assert kernels.derivation["physics_substeps_per_control"] == kernels.derivation["quaternion_components"] == 4
    assert native.source_bindings() == before
    assert "_native_actor_count" not in native.original.__dict__


@pytest.mark.parametrize("actors,controls", [(2, 30), (16, 30), (True, 30), (3, 29), (3, 6001), (3, 30.)])
def test_unregistered_actor_or_duration_cannot_be_coerced(actors, controls):
    with pytest.raises(ValueError, match="explicit N3..15"):
        native.prepare_kernels(actors=actors, controls=controls)


@pytest.mark.parametrize("defect,pattern", [("body", "all24 body positions"), ("dof", "FP32"), ("quaternion", "unit XYZW")])
def test_off_root_body_DOF_and_any_substep_quaternion_corruption_is_rejected(defect, pattern, tmp_path):
    state, ground, _, result, _ = raw_fixture(15, 30, tmp_path)
    if defect == "body":
        state["body_pos"][17, 14, 23, 1] += .1
    elif defect == "dof":
        state["dof_vel"] = state["dof_vel"].astype(np.float64)
    else:
        ground["body_link_quat_w"][17, 0, 14, 23, 3] = .5
    np.savez_compressed(tmp_path/"physical_states.npz", **state)
    np.savez_compressed(tmp_path/"ground_support.npz", **ground)
    with pytest.raises(ValueError, match=pattern):
        native.prepare_kernels(actors=15, controls=30).body(tmp_path/"physical_states.npz", tmp_path/"ground_support.npz", 30, result)


@pytest.mark.parametrize("defect,pattern", [("missing_actor", "complete"), ("clock", "unordered"),
    ("duration", "duration differs"), ("ID_width", "complete")])
def test_joint_event_cannot_split_scene_or_change_the_registered_duration(defect, pattern, tmp_path):
    _, _, plans, _, _ = raw_fixture(15, 30, tmp_path)
    controls = 30
    if defect == "missing_actor":
        plans["actor_ids"] = plans["actor_ids"][:-1]
    elif defect == "clock":
        plans["control_step"][15:] = 0
    elif defect == "ID_width":
        plans["actor_ids"] = plans["actor_ids"][:, None]
    else:
        controls = 6000
    with pytest.raises(ValueError, match=pattern):
        native.prepare_kernels(actors=15, controls=30).joint(plans, controls)


@pytest.mark.parametrize("defect,pattern", [("actor_substeps", "validity or reset clocks"), ("physics_clock", "validity or reset clocks"),
    ("floor_force", "Saved support flags"), ("root_width", "physical channel: root_rot"),
    ("peer_width", "physical channel: peer_contact"), ("target", r"age\+1")])
def test_native_axes_clock_support_and_reference_reconstruction_reject_corruption(defect, pattern, tmp_path):
    state, ground, _, result, _ = raw_fixture(15, 30, tmp_path)
    if defect == "actor_substeps":
        ground["actor_substep_valid_count"] = np.full((30, 4), 4, np.int64)
    elif defect == "physics_clock":
        ground["physics_step"][5, 0] += 1
    elif defect == "floor_force":
        ground["foot_floor_normal_force_w"][5, 0, 14, 3, 0] = 0
    elif defect == "root_width":
        state["root_rot"] = np.zeros((30, 15, 15), np.float32)
    elif defect == "peer_width":
        state["peer_contact"] = np.zeros((30, 15, 4), bool)
    else:
        state["target_xy"][7, 14, 0] += .1
    np.savez_compressed(tmp_path/"physical_states.npz", **state)
    np.savez_compressed(tmp_path/"ground_support.npz", **ground)
    paths = [native.original.record(tmp_path/name) for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")]
    synthetic_base = {"fall_termination_actor_controls": 1, "route_vertices_reached": result["route_vertices_reached"],
                      "same_episode_position_intervals": 29 * 15 - 1}
    kernels = native.prepare_kernels(actors=15, controls=30)
    with pytest.raises(ValueError, match=pattern):
        kernels.analyze(result, synthetic_base, paths, kernels.helpers)


@pytest.mark.parametrize("actors,controls,planner", [(3, 30, "diffusion"), (15, 6000, "sfm")])
def test_fresh_source_CLI_does_not_import_torch_or_start_SDK(actors, controls, planner):
    result = subprocess.run([sys.executable, str(native.ROOT/"scripts/audit_shared_social_native.py"),
        "--actors", str(actors), "--control-steps", str(controls), "--planner", planner, "--check-sources"],
        cwd=native.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["Torch_imported"] is False and report["SDK_started"] is False
    assert not report["actual_native_capture_audited"]


@pytest.mark.parametrize("receipt", [{"schema": "shared_social.actual_navigation200s_execution.v1", "actual_exit_code": 0},
    {"schema": native.EXECUTION_SCHEMA, "actual_exit_code": False, "terminal_tool_chunk": "CPU_fixture_only"},
    {"schema": native.EXECUTION_SCHEMA, "actual_exit_code": 0}])
def test_formal_admission_rejects_old_fixture_or_missing_actual_exit_before_resources(receipt, tmp_path):
    path = tmp_path/"synthetic_metadata.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="actual successful native SDK exit"):
        native.admit_execution(path, actors=3, controls=30, planner="diffusion", scene_config=tmp_path/"missing",
                               scene_sha256="0" * 64)


def test_source_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(native, "HIGH_LEVEL_SHA", "0" * 64)
    with pytest.raises(ValueError, match="registered-duration"):
        native.prepare_kernels(actors=3, controls=30)


def declaration_fixture(actors, controls, planner):
    """In-memory validator declarations, never published as native evidence."""
    scene = {"path": "/CPU_fixture_only/scene.json", "bytes": 1, "sha256": "1" * 64}
    request = {"configuration_file": scene, "configuration": {"scene": "CPU_fixture_only"}}
    arrays = {"path": "/CPU_fixture_only/initial.npz", "bytes": 1, "sha256": "2" * 64}
    checkpoint = {"path": "/CPU_fixture_only/checkpoint.pt", "bytes": 1, "sha256": "3" * 64}
    args = {"num_envs": actors, "evaluate_steps": controls, "scene_config": scene["path"],
        "manifest": str(native.MANIFEST), "native_dynbench": True, "output": "/CPU_fixture_only/run"}
    migration = {"entry": native.original.record(native.ENTRY), "planner": planner,
        "native_initial_capture": {"sources": native.initial.source_bindings(), "actors": actors}}
    names = ["Pelvis"] + [f"CPU_fixture_body_{index}" for index in range(1, 24)]
    result = {"status": "passed", "cleanup_succeeded": True, "cleanup_errors": [], "actual_control_steps": controls,
        "social_sources_and_artifacts_unchanged": True,
        "state_capture": {"body_names": names}, "physics": {"body_names": names},
        "arguments": args, "social_entry": {"native_current_migration": migration, "native_scene_request": request,
            "actors": actors, "requested_controls": controls, "physical_execution_verified": True,
            "primary_200_seconds": controls == 6000}, "shared_navigation_migration": migration,
        **{key: {"observed_count": count, "live_before_app_shutdown": 0} for key, count in
            (("navigation_owned_contact_views", actors + 1), ("navigation_SDK_sensor_objects", 24),
             ("navigation_borrowed_backend_views", 1))},
        "native_scene_live_observation": {"schema": "dynbench_live_runtime_observation.v1",
            "status": "passed_live_stage_and_sampled_native_floor_ownership", "scene": "CPU_fixture_only",
            "observed_current_singleton_context_identity": True, "observed_active_stage_matches_context": True,
            "source_layers_rehashed_before_after_unchanged": True, "observed_floor_geometry_matches_recipe": True,
            "generated_or_foreign_terrain_collision_paths": [], "reset_origins_tested": actors,
            "exact_dynamic_robot_roots": [f"/World/envs/env_{index}/Robot" for index in range(actors)],
            "submitted_step_counter_before": 17, "submitted_step_counter_after": 17, "query_physics_advance": 0},
        "paired_initial_condition": {"arrays": arrays}, "evaluation_actor": {"checkpoint": checkpoint},
        "trajectory_source": "diffusion" if planner == "diffusion" else "public_sfm_joint_future"}
    if planner == "sfm":
        result["high_level_planner"] = {"schema": "local.public_sfm.shared_AMP_native_bridge.v1", "actors": actors,
            "native_resources": request, "actual_world_boundary_checked": True,
            "diffusion_weights_loaded": False, "diffusion_guidance_applied": False}
    arguments = dict(args)
    arguments.pop("output")
    snapshot = {"arrays": arrays, "contract": {"capture_kind": native.initial.LIVE_KIND,
        "clock": {"common_body_names": names},
        "native_protocol": {"actors": actors}, "native_resources": request, "arguments_excluding_output": arguments,
        "resources": {"manifest": {"sha256": native.MANIFEST_SHA}}, "actor": {"checkpoint": checkpoint}}}
    receipt = {"schema": native.EXECUTION_SCHEMA, "actual_exit_code": 0, "terminal_tool_chunk": "CPU_fixture_not_real_SDK",
        "actors": actors, "control_steps": controls, "planner": planner, "scene_configuration": scene}
    return receipt, result, request, snapshot


@pytest.mark.parametrize("actors,controls,planner", [(3, 30, "diffusion"), (15, 6000, "sfm")])
def test_native_declaration_validation_does_not_independently_claim_exit_or_quality(actors, controls, planner):
    values = declaration_fixture(actors, controls, planner)
    before = deepcopy(values)
    validation = native.validate_result_contract(*values, actors=actors, controls=controls, planner=planner)
    assert validation["declared_native_stage_floor_source_and_exposure_admitted"]
    assert not validation["actual_process_exit_independently_observed_by_reader"] and not validation["paper_effect_achieved"]
    assert values == before


@pytest.mark.parametrize("defect,pattern", [("exit_bool", "externally observed"), ("actor_count", "externally observed"),
    ("smoke_as_full", "requested exposure"), ("scene", "requested exposure"), ("source", "source bindings"),
    ("view", "owned view"), ("sensor", "owned view"), ("stage", "native stage"), ("roots", "native stage"),
    ("query", "native stage"), ("CPU_initial", "CPU fixture"), ("initial_args", "CPU fixture"),
    ("actor", "CPU fixture"), ("body_names", "CPU fixture"), ("source_closure", "requested exposure"),
    ("planner", "Wrong native planner"), ("SFM_weights", "no Diffusion weights")])
def test_native_declared_scene_initial_actor_and_cleanup_mismatches_are_rejected(defect, pattern):
    receipt, result, request, snapshot = declaration_fixture(15, 6000, "sfm")
    if defect == "exit_bool":
        receipt["actual_exit_code"] = False
    elif defect == "actor_count":
        receipt["actors"] = 4
    elif defect == "smoke_as_full":
        result["actual_control_steps"] = 30
    elif defect == "scene":
        result["arguments"]["scene_config"] = "/wrong/scene.json"
    elif defect == "source":
        result["shared_navigation_migration"]["entry"] = {"path": "/wrong/source.py"}
    elif defect == "view":
        result["navigation_owned_contact_views"]["observed_count"] = 5
    elif defect == "sensor":
        result["navigation_SDK_sensor_objects"]["observed_count"] = 15 * 24
    elif defect == "stage":
        result["native_scene_live_observation"]["observed_active_stage_matches_context"] = False
    elif defect == "roots":
        result["native_scene_live_observation"]["exact_dynamic_robot_roots"].reverse()
    elif defect == "query":
        result["native_scene_live_observation"]["submitted_step_counter_after"] += 1
    elif defect == "CPU_initial":
        snapshot["contract"]["capture_kind"] = "CPU_contract_fixture"
    elif defect == "initial_args":
        snapshot["contract"]["arguments_excluding_output"]["num_envs"] = 4
    elif defect == "actor":
        snapshot["contract"]["actor"]["checkpoint"] = {"path": "/wrong/model.pt"}
    elif defect == "body_names":
        snapshot["contract"]["clock"]["common_body_names"] = ["wrong"] * 24
    elif defect == "source_closure":
        result["social_sources_and_artifacts_unchanged"] = False
    elif defect == "planner":
        result["trajectory_source"] = "diffusion"
    else:
        result["high_level_planner"]["diffusion_weights_loaded"] = True
    with pytest.raises(ValueError, match=pattern):
        native.validate_result_contract(receipt, result, request, snapshot, actors=15, controls=6000, planner="sfm")


@pytest.mark.parametrize("actors", [3, 15])
def test_initial_link_checks_every_registered_anchor_counter_episode_and_actor(actors, tmp_path, monkeypatch):
    _, _, plans, _, _ = raw_fixture(actors, 30, tmp_path)
    document = tmp_path/"paired_initial_condition.json"
    document.write_text('{"CPU_contract_fixture":true}\n')
    array_path = tmp_path/"synthetic_initial_arrays.npz"
    position = np.zeros((actors, 24, 3), np.float32)
    position[:, 0, :2] = plans["raw_xy"][:actors, 0]
    velocity = np.zeros_like(position)
    velocity[:, 0, 0] = 1
    np.savez_compressed(array_path, common_rigid_body_pos=position, common_rigid_body_vel=velocity,
                        world_episode_ids=np.ones(actors, np.int64))
    snapshot = {"arrays": native.original.record(array_path), "contract": {"capture_kind": native.initial.LIVE_KIND,
        "native_protocol": {"actors": actors}, "clock": {"physics_step_after": 17}}}
    result = {"paired_initial_condition": {"document": native.original.record(document), "arrays": snapshot["arrays"]}}
    kernels = native.prepare_kernels(actors=actors, controls=30)
    # Isolate the separately tested native archive reader. This checks links
    # only; this fixture is never submitted to actual archive/SDK admission.
    monkeypatch.setitem(kernels.initial_link.__globals__, "read_snapshot", lambda _: snapshot)
    def call(selected):
        return kernels.initial_link(document, result, tmp_path/"ground_support.npz", selected, tmp_path)
    out = call(plans)
    assert out["all_registered_first_raw_reference_position_velocity_anchors_equal"]
    assert out["first_native_control_counter_and_episode_equal"]
    assert not out["paired_other_arm_comparison_performed"]
    changed = deepcopy(plans)
    changed["reference_velocity_xy"][actors - 1, 0, 0] += .1
    with pytest.raises(ValueError, match="anchor differs"):
        call(changed)
    snapshot["contract"]["native_protocol"]["actors"] = 4 if actors != 4 else 3
    with pytest.raises(ValueError, match="CPU fixture"):
        call(plans)
    snapshot["contract"]["native_protocol"]["actors"] = actors
    snapshot["contract"]["clock"]["physics_step_after"] += 1
    with pytest.raises(ValueError, match="counter/episode"):
        call(plans)
