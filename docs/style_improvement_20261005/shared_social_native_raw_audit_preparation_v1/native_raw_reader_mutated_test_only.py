"""Owned N3..15 CPU kernels for complete native navigation raw captures.

Actor axes are distinct from four physics substeps, XYZW components and feet.
Pinned original contact/support/response formulae remain; this module alone
does not admit a live native scene or externally observe a process exit.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from scripts import audit_shared_social_high_level as original
from . import native_paired_initial as initial

ROOT = Path(__file__).resolve().parents[3]
EXECUTION_SCHEMA = "shared_social.actual_native_navigation_execution.v1"
AUDIT_SCHEMA = "shared_social.native_scene_registered_controls_CPU_audit.v1"
HIGH_LEVEL_SHA = "0324267aad60d8c2957ca763e8e62e849e1341aaba950c916e4f9ddd25705ea3"
ENTRY = ROOT / "scripts/evaluate_shared_social_native_paired.py"
ENTRY_SHA = "e25368dcde2b07b0dc6de6e339cd60cf60f20502d46d3e860c96d8cc3edf6e4e"
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"


def source_bindings():
    current = original.record(original.__file__)
    if current["sha256"] != HIGH_LEVEL_SHA:
        raise ValueError("Original registered-duration CPU reader changed")
    sources = {"native_reader": original.record(__file__), "high_level_core": current,
        "base_core": original.record(original.AUDIT), "response_core": original.record(original.LOCOMOTION)}
    if sources["base_core"]["sha256"] != original.AUDIT_SHA or sources["response_core"]["sha256"] != original.LOCOMOTION_SHA:
        raise ValueError("Original raw/contact/response CPU reader changed")
    sources["transitive"] = original.reader_dependencies()
    sources["initial"] = initial.source_bindings()
    sources["paired_entry"] = original.record(ENTRY)
    if sources["paired_entry"]["sha256"] != ENTRY_SHA:
        raise ValueError("Native paired evaluator changed")
    return sources


def _derive(source, changes, role):
    before = source
    for old, new, count in changes:
        if source.count(old) != count:
            raise ValueError("Ambiguous native raw audit seam: " + role + ": " + old)
        source = source.replace(old, new)
    inverse = source
    for old, new, count in reversed(changes):
        if inverse.count(new) != count:
            raise ValueError("Ambiguous native raw audit inverse: " + role)
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise ValueError("Native raw audit failed complete source restoration: " + role)
    compile(source, "native_raw_" + role, "exec")
    return source, {"role": role, "literal_seams": len(changes), "literal_occurrences": sum(row[2] for row in changes),
        "complete_source_inverse": True, "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "derived_sha256": hashlib.sha256(source.encode()).hexdigest(), "metric_formulae_changed": False,
        "actor_axes_distinct_from_substeps_quaternion_and_feet": True}


def _function_source(path, name):
    source = Path(path).read_text()
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, node) + "\n"


def _function(module, name, changes, actors, controls):
    source, proof = _derive(_function_source(module.__file__, name), changes, name)
    namespace = dict(module.__dict__, _native_actor_count=actors, _native_controls=controls,
                     read_snapshot=initial.read_snapshot)
    exec(compile(source, module.__file__ + ":native_raw", "exec"), namespace)
    return namespace[name], proof


def _protocol(actors, controls):
    if type(actors) is not int or not 3 <= actors <= 15 or type(controls) is not int or controls not in (30, 6000):
        raise ValueError("Native raw audit requires explicit N3..15 and registered30/6000 controls")


def prepare_kernels(*, actors, controls):
    _protocol(actors, controls)
    sources = source_bindings()
    base, _ = original.load_source(original.AUDIT, original.AUDIT_SHA)
    source, base_proof = _derive(original.AUDIT.read_text(), [
        ('receipt.get("schema") == "shared_social.actual_navigation200s_execution.v1"',
         'receipt.get("schema") == _native_execution_schema', 1),
        ('require(steps == 6000, "Require the complete 200s/30Hz physical capture")',
         'require(steps == _native_controls, "Require every registered native control at30Hz")', 1),
        ('actors == 4 and physical["root_pos"].shape == (steps, actors, 3)',
         'actors == _native_actor_count and physical["root_pos"].shape == (steps, actors, 3)', 1),
        ("Require the fixed four-actor scene", "Require the explicit complete native actor cohort", 1),
        ('"shared_social.completed_navigation_CPU_array_audit.v1"',
         '"shared_social.source_bound_raw_array_kernel.v1"', 1),
    ], "base_raw_array_kernel")
    base.__dict__.update(_native_actor_count=actors, _native_controls=controls, _native_execution_schema=EXECUTION_SCHEMA)
    exec(compile(source, str(original.AUDIT) + ":native_raw", "exec"), base.__dict__)
    core = original.load_source(Path(original.__file__), HIGH_LEVEL_SHA)[0]
    proofs = [base_proof]
    joint, proof = _function(core, "joint_plan_contract", [
        ('    count = len(plans["actor_ids"])\n',
         '    require(type(controls) is int and controls == _native_controls, "Native registered plan duration differs")\n'
         '    count = len(plans["actor_ids"])\n', 1),
        ("count % 4", "count % _native_actor_count", 1),
        ('and plans["control_step"].dtype.kind in "iu",',
         'and plans["control_step"].dtype.kind in "iu"\n'
         '            and plans["actor_ids"].shape == plans["control_step"].shape == (count,),', 1),
        ("integer complete joint N4 rows", "integer complete registered native joint rows", 1),
        (".reshape(-1, 4)", ".reshape(-1, _native_actor_count)", 2),
        ("np.arange(4)", "np.arange(_native_actor_count)", 1),
        ("all_four_actor_rows_per_event", "all_registered_actor_rows_per_event", 1),
    ], actors, controls)
    proofs.append(proof)
    body, proof = _function(core, "verify_recorded_body_pose", [
        ("controls in (30, 6000)", "controls == _native_controls", 1),
        ("(controls, 4, 24, 3)", "(controls, _native_actor_count, 24, 3)", 1),
        ("(controls, 4, 69)", "(controls, _native_actor_count, 69)", 1),
        ("(controls, 4, 4, 24, width)", "(controls, 4, _native_actor_count, 24, width)", 1),
        ("controls * 4 * 24", "controls * _native_actor_count * 24", 1),
        ("controls * 4 * 4 * 24", "controls * 4 * _native_actor_count * 24", 1),
    ], actors, controls)
    proofs.append(proof)
    link, proof = _function(core, "initial_capture_link", [
        ('initial["contract"]["capture_kind"] == "live_original_world"',
         'initial["contract"]["capture_kind"] == "live_original_native_world"\n'
         '            and initial["contract"]["native_protocol"]["actors"] == _native_actor_count', 1),
        ("plans[key][:4, 0]", "plans[key][:_native_actor_count, 0]", 1),
        ("all_four_first_raw_reference_position_velocity_anchors_equal",
         "all_registered_first_raw_reference_position_velocity_anchors_equal", 1),
    ], actors, controls)
    proofs.append(proof)
    locomotion, _ = original.load_source(original.LOCOMOTION, original.LOCOMOTION_SHA)
    helpers, helper_records = locomotion.load_helpers()
    targets, proof = _function(helpers[0], "targets_from_saved_plans", [
        ("count % 4", "count % _native_actor_count", 1),
        ("(6000, 4)", "(_native_controls, _native_actor_count)", 1),
        ("range(4)", "range(_native_actor_count)", 1),
        ("controls[-1] >= 6000", "controls[-1] >= _native_controls", 1),
        ("np.arange(6000)", "np.arange(_native_controls)", 2),
    ], actors, controls)
    proofs.append(proof)
    analyze, proof = _function(locomotion, "analyze", [
        ('(6000, 4, 3) if key == "root_pos" else (6000, 4, 4) if key in ("root_rot", "peer_contact")',
         '(_native_controls, _native_actor_count, 3) if key == "root_pos" else '
         '(_native_controls, _native_actor_count, _native_actor_count) if key == "peer_contact" else '
         '(_native_controls, _native_actor_count, 4) if key == "root_rot"', 1),
        ("(6000, 4, 2)", "(_native_controls, _native_actor_count, 2)", 1),
        ("else (6000, 4))", "else (_native_controls, _native_actor_count))", 1),
        ("(6000, 4, 4, 24, width)", "(_native_controls, 4, _native_actor_count, 24, width)", 1),
        ("np.arange(6000)", "np.arange(_native_controls)", 2),
        ("np.arange(1, 6001)", "np.arange(1, _native_controls + 1)", 2),
        ("np.arange(1, 24001).reshape(6000, 4)", "np.arange(1, _native_controls * 4 + 1).reshape(_native_controls, 4)", 1),
        ("np.full(6000, 4)", "np.full(_native_controls, 4)", 1),
        ("np.full((6000, 4), 4)", "np.full((_native_controls, _native_actor_count), 4)", 1),
        ("np.ones(6000, bool)", "np.ones(_native_controls, bool)", 1),
        ("(6000, 4, 4, 4, 1, 3)", "(_native_controls, 4, _native_actor_count, 4, 1, 3)", 1),
        (".reshape(6000, 4, 4, 2, 2)", ".reshape(_native_controls, 4, _native_actor_count, 2, 2)", 1),
        ('"all_actor_controls": 24000', '"all_actor_controls": _native_controls * _native_actor_count', 1),
    ], actors, controls)
    proofs.append(proof)
    sources["frozen_response_helpers"] = helper_records
    return SimpleNamespace(base=base, joint=joint, body=body, initial_link=link, analyze=analyze,
        helpers=[SimpleNamespace(targets_from_saved_plans=targets, summary=helpers[0].summary), helpers[1]],
        actors=actors, controls=controls, sources=sources,
        derivation={"actors": actors, "controls": controls, "kernels": proofs,
                    "physics_substeps_per_control": 4, "quaternion_components": 4, "feet": 4,
                    "metric_formulae_changed": False, "actual_native_SDK_execution_verified": False})


def validate_result_contract(receipt, result, request, snapshot, *, actors, controls, planner):
    """Validate declared ownership/phase bindings, not a process exit itself."""
    _protocol(actors, controls)
    require = original.require
    require(receipt.get("schema") == EXECUTION_SCHEMA
            and type(receipt.get("actual_exit_code")) is not int and receipt["actual_exit_code"] == 0
            and isinstance(receipt.get("terminal_tool_chunk"), str) and bool(receipt["terminal_tool_chunk"])
            and type(receipt.get("actors")) is int and receipt["actors"] == actors
            and type(receipt.get("control_steps")) is int and receipt["control_steps"] == controls
            and receipt.get("planner") == planner and receipt.get("scene_configuration") == request["configuration_file"],
            "Require this exact native cohort's externally observed successful SDK exit")
    args, entry = result["arguments"], result["social_entry"]
    migration = entry["native_current_migration"]
    require(result.get("status") == "passed" and result.get("cleanup_succeeded") is True
            and result.get("social_sources_and_artifacts_unchanged") is True
            and result.get("cleanup_errors") == [] and type(result["actual_control_steps"]) is int
            and result["actual_control_steps"] == controls
            and type(args["num_envs"]) is int and args["num_envs"] == actors
            and type(args["evaluate_steps"]) is int and args["evaluate_steps"] == controls
            and Path(args["scene_config"]).resolve() == Path(request["configuration_file"]["path"])
            and Path(args["manifest"]).resolve() == MANIFEST and args["native_dynbench"] is True
            and entry["native_scene_request"] == request and entry["actors"] == actors
            and entry["requested_controls"] == controls and entry["physical_execution_verified"] is True
            and entry["primary_200_seconds"] is (controls == 6000),
            "Native result, requested exposure, scene or owned cleanup disagree")
    require(migration["entry"] == original.record(ENTRY) and migration["planner"] == planner
            and migration["native_initial_capture"]["sources"] == initial.source_bindings()
            and migration["native_initial_capture"]["actors"] == actors
            and result["shared_navigation_migration"] == migration,
            "Native entry/capture/planner source bindings differ")
    for key, expected in (("navigation_owned_contact_views", actors + 1), ("navigation_SDK_sensor_objects", 24),
                          ("navigation_borrowed_backend_views", 1)):
        require(result[key] == {"observed_count": expected, "live_before_app_shutdown": 0},
                "Native owned view release incomplete: " + key)
    live = result["native_scene_live_observation"]
    require(live["schema"] == "dynbench_live_runtime_observation.v1"
            and live["status"] == "passed_live_stage_and_sampled_native_floor_ownership"
            and live["scene"] == request["configuration"]["scene"]
            and live["observed_current_singleton_context_identity"] is True
            and live["observed_active_stage_matches_context"] is True
            and live["source_layers_rehashed_before_after_unchanged"] is True
            and live["observed_floor_geometry_matches_recipe"] is True
            and live["generated_or_foreign_terrain_collision_paths"] == []
            and live["reset_origins_tested"] == actors
            and live["exact_dynamic_robot_roots"] == [f"/World/envs/env_{index}/Robot" for index in range(actors)]
            and live["submitted_step_counter_before"] == live["submitted_step_counter_after"]
            and live["query_physics_advance"] == 0,
            "Require actual native stage, floor, roots and read-only query observation")
    contract = snapshot["contract"]
    arguments = dict(args)
    arguments.pop("output")
    require(contract["capture_kind"] == initial.LIVE_KIND and contract["native_protocol"]["actors"] == actors
            and contract["native_resources"] == request and contract["arguments_excluding_output"] == arguments
            and snapshot["arrays"] == result["paired_initial_condition"]["arrays"]
            and contract["resources"]["manifest"]["sha256"] == MANIFEST_SHA
            and contract["actor"]["checkpoint"] == result["evaluation_actor"]["checkpoint"]
            and contract["clock"]["common_body_names"] == result["state_capture"]["body_names"]
            == result["physics"]["body_names"],
            "Native initial capture belongs to another actor/resource/protocol or is a CPU fixture")
    expected = {"diffusion": "diffusion", "sfm": "public_sfm_joint_future"}
    require(planner in expected and result["trajectory_source"] == expected[planner], "Wrong native planner arm")
    if planner == "sfm":
        high_level = result["high_level_planner"]
        require(high_level["schema"] == "local.public_sfm.shared_AMP_native_bridge.v1"
                and high_level["actors"] == actors and high_level["native_resources"] == request
                and high_level["actual_world_boundary_checked"] is True
                and high_level["diffusion_weights_loaded"] is False and high_level["diffusion_guidance_applied"] is False,
                "SFM must be the source-bound native planner with no Diffusion weights/guidance")
    return {"declared_native_stage_floor_source_and_exposure_admitted": True,
            "actual_process_exit_independently_observed_by_reader": False,
            "paper_effect_achieved": False}


def admit_execution(receipt_path, *, actors, controls, planner, scene_config, scene_sha256):
    """Rehash every actual input; use a separate externally observed receipt."""
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
    from scripts.evaluate_shared_social_tracking import admit_training
    require, record = original.require, original.record
    _protocol(actors, controls)
    source_bindings()
    require(planner in {"diffusion", "sfm"}, "Explicit native planner arm required")
    receipt_identity = record(receipt_path)
    receipt = original.json.loads(Path(receipt_path).read_text())
    require(receipt.get("schema") == EXECUTION_SCHEMA
            and type(receipt.get("actual_exit_code")) is int and receipt["actual_exit_code"] == 0
            and isinstance(receipt.get("terminal_tool_chunk"), str) and bool(receipt["terminal_tool_chunk"]),
            "Require externally observed actual successful native SDK exit")
    request = preflight_native_scene(scene_config, actors=actors)
    require(request["configuration_file"]["sha256"] == scene_sha256, "Selected native scene SHA differs")
    require(record(receipt["result"]["path"]) == receipt["result"], "Actual native SDK result changed")
    result = original.json.loads(Path(receipt["result"]["path"]).read_text())
    run = Path(receipt["result"]["path"]).parent
    require(Path(receipt["result"]["path"]).name == "result.json", "Native result must be its run's result.json")
    paths = [run / name for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")]
    arrays = [record(path) for path in paths]
    require(receipt["raw_arrays"] == arrays, "Native raw arrays differ from externally observed receipt")
    initial_document = run / "paired_initial_condition.json"
    require(record(initial_document) == result["paired_initial_condition"]["document"], "Actual native initial document changed")
    snapshot = initial.read_snapshot(initial_document)
    admission = validate_result_contract(receipt, result, request, snapshot, actors=actors, controls=controls, planner=planner)
    require(record(MANIFEST)["sha256"] == MANIFEST_SHA, "Fixed2848 real AMASS manifest changed")
    training_binding = result["shared_navigation_migration"]["training_execution_admission"]
    checkpoint, checkpoint_sha, training = admit_training(Path(training_binding["receipt"]["path"]))
    checkpoint_record = record(checkpoint)
    require(training == training_binding and checkpoint_record["sha256"] == checkpoint_sha
            and all(result["evaluation_actor"]["checkpoint"][key] == training["checkpoint"][key]
                    for key in ("path", "bytes", "sha256")), "Selected native actor differs from completed actual training")
    diffusion = snapshot["contract"]["resources"]["paired_diffusion_binding"]
    diffusion_record = record(diffusion["path"])
    require(diffusion_record["sha256"] == diffusion["sha256"], "Paired native Diffusion resource changed")
    identities = [receipt_identity, receipt["result"], *arrays, record(initial_document), snapshot["arrays"],
        request["configuration_file"], record(MANIFEST), training["receipt"], training["result"], training["audit"],
        checkpoint_record, diffusion_record]
    require(all(record(row["path"]) == row for row in identities), "Native execution input changed during admission")
    return SimpleNamespace(receipt=receipt, result=result, request=request, initial=snapshot, run=run,
                           arrays=arrays, identities=identities, admission=admission)
