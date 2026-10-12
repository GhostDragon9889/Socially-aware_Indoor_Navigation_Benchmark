#!/usr/bin/env python3
"""CPU audit of every published N4 control, including smoke and failures.

Reuses pinned contact/support/response calculations. A 30-control capture is
integration evidence only; a complete 6000-control run is still not author
protocol equivalence. No policy, training data or SDK is loaded here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.paired_initial import read_snapshot  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402

AUDIT = ROOT / "scripts/audit_shared_social_navigation.py"
AUDIT_SHA = "8fb54bbd2048eb755713b025d7f5331c6296520fa3870a5ea4fec4afe48d0414"
LOCOMOTION = ROOT / "scripts/diagnose_shared_social_locomotion.py"
LOCOMOTION_SHA = "1f0c2d07d494f1da18e98f3fd7a895938980485195257f9132cb7708253e6fac"
DEPENDENCIES = (
    ("CrowdSim/humanoid/analysis/support_phases.py", "d27928fc366edb32d0b06eaafcb83820ce214fb42dfc12f401f8eec6dc67dbe7"),
    ("scripts/compare_shared_social_evidence.py", "76e589b05101480ac21ae229e7c8ecdefde049073670deaddb88a4306ed77a35"),
    ("CrowdSim/humanoid/foot_support.py", "8c9cb88ddcbe55599ebe79abf80ea7a77a85eedfaa17324b937a862b284267e8"),
    ("CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py", "991b6107f113d961e9b2cccca1efdb8bbc9d1d484188c1b3b716e4cb11e3e15e"),
    ("CrowdSim/humanoid/analysis/paired_initial.py", "3395ebc1e4de385788fec738f49994af2887bb1b0af998363608c8625e17022d"),
)
COUNTS = {"analyze": {6000: 13, 6001: 2, 24000: 1, 24001: 1}, "targets_from_saved_plans": {6000: 4}}
FORBIDDEN = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_source(path, sha):
    identity = record(path)
    require(identity["sha256"] == sha, "Pinned original CPU reader changed")
    module = ModuleType("_isolated_original_" + path.stem)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module, identity


def reader_dependencies():
    identities = []
    for relative, sha in DEPENDENCIES:
        identity = record(ROOT / relative)
        if identity["sha256"] != sha:
            from CrowdSim.humanoid.analysis.cpu_dependency_migration import verify_dependency
            verify_dependency(ROOT / relative, sha)
        identities.append(identity)
    return identities


def derive_base_audit(source, controls):
    require(type(controls) is int and controls in (30, 6000), "Only registered smoke30 and full200s are supported")
    replacements = [
        ('receipt.get("schema") == "shared_social.actual_navigation200s_execution.v1"',
         'receipt.get("schema") == ' + repr("shared_social.actual_navigation_smoke_execution.v1"
            if controls == 30 else "shared_social.actual_navigation200s_execution.v1")),
        ('require(steps == 6000, "Require the complete 200s/30Hz physical capture")',
         f'require(steps == {controls}, "Require all registered controls at30Hz")'),
    ]
    derived = source
    for old, new in replacements:
        require(derived.count(old) == 1, "Ambiguous registered-duration base audit change")
        derived = derived.replace(old, new, 1)
    inverse = derived
    for old, new in reversed(replacements):
        require(inverse.count(new) == 1, "Ambiguous registered-duration base audit inverse")
        inverse = inverse.replace(new, old, 1)
    require(inverse == source, "Duration audit did not restore whole original bytes")
    compile(derived, "source_bound_registered_duration_audit", "exec")
    return derived, {"whole_source_inverse": True, "replacement_count": 2,
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "contact_episode_support_metric_math_changed": False}


def duration_function(function, controls):
    """Change only registered array lengths/counts, with node-specific inverse.

    Python3.12 AST definitions: https://docs.python.org/3.12/library/ast.html
    Node positions avoid changing existing30Hz constants during restoration.
    """
    require(type(controls) is int and controls in (30, 6000) and function.__name__ in COUNTS, "Unregistered function or duration")
    original = ast.parse(inspect.getsource(function))
    selected = [(index, node.value) for index, node in enumerate(ast.walk(original))
                if isinstance(node, ast.Constant) and type(node.value) is int and node.value in {6000, 6001, 24000, 24001}]
    require(dict(Counter(value for _, value in selected)) == COUNTS[function.__name__],
            "Original duration literal inventory changed")
    remap = {6000: controls, 6001: controls + 1, 24000: controls * 4, 24001: controls * 4 + 1}
    derived = copy.deepcopy(original)
    nodes = list(ast.walk(derived))
    for index, old in selected:
        nodes[index].value = remap[old]
    inverse = copy.deepcopy(derived)
    restored = list(ast.walk(inverse))
    for index, old in selected:
        require(restored[index].value == remap[old], "Duration node inverse mismatch")
        restored[index].value = old
    require(ast.dump(inverse, include_attributes=False) == ast.dump(original, include_attributes=False),
            "Duration derivation changed another AST node")
    namespace = dict(function.__globals__)
    exec(compile(derived, function.__code__.co_filename + ":registered_duration", "exec"), namespace)
    return namespace[function.__name__], {"function": function.__name__, "constant_inventory": COUNTS[function.__name__],
        "numeric_literals_actually_changed": sum(remap[old] != old for _, old in selected),
        "exact_AST_inverse": True, "all_original_AST_nodes_unchanged_at6000": controls == 6000,
        "original_AST_sha256": hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        "derived_AST_sha256": hashlib.sha256(ast.dump(derived, include_attributes=False).encode()).hexdigest()}


def joint_plan_contract(plans, controls):
    count = len(plans["actor_ids"])
    require(count > 0 and count % 4 == 0 and plans["actor_ids"].dtype.kind in "iu"
            and plans["control_step"].dtype.kind in "iu", "Plans require integer complete joint N4 rows")
    actors = plans["actor_ids"].reshape(-1, 4)
    clock = plans["control_step"].reshape(-1, 4)
    require(np.array_equal(actors, np.tile(np.arange(4), (len(actors), 1)))
            and np.equal(clock, clock[:, :1]).all() and clock[0, 0] == 0
            and (np.diff(clock[:, 0]) > 0).all() and clock[-1, 0] < controls,
            "Missing, split, duplicated or unordered shared-scene plan events")
    for key, knots in (("raw_xy", 51), ("reference_xy", 151), ("reference_velocity_xy", 151)):
        value = plans[key]
        require(value.shape == (count, knots, 2) and value.dtype == np.float32 and np.isfinite(value).all(),
                "Invalid complete FP32 plan channel: " + key)
    return {"complete_joint_events": len(actors), "all_four_actor_rows_per_event": True,
            "first_control": int(clock[0, 0]), "last_control": int(clock[-1, 0])}


def verify_recorded_body_pose(physical_path, ground_path, controls, result):
    """Bind all recorded body positions to the native fourth substep.

    DOF channels are checked for complete finite FP32 captures; these do not
    have a second native DOF capture here and are not independently matched.
    """
    physics = result["physics"]
    capture = result["state_capture"]
    contact = result["physical_contact_measurement"]
    require(type(controls) is int and controls in (30, 6000)
            and physics["physics_hz"] == 120 and physics["control_hz"] == 30
            and capture["frequency_hz"] == 30 and contact["physics_substeps_per_control"] == 4
            and contact["physics_dt_s"] == 1 / 120 and contact["control_dt_s"] == 1 / 30,
            "Body pose/native capture must declare 30Hz controls and four120Hz substeps")
    require(Path(capture["path"]).resolve() == Path(physical_path).resolve(),
            "Body pose capture path differs from this run's physical archive")
    with np.load(physical_path, allow_pickle=False) as physical, np.load(ground_path, allow_pickle=False) as ground:
        require(ground["common_body_names"].tolist() == capture["body_names"]
                and len(capture["body_names"]) == 24, "Recorded body pose mapping differs from native common mapping")
        for key, shape in (("body_pos", (controls, 4, 24, 3)),
                           *((key, (controls, 4, 69)) for key in ("dof_pos", "dof_vel", "dof_forces"))):
            value = physical[key]
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    "Invalid complete FP32 body/DOF capture: " + key)
        position = ground["body_link_pos_w"]
        quaternion = ground["body_link_quat_w"]
        for key, value, width in (("body_link_pos_w", position, 3), ("body_link_quat_w", quaternion, 4)):
            require(value.shape == (controls, 4, 4, 24, width) and value.dtype == np.float32
                    and np.isfinite(value).all(), "Invalid complete native body channel: " + key)
        require(np.array_equal(physical["body_pos"], position[:, -1]),
                "Recorded all24 body positions differ from same-control native fourth substep")
        error = float(np.abs(np.linalg.norm(quaternion, axis=-1) - 1).max())
        require(error <= 1e-3, "Native all24/all-substep body quaternions are not unit XYZW")
    return {"status": "recorded_body_pose_verified", "all24_same_control_native_body_positions_equal": True,
            "body_position_samples": controls * 4 * 24, "native_body_quaternion_samples": controls * 4 * 4 * 24,
            "all_native_body_quaternion_norm_max_abs_error": error, "quaternion_norm_tolerance": 1e-3,
            "finite_FP32_69DOF_channels": ["dof_pos", "dof_vel", "dof_forces"],
            "DOF_channels_independently_matched_to_second_native_capture": False,
            "all_failures_and_reset_frames_included": True, "style_score_or_author_equivalence_established": False}


def initial_capture_link(initial_path, result, ground_path, plans, run):
    """Bind the live t0 record to initial plan/reference and first native clock."""
    initial_path = Path(initial_path).resolve()
    require(initial_path == run / "paired_initial_condition.json", "Initial snapshot must belong to this exact run")
    claimed = result["paired_initial_condition"]
    require(record(initial_path) == claimed["document"], "Initial document changed after SDK result publication")
    initial = read_snapshot(initial_path)
    require(initial["arrays"] == claimed["arrays"] and initial["contract"]["capture_kind"] == "live_original_world",
            "Require this run's exact live initial archive, not a CPU fixture")
    with np.load(initial["arrays"]["path"], allow_pickle=False) as data:
        position = data["common_rigid_body_pos"][:, 0, :2]
        velocity = data["common_rigid_body_vel"][:, 0, :2]
        episode = data["world_episode_ids"]
    with np.load(ground_path, allow_pickle=False) as ground:
        require(int(ground["physics_step_before"][0]) == initial["contract"]["clock"]["physics_step_after"]
                and np.array_equal(ground["episode_ids"][0], episode),
                "Initial capture counter/episode differs from first native control")
    for key, expected in (("raw_xy", position), ("reference_xy", position), ("reference_velocity_xy", velocity)):
        require(np.array_equal(plans[key][:4, 0], expected), "Initial plan/reference anchor differs from measured initial root: " + key)
    require(record(initial_path) == claimed["document"] and record(initial["arrays"]["path"]) == initial["arrays"],
            "Initial capture changed during CPU link check")
    return {"status": "live_initial_capture_linked", "document": claimed["document"], "arrays": claimed["arrays"],
            "all_four_first_raw_reference_position_velocity_anchors_equal": True,
            "first_native_control_counter_and_episode_equal": True,
            "paired_other_arm_comparison_performed": False}


def admit_external_closure(receipt_path, receipt, controls, post_owner, locomotion):
    """Accept direct observed exits or the existing fully bound parent chain."""
    require(type(receipt.get("actual_exit_code")) is int and receipt["actual_exit_code"] == 0,
            "Navigation did not exit successfully")
    if post_owner is None:
        require(isinstance(receipt.get("terminal_tool_chunk"), str) and receipt["terminal_tool_chunk"],
                "Require externally observed actual successful SDK exit")
        return {"kind": "directly_observed_navigation_exit", "terminal_tool_chunk": receipt["terminal_tool_chunk"]}, []
    require(controls == 6000, "Existing parent admission covers complete200s only")
    owner_identity = record(post_owner)
    owner = json.loads(Path(post_owner).read_text())
    require(owner.get("schema") == "shared_social.actual_post_training_owner_exit.v1"
            and type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
            and isinstance(owner.get("terminal_tool_chunk"), str) and owner["terminal_tool_chunk"],
            "Require externally observed actual successful completed-parent exit")
    # Reuse exact original parent/child/CPU-audit/array admission, including all
    # five successful child exits, equal measurement maps and sealed captures.
    _, _, _, identities = locomotion.admit(receipt_path, post_owner)
    return {"kind": "completed_parent_and_all_child_exits", "parent_exit_receipt": owner_identity,
            "parent_terminal_tool_chunk": owner["terminal_tool_chunk"],
            "original_full_parent_child_and_CPU_array_admission_reused": True}, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--control-steps", type=int, choices=(30, 6000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-owner-receipt", type=Path)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--paired-initial", type=Path)
    initial.add_argument("--historical-no-initial", action="store_true")
    args = parser.parse_args()
    require(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), "Start a fresh CPU-only audit process")
    reader_identity = record(__file__)
    dependency_identities = reader_dependencies()
    require(not args.output.exists(), "Use a new audit output file")
    base_path = args.output.with_name(args.output.stem + "_base.json")
    require(not base_path.exists(), "Base audit output already exists")
    actual_receipt = record(args.execution_receipt)
    receipt = json.loads(args.execution_receipt.read_text())
    locomotion, locomotion_identity = load_source(LOCOMOTION, LOCOMOTION_SHA)
    exit_closure, closure_identities = admit_external_closure(args.execution_receipt, receipt,
        args.control_steps, args.post_owner_receipt, locomotion)
    result_record = receipt["result"]
    require(record(result_record["path"]) == result_record, "Actual SDK result changed")
    result = json.loads(Path(result_record["path"]).read_text())
    require(result["actual_control_steps"] == args.control_steps and result["arguments"]["evaluate_steps"] == args.control_steps
            and result["arguments"]["num_envs"] == 4, "Registered and actual N4 control exposure differ")
    for key, expected in (("navigation_owned_contact_views", 5), ("navigation_SDK_sensor_objects", 24),
                          ("navigation_borrowed_backend_views", 1)):
        require(result[key] == {"observed_count": expected, "live_before_app_shutdown": 0},
                "Actual native ownership release incomplete: " + key)
    base, base_identity = load_source(AUDIT, AUDIT_SHA)
    source, base_proof = derive_base_audit(AUDIT.read_text(), args.control_steps)
    exec(compile(source, str(AUDIT) + ":registered_duration", "exec"), base.__dict__)
    saved_argv = sys.argv
    try:
        sys.argv = [str(AUDIT), "--execution-receipt", str(args.execution_receipt), "--output", str(base_path)]
        code = base.main()
    finally:
        sys.argv = saved_argv
    require(code == 0, "Reused base audit function failed")
    base_result = json.loads(base_path.read_text())
    helpers, helper_identities = locomotion.load_helpers()
    targets, targets_proof = duration_function(helpers[0].targets_from_saved_plans, args.control_steps)
    analyze, analyze_proof = duration_function(locomotion.analyze, args.control_steps)
    adapted_helpers = [SimpleNamespace(targets_from_saved_plans=targets, summary=helpers[0].summary), helpers[1]]
    response = analyze(result, base_result, base_result["arrays"], adapted_helpers)
    run = Path(result_record["path"]).parent
    body_pose = verify_recorded_body_pose(run / "physical_states.npz", run / "ground_support.npz", args.control_steps, result)
    with np.load(run / "reference_plans.npz", allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    joint = joint_plan_contract(plans, args.control_steps)
    if args.paired_initial:
        initial_link = initial_capture_link(args.paired_initial, result, run / "ground_support.npz", plans, run)
    else:
        require("paired_initial_condition" not in result, "A new paired run must not bypass its initial capture")
        initial_link = {"status": "historical_no_initial_record", "new_pairing_established": False}
    require(record(args.execution_receipt) == actual_receipt and record(result_record["path"]) == result_record
            and all(record(item["path"]) == item for item in base_result["arrays"])
            and all(record(item["path"]) == item for item in closure_identities), "SDK receipt/result/captures changed during audit")
    sources = [base_identity, locomotion_identity, *helper_identities, *dependency_identities]
    require(all(record(item["path"]) == item for item in [reader_identity, *sources]), "Original CPU audit sources changed")
    require(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), "CPU audit imported Torch or SDK")
    report = {"schema": "shared_social.all_registered_controls_native_CPU_audit.v1", "status": "passed",
              "reader": reader_identity, "execution_receipt": actual_receipt, "result": result_record,
              "actual_controls": args.control_steps, "actor_controls": args.control_steps * 4,
              "physics_substeps": args.control_steps * 4, "duration_s": args.control_steps / 30,
              "native_actor_substep_samples": args.control_steps * 16,
              "scope": "integration_smoke30" if args.control_steps == 30 else "complete_local_200s",
              "externally_observed_execution_closure": exit_closure,
              "base_audit": record(base_path), "base_audit_function_return_code": code,
              "original_sources": sources, "base_derivation": base_proof,
              "duration_AST_derivations": [targets_proof, analyze_proof], "native_response_support": response,
              "recorded_body_pose": body_pose,
              "joint_plan_events": joint, "initial_capture_link": initial_link,
              "all_failure_contact_and_reset_frames_retained": True, "Torch_imported": False, "SDK_executed_by_reader": False,
              "actual_CPU_process_exit_verified_by_reader": False, "paper_effect_achieved": False,
              "author_protocol_recovered": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "actual_controls", "actor_controls", "scope", "initial_capture_link")}))


if __name__ == "__main__":
    main()
