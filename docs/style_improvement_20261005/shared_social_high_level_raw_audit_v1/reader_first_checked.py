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
LOCOMOTION_SHA = "2c975fe2b5bbe82435ff29dee2ed9aefe219f91270b168f698eb9eebf8a745be"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--control-steps", type=int, choices=(30, 6000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--paired-initial", type=Path)
    initial.add_argument("--historical-no-initial", action="store_true")
    args = parser.parse_args()
    require(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), "Start a fresh CPU-only audit process")
    require(not args.output.exists(), "Use a new audit output file")
    base_path = args.output.with_name(args.output.stem + "_base.json")
    require(not base_path.exists(), "Base audit output already exists")
    actual_receipt = record(args.execution_receipt)
    receipt = json.loads(args.execution_receipt.read_text())
    require(receipt.get("actual_exit_code") == 0 and isinstance(receipt.get("terminal_tool_chunk"), str)
            and receipt["terminal_tool_chunk"], "Require externally observed actual successful SDK exit")
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
    locomotion, locomotion_identity = load_source(LOCOMOTION, LOCOMOTION_SHA)
    helpers, helper_identities = locomotion.load_helpers()
    targets, targets_proof = duration_function(helpers[0].targets_from_saved_plans, args.control_steps)
    analyze, analyze_proof = duration_function(locomotion.analyze, args.control_steps)
    adapted_helpers = [SimpleNamespace(targets_from_saved_plans=targets, summary=helpers[0].summary), helpers[1]]
    response = analyze(result, base_result, base_result["arrays"], adapted_helpers)
    run = Path(result_record["path"]).parent
    with np.load(run / "reference_plans.npz", allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    joint = joint_plan_contract(plans, args.control_steps)
    if args.paired_initial:
        initial_link = initial_capture_link(args.paired_initial, result, run / "ground_support.npz", plans, run)
    else:
        require("paired_initial_condition" not in result, "A new paired run must not bypass its initial capture")
        initial_link = {"status": "historical_no_initial_record", "new_pairing_established": False}
    require(record(args.execution_receipt) == actual_receipt and record(result_record["path"]) == result_record
            and all(record(item["path"]) == item for item in base_result["arrays"]), "SDK receipt/result/captures changed during audit")
    sources = [base_identity, locomotion_identity, *helper_identities]
    require(all(record(item["path"]) == item for item in sources), "Original frozen audit sources changed")
    require(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), "CPU audit imported Torch or SDK")
    report = {"schema": "shared_social.all_registered_controls_native_CPU_audit.v1", "status": "passed",
              "reader": record(__file__), "execution_receipt": actual_receipt, "result": result_record,
              "actual_controls": args.control_steps, "actor_controls": args.control_steps * 4,
              "physics_substeps": args.control_steps * 4, "duration_s": args.control_steps / 30,
              "scope": "integration_smoke30" if args.control_steps == 30 else "complete_local_200s",
              "base_audit": record(base_path), "base_audit_function_return_code": code,
              "original_sources": sources, "base_derivation": base_proof,
              "duration_AST_derivations": [targets_proof, analyze_proof], "native_response_support": response,
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
