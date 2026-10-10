"""Register CPU source/tests and actual historical kernel equality checks.

Historical N4 fixture arrays are used only to check metric preservation. They
never pass the new native scene/initial/SDK admission and are not new results.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_audit as native  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_returncode": result.returncode,
           "actual_child_exit_observed": True, "log": record(log), "CPU_only": True}
    write_new(role + "_execution.json", row)
    return row


def historical_kernel_review(arm):
    import numpy as np
    previous = ROOT / "docs/style_improvement_20261005/shared_social_S1_paired_actual_v1"
    receipt_path = previous / (arm + "200s_execution.json")
    old_audit_path = previous / (arm + "200s_raw_CPU_audit_v1.json")
    receipt, old = json.loads(receipt_path.read_text()), json.loads(old_audit_path.read_text())
    if (receipt["schema"] != "shared_social.actual_navigation200s_execution.v1"
            or type(receipt["actual_exit_code"]) is not int or receipt["actual_exit_code"] != 0
            or not receipt["terminal_tool_chunk"] or old["actual_controls"] != 6000 or old["actor_controls"] != 24000
            or record(receipt["result"]["path"]) != receipt["result"] or old["result"] != receipt["result"]):
        raise ValueError("Require the unchanged actual historical full N4 fixture")
    result = json.loads(Path(receipt["result"]["path"]).read_text())
    geometry = result["physics"]["scene_geometry"]
    if (result["arguments"]["num_envs"] != 4
            or result["social_entry"]["schema"] != "local.neutral_smpl.social_four_actor_navigation_evaluation.v1"
            or geometry["schema"] != "navisaaclab.reconstruction_scene.v1"
            or geometry["name"] != "four_actor_bounded_obstacle_fixture"
            or Path(result["arguments"]["scene_config"]).resolve()
            != ROOT / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
            or any(result.get(key) is not None for key in
                   ("native_scene", "native_scene_provider", "native_scene_live_observation"))):
        raise ValueError("This regression is explicitly the historical cuboid fixture, not a native result")
    base_before = json.loads(Path(old["base_audit"]["path"]).read_text())
    if record(old["base_audit"]["path"]) != old["base_audit"]:
        raise ValueError("Original historical base measurements changed")
    paths = [Path(row["path"]) for row in base_before["arrays"]]
    if [record(path) for path in paths] != base_before["arrays"]:
        raise ValueError("Actual historical raw captures changed")
    kernels = native.prepare_kernels(actors=4, controls=6000)
    # The private numerical kernel's receipt boundary is set explicitly to
    # the already observed old fixture. Formal native admission is NOT called.
    kernels.base._native_execution_schema = "shared_social.actual_navigation200s_execution.v1"
    target = HERE / f"{arm}_historical_N4_kernel_base.json"
    saved = sys.argv
    try:
        sys.argv = ["historical_N4_metric_kernel_only", "--execution-receipt", str(receipt_path), "--output", str(target)]
        code = kernels.base.main()
    finally:
        sys.argv = saved
    if code != 0:
        raise ValueError("Historical numerical kernel failed")
    after = json.loads(target.read_text())
    before_metrics = {key: value for key, value in base_before.items() if key != "schema"}
    after_metrics = {key: value for key, value in after.items() if key != "schema"}
    body = kernels.body(paths[0], paths[1], 6000, result)
    response = kernels.analyze(result, after, after["arrays"], kernels.helpers)
    with np.load(paths[2], allow_pickle=False) as data:
        plans = {key: data[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    joint = kernels.joint(plans, 6000)
    joint_before = dict(old["joint_plan_events"])
    joint_before["all_registered_actor_rows_per_event"] = joint_before.pop("all_four_actor_rows_per_event")
    if (before_metrics != after_metrics or body != old["recorded_body_pose"]
            or response != old["native_response_support"] or joint != joint_before):
        raise ValueError("Original historical metric calculations changed: " + arm)
    if [record(path) for path in paths] != base_before["arrays"]:
        raise ValueError("Historical captures changed during CPU regression")
    return {"arm": arm, "original_actual_SDK_receipt": record(receipt_path), "original_actual_raw_audit": record(old_audit_path),
        "original_base_audit": old["base_audit"], "arrays": after["arrays"], "adapted_base_audit": record(target),
        "base_every_field_except_schema_equal": True, "all_body_pose_fields_equal": True,
        "all_response_support_fields_equal": True, "joint_fields_equal_after_explicit_actor_label_rename": True,
        "actor_controls": 24000, "historical_N4_cuboid_fixture_only": True,
        "formal_native_SDK_and_initial_admission_run": False, "new_native_capture_or_quality_measured": False}


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    sources = [record(ROOT / name) for name in ("CrowdSim/humanoid/analysis/native_navigation_audit.py",
               "scripts/audit_shared_social_native.py", "tests/test_native_navigation_audit.py")]
    sources.append(record(__file__))
    baseline_bindings = native.source_bindings()
    tests = [sys.executable, "-m", "pytest", "-q", "tests/test_native_navigation_audit.py", "tests/test_shared_social_high_level_audit.py"]
    commands = []
    for actors, controls, planner in ((3, 30, "diffusion"), (3, 6000, "diffusion"), (15, 30, "sfm"), (15, 6000, "sfm")):
        commands.append({"role": f"{planner}_N{actors}_C{controls}_sources", "command": [sys.executable,
            str(ROOT / "scripts/audit_shared_social_native.py"), "--actors", str(actors), "--control-steps", str(controls),
            "--planner", planner, "--check-sources"]})
    ruff = ["/home/lfm/anaconda3/bin/ruff", "check", *(str(Path(row["path"]).relative_to(ROOT)) for row in sources)]
    write_new("CPU_registration.json", {"schema": "shared_social.native_raw_CPU_registration.v1", "sources": sources,
        "kernel_source_bindings": baseline_bindings, "tests": tests, "commands": commands, "ruff": ruff,
        "mutation": "First native exit guard: is int -> is not int; then exact byte restore",
        "historical_kernel_cases": ["diffusion", "sfm"], "historical_N4_cases_are_metric_preservation_only": True,
        "actual_new_native_SDK_or_quality_evaluation_planned": False, "CPU_only": True,
        "training_parameters_or_sources_modified": False, "paper_effect_achieved": False})
    source = ROOT / "CrowdSim/humanoid/analysis/native_navigation_audit.py"
    original = source.read_bytes()
    needle = b'and type(receipt.get("actual_exit_code")) is int'
    changed = b'and type(receipt.get("actual_exit_code")) is not int'
    if original.count(needle) != 2:
        raise ValueError("Native exit guard inventory changed")
    (HERE / "native_raw_reader_before_mutation.py").write_bytes(original)
    mutation_source = original.replace(needle, changed, 1)
    (HERE / "native_raw_reader_mutated_test_only.py").write_bytes(mutation_source)
    try:
        source.write_bytes(mutation_source)
        mutation = child("mutation_expected_failure", tests, env)
    finally:
        source.write_bytes(original)
    if mutation["actual_child_returncode"] == 0 or source.read_bytes() != original:
        raise ValueError("Native exit mutation was not detected or exact restore failed")
    write_new("mutation_review.json", {"observed_child": mutation, "mutation_detected": True,
        "exact_source_bytes_restored": True, "restored_sha256": hashlib.sha256(original).hexdigest(), "SDK_started": False})
    observed = []
    for row in [*commands, {"role": "tests_final", "command": tests}, {"role": "ruff_final", "command": ruff}]:
        result = child(row["role"], row["command"], env)
        observed.append(result)
        if result["actual_child_returncode"] != 0:
            raise ValueError("Actual CPU child failed: " + row["role"])
    historical = [historical_kernel_review(arm) for arm in ("diffusion", "sfm")]
    write_new("historical_N4_metric_preservation.json", {"schema": "shared_social.historical_N4_raw_kernel_equality.v1",
        "cases": historical, "CPU_only": True, "original_captures_unchanged": True,
        "new_native_SDK_capture_or_quality_measured": False, "paper_effect_achieved": False})
    for row in observed[:4]:
        result = json.loads(Path(row["log"]["path"]).read_text())
        if result["Torch_imported"] is not False or result["SDK_started"] is not False:
            raise ValueError("Native CPU source check imported Torch/SDK")
    if any(record(row["path"]) != row for row in sources) or native.source_bindings() != baseline_bindings:
        raise ValueError("CPU source or immutable kernel dependency changed")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 training source guard changed")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_raw_CPU_child_execution.v1",
        "registration": record(HERE / "CPU_registration.json"), "children": observed,
        "mutation": record(HERE / "mutation_review.json"),
        "historical_N4_equality": record(HERE / "historical_N4_metric_preservation.json"),
        "sources_unchanged": True, "active_training_guards_unchanged": True, "training_guard_count": len(guards),
        "native_actual_capture_audited": False, "actual_new_SDK_exit_observed": False,
        "paper_effect_achieved": False, "default_model_promoted": False})
    print(json.dumps({"status": "native_raw_CPU_preparation_passed", "actual_children_exit0": len(observed),
        "actual_historical_N4_metric_preservation_cases": 2, "new_native_SDK_capture_audited": False,
        "receipt": record(HERE / "CPU_preparation_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
