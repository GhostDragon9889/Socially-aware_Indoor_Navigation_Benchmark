"""Register native style CPU contracts and actual historical metric equality.

Historical N4 captures remain historical cuboid data. The formal new native
SDK/resource admission is deliberately not called in the numerical regression.
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
from CrowdSim.humanoid.analysis import native_navigation_style as style  # noqa: E402


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("xb") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_exit_code": completed.returncode,
           "actual_child_exit_observed": True, "log": style.metric.record(log), "SDK_started": False}
    write_new(role + "_execution.json", row)
    return row


def historical_metric_review():
    import numpy as np

    metric = style.metric
    previous = ROOT / "docs/style_improvement_20261005/shared_social_S1_paired_actual_v1"
    actual = metric.bound(metric.record(previous / "navigation_style_pair_execution.json"))
    metric.require(type(actual["actual_exit_code"]) is int and actual["actual_exit_code"] == 0
                   and actual["terminal_tool_chunk"], "Require the original actually completed CPU style evaluation")
    before = metric.bound(actual["report"])
    runs, inputs, summaries = {}, [metric.record(previous / "navigation_style_pair_execution.json"), actual["report"]], {}
    for arm in ("diffusion", "sfm"):
        owner = previous / (arm + "200s_raw_CPU_execution.json")
        original_samples, result, base, admission, identities = metric.read_completed_run(owner)
        geometry = result["physics"]["scene_geometry"]
        metric.require(result["social_entry"]["schema"] == "local.neutral_smpl.social_four_actor_navigation_evaluation.v1"
            and geometry["schema"] == "navisaaclab.reconstruction_scene.v1"
            and geometry["name"] == "four_actor_bounded_obstacle_fixture"
            and any(result.get(key) is not None for key in
                ("native_scene", "native_scene_provider", "native_scene_live_observation")) is False,
            "This is only the unchanged historical N4 cuboid numerical regression")
        metric.validate_planner(arm, result)
        with np.load(admission["arrays"][0]["path"], allow_pickle=False) as physical, \
                np.load(admission["arrays"][1]["path"], allow_pickle=False) as ground:
            samples, capture = style.capture_samples(physical, ground, result, base, actors=4)
        for key in ("local_pose", "local_pose_velocity", "speed_m_s", "end_frame_indices", "actor_indices"):
            metric.require(np.array_equal(samples[key], original_samples[key]), "Historical style sample changed: " + key)
        metric.require(samples["summary"] == before["runs"][arm]["inclusive_summary"], "Historical inclusive summary changed")
        runs[arm], summaries[arm] = samples, capture
        inputs.extend(identities)
    reference, pose, centers, reference_admission, reference_inputs = metric.load_frozen_reference()
    rows = {}
    for arm, samples in runs.items():
        pose_scores = pose.compare_pose_only(samples["local_pose"], reference["local_pose"], centers)
        motion_scores = metric.compare_speed_conditioned(samples, reference)
        metric.require(pose_scores == before["runs"][arm]["pose45"], "Historical full pose45 scores changed")
        metric.require(motion_scores == before["runs"][arm]["style90_all_population"], "Historical full motion90 scores changed")
        rows[arm] = {"inclusive_summary_every_field_equal": True, "pose45_every_field_equal": True,
            "motion90_population_every_field_equal": True, "capture_scope": summaries[arm],
            "pose45": pose_scores, "style90_all_population": motion_scores}
    matched = metric.matched_motion_bins(runs, reference)
    metric.require(matched == before["style90_matched_pair"], "Historical common-bin paired scores changed")
    inputs.extend(reference_inputs)
    metric.require(all(metric.record(row["path"]) == row for row in inputs), "Historical actual evidence changed")
    metric.require(reference_admission == before["reference"], "Frozen actual reference admission changed")
    write_new("historical_N4_style_metric_preservation.json", {"schema": "shared_social.historical_N4_style_equality.v1",
        "original_actual_style_execution": inputs[0], "original_style_report": actual["report"], "inputs": inputs,
        "runs": rows, "style90_matched_pair": matched, "paired_common_bin_every_field_equal": True,
        "reference": reference_admission, "reference_or_basis_refit": False, "Test34_read": False,
        "historical_N4_cuboid_fixture_only": True, "formal_new_native_SDK_resource_admission_run": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False})


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    source_names = ("CrowdSim/humanoid/analysis/native_navigation_style.py", "scripts/evaluate_shared_social_native_style.py",
                    "tests/test_native_navigation_style.py")
    sources = [style.metric.record(ROOT / name) for name in source_names] + [style.metric.record(__file__)]
    bindings = style.source_bindings()
    tests = [sys.executable, "-m", "pytest", "-q", "tests/test_native_navigation_style.py", "tests/test_shared_social_navigation_style.py"]
    commands = [{"role": f"source_N{actors}", "command": [sys.executable,
        str(ROOT / "scripts/evaluate_shared_social_native_style.py"), "--actors", str(actors), "--check-sources"]}
        for actors in (3, 15)]
    ruff = ["/home/lfm/anaconda3/bin/ruff", "check", *(str(Path(row["path"]).relative_to(ROOT)) for row in sources)]
    write_new("CPU_registration.json", {"schema": "shared_social.native_style_CPU_registration.v1", "sources": sources,
        "bindings": bindings, "commands": commands, "tests": tests, "ruff": ruff,
        "mutation": "First owner actual-exit guard is int -> is not int; exact byte restore",
        "historical_actual_N4_cases": ["diffusion", "sfm"], "metrics_or_basis_refit_planned": False,
        "new_native_SDK_or_quality_evaluation_planned": False, "fixed2848_training_changed": False})
    source = ROOT / source_names[0]
    original = source.read_bytes()
    needle = b'type(owner.get("actual_exit_code")) is int'
    if original.count(needle) != 2:
        raise ValueError("Exit guard literal inventory changed")
    (HERE / "native_style_reader_before_mutation.py").write_bytes(original)
    mutated = original.replace(needle, b'type(owner.get("actual_exit_code")) is not int', 1)
    (HERE / "native_style_reader_mutated_TEST_ONLY.py").write_bytes(mutated)
    try:
        source.write_bytes(mutated)
        mutation = child("mutation_expected_failure", tests, env)
    finally:
        source.write_bytes(original)
    if mutation["actual_child_exit_code"] == 0 or source.read_bytes() != original:
        raise ValueError("Mutation not detected or exact restore failed")
    write_new("mutation_review.json", {"actual_child": mutation, "mutation_detected": True,
        "exact_source_bytes_restored": True, "restored_sha256": hashlib.sha256(original).hexdigest(), "SDK_started": False})
    observed = []
    for row in [*commands, {"role": "tests_final", "command": tests}, {"role": "ruff_final", "command": ruff}]:
        completed = child(row["role"], row["command"], env)
        observed.append(completed)
        if completed["actual_child_exit_code"] != 0:
            raise ValueError("Actual CPU child failed: " + row["role"])
    historical_metric_review()
    if any(style.metric.record(row["path"]) != row for row in sources) or style.source_bindings() != bindings:
        raise ValueError("Style source/reference resource changed")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(style.metric.record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 source guard changed")
    for row in observed[:2]:
        report = json.loads(Path(row["log"]["path"]).read_text())
        if report["Torch_imported"] is not False or report["SDK_started"] is not False:
            raise ValueError("Style source preparation imported Torch/SDK")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise ValueError("Actual CPU historical style initialized CUDA")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_style_CPU_child_execution.v1",
        "registration": style.metric.record(HERE / "CPU_registration.json"), "children": observed,
        "mutation": style.metric.record(HERE / "mutation_review.json"),
        "historical_style_equality": style.metric.record(HERE / "historical_N4_style_metric_preservation.json"),
        "source_and_reference_bindings_unchanged": True, "S3_training_guard_count": len(guards),
        "S3_training_guards_unchanged": True, "CUDA_initialized": False, "SDK_started": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False, "default_model_promoted": False})
    print(json.dumps({"status": "native_style_CPU_preparation_passed", "actual_children_exit0": len(observed),
        "actual_historical_style_cases_equal": 2, "new_native_capture_or_quality_measured": False,
        "receipt": style.metric.record(HERE / "CPU_preparation_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
