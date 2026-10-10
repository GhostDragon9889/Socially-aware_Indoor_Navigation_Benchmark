"""Register and observe CPU checks; never start Kit, CUDA or a physical World."""
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
from CrowdSim.humanoid.analysis import native_paired_initial as initial  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_returncode": result.returncode,
           "actual_child_exit_observed": True, "log": record(log), "CPU_only": True}
    write_new(role + "_execution.json", row)
    return row


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    entry = ROOT / "scripts/evaluate_shared_social_native_paired.py"
    scenes = ROOT / "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/scenes"
    receipt = ROOT / "docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json"
    diffusion = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt"
    diffusion_sha = "4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff"
    cases = [(3, "diffusion"), (15, "sfm")]
    selected_sources = [record(ROOT / name) for name in (
        "CrowdSim/humanoid/analysis/native_paired_initial.py", "CrowdSim/humanoid/analysis/paired_initial.py",
        "scripts/evaluate_shared_social_native_paired.py", "scripts/compare_shared_social_native_initial_conditions.py",
        "scripts/evaluate_shared_social_native.py", "tests/test_native_paired_initial.py")]
    selected_sources.extend(initial.source_bindings().values())
    selected_sources.append(record(__file__))
    test_command = [sys.executable, "-m", "pytest", "-q", "tests/test_native_paired_initial.py"]
    commands = []
    resource_cases = []
    for actors, planner in cases:
        config = record(scenes / f"Warehouse_N{actors:02}.json")
        prefix = [sys.executable, str(entry), "--actors", str(actors), "--planner", planner]
        commands.append({"role": f"{planner}_N{actors}_source", "command": prefix + ["--check-sources"]})
        commands.append({"role": f"{planner}_N{actors}_preflight", "command": prefix + [
            "--scene-config", config["path"], "--scene-sha256", config["sha256"],
            "--execution-receipt", str(receipt), "--diffusion-checkpoint", str(diffusion),
            "--diffusion-sha256", diffusion_sha, "--control-steps", "30", "--preflight-only",
            "--output", str(ROOT / "output/diffusion_amp" / f"native_initial_preflight_only_v2_{planner}_N{actors}")]})
        resource_cases.append({"actors": actors, "planner_source_preparation": planner, "configuration": config})
    final_tests = test_command + ["tests/test_shared_social_paired_initial.py",
        "tests/test_native_social_force_runtime.py", "tests/test_native_shared_social_evaluation.py"]
    ruff = ["/home/lfm/anaconda3/bin/ruff", "check", "CrowdSim/humanoid/analysis/native_paired_initial.py",
        "scripts/evaluate_shared_social_native_paired.py", "scripts/compare_shared_social_native_initial_conditions.py",
        "tests/test_native_paired_initial.py", str(Path(__file__).relative_to(ROOT))]
    write_new("CPU_registration.json", {"schema": "shared_social.native_initial_CPU_registration.v1",
        "sources": selected_sources, "cases": resource_cases, "commands": commands,
        "final_tests_command": final_tests, "ruff_command": ruff,
        "mutation_command": test_command, "mutation": "require_live -> not require_live; then exact restore",
        "CPU_only": True, "synthetic_test_archives_not_live": True, "actual_native_SDK_execution_planned": False,
        "training_parameters_or_sources_modified": False, "paper_effect_achieved": False})

    # Actual admitted map channels only. No synthetic body/RNG rows are attached
    # to these resources and no initial-state or live-World receipt is published.
    import numpy as np
    map_cases = []
    for case in resource_cases:
        from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
        request = preflight_native_scene(case["configuration"]["path"], actors=case["actors"])
        scene = initial.admit_native_resource_scene(request)
        protocol = initial.native_protocol(case["actors"], scene)
        target = HERE / f"Warehouse_N{case['actors']:02}_map_channels_only.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **initial.map_arrays(scene))
        with np.load(target, allow_pickle=False) as payload:
            for key in payload.files:
                if initial.original._description(payload[key]) != protocol["map_channels"][key]:
                    raise ValueError("Actual CPU map archive channel differs")
        _, proof = initial.count_backend(case["actors"])
        map_cases.append({"request": request, "protocol": protocol, "count_derivation": proof,
            "archive": record(target), "map_channels_only": True, "human_body_or_RNG_captured": False,
            "actual_native_SDK_execution_verified": False})
    write_new("actual_map_CPU_contracts.json", {"schema": "shared_social.native_initial_map_CPU_contracts.v1",
        "cases": map_cases, "Torch_imported": "torch" in sys.modules, "SDK_started": False,
        "live_initial_pairing_verified": False, "paper_effect_achieved": False})
    if "torch" in sys.modules:
        raise ValueError("CPU resource source preparation unexpectedly imported Torch")

    # Required review experiment: preserve the exact candidate, invert one
    # admission condition, observe test failure, then restore bytes in finally.
    source = ROOT / "CrowdSim/humanoid/analysis/native_paired_initial.py"
    baseline = source.read_bytes()
    old, new = b"if require_live and any(", b"if not require_live and any("
    if baseline.count(old) != 1:
        raise ValueError("Ambiguous strict live-admission mutation")
    (HERE / "native_initial_before_mutation.py").write_bytes(baseline)
    mutated = baseline.replace(old, new, 1)
    (HERE / "native_initial_mutated_test_only.py").write_bytes(mutated)
    try:
        source.write_bytes(mutated)
        mutation = run_child("mutation_expected_failure", test_command, env)
    finally:
        source.write_bytes(baseline)
    if source.read_bytes() != baseline or mutation["actual_child_returncode"] == 0:
        raise ValueError("Mutation did not fail the suite or exact restore failed")
    write_new("mutation_review.json", {"condition_before": old.decode(), "condition_during_test": new.decode(),
        "actual_child": mutation, "mutation_detected": True, "restored_exact_source_bytes": True,
        "restored_sha256": hashlib.sha256(baseline).hexdigest(), "SDK_started": False})

    observed = []
    for row in [*commands, {"role": "tests_final", "command": final_tests}, {"role": "ruff_final", "command": ruff}]:
        child = run_child(row["role"], row["command"], env)
        observed.append(child)
        if child["actual_child_returncode"] != 0:
            raise ValueError("Actual CPU preparation child failed: " + row["role"])
    for row in observed[:4]:
        report = json.loads(Path(row["log"]["path"]).read_text())
        if row["role"].endswith("source"):
            if report["Torch_imported"] is not False or report["SDK_started"] is not False:
                raise ValueError("Source check started Torch/SDK")
        elif "native_initial_capture" not in report["native_current_migration"]:
            raise ValueError("Preflight is missing the actual new native initial hook binding")
    if any(record(row["path"]) != row for row in selected_sources):
        raise ValueError("CPU preparation sources changed")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 owner source guard changed")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_initial_CPU_child_execution.v1",
        "registration": record(HERE / "CPU_registration.json"), "observed_children": observed,
        "mutation": record(HERE / "mutation_review.json"), "actual_map_contracts": record(HERE / "actual_map_CPU_contracts.json"),
        "CPU_preparation_passed": True, "sources_unchanged": True, "active_training_guards_unchanged": True,
        "active_training_guard_count": len(guards), "native_SDK_exit_observed": False,
        "live_native_initial_pairing_verified": False, "native200s_quality_measured": False,
        "paper_effect_achieved": False, "default_model_promoted": False})
    print(json.dumps({"status": "native_initial_CPU_preparation_passed", "actual_children_exit0": len(observed),
        "mutation_detected_and_exact_source_restored": True, "actual_maps": len(map_cases),
        "live_native_initial_pairing_verified": False, "receipt": record(HERE / "CPU_preparation_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
