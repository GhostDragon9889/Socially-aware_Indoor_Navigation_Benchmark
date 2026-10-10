"""Register seed interface checks, original resource matrix and cold S1 preflight.

The S1 preflights exercise compatibility only. The registered future physical
matrix requires the preselected completed S3 candidate and new versioned audit
admission. No SDK, policy inference or new human capture is performed here.
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
from scripts import evaluate_shared_social_native_seeded as entry  # noqa: E402


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("xb") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_exit_code": completed.returncode,
           "actual_child_exit_observed": True, "log": entry.record(log), "SDK_started": False}
    write_new(role + "_execution.json", row)
    return row


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    sources = [entry.record(ROOT / name) for name in ("scripts/evaluate_shared_social_native_seeded.py",
        "CrowdSim/humanoid/analysis/native_seed_protocol.py", "scripts/register_shared_social_native_seeds.py",
        "tests/test_native_seed_protocol.py")] + [entry.record(__file__)]
    bindings = entry.source_bindings()
    tests = [sys.executable, "-m", "pytest", "-q", "tests/test_native_seed_protocol.py", "tests/test_native_shared_social_evaluation.py"]
    commands = []
    for actors in (3, 15):
        for seed in (7, 19, 42):
            for planner in ("diffusion", "sfm"):
                commands.append({"role": f"{planner}_N{actors}_seed{seed}_source", "command": [sys.executable,
                    str(ROOT / "scripts/evaluate_shared_social_native_seeded.py"), "--actors", str(actors),
                    "--planner", planner, "--seed", str(seed), "--check-sources"]})
    base = ROOT / "docs/style_improvement_20261005"
    scenes = base / "shared_social_native_SFM_preparation_v1/scenes"
    S1 = base / "shared_social_evaluation_v1/popart_pilot2000_execution.json"
    diffusion = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt"
    diffusion_sha = "4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff"
    for actors, seed, planner in ((3, 19, "diffusion"), (15, 42, "sfm")):
        config = entry.record(scenes / f"Warehouse_N{actors:02}.json")
        commands.append({"role": f"S1_COMPAT_{planner}_N{actors}_seed{seed}_preflight", "command": [sys.executable,
            str(ROOT / "scripts/evaluate_shared_social_native_seeded.py"), "--actors", str(actors), "--planner", planner,
            "--seed", str(seed), "--scene-config", config["path"], "--scene-sha256", config["sha256"],
            "--execution-receipt", str(S1), "--diffusion-checkpoint", str(diffusion), "--diffusion-sha256", diffusion_sha,
            "--control-steps", "30", "--preflight-only", "--output",
            str(ROOT / "output/diffusion_amp" / f"native_seeded_CPU_preflight_only_v1_{planner}_N{actors}_seed{seed}")]})
    commands.append({"role": "matrix_registration", "command": [sys.executable,
        str(ROOT / "scripts/register_shared_social_native_seeds.py"), "--output-root",
        str(ROOT / "output/diffusion_amp/native_S3_seeded_registered_v1"), "--output", str(HERE / "registered_S3_native_matrix.json")]})
    ruff = ["/home/lfm/anaconda3/bin/ruff", "check", *(str(Path(row["path"]).relative_to(ROOT)) for row in sources)]
    write_new("CPU_registration.json", {"schema": "shared_social.native_seeded_CPU_registration.v1", "sources": sources,
        "bindings": bindings, "tests": tests, "commands": commands, "ruff": ruff,
        "mutation": "validate_seed first type guard is not int -> is int; exact byte restore",
        "cold_S1_preflights_are_compatibility_only": True, "future_matrix_preselects_S3_terminal_only": True,
        "new_native_SDK_or_quality_evaluation_planned": False, "fixed2848_training_changed": False})
    source = ROOT / "scripts/evaluate_shared_social_native_seeded.py"
    original = source.read_bytes()
    needle = b'if type(seed) is not int or not 0 <= seed < 2**32:'
    if original.count(needle) != 1:
        raise ValueError("Seed type mutation inventory changed")
    mutated = original.replace(needle, b'if type(seed) is int or not 0 <= seed < 2**32:', 1)
    (HERE / "seeded_entry_before_mutation.py").write_bytes(original)
    (HERE / "seeded_entry_mutated_TEST_ONLY.py").write_bytes(mutated)
    try:
        source.write_bytes(mutated)
        mutation = child("mutation_expected_failure", tests, env)
    finally:
        source.write_bytes(original)
    if mutation["actual_child_exit_code"] == 0 or source.read_bytes() != original:
        raise ValueError("Seed mutation not detected or exact restore failed")
    write_new("mutation_review.json", {"actual_child": mutation, "mutation_detected": True,
        "exact_source_bytes_restored": True, "restored_sha256": hashlib.sha256(original).hexdigest(), "SDK_started": False})
    observed = []
    for row in [*commands, {"role": "tests_final", "command": tests}, {"role": "ruff_final", "command": ruff}]:
        result = child(row["role"], row["command"], env)
        observed.append(result)
        if result["actual_child_exit_code"] != 0:
            raise ValueError("Actual seed CPU child failed: " + row["role"])
    from CrowdSim.humanoid.application.configuration import parse_amp_args
    comparisons = []
    for actors, seed, planner in ((3, 19, "diffusion"), (15, 42, "sfm")):
        current = json.loads((HERE / f"S1_COMPAT_{planner}_N{actors}_seed{seed}_preflight.log").read_text())
        previous = json.loads((base / "shared_social_native_initial_preparation_v3" / f"{planner}_N{actors}_preflight.log").read_text())
        args, old = vars(parse_amp_args(current["driver_arguments"])), vars(parse_amp_args(previous["driver_arguments"]))
        if args["seed"] != seed or old["seed"] != 7 or args["num_envs"] != actors:
            raise ValueError("Actual prepared driver seed/cohort differs")
        args.pop("seed"), old.pop("seed")
        args.pop("output"), old.pop("output")
        if args != old or current["physical_execution_verified"] is not False:
            raise ValueError("Another driver argument changed or CPU preflight claims physical execution")
        comparisons.append({"actors": actors, "planner": planner, "seed": seed,
            "all_driver_arguments_except_seed_output_equal_to_sealed_seed7_preflight": True,
            "native_paired_driver_code_math_unchanged": True, "actual_native_SDK_execution_verified": False})
    write_new("actual_seed_forwarding_CPU_comparison.json", {"schema": "shared_social.native_seed_driver_args_CPU_comparison.v1",
        "cases": comparisons, "S1_resource_compatibility_only": True, "new_native_capture_or_quality_measured": False})
    for row in observed[:12]:
        result = json.loads(Path(row["log"]["path"]).read_text())
        if result["Torch_imported"] is not False or result["SDK_started"] is not False:
            raise ValueError("Source preparation imported Torch/SDK")
    if any(entry.record(row["path"]) != row for row in sources) or entry.source_bindings() != bindings:
        raise ValueError("Seed sources changed")
    owner = json.loads((base / "shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(entry.record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 source guard changed")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_seeded_CPU_child_execution.v1",
        "registration": entry.record(HERE / "CPU_registration.json"), "children": observed,
        "mutation": entry.record(HERE / "mutation_review.json"),
        "actual_driver_argument_comparison": entry.record(HERE / "actual_seed_forwarding_CPU_comparison.json"),
        "registered_future_matrix": entry.record(HERE / "registered_S3_native_matrix.json"),
        "source_bindings_unchanged": True, "S3_training_guard_count": len(guards), "S3_training_guards_unchanged": True,
        "new_seeded_raw_style_admission_complete": False, "new_native_capture_or_quality_measured": False,
        "paper_effect_achieved": False, "default_model_promoted": False})
    print(json.dumps({"status": "native_seeded_CPU_preparation_passed", "actual_children_exit0": len(observed),
        "new_native_capture_or_quality_measured": False, "receipt": entry.record(HERE / "CPU_preparation_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
