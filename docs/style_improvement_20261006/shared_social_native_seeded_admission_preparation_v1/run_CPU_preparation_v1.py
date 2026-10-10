"""Observe seeded raw/style CPU checks and frozen historical numerical regressions."""
from __future__ import annotations

from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_seeded_style as style  # noqa: E402
from scripts import audit_shared_social_native_seeded as raw_CLI  # noqa: E402
from scripts import evaluate_shared_social_native_seeded_style as style_CLI  # noqa: E402

record = style.metric.record
NAMES = ("CrowdSim/humanoid/analysis/native_navigation_seeded_audit.py",
    "CrowdSim/humanoid/analysis/native_navigation_seeded_style.py", "scripts/audit_shared_social_native_seeded.py",
    "scripts/evaluate_shared_social_native_seeded_style.py", "tests/test_native_seeded_audit.py",
    "tests/test_native_seeded_style.py")
HISTORICAL = (
    ("shared_social_native_raw_audit_preparation_v2", "925f5befbe3998a9e95c20976795c1c4788e07ddc51f864c691bf4a4e8c13395"),
    ("shared_social_native_style_preparation_v2", "b55d392658da9024391eb5e72868a812221155b581bafb5d6f8631ea09b1462e"),
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("xb") as stream:
        observed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_exit_code": observed.returncode,
        "actual_child_exit_observed": True, "log": record(log), "SDK_started": False}
    write_new(HERE / (role + "_execution.json"), row)
    return row


def historical_regression():
    """Reuse both sealed review functions; do not call new native admission."""
    helpers, originals = [], []
    for folder, sha in HISTORICAL:
        path = ROOT / "docs/style_improvement_20261005" / folder / "run_CPU_preparation_v2.py"
        identity = record(path)
        if identity["sha256"] != sha:
            raise ValueError("Sealed historical regression helper changed")
        namespace = ModuleType("_owned_seeded_" + folder)
        namespace.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path) + ":owned_review", "exec"), namespace.__dict__)
        namespace.HERE = HERE / folder
        namespace.HERE.mkdir(exist_ok=False)
        helpers.append(namespace)
        originals.append(identity)
    raw_helper, style_helper = helpers
    bound_raw = ModuleType("_historical_numerics_only_seed19")
    bound_raw.__dict__.update(style.raw.__dict__)
    bound_raw.prepare_kernels = partial(style.raw.prepare_kernels, seed=19)
    raw_helper.native = bound_raw
    raw_cases = [raw_helper.historical_kernel_review(arm) for arm in ("diffusion", "sfm")]
    style_helper.style = style
    style_helper.historical_metric_review()
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise ValueError("Historical CPU regression initialized CUDA")
    write_new(HERE / "historical_N4_regression.json", {"schema": "shared_social.seeded_historical_N4_numerics.v1",
        "unchanged_review_function_sources": originals, "raw_cases": raw_cases,
        "style_review": record(style_helper.HERE / "historical_N4_style_metric_preservation.json"),
        "private_base_kernel_receipt_selector_set_to_original_observed_N4_schema": True,
        "public_seeded_receipt_admission_relaxed": False, "formal_seeded_native_admission_executed": False,
        "historical_N4_cuboid_data_only": True, "new_native_capture_or_quality_measured": False,
        "Torch_CPU_reference_load_possible": True, "CUDA_initialized": False, "SDK_started": False})
    print(json.dumps({"status": "historical_actual_N4_raw_and_style_equal", "cases": 2,
                      "new_native_capture_or_quality_measured": False}))
    return 0


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    sources = [record(ROOT / name) for name in NAMES] + [record(__file__)]
    bindings = style.source_bindings()
    matrix = ROOT / "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/registered_S3_native_matrix.json"
    if record(matrix)["sha256"] != "e8a72e6d0971459eac68ac499a2f05b6253cda7804502afea6b5dc591730d50d":
        raise ValueError("Pre-registered156 native cohorts changed")
    tests = [sys.executable, "-m", "pytest", "-q", "tests/test_native_seeded_audit.py", "tests/test_native_seeded_style.py",
             "tests/test_native_navigation_audit.py", "tests/test_native_navigation_style.py"]
    commands = []
    for actors in (3, 15):
        for seed in (7, 19, 42):
            for planner in ("diffusion", "sfm"):
                for controls in (30, 6000):
                    commands.append({"role": f"raw_{planner}_N{actors}_C{controls}_seed{seed}", "command": [sys.executable,
                        str(raw_CLI.__file__), "--actors", str(actors), "--control-steps", str(controls),
                        "--planner", planner, "--seed", str(seed), "--check-sources"]})
            commands.append({"role": f"style_N{actors}_seed{seed}", "command": [sys.executable,
                str(style_CLI.__file__), "--actors", str(actors), "--seed", str(seed), "--check-sources"]})
    commands.extend(({"role": "tests_final", "command": tests}, {"role": "ruff_final", "command":
        ["/home/lfm/anaconda3/bin/ruff", "check", *(str(Path(row["path"]).relative_to(ROOT)) for row in sources)]},
        {"role": "historical_actual_N4_regression", "command": [sys.executable, str(Path(__file__)), "--historical-only"]}))
    write_new(HERE / "CPU_registration.json", {"schema": "shared_social.native_seeded_admission_CPU_registration.v1",
        "sources": sources, "bindings": bindings, "registered_future_matrix": record(matrix), "commands": commands,
        "seeded_raw_CLI_interface_proof": raw_CLI.prepared_CLI()[1],
        "seeded_style_CLI_interface_proof": style_CLI.prepared_CLI()[1],
        "mutation_test_scope": "New raw tests only; avoids detecting mutation merely via style's source SHA pin",
        "mutation": "require_registered_seed equality == -> !=; exact restore before full suite",
        "historical_N4_cases_are_numerical_regression_only": True, "new_native_SDK_execution_planned": False})
    source = ROOT / NAMES[0]
    before = source.read_bytes()
    needle = b'type(value) is int and value == seed'
    if before.count(needle) != 1:
        raise ValueError("Registered seed equality condition inventory changed")
    mutated = before.replace(needle, b'type(value) is int and value != seed', 1)
    (HERE / "seeded_raw_before_mutation.py").write_bytes(before)
    (HERE / "seeded_raw_mutated_TEST_ONLY.py").write_bytes(mutated)
    try:
        source.write_bytes(mutated)
        mutation = child("mutation_expected_failure", [sys.executable, "-m", "pytest", "-q",
                         "tests/test_native_seeded_audit.py"], env)
    finally:
        source.write_bytes(before)
    if mutation["actual_child_exit_code"] == 0 or source.read_bytes() != before:
        raise ValueError("Seed condition mutation stayed green or source was not restored")
    write_new(HERE / "mutation_review.json", {"actual_child": mutation, "mutation_detected": True,
        "exact_source_bytes_restored": True, "restored_sha256": hashlib.sha256(before).hexdigest()})
    observed = []
    for row in commands:
        completed = child(row["role"], row["command"], env)
        observed.append(completed)
        if completed["actual_child_exit_code"] != 0:
            raise ValueError("Actual CPU child failed; partial evidence retained: " + row["role"])
    for row in observed[:30]:
        doc = json.loads(Path(row["log"]["path"]).read_text())
        if doc["Torch_imported"] is not False or doc["SDK_started"] is not False:
            raise ValueError("Seeded source checks imported Torch/SDK")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if len(guards) != 23 or any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 source guard changed")
    if any(record(row["path"]) != row for row in sources) or style.source_bindings() != bindings:
        raise ValueError("Seeded admission source or frozen metric resources changed")
    write_new(HERE / "CPU_preparation_execution.json", {"schema": "shared_social.native_seeded_admission_CPU_execution.v1",
        "registration": record(HERE / "CPU_registration.json"), "children": observed,
        "mutation": record(HERE / "mutation_review.json"), "historical_regression": record(HERE / "historical_N4_regression.json"),
        "sources_and_frozen_resources_unchanged": True, "S3_guard_count": len(guards), "S3_sources_unchanged": True,
        "explicit_seed_raw_and_style_CPU_interfaces_prepared": True, "actual_seeded_native_SDK_exit_observed": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False, "default_model_promoted": False})
    print(json.dumps({"status": "native_seeded_admission_CPU_preparation_passed", "actual_children_exit0": len(observed),
        "receipt": record(HERE / "CPU_preparation_execution.json"), "new_native_capture_or_quality_measured": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(historical_regression() if sys.argv[1:] == ["--historical-only"] else main())
