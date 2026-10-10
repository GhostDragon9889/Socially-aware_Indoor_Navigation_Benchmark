"""Observe native case/exit bridge CPU checks without launching SDK work."""
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
from CrowdSim.humanoid.application import native_execution_bridge as bridge  # noqa: E402

record = bridge.record
NAMES = ("CrowdSim/humanoid/application/native_execution_bridge.py",
         "scripts/prepare_shared_social_native_execution.py", "tests/test_native_execution_bridge.py")


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    path = HERE / (role + ".log")
    with path.open("xb") as stream:
        observed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {"role": role, "command": command, "actual_child_exit_code": observed.returncode,
           "actual_child_exit_observed": True, "log": record(path), "SDK_started": False}
    write_new(role + "_execution.json", row)
    return row


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    sources = [record(ROOT / name) for name in NAMES] + [record(__file__)]
    matrix = record(bridge.MATRIX)
    bindings = {"raw": bridge.raw.source_bindings(), "style": bridge.style.source_bindings()}
    cases = ("Office_N03_seed19", "Warehouse_N15_seed42", "Full_Warehouse_N03_seed7", "Hospital_N11_seed19")
    tests = [sys.executable, "-m", "pytest", "-q", "tests/test_native_execution_bridge.py"]
    commands = [{"role": "source_" + case, "command": [sys.executable, str(ROOT / NAMES[1]),
        "--case-id", case, "--mode", "source-preflight"]} for case in cases]
    commands.extend(({"role": "tests_final", "command": tests}, {"role": "ruff_final", "command":
        ["/home/lfm/anaconda3/bin/ruff", "check", *(str(Path(row["path"]).relative_to(ROOT)) for row in sources)]}))
    write_new("CPU_registration.json", {"schema": "shared_social.native_execution_bridge_CPU_registration.v1",
        "sources": sources, "bindings": bindings, "matrix": matrix, "commands": commands,
        "mutation": "Tool exit type is int -> is not int; exact restore", "actual_S3_execution_readiness_checked": False,
        "actual_native_SDK_exit_publication_tested": False, "GPU_or_SDK_started": False})
    path = ROOT / NAMES[0]
    original = path.read_bytes()
    needle = b'type(response.get("exit_code")) is int'
    if original.count(needle) != 1:
        raise ValueError("Actual terminal type guard inventory changed")
    mutated = original.replace(needle, b'type(response.get("exit_code")) is not int', 1)
    (HERE / "bridge_before_mutation.py").write_bytes(original)
    (HERE / "bridge_mutated_TEST_ONLY.py").write_bytes(mutated)
    try:
        path.write_bytes(mutated)
        mutation = child("mutation_expected_failure", tests, env)
    finally:
        path.write_bytes(original)
    if mutation["actual_child_exit_code"] == 0 or path.read_bytes() != original:
        raise ValueError("Tool exit type mutation not detected or exact restore failed")
    write_new("mutation_review.json", {"actual_child": mutation, "mutation_detected": True,
        "exact_source_bytes_restored": True, "restored_sha256": hashlib.sha256(original).hexdigest()})
    observed = []
    for row in commands:
        result = child(row["role"], row["command"], env)
        observed.append(result)
        if result["actual_child_exit_code"] != 0:
            raise ValueError("CPU child failed; preserve evidence: " + row["role"])
    plans = []
    for row in observed[:4]:
        plan = json.loads(Path(row["log"]["path"]).read_text())
        if (plan["Torch_imported"] is not False or plan["SDK_started"] is not False
                or plan["actual_execution_admitted"] is not False):
            raise ValueError("Source preflight claimed actual execution or imported Torch/SDK")
        for stage in plan["stages"]:
            if Path(stage["output"]).exists() or Path(stage["SDK_receipt"]).exists():
                raise ValueError("CPU preparation created a future SDK output/receipt")
        plans.append({"case": plan["case"]["case_id"], "stage_count": len(plan["stages"]),
                      "initial_comparison_count": len(plan["initial_comparisons"]), "new_outputs_created": False})
    owner = json.loads((bridge.S3 / "terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if (len(guards) != 23 or any(record(row["path"]) != row for row in guards)
            or any(record(row["path"]) != row for row in sources) or record(bridge.MATRIX) != matrix
            or bridge.raw.source_bindings() != bindings["raw"] or bridge.style.source_bindings() != bindings["style"]):
        raise ValueError("Active S3 sources, bridge, matrix or frozen readers changed")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_execution_bridge_CPU_execution.v1",
        "registration": record(HERE / "CPU_registration.json"), "children": observed,
        "mutation": record(HERE / "mutation_review.json"), "plans": plans, "sources_and_matrix_unchanged": True,
        "S3_guard_count": len(guards), "S3_guarded_sources_unchanged": True,
        "actual_native_SDK_exit_or_readiness_tested": False, "tool_owner_driver_integration_still_required": True,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False})
    print(json.dumps({"status": "native_execution_bridge_CPU_prepared", "actual_children_exit0": len(observed),
        "receipt": record(HERE / "CPU_preparation_execution.json"), "SDK_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
