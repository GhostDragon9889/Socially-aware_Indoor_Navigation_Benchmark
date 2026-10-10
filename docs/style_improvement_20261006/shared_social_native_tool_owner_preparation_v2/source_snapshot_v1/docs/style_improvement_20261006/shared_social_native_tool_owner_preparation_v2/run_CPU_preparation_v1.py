"""Execute cold native owner registrations and CPU contracts, retaining failures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_shared_social_native_owner as owner  # noqa: E402


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child(role, command, env):
    log = HERE / (role + ".log")
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    observed = {"role": role, "command": command, "actual_child_exit_code": result.returncode,
        "actual_child_exit_observed": True, "log": owner.bridge.record(log), "SDK_started": False}
    write_new(role + "_execution.json", observed)
    return observed


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    names = ["scripts/prepare_shared_social_native_owner.py", "scripts/shared_social_native_tool_owner.js",
        "tests/test_native_tool_owner.py", "tests/native_tool_owner_CPU_contracts.js"]
    sources = [owner.bridge.record(ROOT / name) for name in names] + [owner.bridge.record(__file__)]
    cases = ("Office_N03_seed19", "Warehouse_N15_seed42", "Full_Warehouse_N03_seed7", "Hospital_N11_seed19")
    cli = [sys.executable, str(ROOT / names[0])]
    commands = []
    for case in cases:
        registration = str(HERE / (case + "_registration.json"))
        commands.extend((
            {"role": case + "_source", "command": [*cli, "--mode", "source-preflight", "--case-id", case,
                "--registration", registration]},
            {"role": case + "_envelope", "command": [*cli, "--mode", "owner-envelope", "--registration", registration]}))
    commands.extend((
        {"role": "tests_final", "command": [sys.executable, "-m", "pytest", "-q",
            "tests/test_native_tool_owner.py", "tests/test_native_execution_bridge.py"]},
        {"role": "ruff_final", "command": ["/home/lfm/anaconda3/bin/ruff", "check", names[0], names[2],
            str(Path(__file__).relative_to(ROOT))]}))
    write_new("CPU_registration.json", {"schema": "shared_social.native_tools_owner_CPU_registration.v1",
        "sources": sources, "commands": commands, "bridge_sources": owner.bridge.prepare_case(cases[0]),
        "GPU_or_SDK_started": False, "actual_native_execution_readiness_expected": False})
    observed = []
    for row in commands:
        result = child(row["role"], row["command"], env)
        observed.append(result)
        if result["actual_child_exit_code"] != 0:
            raise ValueError("CPU child failed; preserve evidence: " + row["role"])
    gate = child("actual_S3_gate_expected_refusal", [*cli, "--mode", "execution-preflight", "--registration",
        str(HERE / (cases[0] + "_registration.json"))], env)
    if gate["actual_child_exit_code"] == 0 or "external_owner_completion.json" not in Path(gate["log"]["path"]).read_text():
        raise ValueError("Expected current missing actual S3 completion to refuse execution")
    for case in cases:
        envelope = json.loads((HERE / (case + "_envelope.log")).read_text())
        plan, _ = owner.admitted_plan(Path(envelope["registration"]["path"]))
        if envelope["plan"] != owner.owner_view(plan):
            raise ValueError("Runtime owner view differs from full readonly registration")
        if len(plan["actions"]) != 21 or any(Path(row["output"]).exists() for row in plan["case_plan"]["stages"]):
            raise ValueError("Serial plan differs or CPU preparation created SDK outputs")
        if Path(plan["observation_directory"]).exists():
            raise ValueError("Rejected current gate created native observation directory")
    guards = json.loads((owner.bridge.S3 / "terminal_owner_state.json").read_text())["bindings"]["guarded_sources"]
    if (len(guards) != 23 or any(owner.bridge.record(row["path"]) != row for row in guards)
            or any(owner.bridge.record(row["path"]) != row for row in sources)):
        raise ValueError("S3 guards or reviewed owner sources changed")
    write_new("CPU_preparation_execution.json", {"schema": "shared_social.native_tools_owner_CPU_execution.v1",
        "registration": owner.bridge.record(HERE / "CPU_registration.json"), "children": observed,
        "expected_current_S3_gate_refusal": gate, "S3_guarded_sources": guards,
        "source_interfaces_and_case_order_prepared": True, "actual_S3_readiness_admitted": False,
        "actual_native_SDK_or_quality_measured": False, "paper_effect_achieved": False})
    print(json.dumps({"status": "native_tools_owner_CPU_prepared", "actual_children_exit0": len(observed),
        "expected_current_gate_exit": gate["actual_child_exit_code"],
        "execution": owner.bridge.record(HERE / "CPU_preparation_execution.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
