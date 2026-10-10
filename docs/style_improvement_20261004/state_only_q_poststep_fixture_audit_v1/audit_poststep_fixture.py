"""Future post-step metadata admission; reuse the sealed original raw math.

Execution requires actual readonly terminal, preflight and fixture records.
Importing this entry neither loads a model nor reads a capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "docs/style_improvement_20261003/state_only_matched_q_physical_preparation_v1"
SOURCE_PLAN = ROOT / "docs/style_improvement_20261003/state_only_q_poststep_evaluation_source_plan_v1/commands_and_source_plan.json"
PLAN_SHA = "6a09e3d030c47984e173e68221e08854826274a743786a4557512611c727df5b"
OLD_SHA = "37af492e3bf1cd32eb1e1f5088b4f207834d3a901ce9901c6acda6c4db9acb03"
OLD_RECIPE_SHA = "22b9eac0cea44d88cd6a58b3409259c587435a579b063177964c00f73f114d1d"
HELPER = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1/physical_evidence.py"
HELPER_SHA = "30202b34a13b5756ffa5c9b41110103a913c3883ed1d88d8389546322a8fc05a"
TERMINAL_READER = ROOT / "docs/style_improvement_20261003/state_only_q_poststep_training_audit_preparation_v1/audit_poststep_training_finalization_v3.py"
TERMINAL_READER_SHA = "55cc08a194d0b43f68e24e056e34170402ee9e879b9745601a1ec97dc56a5699"
TRAINING_PLAN_SHA = "acd1027740bf618400a7b4b3b9009ef37f55aafc2c7ebabad3ef2f7f6dc55a90"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def sealed(path, sha):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and not path.stat().st_mode & 0o222, "Actual readonly regular absolute input required")
    row = record(path)
    require(row["sha256"] == sha, "Actual input SHA differs")
    return json.loads(path.read_text()), row


def derive_reader():
    """Change exactly seven schema occurrences and one copied-file count."""
    sealed(SOURCE_PLAN, PLAN_SHA)
    for path, sha in [(OLD / "audit_fixture.py", OLD_SHA), (HELPER, HELPER_SHA)]:
        require(record(path)["sha256"] == sha and not path.stat().st_mode & 0o222,
                "Original reader/helper bytes changed")
    original = (OLD / "audit_fixture.py").read_text()
    before, after = "state_only_matched_q.", "state_only_q.poststep_"
    require(original.count(before) == 7 and after not in original, "Original seven schemas differ")
    source = original.replace(before, after)
    old_count, new_count = 'document["copied_files_hashed"] == 936', 'document["copied_files_hashed"] == 937'
    require(source.count(old_count) == 1 and new_count not in source, "Original copied-file guard differs")
    source = source.replace(old_count, new_count)
    require(source.replace(new_count, old_count).replace(after, before) == original,
            "Inverse entire source bytes differ")
    return source


def prepare(binding_path, binding_sha):
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
                    for name in sys.modules), "Fresh CPU reader without model/SDK imports required")
    plan, _ = sealed(SOURCE_PLAN, PLAN_SHA)
    original_recipe, _ = sealed(OLD / "commands.json", OLD_RECIPE_SHA)
    binding, binding_row = sealed(binding_path, binding_sha)
    require(Path(binding_path).name == "commands.json"
            and binding["schema"] == "state_only_q.poststep_fixture_runtime_binding.v1"
            and binding["status"] == "actual_terminal_and_preflight_registered",
            "Pending templates and historical bindings cannot admit future captures")
    require(binding["evaluation_source_plan"] == record(SOURCE_PLAN)
            and binding["command"] == plan["fixture_GPU_argv"]
            and binding["SDK_free_preflight_command"] == plan["SDK_free_fixture_preflight_argv"]
            and binding["GPU_environment"] == plan["fixture_GPU_env"]
            and binding["requested_profile"] == plan["selected_profile"]
            and binding["fixture_raw_math"] == plan["original_raw_math"]
            and binding["production_source"] == original_recipe["production_source"]
            and binding["planner"] == original_recipe["planner"]
            and binding["terminal_facts"] == original_recipe["terminal_facts"],
            "Fixed original physics/model/counter recipe differs")
    require(binding["checkpoint"]["path"] == plan["future_checkpoint"]["path"]
            and binding["terminal_admission"]["path"] == plan["future_terminal_audit"]["path"]
            and binding["training_source_manifest"]["sha256"] == plan["future_terminal_audit"]["source_manifest_sha256"]
            and binding["output"] == plan["fixture_GPU_argv"][plan["fixture_GPU_argv"].index("--output") + 1],
            "Actual post-step model/source/output paths differ")
    for key in ("actual_preflight", "actual_preflight_receipt", "terminal_admission"):
        document, observed = sealed(binding[key]["path"], binding[key]["sha256"])
        require(observed == binding[key], "Actual metadata record differs: " + key)
    terminal, _ = sealed(binding["terminal_admission"]["path"], binding["terminal_admission"]["sha256"])
    require(record(TERMINAL_READER)["sha256"] == TERMINAL_READER_SHA
            and terminal["actual_execution_adapter"] == record(TERMINAL_READER)
            and terminal["original_full_payload_numeric_checks_unchanged"] is True
            and terminal["check_count"] > 0, "Actual complete finalization-v3 reader required")
    launch_binding_row = terminal["root_owner_source_binding"]
    launch_binding, observed = sealed(launch_binding_row["path"], launch_binding_row["sha256"])
    require(observed == launch_binding_row
            and launch_binding["schema"] == "state_only_q.poststep_reader_root_owner_binding.v1"
            and launch_binding["source_manifest"] == binding["training_source_manifest"],
            "Actual terminal root-owner source binding differs")
    launch_row = launch_binding["actual_launch_records"]["fullfresh2000"]
    launch, observed = sealed(launch_row["path"], launch_row["sha256"])
    root_plan_row = launch_binding["root_plan"]
    root_plan, observed_plan = sealed(root_plan_row["path"], root_plan_row["sha256"])
    seal_row = binding["terminal_root_seal"]
    seal, observed_seal = sealed(seal_row["path"], seal_row["sha256"])
    require(observed == launch_row and observed_plan == root_plan_row and observed_seal == seal_row
            and root_plan_row["sha256"] == TRAINING_PLAN_SHA
            and launch_binding["actual_training_env"] == {**root_plan["env"], "CUDA_VISIBLE_DEVICES": "0"}
            and launch["schema"] == "state_only_q.poststep_actual_training_root_owner.v1"
            and launch["status"] == "closed" and type(launch["actual_exit_code"]) is int
            and launch["actual_exit_code"] == 0 and launch["child_exit_observed"] is True
            and launch["command"] == root_plan["commands"]["fullfresh2000"]
            and launch["env"] == launch_binding["actual_training_env"]
            and type(launch["actual_tool_session_id"]) is int and launch["actual_tool_session_id"] > 0
            and seal["actual_root_GPU_session_id"] == launch["actual_tool_session_id"]
            and seal["actual_root_GPU_launch_chunk"] == launch["launch_chunk"]
            and seal["actual_root_GPU_completion_chunk"] == launch["completion_chunk"],
            "Actual fresh2000 closure and terminal root seal differ")
    namespace = {"__file__": __file__, "__name__": "_poststep_original_fixture_metadata"}
    exec(compile(derive_reader(), __file__, "exec"), namespace)
    namespace.update(HERE=Path(binding_path).parent, PLAN_SHA256=PLAN_SHA,
                     PREFLIGHT=Path(binding["actual_preflight"]["path"]),
                     PREFLIGHT_RECEIPT=Path(binding["actual_preflight_receipt"]["path"]))
    original_source, original_capture = namespace["source_binding"], namespace["admit_capture"]

    def source_binding(registered):
        result = original_source(registered)
        receipt = json.loads(Path(registered["actual_preflight_receipt"]["path"]).read_text())
        require(type(receipt["actual_child_return_code"]) is int and receipt["actual_child_return_code"] == 0
                and receipt["env"] == original_recipe["CPU_environment"]
                and isinstance(receipt["actual_root_tool_chunk"], str) and receipt["actual_root_tool_chunk"],
                "Actual closed CPU preflight receipt required")
        result.update(evaluation_source_plan=record(SOURCE_PLAN), actual_runtime_binding=binding_row,
                      original_metadata_reader=record(OLD / "audit_fixture.py"),
                      actual_terminal_reader=record(TERMINAL_READER))
        return result

    def admit_capture(registered, owner_path, owner_sha):
        result = original_capture(registered, owner_path, owner_sha)
        owner, _ = sealed(owner_path, owner_sha)
        require(owner["env"] == registered["GPU_environment"]
                and owner["MMD_launch_filter"] is False and owner["default_promoted"] is False
                and owner["sealed34_outputs_read"] is False, "Actual fixed GPU environment and full-data scope required")
        return result

    namespace.update(source_binding=source_binding, admit_capture=admit_capture)
    return namespace, binding, binding_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for actual raw reading")
    require(not args.output.exists(), "Preserve previous reports and failures")
    try:
        namespace, binding, _ = prepare(args.binding, args.binding_sha256)
        require(str(args.output.absolute()) == binding["future_raw_report"], "Registered fresh report path required")
        report = namespace["audit"](binding, args.owner, args.owner_sha256)
        report.update(execution_reader=record(__file__), original_raw_math_unchanged=True)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"schema": "state_only_q.poststep_actual_fixture_raw_audit.v1",
                       "status": "failed", "failure": repr(error),
                       "actual_capture_independently_audited": False}, stream, indent=2)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "physical_zero_gate_passed": report["physical_zero_gate_passed"]}))


if __name__ == "__main__":
    main()
