#!/usr/bin/env python3
"""Register and cold admit one serial native tools owner; never launch SDK here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import native_execution_bridge as bridge  # noqa: E402
from social_navigation.external_runtime import selected_python  # noqa: E402

DRIVER = ROOT / "scripts/shared_social_native_tool_owner.js"
COMPLETION = bridge.S3 / "external_owner_completion.json"
BRIDGE_SHA = "249e66e60d5378a4c52e00c35803501059ceb97c22bb03090e8add995aa0a7e6"
PLAN_SCHEMA = "shared_social.registered_native_serial_tool_plan.v1"


def action(role, argv, directory, *, cuda="", sdk=False, failure_publication=None):
    return {"role": role, "argv": argv, "cuda": cuda, "cmd": bridge.tool_command(argv, cuda=cuda),
        "workdir": str(ROOT), "requires_host_GPU": sdk,
        "observation": str(directory / (role + "_actual_tool.json")), "failure_publication": failure_publication}


def build_plan(case_id, registration_path):
    bridge.require(bridge.record(bridge.__file__)["sha256"] == BRIDGE_SHA,
                   "Require the CPU-reviewed native exit bridge")
    case = bridge.prepare_case(case_id)
    python = str(selected_python(ROOT))
    # Preserve the frozen command derivation; adapt only executable selection.
    commands = [case["style_command"], *[row["command"] for row in case["initial_comparisons"]]]
    for row in case["stages"]:
        commands.extend((row["SDK_command"], row["raw_command"]))
    for command in commands:
        bridge.require(command[0] == str(bridge.PYTHON), "Frozen native command executable changed")
        command[0] = python
    parent = Path(case["stages"][0]["SDK_receipt"]).parent
    cli = [python, str(Path(__file__).resolve()), "--registration", str(registration_path)]
    actions = []
    for controls in (30, 6000):
        for stage in case["stages"]:
            if stage["controls"] != controls:
                continue
            prefix = stage["planner"] + "_C" + str(controls)
            sdk = action(prefix + "_SDK", stage["SDK_command"], parent, cuda="0", sdk=True)
            raw = action(prefix + "_raw_CPU", stage["raw_command"], parent)
            publisher = [python, str(ROOT / "scripts/prepare_shared_social_native_execution.py"),
                "--case-id", case_id, "--planner", stage["planner"], "--control-steps", str(controls),
                "--owner-completion", str(COMPLETION)]
            sdk_publish = action(prefix + "_SDK_publication", [*publisher, "--mode", "SDK-exit",
                "--tool-observation", sdk["observation"]], parent)
            raw_publish = action(prefix + "_raw_CPU_publication", [*publisher, "--mode", "raw-CPU-exit",
                "--tool-observation", raw["observation"]], parent)
            sdk["failure_publication"], raw["failure_publication"] = sdk_publish, raw_publish
            actions.extend((sdk, sdk_publish, raw, raw_publish))
        comparison = next(row for row in case["initial_comparisons"] if row["controls"] == controls)
        actions.append(action("initial_pair_C" + str(controls), comparison["command"], parent))
    actions.extend((action("seal_captures", [*cli, "--mode", "seal-captures"], parent),
        action("style_200s", case["style_command"], parent),
        action("collect_measurements", [*cli, "--mode", "collect"], parent)))
    return {"schema": PLAN_SCHEMA, "case_id": case_id, "actors": case["case"]["actors"], "seed": case["case"]["seed"],
        "checkout_root": str(ROOT), "sdk_python": python,
        "registration_path": str(registration_path), "owner_completion": str(COMPLETION),
        "matrix": case["matrix"], "scene_configuration": case["request"]["configuration_file"],
        "case_plan": case, "sources": {"bridge": bridge.record(bridge.__file__),
            "planner": bridge.record(__file__), "tools_owner": bridge.record(DRIVER)},
        "actions": actions, "observation_directory": str(parent), "state_path": str(parent / "serial_tool_owner_state.json"),
        "completed_report": str(parent / "completed_case_measurements.json"),
        "execution_preflight_command": bridge.tool_command([*cli, "--mode", "execution-preflight"], cuda=""),
        "guard_command": bridge.tool_command([*cli, "--mode", "guard"], cuda=""),
        "SDK_started": False, "paper_effect_achieved": False}


def admitted_plan(path):
    plan, identity = bridge.readonly_json(path)
    bridge.require(plan.get("schema") == PLAN_SCHEMA and Path(plan["registration_path"]) == path.resolve(),
                   "Require this readonly native owner registration")
    bridge.require(build_plan(plan["case_id"], path.resolve()) == plan, "Registered commands, resources or owner sources changed")
    return plan, identity


def owner_view(plan):
    """Compact tools-facing commands; the full registration retains source graphs."""
    def visible(row):
        value = {key: row[key] for key in ("role", "cmd", "workdir", "requires_host_GPU", "observation")}
        value["failure_publication"] = visible(row["failure_publication"]) if row["failure_publication"] else None
        return value
    value = {key: plan[key] for key in ("schema", "case_id", "registration_path", "SDK_started",
        "checkout_root", "sdk_python",
        "observation_directory", "state_path", "execution_preflight_command", "guard_command")}
    value["actions"] = [visible(row) for row in plan["actions"]]
    return value


def ordinary_owned(path, parent):
    bridge.require(path.is_relative_to(parent) and not any(item.is_symlink() for item in (path, *path.parents))
        and path.is_file() and path.stat().st_nlink == 1, "Require ordinary single-link owned captures")


def seal_captures(plan):
    identities = []
    for stage in plan["case_plan"]["stages"]:
        selected = bridge.raw.admit_execution(stage["SDK_receipt"], actors=plan["actors"], controls=stage["controls"],
            planner=stage["planner"], seed=plan["seed"], scene_config=plan["scene_configuration"]["path"],
            scene_sha256=plan["scene_configuration"]["sha256"])
        rows = [selected.receipt["result"], *selected.arrays,
            selected.result["paired_initial_condition"]["document"], selected.initial["arrays"]]
        for row in rows:
            path = Path(row["path"])
            ordinary_owned(path, Path(stage["output"]))
            bridge.require(bridge.record(path) == row, "Actual capture changed before sealing")
            identities.append(row)
    # Validate all files first; retain any partial chmod on an unexpected OS failure.
    for row in identities:
        path = Path(row["path"])
        path.chmod(0o444)
        bridge.require(bridge.record(path) == row and not path.stat().st_mode & 0o222, "Capture changed during sealing")
    return {"status": "actual_native_captures_readonly", "files": identities,
        "chmod_changes_capture_bytes": False, "paper_effect_achieved": False}


def collect(plan):
    observations = []
    for stage in plan["actions"][:-1]:
        packet, identity = bridge.readonly_json(stage["observation"])
        terminal = bridge.validate_tool_observation(packet, command=stage["argv"], cuda=stage["cuda"])
        bridge.require(terminal["actual_exit_code"] == 0, "Every preceding native tool stage must actually exit0")
        observations.append({"role": stage["role"], "observation": identity, **terminal})
    for row in plan["case_plan"]["initial_comparisons"]:
        comparison = json.loads(Path(row["command"][-1]).read_text())
        bridge.require(comparison["status"] == "native_initial_conditions_matched"
            and comparison["actors"] == plan["actors"] and comparison["same_initial_body_map_and_RNG_bytes"] is True,
            "Actual smoke and full initial pairings required")
    report_path = Path(plan["case_plan"]["style_command"][-1])
    style = json.loads(report_path.read_text())
    bridge.require(style["schema"] == bridge.style.STYLE_SCHEMA and style["status"] == "passed_CPU_measurement"
        and type(style["seed"]) is int and style["seed"] == plan["seed"] and style["actors"] == plan["actors"]
        and style["control_steps"] == 6000 and style["duration_s"] == 200
        and style["sources"] == {"entry": bridge.record(plan["case_plan"]["style_command"][1]),
            "reader_bindings": bridge.style.source_bindings()}
        and style["basis_refit"] is False and style["metric_math_changed"] is False
        and style["failed_episodes_filtered"] is False and style["paper_effect_achieved"] is False
        and style["numerical_comparison_to_paper_valid"] is False, "Require the full frozen inclusive style report")
    ordinary_owned(report_path, Path(plan["observation_directory"]))
    report_path.chmod(0o444)
    value = {"schema": "shared_social.completed_registered_native_case_measurements.v1", "case_id": plan["case_id"],
        "seed": plan["seed"], "actors": plan["actors"], "observed_tool_stages": observations,
        "status": "completed_measurements_quality_unclassified", "style_report": bridge.record(report_path),
        "paper_effect_achieved": False, "default_model_promoted": False}
    return bridge._write_new(plan["completed_report"], value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("source-preflight", "owner-envelope", "execution-preflight",
        "guard", "seal-captures", "collect"))
    parser.add_argument("--case-id")
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args(argv)
    bridge.require(not any(name.split(".")[0] in bridge.raw.original.FORBIDDEN for name in sys.modules),
                   "Native owner preparation requires a fresh CPU process")
    if args.mode == "source-preflight":
        if not args.case_id:
            parser.error("Source preparation requires an explicit registered case")
        plan = build_plan(args.case_id, args.registration.resolve())
        # Publication is explicit; no simulation or future native output directories.
        bridge._write_new(args.registration, plan)
        print(json.dumps({"registration": bridge.record(args.registration), "SDK_started": False, "actions": len(plan["actions"])}))
        return 0
    plan, identity = admitted_plan(args.registration)
    if args.mode == "owner-envelope":
        print(json.dumps({"registration": identity, "plan": owner_view(plan)}))
        return 0
    completion = bridge.admit_S3_owner_completion(plan["owner_completion"])
    if args.mode == "execution-preflight":
        paths = [Path(row["output"]) for row in plan["case_plan"]["stages"]]
        paths.append(Path(plan["observation_directory"]))
        bridge.require(all(not path.exists() and not path.is_symlink() for path in paths),
                       "Require unused native outputs and observations; preserve failures, no automatic restart")
        value = {"status": "native_serial_S3_execution_ready", "registration": identity,
            "owner_plan": owner_view(plan), "external_owner_completion": completion["external_owner_completion"],
            "SDK_started": False}
    elif args.mode == "guard":
        value = {"status": "native_serial_sources_and_S3_guarded", "registration": identity, "SDK_started": False}
    elif args.mode == "seal-captures":
        value = seal_captures(plan)
    else:
        value = collect(plan)
    print(json.dumps(value, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
