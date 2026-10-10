"""Prepare fixed 22-motion/200s evaluation from frozen new935 parsers on CPU.

The initial draft explicitly lacks a reader freeze. A second invocation with
the actual independent reader freeze seals registration and commands; neither
invocation reads a candidate checkpoint or admits future model quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "docs/style_improvement_20261003/state_prior_evaluation_registration_v1/registration.json"
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"
CODE = BUNDLE / "code"
PROFILE = CODE / "configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_state_prior_stratified_isolated_snapshots_v1/step_2000.pt"
AUDIT_DIR = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_audit_v1"
AUDITOR = AUDIT_DIR / "audit_stratified_training.py"
ADMISSION = AUDIT_DIR / "actual_terminal2000_v1/audit.json"
OUTPUTS = {key: ROOT / "output/diffusion_amp" / name for key, name in
           (("tracking", "stratified_prior_tracking2000_v1"), ("navigation", "stratified_prior_nav2000_v1"))}
SDKPY = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
MANIFEST_SHA = "2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359"
PROFILE_SHA = "bcbc8942c02560c1790afc1b0b7e310463e271035d513607ba3d223fc8fb78ed"
REWARD = {"amp_reward_scale": 2.0, "amp_reward_weight": 2.0,
          "task_reward_weight": 0.5, "direction_mode": "backwards_penalty",
          "amp_probability_epsilon": 0.0001}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parsed_values(arguments):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}


def prepare_draft():
    for path in [HERE / "registration_draft.json", HERE / "registration.json", HERE / "commands.json", *OUTPUTS.values()]:
        require(not path.exists() and not path.is_symlink(), "Exclusive new preparation/evaluation paths required: " + str(path))
    old = json.loads(OLD.read_text())
    source_record = record(BUNDLE / "source_manifest.json")
    source = json.loads(Path(source_record["path"]).read_text())
    require(source_record["sha256"] == MANIFEST_SHA and len(source["files"]) == 935, "Final new935v2 source required")
    names = {str(path.relative_to(BUNDLE)) for path in CODE.rglob("*") if path.is_file() and not path.is_symlink()}
    require(names == {row["path"] for row in source["files"]}, "Exactly declared owned sources required")
    for row in source["files"]:
        path = BUNDLE / row["path"]
        actual = record(path)
        require(path.resolve().is_relative_to(CODE) and actual["sha256"] == row["sha256"]
                and actual["bytes"] == row["bytes"] and not path.stat().st_mode & 0o222,
                "Frozen owned source changed: " + row["path"])
    require(record(PROFILE)["sha256"] == PROFILE_SHA, "Final absolute stratified expert sampling profile required")
    targets = {}
    for protocol in OUTPUTS:
        section = deepcopy(old[protocol])
        argv = section["argv_without_future_SHA"] if protocol == "tracking" else section["argv"]
        flag = "--checkpoint" if protocol == "tracking" else "--resume"
        argv[argv.index(flag) + 1] = str(CHECKPOINT)
        argv[argv.index("--output") + 1] = str(OUTPUTS[protocol])
        if protocol == "navigation":
            argv[argv.index("--config") + 1] = str(PROFILE)
            argv[argv.index("--scene-config") + 1] = str(CODE / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml")
        section["source_execution"] = "Exact new935v2 frozen full-network entry/guarded owned imports; independent stratified expert sampling terminal admission required"
        targets[protocol] = section
    launcher_path = BUNDLE / "launch_isolated.py"
    spec = importlib.util.spec_from_file_location("registered_new935_parser_isolation", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    ledger = launcher.ImportLedger(HERE / "configuration_only_imports.json")
    launcher.activate(ledger)
    status = "failed"
    try:
        from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
        from scripts.benchmark_paper_motion_tracking import parse_args as parse_tracking

        targets["tracking"]["parsed"] = parsed_values(parse_tracking(targets["tracking"]["argv_without_future_SHA"]))
        targets["navigation"]["parsed"] = parsed_values(parse_amp_args(targets["navigation"]["argv"]))
        profile = load_amp_profile(PROFILE)
        require(profile["training"]["warm_start_action_discriminator_from_state_prior"] is True
                and profile["training"]["discriminator_input"] == "state_action", "Coupled state prior required")
        require(all(type(profile["training"].get(key)) is type(value) and profile["training"][key] == value
                    for key, value in REWARD.items()) and profile["reward"]["amp_weight"] == 2.0,
                "Five-field stratified expert sampling contract required")
        require(not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab."))
                        for name in sys.modules), "SDK-free parsers required")
        for protocol in OUTPUTS:
            expected = old[protocol]["parsed"]
            allowed = {"checkpoint", "checkpoint_sha256", "output"} if protocol == "tracking" else {"config", "resume", "output", "scene_config"}
            require({key: value for key, value in targets[protocol]["parsed"].items() if key not in allowed} ==
                    {key: value for key, value in expected.items() if key not in allowed},
                    "Original matched protocol values changed: " + protocol)
        status = "passed"
    finally:
        ledger.finish(status)
    sources = deepcopy(old["sources"])
    sources.update({"isolated_source_manifest": source_record, "isolated_launcher": record(launcher_path),
                    "registered_profile": record(PROFILE), "scene": record(CODE / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"),
                    "state_prior_auditor": None, "state_prior_auditor_source_freeze": None,
                    "prior_registration": record(OLD),
                    "entrypoint_bytes": {name: record(CODE / name) for name in ("scripts/train_paper_amp.py", "scripts/benchmark_paper_motion_tracking.py")},
                    "style90_entry": record(CODE / "scripts/evaluate_pedestrian_style.py"),
                    "style90_math": record(CODE / "CrowdSim/humanoid/style_evaluation.py")})
    for key in ("style90_entry", "style90_math"):
        require(sources[key]["sha256"] == old["sources"][key]["sha256"], "Original primary metric bytes changed")
    for key, expected in (("frozen_style_reference", "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"),
                          ("pose45_basis", "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb")):
        require(record(sources[key]["path"]) == sources[key] and sources[key]["sha256"] == expected,
                "Fixed AMASS metric reference changed")
    registration = deepcopy(old)
    registration.update({"schema": "stratified_prior.future_evaluation_registration.v1",
                         "status": "prepared_protocol_and_isolated_parsers_passed_reader_freeze_pending",
                         "sources": sources, **targets})
    registration["future_checkpoint"].update({"path": str(CHECKPOINT), "terminal_admission_path": str(ADMISSION),
                                              "required_reward_contract": deepcopy(REWARD)})
    registration["matched_protocol_and_source"].pop("new934_source_required")
    registration["matched_protocol_and_source"]["new935_source_required"] = True
    sampling = json.loads((ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_audit_v1/actual_pilot4_v1/audit.json").read_text())["checkpoints"][0]["actual_expert_sampling_saved_state"]
    registration["future_checkpoint"]["required_expert_sampling_contract"] = {key: sampling[key] for key in ("saved_schema", "configuration", "registry_sha256", "probabilities_sha256", "manifest_sha256", "config_sha256")}
    registration["sources"]["state_prior_auditor_peer"] = record(ROOT / "docs/style_improvement_20261003/state_prior_stratified_reader_independent_peer_v1/peer_review.json")
    registration["pending"]["independent_reader_source_freeze"] = True
    write(HERE / "registration_draft.json", registration)
    print(json.dumps({"status": "CPU_prepared_reader_freeze_pending", "draft": record(HERE / "registration_draft.json"),
                      "owned_sources_fully_hashed": 935, "GPU_used": False, "future_effect": False}))


def finalize(freeze_path):
    for path in [HERE / "registration.json", HERE / "commands.json", *OUTPUTS.values()]:
        require(not path.exists() and not path.is_symlink(), "Exclusive final registration/evaluation paths required")
    registration = json.loads((HERE / "registration_draft.json").read_text())
    freeze_path = freeze_path.absolute()
    freeze_record = record(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    reader = record(AUDITOR)
    files = freeze.get("source_files", [])
    require(freeze.get("schema") == "stratified_prior.independent_auditor_source_freeze.v1"
            and freeze.get("status") == "prepared_only_frozen_pending_independent_peer" and freeze.get("source_manifest") == registration["sources"]["isolated_source_manifest"]
            and reader in files and len({row["path"] for row in files}) == len(files)
            and all(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222 for row in files),
            "Actual independent frozen reader/dependencies/source required")
    require(record(BUNDLE / "source_manifest.json") == registration["sources"]["isolated_source_manifest"],
            "Final source manifest changed since CPU draft")
    registration["sources"]["state_prior_auditor"] = reader
    registration["sources"]["state_prior_auditor_source_freeze"] = freeze_record
    registration["status"] = "prepared_protocol_and_isolated_parsers_passed_future_terminal_pending"
    registration["pending"]["independent_reader_source_freeze"] = False
    peer = json.loads(Path(registration["sources"]["state_prior_auditor_peer"]["path"]).read_text())
    require(peer["schema"] == "stratified_prior.independent_training_payload_reader_peer.v1"
            and peer["status"] == "passed_prepared_only" and peer["reader_freeze"] == freeze_record
            and peer["main"] == reader and peer["required_findings"] == [], "Exact independent q-reader source peer required")
    write(HERE / "registration.json", registration)
    executor = HERE / "run_registered_evaluation.py"
    raw = executor.read_text()
    require(raw.count("UNFROZEN_REGISTRATION_SHA256") == 1, "Fresh unbound executor required")
    executor.write_text(raw.replace("UNFROZEN_REGISTRATION_SHA256", record(HERE / "registration.json")["sha256"]))
    sources = registration["sources"]
    commands = {"schema": "stratified_prior.future_registered_evaluation_commands.v1", "status": "prepared_not_executed",
                "future_effect": False, "actual_SHA_and_terminal_admission_required": True,
                "CPU_environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                    "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                "GPU_environment_root_single_App_only": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4",
                                                         "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                "registered_protocol": record(HERE / "registration.json"), "executor": record(executor),
                "CPU_metric_sources_frozen": {key: sources[key] for key in
                                              ("style90_entry", "style90_math", "pose45_basis", "pose45_original_evaluator")}}
    for protocol in OUTPUTS:
        commands[protocol + "_GPU_root_only"] = [SDKPY, str(executor), "--protocol", protocol, "--checkpoint", str(CHECKPOINT),
            "--checkpoint-sha256", "ACTUAL_SEALED_STEP2000_SHA256", "--checkpoint-admission", str(ADMISSION),
            "--output", str(OUTPUTS[protocol]), "--import-report", str(BUNDLE / ("stratified_prior_" + protocol + "2000_v1_GPU_imports.json"))]
    write(HERE / "commands.json", commands)
    print(json.dumps({"status": "registered_future_terminal_pending", "registration": record(HERE / "registration.json"),
                      "GPU_used": False, "future_effect": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor-source-freeze", type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    if args.auditor_source_freeze is None:
        prepare_draft()
    else:
        finalize(args.auditor_source_freeze)


if __name__ == "__main__":
    main()
