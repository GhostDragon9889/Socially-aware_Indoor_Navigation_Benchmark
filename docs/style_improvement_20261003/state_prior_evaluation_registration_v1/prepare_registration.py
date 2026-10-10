"""Prepare fixed terminal evaluation from new934 SDK-free parsers on CPU."""
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
OLD = ROOT / "docs/style_improvement_20261002/terrain_protected_evaluation_registration_v1/registration.json"
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
CODE = BUNDLE / "code"
PROFILE = CODE / "configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml"
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt"
ADMISSION = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1/terminal2000/audit.json"
OUTPUTS = {key: ROOT / "output/diffusion_amp" / name for key, name in
           (("tracking", "state_prior_tracking2000_v1"), ("navigation", "state_prior_nav2000_v1"))}
SDKPY = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
MANIFEST_SHA = "3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular artifact required: " + str(path))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor-source-freeze", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    for path in [HERE / "registration.json", HERE / "commands.json", *OUTPUTS.values()]:
        if path.exists() or path.is_symlink():
            raise ValueError("Exclusive new preparation/evaluation paths required: " + str(path))
    old = json.loads(OLD.read_text())
    source = json.loads((BUNDLE / "source_manifest.json").read_text())
    source_record = record(BUNDLE / "source_manifest.json")
    assert source_record["sha256"] == MANIFEST_SHA and len(source["files"]) == 934
    for row in source["files"]:
        actual = record(BUNDLE / row["path"])
        assert actual["sha256"] == row["sha256"] and actual["bytes"] == row["bytes"]
        assert not (BUNDLE / row["path"]).stat().st_mode & 0o222
    freeze_path = args.auditor_source_freeze.absolute()
    freeze = json.loads(freeze_path.read_text())
    auditor = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1/audit_state_prior_training.py"
    reader = record(auditor)
    assert freeze["schema"] == "public_state_prior.independent_auditor_source_freeze.v1" and freeze["status"] == "frozen"
    assert any(row == reader for row in freeze["source_files"]), "Frozen independent reader identity required"
    assert all(record(row["path"]) == row for row in freeze["source_files"])
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
        section["source_execution"] = "Exact new934 copied old full-network adapter/entry/guarded owned imports; coupled public state prior terminal admission required"
        targets[protocol] = section
    launcher_path = BUNDLE / "launch_isolated.py"
    spec = importlib.util.spec_from_file_location("registered_new934_parser_isolation", launcher_path)
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
        assert profile["training"]["warm_start_action_discriminator_from_state_prior"] is True
        assert profile["training"]["discriminator_input"] == "state_action"
        assert not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)
        for protocol in OUTPUTS:
            expected = old[protocol]["parsed"]
            allowed = {"checkpoint", "checkpoint_sha256", "output"} if protocol == "tracking" else {"config", "resume", "output", "scene_config"}
            assert {key: value for key, value in targets[protocol]["parsed"].items() if key not in allowed} == {
                key: value for key, value in expected.items() if key not in allowed}
        status = "passed"
    finally:
        ledger.finish(status)
    sources = {"fixed_manifest": deepcopy(old["sources"]["fixed_manifest"]),
        "frozen_style_reference": deepcopy(old["sources"]["frozen_style_reference"]),
        "isolated_source_manifest": source_record, "isolated_launcher": record(launcher_path),
        "registered_profile": record(PROFILE), "scene": record(CODE / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"),
        "state_prior_auditor": reader, "state_prior_auditor_source_freeze": record(freeze_path),
        "prior_registration": record(OLD),
        "entrypoint_bytes": {name: record(CODE / name) for name in ("scripts/train_paper_amp.py", "scripts/benchmark_paper_motion_tracking.py")},
        "style90_entry": record(CODE / "scripts/evaluate_pedestrian_style.py"),
        "style90_math": record(CODE / "CrowdSim/humanoid/style_evaluation.py"),
        "pose45_basis": record(ROOT / "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz"),
        "pose45_original_evaluator": record(ROOT / "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py"),
        "selected_baseline_style_comparison": record(ROOT / "docs/style_improvement_20261002/backward_penalty_navigation_style_v1/comparison.json")}
    assert sources["pose45_basis"]["sha256"] == "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb"
    assert sources["frozen_style_reference"]["sha256"] == "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
    registration = {"schema": "public_state_prior.future_evaluation_registration.v1",
        "status": "prepared_protocol_and_isolated_parsers_passed_future_terminal_pending", "CPU_only": True,
        "GPU_used": False, "Kit_started": False, "future_effect": False, "sealed34_outputs_read": False,
        "future_checkpoint": {"path": str(CHECKPOINT), "sha256": None, "bytes": None, "terminal_admission_path": str(ADMISSION),
            "actual_payload_preflight_performed": False, "actual_model_effect_evaluated": False,
            "required_updates": 2000, "required_transitions": 16384000, "required_D_input": 2019,
            "public_D_and_AMP_RMS_transfer_required": True, "required_initialization_strategy": "public_pacer_state_prior_zero_action_extension",
            "required_AMP_count": 24776000.0, "required_full_payload_resume_and_actual_imports": True},
        "sources": sources, **targets,
        "metrics": deepcopy(old["metrics"]),
        "pose45": {"reference": "saved AMASS-only centers_50/centers_100, fixed b25dffa basis", "refit_allowed": False,
            "generated_poses_used_for_fit": False, "replaces_primary_90D": False, "author_official_centers": False},
        "acceptance": {"execution_pass_is_quality_pass": False, "all_failed_reset_idle_frames_retained": True,
            "navigation_minimum_gate": {"complete_controls": 6000, "falls": 0, "minimum_route_vertices": 21,
                "inclusive_pose_velocity_mmd2_max": 0.07238060315284697},
            "complete22_tracking_and_fixed45D_reports_required": True, "unsupported_bins_are_unknown_not_zero": True,
            "actual_metrics_and_source_admission_required": True, "native_follow_on_or_default_promotion_automatic": False},
        "matched_protocol_and_source": {"original_all_CLI_values_retained_except_registered_aliases": True,
            "full_model_adapter_both_variants": True, "new934_source_required": True, "old931_or_D_false_reader_allowed": False},
        "pending": {"future_full_payload_admitted": False, "actual_tracking_evaluated": False, "actual_navigation_evaluated": False,
            "primary90D_or_pose45_actual_effect_evaluated": False}}
    write(HERE / "registration.json", registration)
    executor = HERE / "run_registered_evaluation.py"
    raw = executor.read_text()
    assert raw.count("UNFROZEN_REGISTRATION_SHA256") == 1
    executor.write_text(raw.replace("UNFROZEN_REGISTRATION_SHA256", record(HERE / "registration.json")["sha256"]))
    commands = {"schema": "public_state_prior.future_registered_evaluation_commands.v1", "status": "prepared_not_executed",
        "future_effect": False, "actual_SHA_and_terminal_admission_required": True,
        "CPU_environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        "GPU_environment_root_single_App_only": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        "registered_protocol": record(HERE / "registration.json"), "executor": record(executor),
        "CPU_metric_sources_frozen": {key: sources[key] for key in ("style90_entry", "style90_math", "pose45_basis", "pose45_original_evaluator")}}
    for protocol in OUTPUTS:
        commands[protocol + "_GPU_root_only"] = [SDKPY, str(executor), "--protocol", protocol, "--checkpoint", str(CHECKPOINT),
            "--checkpoint-sha256", "ACTUAL_SEALED_STEP2000_SHA256", "--checkpoint-admission", str(ADMISSION),
            "--output", str(OUTPUTS[protocol]), "--import-report", str(BUNDLE / ("state_prior_" + protocol + "2000_v1_GPU_imports.json"))]
    write(HERE / "commands.json", commands)
    print(json.dumps({"status": "prepared", "registration": record(HERE / "registration.json"), "GPU_used": False, "future_effect": False}))


if __name__ == "__main__":
    main()
