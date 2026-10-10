"""Register future state-action full22/200s protocols without running them.

Parses actual isolated old146 entrypoints. Future checkpoint existence/full
payload admission are deliberately pending, not a fake resource preflight.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_v1"
CODE = BUNDLE / "code"
OUT = DOCS / "state_action_evaluation_registration_v1"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def json_read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or OUT.exists():
        raise ValueError("CPU-only fresh registration required")
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args, validate_training_source

    original = json_read(ROOT / "output/diffusion_amp/shared_mixed_curriculum_step2000_ground_support_200s_v1/result.json")
    old_tracking = json_read(ROOT / "output/diffusion_amp/motion_tracking_mixed_curriculum_step2000_full_validation_v1/result.json")
    code_manifest = json_read(BUNDLE / "source_manifest_v3.json")
    declared = {row["path"]: row for row in code_manifest["files"]}
    for name in ("scripts/train_paper_amp.py", "scripts/benchmark_paper_motion_tracking.py",
                 "CrowdSim/humanoid/application/configuration.py", "CrowdSim/humanoid/training/trainer.py"):
        actual = record(CODE / name)
        expected = declared["code/" + name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError("Frozen isolated entrypoint identity changed:" + name)
    profile_path = CODE / "configs/diffusion_amp/isolated_state_action_absolute.yaml"
    profile = load_amp_profile(profile_path)
    if (profile["training"]["discriminator_input"] != "state_action" or
            profile["training"]["warm_start_discriminator"] is not False or
            profile["initialization"]["discriminator"] != "independent_random_history_and_action"):
        raise ValueError("State-action training profile differs")
    checkpoint = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt"
    manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    diffusion = ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"
    nav_output = ROOT / "output/diffusion_amp/shared_mixed_state_action_isolated_ground_support_200s_v1"
    tracking_output = ROOT / "output/diffusion_amp/motion_tracking_mixed_state_action_isolated_full_validation_v1"
    scene = CODE / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
    nav_args = dict(original["arguments"])
    nav_args.update(config=str(profile_path), manifest=str(manifest), diffusion_checkpoint=str(diffusion),
                    scene_config=str(scene), output=str(nav_output), resume=str(checkpoint))
    # Materialize all existing flags with exactly the baseline values; no new
    # native/inference/arm options exist in the old146 isolated driver.
    nav_argv = []
    for key, value in nav_args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        nav_argv.append(flag)
        if value is not True:
            nav_argv.append(str(value))
    parsed_nav = vars(parse_amp_args(nav_argv))
    if parsed_nav != nav_args:
        raise ValueError("Baseline-matching navigation flags parsed differently")
    validate_training_source(parse_amp_args(nav_argv), profile)
    tracking_argv = ["--checkpoint", str(checkpoint), "--manifest", str(manifest), "--split", "validation",
        "--minimum-evaluation-seconds", "1", "--failure-threshold-m", ".5",
        "--maximum-terrain-extent-m", "256", "--deterministic-evaluation", "--seed", "7",
        "--cpu-threads", "4", "--output", str(tracking_output)]
    spec = importlib.util.spec_from_file_location("registered_isolated_tracking_parser",
        CODE / "scripts/benchmark_paper_motion_tracking.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    parsed_tracking = vars(benchmark.parse_args(tracking_argv))
    parsed_tracking = {key: str(value) if isinstance(value, Path) else value for key, value in parsed_tracking.items()}
    expected_tracking = dict(old_tracking["arguments"])
    expected_tracking.update(checkpoint=str(checkpoint), checkpoint_sha256=None, output=str(tracking_output))
    if parsed_tracking != expected_tracking:
        raise ValueError("Complete22 tracking protocol differs from actual mixed2000 validation")
    module_paths = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                    if name.startswith(("CrowdSim", "protomotions")) and getattr(module, "__file__", None)}
    if not all(Path(path).is_relative_to(CODE) for path in module_paths.values()):
        raise ValueError("CPU profile parser imported live business code")
    if any(name in sys.modules for name in ("torch", "numpy", "omni", "isaacsim", "isaaclab")):
        raise ValueError("Configuration-only registration imported numerical/SDK runtime")
    data = json_read(manifest)
    reference = DOCS / "backward_penalty_style_v1/heldout_reference.json"
    if record(reference)["sha256"] != "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564":
        raise ValueError("Frozen style reference changed")
    if len(data["splits"]["validation"]["clips"]) != 22:
        raise ValueError("Complete validation set changed")
    commands = json_read(BUNDLE / "commands_v3.json")
    output = {"schema": "state_action.future_evaluation_registration.v1",
        "status": "protocol_and_isolated_parsers_passed_future_checkpoint_pending",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
        "Kit_started": False, "GPU_used": False, "sealed34_outputs_read": False,
        "future_checkpoint": {"path": str(checkpoint), "sha256": None, "bytes": None,
            "actual_payload_preflight_performed": False, "actual_model_effect_evaluated": False,
            "required_update_count": 2000, "required_transitions": 16384000,
            "required_D_input": 2019, "public_D_or_AMP_RMS_transfer_allowed": False,
            "required_full_payload_and_actual_import_ledger_admission": True},
        "sources": {"isolated_source_manifest_v3": record(BUNDLE / "source_manifest_v3.json"),
            "isolated_launcher_v3": record(BUNDLE / "launch_isolated_v3.py"),
            "actual_fresh_training_commands_v3": record(BUNDLE / "commands_v3.json"),
            "fixed_manifest": record(manifest), "frozen_style_reference": record(reference),
            "tracking_baseline_result": record(ROOT / "output/diffusion_amp/motion_tracking_mixed_curriculum_step2000_full_validation_v1/result.json"),
            "navigation_baseline_result": record(ROOT / "output/diffusion_amp/shared_mixed_curriculum_step2000_ground_support_200s_v1/result.json"),
            "registered_profile": record(profile_path), "scene": record(scene),
            "entrypoint_bytes": {name: record(CODE / name) for name in
                ("scripts/train_paper_amp.py", "scripts/benchmark_paper_motion_tracking.py")}},
        "fresh_training_observed_launched_by_parent": True,
        "fresh_training_output": commands["commands"]["fullfresh2000"][commands["commands"]["fullfresh2000"].index("--output") + 1],
        "tracking": {"argv_without_future_SHA": tracking_argv, "parsed": parsed_tracking,
            "required_actual_launch_addition": ["--checkpoint-sha256", "ACTUAL_SEALED_SHA256"],
            "all_validation_clips": 22, "motion_ids": None, "max_clips": None, "max_steps": None,
            "expected_complete_metric_frames_from_same_fixture": 11982,
            "no_alignment_or_fall_filter": True, "physics": "same120Hz/30Hz SMPL PD, rootXY-only oracle reference",
            "source_execution": "Use exact isolated CODE benchmark and v3 import guard/ledger; no current-tree benchmark substitution."},
        "navigation": {"argv": nav_argv, "parsed": parsed_nav, "actors": 4, "controls": 6000,
            "duration_s": 200, "guide_weight": 20, "near_horizon_waypoint_mass": 0,
            "source_execution": "Existing v3 launcher train_paper_amp entry; sole App/world; no pure inference/arm/native provider change.",
            "runtime_argument_difference_from_mixed2000_baseline": ["config", "resume", "output", "absolute_resource_paths_to_same_bytes"],
            "train_profile_mode_is_not_deployment_curriculum": True},
        "metrics": {"primary": "Inclusive fixedc274 90D pose+velocity Gaussian MMD2 sigma1, original speedbins/min30/max256",
            "secondary": ["upright subset with original .8 root_up/.7 pelvis thresholds", "physical speedW1",
                "native floor normal-force support and body-origin velocity proxy", "route arrivals", "full22 .5m success/falls"],
            "all_failed_frames_kept": True, "cross_reset_derivative_intervals_only_excluded": True,
            "unsupported_bins": "unknown, never zero", "short_vs200_MMD_direct_comparison": False},
        "acceptance": {"execution_pass_is_quality_pass": False, "training_budget": "fresh16.384M versus mixed2000 same16.384M",
            "navigation_promotion_minimum": {"complete_controls": 6000, "falls": 0,
                "arrivals_not_below_negative_guide20": 21, "inclusive_style_not_worse_than_negative_guide20": True},
            "tracking_and_style_results_must_both_be_reported": True,
            "new_fall_tradeoff_cannot_be_hidden_by_MMD_or_arrivals": True,
            "author_paper_threshold_or_equivalence_claim": False,
            "single_seed_population_causality_claim": False},
        "limitations": ["Future checkpoint SHA/payload/resource preflight and all GPU effects are pending, not fabricated.",
            "State-action changes D initialization/RMS and pre-step conditioning, not merely69 input dimensions.",
            "Expert actions are next-frame normalized PD position proxies, not measured expert controls or inverse dynamics.",
            "Local2848 derived segments are not confirmed author originalIDs; no author8-scene replication.",
            "Validation has been repeatedly used; sealed34 outputs are not consulted for this variant/protocol.",
            "Native floor-force velocity proxy is not contact-point slip or AMASS force ground truth."]}
    OUT.mkdir()
    write(OUT / "registration.json", output)
    write(OUT / "configuration_only_imports.json", {"owned_modules": module_paths,
        "torch_imported": False, "numpy_imported": False, "SDK_imported": False})
    print(json.dumps({"status": output["status"], "tracking_clips": 22, "navigation_controls": 6000,
        "future_checkpoint_exists_at_registration": checkpoint.exists()}))


if __name__ == "__main__":
    run()
