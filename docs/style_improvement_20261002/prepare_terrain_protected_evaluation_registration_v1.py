"""CPU register isolated931 full22/200s protocols; future effects stay pending."""

from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_protected_v1"
CODE = BUNDLE / "code"
OUT = DOCS / "terrain_protected_evaluation_registration_v1"
OLD = DOCS / "state_action_evaluation_registration_v1/registration.json"


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def replace_flag(argv, flag, value):
    if argv.count(flag) != 1:
        raise ValueError("Expected one fixed protocol flag:" + flag)
    argv[argv.index(flag) + 1] = str(value)


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or OUT.exists():
        raise ValueError("CPU-only and fresh registration required")
    old = load(OLD)
    experiment = DOCS / "terrain_protected_experiment_registration_v1/registration.json"
    source = BUNDLE / "source_manifest.json"
    launcher = BUNDLE / "launch_isolated.py"
    spec = importlib.util.spec_from_file_location(
        "protected_registered_isolation", launcher
    )
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    manifest = isolated.verify()
    if len(manifest["files"]) != 931:
        raise ValueError("Only the registered isolated931 copy accepted")
    isolated.activate()
    from CrowdSim.humanoid.application.configuration import (
        load_amp_profile,
        parse_amp_args,
        validate_training_source,
    )

    checkpoint = (
        ROOT
        / "output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_snapshots_v1/step_2000.pt"
    )
    profile_path = (
        CODE / "configs/diffusion_amp/isolated_terrain_protected_absolute.yaml"
    )
    scene = CODE / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
    profile = load_amp_profile(profile_path)
    if (
        profile["training"]["task_terrain_statistics_mode"]
        != "preserve_declared_suffix"
        or profile["training"]["discriminator_input"] != "state_action"
        or profile["training"]["warm_start_discriminator"] is not False
    ):
        raise ValueError("New protected state-action profile differs")
    old_profile = load_amp_profile(
        ROOT
        / "output/isolated_training/mixed_state_action_v1/code/configs/diffusion_amp/isolated_state_action_absolute.yaml"
    )
    changed_fields = [
        k
        for k in sorted(set(old_profile["training"]) | set(profile["training"]))
        if old_profile["training"].get(
            k, "legacy" if k == "task_terrain_statistics_mode" else None
        )
        != profile["training"].get(k)
    ]
    if changed_fields != ["task_terrain_statistics_mode"]:
        raise ValueError("Training differs beyond the declared single terrain policy")
    nav_argv = list(old["navigation"]["argv"])
    nav_output = ROOT / "output/diffusion_amp/protected_nav2000_v1"
    track_output = ROOT / "output/diffusion_amp/protected_tracking2000_v1"
    for flag, value in (
        ("--config", profile_path),
        ("--scene-config", scene),
        ("--resume", checkpoint),
        ("--output", nav_output),
    ):
        replace_flag(nav_argv, flag, value)
    nav_parsed = vars(parse_amp_args(nav_argv))
    expected_nav = dict(old["navigation"]["parsed"])
    expected_nav.update(
        config=str(profile_path),
        scene_config=str(scene),
        resume=str(checkpoint),
        output=str(nav_output),
    )
    if nav_parsed != expected_nav:
        raise ValueError("New navigation parser changes frozen old57 arguments")
    validate_training_source(parse_amp_args(nav_argv), profile)
    track_argv = list(old["tracking"]["argv_without_future_SHA"])
    replace_flag(track_argv, "--checkpoint", checkpoint)
    replace_flag(track_argv, "--output", track_output)
    spec = importlib.util.spec_from_file_location(
        "protected_registered_tracking_parser",
        CODE / "scripts/benchmark_paper_motion_tracking.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    track_parsed = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(benchmark.parse_args(track_argv)).items()
    }
    expected_track = dict(old["tracking"]["parsed"])
    expected_track.update(
        checkpoint=str(checkpoint), checkpoint_sha256=None, output=str(track_output)
    )
    if track_parsed != expected_track:
        raise ValueError("Full22 protocol changed")
    declared = {r["path"]: r for r in manifest["files"]}
    entries = {}
    for name in (
        "scripts/train_paper_amp.py",
        "scripts/benchmark_paper_motion_tracking.py",
    ):
        info = record(CODE / name)
        expected = declared["code/" + name]
        if info["sha256"] != expected["sha256"] or info["bytes"] != expected["bytes"]:
            raise ValueError("Registered931 entrypoint byte mismatch")
        if info["sha256"] != old["sources"]["entrypoint_bytes"][name]["sha256"]:
            raise ValueError("Control/benchmark source changed beyond old926 RMSdelta")
        entries[name] = info
    reference = Path(old["sources"]["frozen_style_reference"]["path"])
    if (
        record(reference) != old["sources"]["frozen_style_reference"]
        or record(scene)["sha256"] != old["sources"]["scene"]["sha256"]
    ):
        raise ValueError("Frozen90D/reference/fixture identity differs")
    if any(
        k in sys.modules for k in ("torch", "numpy", "omni", "isaacsim", "isaaclab")
    ):
        raise ValueError(
            "Pure configuration preparation imported numerical/SDK runtime"
        )
    modules = {
        k: str(Path(v.__file__).resolve())
        for k, v in sys.modules.items()
        if k.split(".")[0] in {"CrowdSim", "protomotions"}
        and getattr(v, "__file__", None)
    }
    if any(not Path(v).is_relative_to(CODE) for v in modules.values()):
        raise ValueError("Actual CPU-owned import escaped isolated931 code")
    data = load(ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")
    if len(data["splits"]["validation"]["clips"]) != 22:
        raise ValueError("All22 validation source changed")
    registration = deepcopy(old)
    registration.update(
        schema="protected_terrain.future_evaluation_registration.v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        fresh_training_observed_launched_by_parent=False,
        fresh_training_output=str(
            ROOT
            / "output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_v1"
        ),
    )
    registration["future_checkpoint"].update(
        path=str(checkpoint),
        required_terrain_policy="preserve_declared_suffix",
        required_reference_count=61931520001.0,
        required_warm_count=200000.0,
        required_post_update_calls=2000,
    )
    sources = registration["sources"]
    sources.pop("isolated_source_manifest_v3")
    sources.pop("isolated_launcher_v3")
    sources.pop("actual_fresh_training_commands_v3")
    sources.update(
        isolated_source_manifest=record(source),
        isolated_launcher=record(launcher),
        actual_fresh_training_commands=record(BUNDLE / "commands_v2.json"),
        registered_profile=record(profile_path),
        scene=record(scene),
        entrypoint_bytes=entries,
        original_protocol_registration=record(OLD),
        protected_experiment_registration=record(experiment),
    )
    registration["tracking"].update(
        argv_without_future_SHA=track_argv,
        parsed=track_parsed,
        source_execution="Exact isolated931 old full-model adapter/benchmark and guarded import ledger; no FrozenAMPActor/current-tree substitution",
    )
    registration["navigation"].update(
        argv=nav_argv,
        parsed=nav_parsed,
        source_execution="Exact isolated931 old full-model adapter/sole driver/guarded ledger; no pure inference, arm or native provider change",
    )
    registration["matched_protocol_and_source"] = {
        "training_single_variable": "task_terrain_statistics_mode legacy -> preserve_declared_suffix",
        "old_state_action_full2000_source": str(
            ROOT / "output/isolated_training/mixed_state_action_v1/code"
        ),
        "new_source931_epoch": manifest["source_epoch"],
        "two_entrypoints_bitwise_same_old926": True,
        "full_model_adapter_both_variants": True,
        "FrozenAMPActor_reduction_used": False,
        "against_earlier_state_only_or_negative_baselines": "Multiple changes in curriculum,D/action initialization,timestamps,budget/RNG; not one-variable comparison",
        "all_validation22": True,
        "fixed_c27490D_reference": True,
        "no_test34_read": True,
        "45D_pose_centers": "Existing frozen AMASS-only centers; do not refit, select good frames or promote this secondary method to original90D",
    }
    registration["pending"] = {
        "future_checkpoint_exists_now": checkpoint.exists(),
        "future_full_payload_admitted": False,
        "future_training_full2000_started_by_this_preparation": False,
        "physical_evaluation_started": False,
    }
    if checkpoint.exists() or nav_output.exists() or track_output.exists():
        raise ValueError(
            "This preregistration requires fresh unevaluated checkpoint/output paths"
        )
    OUT.mkdir()
    for path, value in (
        (OUT / "registration.json", registration),
        (
            OUT / "configuration_only_imports.json",
            {
                "owned_modules": modules,
                "SDK_imported": False,
                "Torch_imported": False,
                "NumPy_imported": False,
            },
        ),
    ):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": registration["status"],
                "tracking_clips": 22,
                "navigation_controls": 6000,
                "future_checkpoint_admitted": False,
                "copied931": True,
            }
        )
    )


if __name__ == "__main__":
    run()
