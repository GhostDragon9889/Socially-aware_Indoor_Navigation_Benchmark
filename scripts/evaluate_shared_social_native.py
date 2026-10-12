#!/usr/bin/env python3
"""Prepared N3..15 native Diffusion/SFM + frozen shared Social AMP entry.

Native SDK smoke and exact paired initial-state capture remain required before
admitting any200s result. Source/metadata preflight does not claim those gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from CrowdSim.humanoid.runtime.native_social_navigation_lifecycle import NativeNavigationLifecycle  # noqa: E402
from CrowdSim.humanoid.runtime import native_social_observation_producer, social_popart_inference  # noqa: E402
from scripts import evaluate_shared_social_navigation as canonical  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402
from scripts.native_shared_social_derivation import derive_native_sfm_driver  # noqa: E402

NATIVE_PARENT = ROOT / "resources/runtime_sources/style_improvement_20261004/social_native_evaluation_staging_v1/evaluate_native_social.py"
NATIVE_PARENT_SHA = "4090ef7f37b46b853fbb066c4e0486ccc1a3a9f50eceaf69c614bcc435ad473b"
CANONICAL_SHA = "0e21b47d7d983851f1cf39349e3cb44476fec7381a476fa48df98f29f088d097"


def prepare_entry(*, actors, planner):
    if type(actors) is not int or not 3 <= actors <= 15 or planner not in {"diffusion", "sfm"}:
        raise ValueError("Require native N3..15 and one explicitly selected planner")
    parent, original = record(NATIVE_PARENT), record(canonical.PARENT)
    if (parent["sha256"] != NATIVE_PARENT_SHA or NATIVE_PARENT.stat().st_mode & 0o222
            or original["sha256"] != canonical.PARENT_SHA
            or record(canonical.__file__)["sha256"] != CANONICAL_SHA):
        raise ValueError("Native/canonical evaluator source changed")
    staging = ModuleType("_owned_native_social_sealed_entry")
    staging.__file__ = str(NATIVE_PARENT)
    exec(compile(NATIVE_PARENT.read_bytes(), str(NATIVE_PARENT), "exec"), staging.__dict__)
    source, native_proof = staging._derive_parent(canonical.PARENT.read_text())
    old = '        module = _load("_social_native_observations", sources["producer"])\n'
    new = '        module = _load("_social_navigation_observations", sources["producer"])\n'
    if source.count(old) != 1:
        raise ValueError("Ambiguous native observation facade seam")
    seam = source.replace(old, new, 1)
    if seam.count(new) != 1 or seam.replace(new, old, 1) != source:
        raise ValueError("Native producer compatibility seam is not exactly reversible")
    derived, migration = canonical.derive_entry(seam)
    metadata_old = "    if args.preflight_only:\n"
    metadata_new = '    preparation["native_current_migration"] = _migration_binding\n' + metadata_old
    if derived.count(metadata_old) != 1:
        raise ValueError("Ambiguous native migration metadata seam")
    previous = derived
    derived = derived.replace(metadata_old, metadata_new, 1)
    if derived.count(metadata_new) != 1 or derived.replace(metadata_new, metadata_old, 1) != previous:
        raise ValueError("Native preflight metadata seam is not reversible")
    bootstrap = {}
    lifecycle = NativeNavigationLifecycle(bootstrap, actors=actors)
    binding = {"entry": record(__file__), "native_staging_entry": parent, "original_four_actor_entry": original,
        "native_entry_derivation": native_proof, "canonical_entry_migration": migration,
        "producer_compatibility_seam_exact_inverse": True,
        "preflight_migration_metadata_seam_exact_inverse": True,
        "native_observation_migration": native_social_observation_producer.describe_migration(),
        "native_lifecycle": bootstrap["evaluation_lifecycle"], "planner": planner,
        "actual_native_SDK_execution_verified": False, "exact_native_paired_initial_state_verified": False,
        "native_stage": "source_preparation_requires_actual_smoke"}
    namespace = ModuleType("_migrated_native_shared_social_evaluation")
    namespace.__file__ = str(NATIVE_PARENT)
    namespace.__dict__.update(_native_parent_record=original, _native_entry_derivation=native_proof,
        _native_resource_records=staging._native_resource_records,
        _migration_entry=str(Path(__file__).resolve()), _migration_lifecycle=lifecycle,
        _migration_bootstrap=bootstrap, _migration_binding=binding,
        _migration_observation_producer=native_social_observation_producer,
        _migration_producer_path=native_social_observation_producer.__file__, _migration_checkpoint_loader=None)
    exec(compile(derived, str(NATIVE_PARENT) + ":current_migration", "exec"), namespace.__dict__)
    from CrowdSim.humanoid.runtime.evaluation_source_migration import migrate_pins
    binding["dependency_migration"] = migrate_pins(namespace)
    if planner == "sfm":
        base_derive = namespace._derive
        namespace._derive = lambda value: derive_native_sfm_driver(value, base_derive)
        binding["SFM_sources"] = [record(ROOT / name) for name in (
            "scripts/native_shared_social_derivation.py", "scripts/evaluate_shared_social_sfm.py",
            "CrowdSim/humanoid/trajectory/native_social_force_runtime.py",
            "CrowdSim/humanoid/trajectory/social_force_runtime.py",
            "CrowdSim/humanoid/trajectory/social_force.py", "CrowdSim/world/sfm.py")]
    binding["prepared_entry_sha256"] = hashlib.sha256(derived.encode()).hexdigest()
    return namespace, binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)
    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--scene-sha256")
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--diffusion-sha256")
    parser.add_argument("--control-steps", type=int, choices=(4, 30, 6000), default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args(argv)
    namespace, binding = prepare_entry(actors=options.actors, planner=options.planner)
    guarded = [record(ROOT / name) for name in (
        "scripts/evaluate_shared_social_native.py", "scripts/evaluate_shared_social_navigation.py",
        "CrowdSim/humanoid/runtime/native_social_navigation_lifecycle.py",
        "CrowdSim/humanoid/runtime/native_social_observation_producer.py")]
    guarded.extend(binding.get("SFM_sources", []))
    if options.check_sources:
        sources = namespace._sources()
        namespace._profile(namespace.HERE / "evaluation_profile.json")
        physical, proof = namespace._derive(namespace.DRIVER.read_text())
        compile(physical, str(namespace.DRIVER) + ":native_social", "exec")
        print(json.dumps({"status": "native_source_checks_passed", "Torch_imported": "torch" in sys.modules,
            "SDK_started": False, "migration": binding, "physical_derivation": proof, "bound_sources": len(sources)}))
        return 0
    if any(value is None for value in (options.scene_config, options.scene_sha256, options.execution_receipt,
            options.diffusion_checkpoint, options.diffusion_sha256, options.output)):
        parser.error("Native evaluation requires explicit scene/SHA, completed training receipt, paired DP/SHA and new output")
    checkpoint, sha, admission = admit_training(options.execution_receipt.resolve())
    binding["training_execution_admission"] = admission
    if admission["value_normalization"] == "popart":
        namespace._migration_checkpoint_loader = ModuleType("_native_social_explicit_popart_loader")
        namespace._migration_checkpoint_loader.FrozenSocialCheckpointInference = (
            social_popart_inference.FrozenSocialPopArtCheckpointInference)
        binding["explicit_checkpoint_reader"] = record(social_popart_inference.__file__)
    forwarded = [str(Path(__file__)), "--amp-checkpoint", str(checkpoint), "--amp-sha256", sha,
        "--training-result", admission["result"]["path"], "--training-result-sha256", admission["result"]["sha256"],
        "--diffusion-checkpoint", str(options.diffusion_checkpoint.resolve()), "--diffusion-sha256", options.diffusion_sha256,
        "--actors", str(options.actors), "--scene-config", str(options.scene_config.resolve()),
        "--scene-sha256", options.scene_sha256, "--control-steps", str(options.control_steps),
        "--output", str(options.output.resolve())]
    if options.preflight_only:
        forwarded.append("--preflight-only")
    saved = sys.argv
    try:
        sys.argv = forwarded
        code = namespace.main()
    finally:
        sys.argv = saved
    if any(record(row["path"]) != row for row in guarded):
        raise ValueError("Native evaluator migration source changed during invocation")
    if options.preflight_only:
        return 0
    report = json.loads((options.output / "result.json").read_text())
    return 0 if code in (None, 0) and report.get("status") == "passed" and report.get("cleanup_succeeded") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
