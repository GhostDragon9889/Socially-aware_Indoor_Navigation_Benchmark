#!/usr/bin/env python3
"""Migrate the sealed four-person Diffusion/Social AMP physical evaluator.

The planner, scene, reference timing and measurements retain their original
code. Only entry provenance, writable manifest admission, native bootstrap and
terminal owner release change. Actual smoke must precede 200-second evaluation.
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
from CrowdSim.humanoid.runtime.social_navigation_lifecycle import FourActorNavigationLifecycle  # noqa: E402
from CrowdSim.humanoid.runtime import social_observation_producer  # noqa: E402
from CrowdSim.humanoid.runtime import social_popart_inference  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402

PARENT = ROOT / "docs/style_improvement_20261004/social_navigation_evaluation_staging_v1/evaluate_four_actor_social.py"
PARENT_SHA = "c31c87391ab62c2c33369fc437c6b8967546987a505e885b7954ba9349107d8e"


def derive_entry(original):
    replacements = [
        ('        owned["world"] = factory(*positional, **keywords)\n',
         '        constructor = (_migration_lifecycle._sdk61.construct_world\n'
         '                       if _migration_lifecycle._sdk61 is not None else lambda f, *a, **k: f(*a, **k))\n'
         '        owned["world"] = constructor(factory, *positional, **keywords)\n'),
        ('    result["entry"] = _identity(Path(__file__), readonly=True)\n',
         '    result["parent_entry"] = _identity(Path(__file__), readonly=True)\n'
         '    result["entry"] = _identity(Path(_migration_entry))\n'
         '    result["scripted_origin_adapter"] = _identity(Path(_migration_producer_path))\n'),
        ('"manifest": _identity(MANIFEST, readonly=True, expected_sha=MANIFEST_SHA)',
         '"manifest": _identity(MANIFEST, expected_sha=MANIFEST_SHA)'),
        ('readonly=role != "scene", expected_sha=identity["sha256"]',
         'readonly=role not in {"scene", "manifest"}, expected_sha=identity["sha256"]'),
        ('        owned["app"] = launcher(options).app\n',
         '        options["kit_args"] = options.get("kit_args", "") + " " + _migration_lifecycle.kit_args\n'
         '        owned["app"] = launcher(options).app\n'
         '        _migration_lifecycle.verify_loaded()\n'),
        ('        producer.begin_episode()\n',
         '        producer.begin_episode()\n        owned["producer"] = producer\n'),
        ('        report["social_entry"] = preparation\n',
         '        report["social_entry"] = preparation\n'
         '        report["shared_navigation_migration"] = _migration_binding\n'),
        ('        module = _load("_social_navigation_observations", sources["producer"])\n',
         '        module = _migration_observation_producer\n'),
        ('        loader = _load("_social_navigation_checkpoint", sources["loader"])\n',
         '        loader = (_migration_checkpoint_loader if _migration_checkpoint_loader is not None\n'
         '                  else _load("_social_navigation_checkpoint", sources["loader"]))\n'),
    ]
    start = original.index("        sim = None\n", original.index("    def finish("))
    end = original.index('        if report["status"] == "passed" and report["social_observation_capture"]', start)
    old_cleanup = original[start:end]
    new_cleanup = '''        with (path / "cleanup_pending.json").open("x") as stream:
            json.dump({"status": "cleanup_pending", "actual_control_steps": report.get("actual_control_steps"),
                       "sources": _migration_binding, "actual_process_exit_observed": False}, stream, indent=2)
            stream.write("\\n")
        report.update(_migration_bootstrap)
        _migration_lifecycle.report = report
        _migration_lifecycle.close(app, world, owned.get("producer"))
        owned["app"] = owned["world"] = owned["producer"] = None
        report["cleanup_errors"].extend(errors)
        if errors or not report["cleanup_succeeded"]:
            report["cleanup_succeeded"] = False
            report["status"] = "failed"
'''
    replacements.append((old_cleanup, new_cleanup))
    derived = original
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError("Ambiguous navigation entry migration")
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(replacements):
        if inverse.count(after) != 1:
            raise ValueError("Ambiguous navigation migration inverse")
        inverse = inverse.replace(after, before, 1)
    if inverse != original:
        raise ValueError("Navigation entry migration is not exactly reversible")
    compile(derived, str(PARENT) + ":navigation_migration", "exec")
    return derived, {"parent_sha256": hashlib.sha256(original.encode()).hexdigest(),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "replacement_count": len(replacements), "exact_inverse": True,
                     "physical_driver_changes_added": False, "metric_math_changes_added": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--diffusion-sha256")
    parser.add_argument("--control-steps", type=int, choices=(4, 30, 6000), default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args(argv)
    parent_record = record(PARENT)
    if parent_record["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
        raise ValueError("Sealed four-person physical evaluator changed")
    source, derivation = derive_entry(PARENT.read_text())
    bootstrap = {}
    lifecycle = FourActorNavigationLifecycle(bootstrap)
    binding = {"entry": record(Path(__file__)), "parent": parent_record, "derivation": derivation,
               "native_lifecycle": bootstrap["evaluation_lifecycle"], "actual_execution_verified": False,
               "observation_producer_migration": social_observation_producer.describe_migration()}
    namespace = ModuleType("_migrated_shared_social_navigation")
    namespace.__file__ = str(PARENT)
    namespace.__dict__.update(_migration_entry=str(Path(__file__).resolve()), _migration_lifecycle=lifecycle,
                              _migration_bootstrap=bootstrap, _migration_binding=binding,
                              _migration_observation_producer=social_observation_producer,
                              _migration_producer_path=social_observation_producer.__file__,
                              _migration_checkpoint_loader=None)
    exec(compile(source, str(PARENT) + ":navigation_migration", "exec"), namespace.__dict__)
    from CrowdSim.humanoid.runtime.evaluation_source_migration import migrate_pins
    binding["dependency_migration"] = migrate_pins(namespace)
    if options.check_sources:
        sources = namespace._sources()
        physical, proof = namespace._derive(namespace.DRIVER.read_text())
        compile(physical, str(namespace.DRIVER) + ":social_derived", "exec")
        print(json.dumps({"status": "source_checks_passed", "Torch_imported": "torch" in sys.modules,
                          "SDK_started": False, "migration": binding, "physical_derivation": proof,
                          "bound_sources": len(sources)}))
        return 0
    if any(value is None for value in (options.execution_receipt, options.diffusion_checkpoint, options.diffusion_sha256, options.output)):
        parser.error("--execution-receipt, --diffusion-checkpoint, --diffusion-sha256 and --output are required")
    checkpoint, sha, admission = admit_training(options.execution_receipt.resolve())
    binding["training_execution_admission"] = admission
    if admission["value_normalization"] == "popart":
        binding["explicit_checkpoint_reader"] = {
            "adapter": record(social_popart_inference.__file__), "numeric_migration": admission["numeric_migration"]}
        namespace._migration_checkpoint_loader = ModuleType("_navigation_popart_facade")
        namespace._migration_checkpoint_loader.FrozenSocialCheckpointInference = (
            social_popart_inference.FrozenSocialPopArtCheckpointInference)
    forwarded = [str(Path(__file__)), "--amp-checkpoint", str(checkpoint), "--amp-sha256", sha,
        "--training-result", admission["result"]["path"], "--training-result-sha256", admission["result"]["sha256"],
        "--diffusion-checkpoint", str(options.diffusion_checkpoint.resolve()), "--diffusion-sha256", options.diffusion_sha256,
        "--control-steps", str(options.control_steps), "--output", str(options.output.resolve())]
    if options.preflight_only:
        forwarded.append("--preflight-only")
    saved_argv = sys.argv
    try:
        sys.argv = forwarded
        code = namespace.main()
    finally:
        sys.argv = saved_argv
    if options.preflight_only:
        return 0
    result = json.loads((options.output / "result.json").read_text())
    return 0 if code in (None, 0) and result.get("status") == "passed" and result.get("cleanup_succeeded") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
