#!/usr/bin/env python3
"""Evaluate only the registered episode-reference AMP terminal model.

Reuses the unchanged full22, frozen style and N4 physical evaluation pipeline.
The original module is adapted only in a caller-owned private namespace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_evaluation_admission as candidate  # noqa: E402
from CrowdSim.humanoid.application import native_execution_bridge as bridge  # noqa: E402


def prepare_pipeline(registration, serial_release=None):
    release_identity = None
    if serial_release is not None:
        from CrowdSim.humanoid.application import episode_serial_release
        _, release_identity = episode_serial_release.admit_release(serial_release)
    parent = candidate.record(candidate.PARENT)
    if parent != registration["episode_reference_experiment"]["parent"]:
        raise ValueError("Registered independent evaluation parent changed")
    module = ModuleType("_episode_reference_independent_post_training")
    module.__file__ = str(candidate.PARENT)
    exec(compile(candidate.PARENT.read_bytes(), str(candidate.PARENT), "exec"), module.__dict__)
    original_admit = module.admit_training

    def admit(receipt):
        current, _ = candidate.admit_registration(candidate.registration_path(registration))
        if current != registration:
            raise ValueError("Episode evaluation registration changed before admission")
        checkpoint, sha, accepted = candidate.admit_episode_training(receipt, registration, original_admit)
        if release_identity is not None:
            _, identity = episode_serial_release.admit_release(serial_release)
            if identity != release_identity:
                raise ValueError("Episode evaluation release changed")
            accepted = {**accepted, "episode_evaluation_release": {
                "basis": "completed_S3R_independent_evaluation", "serial_release": identity,
                "historical_training_reexecuted": False, "original_S3_certificate_created": False}}
        return checkpoint, sha, accepted

    module.admit_training = admit
    # Parent indices and child commands remain unchanged; new sources are guarded
    # before and after every child alongside the original numerical sources.
    extra = [Path(__file__), Path(candidate.__file__), candidate.registration_path(registration),
             candidate.PLAN, candidate.PROBE]
    if release_identity is not None:
        extra.extend((Path(serial_release), Path(episode_serial_release.__file__)))
    training_binding = registration["episode_reference_experiment"]["training_binding"]
    extra.extend(candidate.artifact_path(training_binding[key]["path"]) for key in ("entry", "World", "curriculum", "reference_primitive"))
    module.SOURCE_PATHS = [*module.SOURCE_PATHS, *(str(path.resolve().relative_to(ROOT)) for path in extra)]
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-sources", action="store_true")
    checks.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--serial-release", type=Path,
                        help="Explicit completed S3-R release for historical checkpoint evaluation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    registration, identity = candidate.admit_registration()
    module = prepare_pipeline(registration, args.serial_release)
    if args.check_sources:
        print(json.dumps({"status": "episode_evaluation_source_checks_passed", "registration": identity,
            "guarded_source_count": len(module.SOURCE_PATHS), "Torch_imported": "torch" in sys.modules,
            "SDK_started": False, "training_or_quality_proven": False}))
        return 0
    if args.execution_receipt is None:
        parser.error("--execution-receipt is required")
    _, _, admission = module.admit_training(args.execution_receipt.resolve())
    if args.preflight_only:
        print(json.dumps({"status": "episode_terminal_evaluation_admitted", "admission": admission,
            "Torch_imported": "torch" in sys.modules, "SDK_started": False, "quality_improvement_proven": False}))
        return 0
    if args.output is None or args.output.resolve() != candidate.OUTPUT:
        parser.error("Use the registered independent episode evaluation output: " + str(candidate.OUTPUT))
    candidate.training.pinned(ROOT / "CrowdSim/humanoid/application/native_execution_bridge.py",
                              candidate.training.BRIDGE_SHA)
    if args.serial_release is None:
        bridge.admit_S3_owner_completion(candidate.training.COMPLETION)
    return module.main(["--execution-receipt", str(args.execution_receipt.resolve()),
                        "--registration", str(candidate.REGISTRATION), "--output", str(args.output.resolve())])


if __name__ == "__main__":
    raise SystemExit(main())
