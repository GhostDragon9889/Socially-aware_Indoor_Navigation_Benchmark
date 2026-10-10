#!/usr/bin/env python3
"""Evaluate the fresh episode candidate after an actually completed serial run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_serial_release as release  # noqa: E402
from CrowdSim.humanoid.application import episode_evaluation_admission as admission  # noqa: E402
from scripts import run_shared_social_episode_evaluation as original  # noqa: E402
from scripts import train_shared_social_episode_after_recovery as training  # noqa: E402

STEPS = [name + suffix for name in ("smoke8", "smoke256", "pilot2000") for suffix in ("_training", "_CPU_audit")]


def validate_training_owner(document):
    release.recovery.require(document.get("schema") == "shared_social.episode_after_recovery_training_completion.v1"
                             and document.get("status") == "completed_fresh_training_and_CPU_audits"
                             and [row.get("name") for row in document.get("steps", [])] == STEPS
                             and all(row.get("status") == "passed"
                                     and release.recovery.exact_integer(row.get("actual_exit_code"), 0)
                                     and row.get("exit_observation") == "subprocess.Popen.wait"
                                     for row in document["steps"]),
                             "Require all three real candidate training and CPU audit stages")


def prepare_pipeline(release_path):
    release.admit_release(release_path)
    registration, _ = admission.admit_registration()
    module = original.prepare_pipeline(registration)
    _, training_binding, _ = training.prepare_training(release_path, "pilot2000")
    parent_admit = module.admit_training

    def admit(receipt_path):
        release.admit_release(release_path)
        checkpoint, sha, accepted = parent_admit(receipt_path)
        receipt = json.loads(Path(receipt_path).read_text())
        owner_identity = receipt["episode_training_owner_completion"]
        release.recovery.require(release.recovery.readonly_identity(owner_identity["path"]) == owner_identity,
                                 "Candidate training owner completion changed")
        completed = json.loads(Path(owner_identity["path"]).read_text())
        validate_training_owner(completed)
        release.recovery.require(completed["result"] == accepted["result"]
                                 and completed["audit"] == accepted["audit"]
                                 and completed["checkpoint"] == accepted["checkpoint"], "Candidate owner/result/audit disagree")
        result = json.loads(Path(accepted["result"]["path"]).read_text())
        provenance = json.loads((checkpoint.parent / "resolved_configuration.json").read_text())["provenance"]
        release.recovery.require(result.get("episode_after_recovery_execution") == training_binding
                                 and provenance.get("episode_after_recovery_execution") == training_binding,
                                 "Candidate result and saved training provenance lack the actual serial execution binding")
        return checkpoint, sha, {**accepted, "episode_after_recovery_execution": {
            "training_owner_completion": owner_identity, "training_binding": training_binding,
            "evaluation_entry": release.record(__file__), "serial_release": release.record(release_path)}}

    module.admit_training = admit
    module.SOURCE_PATHS.extend(str(Path(path).resolve().relative_to(ROOT)) for path in
        (Path(__file__), Path(training.__file__), Path(release.__file__),
         Path(release.consumers.__file__), Path(release.clock_grid.__file__), Path(release_path)))
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serial-release", required=True, type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    module = prepare_pipeline(args.serial_release)
    if args.check_sources:
        print(json.dumps({"status": "episode_after_recovery_evaluation_preflight_passed",
                          "source_count": len(module.SOURCE_PATHS), "Torch_imported": "torch" in sys.modules,
                          "SDK_started": False, "metric_math_changed": False}))
        return 0
    if args.execution_receipt is None:
        parser.error("--execution-receipt is required")
    return module.main(["--execution-receipt", str(args.execution_receipt.resolve()),
                        "--registration", str(admission.REGISTRATION), "--output", str(admission.OUTPUT)])


if __name__ == "__main__":
    raise SystemExit(main())
