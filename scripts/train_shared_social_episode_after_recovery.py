#!/usr/bin/env python3
"""Run the registered fresh episode candidate under an explicit S3-R release.

Original candidate CLI/gates remain unchanged. This entry selects its already
prepared private module after the distinct completed-recovery serial gate and
adds execution provenance to result and checkpoint; model/physics are unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_serial_release as release  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import train_shared_social_episode_reference as original  # noqa: E402


def provenance_facade(loaded, binding):
    """Return the actual original trainer with an additional provenance field."""
    original_factory = loaded.SocialAMPTrainer

    class DeclaredFactory:
        @staticmethod
        def prepare(config, *, device, provenance):
            declared = {**provenance, "episode_after_recovery_execution": deepcopy(binding)}
            return original_factory.prepare(config, device=device, provenance=declared)

    facade = ModuleType(loaded.__name__ + "_declared_execution_provenance")
    facade.__dict__.update(loaded.__dict__)
    facade.SocialAMPTrainer = DeclaredFactory
    return facade


def prepare_training(release_path, stage):
    document, identity = release.admit_release(release_path)
    matches = [row for row in release.stages() if row["stage"] == stage]
    release.recovery.require(len(matches) == 1, "Select one exact registered candidate stage")
    selected = matches[0]
    module, original_binding = original.prepare_entry()
    release.recovery.require(original_binding == document["candidate_training_binding"], "Candidate binding differs")
    binding = {"schema": "shared_social.episode_after_recovery_execution_entry.v1",
               "entry": record(__file__), "serial_release_helper": record(release.__file__),
               "serial_release": identity, "candidate_entry": original_binding["entry"],
               "consumer_World_admission": release.consumers.describe(),
               "reference_clock_grid_derivation": release.clock_grid.describe(),
               "stage": stage, "profile": selected["profile"], "output": selected["output_directory"],
               "fresh_initialization": True, "weights_or_moments_transferred": False,
               "model_optimizer_control_expert_reward_math_changed": False,
               "original_continuous_S3_completion_claimed": False}
    parent_run, parent_load = module._run, module._load

    def load(path, name, item):
        loaded = parent_load(path, name, item)
        if Path(path) == module.PINS["trainer"][0]:
            return provenance_facade(loaded, binding)
        if Path(path) == module.PINS["probe_setup"][0]:
            return release.clock_grid.adapt_probe(loaded)
        for key in ("expert", "collector"):
            if Path(path) == module.PINS[key][0]:
                return release.consumers.adapt(loaded, key)
        return loaded

    def run(args, profile, profile_record, report, rows):
        release.recovery.require(args.output == Path(selected["output_directory"])
                                 and profile_record == selected["profile"], "Use the registered stage output/profile")
        report["episode_after_recovery_execution"] = deepcopy(binding)
        return parent_run(args, profile, profile_record, report, rows)

    module._load, module._run = load, run
    return module, binding, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serial-release", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("smoke8", "smoke256", "pilot2000"))
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    module, binding, selected = prepare_training(args.serial_release, args.stage)
    module._sources()
    module._profile(Path(selected["profile"]["path"]))
    if args.check_sources:
        print(json.dumps({"status": "episode_after_recovery_preflight_passed", "binding": binding,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    saved = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--config", selected["profile"]["path"],
                    "--output", selected["output_directory"], "--device", "cuda:0"]
        code = module.main()
    finally:
        sys.argv = saved
    release.admit_release(args.serial_release)
    release.recovery.require(record(__file__) == binding["entry"]
                             and record(release.__file__) == binding["serial_release_helper"]
                             and release.consumers.describe() == binding["consumer_World_admission"]
                             and release.clock_grid.describe() == binding["reference_clock_grid_derivation"], "Execution source changed")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
