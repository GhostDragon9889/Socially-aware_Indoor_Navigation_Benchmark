#!/usr/bin/env python3
"""Reuse the sealed five-stage evaluation for a registered training budget.

Only terminal-budget admission changes. Tracking windows, frozen style,
Diffusion, controls, contacts and subprocess/cleanup gates remain unchanged.
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

PARENT = ROOT / "scripts/run_shared_social_post_training.py"
PARENT_SHA = "2890d628361fbe1c548d822a82bbcbedde215f6395db44d7327a1903a7199cf8"
SCHEMA = "shared_social.registered_budget_evaluation.v1"


def registered_budget(registration, config):
    """Derive exposure from declared actors/T/updates, never from filenames."""
    if registration.get("schema") != SCHEMA:
        raise ValueError("Require an explicit registered terminal-budget evaluation")
    runtime = config["runtime"]
    values = [runtime[key] for key in ("actors", "rollout_controls", "updates")]
    if any(type(value) is not int or value < 1 for value in values) or values[0] % 4:
        raise ValueError("Registered actor/control/update counts must be positive exact integers and complete groups")
    actors, controls, updates = values
    transitions = actors * controls * updates
    if (type(registration.get("planned_updates")) is not int or registration["planned_updates"] != updates
            or type(registration.get("planned_physical_transitions")) is not int
            or registration["planned_physical_transitions"] != transitions
            or registration.get("preselected_checkpoint") != f"step_{updates:06d}.pt"):
        raise ValueError("Registered terminal checkpoint and exposure disagree with the actual configuration")
    return updates, transitions


def derive(source):
    replacements = (
        ('    expected_config = json.loads(Path(registration["configuration"]["path"]).read_text())\n',
         '    expected_config = json.loads(Path(registration["configuration"]["path"]).read_text())\n'
         '    expected_updates, expected_transitions = _registered_budget(registration, expected_config)\n'),
        ('registration.get("schema") != "shared_social.popart_paired_experiment_registration.v1"',
         'registration.get("schema") != "shared_social.registered_budget_evaluation.v1"'),
        ('or result["completed_updates"] != 2000 or result["actual_training_transitions"] != 16384000',
         'or result["completed_updates"] != expected_updates or result["actual_training_transitions"] != expected_transitions'),
        ('Require the complete preregistered fresh S1 terminal run',
         'Require the complete preregistered terminal training budget'),
    )
    derived = source
    for old, new in replacements:
        if derived.count(old) != 1:
            raise ValueError("Ambiguous registered-budget evaluation admission")
        derived = derived.replace(old, new, 1)
    inverse = derived
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous registered-budget evaluation inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != source:
        raise ValueError("Budget admission did not restore the entire original pipeline")
    compile(derived, str(PARENT) + ":registered_budget", "exec")
    return derived, {"complete_source_inverse": True, "replacement_count": len(replacements),
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "tracking_style_navigation_and_child_exit_math_changed": False}


def prepare():
    parent = record(PARENT)
    if parent["sha256"] != PARENT_SHA:
        raise ValueError("Original fixed2000 post-training pipeline changed")
    source, proof = derive(PARENT.read_text())
    module = ModuleType("_registered_budget_evaluation")
    module.__file__ = str(PARENT)
    module.__dict__["_registered_budget"] = registered_budget
    exec(compile(source, str(PARENT) + ":registered_budget", "exec"), module.__dict__)
    module.SOURCE_PATHS.append("scripts/run_shared_social_budget_evaluation.py")
    return module, {"parent": parent, "entry": record(__file__), "derivation": proof}


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--check-sources", action="store_true")
    args, forwarded = selector.parse_known_args(argv)
    module, binding = prepare()
    if args.check_sources:
        print(json.dumps({"status": "source_checks_passed", **binding,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    code = module.main(forwarded)
    if record(PARENT) != binding["parent"] or record(__file__) != binding["entry"]:
        raise ValueError("Registered-budget entry changed during evaluation")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
