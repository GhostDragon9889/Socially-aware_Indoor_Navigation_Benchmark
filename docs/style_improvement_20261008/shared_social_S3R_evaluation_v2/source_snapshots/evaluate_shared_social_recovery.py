#!/usr/bin/env python3
"""Evaluate S3-R with the original tracking/style/navigation mathematics.

Caller-owned module namespaces admit explicit recovery metadata and route child
entries through this adapter. Original files, frozen references, physical code,
checkpoint reader, child exit checks and cleanup gates are preserved.
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
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import run_shared_social_budget_evaluation as budget  # noqa: E402

PARENTS = {
    "actor_cpu": ("scripts/audit_shared_social_inference.py", "5f01bb007bfe28802ec282e7ad286809b605ef8a1e6102263a839d23dd28ce1b"),
    "tracking": ("scripts/evaluate_shared_social_tracking.py", "ff866433ce91feb2a006a137ed437ae7bab92b8b6b0c9daaf09f244521b3767c"),
    "tracking_style": ("scripts/run_shared_social_evaluation.py", "ae2e78bca664eaac6ffe023159ea4d714ec6cc892065218506bbc19ecd1cebf2"),
    "navigation": ("scripts/evaluate_shared_social_navigation.py", "b556de1a9c9c1a6b5984e1484c9a67f490f549f4346649746a84dd7ab9726558"),
    "post": ("scripts/run_shared_social_post_training.py", "2890d628361fbe1c548d822a82bbcbedde215f6395db44d7327a1903a7199cf8"),
}
EXTRA_SOURCES = ["scripts/evaluate_shared_social_recovery.py",
                 "CrowdSim/humanoid/application/recovery_evaluation_admission.py"]
BUDGET_SHA = "6fa571eaa503f1e9cd41965572fb3e2874fe542cece46d86a2df50f7557b4512"


def derive(source, stage):
    parent_proof = None
    if stage == "post":
        source, parent_proof = budget.derive(source)
    edits = []
    if stage == "actor_cpu":
        edits.append(('result.get("canonical_entry", {}).get("value_normalization") != "popart"',
                      'admission["value_normalization"] != "popart"', 1))
    elif stage == "tracking_style":
        edits.extend([
            ('    from scripts.evaluate_shared_social_tracking import admit_training\n',
             '    admit_training = _recovery_admit_training\n', 1),
            ('[sys.executable, str(tracking), "--execution-receipt",',
             '[sys.executable, str(_recovery_entry), "--stage", "tracking", "--execution-receipt",', 2),
            ('    source_records = [identity(path) for path in sources]\n',
             '    sources.extend(ROOT / path for path in _recovery_sources)\n'
             '    source_records = [identity(path) for path in sources]\n', 1),
            ('"shared2000=" + str(output / "full22")', '"S3_R12500=" + str(output / "full22")', 1),
        ])
    elif stage == "post":
        edits.extend([
            ('[python, str(ROOT / SOURCE_PATHS[1]), "--execution-receipt",',
             '[python, str(_recovery_entry), "--stage", "actor_cpu", "--execution-receipt",', 1),
            ('[python, str(ROOT / SOURCE_PATHS[2]), "--execution-receipt",',
             '[python, str(_recovery_entry), "--stage", "tracking_style", "--execution-receipt",', 1),
            ('navigation = [python, str(ROOT / "scripts/evaluate_shared_social_navigation.py"),',
             'navigation = [python, str(_recovery_entry), "--stage", "navigation",', 1),
        ])
    elif stage not in {"tracking", "navigation"}:
        raise ValueError("Unknown recovery evaluation stage")
    derived = source
    for old, new, count in edits:
        if derived.count(old) != count:
            raise ValueError("Ambiguous recovery evaluation routing: " + stage)
        derived = derived.replace(old, new)
    inverse = derived
    for old, new, count in reversed(edits):
        if inverse.count(new) != count:
            raise ValueError("Ambiguous recovery evaluation inverse: " + stage)
        inverse = inverse.replace(new, old)
    if inverse != source:
        raise ValueError("Recovery routing did not restore the complete original evaluation")
    compile(derived, "<recovery_evaluation_" + stage + ">", "exec")
    return derived, {"complete_source_inverse": True, "replacement_count": sum(row[2] for row in edits),
        "registered_budget_derivation": parent_proof,
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "tracking_style_navigation_metric_or_physics_math_changed": False,
        "training_metadata_admission_and_child_entry_routing_changed": True}


def prepare(stage):
    relative, sha = PARENTS[stage]
    path = ROOT / relative
    parent = record(path)
    if parent["sha256"] != sha:
        raise ValueError("Original recovery evaluation parent changed: " + stage)
    if stage == "post" and record(budget.__file__)["sha256"] != BUDGET_SHA:
        raise ValueError("Registered-budget derivation source changed")
    source, proof = derive(path.read_text(), stage)
    binding = {"stage": stage, "parent": parent, "entry": record(__file__),
               "admission_source": record(admission.__file__), "derivation": proof}

    def admit_receipt(receipt_path):
        checkpoint, digest, accepted = admission.admit_training(receipt_path)
        return checkpoint, digest, {**accepted, "recovery_evaluation_entry": binding}

    module = ModuleType("_recovery_evaluation_" + stage)
    module.__file__ = str(path)
    module.__dict__.update(_recovery_entry=Path(__file__).resolve(), _recovery_sources=EXTRA_SOURCES,
                          _recovery_admit_training=admit_receipt, _registered_budget=budget.registered_budget)
    exec(compile(source, str(path) + ":recovery_admission", "exec"), module.__dict__)
    module.admit_training = admit_receipt
    if stage == "post":
        module.SOURCE_PATHS.extend([*EXTRA_SOURCES, "scripts/run_shared_social_budget_evaluation.py"])
    return module, binding


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stage", required=True, choices=PARENTS)
    parser.add_argument("--check-adapter", action="store_true")
    options, forwarded = parser.parse_known_args(argv)
    module, binding = prepare(options.stage)
    if options.check_adapter:
        print(json.dumps({"status": "source_checks_passed", **binding,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    saved = sys.argv
    try:
        sys.argv = [module.__file__, *forwarded]
        code = module.main()
    finally:
        sys.argv = saved
    if (record(binding["parent"]["path"]) != binding["parent"]
            or record(__file__) != binding["entry"]
            or record(admission.__file__) != binding["admission_source"]):
        raise ValueError("Recovery evaluation entry changed during execution")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
