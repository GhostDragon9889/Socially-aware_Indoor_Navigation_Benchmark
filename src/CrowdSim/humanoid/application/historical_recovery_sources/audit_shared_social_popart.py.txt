#!/usr/bin/env python3
"""CPU completed-run audit for the explicitly declared Pop-Art checkpoint family.

Keeps the existing weight/Adam/RMS/replay/rollout/count audit. Adds exact source
contract and the saved actual output-preservation exposure checks. No NN/SDK.
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
from CrowdSim.humanoid.runtime import social_popart_training  # noqa: E402

PARENT = ROOT / "scripts/audit_shared_social_training.py"
PARENT_SHA = "4facc6cb9bd960b7fb23ac29bd55035d829106f5548ce97ea3ff528c2e460fba"


def derive_auditor(source):
    replacements = (
        ('    assert state["format"] == "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"\n',
         '    assert state["format"] == _popart_binding["checkpoint_format"]\n'
         '    assert state["social_training_contract"]["value_normalization"] == _popart_binding\n'),
        ('    assert metric_rows[-1]["actual_normalizer_counts"] == actual_counts\n',
         '    assert metric_rows[-1]["actual_normalizer_counts"] == actual_counts\n'
         '    adjustment = metric_rows[-1]["value_normalization"]\n'
         '    assert adjustment["strategy"] == _popart_binding["strategy"]\n'
         '    assert adjustment["old_count"] == (updates - 1) * actors * controls\n'
         '    assert adjustment["new_count"] == transitions\n'
         '    assert adjustment["optimizer_state_modified"] is False\n'),
        ('    assert report["CUDA_initialized"] is False\n',
         '    assert report["CUDA_initialized"] is False\n'
         '    assert identity(Path(__file__)) == _auditor_binding["entry"]\n'
         '    assert _describe_popart() == _popart_binding\n'
         '    report["explicit_value_normalization"] = _popart_binding\n'
         '    report["actual_terminal_value_adjustment"] = adjustment\n'
         '    report["auditor_source_derivation"] = _auditor_binding\n'),
    )
    derived = source
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError("Ambiguous Pop-Art CPU audit migration")
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(replacements):
        if inverse.count(after) != 1:
            raise ValueError("Ambiguous Pop-Art audit inverse")
        inverse = inverse.replace(after, before, 1)
    if inverse != source:
        raise ValueError("Pop-Art CPU audit is not exactly reversible")
    compile(derived, str(PARENT) + ":popart", "exec")
    return derived, {"parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "replacement_count": len(replacements), "exact_inverse": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true")
    options, forwarded = parser.parse_known_args()
    parent = record(PARENT)
    if parent["sha256"] != PARENT_SHA:
        raise ValueError("Original CPU audit source changed")
    source, proof = derive_auditor(PARENT.read_text())
    binding = {"parent": parent, "entry": record(__file__), "derivation": proof}
    trainer = social_popart_training.describe()
    if options.check_sources:
        print(json.dumps({"status": "source_checks_passed", "Torch_imported": "torch" in sys.modules,
                          "SDK_started": False, "auditor": binding, "trainer": trainer}))
        return 0
    namespace = ModuleType("_explicit_social_popart_auditor")
    namespace.__file__ = __file__
    namespace.__dict__.update(_popart_binding=trainer, _describe_popart=social_popart_training.describe,
                              _auditor_binding=binding)
    exec(compile(source, str(PARENT) + ":popart", "exec"), namespace.__dict__)
    saved = sys.argv
    try:
        sys.argv = [__file__, *forwarded]
        namespace.main()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
