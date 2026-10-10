#!/usr/bin/env python3
"""CPU audit of actual recovery segment and retained checkpoint training counts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_shared_social_popart as popart  # noqa: E402
from CrowdSim.humanoid.runtime import social_training_recovery as recovery  # noqa: E402


def derive(source):
    source, parent_proof = popart.derive_auditor(source)
    edits = (
        ('    assert result["training_update_returned_count"] == updates\n',
         '    assert result["training_update_returned_count"] == updates - _base\n'
         '    assert result["recovery"]["initial_update_count"] == _base\n'
         '    assert result["recovery"]["physical_world_restarted"] is True\n'
         '    assert result["recovery"]["physical_world_curriculum_history_restored"] is False\n'
         '    assert result["recovery"]["global_training_rng_restored_after_setup"] is True\n'),
        ('    assert state["runner_state"] is None and state["physical_runner_state_included"] is False\n',
         '    assert state["runner_state"] is None and state["physical_runner_state_included"] is False\n'
         '    assert state["provenance"]["training_recovery"] == result["recovery"]\n'
         '    assert all(float(item["step"]) == updates * 12 for item in state["optimizer"]["state"].values())\n'
         '    assert all(float(item["step"]) == updates * 2 for item in state["discriminator_optimizer"]["state"].values())\n'),
        ('    assert len(metric_rows) == updates\n', '    assert len(metric_rows) == updates - _base\n'),
        ('list(range(1, updates + 1))', 'list(range(_base + 1, updates + 1))'),
        ('updates * controls * 4\n', '(updates - _base) * controls * 4\n'),
        ('    assert report["CUDA_initialized"] is False\n',
         '    assert report["CUDA_initialized"] is False\n'
         '    prefix_path = run / "retained_metrics_prefix.jsonl"\n'
         '    metadata, original_prefix = _read_retained_metrics()\n'
         '    assert prefix_path.read_bytes() == original_prefix\n'
         '    assert metadata == result["recovery"]["binding"]["retained_metrics"]\n'
         '    assert metric_rows[-1]["recovery_segment_update"] == updates - _base\n'
         '    assert metric_rows[-1]["completed_physical_controls"] == (updates - _base) * controls\n'
         '    assert metric_rows[-1]["cumulative_expert_windows_drawn"] == updates * experts\n'
         '    report["schema"] = "shared_social.completed_optimizer_recovery_CPU_audit.v1"\n'
         '    report["retained_updates"] = _base\n'
         '    report["actual_recovery_segment_updates"] = updates - _base\n'
         '    report["actual_recovery_segment_transitions"] = (updates - _base) * actors * controls\n'
         '    report["original_continuous_S3_run_completed"] = False\n'
         '    report["recovery"] = result["recovery"]\n'),
    )
    derived = source
    for old, new in edits:
        if derived.count(old) != 1:
            raise ValueError("Ambiguous recovery audit migration")
        derived = derived.replace(old, new, 1)
    inverse = derived
    for old, new in reversed(edits):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous recovery audit inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != source:
        raise ValueError("Recovery audit did not preserve original finite/shape/history checks")
    return derived, {"parent": parent_proof, "exact_inverse": True, "replacement_count": len(edits),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if popart.record(popart.PARENT)["sha256"] != popart.PARENT_SHA:
        raise ValueError("Original CPU auditor changed")
    source, proof = derive(popart.PARENT.read_text())
    binding = {"parent": popart.record(popart.PARENT), "entry": popart.record(__file__), "derivation": proof}
    module = ModuleType("_optimizer_recovery_CPU_auditor")
    module.__file__ = __file__
    module.__dict__.update(_popart_binding=popart.social_popart_training.describe(),
                          _describe_popart=popart.social_popart_training.describe,
                          _auditor_binding=binding, _base=recovery.BASE_UPDATES,
                          _read_retained_metrics=recovery.read_retained_metrics)
    exec(compile(source, __file__ + ":recovery", "exec"), module.__dict__)
    saved = sys.argv
    try:
        sys.argv = [__file__, str(args.run_directory), "--output", str(args.output)]
        module.main()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
