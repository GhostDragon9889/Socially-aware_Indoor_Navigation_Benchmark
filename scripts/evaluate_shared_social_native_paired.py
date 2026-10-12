#!/usr/bin/env python3
"""Native N3..15 DP/SFM entry with identical pre-planner body/map/RNG capture.

Delegates scene, frozen AMP, Pop-Art and terminal admission to the prepared
native entry. CPU preflight is not evidence of live native SDK execution.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_paired_initial as initial  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import evaluate_shared_social_native as native_entry  # noqa: E402

PARENT = ROOT / "scripts/evaluate_shared_social_native.py"
PARENT_SHA = "dcc6033320430b381a9843a95e9f34005fbf54748a813d798d3cc228ebbf0f11"


def derive_driver(original, base_derive, *, planner):
    """Hook an already derived native arm; do not apply SFM derivation twice."""
    if planner not in {"diffusion", "sfm"}:
        raise ValueError("Native initial capture requires one explicit DP/SFM arm")
    source, base_proof = base_derive(original)
    before = source
    marker = ("        planner = (PaperDiffusionPlanner(checkpoint_path=args.diffusion_checkpoint, device=world.device,\n"
              if planner == "diffusion" else
              "        if curriculum is not None or not args.evaluate_steps or not args.resume or args.compile_diffusion_forward:\n")
    changes = [
        ("from __future__ import annotations\n", "from __future__ import annotations\n"
         "from CrowdSim.humanoid.analysis.native_paired_initial import capture_initial_condition as _paired_initial\n"),
        (marker, '        report["paired_initial_condition"] = _paired_initial(world, ground_capture, output, report, args, provenance)\n' + marker),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Ambiguous native before-planner capture seam")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(changes):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous native before-planner capture inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("Native capture hooks did not restore the complete derived arm")
    def helpers(value):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(value).body
                if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if helpers(source) != helpers(before):
        raise ValueError("Native capture changed an existing control/reference/measurement helper")
    compile(source, "native_same_AMP_before_planner_driver", "exec")
    return source, {"base_native_driver_derivation": base_proof, "planner": planner,
                    "capture_hook_replacements": len(changes), "complete_inverse": True,
                    "existing_helpers_AST_equal": True, "existing_control_or_metric_math_changed": False,
                    "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "actual_native_SDK_execution_verified": False}


def prepare_entry(*, actors, planner):
    if (Path(native_entry.__file__).resolve() != PARENT or record(PARENT)["sha256"] != PARENT_SHA):
        raise ValueError("Prepared native entry origin/source changed")
    sources = initial.source_bindings()
    namespace, binding = native_entry.prepare_entry(actors=actors, planner=planner)
    base_derive = namespace._derive
    namespace._derive = lambda value: derive_driver(value, base_derive, planner=planner)
    binding["unpaired_entry"] = binding["entry"]
    binding["entry"] = record(__file__)
    binding["native_initial_capture"] = {"sources": sources, "actors": actors,
        "schema": initial.SCHEMA, "capture_kind": initial.LIVE_KIND,
        "same_hook_before_either_planner": True, "RNG_reseed_or_restore_in_hook": False,
        "extra_physics_steps_in_hook": 0, "actual_native_SDK_capture_verified": False}
    namespace._migration_entry = str(Path(__file__).resolve())
    return namespace, binding


def main(argv=None):
    """Reuse the parent CLI and its owned lifecycle in an isolated namespace."""
    parent = record(PARENT)
    if parent["sha256"] != PARENT_SHA:
        raise ValueError("Prepared native entry changed")
    sources, entry = initial.source_bindings(), record(__file__)
    namespace = ModuleType("_owned_native_paired_entry_CLI")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(PARENT.read_bytes(), str(PARENT) + ":native_paired", "exec"), namespace.__dict__)
    namespace.prepare_entry = prepare_entry
    try:
        return namespace.main(argv)
    finally:
        if record(PARENT) != parent or record(__file__) != entry or initial.source_bindings() != sources:
            raise ValueError("Native paired entry/source changed during invocation; partial files retained")


if __name__ == "__main__":
    raise SystemExit(main())
