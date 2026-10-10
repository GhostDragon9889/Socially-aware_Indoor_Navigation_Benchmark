#!/usr/bin/env python3
"""Fresh N4 Diffusion or SFM evaluation with the same pre-planner snapshot.

Uses unchanged existing actor/control/reference/measurement/cleanup paths.
Actual GPU smoke and full paired runs must follow the active S2 owner's gates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANONICAL = ROOT / "scripts/evaluate_shared_social_navigation.py"
CANONICAL_SHA = "b556de1a9c9c1a6b5984e1484c9a67f490f549f4346649746a84dd7ab9726558"
SFM_ENTRY_SHA = "9a375da2698327b28df480e6128d8741d92800bde596f16ef7543a21d5c658de"


def derive_driver(original, base_derive, *, planner):
    """Add the identical capture hook after actor admission in each arm."""
    if planner not in {"diffusion", "sfm"}:
        raise ValueError("Only the explicit Diffusion and public SFM arms are supported")
    if planner == "sfm":
        from scripts.evaluate_shared_social_sfm import derive_driver as sfm_derive
        source, base_proof = sfm_derive(original, base_derive)
        marker = "        if curriculum is not None or not args.evaluate_steps or not args.resume or args.compile_diffusion_forward:\n"
    else:
        source, base_proof = base_derive(original)
        marker = "        planner = (PaperDiffusionPlanner(checkpoint_path=args.diffusion_checkpoint, device=world.device,\n"
    before = source
    replacements = [
        ("from __future__ import annotations\n", "from __future__ import annotations\n"
         "from CrowdSim.humanoid.analysis.paired_initial import capture_initial_condition as _paired_initial\n"),
        (marker, '        report["paired_initial_condition"] = _paired_initial(world, ground_capture, output, report, args, provenance)\n' + marker),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Ambiguous before-planner paired snapshot hook")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous paired snapshot inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("Paired hooks failed complete original byte restoration")
    def helpers(text):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if helpers(source) != helpers(before):
        raise ValueError("Paired snapshot changed an existing physical helper")
    compile(source, "same_amp_paired_initial_driver", "exec")
    return source, {"base_driver_derivation": base_proof, "high_level_arm": planner,
                    "paired_capture_replacements": len(replacements), "complete_inverse": True,
                    "existing_helpers_AST_equal": True, "existing_control_or_metric_math_changed": False,
                    "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "actual_physical_execution_verified": False}


def derive_entry(parent, base_derive, *, planner):
    source, base_proof = base_derive(parent)
    before = source
    replacements = [
        ("from __future__ import annotations\n", "from __future__ import annotations\n"
         "from scripts.evaluate_shared_social_high_level import derive_driver as _paired_derive\n"),
        ("def _derive(source):\n", "def _base_derive(source):\n"),
        ("def main():\n", "def _derive(source):\n"
         f"    return _paired_derive(source, _base_derive, planner={planner!r})\n\n\ndef main():\n"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Ambiguous paired entry hook")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous paired entry inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("Paired entry did not restore exact canonical migrated parent")
    compile(source, "same_amp_paired_initial_entry", "exec")
    return source, {"base_entry_derivation": base_proof, "paired_entry_inverse": True,
                    "high_level_arm": planner, "entry_replacements": len(replacements),
                    "derived_sha256": hashlib.sha256(source.encode()).hexdigest()}


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--planner", choices=("diffusion", "sfm"), required=True)
    selected, forwarded = selector.parse_known_args(argv)
    from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
    from CrowdSim.humanoid.analysis.paired_initial import FOOT_CAPTURE_SHA
    actual = record(CANONICAL)
    sfm = record(ROOT / "scripts/evaluate_shared_social_sfm.py")
    if (actual["sha256"] != CANONICAL_SHA or sfm["sha256"] != SFM_ENTRY_SHA
            or record(ROOT / "CrowdSim/humanoid/foot_capture.py")["sha256"] != FOOT_CAPTURE_SHA):
        raise ValueError("Existing canonical or SFM entry changed")
    namespace = ModuleType("_isolated_same_AMP_paired_initial_entry")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(CANONICAL.read_bytes(), str(CANONICAL), "exec"), namespace.__dict__)
    base = namespace.derive_entry
    namespace.derive_entry = lambda parent: derive_entry(parent, base, planner=selected.planner)
    bound = [record(ROOT / name) for name in (
        "scripts/evaluate_shared_social_high_level.py", "CrowdSim/humanoid/analysis/paired_initial.py",
        "CrowdSim/humanoid/foot_capture.py", "CrowdSim/humanoid/trajectory/social_force_runtime.py",
        "CrowdSim/humanoid/trajectory/social_force.py", "CrowdSim/world/sfm.py",
        "configs/diffusion_amp/shared_obstacle_timed_validation.yaml")]
    result = namespace.main(forwarded)
    if (any(record(item["path"]) != item for item in bound)
            or record(CANONICAL) != actual or record(sfm["path"]) != sfm):
        raise ValueError("Paired source/fixture changed during invocation")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
