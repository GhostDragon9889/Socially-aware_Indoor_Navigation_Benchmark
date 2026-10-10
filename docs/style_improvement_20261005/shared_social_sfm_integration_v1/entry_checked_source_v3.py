#!/usr/bin/env python3
"""Same-AMP N4 fixture SFM arm; actual SDK smoke must follow active S2 gates.

The accepted Diffusion checkpoint is an unused paired-resource binding. No
Diffusion network is instantiated in this arm. Original control, causal
reference, physical captures, actor admission and terminal lifecycle remain.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANONICAL = ROOT / "scripts/evaluate_shared_social_navigation.py"
CANONICAL_SHA = "b556de1a9c9c1a6b5984e1484c9a67f490f549f4346649746a84dd7ab9726558"


def derive_driver(original, base_derive):
    source, base_proof = base_derive(original)
    before = source
    replacements = [
        ("    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner\n",
         "    from CrowdSim.humanoid.trajectory.social_force_runtime import SocialForceSharedPlanner\n"),
        ('        planner = (PaperDiffusionPlanner(checkpoint_path=args.diffusion_checkpoint, device=world.device,\n'
         '                                        compile_forward=args.compile_diffusion_forward) if curriculum is None else None)\n',
         '        if curriculum is not None or not args.evaluate_steps or not args.resume or args.compile_diffusion_forward:\n'
         '            raise ValueError("SFM arm requires uncompiled pure shared-actor evaluation")\n'
         '        planner = SocialForceSharedPlanner(world=world, scene_source=args.scene_config, seed=args.seed)\n'
         '        report["high_level_planner"] = planner.describe()\n'),
        ('                      "diffusion_checkpoint": str(Path(args.diffusion_checkpoint).resolve()),\n'
         '                      "diffusion_sha256": file_sha256(args.diffusion_checkpoint),\n',
         '                      "unused_paired_diffusion_checkpoint": str(Path(args.diffusion_checkpoint).resolve()),\n'
         '                      "unused_paired_diffusion_sha256": file_sha256(args.diffusion_checkpoint),\n'
         '                      "high_level_planner": "public_sfm_joint_future",\n'),
        ('        provenance["diffusion_used_for_training"] = curriculum is None\n',
         '        provenance["diffusion_used_for_training"] = False\n'),
        ('        report["trajectory_source"] = args.trajectory_source\n',
         '        report["trajectory_source"] = "public_sfm_joint_future"\n'),
        ('        report["diffusion_guidance"] = guidance_contract\n',
         '        guidance_contract = {"mode": "not_applied_SFM", "author_configuration": False,\n'
         '            "joint_replanning": "entire shared scene whenever any actor is due",\n'
         '            "diffusion_objective_weights_applied": False}\n'
         '        report["diffusion_guidance"] = guidance_contract\n'),
    ]
    start = source.index('        report["runtime_execution"] = {')
    end = source.index('        if args.trajectory_source == "amass_root":\n', start)
    replacements.append((source[start:end], '''        report["runtime_execution"] = {"torch": str(torch.__version__), "physics_only": args.physics_only,
            "compiled_forward": False, "diffusion_steps": 0, "precision": "CPU_float64_predict_FP32_output",
            "consumed_planner_output": "raw51 XY only; common causal reference follows",
            "diffusion_used_for_rollouts": False, "diffusion_samples_per_actor": 0,
            "sample_filtering": "none; one deterministic joint SFM future; invalid predictions fail with partial captures"}
        report["runtime_execution"]["numerics"] = report["runtime_numerics"]
        report["method"] = "Public SFM joint future + frozen shared Social AMP"
        report["navigation_guidance"] = {"mode": "common_static_A_star_5s_endpoint",
            "waypoint_times_s": [float(i/2) for i in range(1, 11)], "SFM_goal": "last registered static A* waypoint",
            "constant_goal_over_predicted_horizon": True, "diffusion_goal_weights_applied": False,
            "author_configuration": False}
'''))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Ambiguous SFM-only physical driver substitution")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous SFM physical inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("SFM edits did not restore entire existing Social physical driver")
    def helpers(value):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(value).body
                if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if helpers(source) != helpers(before):
        raise ValueError("SFM changed a shared control/capture helper")
    compile(source, "shared_social_sfm_derived_driver", "exec")
    return source, {"base_social_derivation": base_proof, "SFM_replacement_count": len(replacements),
                    "SFM_inverse_complete_bytes_equal": True, "shared_helper_AST_equal": True,
                    "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "shared_control_reference_actor_measurement_changed": False,
                    "physical_execution_verified": False}


def main(argv=None):
    # Isolated module globals; no mutation of the imported canonical evaluator.
    from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
    from scripts import evaluate_shared_social_sfm as current_source
    actual = record(CANONICAL)
    if actual["sha256"] != CANONICAL_SHA:
        raise ValueError("Canonical paired evaluator changed")
    namespace = ModuleType("_isolated_same_amp_SFM_entry")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(CANONICAL.read_bytes(), str(CANONICAL), "exec"), namespace.__dict__)
    original_entry_deriver = namespace.derive_entry

    def entry_deriver(parent):
        source, base_proof = original_entry_deriver(parent)
        before = source
        replacements = [
            ("from __future__ import annotations\n", "from __future__ import annotations\n"
             "from scripts.evaluate_shared_social_sfm import derive_driver as _sfm_derive\n"),
            ("def _derive(source):\n", "def _base_derive(source):\n"),
            ("def main():\n", "def _derive(source):\n    return _sfm_derive(source, _base_derive)\n\n\ndef main():\n"),
        ]
        for old, new in replacements:
            if source.count(old) != 1:
                raise ValueError("Expected one SFM entry hook site")
            source = source.replace(old, new, 1)
        inverse = source
        for old, new in reversed(replacements):
            if inverse.count(new) != 1:
                raise ValueError("Ambiguous SFM entry inverse")
            inverse = inverse.replace(new, old, 1)
        if inverse != before:
            raise ValueError("SFM entry hook did not restore exact base entry")
        proof = {"base_entry_derivation": base_proof, "canonical_paired_entry": actual,
                 "SFM_entry_hook_inverse": True, "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
                 "SFM_entry": record(Path(__file__)), "SFM_driver_hook": record(current_source.__file__)}
        compile(source, "same_amp_SFM_entry", "exec")
        return source, proof

    namespace.derive_entry = entry_deriver
    bound = [record(ROOT / name) for name in ("scripts/evaluate_shared_social_sfm.py",
        "CrowdSim/humanoid/trajectory/social_force_runtime.py", "CrowdSim/humanoid/trajectory/social_force.py",
        "CrowdSim/world/sfm.py", "configs/diffusion_amp/shared_obstacle_timed_validation.yaml")]
    result = namespace.main(argv)
    if any(record(item["path"]) != item for item in bound) or record(CANONICAL) != actual:
        raise ValueError("SFM source/fixture changed during actual invocation")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
