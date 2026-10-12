"""Native SFM planner-only substitution of the existing shared AMP driver.

This is source preparation, not a standalone SDK launcher. The native entry
must supply its own admitted scene, frozen AMP reader and terminal lifecycle.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFM_DERIVER = ROOT / "scripts/evaluate_shared_social_sfm.py"
SFM_DERIVER_SHA = "df5111e8c7d1cb39f821cc679f0251123ff1d9e20afe6752aa027117dd5ba239"


def derive_native_sfm_driver(original, base_derive):
    from scripts.evaluate_shared_social_sfm import derive_driver
    data = SFM_DERIVER.read_bytes()
    if hashlib.sha256(data).hexdigest() != SFM_DERIVER_SHA:
        raise ValueError("Same-AMP SFM driver derivation changed")
    source, base_proof = derive_driver(original, base_derive)
    before = source
    replacements = [
        ("    from CrowdSim.humanoid.trajectory.social_force_runtime import SocialForceSharedPlanner\n",
         "    from CrowdSim.humanoid.trajectory.native_social_force_runtime import NativeSocialForcePlanner\n"),
        ("        planner = SocialForceSharedPlanner(world=world, scene_source=args.scene_config, seed=args.seed)\n",
         "        planner = NativeSocialForcePlanner(world=world, scene_source=args.scene_config, seed=args.seed)\n"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Ambiguous native SFM planner seam")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous native SFM planner inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("Native planner seams did not restore exact complete shared SFM driver")
    def helpers(value):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(value).body
                if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if helpers(source) != helpers(before):
        raise ValueError("Native SFM changed a shared control/reference/capture helper")
    compile(source, "native_shared_SFM_AMP_derived_driver", "exec")
    return source, {"base_same_AMP_SFM_derivation": base_proof, "native_planner_replacements": 2,
                    "exact_inverse_to_shared_SFM_driver": True, "shared_helpers_AST_equal": True,
                    "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "shared_control_reference_measurement_math_changed": False,
                    "physical_execution_verified": False}
