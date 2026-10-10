"""Migrate the sealed native observation facade's scripted heading admission.

Native queries, observations, clocks and resets retain the original methods.
Only the existing scripted-origin guard is composed into the owned core module.
Import is stdlib only. Native SDK construction and quality remain separate gates.
"""
from __future__ import annotations

import ast
import hashlib
from types import ModuleType

from .social_evaluation_lifecycle import ROOT, record
from . import social_observation_producer

PARENT = ROOT / "docs/style_improvement_20261004/social_native_evaluation_staging_v1/native_producer.py"
PARENT_SHA = "7d2a51563f59811a6c6d16d82f880e99accaffbfecc3074e44c48bad64666621"


def derive_native_origin(source):
    old = "        derived, instance.derivation = _derive(source.decode())\n"
    new = ("        native_source, native_proof = _derive(source.decode())\n"
           "        derived, origin_proof = _migration_origin_deriver(native_source)\n"
           "        instance.derivation = {\"native_lifecycle\": native_proof, \"scripted_origin\": origin_proof}\n")
    if source.count(old) != 1:
        raise ValueError("Ambiguous native producer origin seam")
    derived = source.replace(old, new, 1)
    if derived.count(new) != 1 or derived.replace(new, old, 1) != source:
        raise ValueError("Native origin seam did not restore complete source")
    def untouched(value):
        tree = ast.parse(value)
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name != "from_world"}
    if untouched(derived) != untouched(source):
        raise ValueError("Native origin migration changed a query/observation/reset method")
    compile(derived, str(PARENT) + ":scripted_origin", "exec")
    return derived, {"replacement_count": 1, "exact_inverse": True, "other_methods_AST_equal": True,
                     "parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "observation_query_clock_reset_math_changed": False, "SDK_executed": False}


_parent_identity = record(PARENT)
if _parent_identity["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
    raise ValueError("Sealed native producer changed")
_identity = record(__file__)
_origin_identity = record(social_observation_producer.__file__)
_source, _derivation = derive_native_origin(PARENT.read_text())
_parent = ModuleType("_owned_native_social_observations_scripted_origin")
_parent.__file__ = str(PARENT)
_parent._migration_origin_deriver = social_observation_producer.derive_origin_guard
exec(compile(_source, str(PARENT) + ":scripted_origin", "exec"), _parent.__dict__)
_original_sources = _parent._sources


def _sources():
    if record(__file__) != _identity or record(social_observation_producer.__file__) != _origin_identity:
        raise ValueError("Native scripted-origin migration changed after import")
    return {**_original_sources(), "native_origin_adapter": dict(_identity),
            "canonical_scripted_origin_adapter": dict(_origin_identity)}


_parent._sources = _sources
RawSocialNativeEvaluationObservations = _parent.RawSocialNativeEvaluationObservations


def describe_migration():
    _sources()
    return {"implementation": dict(_identity), "parent": dict(_parent_identity),
            "canonical_scripted_origin": dict(_origin_identity), "derivation": dict(_derivation),
            "actual_native_TorchScript_admission_verified": False}
