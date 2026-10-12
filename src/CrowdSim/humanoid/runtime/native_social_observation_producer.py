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

PARENT = ROOT / "resources/runtime_sources/style_improvement_20261004/social_native_evaluation_staging_v1/native_producer.py"
PARENT_SHA = "e76deb05e0cc07af97c18a84307d454c92817538ddd2110fe8a77f8076459b4b"


def derive_native_origin(source):
    old = "        derived, instance.derivation = _derive(source.decode())\n"
    new = ("        native_source, native_proof = _derive(source.decode())\n"
           "        derived, origin_proof = _migration_origin_deriver(native_source)\n"
           "        instance.derivation = {\"native_lifecycle\": native_proof, \"scripted_origin\": origin_proof}\n")
    if source.count(old) != 1:
        raise ValueError("Ambiguous native producer origin seam")
    core_old = '        exec(compile(derived, backend.__file__, "exec"), backend.__dict__)\n'
    core_new = core_old + '        _migration_core_pins(backend)\n'
    if source.count(core_old) != 1:
        raise ValueError("Ambiguous native observation core seam")
    derived = source.replace(old, new, 1).replace(core_old, core_new, 1)
    inverse = derived.replace(core_new, core_old, 1).replace(new, old, 1)
    if derived.count(new) != 1 or derived.count(core_new) != 1 or inverse != source:
        raise ValueError("Native origin seam did not restore complete source")
    def untouched(value):
        tree = ast.parse(value)
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name != "from_world"}
    if untouched(derived) != untouched(source):
        raise ValueError("Native origin migration changed a query/observation/reset method")
    compile(derived, str(PARENT) + ":scripted_origin", "exec")
    return derived, {"replacement_count": 2, "exact_inverse": True, "other_methods_AST_equal": True,
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
from .evaluation_source_migration import migrate_observation_pins
_parent._migration_core_pins = migrate_observation_pins
exec(compile(_source, str(PARENT) + ":scripted_origin", "exec"), _parent.__dict__)
# The sealed producer retains its exact observation methods. Its owned module
# selects explicit current scene providers; the archived module and evidence
# are unchanged, and this version cannot certify a historical native run.
_historical_pins = dict(_parent.PINS)
_current_pins = {
    **_historical_pins,
    "native_deployment": (_historical_pins["native_deployment"][0], 5610,
        "a3a8097e5d63b03e5f96f24e6246c333b2103f3628c6209139e2aff2c564cd96"),
    "provider": (_historical_pins["provider"][0], 53151,
        "68bd2171ad9707c3f280c374d73538b8c532764a42e1f90335c6375090595d28"),
    "grid": (_historical_pins["grid"][0], 41778,
        "b5563bd5a509cdb02ae33e57ad34ab7a0524825ad16e481ef631531d59308165"),
    "live": (_historical_pins["live"][0], 13565,
        "4c7f5384658778623b96ceeca0c52cda938d57e644b8084f4461e21ebac42bca"),
}
_parent.PINS = {role: (str((ROOT / path).resolve()), size, sha)
                for role, (path, size, sha) in _current_pins.items()}
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
            "source_version": "native_observations.current_scene_sources.v2",
            "historical_scene_pins": dict(_historical_pins),
            "actual_native_TorchScript_admission_verified": False}
