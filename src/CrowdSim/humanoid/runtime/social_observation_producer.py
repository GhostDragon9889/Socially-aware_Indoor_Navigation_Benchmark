"""Source-bound social observation producer with TorchScript origin admission.

Geometry, clocks, reset notifications and observation calculations retain the
sealed implementation. The only functional change admits its original scripted
heading function by exact loaded object, qualified name and pinned source.
Import is stdlib only, before the caller starts Isaac Sim and imports Torch.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import ModuleType

from .social_evaluation_lifecycle import ROOT, record

PARENT = ROOT / "docs/style_improvement_20261004/social_evaluation_observation_staging_v1/observation_producer.py"
PARENT_SHA = "2379d65876775173fbdd111d846fd847beb6b30109ecb7e04aa30f7a569064f7"


def derive_origin_guard(source):
    tree = ast.parse(source)
    original = ast.get_source_segment(source, next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_module"))
    old = "    module = sys.modules.get(value.__module__)\n"
    new = '''    if key == "rotations":
        torch = sys.modules.get("torch")
        module = sys.modules.get("protomotions.utils.rotations")
        if (torch is None or module is None
                or Path(getattr(module, "__file__", "")).resolve() != ROOT / PINS[key][0]
                or type(value) is not torch.jit.ScriptFunction
                or value is not getattr(module, "calc_heading_quat_inv", None)
                or value.qualified_name != "__torch__.protomotions.utils.rotations.calc_heading_quat_inv"):
            raise TypeError("Original source-bound scripted heading function required")
        identity = _record(ROOT / PINS[key][0])
        if identity["bytes"] != PINS[key][1] or identity["sha256"] != PINS[key][2]:
            raise TypeError("Scripted heading source changed")
        return module
    module = sys.modules.get(value.__module__)
'''
    if original.count(old) != 1 or source.count(original) != 1:
        raise ValueError("Ambiguous scripted-origin migration")
    replacement = original.replace(old, new, 1)
    derived = source.replace(original, replacement, 1)
    if derived.count(replacement) != 1 or derived.replace(replacement, original, 1) != source:
        raise ValueError("Scripted-origin migration is not exactly reversible")
    compile(derived, str(PARENT) + ":scripted_origin", "exec")
    return derived, {"parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(), "exact_inverse": True,
        "changed_function": "_module", "observation_math_changed": False}


_parent_identity = record(PARENT)
if _parent_identity["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
    raise ValueError("Sealed observation producer changed")
_identity = record(Path(__file__))
_source, _derivation = derive_origin_guard(PARENT.read_text())
_parent = ModuleType("_shared_eval_original_producer_scripted_origin")
_parent.__file__ = str(PARENT)
exec(compile(_source, str(PARENT) + ":scripted_origin", "exec"), _parent.__dict__)
from .evaluation_source_migration import migrate_observation_pins
migrate_observation_pins(_parent)
_parent_sources = _parent._sources


def _sources():
    if record(Path(__file__)) != _identity:
        raise ValueError("Scripted-origin adapter changed after import")
    return {**_parent_sources(), "scripted_origin_adapter": dict(_identity)}


_parent._sources = _sources  # Only this entry-owned module namespace; no SDK or original module mutation.
RawSocialEvaluationObservations = _parent.RawSocialEvaluationObservations


def describe_migration():
    return {"implementation": dict(_identity), "parent": dict(_parent_identity),
            "derivation": dict(_derivation), "actual_TorchScript_admission_verified": False}
