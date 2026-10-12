#!/usr/bin/env python3
"""Explicit-seed native paired evaluation, preserving the frozen seed7 entry.

Only CLI seed admission and its existing driver argument change. Original
capture/control/planner/measurement helpers and registered resources remain
unchanged. Versioned raw/style admission is required for this new entry.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_shared_social_native_paired as paired  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402

PAIRED = ROOT / "scripts/evaluate_shared_social_native_paired.py"
PAIRED_SHA = "1b77d62a561aef8f465e5a28dcb4e68e1d0625094c23022540f06f3ae66b2e9d"
CLI_PARENT = paired.PARENT
CLI_PARENT_SHA = paired.PARENT_SHA


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Native evaluation seed must be an explicit integer in [0,2**32)")


def source_bindings():
    parent, original_cli = record(PAIRED), record(CLI_PARENT)
    if (Path(paired.__file__).resolve() != PAIRED or parent["sha256"] != PAIRED_SHA
            or original_cli["sha256"] != CLI_PARENT_SHA):
        raise ValueError("Frozen native paired/CLI source changed")
    return {"seeded_entry": record(__file__), "seed7_paired_entry": parent,
            "seed7_CLI": original_cli, "native_initial_capture": paired.initial.source_bindings()}


def derive_seed_interface(source, changes, *, role):
    """Bounded main-only interface edits with complete source inverse."""
    original = source
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Ambiguous native seed seam: " + role)
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(changes):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous native seed inverse: " + role)
        inverse = inverse.replace(new, old, 1)
    def helpers(value):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(value).body
                if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if inverse != original or helpers(original) != helpers(source):
        raise ValueError("Native seed adapter changed another helper or failed complete inverse")
    compile(source, "native_seed_interface:" + role, "exec")
    return source, {"role": role, "literal_seams": len(changes), "complete_inverse": True,
        "unchanged_helpers_AST_equal": True, "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "control_planner_capture_or_metric_math_changed": False, "actual_native_SDK_execution_verified": False}


def seed_main_source(source):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return ast.get_source_segment(source, node) + "\n"


def prepare_entry(*, actors, planner, seed):
    validate_seed(seed)
    sources = source_bindings()
    namespace, binding = paired.prepare_entry(actors=actors, planner=planner)
    prepared_source = _prepared_source()
    if hashlib.sha256(prepared_source.encode()).hexdigest() != binding["prepared_entry_sha256"]:
        raise ValueError("Recovered prepared native source differs from the actual owned namespace")
    source = seed_main_source(prepared_source)
    changes = [
        ('    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)\n',
         '    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)\n'
         '    parser.add_argument("--seed", type=int, required=True)\n'),
        ('    args = parser.parse_args()\n',
         '    args = parser.parse_args()\n'
         '    if type(args.seed) is not int or args.seed != _native_registered_seed:\n'
         '        raise ValueError("Forwarded native seed differs from the registered entry")\n'),
        ('actors=args.actors, seed=7,', 'actors=args.actors, seed=args.seed,'),
    ]
    source, proof = derive_seed_interface(source, changes, role="owned_prepared_main")
    binding["seed7_paired_entry"] = binding["entry"]
    binding["entry"] = sources["seeded_entry"]
    binding["registered_native_seed"] = {"seed": seed, "sources": sources, "main_derivation": proof,
        "seed_forwarded_to_original_driver": True, "extra_reseed_in_capture_hook": False,
        "author_seed_protocol_recovered": False, "actual_native_SDK_execution_verified": False}
    namespace._migration_entry = str(Path(__file__).resolve())
    namespace._native_registered_seed = seed
    exec(compile(source, str(CLI_PARENT) + ":explicit_seed_main", "exec"), namespace.__dict__)
    return namespace, binding


def _prepared_source():
    """Reconstruct only the pinned prepared entry source; no runtime objects."""
    native = paired.native_entry
    if record(native.NATIVE_PARENT)["sha256"] != native.NATIVE_PARENT_SHA:
        raise ValueError("Frozen native staging source changed")
    staging = ModuleType("_owned_seed_native_staging")
    staging.__file__ = str(native.NATIVE_PARENT)
    exec(compile(native.NATIVE_PARENT.read_bytes(), str(native.NATIVE_PARENT), "exec"), staging.__dict__)
    source, _ = staging._derive_parent(native.canonical.PARENT.read_text())
    old = '        module = _load("_social_native_observations", sources["producer"])\n'
    new = '        module = _load("_social_navigation_observations", sources["producer"])\n'
    source, _ = derive_seed_interface(source, [(old, new)], role="original_producer_compatibility")
    source, _ = native.canonical.derive_entry(source)
    old = "    if args.preflight_only:\n"
    new = '    preparation["native_current_migration"] = _migration_binding\n' + old
    source, _ = derive_seed_interface(source, [(old, new)], role="original_preflight_binding")
    return source


def outer_CLI_source():
    source_bindings()
    source = CLI_PARENT.read_text()
    changes = [
        ('    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)\n',
         '    parser.add_argument("--planner", choices=("diffusion", "sfm"), required=True)\n'
         '    parser.add_argument("--seed", type=int, required=True)\n'),
        ('prepare_entry(actors=options.actors, planner=options.planner)',
         'prepare_entry(actors=options.actors, planner=options.planner, seed=options.seed)'),
        ('        "--actors", str(options.actors),',
         '        "--seed", str(options.seed), "--actors", str(options.actors),'),
    ]
    return derive_seed_interface(source, changes, role="outer_native_CLI")


def main(argv=None):
    sources = source_bindings()
    source, proof = outer_CLI_source()
    namespace = ModuleType("_owned_explicit_seed_native_CLI")
    namespace.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(CLI_PARENT) + ":explicit_seed_CLI", "exec"), namespace.__dict__)
    def prepare_with_CLI_proof(**options):
        prepared, binding = prepare_entry(**options)
        binding["registered_native_seed"]["outer_CLI_derivation"] = proof
        return prepared, binding
    namespace.prepare_entry = prepare_with_CLI_proof
    try:
        return namespace.main(argv)
    finally:
        if source_bindings() != sources:
            raise ValueError("Native explicit-seed sources changed during invocation; partial files retained")


if __name__ == "__main__":
    raise SystemExit(main())
