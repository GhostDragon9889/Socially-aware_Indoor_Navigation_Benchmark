"""Explicit-seed admission around the unchanged native CPU measurement kernels.

The old seed7 reader stays source-bound. Each new reader owns a separate module
namespace; only source/schema selectors and seed admission differ. This reader
does not observe a process exit or establish the authors' evaluation protocol.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

from . import native_navigation_audit as frozen
from .cpu_dependency_migration import verify_reader
from scripts import evaluate_shared_social_native_seeded as entry

ROOT, original, initial = frozen.ROOT, frozen.original, frozen.initial
MANIFEST, MANIFEST_SHA = frozen.MANIFEST, frozen.MANIFEST_SHA
ENTRY = ROOT / "scripts/evaluate_shared_social_native_seeded.py"
ENTRY_SHA = "6a318599242ae701ad54ffb59f5229ecfcd07c893e70ea42961540ed80858367"
FROZEN = ROOT / "CrowdSim/humanoid/analysis/native_navigation_audit.py"
FROZEN_SHA = "f80f005bb2796bf37168281f243ab462ee207223609b735656a41b7febe9fe8b"
EXECUTION_SCHEMA = "shared_social.actual_native_seeded_navigation_execution.v1"
AUDIT_SCHEMA = "shared_social.native_seeded_scene_registered_controls_CPU_audit.v1"
_protocol = frozen._protocol


def source_bindings():
    record, require = original.record, original.require
    migration = verify_reader(FROZEN, FROZEN_SHA)
    require(Path(frozen.__file__).resolve() == FROZEN.resolve()
            and Path(entry.__file__).resolve() == ENTRY and record(ENTRY)["sha256"] == ENTRY_SHA,
            "Frozen raw kernel or registered seeded entry changed")
    sources = frozen.source_bindings()
    return {**sources, "reader_migration": migration,
            "native_reader": record(__file__), "seed7_native_reader": record(FROZEN),
            "seed7_paired_entry": sources["paired_entry"], "paired_entry": record(ENTRY),
            "seed_entry_sources": entry.source_bindings()}


def require_registered_seed(value, seed, role):
    entry.validate_seed(seed)
    original.require(type(value) is int and value == seed,
                     "Registered native seed differs or is not an integer: " + role)


def validate_seed_contract(receipt, result, snapshot, *, actors, planner, seed):
    """Check actual receipt, driver, initial arguments and exact entry proofs."""
    require_registered_seed(receipt.get("seed"), seed, "external SDK receipt")
    require_registered_seed(result["arguments"].get("seed"), seed, "actual driver arguments")
    require_registered_seed(snapshot["contract"]["arguments_excluding_output"].get("seed"), seed,
                            "initial capture arguments")
    migration = result["social_entry"]["native_current_migration"]
    registered = migration.get("registered_native_seed", {})
    original.require(isinstance(registered, dict), "Registered native seed proof must be a mapping")
    require_registered_seed(registered.get("seed"), seed, "entry registration")
    _, expected = entry.prepare_entry(actors=actors, planner=planner, seed=seed)
    proof = dict(expected["registered_native_seed"], outer_CLI_derivation=entry.outer_CLI_source()[1])
    original.require(registered == proof and migration.get("seed7_paired_entry") == expected["seed7_paired_entry"],
                     "Registered seed entry sources or interface proofs differ")


def _reader(seed):
    """Execute pinned parent bytes in an owned namespace, preserving all math."""
    entry.validate_seed(seed)
    source_bindings()
    namespace = ModuleType("CrowdSim.humanoid.analysis._owned_seeded_native_raw")
    namespace.__package__ = "CrowdSim.humanoid.analysis"
    namespace.__file__ = str(FROZEN)
    exec(compile(FROZEN.read_bytes(), str(FROZEN) + ":seeded_admission", "exec"), namespace.__dict__)
    namespace.ENTRY, namespace.ENTRY_SHA = ENTRY, ENTRY_SHA
    namespace.EXECUTION_SCHEMA, namespace.AUDIT_SCHEMA = EXECUTION_SCHEMA, AUDIT_SCHEMA
    namespace.source_bindings = source_bindings
    inherited_validator = namespace.validate_result_contract

    def validate(receipt, result, request, snapshot, *, actors, controls, planner):
        validate_seed_contract(receipt, result, snapshot, actors=actors, planner=planner, seed=seed)
        admitted = inherited_validator(receipt, result, request, snapshot,
                                       actors=actors, controls=controls, planner=planner)
        return {**admitted, "registered_seed": seed,
                "receipt_driver_initial_and_entry_seed_equal": True}

    namespace.validate_result_contract = validate
    return namespace


def prepare_kernels(*, actors, controls, seed):
    kernels = _reader(seed).prepare_kernels(actors=actors, controls=controls)
    kernels.derivation["registered_seed"] = seed
    return kernels


def validate_result_contract(receipt, result, request, snapshot, *, actors, controls, planner, seed):
    return _reader(seed).validate_result_contract(receipt, result, request, snapshot,
        actors=actors, controls=controls, planner=planner)


def admit_execution(receipt_path, *, actors, controls, planner, scene_config, scene_sha256, seed):
    bindings = source_bindings()
    _protocol(actors, controls)
    entry.validate_seed(seed)
    receipt = original.json.loads(Path(receipt_path).read_text())
    original.require(receipt.get("schema") == EXECUTION_SCHEMA
        and type(receipt.get("actual_exit_code")) is int and receipt["actual_exit_code"] == 0
        and isinstance(receipt.get("terminal_tool_chunk"), str) and bool(receipt["terminal_tool_chunk"]),
        "Require externally observed actual successful seeded native SDK exit")
    require_registered_seed(receipt.get("seed"), seed, "external SDK receipt before resources")
    selected = _reader(seed).admit_execution(receipt_path, actors=actors, controls=controls,
        planner=planner, scene_config=scene_config, scene_sha256=scene_sha256)
    original.require(source_bindings() == bindings, "Seeded native admission sources changed")
    selected.identities.extend(original.record(path) for path in (Path(__file__), FROZEN, ENTRY))
    return selected
