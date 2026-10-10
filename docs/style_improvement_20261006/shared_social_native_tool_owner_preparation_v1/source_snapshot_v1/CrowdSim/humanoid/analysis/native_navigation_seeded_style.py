"""Admit same-seed native pairs for the unchanged frozen motion-style metric."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import ModuleType

from . import native_navigation_seeded_audit as raw
from . import native_navigation_style as frozen

ROOT, metric = raw.ROOT, frozen.metric
OWNER_SCHEMA = "shared_social.actual_native_seeded_raw_CPU_audit_execution.v1"
STYLE_SCHEMA = "shared_social.paired_native_seeded_navigation_inclusive_style.v1"
FROZEN = ROOT / "CrowdSim/humanoid/analysis/native_navigation_style.py"
FROZEN_SHA = "f0ec149c97e10bc7511fdd7775281d8ee1f28d2b35aaf6e9f73bb7386fd7ff54"
RAW_CLI = ROOT / "scripts/audit_shared_social_native_seeded.py"
RAW_CLI_SHA = "0632eb16f71feeddbd4e608b88f3c69b584512de0b0e0a53a7a6cedc7024a1d8"
RAW_SHA = "b405e4926915abc4f8165d2ff996d15227e218e12d9209555aca349d1e5d7bbd"


def source_bindings():
    metric.require(Path(frozen.__file__).resolve() == FROZEN and metric.record(FROZEN)["sha256"] == FROZEN_SHA
        and metric.record(RAW_CLI)["sha256"] == RAW_CLI_SHA
        and metric.record(raw.__file__)["sha256"] == RAW_SHA, "Frozen style or seeded raw admission source changed")
    return {"native_seeded_style_reader": metric.record(__file__), "frozen_seed7_style_reader": metric.record(FROZEN),
            "unchanged_style_and_reference_bindings": frozen.source_bindings(),
            "seeded_raw_CLI": metric.record(RAW_CLI), "seeded_raw_kernels": raw.source_bindings()}


def _audit_seed(owner, audit, kernels, seed):
    for value, role in ((owner.get("seed"), "external raw CPU receipt"), (audit.get("seed"), "raw CPU report"),
        (kernels.derivation.get("registered_seed"), "raw CPU kernels"),
        (audit["execution_admission"].get("registered_seed"), "raw SDK admission")):
        raw.require_registered_seed(value, seed, role)
    metric.require(audit["execution_admission"].get("receipt_driver_initial_and_entry_seed_equal") is True,
                   "Raw audit did not admit the actual receipt, driver, initial and entry seed")


def _reader(seed):
    raw.entry.validate_seed(seed)
    source_bindings()
    namespace = ModuleType("CrowdSim.humanoid.analysis._owned_seeded_native_style")
    namespace.__package__ = "CrowdSim.humanoid.analysis"
    namespace.__file__ = str(FROZEN)
    exec(compile(FROZEN.read_bytes(), str(FROZEN) + ":seeded_admission", "exec"), namespace.__dict__)
    bound_raw = ModuleType("_owned_native_raw_fixed_seed")
    bound_raw.__dict__.update(raw.__dict__)
    bound_raw.prepare_kernels = partial(raw.prepare_kernels, seed=seed)
    bound_raw.admit_execution = partial(raw.admit_execution, seed=seed)
    namespace.raw = bound_raw
    namespace.OWNER_SCHEMA, namespace.STYLE_SCHEMA = OWNER_SCHEMA, STYLE_SCHEMA
    sources = list(namespace.SOURCES)
    sources[2] = (str(RAW_CLI.relative_to(ROOT)), RAW_CLI_SHA)
    namespace.SOURCES = tuple(sources)
    namespace.source_bindings = source_bindings
    inherited_validator = namespace.validate_audit_contract

    def validate(owner, audit, kernels, *, planner):
        _audit_seed(owner, audit, kernels, seed)
        return inherited_validator(owner, audit, kernels, planner=planner)

    namespace.validate_audit_contract = validate
    return namespace


def validate_audit_contract(owner, audit, kernels, *, planner, seed):
    return _reader(seed).validate_audit_contract(owner, audit, kernels, planner=planner)


def read_completed_run(owner_path, *, actors, planner, scene_config, scene_sha256, seed):
    raw.entry.validate_seed(seed)
    bindings = source_bindings()
    owner = metric.bound(metric.record(owner_path))
    metric.require(owner.get("schema") == OWNER_SCHEMA
        and type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
        and isinstance(owner.get("terminal_tool_chunk"), str) and bool(owner["terminal_tool_chunk"]),
        "Require externally observed successful seeded native raw CPU audit")
    raw.require_registered_seed(owner.get("seed"), seed, "external raw CPU receipt before resources")
    selected = _reader(seed).read_completed_run(owner_path, actors=actors, planner=planner,
        scene_config=scene_config, scene_sha256=scene_sha256)
    metric.require(source_bindings() == bindings, "Seeded native style admission sources changed")
    selected.admission["registered_seed"] = seed
    selected.identities.extend(metric.record(path) for path in (Path(__file__), FROZEN, RAW_CLI))
    return selected


# No fork of extraction, normalization, interval masks or score calculations.
capture_samples = frozen.capture_samples
