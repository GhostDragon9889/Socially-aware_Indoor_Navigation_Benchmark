#!/usr/bin/env python3
"""Canonical shared Social AMP entry; explicit legacy or Pop-Art experiment.

Reuses the sealed real AMASS World, collector, expert, clocks and shutdown.
No physical resume or default-model promotion. Run serially on the host GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from CrowdSim.humanoid.runtime import social_popart_training  # noqa: E402
from CrowdSim.humanoid.runtime import grouped_source_migration  # noqa: E402

PARENT = ROOT / "docs/style_improvement_20261004/shared_social_contact_layout_revision_v6/train_shared_social.py"
PARENT_SHA = "6b64213aea03243bf841af0c8e1b5bb03f06700d030f6d8d16051d356bfb59c4"


def prepare_entry(strategy):
    parent_identity = record(PARENT)
    if parent_identity["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
        raise ValueError("Sealed physical training entry changed")
    own_identity = record(Path(__file__))
    migration = social_popart_training.describe_runtime() if strategy == "popart" else None
    module = ModuleType("_canonical_shared_social_physical_entry")
    module.__file__ = str(PARENT)
    exec(compile(PARENT.read_bytes(), str(PARENT), "exec"), module.__dict__)
    grouped_binding = grouped_source_migration.describe()
    grouped_source_migration.adapt_entry(module)
    original_sources, original_load, original_run = module._sources, module._load, module._run

    def sources():
        if record(Path(__file__)) != own_identity:
            raise ValueError("Canonical training entry changed after admission")
        rows = original_sources()
        if grouped_source_migration.describe() != grouped_binding:
            raise ValueError("Grouped runtime migration changed after admission")
        rows["grouped_runtime_migration"] = grouped_binding["adapter"]
        rows["grouped_runtime_registration"] = grouped_binding["registration"]
        rows["original_entry"], rows["entry"] = rows["entry"], own_identity
        if migration is not None:
            if social_popart_training.describe_runtime() != migration:
                raise ValueError("Pop-Art numerical source changed")
            rows["value_normalization_adapter"] = migration["adapter"]
            rows["value_normalization_helper"] = migration["helper"]
        return rows

    def load(path, name, identity):
        if strategy == "popart" and Path(path) == social_popart_training.TRAINER:
            if record(path) != identity:
                raise ValueError("Trainer basis differs from actual physical source inventory")
            return social_popart_training.load_trainer(name)
        return grouped_source_migration.load(path, name, identity)

    def run(args, profile, profile_record, report, rows):
        report["canonical_entry"] = {"entry": own_identity, "original_entry": parent_identity,
                                    "value_normalization": strategy, "numeric_migration": migration,
                                    "grouped_runtime_migration": grouped_binding,
                                    "World_collector_expert_algorithms_changed": False}
        return original_run(args, profile, profile_record, report, rows)

    # Only this caller-owned namespace is adapted. Original source modules,
    # SDK classes/methods, sampler/reset/physics implementation are unchanged.
    module._sources, module._load, module._run = sources, load, run
    return module, {"entry": own_identity, "parent": parent_identity, "strategy": strategy,
                    "migration": migration, "grouped_runtime_migration": grouped_binding}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-normalization", choices=("legacy", "popart"), default="legacy")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    module, binding = prepare_entry(args.value_normalization)
    if args.check_sources:
        rows = module._sources()
        print(json.dumps({"status": "source_checks_passed", "binding": binding, "bound_sources": len(rows),
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    if args.config is None or args.output is None:
        parser.error("--config and --output are required")
    saved = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--config", str(args.config), "--output", str(args.output), "--device", args.device]
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
