"""Restore two JSON tuple fields, then call the unchanged registered auditor.

The original reader remains the executed function. This explicit execution
adapter is included in its actual report; no checkpoint or frozen source edits.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "docs/style_improvement_20261003/orca_official_all_temporal_formal_preparation_v1"
ORIGINAL = FORMAL / "audit_terminal.py"
ORIGINAL_SHA = "77025f740bd998b254d944f5bc26dc3501bf5b508cc062dc1ef6a169d4e07c07"
PAPER = ROOT / "docs/style_improvement_20261003/orca_interact_usable_staging_v1/paper_interact.py"
PAPER_SHA = "42abd2c7757fc3e1feac8a1b0645c86a2d831744f4f3de8567c0c62ecf0c18b3"
PLAN = FORMAL / "commands_and_audit_plan.json"
PLAN_SHA = "2beed1bd10777658eb1c6ec8e1a2aa3bbc7c8571646ddf8f5fa9f3fbec9ab945"
FIELDS = ("state_mean", "state_std")


def record(path, *, readonly=True):
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or readonly and before.st_mode & 0o222):
        raise ValueError("Ordinary independent readonly source required: " + str(path))
    data = path.read_bytes()
    after = path.lstat()
    if any(getattr(before, key) != getattr(after, key) for key in
           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")):
        raise ValueError("Source changed during read")
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA in this standalone CPU process")
    if "paper_interact" in sys.modules or "torch" in sys.modules:
        raise ValueError("Use a fresh standalone process")
    execution = record(Path(__file__))
    original, paper, plan_row = record(ORIGINAL), record(PAPER, readonly=False), record(PLAN)
    if (original["sha256"] != ORIGINAL_SHA or paper["sha256"] != PAPER_SHA
            or plan_row["sha256"] != PLAN_SHA):
        raise ValueError("Frozen original reader/configuration/plan changed")
    plan = json.loads(PLAN.read_text())
    if original != plan["terminal_auditor"]:
        raise ValueError("The executed original reader is not the registered function")
    config_path = Path(plan["formal_output"]) / "run_config_initial.json"
    config_row = record(config_path)
    model = json.loads(config_path.read_text())["model"]
    if any(type(model.get(key)) is not list or len(model[key]) != 5 for key in FIELDS):
        raise ValueError("Exactly the two declared five-element JSON arrays are required")
    destination = FORMAL / "actual_terminal40000_audit.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve every existing audit; use an exclusive original output")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(PAPER.parent))
    module = importlib.import_module("paper_interact")
    if Path(module.__file__).resolve() != PAPER or record(PAPER, readonly=False) != paper:
        raise ValueError("Actual imported configuration source differs")
    configuration_class = module.PaperDiffusionConfig
    calls = []

    def restore_configuration(**kwargs):
        if calls or kwargs != model:
            raise ValueError("Only the original reader's one exact run-JSON configuration is restored")
        decoded = {**kwargs, **{key: tuple(kwargs[key]) for key in FIELDS}}
        config = configuration_class(**decoded)
        if (asdict(config) != decoded or any(type(getattr(config, key)) is not tuple for key in FIELDS)
                or json.loads(json.dumps(asdict(config), allow_nan=False)) != model):
            raise ValueError("Configuration values or other fields changed during tuple restoration")
        calls.append({"fields": list(FIELDS), "JSON_types": ["list", "list"],
                      "restored_types": ["tuple", "tuple"], "all_values_JSON_equal": True})
        return config

    spec = importlib.util.spec_from_file_location("_unchanged_registered_terminal_reader", ORIGINAL)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with patch.object(module, "PaperDiffusionConfig", restore_configuration):
        result = reader.main(sys.argv[1:] if argv is None else argv)
    if (result != 0 or len(calls) != 1 or module.PaperDiffusionConfig is not configuration_class
            or record(ORIGINAL) != original or record(PAPER, readonly=False) != paper
            or record(PLAN) != plan_row or record(config_path) != config_row
            or record(Path(__file__)) != execution):
        raise ValueError("Original audit, one restored configuration or source restoration failed")
    report = json.loads(destination.read_text())
    if report["status"] != "passed" or report["reader"] != original:
        raise ValueError("The unchanged original strict auditor did not pass")
    report["configuration_JSON_tuple_restoration"] = {
        "execution_adapter": execution, "original_reader_function": original,
        "configuration_class_source": paper, "run_JSON_source": config_row,
        "constructor_calls": calls, "temporary_module_alias_restored": True,
        "original_validate_resume_function_unchanged": True,
        "checkpoint_config_or_weights_modified": False,
        "strict_recipe_corpus_counters_and_tensor_checks_unchanged": True,
    }
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "configuration_fields_restored": list(FIELDS),
                      "execution_adapter": execution, "original_reader": original}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
