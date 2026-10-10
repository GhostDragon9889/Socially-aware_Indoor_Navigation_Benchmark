"""Thin paired-reader revision for recipe-supported FP32 EMA constants.

The original selection, terminal admission, loss, sampling and metric functions
are retained. This entry never changes a stored normalization buffer.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/style_improvement_20261003/orca_all_temporal_paired_validation_v1/evaluate_paired.py"
ORIGINAL_SHA = "9f7732c99c33984a5ffaf24d505dc8261668b84a4ccc3703c1e5c5bc07d7e113"
HELPER = HERE.parent / "full_validation_loss_diagnostic_v1/normalization_contract.py"
HELPER_SHA = "2bfb3f9766a854636029960a0794a2ca79f83e3b0700109264b367c7e2b7bf15"
REGISTRATION = HERE / "source_registration.json"

REPLACEMENTS = (
    ("HERE = Path(__file__).resolve().parent\n",
     'HERE = ROOT / "docs/style_improvement_20261003/orca_all_temporal_paired_validation_v1"\n'),
    ('    normalization = {}\n'
     '    for key in ("state_mean", "state_std"):\n'
     '        require(torch.equal(getattr(old.model, key), getattr(new.model, key)), "Model normalization mismatch")\n'
     '        normalization[key] = getattr(old.model, key).tolist()\n',
     '    normalization = _normalization_contract_v2(old, new, audit)\n'),
    ('    report = {"schema": "official_orca.paired_validation_actual_comparison.v1", "status": "passed",\n',
     '    report = {"schema": "official_orca.paired_validation_EMA_normalization_actual_comparison.v2", "status": "passed",\n'
     '              "reader_revision": _reader_revision_v2(),\n'),
    ('default=ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_validation_v1")',
     'default=ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_validation_EMA_normalization_v2")'),
)


def file_record(path):
    path = Path(path)
    info = path.stat()
    if not path.is_file() or path.is_symlink() or info.st_nlink != 1:
        raise ValueError(f"Ordinary single-link source required: {path}")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data


def source_binding():
    if REGISTRATION.is_symlink() or REGISTRATION.stat().st_mode & 0o222:
        raise ValueError("Readonly ordinary source registration required")
    registration_record, registration_bytes = file_record(REGISTRATION)
    registration = json.loads(registration_bytes)
    if registration.get("schema") != "official_orca.paired_EMA_normalization_source_registration.v2":
        raise ValueError("Unexpected paired-reader source registration")
    records, sources = {}, {}
    for role, path in (("entry", Path(__file__).resolve()), ("original", ORIGINAL), ("normalization_helper", HELPER)):
        row, data = file_record(path)
        if row != registration[role]:
            raise ValueError(f"Registered {role} source bytes changed")
        records[role], sources[role] = row, data.decode("utf-8")
    if records["original"]["sha256"] != ORIGINAL_SHA or records["normalization_helper"]["sha256"] != HELPER_SHA:
        raise ValueError("Frozen original or shared normalization source changed")
    return {"registration": registration_record, **records}, sources


def derive(original):
    derived = original
    for before, after in REPLACEMENTS:
        if derived.count(before) != 1:
            raise ValueError("Missing or ambiguous exact reader replacement")
        derived = derived.replace(before, after, 1)
    reverse = derived
    for before, after in reversed(REPLACEMENTS):
        if reverse.count(after) != 1:
            raise ValueError("Ambiguous reader reverse replacement")
        reverse = reverse.replace(after, before, 1)
    if reverse != original:
        raise ValueError("Reader whole-byte inverse differs")
    before_tree, after_tree = ast.parse(original), ast.parse(derived)
    before_functions = {node.name: node for node in before_tree.body if isinstance(node, ast.FunctionDef)}
    after_functions = {node.name: node for node in after_tree.body if isinstance(node, ast.FunctionDef)}
    if before_functions.keys() != after_functions.keys():
        raise ValueError("Reader function inventory changed")
    unchanged = {}
    for name, node in before_functions.items():
        if name in ("evaluate", "main"):
            continue
        old_text, new_text = ast.get_source_segment(original, node), ast.get_source_segment(derived, after_functions[name])
        if old_text != new_text or ast.dump(node, include_attributes=False) != ast.dump(after_functions[name], include_attributes=False):
            raise ValueError(f"Original reader function changed: {name}")
        unchanged[name] = hashlib.sha256(old_text.encode()).hexdigest()
    compile(derived, str(__file__) + ":exact_reader_delta", "exec")
    return derived, {"replacement_count": len(REPLACEMENTS), "whole_byte_reverse_exact": True,
                     "derived_source_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "unchanged_function_source_sha256": unchanged,
                     "changed_function_scope": {"evaluate": "normalization admission and v2 provenance only",
                                                "main": "independent default output path only"}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    binding, sources = source_binding()  # Includes this entry before derive or imports.
    derived, derivation = derive(sources["original"])
    if argv == ["--check-source-only"]:
        print(json.dumps({"status": "source_only_ready", "sources": binding, "derivation": derivation,
                          "NN_executed": False, "weights_read": False, "selection_or_capture_read": False,
                          "GPU_used": False, "SDK_started": False, "terminal_claimed": False}, allow_nan=False))
        return 0
    # This revision consumes the already frozen selection; it cannot reselect.
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=("evaluate",), required=True)
    mode_parser.parse_known_args(argv)
    helper = {"__file__": str(HELPER), "__name__": "_paired_normalization_contract_v2"}
    exec(compile(sources["normalization_helper"], str(HELPER), "exec"), helper)

    def normalization_contract(old, new, audit):
        reports = {}
        for name, planner, weight_source in (("OLD", old, "raw"), ("NEW", new, "EMA")):
            actual = {}
            for key in ("state_mean", "state_std"):
                buffer = getattr(planner.model, key)
                if str(buffer.dtype) != "torch.float32" or tuple(buffer.shape) != (5,):
                    raise ValueError("Original FP32 five-state normalization buffers required")
                actual[key] = buffer.tolist()
            reports[name] = helper["validate_normalization"](actual, asdict(planner.config), planner.trained_steps,
                weight_source=weight_source, ema_recipe=audit["training_recipe"]["EMA"])
        return {"models": reports, "normalization_buffers_rewritten": False,
                "legacy_exact_equality_registration_flag_overridden_by_v2_source": True,
                "same_stored_normalization_between_models_claimed": False}

    def reader_revision():
        after, _ = source_binding()
        if after != binding:
            raise ValueError("Reader/helper source changed during actual evaluation")
        return {"schema": "official_orca.paired_reader_revision.v2", "entry": binding["entry"],
                "source_binding": binding, "derivation": derivation,
                "original_frozen_evaluator_executed_bitexact": False,
                "original_selection_loss_plan_RNG_metrics_and_terminal_admission_retained": True,
                "normalization_gate_only_accepts_shared_recipe_supported_values": True}

    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "_paired_validation_EMA_normalization_v2",
                 "_normalization_contract_v2": normalization_contract, "_reader_revision_v2": reader_revision}
    exec(compile(derived, str(__file__) + ":exact_reader_delta", "exec"), namespace)
    return namespace["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
